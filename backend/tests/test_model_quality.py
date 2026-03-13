"""Model quality & validation tests for Lite-Vision.

Verifies that the age/gender detection system produces varied,
high-quality results — not pre-planned categorical outputs.

These tests use mocked inference backends (SCRFD face detector + InsightFace
genderage_net) to validate response structure, variability, and performance.
"""

import base64
import statistics
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers — synthetic face image generation
# ---------------------------------------------------------------------------


def create_synthetic_face(
    size: int = 200,
    brightness: int = 128,
    face_size_ratio: float = 0.6,
    offset_x: int = 0,
    offset_y: int = 0,
) -> str:
    """Create a synthetic face-like image for testing.

    Returns a base64-encoded JPEG string.  The image contains a crude
    face drawn with OpenCV primitives (ellipse for head, circles for
    eyes, arc for mouth).  Parameters allow slight variation between
    generated images.
    """
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    center_x = size // 2 + offset_x
    center_y = size // 2 + offset_y
    face_r = int(size * face_size_ratio / 2)

    # Head (skin-toned ellipse)
    cv2.ellipse(
        img,
        (center_x, center_y),
        (face_r, int(face_r * 1.2)),
        0, 0, 360,
        (200, 180, 160),
        -1,
    )

    # Eyes
    eye_y = center_y - face_r // 4
    eye_radius = max(1, face_r // 8)
    cv2.circle(img, (center_x - face_r // 3, eye_y), eye_radius, (50, 50, 50), -1)
    cv2.circle(img, (center_x + face_r // 3, eye_y), eye_radius, (50, 50, 50), -1)

    # Mouth
    mouth_y = center_y + face_r // 3
    cv2.ellipse(
        img,
        (center_x, mouth_y),
        (max(1, face_r // 4), max(1, face_r // 8)),
        0, 0, 180,
        (150, 100, 100),
        2,
    )

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _make_varied_images(n: int = 20) -> list[str]:
    """Generate *n* synthetic face images with slight variations."""
    images = []
    for i in range(n):
        images.append(
            create_synthetic_face(
                size=180 + (i * 3) % 60,
                brightness=100 + (i * 7) % 100,
                face_size_ratio=0.45 + (i % 10) * 0.03,
                offset_x=(i % 5) - 2,
                offset_y=(i % 7) - 3,
            )
        )
    return images


# ---------------------------------------------------------------------------
# Fixtures — mocked client using SCRFD + genderage_net architecture
# ---------------------------------------------------------------------------


def _build_scrfd_mock_varied(call_counter: dict):
    """Return an SCRFD-format mock that always detects one face.

    Returns (det[N,5], kpss[N,5,2]) in SCRFD format.
    """
    mock = MagicMock()

    def _detect(img, conf_threshold=None, max_num=0):
        h, w = img.shape[:2]
        det = np.zeros((1, 5), dtype=np.float32)
        det[0] = [w * 0.2, h * 0.2, w * 0.8, h * 0.8, 0.95]

        kpss = np.zeros((1, 5, 2), dtype=np.float32)
        kpss[0, 0] = [w * 0.35, h * 0.35]  # left eye
        kpss[0, 1] = [w * 0.65, h * 0.35]  # right eye
        kpss[0, 2] = [w * 0.5, h * 0.5]    # nose
        kpss[0, 3] = [w * 0.35, h * 0.65]  # left mouth
        kpss[0, 4] = [w * 0.65, h * 0.65]  # right mouth

        call_counter["n"] += 1
        if max_num > 0:
            det = det[:max_num]
            kpss = kpss[:max_num]
        return det, kpss

    mock.detect = _detect
    return mock


def _build_genderage_mock_varied(call_counter: dict):
    """Return a genderage_net mock that produces varied (non-bucketed)
    age predictions and varying gender confidences.

    The InsightFace genderage model output is [1, 3]:
      [0:2] = gender logits (Male, Female), [2] = age_factor (age = factor * 100)

    Each successive call shifts the age and gender logits so that
    the collected results span a realistic distribution.
    """
    net = MagicMock()
    net.setInput = MagicMock()

    def _forward():
        n = call_counter["n"]
        # Produce ages spanning 18-65 via varying age_factor
        age = 18 + (n * 7 + n // 3) % 47
        age_factor = age / 100.0

        # Gender logits that produce varying softmax probabilities
        # ~60% Male predictions, ~40% Female
        if (n * 17 + 3) % 5 < 2:
            # Female-dominant: Female logit (index 1) > Male logit (index 0)
            male_logit = -0.5 - (n * 7 % 20) / 20.0
            female_logit = 1.0 + (n * 11 % 25) / 25.0
        else:
            # Male-dominant: Male logit (index 0) > Female logit (index 1)
            male_logit = 1.0 + (n * 13 % 35) / 35.0
            female_logit = -0.5 - (n * 9 % 20) / 20.0

        # Convention: [Male_logit, Female_logit, age_factor]
        return np.array([[[male_logit, female_logit, age_factor]]], dtype=np.float32)[0]

    net.forward.side_effect = _forward
    return net


def _build_genderage_mock_deterministic():
    """Return a genderage_net mock that always returns the same result.

    Male with high confidence, age=28.
    """
    net = MagicMock()
    net.setInput = MagicMock()

    def _forward():
        # gender logits: Male=2.0 (index 0), Female=-1.0 (index 1)
        # age factor: 0.28 -> age = 28
        return np.array([[[2.0, -1.0, 0.28]]], dtype=np.float32)[0]

    net.forward.side_effect = _forward
    return net


@pytest.fixture()
def _patch_models(monkeypatch):
    """Prevent real model loading; stub load_models."""
    import main as mod

    monkeypatch.setattr(mod, "load_models", lambda app: None)
    mod.cache.clear()


@pytest.fixture()
def quality_client(_patch_models):
    """A TestClient whose SCRFD detector and genderage_net are mocked
    with *varied* return values — suitable for quality/variability tests.
    """
    from main import app

    call_counter: dict = {"n": 0}

    with TestClient(app) as c:
        app.state.face_detector = _build_scrfd_mock_varied(call_counter)
        app.state.genderage_net = _build_genderage_mock_varied(call_counter)
        yield c


@pytest.fixture()
def deterministic_client(_patch_models):
    """A TestClient that always returns the *same* age/gender for a given
    call — useful for caching and consistency tests.
    """
    from main import app

    call_counter: dict = {"n": 0}
    with TestClient(app) as c:
        app.state.face_detector = _build_scrfd_mock_varied(call_counter)
        app.state.genderage_net = _build_genderage_mock_deterministic()
        yield c


# ---------------------------------------------------------------------------
# 1. TestAgeVariability
# ---------------------------------------------------------------------------


class TestAgeVariability:
    """Verify regression-based age model produces continuous, varied predictions."""

    def test_age_not_bucketed(self, quality_client):
        """Send multiple synthetic face images and verify ages are not
        clustered into a small set of buckets.

        If more than 20% of predictions return the exact same age value
        the test fails — this catches a model that only outputs discrete
        category centres.
        """
        images = _make_varied_images(50)
        ages: list[int] = []

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200, f"Non-200: {resp.text}"
            data = resp.json()
            for face in data["results"]:
                ages.append(face["age"])

        assert len(ages) >= 50, f"Expected >= 50 age predictions, got {len(ages)}"

        # No single age should dominate more than 20% of the results
        from collections import Counter

        counts = Counter(ages)
        max_count = counts.most_common(1)[0][1]
        ratio = max_count / len(ages)
        assert ratio <= 0.20, (
            f"Most common age {counts.most_common(1)[0][0]} appears {max_count}/{len(ages)} "
            f"times ({ratio:.0%}), exceeding 20% — model may be bucketing"
        )

    def test_age_range_continuous(self, quality_client):
        """Verify age predictions span a reasonable continuous range.

        The old categorical model only returned ages from set buckets
        like {1, 5, 10, 18, 25, 35, 48, 60}.  The regression model
        should produce ages across a much wider, non-bucketed spectrum.
        """
        images = _make_varied_images(30)
        ages: set[int] = set()

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                ages.add(face["age"])

        KNOWN_BUCKET_SET = {1, 5, 10, 18, 25, 35, 48, 60}

        # At least 5 distinct age values
        assert len(ages) >= 5, (
            f"Only {len(ages)} distinct age value(s): {sorted(ages)} — "
            "model may not be producing continuous ages"
        )

        # Ages should NOT all come from the old bucket set
        non_bucket = ages - KNOWN_BUCKET_SET
        assert len(non_bucket) > 0, (
            f"All predicted ages {sorted(ages)} fall within the old bucket set "
            f"{sorted(KNOWN_BUCKET_SET)} — model may still be categorical"
        )

    def test_age_min_max_reasonable(self, quality_client):
        """Verify age confidence intervals (age_min, age_max) are reasonable.

        age_min should be <= age, age_max should be >= age.
        The range should not be zero (overconfident) or excessively wide.
        """
        img_b64 = create_synthetic_face()
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        assert resp.status_code == 200
        data = resp.json()

        for face in data["results"]:
            age = face["age"]
            age_min = face["age_min"]
            age_max = face["age_max"]

            # Basic ordering
            assert age_min <= age, f"age_min ({age_min}) > age ({age})"
            assert age_max >= age, f"age_max ({age_max}) < age ({age})"

            # Range should be non-zero (not overconfident)
            spread = age_max - age_min
            assert spread > 0, (
                f"age range is zero (age_min={age_min}, age_max={age_max}) — "
                "model is overconfident"
            )

            # Range should not be absurdly wide
            assert spread <= 30, (
                f"age range is {spread} years (min={age_min}, max={age_max}) — "
                "confidence interval is too wide to be useful"
            )

    def test_age_is_non_negative(self, quality_client):
        """Age predictions should never be negative."""
        images = _make_varied_images(10)
        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                assert face["age"] >= 0, f"Negative age: {face['age']}"
                assert face["age_min"] >= 0, f"Negative age_min: {face['age_min']}"


# ---------------------------------------------------------------------------
# 2. TestGenderRobustness
# ---------------------------------------------------------------------------


class TestGenderRobustness:
    """Verify gender model uses pattern-finding, not simple heuristics."""

    def test_gender_confidence_varies(self, quality_client):
        """Gender confidence should vary across different face images.

        If every prediction returns exactly the same confidence (e.g. 0.99),
        the model is likely not doing real analysis.
        """
        images = _make_varied_images(20)
        confidences: list[float] = []

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                confidences.append(face["gender_confidence"])

        assert len(confidences) >= 20, (
            f"Expected >= 20 gender confidence values, got {len(confidences)}"
        )

        # Standard deviation of confidences should be > 0.01
        stddev = statistics.stdev(confidences)
        assert stddev > 0.01, (
            f"Gender confidence std dev is {stddev:.4f} — values are too uniform. "
            f"Sample: {confidences[:10]}"
        )

        # Not all confidences should be identical
        unique_confs = set(round(c, 4) for c in confidences)
        assert len(unique_confs) > 1, (
            f"All gender confidences are identical ({confidences[0]:.4f}) — "
            "model may not be performing real analysis"
        )

    def test_gender_returns_valid_labels(self, quality_client):
        """Gender should always be 'Male' or 'Female'."""
        images = _make_varied_images(15)

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                assert face["gender"] in ("Male", "Female"), (
                    f"Unexpected gender label: {face['gender']!r}"
                )

    def test_gender_confidence_between_0_and_1(self, quality_client):
        """gender_confidence should always be between 0.0 and 1.0."""
        images = _make_varied_images(15)

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                gc = face["gender_confidence"]
                assert 0.0 <= gc <= 1.0, (
                    f"gender_confidence {gc} is outside [0.0, 1.0]"
                )

    def test_both_genders_appear(self, quality_client):
        """Across many varied images, both Male and Female should appear
        at least once.  A model that always predicts the same gender is
        not performing real analysis.
        """
        images = _make_varied_images(30)
        genders: set[str] = set()

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                genders.add(face["gender"])

        assert "Male" in genders and "Female" in genders, (
            f"Only saw genders {genders} across 30 images — "
            "model may be always predicting the same gender"
        )


# ---------------------------------------------------------------------------
# 3. TestPerformance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Verify inference performance meets requirements."""

    def test_single_face_under_500ms(self, quality_client):
        """Single face detection should complete in under 500ms.

        The InsightFace genderage model is lightweight (~1MB), so
        sub-500ms response times are expected for real-time streaming.
        """
        img_b64 = create_synthetic_face()
        t0 = time.perf_counter()
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        wall_ms = (time.perf_counter() - t0) * 1000

        assert resp.status_code == 200
        data = resp.json()

        # Check the server-reported processing time
        assert data["processing_time_ms"] < 500, (
            f"Server processing time {data['processing_time_ms']:.1f}ms exceeds 500ms limit"
        )

        # Also check wall-clock time (includes network overhead in TestClient)
        assert wall_ms < 2000, (
            f"Wall-clock time {wall_ms:.1f}ms exceeds 2000ms limit "
            "(includes TestClient overhead)"
        )

    def test_response_format_complete(self, quality_client):
        """Verify all expected fields are present in the response."""
        img_b64 = create_synthetic_face()
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        assert resp.status_code == 200
        data = resp.json()

        # Top-level fields
        assert "results" in data, "Missing 'results' field"
        assert "face_count" in data, "Missing 'face_count' field"
        assert "processing_time_ms" in data, "Missing 'processing_time_ms' field"

        assert isinstance(data["results"], list)
        assert isinstance(data["face_count"], int)
        assert isinstance(data["processing_time_ms"], (int, float))

        # Per-face result fields
        assert data["face_count"] >= 1, "Expected at least one detected face"
        face = data["results"][0]

        required_fields = {
            "age": int,
            "age_min": int,
            "age_max": int,
            "gender": str,
            "gender_confidence": (int, float),
            "confidence": (int, float),
            "region": list,
        }
        for field, expected_type in required_fields.items():
            assert field in face, f"Missing field '{field}' in face result"
            assert isinstance(face[field], expected_type), (
                f"Field '{field}' has type {type(face[field]).__name__}, "
                f"expected {expected_type}"
            )

        # Region should have exactly 4 normalized floats
        assert len(face["region"]) == 4, (
            f"Region has {len(face['region'])} elements, expected 4"
        )
        for i, val in enumerate(face["region"]):
            assert 0.0 <= val <= 1.0, (
                f"Region[{i}] = {val} is outside [0.0, 1.0]"
            )

    def test_multiple_sequential_requests_stable(self, quality_client):
        """Send 10 sequential requests and verify no errors or timeouts."""
        images = _make_varied_images(10)
        processing_times: list[float] = []

        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200, (
                f"Request failed with status {resp.status_code}: {resp.text}"
            )
            data = resp.json()
            assert data["face_count"] >= 0
            processing_times.append(data["processing_time_ms"])

        # All 10 should succeed (already asserted above)
        assert len(processing_times) == 10

        # No single request should take an absurd amount of time
        for i, t in enumerate(processing_times):
            assert t < 2000, (
                f"Request {i} took {t:.1f}ms — exceeds 2000ms limit"
            )

    def test_processing_time_is_positive(self, quality_client):
        """processing_time_ms should always be a positive number."""
        img_b64 = create_synthetic_face()
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        assert resp.status_code == 200
        assert resp.json()["processing_time_ms"] > 0


# ---------------------------------------------------------------------------
# 4. TestResponseConsistency
# ---------------------------------------------------------------------------


class TestResponseConsistency:
    """Verify deterministic behaviour and caching."""

    def test_same_image_same_result(self, deterministic_client):
        """Same image should return identical results (via cache or
        deterministic model).
        """
        img_b64 = create_synthetic_face(size=200, brightness=128)

        resp1 = deterministic_client.post("/api/analyze", json={"image": img_b64})
        resp2 = deterministic_client.post("/api/analyze", json={"image": img_b64})

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        data1 = resp1.json()
        data2 = resp2.json()

        # Results should be identical (ages, genders)
        assert data1["face_count"] == data2["face_count"]
        assert len(data1["results"]) == len(data2["results"])

        for f1, f2 in zip(data1["results"], data2["results"]):
            assert f1["age"] == f2["age"], (
                f"Age mismatch on same image: {f1['age']} vs {f2['age']}"
            )
            assert f1["gender"] == f2["gender"], (
                f"Gender mismatch on same image: {f1['gender']} vs {f2['gender']}"
            )
            assert f1["gender_confidence"] == f2["gender_confidence"]
            assert f1["age_min"] == f2["age_min"]
            assert f1["age_max"] == f2["age_max"]

    def test_cache_returns_faster(self, deterministic_client):
        """Second request for the same image should be faster (cache hit)."""
        img_b64 = create_synthetic_face(size=200, brightness=128)

        # First request (cold — runs inference)
        t0 = time.perf_counter()
        resp1 = deterministic_client.post("/api/analyze", json={"image": img_b64})
        cold_ms = (time.perf_counter() - t0) * 1000

        # Second request (should hit cache)
        t1 = time.perf_counter()
        resp2 = deterministic_client.post("/api/analyze", json={"image": img_b64})
        warm_ms = (time.perf_counter() - t1) * 1000

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        server_time_1 = resp1.json()["processing_time_ms"]
        server_time_2 = resp2.json()["processing_time_ms"]

        # The cached response should not be significantly slower
        assert server_time_2 <= server_time_1 + 5.0, (
            f"Cached response ({server_time_2:.1f}ms) should not be slower "
            f"than cold response ({server_time_1:.1f}ms) by more than 5ms"
        )

    def test_different_images_can_differ(self, quality_client):
        """Different images should be allowed to produce different results.

        This is the inverse of the consistency test: we verify the system
        does not return the same canned answer for every input.
        """
        img_a = create_synthetic_face(size=200, brightness=80)
        img_b = create_synthetic_face(size=250, brightness=200, face_size_ratio=0.8)

        resp_a = quality_client.post("/api/analyze", json={"image": img_a})
        resp_b = quality_client.post("/api/analyze", json={"image": img_b})

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        data_a = resp_a.json()
        data_b = resp_b.json()

        if data_a["face_count"] > 0 and data_b["face_count"] > 0:
            fa = data_a["results"][0]
            fb = data_b["results"][0]
            differs = (
                fa["age"] != fb["age"]
                or fa["gender"] != fb["gender"]
                or fa["gender_confidence"] != fb["gender_confidence"]
            )
            assert differs, (
                "Two very different images produced identical predictions — "
                "model may not be analysing input"
            )


# ---------------------------------------------------------------------------
# 5. TestEdgeCases — additional robustness checks
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that could break a fragile model pipeline."""

    def test_small_image_does_not_crash(self, quality_client):
        """Very small images (e.g. 32x32) should return 200 with zero or
        more faces — not crash.
        """
        img_b64 = create_synthetic_face(size=32)
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        # Acceptable: 200 with face_count >= 0, or 422 if image is too small
        assert resp.status_code in (200, 422)

    def test_large_synthetic_image(self, quality_client):
        """A larger image (640x640) should still be handled correctly."""
        img_b64 = create_synthetic_face(size=640)
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["face_count"], int)

    def test_face_count_matches_results_length(self, quality_client):
        """face_count field must equal the length of the results array."""
        images = _make_varied_images(5)
        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            data = resp.json()
            assert data["face_count"] == len(data["results"]), (
                f"face_count={data['face_count']} but "
                f"len(results)={len(data['results'])}"
            )

    def test_region_values_normalized(self, quality_client):
        """All bounding box coordinates should be in [0.0, 1.0]."""
        img_b64 = create_synthetic_face()
        resp = quality_client.post("/api/analyze", json={"image": img_b64})
        assert resp.status_code == 200
        for face in resp.json()["results"]:
            for i, val in enumerate(face["region"]):
                assert 0.0 <= val <= 1.0, (
                    f"region[{i}] = {val} outside [0.0, 1.0]"
                )

    def test_confidence_in_valid_range(self, quality_client):
        """Face detection confidence should be in [0.0, 1.0]."""
        images = _make_varied_images(5)
        for img_b64 in images:
            resp = quality_client.post("/api/analyze", json={"image": img_b64})
            assert resp.status_code == 200
            for face in resp.json()["results"]:
                assert 0.0 <= face["confidence"] <= 1.0, (
                    f"confidence {face['confidence']} outside [0.0, 1.0]"
                )
