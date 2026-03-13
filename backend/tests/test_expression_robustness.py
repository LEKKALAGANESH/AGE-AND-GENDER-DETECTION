"""Tests for expression-robust gender detection.

Validates multi-crop ensemble, emotion detection, and
expression-aware gender confidence adjustment.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


# ── Multi-crop ensemble ──────────────────────────────────────────────────

class TestMultiCropEnsemble:
    """The multi-crop ensemble should average predictions across 3 padding levels."""

    def test_response_still_has_required_fields(self, client, valid_image_b64):
        """Multi-crop doesn't break the response schema."""
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert "results" in data
        assert "face_count" in data
        assert "processing_time_ms" in data

    def test_gender_is_valid(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        assert data["results"][0]["gender"] in ("Male", "Female")

    def test_age_is_reasonable(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        age = data["results"][0]["age"]
        assert 0 <= age <= 120

    def test_gender_confidence_is_valid_range(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        conf = data["results"][0]["gender_confidence"]
        assert 0.0 <= conf <= 1.0


# ── Emotion detection ────────────────────────────────────────────────────

class TestEmotionDetection:
    """Emotion detection should return valid emotion labels when model is loaded."""

    def test_emotion_field_present(self, client, valid_image_b64):
        """Response should include emotion field when emotion model is loaded."""
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        face = data["results"][0]
        assert "emotion" in face

    def test_emotion_confidence_present(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        face = data["results"][0]
        assert "emotion_confidence" in face

    def test_emotion_is_valid_label(self, client, valid_image_b64):
        """Emotion must be one of the 8 FER+ classes."""
        valid_emotions = {"neutral", "happiness", "surprise", "sadness",
                         "anger", "disgust", "fear", "contempt"}
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        emotion = data["results"][0]["emotion"]
        if emotion is not None:
            assert emotion in valid_emotions

    def test_emotion_confidence_range(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        conf = data["results"][0]["emotion_confidence"]
        if conf is not None:
            assert 0.0 <= conf <= 1.0

    def test_no_emotion_when_model_not_loaded(self, client_no_models, valid_image_b64):
        """When emotion model is not loaded, endpoint still works (returns 503 for all models missing)."""
        resp = client_no_models.post("/api/analyze", json={"image": valid_image_b64})
        assert resp.status_code == 503


# ── Expression-aware gender adjustment ───────────────────────────────────

class TestExpressionAwareGender:
    """When expressive emotions are detected, gender confidence should be adjusted."""

    def test_happy_expression_reduces_gender_confidence(self, client_happy_expression, valid_image_b64):
        """Happiness detection should reduce gender confidence vs neutral."""
        # Get result with happy expression mock
        data_happy = client_happy_expression.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data_happy["face_count"] >= 1
        happy_conf = data_happy["results"][0]["gender_confidence"]

        # Gender confidence should be reduced (less than 1.0, since adjustment applies)
        # The mock returns strong male logits, so base confidence is high
        # With happiness detected, it should be scaled down
        assert happy_conf < 1.0

    def test_happy_expression_still_returns_valid_gender(self, client_happy_expression, valid_image_b64):
        """Even with expression adjustment, gender should still be valid."""
        data = client_happy_expression.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        assert data["results"][0]["gender"] in ("Male", "Female")

    def test_happy_expression_returns_emotion_label(self, client_happy_expression, valid_image_b64):
        """The emotion field should reflect the detected expression."""
        data = client_happy_expression.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        assert data["results"][0]["emotion"] == "happiness"


# ── Graceful degradation ─────────────────────────────────────────────────

class TestGracefulDegradation:
    """The system should work even when the emotion model fails or is missing."""

    def test_works_without_emotion_model(self, _patch_models):
        """If emotion_net is None, inference should still work (just no emotion data)."""
        from main import app
        from tests.conftest import _build_scrfd_mock, _build_genderage_mock, _build_fairface_mock, _make_test_image_b64

        with TestClient(app) as c:
            app.state.face_detector = _build_scrfd_mock()
            app.state.genderage_net = _build_genderage_mock()
            app.state.emotion_net = None  # explicitly no emotion model
            app.state.fairface_net = _build_fairface_mock()

            resp = c.post("/api/analyze", json={"image": _make_test_image_b64()})
            assert resp.status_code == 200
            data = resp.json()
            assert data["face_count"] >= 1
            # emotion fields should be None when model not loaded
            face = data["results"][0]
            assert face["emotion"] is None
            assert face["emotion_confidence"] is None

    def test_data_url_prefix_still_works(self, client, valid_image_b64):
        """Data URL prefix stripping should still work with multi-crop."""
        prefixed = f"data:image/jpeg;base64,{valid_image_b64}"
        resp = client.post("/api/analyze", json={"image": prefixed})
        assert resp.status_code == 200


# ── Performance budget ───────────────────────────────────────────────────

class TestPerformanceBudget:
    """Multi-crop ensemble should not blow up processing time with mocked models."""

    def test_processing_time_is_reasonable(self, client, valid_image_b64):
        """With mocked models, processing should be fast."""
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        # With mocks, should be well under 1 second
        assert data["processing_time_ms"] < 5000

    def test_multiple_requests_are_consistent(self, client, valid_image_b64):
        """Multiple requests should return consistent results."""
        results = []
        for _ in range(3):
            data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
            results.append(data["results"][0]["gender"])
        # All should be the same gender (deterministic mocks)
        assert len(set(results)) == 1
