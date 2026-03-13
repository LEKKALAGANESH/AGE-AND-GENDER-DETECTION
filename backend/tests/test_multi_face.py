"""Tests for multi-face detection and per-face differentiation.

Verifies that the inference pipeline produces unique predictions
for each detected face (not identical results due to normalization bugs).
"""

import pytest


class TestMultiFaceDetection:
    """Multiple faces should produce individual predictions."""

    def test_detects_multiple_faces(self, client_multi_face, multi_face_image_b64):
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        assert data["face_count"] == 3

    def test_face_count_matches_results(self, client_multi_face, multi_face_image_b64):
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        assert data["face_count"] == len(data["results"])

    def test_ages_are_different(self, client_multi_face, multi_face_image_b64):
        """Each face should have a different predicted age."""
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        ages = [r["age"] for r in data["results"]]
        # With the varied mock, we expect 3 different ages
        assert len(set(ages)) >= 2, f"Expected varied ages but got {ages}"

    def test_genders_can_differ(self, client_multi_face, multi_face_image_b64):
        """Different faces can have different genders."""
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        genders = [r["gender"] for r in data["results"]]
        # With the varied mock (Male, Female, Male), should have both
        assert "Male" in genders and "Female" in genders, f"Expected mixed genders but got {genders}"

    def test_each_face_has_valid_schema(self, client_multi_face, multi_face_image_b64):
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        for face in data["results"]:
            assert isinstance(face["age"], int)
            assert face["gender"] in ("Male", "Female")
            assert 0.0 <= face["confidence"] <= 1.0
            assert isinstance(face["region"], list) and len(face["region"]) == 4

    def test_each_face_has_unique_region(self, client_multi_face, multi_face_image_b64):
        """Bounding boxes should be at different positions."""
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        regions = [tuple(r["region"]) for r in data["results"]]
        assert len(set(regions)) == 3, f"Expected 3 unique regions but got {regions}"

    def test_processing_time_reasonable(self, client_multi_face, multi_face_image_b64):
        data = client_multi_face.post("/api/analyze", json={"image": multi_face_image_b64}).json()
        assert data["processing_time_ms"] < 10000


class TestMultiFaceEdgeCases:
    """Edge cases with multi-face detection."""

    def test_max_faces_limits_results(self, client_multi_face, multi_face_image_b64):
        """max_faces=2 should return only 2 faces even when 3 are detected."""
        data = client_multi_face.post(
            "/api/analyze", json={"image": multi_face_image_b64}, params={"max_faces": 2}
        ).json()
        assert data["face_count"] == 2

    def test_single_face_still_works(self, client, valid_image_b64):
        """Single-face detection should still work correctly."""
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] == 1
