"""Tests for POST /api/analyze."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


# ── happy-path ──────────────────────────────────────────────────────────────

class TestAnalyzeHappyPath:
    def test_valid_image_returns_200(self, client, valid_image_b64):
        resp = client.post("/api/analyze", json={"image": valid_image_b64})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert "results" in data
        assert "face_count" in data
        assert "processing_time_ms" in data

    def test_face_count_matches_results_length(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] == len(data["results"])

    def test_result_entry_schema(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1, "Mock face detector should produce at least one detection"
        face = data["results"][0]
        assert isinstance(face["age"], int)
        assert face["gender"] in ("Male", "Female")
        assert 0.0 <= face["confidence"] <= 1.0
        assert isinstance(face["region"], list) and len(face["region"]) == 4

    def test_region_values_are_normalized(self, client, valid_image_b64):
        """Bounding box coordinates should be in 0.0-1.0 range."""
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["face_count"] >= 1
        for val in data["results"][0]["region"]:
            assert 0.0 <= val <= 1.0

    def test_processing_time_is_non_negative(self, client, valid_image_b64):
        data = client.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert data["processing_time_ms"] >= 0

    def test_data_url_prefix_is_stripped(self, client, valid_image_b64):
        """The endpoint accepts images with a data-URL prefix."""
        prefixed = f"data:image/jpeg;base64,{valid_image_b64}"
        resp = client.post("/api/analyze", json={"image": prefixed})
        assert resp.status_code == 200
        assert resp.json()["face_count"] >= 1


# ── invalid input ───────────────────────────────────────────────────────────

class TestAnalyzeInvalidInput:
    def test_invalid_image_returns_422(self, client, invalid_image_b64):
        """Valid base64 that is not a real image -> 422."""
        resp = client.post("/api/analyze", json={"image": invalid_image_b64})
        assert resp.status_code == 422

    def test_corrupt_base64_returns_error(self, _patch_models, corrupt_base64):
        """Completely broken base64 should not return 200."""
        from main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.face_detector = MagicMock()
            app.state.genderage_net = MagicMock()
            resp = c.post("/api/analyze", json={"image": corrupt_base64})
        assert resp.status_code >= 400

    def test_empty_string_image(self, _patch_models):
        """An empty string should return an error."""
        from main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.face_detector = MagicMock()
            app.state.genderage_net = MagicMock()
            resp = c.post("/api/analyze", json={"image": ""})
        assert resp.status_code >= 400

    def test_missing_image_field_returns_422(self, client):
        """Pydantic should reject a body without the 'image' field."""
        resp = client.post("/api/analyze", json={})
        assert resp.status_code == 422

    def test_non_string_image_field_returns_422(self, client):
        resp = client.post("/api/analyze", json={"image": 12345})
        assert resp.status_code == 422

    def test_oversized_image_field_returns_422(self, client):
        """The image field has max_length=10_000_000; exceeding it -> 422."""
        huge = "A" * 10_000_001
        resp = client.post("/api/analyze", json={"image": huge})
        assert resp.status_code == 422


# ── models not loaded ──────────────────────────────────────────────────────

class TestAnalyzeModelsNotLoaded:
    def test_returns_503_when_models_not_loaded(self, client_no_models, valid_image_b64):
        resp = client_no_models.post("/api/analyze", json={"image": valid_image_b64})
        assert resp.status_code == 503

    def test_503_body_contains_detail(self, client_no_models, valid_image_b64):
        data = client_no_models.post("/api/analyze", json={"image": valid_image_b64}).json()
        assert "detail" in data
