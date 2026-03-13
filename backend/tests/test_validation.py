"""Tests for request validation and CORS headers."""

import pytest


# ── request body validation ─────────────────────────────────────────────────

class TestRequestValidation:
    def test_image_field_required(self, client):
        resp = client.post("/api/analyze", json={})
        assert resp.status_code == 422
        body = resp.json()
        # FastAPI/Pydantic puts validation errors under "detail"
        assert "detail" in body

    def test_extra_fields_are_ignored(self, client, valid_image_b64):
        """Extra keys in the payload should not cause an error."""
        resp = client.post(
            "/api/analyze",
            json={"image": valid_image_b64, "extra_field": "hello"},
        )
        assert resp.status_code == 200

    def test_wrong_content_type_returns_422(self, client, valid_image_b64):
        """Sending form-encoded data instead of JSON should fail."""
        resp = client.post(
            "/api/analyze",
            data={"image": valid_image_b64},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 422

    def test_max_faces_query_param_respected(self, client, valid_image_b64):
        """max_faces query parameter should be accepted."""
        resp = client.post(
            "/api/analyze?max_faces=5",
            json={"image": valid_image_b64},
        )
        assert resp.status_code == 200

    def test_max_faces_below_minimum_returns_422(self, client, valid_image_b64):
        """max_faces must be >= 1."""
        resp = client.post(
            "/api/analyze?max_faces=0",
            json={"image": valid_image_b64},
        )
        assert resp.status_code == 422

    def test_max_faces_above_maximum_returns_422(self, client, valid_image_b64):
        """max_faces must be <= 100."""
        resp = client.post(
            "/api/analyze?max_faces=999",
            json={"image": valid_image_b64},
        )
        assert resp.status_code == 422


# ── CORS ────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_allows_any_origin(self, client):
        """A preflight OPTIONS request from any origin should be accepted."""
        resp = client.options(
            "/api/analyze",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_cors_header_on_regular_request(self, client, valid_image_b64):
        """Regular POST responses should also carry the CORS header."""
        resp = client.post(
            "/api/analyze",
            json={"image": valid_image_b64},
            headers={"Origin": "https://example.com"},
        )
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_cors_allows_all_methods(self, client):
        resp = client.options(
            "/api/analyze",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        allow = resp.headers.get("access-control-allow-methods", "")
        assert "*" in allow or "DELETE" in allow
