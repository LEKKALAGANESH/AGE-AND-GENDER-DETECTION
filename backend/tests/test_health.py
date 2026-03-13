"""Tests for GET /api/health."""


def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_health_returns_ok_status(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"


def test_health_returns_models_loaded_true_when_loaded(client):
    data = client.get("/api/health").json()
    assert data["models_loaded"] is True


def test_health_returns_models_loaded_false_when_not_loaded(client_no_models):
    data = client_no_models.get("/api/health").json()
    assert data["models_loaded"] is False
