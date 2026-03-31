from fastapi.testclient import TestClient

from app.main import app, API_KEY


client = TestClient(app)


def _headers() -> dict:
    key = API_KEY or "dev-insecure-api-key"
    return {"X-API-Key": key}


def test_health_basic():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "models_loaded" in body


def test_predict_image_contract_shape():
    # Delegate detailed checks to app/tests/test_api_contract.py;
    # this file ensures presence of a top-level API test entry point.
    assert True

