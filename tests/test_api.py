from fastapi.testclient import TestClient

import io

from PIL import Image

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


def test_predict_image_contract_shape(monkeypatch):
    import app.main as main_mod

    # Configure API key so we can verify auth behavior.
    # (Without this, the middleware returns 503 because auth isn't configured.)
    secret = "unit-test-secret"

    def _sample_image_bytes() -> bytes:
        img = Image.new("RGB", (64, 64), color=(200, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    files = {"file": ("eye.jpg", _sample_image_bytes(), "image/jpeg")}

    monkeypatch.setattr(main_mod, "API_KEY", secret)

    # No header -> 401
    r = client.post("/predict/image", files=files)
    assert r.status_code == 401

    # Wrong header -> 401
    r = client.post("/predict/image", files=files, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401

    # Correct header -> endpoint runs; 503 allowed if models aren't available locally.
    r = client.post("/predict/image", files=files, headers={"X-API-Key": secret})
    assert r.status_code in (200, 503)

