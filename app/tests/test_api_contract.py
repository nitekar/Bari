import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app, API_KEY


client = TestClient(app)


def _headers() -> dict:
    key = API_KEY or "dev-insecure-api-key"
    return {"X-API-Key": key}


def _sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "models_loaded" in body


def test_predict_image_contract_fields():
    files = {"file": ("eye.jpg", _sample_image_bytes(), "image/jpeg")}
    r = client.post("/predict/image", headers=_headers(), files=files)
    assert r.status_code in (200, 503)  # 503 allowed when models not present locally
    if r.status_code != 200:
        return
    body = r.json()
    for key in (
        "prediction",
        "confidence",
        "confidence_score",
        "risk_level",
        "class_probabilities",
        "recommendations",
    ):
        assert key in body
    rec = body["recommendations"]
    assert rec["diet_plan"]
    assert rec["foods_to_include"]
    assert rec["foods_to_avoid"]
    assert rec["urgency_level"]


def test_predict_multimodal_requires_age_and_gender():
    files = {"file": ("eye.jpg", _sample_image_bytes(), "image/jpeg")}
    data = {"age": "12", "gender": "1"}
    r = client.post("/predict/multimodal", headers=_headers(), files=files, data=data)
    assert r.status_code in (200, 503)
    if r.status_code != 200:
        return
    body = r.json()
    assert body["risk_level"] in ("low", "moderate", "high")
    assert "recommendations" in body

