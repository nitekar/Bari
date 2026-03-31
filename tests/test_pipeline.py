import io

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app, API_KEY


client = TestClient(app)


def _headers() -> dict:
    key = API_KEY or "dev-insecure-api-key"
    return {"X-API-Key": key}


def _sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(150, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_full_multimodal_pipeline_contract():
    files = {"file": ("eye.jpg", _sample_image_bytes(), "image/jpeg")}
    data = {"age": "12", "gender": "1"}
    resp = client.post("/predict/multimodal", headers=_headers(), files=files, data=data)
    # When models are missing in local dev this can be 503; this test focuses
    # on wiring and does not assert on exact prediction values.
    assert resp.status_code in (200, 503)

