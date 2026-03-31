import io

import numpy as np
from PIL import Image

from app.services.preprocessing import preprocess_image_bytes
from app.services.inference import predict_rf, build_probabilities_dict, CLASS_NAMES


def _sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(120, 120, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_bytes_shape_and_dtype():
    raw = _sample_image_bytes()
    arr, pil = preprocess_image_bytes(raw)
    assert arr.shape[1:] == (224, 224, 3)
    assert arr.dtype == np.float32
    assert pil.size[0] > 0 and pil.size[1] > 0


class _DummyRF:
    def predict_proba(self, X):
        n = len(CLASS_NAMES)
        base = np.linspace(1, n, n, dtype="float32")
        base = base / base.sum()
        return np.tile(base, (X.shape[0], 1))


def test_predict_rf_and_probabilities_dict():
    X = np.zeros((1, 17), dtype="float32")
    pred_idx, conf, probs = predict_rf(X, _DummyRF())
    assert 0 <= pred_idx < len(CLASS_NAMES)
    assert 0.0 <= conf <= 1.0
    d = build_probabilities_dict(probs)
    assert set(d.keys()) == set(CLASS_NAMES)
    assert abs(sum(d.values()) - 1.0) < 1e-3

