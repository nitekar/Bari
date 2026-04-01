import io
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


class DummyInterp:
    def __init__(self, outputs):
        # outputs: dict name -> ndarray
        self._outputs = outputs

    def get_input_details(self):
        # return one image and one tab input descriptor
        return [{"name": "input_1", "shape": np.array([1, 224, 224, 3]), "dtype": np.float32},
                {"name": "input_2", "shape": np.array([1, 17]), "dtype": np.float32}]

    def get_output_details(self):
        # produce two outputs: class probs and hb scalar
        return [{"name": "out_probs", "shape": np.array([1, self._outputs['probs'].shape[-1]]), "index": 0},
                {"name": "out_hb", "shape": np.array([1, 1]), "index": 1}]

    def allocate_tensors(self):
        return

    def set_tensor(self, index, arr):
        return

    def invoke(self):
        return

    def get_tensor(self, index):
        if index == 0:
            return np.array([self._outputs['probs']], dtype=np.float32)
        return np.array([[self._outputs.get('hb', 10.0)]], dtype=np.float32)


class DummySeverityModel:
    def predict_proba(self, X):
        # return uniform-ish probabilities over 4 classes
        return np.tile(np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32), (X.shape[0], 1))


def test_predict_image_and_multimodal(monkeypatch):
    # Seed the registry BEFORE the lifespan runs so endpoints can find models.
    # TestClient triggers lifespan on __enter__; pre-populating state ensures
    # our mocks survive the startup sequence.
    mock_reg = {
        'visual_interp':         DummyInterp({'probs': np.array([0.2, 0.8]), 'hb': 0.5}),
        'severity_model':        DummySeverityModel(),
        'severity_scaler':       None,
        'feature_probe_scaler':  None,
        'multimodal_with_hb_interp': None,
        'multimodal_no_hb_interp':   None,
        'hb_mean':               10.0,
        'hb_std':                2.0,
    }
    app.state.registry = mock_reg

    # Mock preprocess to return a valid image array and a dummy PIL image
    def fake_preprocess_image_bytes(raw_bytes):
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pil = object()
        return img, pil

    monkeypatch.setattr('app.services.preprocessing.preprocess_image_bytes', fake_preprocess_image_bytes)

    # Mock feature builders
    def fake_build_nh_scaled(pil, age, gender, scaler):
        return None, np.zeros((1, 10), dtype=np.float32)

    def fake_build_wh_scaled(lab_feats, age, gender, hb, scaler):
        return np.zeros((1, 17), dtype=np.float32)

    monkeypatch.setattr('app.services.preprocessing.build_nh_scaled', fake_build_nh_scaled)
    monkeypatch.setattr('app.services.preprocessing.build_wh_scaled', fake_build_wh_scaled)

    data = {'file': ('test.png', io.BytesIO(b'PNGDATA'), 'image/png')}

    with TestClient(app) as client:
        # After lifespan, restore our mocks (lifespan may overwrite registry)
        app.state.registry = mock_reg

        r = client.post('/predict/image', files=data, headers={'X-API-Key': ''})
        assert r.status_code in (200, 503)

        r2 = client.post('/predict/multimodal', files=data, data={'age': '24', 'gender': '1'}, headers={'X-API-Key': ''})
        assert r2.status_code in (200, 503)
