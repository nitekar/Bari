import io
import numpy as np
from fastapi.testclient import TestClient

from app.main import app, API_KEY


class DummyVisualInterp:
    """Simulates a visual-only TFLite model with image input, binary cls + Hb output."""
    def __init__(self, outputs):
        self._outputs = outputs

    def get_input_details(self):
        return [{"name": "input_image", "shape": np.array([1, 224, 224, 3]), "dtype": np.float32, "index": 0}]

    def get_output_details(self):
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


class DummyFusionInterp:
    """Simulates a fusion TFLite model with image + tabular inputs."""
    def __init__(self, outputs):
        self._outputs = outputs

    def get_input_details(self):
        return [{"name": "input_image", "shape": np.array([1, 224, 224, 3]), "dtype": np.float32, "index": 0},
                {"name": "input_tab", "shape": np.array([1, 17]), "dtype": np.float32, "index": 1}]

    def get_output_details(self):
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
    client = TestClient(app, raise_server_exceptions=False)

    # Mock preprocess to return a valid image array and a dummy PIL image
    def fake_preprocess_image_bytes(raw_bytes):
        img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pil = object()
        return img, pil

    # Patch at the import location in api.routes (where the functions are used)
    monkeypatch.setattr('api.routes.preprocess_image_bytes', fake_preprocess_image_bytes)

    # Mock feature builders
    def fake_build_nh_scaled(pil, age, gender, scaler):
        return np.zeros((1, 16), dtype=np.float32), {"a_mean": 0.0, "a_std": 0.0, "b_mean": 0.0, "G_mean": 0.0, "l_mean": 0.0, "lap_var": 0.0, "a_hist_0": 0.0, "a_hist_1": 0.0, "a_hist_2": 0.0, "a_hist_3": 0.0, "a_hist_4": 0.0, "a_hist_5": 0.0, "a_hist_6": 0.0, "a_hist_7": 0.0}

    def fake_build_wh_scaled(lab_feats, age, gender, hb, scaler):
        return np.zeros((1, 17), dtype=np.float32)

    monkeypatch.setattr('api.routes.build_nh_scaled', fake_build_nh_scaled)
    monkeypatch.setattr('api.routes.build_wh_scaled', fake_build_wh_scaled)

    # Prepare registry with dummy models -- initialize if lifespan hasn't run
    if not hasattr(app.state, 'registry'):
        app.state.registry = {}
    reg = app.state.registry
    reg['visual_interp'] = DummyVisualInterp({'probs': np.array([0.2, 0.8]), 'hb': 0.5})
    reg['severity_model'] = DummySeverityModel()
    reg['severity_scaler'] = None
    reg['feature_probe_scaler'] = None
    reg['hb_mean'] = 10.0
    reg['hb_std'] = 2.0

    # Ensure app.extra has max_upload_bytes
    if not hasattr(app, 'extra') or app.extra is None:
        app.extra = {}
    app.extra['max_upload_bytes'] = 10 * 1024 * 1024

    _api_key = API_KEY or "dev-insecure-api-key"

    # Test /predict/image
    data = {
        'file': ('test.png', io.BytesIO(b'PNGDATA'), 'image/png')
    }
    r = client.post('/predict/image', files=data, headers={'X-API-Key': _api_key})
    assert r.status_code in (200, 422, 503), f"Unexpected status {r.status_code}: {r.text}"

    # Test /predict/multimodal
    data2 = {
        'file': ('test.png', io.BytesIO(b'PNGDATA'), 'image/png')
    }
    r2 = client.post('/predict/multimodal', files=data2, data={'age': '24', 'gender': '1'}, headers={'X-API-Key': _api_key})
    assert r2.status_code in (200, 422, 503), f"Unexpected status {r2.status_code}: {r2.text}"
