import numpy as np

from app.services.inference import predict_rf, build_probabilities_dict, CLASS_NAMES


class _DummyRF:
    def predict_proba(self, X):
        n = len(CLASS_NAMES)
        base = np.linspace(1, n, n, dtype="float32")
        base = base / base.sum()
        return np.tile(base, (X.shape[0], 1))


def test_rf_prediction_and_probabilities():
    X = np.zeros((1, 17), dtype="float32")
    pred_idx, conf, probs = predict_rf(X, _DummyRF())
    assert 0 <= pred_idx < len(CLASS_NAMES)
    assert 0.0 <= conf <= 1.0
    d = build_probabilities_dict(probs)
    assert set(d.keys()) == set(CLASS_NAMES)

