"""
tests/test_inference.py
Validates that the core inference functions work correctly with synthetic inputs.
No real models are required — DummyInterp covers the TFLite interface contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.inference import (
    CLASS_NAMES,
    VISUAL_CLASS_NAMES,
    _run_tflite,
    _get_output,
    _get_hb_output,
    predict_visual,
    predict_rf,
    predict_fusion,
    late_fusion_weighted_average,
    compare_fusion_vs_individuals,
    adapt_tab_array_for_interpreter,
    build_probabilities_dict,
    build_visual_probabilities_dict,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

class DummyVisualInterp:
    """Mimics a visual TFLite model (2-class + 1 Hb output)."""
    def __init__(self, binary_probs=(0.2, 0.8), hb_norm=0.5):
        self._probs = np.array(binary_probs, dtype="float32")
        self._hb    = np.float32(hb_norm)

    def get_input_details(self):
        # Must include "index" key — same contract as the real tf.lite.Interpreter
        return [{"name": "image", "shape": np.array([1, 224, 224, 3]), "dtype": np.float32, "index": 0}]

    def get_output_details(self):
        return [
            {"name": "probs", "index": 0},
            {"name": "hb",    "index": 1},
        ]

    def allocate_tensors(self): pass
    def set_tensor(self, idx, arr): pass
    def invoke(self): pass

    def get_tensor(self, idx):
        if idx == 0:
            return self._probs.reshape(1, -1)
        return np.array([[self._hb]], dtype="float32")


class DummyFusionInterp:
    """Mimics a fusion TFLite model (4-class + 1 Hb output)."""
    def __init__(self, class_probs=(0.1, 0.2, 0.6, 0.1), hb_norm=0.3):
        self._probs = np.array(class_probs, dtype="float32")
        self._hb    = np.float32(hb_norm)

    def get_input_details(self):
        return [
            {"name": "image", "shape": np.array([1, 224, 224, 3]), "dtype": np.float32, "index": 0},
            {"name": "tab",   "shape": np.array([1, 17]),           "dtype": np.float32, "index": 1},
        ]

    def get_output_details(self):
        return [
            {"name": "probs", "index": 0},
            {"name": "hb",    "index": 1},
        ]

    def allocate_tensors(self): pass
    def set_tensor(self, idx, arr): pass
    def invoke(self): pass

    def get_tensor(self, idx):
        if idx == 0:
            return self._probs.reshape(1, -1)
        return np.array([[self._hb]], dtype="float32")


class DummySeverityModel:
    """Mimics a sklearn classifier with predict_proba."""
    def predict_proba(self, X):
        return np.tile(np.array([[0.1, 0.2, 0.6, 0.1]], dtype="float32"), (X.shape[0], 1))


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_class_names_length():
    assert len(CLASS_NAMES) == 4
    assert len(VISUAL_CLASS_NAMES) == 2


def test_predict_visual_returns_valid_tuple():
    interp = DummyVisualInterp(binary_probs=(0.3, 0.7), hb_norm=0.5)
    img = np.zeros((1, 224, 224, 3), dtype="float32")
    pred, conf, probs, hb = predict_visual(img, interp, hb_mean=10.0, hb_std=2.0)

    assert pred in (0, 1), "pred must be 0 or 1"
    assert 0.0 <= conf <= 1.0, "confidence must be in [0, 1]"
    assert probs.shape == (2,), "probs must be length-2 for binary model"
    assert isinstance(hb, float), "hb must be a float"
    # hb_norm=0.5 → 0.5*2.0 + 10.0 = 11.0
    assert abs(hb - 11.0) < 1e-4, f"expected hb=11.0 got {hb}"


def test_predict_visual_argmax_correct():
    interp = DummyVisualInterp(binary_probs=(0.9, 0.1))
    img = np.zeros((1, 224, 224, 3), dtype="float32")
    pred, conf, _, _ = predict_visual(img, interp, 10.0, 2.0)
    assert pred == 0
    assert abs(conf - 0.9) < 1e-5


def test_predict_rf_returns_valid_tuple():
    model = DummySeverityModel()
    features = np.zeros((1, 17), dtype="float32")
    pred, conf, probs = predict_rf(features, model)

    assert pred in range(4), "pred must be a valid class index"
    assert 0.0 <= conf <= 1.0
    assert probs.shape == (4,)
    # DummySeverityModel returns [0.1, 0.2, 0.6, 0.1] → argmax = 2
    assert pred == 2
    assert abs(conf - 0.6) < 1e-5


def test_predict_fusion_returns_four_classes():
    interp = DummyFusionInterp(class_probs=(0.05, 0.1, 0.75, 0.1))
    img = np.zeros((1, 224, 224, 3), dtype="float32")
    tab = np.zeros((1, 17), dtype="float32")
    pred, conf, probs, hb = predict_fusion(img, tab, interp, hb_mean=10.0, hb_std=2.0)

    assert pred in range(4)
    assert 0.0 <= conf <= 1.0
    assert probs.shape == (4,)
    assert pred == 2  # argmax of (0.05, 0.1, 0.75, 0.1)


def test_late_fusion_weighted_average_output_shape():
    v = np.array([0.3, 0.3, 0.3, 0.1], dtype="float32")
    t = np.array([0.1, 0.2, 0.6, 0.1], dtype="float32")
    pred, conf, combined = late_fusion_weighted_average(v, t, w_visual=0.7, w_tab=0.3)

    assert combined.shape == (4,)
    assert abs(combined.sum() - 1.0) < 1e-5, "combined probs must sum to 1"
    assert 0.0 <= conf <= 1.0
    assert pred == int(np.argmax(combined))


def test_late_fusion_weighted_average_shape_mismatch_raises():
    v = np.array([0.5, 0.5], dtype="float32")
    t = np.array([0.25, 0.25, 0.25, 0.25], dtype="float32")
    with pytest.raises(ValueError):
        late_fusion_weighted_average(v, t)


def test_compare_fusion_vs_individuals_benefit():
    # Fusion is more confident than both
    fusion = np.array([0.1, 0.1, 0.8, 0.0], dtype="float32")
    visual = np.array([0.4, 0.6, 0.0, 0.0], dtype="float32")
    tab    = np.array([0.2, 0.2, 0.5, 0.1], dtype="float32")
    result = compare_fusion_vs_individuals(fusion, visual, tab)
    assert result["verdict"] == "fusion_benefit"


def test_compare_fusion_vs_individuals_no_benefit():
    # Fusion is less confident than visual
    fusion = np.array([0.4, 0.3, 0.2, 0.1], dtype="float32")
    visual = np.array([0.1, 0.1, 0.7, 0.1], dtype="float32")
    tab    = np.array([0.2, 0.2, 0.5, 0.1], dtype="float32")
    result = compare_fusion_vs_individuals(fusion, visual, tab)
    assert result["verdict"] == "no_benefit"


def test_adapt_tab_array_clips_extra_columns():
    interp = DummyFusionInterp()  # expects (1, 17)
    wide_tab = np.ones((1, 20), dtype="float32")
    out = adapt_tab_array_for_interpreter(interp, wide_tab)
    assert out.shape == (1, 17)
    # last 17 columns of wide_tab (all ones)
    np.testing.assert_array_equal(out, np.ones((1, 17), dtype="float32"))


def test_adapt_tab_array_pads_short_input():
    interp = DummyFusionInterp()  # expects (1, 17)
    narrow_tab = np.ones((1, 10), dtype="float32")
    out = adapt_tab_array_for_interpreter(interp, narrow_tab)
    assert out.shape == (1, 17)
    # first 10 cols are 1, last 7 are 0
    assert out[0, 9] == 1.0
    assert out[0, 16] == 0.0


def test_build_probabilities_dict_keys():
    probs = np.array([0.1, 0.2, 0.6, 0.1], dtype="float32")
    d = build_probabilities_dict(probs)
    assert set(d.keys()) == set(CLASS_NAMES)
    assert abs(sum(d.values()) - 1.0) < 1e-4


def test_build_visual_probabilities_dict_keys():
    probs = np.array([0.3, 0.7], dtype="float32")
    d = build_visual_probabilities_dict(probs)
    assert set(d.keys()) == set(VISUAL_CLASS_NAMES)


def test_get_output_picks_matching_size():
    tensors = {
        "big":  np.zeros((1, 512)),
        "probs": np.zeros((1, 4)),
    }
    out = _get_output(tensors, n_classes=4)
    assert out.shape == (4,)


def test_get_hb_output_scalar():
    tensors = {"hb": np.array([[0.5]], dtype="float32")}
    hb = _get_hb_output(tensors)
    assert hb is not None
    assert abs(hb - 0.5) < 1e-5
