"""
services/inference.py
Core inference engine — all model calls live here.
Stateless: models are injected as arguments (loaded once in main.py lifespan).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("anemia-api.inference")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = ["Non-Anemic", "Mild", "Moderate", "Severe"]

class_map: dict[int, str] = {
    0: "Non-Anemic",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
}


# ── Tabular ───────────────────────────────────────────────────────────────────
def predict_tabular(
    scaled_features: np.ndarray,
    model_with_hb: Any,
    model_no_hb: Any,
    use_hb: bool,
) -> tuple[int, float, np.ndarray]:
    """
    Run sklearn RandomForest inference.

    Parameters
    ----------
    scaled_features : (1, n_features) float32 array
    model_with_hb   : fitted sklearn model (uses HB_LEVEL)
    model_no_hb     : fitted sklearn model (no HB_LEVEL)
    use_hb          : select which model to call

    Returns
    -------
    (pred_class_idx, confidence, probabilities array)
    """
    model = model_with_hb if use_hb else model_no_hb
    probs = model.predict_proba(scaled_features)[0]     # (n_classes,)
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf, probs


# ── TFLite helpers ────────────────────────────────────────────────────────────
def _run_tflite_single(interpreter: Any, input_array: np.ndarray) -> np.ndarray:
    """
    Run a single-input TFLite model.

    Parameters
    ----------
    interpreter  : tf.lite.Interpreter (already allocated)
    input_array  : (1, H, W, C) or (1, n_feat) float32

    Returns
    -------
    probs : (n_classes,) float32
    """
    in_det  = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    interpreter.set_tensor(in_det[0]["index"], input_array.astype(in_det[0]["dtype"]))
    interpreter.invoke()
    return interpreter.get_tensor(out_det[0]["index"])[0]


def _run_tflite_multi(
    interpreter: Any,
    img_array: np.ndarray,
    tab_array: np.ndarray,
) -> np.ndarray:
    """
    Run a dual-input TFLite model (image + tabular).

    The function auto-detects which input tensor corresponds to the image
    and which to tabular features by comparing tensor shapes.

    Returns
    -------
    probs : (n_classes,) float32
    """
    in_dets  = interpreter.get_input_details()
    out_dets = interpreter.get_output_details()

    for det in in_dets:
        shape = det["shape"]
        dtype = det["dtype"]
        if len(shape) == 4:                 # (1, H, W, C) → image
            interpreter.set_tensor(det["index"], img_array.astype(dtype))
        else:                               # (1, n_feat)  → tabular
            interpreter.set_tensor(det["index"], tab_array.astype(dtype))

    interpreter.invoke()
    return interpreter.get_tensor(out_dets[0]["index"])[0]


# ── Image ─────────────────────────────────────────────────────────────────────
def predict_image(
    img_array: np.ndarray,
    visual_interpreter: Any,
) -> tuple[int, float, np.ndarray]:
    """
    TFLite visual model inference.

    Parameters
    ----------
    img_array          : (1, 160, 160, 3) float32
    visual_interpreter : tf.lite.Interpreter (allocated)

    Returns
    -------
    (pred_class_idx, confidence, probabilities)
    """
    probs = _run_tflite_single(visual_interpreter, img_array)
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf, probs


# ── Multimodal ────────────────────────────────────────────────────────────────
def predict_multimodal(
    img_array:           np.ndarray,
    tab_array:           np.ndarray,
    mm_interpreter_wh:   Any,
    mm_interpreter_nonh: Any,
    use_hb:              bool,
) -> tuple[int, float, np.ndarray]:
    """
    TFLite multimodal fusion model inference.

    Selects the correct interpreter based on whether HB_LEVEL was provided.

    Parameters
    ----------
    img_array            : (1, 160, 160, 3) float32
    tab_array            : (1, 3) or (1, 2) float32
    mm_interpreter_wh    : Interpreter for fusion_with_hb  (3 tab features)
    mm_interpreter_nonh  : Interpreter for fusion_no_hb    (2 tab features)
    use_hb               : True → use mm_interpreter_wh

    Returns
    -------
    (pred_class_idx, confidence, probabilities)
    """
    interp = mm_interpreter_wh if use_hb else mm_interpreter_nonh
    probs  = _run_tflite_multi(interp, img_array, tab_array)
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred])
    return pred, conf, probs


# ── SHAP explainability ───────────────────────────────────────────────────────
def explain_tabular(
    scaled_features: np.ndarray,
    model_with_hb:   Any,
    model_no_hb:     Any,
    use_hb:          bool,
    feature_names:   list[str],
) -> dict[str, float]:
    """
    Compute SHAP feature importances for a single tabular prediction.

    Returns
    -------
    Dict mapping feature name → mean absolute SHAP contribution
    across all classes.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for /explain/tabular. "
            "Install with: pip install shap"
        ) from exc

    model    = model_with_hb if use_hb else model_no_hb
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(scaled_features)  # list[n_classes]

    # Mean |SHAP| across classes → single importance score per feature
    if isinstance(shap_vals, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)[0]
    else:
        mean_abs = np.abs(shap_vals)[0]

    return {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs)}


# ── Shared output builders ────────────────────────────────────────────────────
def build_probabilities_dict(probs: np.ndarray) -> dict[str, float]:
    """Map 4-class names to rounded probability values."""
    return {
        name: round(float(p), 6)
        for name, p in zip(CLASS_NAMES, probs)
    }


VISUAL_CLASS_NAMES: list[str] = ["Non-Anemic", "Anemic"]


def build_visual_probabilities_dict(probs: np.ndarray) -> dict[str, float]:
    """Map binary visual class names to rounded probability values."""
    return {
        name: round(float(p), 6)
        for name, p in zip(VISUAL_CLASS_NAMES, probs)
    }
