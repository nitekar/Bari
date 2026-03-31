"""
services/inference.py
Core inference engine — all model calls live here.
Stateless: models/scalers are injected as arguments (loaded once at startup).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("anemia-api.inference")

# ── Class labels — must match notebook CLASS_NAMES ────────────────────────────
CLASS_NAMES:        list[str] = ["Non-Anemic", "Mild", "Moderate", "Severe"]
VISUAL_CLASS_NAMES: list[str] = ["Non-Anemic", "Anemic"]


# ── WHO severity from Hb ──────────────────────────────────────────────────────
def anemia_label(hb: float, age_months: float, gender: str = "") -> str:
    """
    WHO age/gender-aware severity label from Hb (g/dL).
    Matches anemia_label() in the notebook config cell.
    """
    if 6 <= age_months < 24:
        if   hb >= 10.5:           return "Non-Anemic"
        elif 9.5  <= hb < 10.5:    return "Mild"
        elif 7.0  <= hb <  9.5:    return "Moderate"
        else:                       return "Severe"
    elif 24 <= age_months <= 60:
        if   hb >= 11.0:           return "Non-Anemic"
        elif 10.0 <= hb < 11.0:    return "Mild"
        elif 7.0  <= hb < 10.0:    return "Moderate"
        else:                       return "Severe"
    # Age outside 6-60 months: fall back to adult WHO threshold
    return "Non-Anemic" if hb >= 12.0 else ("Mild" if hb >= 11.0 else
           "Moderate" if hb >= 8.0 else "Severe")


# ── TFLite runner ─────────────────────────────────────────────────────────────
def _run_tflite(
    interpreter: Any,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Generic TFLite runner for single or dual-input / dual-output models.

    Parameters
    ----------
    interpreter : tf.lite.Interpreter (already allocated)
    inputs      : dict of {partial_name: array}  e.g. {"image": img_arr}
                  Matched to tensor slots by shape: 4-D → image, 2-D → tabular.

    Returns
    -------
    dict {output_name: np.ndarray}
    """
    in_dets  = interpreter.get_input_details()
    out_dets = interpreter.get_output_details()

    for det in in_dets:
        dtype = det["dtype"]
        shape = det["shape"]
        # Match by dimensionality: 4-D tensor is image, 2-D is tabular
        if len(shape) == 4:
            arr = inputs.get("image") or next(v for v in inputs.values() if v.ndim == 4)
        else:
            arr = inputs.get("tab") or next(v for v in inputs.values() if v.ndim == 2)
        interpreter.set_tensor(det["index"], arr.astype(dtype))

    interpreter.invoke()
    return {det["name"]: interpreter.get_tensor(det["index"]) for det in out_dets}


def _get_output(tensors: dict[str, np.ndarray], n_classes: int) -> np.ndarray:
    """Pick the output tensor with shape matching n_classes."""
    for arr in tensors.values():
        if arr.ndim >= 1 and arr.flatten().shape[0] == n_classes:
            return arr.flatten()
    # Fallback: first tensor
    return next(iter(tensors.values())).flatten()


def _get_hb_output(tensors: dict[str, np.ndarray]) -> float | None:
    """Pick the scalar Hb output tensor (shape (1,1) or (1,))."""
    for arr in tensors.values():
        if arr.size == 1:
            return float(arr.flatten()[0])
    return None


# ── Visual model inference ────────────────────────────────────────────────────
def predict_visual(
    img_array: np.ndarray,
    interpreter: Any,
    hb_mean: float,
    hb_std:  float,
) -> tuple[int, float, np.ndarray, float]:
    """
    Run the visual TFLite model (binary cls + Hb regression).

    Returns
    -------
    (binary_pred_idx, confidence, binary_probs, hb_estimated_gdl)
    """
    tensors    = _run_tflite(interpreter, {"image": img_array})
    cls_probs  = _get_output(tensors, n_classes=2)
    hb_norm    = _get_hb_output(tensors)
    hb_gdl     = (hb_norm * hb_std + hb_mean) if hb_norm is not None else 0.0
    pred       = int(np.argmax(cls_probs))
    conf       = float(cls_probs[pred])
    return pred, conf, cls_probs, hb_gdl


# ── Fusion model inference ────────────────────────────────────────────────────
def predict_fusion(
    img_array:  np.ndarray,
    tab_array:  np.ndarray,
    interpreter: Any,
    hb_mean:    float,
    hb_std:     float,
) -> tuple[int, float, np.ndarray, float]:
    """
    Run the fusion TFLite model for the currently deployed artifact.

    This path is intentionally treated as experimental by the API until the
    exported model contract is validated against training.

    Returns
    -------
    (severity_pred_idx, confidence, severity_probs, hb_estimated_gdl)
    """
    tensors    = _run_tflite(interpreter, {"image": img_array, "tab": tab_array})
    cls_probs  = _get_output(tensors, n_classes=4)
    hb_norm    = _get_hb_output(tensors)
    hb_gdl     = (hb_norm * hb_std + hb_mean) if hb_norm is not None else 0.0
    pred       = int(np.argmax(cls_probs))
    conf       = float(cls_probs[pred])
    return pred, conf, cls_probs, hb_gdl


# ── Visual → RF pipeline ──────────────────────────────────────────────────────
def predict_rf(
    scaled_wh_features: np.ndarray,
    severity_model:     Any,
) -> tuple[int, float, np.ndarray]:
    """
    Run the production severity classifier on pre-scaled FEAT_WITH_HB features.

    Returns
    -------
    (severity_pred_idx, confidence, severity_probs)
    """
    probs = severity_model.predict_proba(scaled_wh_features)[0]
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return pred, conf, probs


# ── Output builders ───────────────────────────────────────────────────────────
def build_probabilities_dict(probs: np.ndarray) -> dict[str, float]:
    return {name: round(float(p), 6) for name, p in zip(CLASS_NAMES, probs)}


def build_visual_probabilities_dict(probs: np.ndarray) -> dict[str, float]:
    return {name: round(float(p), 6) for name, p in zip(VISUAL_CLASS_NAMES, probs)}
