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


# ── Fusion helpers: late & early fusion implementations ───────────────────
def late_fusion_weighted_average(
    visual_probs: np.ndarray,
    tab_probs: np.ndarray,
    w_visual: float = 0.5,
    w_tab: float = 0.5,
) -> tuple[int, float, np.ndarray]:
    """
    Combine two probability vectors by weighted averaging.

    Ensures inputs are 1-D arrays of the same length. Returns
    (pred_idx, confidence, combined_probs).
    """
    v = np.asarray(visual_probs, dtype="float32").flatten()
    t = np.asarray(tab_probs, dtype="float32").flatten()
    if v.shape != t.shape:
        raise ValueError("visual_probs and tab_probs must have the same shape")

    # Normalize both vectors to sum to 1 to be robust to minor numerical issues
    v = v / (v.sum() + 1e-12)
    t = t / (t.sum() + 1e-12)

    combined = w_visual * v + w_tab * t
    combined = combined / (combined.sum() + 1e-12)

    pred = int(np.argmax(combined))
    conf = float(combined[pred])
    return pred, conf, combined


def early_fusion_concat_normalize(
    visual_embedding: np.ndarray,
    tab_vector: np.ndarray,
) -> np.ndarray:
    """
    Simple early-fusion feature builder.

    Normalises the visual embedding and tabular vector (L2) and concatenates
    into a single 2-D row suitable for downstream classifiers.
    """
    v = np.asarray(visual_embedding, dtype="float32")
    t = np.asarray(tab_vector, dtype="float32")

    # Flatten to 1-D
    v = v.flatten()
    t = t.flatten()

    v_norm = np.linalg.norm(v) + 1e-12
    t_norm = np.linalg.norm(t) + 1e-12

    v_scaled = v / v_norm
    t_scaled = t / t_norm

    fused = np.concatenate([v_scaled, t_scaled], axis=0)
    return fused.reshape(1, -1).astype("float32")


def compare_fusion_vs_individuals(
    fusion_probs: np.ndarray,
    visual_probs: np.ndarray,
    tab_probs: np.ndarray,
) -> dict:
    """
    Lightweight comparator that checks whether fusion improves the top-class
    confidence compared to individual models. Returns a dict with verdict and
    delta scores.
    """
    fusion = np.asarray(fusion_probs, dtype="float32").flatten()
    v = np.asarray(visual_probs, dtype="float32").flatten()
    t = np.asarray(tab_probs, dtype="float32").flatten()

    # Ensure same length before comparisons — compare on available overlap
    min_len = min(len(fusion), len(v), len(t))
    fusion = fusion[:min_len] / (fusion[:min_len].sum() + 1e-12)
    v = v[:min_len] / (v[:min_len].sum() + 1e-12)
    t = t[:min_len] / (t[:min_len].sum() + 1e-12)

    fusion_top = float(fusion.max())
    visual_top = float(v.max())
    tab_top = float(t.max())

    better_than_visual = fusion_top > visual_top + 1e-6
    better_than_tab = fusion_top > tab_top + 1e-6

    verdict = "fusion_benefit" if (better_than_visual and better_than_tab) else "no_benefit"

    return {
        "verdict": verdict,
        "fusion_top": fusion_top,
        "visual_top": visual_top,
        "tab_top": tab_top,
    }


def adapt_tab_array_for_interpreter(interpreter: Any, tab_array: np.ndarray) -> np.ndarray:
    """
    Adapt a provided tabular array to match the interpreter's expected 2-D input shape.

    If the interpreter expects fewer columns than provided, the function will take
    the last N columns (assumes those are HB/age/gender tail in FEAT_WITH_HB).
    If more columns are expected, it will pad with zeros on the right.
    Returns an array shaped exactly as the interpreter expects.
    """
    in_dets = interpreter.get_input_details()
    # Find a non-4D input (tabular)
    tab_det = None
    for det in in_dets:
        if len(det["shape"]) != 4:
            tab_det = det
            break
    if tab_det is None:
        # No tabular input expected; return an empty array
        return np.zeros((1, 0), dtype="float32")

    expected_shape = tuple(tab_det["shape"])
    # Usually expected_shape is like (1, N)
    if len(expected_shape) == 2:
        expected_cols = expected_shape[1]
        provided = np.asarray(tab_array, dtype="float32")
        if provided.ndim == 1:
            provided = provided.reshape(1, -1)
        # If provided has more columns, take the rightmost columns
        if provided.shape[1] >= expected_cols:
            out = provided[:, -expected_cols:]
        else:
            # pad with zeros
            pad = np.zeros((provided.shape[0], expected_cols - provided.shape[1]), dtype="float32")
            out = np.concatenate([provided, pad], axis=1)
        return out.astype("float32")
    # Fallback: return as float32
    return np.asarray(tab_array, dtype="float32")


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
