"""
utils/tabular_utils.py
Helpers for building and scaling tabular feature vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ── Feature order must match training pipeline (Bari.ipynb Part 4) ───────────
FEAT_WITH_HB: list[str] = ["HB_LEVEL", "Age(Months)", "Gender_F"]
FEAT_NO_HB:   list[str] = ["Age(Months)", "Gender_F"]


def build_feature_vector(
    age: float,
    gender: int,
    hb_level: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Build a raw (unscaled) feature vector.

    Parameters
    ----------
    age      : age in months
    gender   : 0 = Male, 1 = Female  (Gender_F = float(gender))
    hb_level : haemoglobin g/dL (optional)

    Returns
    -------
    (feature_array, use_hb)
        feature_array : shape (1, n_features)  dtype float64
        use_hb        : True if hb_level was provided
    """
    gender_f = float(gender)           # Female=1.0, Male=0.0
    use_hb   = hb_level is not None

    if use_hb:
        vec = np.array([[hb_level, age, gender_f]], dtype="float64")
    else:
        vec = np.array([[age, gender_f]], dtype="float64")

    return vec, use_hb


def scale_features(raw: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Apply a fitted sklearn StandardScaler.

    Parameters
    ----------
    raw    : shape (1, n_features)
    scaler : fitted sklearn.preprocessing.StandardScaler

    Returns
    -------
    scaled : shape (1, n_features)  dtype float32
    """
    return scaler.transform(raw).astype("float32")


def build_and_scale(
    age: float,
    gender: int,
    scaler_with: Any,
    scaler_no:   Any,
    hb_level: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Convenience wrapper: build vector → select scaler → scale.

    Returns
    -------
    (scaled_array, use_hb)
    """
    raw, use_hb = build_feature_vector(age, gender, hb_level)
    scaler      = scaler_with if use_hb else scaler_no
    scaled      = scale_features(raw, scaler)
    return scaled, use_hb
