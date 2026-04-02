"""
utils/tabular_utils.py
Feature vector construction and scaling for the sequential multimodal
severity pipeline. Feature order must match the shipped model artifact.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ── Feature lists — must match notebook config cell (14e69070) ────────────────
_COLOUR_FEATS: list[str] = ["a_mean", "a_std", "b_mean", "G_mean", "l_mean", "lap_var"]
_HIST_FEATS:   list[str] = [f"a_hist_{i}" for i in range(8)]
_DEMO_FEATS:   list[str] = ["Age(Months)", "Gender_F"]

FEAT_NO_HB:   list[str] = _COLOUR_FEATS + _HIST_FEATS + _DEMO_FEATS          # 16 features
FEAT_WITH_HB: list[str] = _COLOUR_FEATS + _HIST_FEATS + ["HB_LEVEL"] + _DEMO_FEATS  # 17 features

_N_COLOUR_HIST = len(_COLOUR_FEATS) + len(_HIST_FEATS)   # 14
_HB_IDX_WH    = FEAT_WITH_HB.index("HB_LEVEL")           # 14


def build_nh_vector(lab_feats: dict[str, float], age: float, gender: int) -> np.ndarray:
    """
    Build FEAT_NO_HB feature vector.

    Parameters
    ----------
    lab_feats : output of extract_lab_features()
    age       : age in months
    gender    : 0=Male, 1=Female

    Returns
    -------
    np.ndarray  shape (1, 16)  dtype float32
    """
    row = {**lab_feats, "Age(Months)": float(age), "Gender_F": float(gender)}
    return np.array([[row.get(f, 0.0) for f in FEAT_NO_HB]], dtype="float32")


def build_wh_vector(
    lab_feats: dict[str, float],
    age: float,
    gender: int,
    hb_estimated: float,
) -> np.ndarray:
    """
    Build FEAT_WITH_HB feature vector for the production severity classifier
    after the visual CNN estimates Hb.

    Parameters
    ----------
    lab_feats    : output of extract_lab_features()
    age          : age in months
    gender       : 0=Male, 1=Female
    hb_estimated : Hb in g/dL estimated by the visual model

    Returns
    -------
    np.ndarray  shape (1, 17)  dtype float32
    """
    row = {
        **lab_feats,
        "HB_LEVEL":    hb_estimated,
        "Age(Months)": float(age),
        "Gender_F":    float(gender),
    }
    return np.array([[row.get(f, 0.0) for f in FEAT_WITH_HB]], dtype="float32")


def scale(raw: np.ndarray, scaler: Any) -> np.ndarray:
    """Apply a fitted sklearn StandardScaler → float32. Pass-through if scaler is None."""
    if scaler is None:
        return raw.astype("float32")
    return scaler.transform(raw).astype("float32")
