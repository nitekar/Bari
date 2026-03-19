"""
services/preprocessing.py
Unified preprocessing façade consumed by inference endpoints.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.utils.image_utils import preprocess_image
from app.utils.tabular_utils import build_and_scale


def preprocess_tabular(
    age: float,
    gender: int,
    scaler_with: Any,
    scaler_no:   Any,
    hb_level: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Preprocess tabular inputs.

    Applies StandardScaler fitted during training.
    Gender is encoded as a binary float (Female=1, Male=0) — one-hot for
    a 2-class variable reduces to a single binary feature.

    Returns
    -------
    (scaled_array shape (1, n_features), use_hb flag)
    """
    return build_and_scale(
        age=age,
        gender=gender,
        scaler_with=scaler_with,
        scaler_no=scaler_no,
        hb_level=hb_level,
    )


def preprocess_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Decode bytes → PIL.Image → (1, 160, 160, 3) float32 array.
    """
    from app.utils.image_utils import load_pil_image
    pil = load_pil_image(raw_bytes)
    return preprocess_image(pil)
