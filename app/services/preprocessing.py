"""
services/preprocessing.py
Preprocessing façade consumed by inference endpoints.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.utils.image_utils import extract_lab_features, preprocess_image
from app.utils.tabular_utils import build_nh_vector, build_wh_vector, scale


def preprocess_image_bytes(raw_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    """
    Decode bytes → PIL image → CLAHE → (1, 224, 224, 3) float32 tensor.

    Returns both the model-ready array and the PIL image (for LAB extraction).
    """
    from app.utils.image_utils import load_pil_image
    pil = load_pil_image(raw_bytes)
    return preprocess_image(pil), pil


def build_nh_scaled(
    pil_image: Image.Image,
    age:       float,
    gender:    int,
    feature_probe_scaler: Any,
) -> np.ndarray:
    """
    Extract LAB features + build + scale FEAT_NO_HB vector.

    This is retained for feature extraction parity and internal checks even
    though the shipped multimodal path uses the Hb-augmented severity vector.
    """
    lab_feats = extract_lab_features(pil_image)
    raw       = build_nh_vector(lab_feats, age, gender)
    return scale(raw, feature_probe_scaler), lab_feats


def build_wh_scaled(
    lab_feats:         dict,
    age:               float,
    gender:            int,
    hb_estimated:      float,
    severity_scaler:   Any,
) -> np.ndarray:
    """
    Build + scale FEAT_WITH_HB vector using the image-estimated Hb.

    This is the canonical production feature vector for the sequential
    multimodal severity classifier.
    """
    raw = build_wh_vector(lab_feats, age, gender, hb_estimated)
    return scale(raw, severity_scaler)
