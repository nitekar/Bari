"""
utils/image_utils.py
Image loading, CLAHE normalisation, LAB feature extraction, and preprocessing.
"""
from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("anemia-api.image")

IMG_SIZE      = 224
ACCEPTED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def load_pil_image(raw_bytes: bytes) -> Image.Image:
    """Decode raw bytes into a PIL Image (RGB)."""
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.verify()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return img
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Image decode error: {exc}") from exc


def _apply_clahe(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to L channel of LAB colour space.
    Input/output: (H, W, 3) uint8 RGB.
    Matches the notebook's _apply_clahe_np exactly.
    """
    img_u8  = np.clip(arr_rgb, 0, 255).astype(np.uint8)
    lab     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq    = clahe.apply(L)
    lab_eq  = cv2.merge([L_eq, A, B])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def extract_lab_features(pil_image: Image.Image) -> dict[str, float]:
    """
    Extract the 14 LAB/colour/texture features used by the RF model.
    Matches extract_lab_features() in the notebook exactly.

    Features (FEAT_NO_HB without demo cols):
      a_mean, a_std, b_mean, G_mean, l_mean, lap_var  — scalar colour
      a_hist_0 … a_hist_7                              — 8-bin a* histogram

    Parameters
    ----------
    pil_image : PIL.Image  (RGB, any size — will be resized to 224×224)

    Returns
    -------
    dict mapping feature name → float value
    """
    img_rgb = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    bgr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq    = clahe.apply(L)
    lab_eq  = cv2.merge([L_eq, A, B])
    _, A_eq, B_eq = cv2.split(lab_eq)

    G = img_rgb[:, :, 1]   # Green channel from RGB

    hist, _ = np.histogram(A_eq.flatten(), bins=8, range=(0, 256))
    hist_n  = hist.astype("float32") / (hist.sum() + 1e-8)

    lap_var = float(cv2.Laplacian(L_eq, cv2.CV_64F).var())

    feats: dict[str, float] = {
        "a_mean" : float(A_eq.mean()),
        "a_std"  : float(A_eq.std()),
        "b_mean" : float(B_eq.mean()),
        "G_mean" : float(G.mean()),
        "l_mean" : float(L_eq.mean()),
        "lap_var": lap_var,
    }
    for i, v in enumerate(hist_n):
        feats[f"a_hist_{i}"] = float(v)

    return feats


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize 224×224 → CLAHE → MobileNetV3 preprocess_input (scale to [-1, 1]).
    Matches the notebook's load_image_raw + apply_clahe_tf + preprocess_input pipeline.

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3)  dtype float32
    """
    arr     = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    arr     = _apply_clahe(arr)                      # CLAHE on L channel
    arr_f   = arr.astype("float32") / 127.5 - 1.0   # MobileNetV3: scale to [-1, 1]
    return np.expand_dims(arr_f, axis=0)             # (1, 224, 224, 3)


def validate_image_content_type(content_type: str) -> None:
    if content_type.lower() not in ACCEPTED_MIME:
        raise ValueError(
            f"Unsupported image type '{content_type}'. "
            f"Accepted: {sorted(ACCEPTED_MIME)}"
        )
