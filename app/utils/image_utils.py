"""
utils/image_utils.py
Low-level image loading, resizing, and normalisation helpers.
"""
from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("anemia-api.image")

IMG_SIZE       = 224    # MobileNetV3Small native input size
ROI_CROP_RATIO = 0.70   # central crop fraction — matches training pipeline
ACCEPTED_MIME  = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def load_pil_image(raw_bytes: bytes) -> Image.Image:
    """
    Decode raw bytes into a PIL Image (RGB).

    Raises
    ------
    ValueError  : if bytes cannot be decoded as an image
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.verify()                    # detect truncated files early
        # re-open: verify() closes the file pointer
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return img
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Image decode error: {exc}") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Central-crop (70 %) → resize 224×224 → MobileNetV3 preprocess_input.

    Matches the training pipeline exactly:
        tf.image.central_crop(0.70) → tf.image.resize(224) →
        mobilenet_v3.preprocess_input  →  pixels in [-1, 1]

    Returns
    -------
    np.ndarray  shape (1, 224, 224, 3)  dtype float32
    """
    w, h = image.size
    crop_w = int(w * ROI_CROP_RATIO)
    crop_h = int(h * ROI_CROP_RATIO)
    left   = (w - crop_w) // 2
    top    = (h - crop_h) // 2
    image  = image.crop((left, top, left + crop_w, top + crop_h))

    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr   = np.array(image, dtype="float32")    # [0, 255] RGB
    arr   = arr / 127.5 - 1.0                   # MobileNetV3: scale to [-1, 1]
    return np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)


def validate_image_content_type(content_type: str) -> None:
    """
    Raise ValueError for unsupported image MIME types.
    """
    if content_type.lower() not in ACCEPTED_MIME:
        raise ValueError(
            f"Unsupported image type '{content_type}'. "
            f"Accepted: {sorted(ACCEPTED_MIME)}"
        )
