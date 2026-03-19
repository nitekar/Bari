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

IMG_SIZE = 160                          # training resolution
ACCEPTED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


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
    Resize → centre-crop → normalise to [0, 1].

    Returns
    -------
    np.ndarray  shape (1, 160, 160, 3)  dtype float32
    """
    # Slight oversize for centre crop (matches training pipeline)
    load_size = IMG_SIZE + 20
    img = image.resize((load_size, load_size), Image.LANCZOS)

    # Centre-crop to IMG_SIZE
    left   = (load_size - IMG_SIZE) // 2
    top    = (load_size - IMG_SIZE) // 2
    right  = left + IMG_SIZE
    bottom = top  + IMG_SIZE
    img    = img.crop((left, top, right, bottom))

    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 160, 160, 3)


def validate_image_content_type(content_type: str) -> None:
    """
    Raise ValueError for unsupported image MIME types.
    """
    if content_type.lower() not in ACCEPTED_MIME:
        raise ValueError(
            f"Unsupported image type '{content_type}'. "
            f"Accepted: {sorted(ACCEPTED_MIME)}"
        )
