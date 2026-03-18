"""
Anemia Screening System — FastAPI Inference API
================================================
Run with:
    uvicorn api.inference_api:app --reload

Endpoints:
    POST /predict  — Accept image + patient data, return diagnosis & recommendations
    GET  /health   — Health check
"""

import io
import json
import os
import sys
import tempfile
from typing import List

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, validator

# ---------------------------------------------------------------------------
# Paths (relative to project root — adjust as needed)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "anemia_multimodal.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "saved_models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "saved_models", "label_encoder.pkl")
NUTRITION_PATH = os.path.join(BASE_DIR, "nutrition", "nutrition_rules.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Mild", "Moderate", "Severe"]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Anemia Screening API",
    description="Multimodal anemia severity classification from conjunctiva images and patient data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model / transformer holders
# ---------------------------------------------------------------------------
_model = None
_scaler = None
_encoder = None
_nutrition_rules = None


def _load_assets():
    """Lazily load model and transformers on first request."""
    global _model, _scaler, _encoder, _nutrition_rules

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)

    if _encoder is None:
        if not os.path.exists(ENCODER_PATH):
            raise RuntimeError(f"Label encoder file not found: {ENCODER_PATH}")
        _encoder = joblib.load(ENCODER_PATH)

    if _nutrition_rules is None:
        with open(NUTRITION_PATH, "r") as fh:
            _nutrition_rules = json.load(fh)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    nutrition_advice: str
    recommended_foods: List[str]
    referral_action: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes, resize to (224, 224) and normalize to [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def _preprocess_tabular(age: int, gender: str) -> np.ndarray:
    """Encode gender and scale age using saved transformers."""
    gender_encoded = _encoder.transform([gender])[0]
    age_scaled = _scaler.transform([[age]])[0][0]
    return np.array([[age_scaled, gender_encoded]], dtype=np.float32)


def _get_nutrition(diagnosis: str, confidence: float) -> dict:
    """Return structured nutritional recommendation for the given diagnosis."""
    rule = _nutrition_rules.get(diagnosis, _nutrition_rules["Normal"])
    return {
        "diagnosis": diagnosis,
        "confidence": round(float(confidence), 4),
        "nutrition_advice": rule["diet_recommendation"],
        "recommended_foods": rule["food_list"],
        "referral_action": rule["referral_action"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health():
    """Return service status."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict anemia severity")
async def predict(
    image: UploadFile = File(..., description="Conjunctiva image (JPEG or PNG)"),
    age: int = Form(..., ge=0, le=120, description="Patient age in years"),
    gender: str = Form(..., description='Patient gender: "Male" or "Female"'),
):
    """
    Predict anemia severity from a conjunctiva image and patient metadata.

    - **image**: JPEG or PNG conjunctiva image
    - **age**: patient age in years (0–120)
    - **gender**: "Male" or "Female"
    """
    # Validate gender
    if gender not in ("Male", "Female"):
        raise HTTPException(
            status_code=422,
            detail='gender must be "Male" or "Female"',
        )

    # Validate content type
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail="image must be JPEG or PNG",
        )

    try:
        _load_assets()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    image_bytes = await image.read()
    img_array = _preprocess_image(image_bytes)
    tab_array = _preprocess_tabular(age, gender)

    probs = _model.predict([img_array, tab_array], verbose=0)[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    diagnosis = CLASS_NAMES[class_idx]

    return _get_nutrition(diagnosis, confidence)
