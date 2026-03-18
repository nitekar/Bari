"""
Anemia Screening Inference API
================================
FastAPI REST endpoint for multimodal anemia severity prediction.

Endpoint:
    POST /predict

Input (multipart/form-data):
    - file   : conjunctiva image (JPEG or PNG)
    - age    : patient age in months (integer)
    - gender : "Male" or "Female"

Output (JSON):
    - diagnosis       : predicted severity class (Normal / Mild / Moderate / Severe)
    - confidence      : model confidence score [0, 1]
    - nutrition_advice: brief dietary recommendation
    - recommended_foods: list of recommended foods
    - referral_action : referral / follow-up instruction
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("anemia_api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
API_DIR      = Path(__file__).resolve().parent
PROJECT_DIR  = API_DIR.parent
NUTRITION_PATH = PROJECT_DIR / "nutrition" / "nutrition_rules.json"
MODELS_DIR   = PROJECT_DIR / "models" / "saved_models"

# ---------------------------------------------------------------------------
# Load nutrition rules
# ---------------------------------------------------------------------------
with open(NUTRITION_PATH, "r") as fh:
    NUTRITION_RULES: dict = json.load(fh)

CLASSES = ["Normal", "Mild", "Moderate", "Severe"]
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Lazy-load model (avoids long import times at module load)
# ---------------------------------------------------------------------------
_model = None


def get_model():
    """Load and cache the Keras model.  Falls back gracefully when no saved
    model exists so that the API can start (useful in CI/testing).

    Returns:
        Loaded Keras model, or None if no model file is found.
    """
    global _model
    if _model is not None:
        return _model

    candidates = [
        MODELS_DIR / "fusion_model_best.h5",
        MODELS_DIR / "visual_model_finetuned.h5",
        # legacy path kept for backward-compatibility
        PROJECT_DIR.parent / "Notebook" / "models" / "mobilenetv2_finetuned_visual_model.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                import tensorflow as tf  # import deferred to keep startup fast
                _model = tf.keras.models.load_model(str(candidate))
                logger.info("Model loaded from: %s", candidate)
                return _model
            except Exception as exc:
                logger.warning("Could not load model from %s: %s", candidate, exc)

    logger.warning(
        "No trained model found. /predict will return demo responses. "
        "Train and save a model to %s to enable real inference.",
        MODELS_DIR,
    )
    return None


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize a PIL image to 224×224 and normalise pixels to [0, 1].

    Args:
        image: PIL Image object (any mode; will be converted to RGB).

    Returns:
        Float32 numpy array of shape (1, 224, 224, 3).
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    image: Image.Image,
    age_months: int,
    gender: str,
) -> dict:
    """Run the full inference pipeline.

    Args:
        image: Raw PIL image of the patient's conjunctiva.
        age_months: Patient age in months.
        gender: "Male" or "Female".

    Returns:
        Dictionary with keys: diagnosis, confidence, class_probabilities,
        nutrition_advice, recommended_foods, referral_action.
    """
    model = get_model()
    img_array = preprocess_image(image)

    # Encode gender: Female=0, Male=1
    gender_enc = 0.0 if gender.strip().lower() in ("female", "f") else 1.0
    # Scale age roughly (mean ~30 months, std ~20 in our dataset — best-effort)
    age_scaled = (float(age_months) - 30.0) / 20.0

    if model is None:
        # Demo mode — return deterministic dummy prediction
        logger.warning("Running in demo mode (no model loaded).")
        demo_class = "Mild"
        return _build_response(demo_class, 0.75, [0.10, 0.75, 0.10, 0.05])

    # Determine whether the loaded model is multimodal (two inputs) or visual only
    n_inputs = len(model.inputs)

    if n_inputs >= 2:
        # Multimodal model: [image, tabular]
        tabular = np.array([[age_scaled, gender_enc]], dtype=np.float32)
        probs = model.predict([img_array, tabular], verbose=0)[0]
    else:
        # Visual-only model
        probs = model.predict(img_array, verbose=0)[0]

    # Normalise in case the model has more/fewer output neurons than expected
    if len(probs) == len(CLASSES):
        class_probs = probs.tolist()
    elif len(probs) == 2:
        # Binary legacy model — map to 4-class heuristically
        anemic_prob = float(probs[0])
        # Distribute anemic probability across Mild / Moderate / Severe proportionally
        class_probs = [
            1 - anemic_prob,
            anemic_prob * 0.40,
            anemic_prob * 0.40,
            anemic_prob * 0.20,
        ]
    else:
        # Unexpected output — softmax over available outputs
        soft = np.exp(probs) / np.exp(probs).sum()
        class_probs = (soft[:4] / soft[:4].sum()).tolist()

    predicted_idx = int(np.argmax(class_probs))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(class_probs[predicted_idx])

    return _build_response(predicted_class, confidence, class_probs)


def _build_response(diagnosis: str, confidence: float, class_probs: list) -> dict:
    """Assemble the API response dict for a given diagnosis."""
    rule = NUTRITION_RULES.get(diagnosis, NUTRITION_RULES["Normal"])
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(CLASSES, class_probs)
        },
        "nutrition_advice": rule["diet_recommendation"],
        "recommended_foods": rule["food_list"],
        "referral_action": rule["referral_action"],
        "urgency": rule.get("urgency", "low"),
    }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bari Anemia Screening API",
    description=(
        "Multimodal AI system for anemia severity prediction from conjunctiva images "
        "and patient metadata, with nutritional guidance and referral recommendations."
    ),
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
# Response schema
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    class_probabilities: dict
    nutrition_advice: str
    recommended_foods: List[str]
    referral_action: str
    urgency: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def home():
    """Health check endpoint."""
    return {
        "message": "Bari Anemia Screening API is running",
        "version": "1.0.0",
        "classes": CLASSES,
    }


@app.get("/health", tags=["health"])
def health():
    """Detailed health status including model availability."""
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASSES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict_endpoint(
    file: UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
    age: int = Form(..., ge=0, le=1200, description="Patient age in months"),
    gender: str = Form(..., description="Patient gender: Male or Female"),
):
    """Predict anemia severity from a conjunctiva image and patient metadata.

    Returns the predicted severity class, model confidence, per-class
    probabilities, nutritional guidance, and referral recommendations.
    """
    # Validate gender
    if gender.strip().lower() not in ("male", "m", "female", "f"):
        raise HTTPException(
            status_code=422,
            detail="gender must be 'Male' or 'Female'",
        )

    # Validate file type
    allowed_content_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type and file.content_type.lower() not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        logger.error("Failed to read image: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    # Run inference
    try:
        result = predict(image, age_months=age, gender=gender)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    logger.info(
        "Prediction | age=%d gender=%s | %s (%.3f)",
        age, gender, result["diagnosis"], result["confidence"]
    )
    return result
