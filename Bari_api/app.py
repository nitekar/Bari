"""
Bari Anemia Detection API (legacy entry point)
================================================
This file maintains backward compatibility.  For the full multimodal API
with 4-class severity prediction and nutritional guidance, use
``project/api/inference_api.py``.
"""

import json
import os

import tensorflow as tf
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from utils import preprocess_image, read_image, predict_class

app = FastAPI(
    title="Bari Anemia Detection API",
    description="Anemia severity classification from conjunctiva images.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model (gracefully skip if not found — e.g. in CI / testing)
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../Notebook/models/mobilenetv2_finetuned_visual_model.h5"
)
model = None
if os.path.exists(_MODEL_PATH):
    model = tf.keras.models.load_model(_MODEL_PATH)

# 4-class severity labels matching the training convention
CLASSES = ["Normal", "Mild", "Moderate", "Severe"]

# ---------------------------------------------------------------------------
# Load nutrition rules for enriched responses
# ---------------------------------------------------------------------------
_NUTRITION_PATH = os.path.join(
    os.path.dirname(__file__), "../project/nutrition/nutrition_rules.json"
)
_NUTRITION_RULES: dict = {}
if os.path.exists(_NUTRITION_PATH):
    with open(_NUTRITION_PATH) as fh:
        _NUTRITION_RULES = json.load(fh)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def home():
    """Health check."""
    return {
        "message": "Bari Anemia Detection API is running",
        "model_loaded": model is not None,
        "classes": CLASSES,
    }


@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
    age: int = Form(default=0, ge=0, le=1200, description="Patient age in months"),
    gender: str = Form(default="Unknown", description="Male or Female"),
):
    """Predict anemia severity from a conjunctiva image.

    Returns predicted severity class, confidence, and nutritional guidance.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Please train and save the model to "
                f"{_MODEL_PATH} before using this endpoint."
            ),
        )

    image = await read_image(file)
    processed_image = preprocess_image(image)
    label, confidence = predict_class(model, processed_image, CLASSES)

    # Attach nutrition guidance when available
    rule = _NUTRITION_RULES.get(label, {})
    response: dict = {
        "diagnosis": label,
        "confidence": confidence,
    }
    if rule:
        response["nutrition_advice"] = rule.get("diet_recommendation", "")
        response["recommended_foods"] = rule.get("food_list", [])
        response["referral_action"] = rule.get("referral_action", "")

    return response