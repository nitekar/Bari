"""
main.py — Anemia Screening REST API
=====================================
Endpoints
---------
GET  /health               Health check
POST /predict/multimodal   TFLite fusion model (image + tabular)

Run
---
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Local imports ─────────────────────────────────────────────────────────────
from app.schemas.response import HealthResponse, PredictionResponse
from app.services.inference import build_probabilities_dict, predict_multimodal
from app.services.nutrition import get_full_guidance
from app.services.preprocessing import preprocess_image_bytes, preprocess_tabular
from app.utils.image_utils import validate_image_content_type

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("anemia-api")

# ── Model paths ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MODEL_PATHS: dict[str, str] = {
    "tab_wh":  os.environ.get("TAB_WH_PATH",  os.path.join(_ROOT, "Notebook", "models", "tabular_with_hb.pkl")),
    "tab_nh":  os.environ.get("TAB_NH_PATH",  os.path.join(_ROOT, "Notebook", "models", "tabular_no_hb.pkl")),
    "mm_wh":   os.environ.get("MM_WH_PATH",   os.path.join(_ROOT, "models", "saved_models", "multimodal_model.tflite")),
    "mm_nh":   os.environ.get("MM_NH_PATH",   os.path.join(_ROOT, "models", "saved_models", "multimodal_no_hb_model.tflite")),
}

# ── Model registry (populated at startup) ────────────────────────────────────
_registry: dict[str, Any] = {
    "scaler_wh":    None,
    "scaler_nh":    None,
    "mm_wh_interp": None,
    "mm_nh_interp": None,
}


# ── Lifespan: load models once at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models …")

    # Load scalers from pkl bundles (needed for tabular preprocessing)
    for key, path_key in [("tab_wh", "tab_wh"), ("tab_nh", "tab_nh")]:
        path = MODEL_PATHS[path_key]
        if os.path.exists(path):
            bundle = joblib.load(path)
            scaler_key = f"scaler_{'wh' if key == 'tab_wh' else 'nh'}"
            _registry[scaler_key] = bundle.get("scaler") if isinstance(bundle, dict) else None
            logger.info(f"  [OK] scaler from {path_key}")
        else:
            logger.warning(f"  [MISSING] {path_key} → {path}")

    # Load multimodal TFLite interpreters
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter

        for reg_key, path_key in [("mm_wh_interp", "mm_wh"), ("mm_nh_interp", "mm_nh")]:
            path = MODEL_PATHS[path_key]
            if os.path.exists(path):
                interp = TFLiteInterpreter(model_path=path)
                interp.allocate_tensors()
                _registry[reg_key] = interp
                logger.info(f"  [OK] {path_key} → {path}")
            else:
                logger.warning(f"  [MISSING] {path_key} → {path}")

    except ImportError:
        logger.warning("TensorFlow not installed — multimodal endpoint unavailable.")

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bari Anemia Screening API",
    description="Multimodal anemia severity classification from conjunctiva images and patient data.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "code": 500},
    )


# ── HELPER ────────────────────────────────────────────────────────────────────
def _make_response(
    pred_idx: int,
    confidence: float,
    probs: np.ndarray,
    age: float | None = None,
    gender: int | None = None,
) -> PredictionResponse:
    from app.services.inference import CLASS_NAMES
    guide = get_full_guidance(pred_idx, age_months=age, gender=gender)
    return PredictionResponse(
        prediction=CLASS_NAMES[pred_idx],
        confidence=round(confidence, 4),
        class_probabilities=build_probabilities_dict(probs),
        nutrition=guide["advice"],
        recommended_foods=guide["foods"],
        referral_action=guide["referral"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns API status and which models are loaded."""
    return HealthResponse(
        status="API running",
        version="2.0.0",
        models_loaded={
            "multimodal_with_hb":  _registry["mm_wh_interp"] is not None,
            "multimodal_no_hb":    _registry["mm_nh_interp"] is not None,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — Multimodal prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/multimodal",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anemia from conjunctiva image + patient data",
)
async def predict_multimodal_endpoint(
    file:     UploadFile   = File(...,  description="Conjunctiva image (JPEG / PNG)"),
    age:      float        = Form(...,  ge=0, le=1200, description="Age in months"),
    gender:   int          = Form(...,  ge=0, le=1,    description="0=Male | 1=Female"),
    hb_level: float | None = Form(None, ge=0, le=25,   description="Hemoglobin g/dL (optional)"),
) -> PredictionResponse:
    """
    **Multimodal fusion prediction**

    Combines the conjunctiva image with patient tabular data.

    - `hb_level` provided → uses model trained with [image + age + gender + HB]
    - `hb_level` absent   → uses model trained with [image + age + gender]
    """
    if hb_level is not None and _registry["mm_wh_interp"] is None:
        raise HTTPException(503, "Multimodal-with-HB model is not loaded.")
    if hb_level is None and _registry["mm_nh_interp"] is None:
        raise HTTPException(503, "Multimodal-no-HB model is not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")

    try:
        img_arr = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image decode error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(422, f"Image preprocessing error: {exc}") from exc

    try:
        tab_arr, use_hb = preprocess_tabular(
            age=age,
            gender=gender,
            scaler_with=_registry["scaler_wh"],
            scaler_no=_registry["scaler_nh"],
            hb_level=hb_level,
        )
    except Exception as exc:
        raise HTTPException(400, f"Tabular preprocessing error: {exc}") from exc

    try:
        pred, conf, probs = predict_multimodal(
            img_array=img_arr,
            tab_array=tab_arr,
            mm_interpreter_wh=_registry["mm_wh_interp"],
            mm_interpreter_nonh=_registry["mm_nh_interp"],
            use_hb=use_hb,
        )
    except Exception as exc:
        logger.error(f"Multimodal inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Multimodal model inference failed.") from exc

    logger.info(f"/predict/multimodal → pred_idx={pred} ({conf:.3f}) | hb={hb_level}")
    return _make_response(pred, conf, probs, age=age, gender=gender)
