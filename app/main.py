"""
main.py — Anemia Screening REST API
=====================================
Endpoints
---------
GET  /                     Health check
POST /predict/tabular      Random Forest (with or without HB_LEVEL)
POST /predict/image        TFLite visual model (conjunctiva image)
POST /predict/multimodal   TFLite fusion model (image + tabular)
POST /explain/tabular      SHAP feature explanation (optional)

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
from app.schemas.request import ExplainRequest, MultimodalRequest, TabularRequest
from app.schemas.response import (
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    PredictionResponse,
)
from app.services.inference import (
    build_probabilities_dict,
    explain_tabular,
    predict_image,
    predict_multimodal,
    predict_tabular,
)
from app.services.nutrition import get_full_guidance, get_short_advice
from app.services.preprocessing import preprocess_image_bytes, preprocess_tabular
from app.utils.image_utils import validate_image_content_type
from app.utils.tabular_utils import FEAT_NO_HB, FEAT_WITH_HB

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
    "tab_wh":  os.environ.get("TAB_WH_PATH",   os.path.join(_ROOT, "Notebook", "models", "tabular_with_hb.pkl")),
    "tab_nh":  os.environ.get("TAB_NH_PATH",   os.path.join(_ROOT, "Notebook", "models", "tabular_no_hb.pkl")),
    "vis":     os.environ.get("VIS_PATH",      os.path.join(_ROOT, "models", "saved_models", "visual_model.tflite")),
    "mm_wh":   os.environ.get("MM_WH_PATH",    os.path.join(_ROOT, "models", "saved_models", "multimodal_model.tflite")),
    "mm_nh":   os.environ.get("MM_NH_PATH",    os.path.join(_ROOT, "models", "saved_models", "multimodal_no_hb_model.tflite")),
}

# ── Model registry (populated at startup) ────────────────────────────────────
_registry: dict[str, Any] = {
    "tab_wh":      None,
    "tab_nh":      None,
    "scaler_wh":   None,
    "scaler_nh":   None,
    "vis_interp":  None,
    "mm_wh_interp": None,
    "mm_nh_interp": None,
}


# ── Lifespan: load all models once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models …")

    # ── sklearn models + scalers ──────────────────────────────────────────────
    for key, path_key in [("tab_wh", "tab_wh"), ("tab_nh", "tab_nh")]:
        path = MODEL_PATHS[path_key]
        if os.path.exists(path):
            bundle = joblib.load(path)
            if isinstance(bundle, dict):
                _registry[key]                  = bundle["model"]
                _registry[f"scaler_{key.split('_')[1]}"] = bundle.get("scaler")
            else:
                _registry[key] = bundle
            logger.info(f"  [OK] {path_key} → {path}")
        else:
            logger.warning(f"  [MISSING] {path_key} → {path}")

    # ── TFLite interpreters ───────────────────────────────────────────────────
    try:
        try:
            from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        except ImportError:
            import tensorflow as tf
            TFLiteInterpreter = tf.lite.Interpreter

        for reg_key, path_key in [
            ("vis_interp",   "vis"),
            ("mm_wh_interp", "mm_wh"),
            ("mm_nh_interp", "mm_nh"),
        ]:
            path = MODEL_PATHS[path_key]
            if os.path.exists(path):
                interp = TFLiteInterpreter(model_path=path)
                interp.allocate_tensors()
                _registry[reg_key] = interp
                logger.info(f"  [OK] {path_key} → {path}")
            else:
                logger.warning(f"  [MISSING] {path_key} → {path}")

    except ImportError:
        logger.warning("TensorFlow not installed — image/multimodal endpoints unavailable.")

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Anemia Screening API",
    description=(
        "Multimodal anemia severity classification from conjunctiva images "
        "and patient tabular data. Supports tabular-only, image-only, and "
        "fusion inference with nutritional guidance."
    ),
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: assemble PredictionResponse from raw inference output
# ─────────────────────────────────────────────────────────────────────────────
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
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Returns API status and which models are loaded."""
    return HealthResponse(
        status="API running",
        version="2.0.0",
        models_loaded={
            "tabular_with_hb":       _registry["tab_wh"]       is not None,
            "tabular_no_hb":         _registry["tab_nh"]        is not None,
            "visual_tflite":         _registry["vis_interp"]    is not None,
            "multimodal_with_hb":    _registry["mm_wh_interp"]  is not None,
            "multimodal_no_hb":      _registry["mm_nh_interp"]  is not None,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — Tabular prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/tabular",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anemia from tabular data (age, gender, optional HB level)",
)
async def predict_tabular_endpoint(body: TabularRequest) -> PredictionResponse:
    """
    **Tabular prediction**

    - If `hb_level` is provided → RandomForest trained with [HB, Age, Gender]
    - If `hb_level` is absent   → RandomForest trained with [Age, Gender]

    Gender encoding: `0 = Male`, `1 = Female`
    """
    # Validate models loaded
    if _registry["tab_wh"] is None and body.hb_level is not None:
        raise HTTPException(503, "tabular_with_hb model is not loaded.")
    if _registry["tab_nh"] is None and body.hb_level is None:
        raise HTTPException(503, "tabular_no_hb model is not loaded.")

    try:
        scaled, use_hb = preprocess_tabular(
            age=body.age,
            gender=body.gender,
            scaler_with=_registry["scaler_wh"],
            scaler_no=_registry["scaler_nh"],
            hb_level=body.hb_level,
        )
    except Exception as exc:
        raise HTTPException(400, f"Preprocessing error: {exc}") from exc

    try:
        pred, conf, probs = predict_tabular(
            scaled_features=scaled,
            model_with_hb=_registry["tab_wh"],
            model_no_hb=_registry["tab_nh"],
            use_hb=use_hb,
        )
    except Exception as exc:
        logger.error(f"Tabular inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Tabular model inference failed.") from exc

    logger.info(f"/predict/tabular → {pred} ({conf:.3f}) | hb={body.hb_level}")
    return _make_response(pred, conf, probs, age=body.age, gender=body.gender)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — Image prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anemia from a conjunctiva image (TFLite)",
)
async def predict_image_endpoint(
    file: UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
) -> PredictionResponse:
    """
    **Image prediction**

    Upload a conjunctiva eye image. The model returns severity class
    probabilities using a quantized ResNet50 + CBAM (TFLite).
    """
    if _registry["vis_interp"] is None:
        raise HTTPException(503, "Visual TFLite model is not loaded.")

    # Validate MIME type
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
        pred, conf, probs = predict_image(img_arr, _registry["vis_interp"])
    except Exception as exc:
        logger.error(f"Visual inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Visual model inference failed.") from exc

    logger.info(f"/predict/image → {pred} ({conf:.3f})")
    return _make_response(pred, conf, probs)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — Multimodal prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict/multimodal",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anemia from image + tabular data (TFLite fusion model)",
)
async def predict_multimodal_endpoint(
    file:     UploadFile    = File(...,  description="Conjunctiva image (JPEG / PNG)"),
    age:      float         = Form(...,  ge=0, le=1200, description="Age in months"),
    gender:   int           = Form(...,  ge=0, le=1,    description="0=Male | 1=Female"),
    hb_level: float | None  = Form(None, ge=0, le=25,   description="HB level g/dL (optional)"),
) -> PredictionResponse:
    """
    **Multimodal fusion prediction**

    Combines the conjunctiva image with patient tabular data.

    - `hb_level` present → `multimodal_model.tflite` (3 tabular features)
    - `hb_level` absent  → `multimodal_no_hb_model.tflite` (2 tabular features)
    """
    # Check models
    if hb_level is not None and _registry["mm_wh_interp"] is None:
        raise HTTPException(503, "Multimodal-with-HB TFLite model is not loaded.")
    if hb_level is None and _registry["mm_nh_interp"] is None:
        raise HTTPException(503, "Multimodal-no-HB TFLite model is not loaded.")

    # Validate image
    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")

    # Preprocess image
    try:
        img_arr = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image decode error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(422, f"Image preprocessing error: {exc}") from exc

    # Preprocess tabular
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

    # Inference
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

    logger.info(f"/predict/multimodal → {pred} ({conf:.3f}) | hb={hb_level}")
    return _make_response(pred, conf, probs, age=age, gender=gender)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — SHAP Explainability (optional)
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/explain/tabular",
    response_model=ExplainResponse,
    tags=["Explainability"],
    summary="SHAP feature explanation for tabular prediction",
)
async def explain_tabular_endpoint(body: ExplainRequest) -> ExplainResponse:
    """
    **SHAP explainability**

    Returns the mean absolute SHAP contribution of each feature
    for the predicted class. Requires `shap` to be installed.
    """
    if _registry["tab_wh"] is None and body.hb_level is not None:
        raise HTTPException(503, "tabular_with_hb model not loaded.")
    if _registry["tab_nh"] is None and body.hb_level is None:
        raise HTTPException(503, "tabular_no_hb model not loaded.")

    try:
        scaled, use_hb = preprocess_tabular(
            age=body.age,
            gender=body.gender,
            scaler_with=_registry["scaler_wh"],
            scaler_no=_registry["scaler_nh"],
            hb_level=body.hb_level,
        )
    except Exception as exc:
        raise HTTPException(400, f"Preprocessing error: {exc}") from exc

    feature_names = FEAT_WITH_HB if use_hb else FEAT_NO_HB

    try:
        shap_scores = explain_tabular(
            scaled_features=scaled,
            model_with_hb=_registry["tab_wh"],
            model_no_hb=_registry["tab_nh"],
            use_hb=use_hb,
            feature_names=feature_names,
        )
    except ImportError as exc:
        raise HTTPException(501, str(exc)) from exc
    except Exception as exc:
        logger.error(f"SHAP failed: {exc}", exc_info=True)
        raise HTTPException(500, f"SHAP computation failed: {exc}") from exc

    # Also run prediction for the response
    try:
        pred, conf, _ = predict_tabular(
            scaled_features=scaled,
            model_with_hb=_registry["tab_wh"],
            model_no_hb=_registry["tab_nh"],
            use_hb=use_hb,
        )
    except Exception as exc:
        raise HTTPException(500, "Model prediction failed.") from exc

    from app.services.inference import CLASS_NAMES
    return ExplainResponse(
        prediction=CLASS_NAMES[pred],
        confidence=round(conf, 4),
        top_features=dict(
            sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
        ),
        nutrition=get_short_advice(pred),
        note=(
            "SHAP values show mean |contribution| per feature across all classes. "
            "Higher = more influential."
        ),
    )
