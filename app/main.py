"""
main.py — Anemia Screening REST API
=====================================
Pipeline
--------
  Visual CNN (TFLite) → estimates Hb + binary class
  LAB features extracted from image (OpenCV)
  [LAB + estimated Hb + Age + Gender] → RF With HB → WHO severity   (mode=tabular)
  [image + LAB features] → Fusion TFLite → WHO severity + Hb         (mode=fusion)

Endpoints
---------
  GET  /health                  Health check
  POST /predict/multimodal      Full pipeline (image + age + gender → severity + Hb)
  POST /predict/image           Quick binary screen (image only)

Run
---
  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas.response import HealthResponse, PredictionResponse
from app.services.inference import (
    anemia_label,
    build_probabilities_dict,
    build_visual_probabilities_dict,
    predict_fusion,
    predict_rf,
    predict_visual,
)
from app.services.nutrition import get_full_guidance, get_binary_guidance
from app.services.preprocessing import (
    build_nh_scaled,
    build_wh_scaled,
    preprocess_image_bytes,
)
from app.utils.image_utils import validate_image_content_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("anemia-api")

# ── Model paths ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MODEL_PATHS: dict[str, str] = {
    "tab_wh":    os.environ.get("TAB_WH_PATH",    os.path.join(_ROOT, "Notebook", "models", "tabular_with_hb.pkl")),
    "scaler_nh": os.environ.get("SCALER_NH_PATH", os.path.join(_ROOT, "Notebook", "models", "scaler_no_hb.pkl")),
    "fusion":    os.environ.get("FUSION_PATH",    (
        os.path.join(_ROOT, "models", "saved_models", "fusion_model.tflite")
        if os.path.exists(os.path.join(_ROOT, "models", "saved_models", "fusion_model.tflite"))
        else os.path.join(_ROOT, "models", "saved_models", "multimodal_model.tflite")
    )),
    "visual":    os.environ.get("VISUAL_PATH",    os.path.join(_ROOT, "models", "saved_models", "visual_model.tflite")),
}

# ── Registry ──────────────────────────────────────────────────────────────────
_reg: dict[str, Any] = {
    "rf_model":       None,   # sklearn RF (with HB)
    "scaler_wh":      None,   # StandardScaler for FEAT_WITH_HB
    "scaler_nh":      None,   # StandardScaler for FEAT_NO_HB
    "hb_mean":        10.0,   # fallback — overwritten from pkl
    "hb_std":          2.0,   # fallback
    "fusion_interp":  None,   # TFLite fusion model
    "visual_interp":  None,   # TFLite visual model
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models …")

    # RF model + scaler_wh + HB stats
    path = MODEL_PATHS["tab_wh"]
    _EXPECTED_FEATURES = 17  # FEAT_WITH_HB length
    if os.path.exists(path):
        bundle = joblib.load(path)
        feats  = bundle.get("feature_names", [])
        if len(feats) == _EXPECTED_FEATURES:
            _reg["rf_model"]  = bundle.get("model")
            _reg["scaler_wh"] = bundle.get("scaler")
            _reg["hb_mean"]   = bundle.get("hb_mean", _reg["hb_mean"])
            _reg["hb_std"]    = bundle.get("hb_std",  _reg["hb_std"])
            logger.info(f"  [OK] RF model + scaler_wh  (HB mean={_reg['hb_mean']:.2f} std={_reg['hb_std']:.2f})")
        else:
            logger.warning(
                f"  [SKIP] tab_wh has {len(feats)} features (expected {_EXPECTED_FEATURES}) — "
                "re-run notebook to regenerate. RF severity endpoint disabled."
            )
    else:
        logger.warning(f"  [MISSING] tab_wh → {path}")

    # scaler_nh (for fusion model tabular branch)
    path = MODEL_PATHS["scaler_nh"]
    if os.path.exists(path):
        bundle = joblib.load(path)
        _reg["scaler_nh"] = bundle.get("scaler")
        logger.info("  [OK] scaler_nh")
    else:
        logger.warning(f"  [MISSING] scaler_nh → {path}")

    # TFLite models
    try:
        import tensorflow as tf
        Interp = tf.lite.Interpreter
        for key, path_key in [("fusion_interp", "fusion"), ("visual_interp", "visual")]:
            path = MODEL_PATHS[path_key]
            if os.path.exists(path):
                interp = Interp(model_path=path)
                interp.allocate_tensors()
                _reg[key] = interp
                logger.info(f"  [OK] {path_key} TFLite")
            else:
                logger.warning(f"  [MISSING] {path_key} → {path}")
    except ImportError:
        logger.warning("TensorFlow not installed — TFLite endpoints unavailable.")

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Bari Anemia Screening API",
    description=(
        "Multimodal anemia severity classification from conjunctiva images.\n\n"
        "**Pipeline:** Visual CNN → estimated Hb → RF severity classifier\n\n"
        "No blood test required — Hb is estimated non-invasively from the image."
    ),
    version="3.0.0",
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "code": 500})


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="API running",
        version="3.0.0",
        models_loaded={
            "rf_with_hb":   _reg["rf_model"]      is not None,
            "fusion_tflite": _reg["fusion_interp"] is not None,
            "visual_tflite": _reg["visual_interp"] is not None,
        },
    )


# ── Multimodal prediction ─────────────────────────────────────────────────────
@app.post(
    "/predict/multimodal",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anemia severity from conjunctiva image + patient data",
)
async def predict_multimodal_endpoint(
    file:   UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
    age:    float      = Form(..., ge=0, le=1200, description="Age in months"),
    gender: int        = Form(..., ge=0, le=1,    description="0=Male | 1=Female"),
    mode:   Literal["tabular", "fusion"] = Query(
        "tabular",
        description=(
            "tabular = Visual CNN → estimated Hb → RF severity (default)\n"
            "fusion  = Fusion TFLite (image + LAB features)"
        ),
    ),
) -> PredictionResponse:
    """
    **Non-invasive severity prediction**

    - `mode=tabular` (default): Visual model estimates Hb → RF classifies severity
    - `mode=fusion`: End-to-end fusion model (image + colour features → severity)

    No blood test required. Hb is estimated from the conjunctiva image.
    """
    if _reg["visual_interp"] is None:
        raise HTTPException(503, "Visual model not loaded.")
    # If RF model is unavailable (e.g. old format pkl), fall back to visual-only
    if mode == "tabular" and _reg["rf_model"] is None:
        logger.warning("RF model unavailable — falling back to visual-only mode.")
        mode = "visual_only"
    if mode == "fusion" and _reg["fusion_interp"] is None:
        raise HTTPException(503, "Fusion TFLite model not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")

    try:
        img_arr, pil = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image error: {exc}") from exc

    hb_mean = _reg["hb_mean"]
    hb_std  = _reg["hb_std"]
    gender_str = "F" if gender == 1 else "M"

    if mode == "tabular":
        # Step 1: Visual model → binary + estimated Hb
        try:
            _, _, _, hb_gdl = predict_visual(img_arr, _reg["visual_interp"], hb_mean, hb_std)
        except Exception as exc:
            logger.error(f"Visual inference failed: {exc}", exc_info=True)
            raise HTTPException(500, "Visual model inference failed.") from exc

        # Step 2: Extract LAB features, build FEAT_WITH_HB, scale
        try:
            _, lab_feats  = build_nh_scaled(pil, age, gender, _reg["scaler_nh"])
            tab_wh_scaled = build_wh_scaled(lab_feats, age, gender, hb_gdl, _reg["scaler_wh"])
        except Exception as exc:
            raise HTTPException(500, f"Feature extraction failed: {exc}") from exc

        # Step 3: RF → severity
        try:
            pred_idx, conf, probs = predict_rf(tab_wh_scaled, _reg["rf_model"])
        except Exception as exc:
            logger.error(f"RF inference failed: {exc}", exc_info=True)
            raise HTTPException(500, "RF model inference failed.") from exc

        severity = CLASS_NAMES[pred_idx]
        logger.info(f"/predict/multimodal [tabular] → {severity} ({conf:.3f}) Hb≈{hb_gdl:.2f}")
        guide = get_full_guidance(pred_idx, age_months=age, gender=gender)
        return PredictionResponse(
            prediction=severity,
            confidence=round(conf, 4),
            class_probabilities=build_probabilities_dict(probs),
            hb_estimate_gdl=round(hb_gdl, 2),
            nutrition=guide["advice"],
            recommended_foods=guide["foods"],
            referral_action=guide["referral"],
        )

    else:  # fusion
        # Extract LAB features + build scaled FEAT_NO_HB for fusion tabular branch
        try:
            tab_nh_scaled, _ = build_nh_scaled(pil, age, gender, _reg["scaler_nh"])
        except Exception as exc:
            raise HTTPException(500, f"Feature extraction failed: {exc}") from exc

        try:
            pred_idx, conf, probs, hb_gdl = predict_fusion(
                img_arr, tab_nh_scaled, _reg["fusion_interp"], hb_mean, hb_std
            )
        except Exception as exc:
            logger.error(f"Fusion inference failed: {exc}", exc_info=True)
            raise HTTPException(500, "Fusion model inference failed.") from exc

        severity = anemia_label(hb_gdl, age, gender_str)
        logger.info(f"/predict/multimodal [fusion] → {severity} ({conf:.3f}) Hb≈{hb_gdl:.2f}")
        guide = get_full_guidance(pred_idx, age_months=age, gender=gender)
        return PredictionResponse(
            prediction=severity,
            confidence=round(conf, 4),
            class_probabilities=build_probabilities_dict(probs),
            hb_estimate_gdl=round(hb_gdl, 2),
            nutrition=guide["advice"],
            recommended_foods=guide["foods"],
            referral_action=guide["referral"],
        )


# ── Quick binary screen ───────────────────────────────────────────────────────
@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Quick binary screen (Anemic / Non-Anemic) from image only",
)
async def predict_image_endpoint(
    file: UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
) -> PredictionResponse:
    """
    **Quick binary screen**

    Uses the visual CNN to classify as Non-Anemic or Anemic and estimate Hb.
    No clinical data required. For 4-class severity use `/predict/multimodal`.
    """
    if _reg["visual_interp"] is None:
        raise HTTPException(503, "Visual model not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")

    try:
        img_arr, _ = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image error: {exc}") from exc

    try:
        pred_idx, conf, probs, hb_gdl = predict_visual(
            img_arr, _reg["visual_interp"], _reg["hb_mean"], _reg["hb_std"]
        )
    except Exception as exc:
        logger.error(f"Visual inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Visual model inference failed.") from exc

    from app.services.inference import VISUAL_CLASS_NAMES
    label = VISUAL_CLASS_NAMES[pred_idx]
    guide = get_binary_guidance(pred_idx)
    logger.info(f"/predict/image → {label} ({conf:.3f}) Hb≈{hb_gdl:.2f}")

    return PredictionResponse(
        prediction=label,
        confidence=round(conf, 4),
        class_probabilities=build_visual_probabilities_dict(probs),
        hb_estimate_gdl=round(hb_gdl, 2),
        nutrition=guide["advice"],
        recommended_foods=guide["foods"],
        referral_action=guide["referral"],
    )


# Import here to avoid circular reference in response builder above
from app.services.inference import CLASS_NAMES  # noqa: E402
