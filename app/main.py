from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Rate Limiting ─────────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas.response import HealthResponse, PredictionResponse
from app.services.inference import (
    build_probabilities_dict,
    build_visual_probabilities_dict,
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
from utils.nutrition import build_structured_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("anemia-api")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "")
ALLOW_INSECURE_DEFAULT_API_KEY = (
    os.environ.get("ALLOW_INSECURE_DEFAULT_API_KEY", "false").strip().lower() == "true"
)
if not API_KEY and ALLOW_INSECURE_DEFAULT_API_KEY:
    API_KEY = "dev-insecure-api-key"
    logger.warning("Using insecure development API key fallback. Do not use in production.")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Allowed CORS origins (restrict in production)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "https://web-production-c7c1.up.railway.app,http://localhost:8081,http://localhost:19006"
    ).split(",")
]

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

# ── Model paths ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

MODEL_PATHS: dict[str, str] = {
    "severity_bundle": os.environ.get(
        "TAB_WH_PATH",
        os.path.join(_ROOT, "Notebook", "models", "tabular_with_hb.pkl"),
    ),
    "feature_probe_scaler": os.environ.get(
        "SCALER_NH_PATH",
        os.path.join(_ROOT, "Notebook", "models", "scaler_no_hb.pkl"),
    ),
    "visual": os.environ.get(
        "VISUAL_PATH",
        os.path.join(_ROOT, "models", "saved_models", "visual_model.tflite"),
    ),
}

# ── Registry ──────────────────────────────────────────────────────────────────
_reg: dict[str, Any] = {
    "severity_model":      None,   # sklearn severity classifier
    "severity_scaler":     None,   # StandardScaler for FEAT_WITH_HB
    "feature_probe_scaler": None,  # StandardScaler for FEAT_NO_HB
    "hb_mean":             10.0,   # fallback — overwritten from pkl
    "hb_std":               2.0,   # fallback
    "visual_interp":       None,   # TFLite visual model
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models …")

    # Severity classifier bundle + scaler + Hb stats
    path = MODEL_PATHS["severity_bundle"]
    _EXPECTED_FEATURES = 17  # FEAT_WITH_HB length
    if os.path.exists(path):
        bundle = joblib.load(path)
        feats  = bundle.get("feature_names", [])
        if len(feats) == _EXPECTED_FEATURES:
            _reg["severity_model"] = bundle.get("model")
            _reg["severity_scaler"] = bundle.get("scaler")
            _reg["hb_mean"] = bundle.get("hb_mean", _reg["hb_mean"])
            _reg["hb_std"] = bundle.get("hb_std", _reg["hb_std"])
            logger.info(
                "  [OK] severity classifier + scaler  (HB mean=%.2f std=%.2f)",
                _reg["hb_mean"],
                _reg["hb_std"],
            )
        else:
            logger.warning(
                f"  [SKIP] severity bundle has {len(feats)} features (expected {_EXPECTED_FEATURES}) — "
                "re-run notebook to regenerate. RF severity endpoint disabled."
            )
    else:
        logger.warning(f"  [MISSING] severity bundle → {path}")

    # The no-Hb scaler is retained for feature extraction parity and future model checks.
    path = MODEL_PATHS["feature_probe_scaler"]
    if os.path.exists(path):
        bundle = joblib.load(path)
        _reg["feature_probe_scaler"] = bundle.get("scaler")
        logger.info("  [OK] feature probe scaler")
    else:
        logger.warning(f"  [MISSING] feature probe scaler → {path}")

    # TFLite models
    try:
        import tensorflow as tf
        Interp = tf.lite.Interpreter
        for key, path_key in [("visual_interp", "visual")]:
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
        "Anemia screening from conjunctiva images.\n\n"
        "**Production pipeline:** image → visual model → estimated Hb + patient data "
        "→ tabular severity classifier.\n\n"
        "The `/predict/multimodal` endpoint is the canonical shipped multimodal path."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Attach rate limiter to app ────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — restricted origins ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ── API Key middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify X-API-Key header on all non-health/non-docs endpoints."""
    path = request.url.path

    # Allow health checks, docs, and OPTIONS (CORS preflight) through
    public_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json"}
    if path in public_paths or request.method == "OPTIONS":
        return await call_next(request)

    # Fail closed when API key is not configured.
    if not API_KEY:
        logger.error("Rejected request to %s — API key is not configured", path)
        return JSONResponse(
            status_code=503,
            content={"detail": "API key authentication is not configured."},
        )

    # Check API key
    provided_key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided_key, API_KEY):
        logger.warning(f"Rejected request to {path} — invalid API key")
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key."},
        )

    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "code": 500})


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request) -> HealthResponse:
    return HealthResponse(
        status="API running",
        version="3.0.0",
        models_loaded={
            "severity_classifier": _reg["severity_model"] is not None,
            "visual_tflite": _reg["visual_interp"] is not None,
        },
    )


# ── Multimodal prediction ─────────────────────────────────────────────────────
@app.post(
    "/predict/multimodal",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Sequential multimodal severity prediction from image + patient data",
)
@limiter.limit("30/minute")
async def predict_multimodal_endpoint(
    request: Request,
    file:   UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
    age:    float      = Form(..., ge=0, le=1200, description="Age in months"),
    gender: int        = Form(..., ge=0, le=1,    description="0=Male | 1=Female"),
    hb_level: float | None = Form(
        None,
        ge=0,
        le=25,
        description="Optional measured Hb in g/dL. If provided, it is used by the severity classifier.",
    ),
) -> PredictionResponse:
    """
    **Sequential multimodal severity prediction**

        The shipped production path is:
      image → visual model → estimated Hb
      estimated Hb + age + gender + LAB features → RF severity classifier

        No blood test is required at inference time. If `hb_level` is provided,
        the classifier uses measured Hb for severity while still returning image-estimated Hb.
    """
    if _reg["visual_interp"] is None:
        raise HTTPException(503, "Visual model not loaded.")
    if _reg["severity_model"] is None:
        raise HTTPException(503, "Severity classifier not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

    try:
        img_arr, pil = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image error: {exc}") from exc

    hb_mean = _reg["hb_mean"]
    hb_std  = _reg["hb_std"]
    try:
        _, _, _, hb_gdl = predict_visual(img_arr, _reg["visual_interp"], hb_mean, hb_std)
    except Exception as exc:
        logger.error(f"Visual inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Visual model inference failed.") from exc

    hb_for_severity = float(hb_level) if hb_level is not None else float(hb_gdl)

    try:
        _, lab_feats = build_nh_scaled(pil, age, gender, _reg["feature_probe_scaler"])
        tab_wh_scaled = build_wh_scaled(
            lab_feats,
            age,
            gender,
            hb_for_severity,
            _reg["severity_scaler"],
        )
    except Exception as exc:
        raise HTTPException(500, f"Feature extraction failed: {exc}") from exc

    try:
        pred_idx, conf, probs = predict_rf(tab_wh_scaled, _reg["severity_model"])
    except Exception as exc:
        logger.error(f"Severity classifier inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Severity classifier inference failed.") from exc

    severity = CLASS_NAMES[pred_idx]
    hb_source = "measured" if hb_level is not None else "estimated"
    logger.info(
        "/predict/multimodal [sequential] → %s (%.3f) Hb_used=%.2f (%s) Hb_est=%.2f age=%s gender=%s",
        severity,
        conf,
        hb_for_severity,
        hb_source,
        hb_gdl,
        age,
        gender,
    )
    guide = get_full_guidance(pred_idx, age_months=age, gender=gender)
    rec = build_structured_recommendations(severity, conf, age_months=age)

    return PredictionResponse(
        prediction=severity,
        confidence=round(conf, 4),
        confidence_score=round(conf, 4),
        risk_level=rec["risk_level"],
        class_probabilities=build_probabilities_dict(probs),
        hb_estimate_gdl=round(hb_gdl, 2),
        nutrition=guide["advice"],
        recommended_foods=guide["foods"],
        referral_action=guide["referral"],
        recommendations=rec["recommendations"],
    )


# ── Quick binary screen ───────────────────────────────────────────────────────
@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Quick binary screen (Anemic / Non-Anemic) from image only",
)
@limiter.limit("30/minute")
async def predict_image_endpoint(
    request: Request,
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
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

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
    # Age is unknown for the quick screen; use a safe default in the 6–60 month band
    default_age_months = 24.0
    rec = build_structured_recommendations(label, conf, age_months=default_age_months)
    logger.info(f"/predict/image → {label} ({conf:.3f}) Hb≈{hb_gdl:.2f}")

    return PredictionResponse(
        prediction=label,
        confidence=round(conf, 4),
        confidence_score=round(conf, 4),
        risk_level=rec["risk_level"],
        class_probabilities=build_visual_probabilities_dict(probs),
        hb_estimate_gdl=round(hb_gdl, 2),
        nutrition=guide["advice"],
        recommended_foods=guide["foods"],
        referral_action=guide["referral"],
        recommendations=rec["recommendations"],
    )


# Import here to avoid circular reference in response builder above
from app.services.inference import CLASS_NAMES  # noqa: E402
