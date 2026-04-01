from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Annotated

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
from app.services.inference import (
    predict_fusion,
    late_fusion_weighted_average,
    compare_fusion_vs_individuals,
    adapt_tab_array_for_interpreter,
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
    # Multimodal fusion models (optional)
    "multimodal_with_hb": os.environ.get(
        "MM_WH_PATH",
        os.path.join(_ROOT, "models", "saved_models", "multimodal_model.tflite"),
    ),
    "multimodal_no_hb": os.environ.get(
        "MM_NH_PATH",
        os.path.join(_ROOT, "models", "saved_models", "multimodal_no_hb_model.tflite"),
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
        "multimodal_with_hb_interp": None,
        "multimodal_no_hb_interp":   None,
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
        # Multimodal fusion models (optional)
        for key, path_key in [("multimodal_with_hb_interp", "multimodal_with_hb"), ("multimodal_no_hb_interp", "multimodal_no_hb")]:
            path = MODEL_PATHS[path_key]
            if os.path.exists(path):
                try:
                    interp = Interp(model_path=path)
                    interp.allocate_tensors()
                    _reg[key] = interp
                    logger.info(f"  [OK] {path_key} TFLite")
                except Exception as exc:
                    logger.warning(f"  [ERROR] loading {path_key}: {exc}")
            else:
                logger.info(f"  [MISSING] {path_key} → {path}")
    except ImportError:
        logger.warning("TensorFlow not installed — TFLite endpoints unavailable.")

    logger.info("Startup complete.")
    # Expose registry on app state for route modules to consume.
    app.state.registry = _reg
    # Expose max upload size for route handlers
    app.extra = getattr(app, "extra", {})
    app.extra["max_upload_bytes"] = MAX_UPLOAD_BYTES
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


from api import routes as api_routes

# Include migrated API routes implemented in `api/routes.py`
app.include_router(api_routes.router)
