"""API routes moved from `app/main.py`.

Endpoints here are thin and access the application registry via
`request.app.state.registry`. Heavy logic remains in `app.services` so tests
and behaviour remain unchanged during migration.
"""
from __future__ import annotations

from typing import Any
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.response import HealthResponse
from app.services.inference import (
    build_probabilities_dict,
    build_visual_probabilities_dict,
    late_fusion_weighted_average,
    predict_rf,
    predict_visual,
    predict_fusion,
    adapt_tab_array_for_interpreter,
    compare_fusion_vs_individuals,
)
from app.services.preprocessing import (
    build_nh_scaled,
    build_wh_scaled,
    preprocess_image_bytes,
)
from app.services.nutrition import get_full_guidance, get_binary_guidance
from app.utils.image_utils import validate_image_content_type
from utils.nutrition import build_structured_recommendations

router = APIRouter()
logger = logging.getLogger("anemia-api")


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API health check — returns status and loaded model inventory",
)
async def health_check(request: Request) -> HealthResponse:
    reg = getattr(request.app.state, "registry", {})
    models_loaded = {
        "visual":               reg.get("visual_interp") is not None,
        "severity_classifier":  reg.get("severity_model") is not None,
        "severity_scaler":      reg.get("severity_scaler") is not None,
        "multimodal_with_hb":   reg.get("multimodal_with_hb_interp") is not None,
        "multimodal_no_hb":     reg.get("multimodal_no_hb_interp") is not None,
    }
    all_critical = models_loaded["visual"] and models_loaded["severity_classifier"]
    return HealthResponse(
        status="ok" if all_critical else "degraded",
        version="3.0.0",
        models_loaded=models_loaded,
    )


@router.post(
    "/predict/multimodal",
    response_model=None,
    tags=["Prediction"],
    summary="Sequential multimodal severity prediction from image + patient data",
)
async def predict_multimodal_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Conjunctiva image (JPEG / PNG)"),
    age: float = Form(..., ge=0, le=1200, description="Age in months"),
    gender: int = Form(..., ge=0, le=1, description="0=Male | 1=Female"),
    hb_level: float | None = Form(None, ge=0, le=25, description="Optional measured Hb in g/dL."),
) -> Any:
    reg = request.app.state.registry

    if reg.get("visual_interp") is None:
        raise HTTPException(503, "Visual model not loaded.")
    if reg.get("severity_model") is None:
        raise HTTPException(503, "Severity classifier not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")
    if len(raw) > request.app.extra.get("max_upload_bytes", 10 * 1024 * 1024):
        raise HTTPException(413, "File too large.")

    try:
        img_arr, pil = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image error: {exc}") from exc

    hb_mean = reg.get("hb_mean")
    hb_std = reg.get("hb_std")
    try:
        _, _, _, hb_gdl = predict_visual(img_arr, reg.get("visual_interp"), hb_mean, hb_std)
    except Exception as exc:
        logger.error(f"Visual inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Visual model inference failed.") from exc

    hb_for_severity = float(hb_level) if hb_level is not None else float(hb_gdl)

    try:
        _, lab_feats = build_nh_scaled(pil, age, gender, reg.get("feature_probe_scaler"))
        tab_wh_scaled = build_wh_scaled(lab_feats, age, gender, hb_for_severity, reg.get("severity_scaler"))
    except Exception as exc:
        raise HTTPException(500, f"Feature extraction failed: {exc}") from exc

    try:
        pred_idx, conf, probs = predict_rf(tab_wh_scaled, reg.get("severity_model"))
    except Exception as exc:
        logger.error(f"Severity classifier inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Severity classifier inference failed.") from exc

    fusion_used = False
    fusion_strategy = "individual"
    try:
        # ── Gather individual model outputs for comparison ────────────────
        _, v_conf, v_probs, _ = predict_visual(img_arr, reg.get("visual_interp"), hb_mean, hb_std)
        _, rf_conf, rf_probs = predict_rf(tab_wh_scaled, reg.get("severity_model"))
        logger.debug("Visual top-conf=%.4f  RF top-conf=%.4f", v_conf, rf_conf)

        # ── Tier 1: TFLite end-to-end fusion model ────────────────────────
        mm_key = "multimodal_with_hb_interp" if (hb_level is not None) else "multimodal_no_hb_interp"
        mm_interp = reg.get(mm_key)
        if mm_interp is not None:
            tab_for_mm = adapt_tab_array_for_interpreter(mm_interp, tab_wh_scaled)
            f_pred_idx, f_conf, f_probs, f_hb = predict_fusion(img_arr, tab_for_mm, mm_interp, hb_mean, hb_std)
            logger.debug("TFLite fusion top-conf=%.4f  pred=%d", f_conf, f_pred_idx)
            comp = compare_fusion_vs_individuals(f_probs, v_probs, rf_probs)
            logger.info(
                "Fusion comparison — verdict=%s  fusion=%.4f  visual=%.4f  tab=%.4f",
                comp["verdict"], comp["fusion_top"], comp["visual_top"], comp["tab_top"],
            )
            if comp.get("verdict") == "fusion_benefit":
                pred_idx, conf, probs = f_pred_idx, f_conf, f_probs
                hb_gdl = f_hb or hb_gdl
                fusion_used = True
                fusion_strategy = "tflite_fusion"

        # ── Tier 2: late weighted average (visual 70 % / tabular 30 %) ───
        # Runs when TFLite fusion is unavailable or shows no benefit.
        # Maps 2-class visual probs to 4-class space before averaging.
        if not fusion_used:
            # visual_probs is binary (2 classes); pad to 4-class to match tab
            import numpy as _np
            v4 = _np.zeros(4, dtype="float32")
            # index 0 = Non-Anemic (binary 0), index 1+ share the anemic mass
            v4[0] = float(v_probs[0])
            v4[1:] = float(v_probs[1]) / 3.0  # spread anemic mass equally
            wa_pred, wa_conf, wa_probs = late_fusion_weighted_average(v4, rf_probs, w_visual=0.7, w_tab=0.3)
            logger.debug("Weighted-avg fusion top-conf=%.4f  pred=%d", wa_conf, wa_pred)
            wa_comp = compare_fusion_vs_individuals(wa_probs, v4, rf_probs)
            if wa_comp.get("verdict") == "fusion_benefit":
                pred_idx, conf, probs = wa_pred, wa_conf, wa_probs
                fusion_used = True
                fusion_strategy = "weighted_average"
                logger.info("Using weighted-average fusion (visual=0.7 / tabular=0.3)")
            else:
                # Tier 3: best individual model — pick whichever is more confident
                if rf_conf >= v_conf:
                    fusion_strategy = "individual_rf"
                    logger.info("Fusion skipped — using RF prediction (conf=%.4f)", rf_conf)
                else:
                    fusion_strategy = "individual_visual_mapped"
                    logger.info("Fusion skipped — using visual-mapped prediction (conf=%.4f)", v_conf)

    except Exception as exc:
        logger.warning("Fusion block failed (%s) — using baseline RF prediction", exc)

    from app.services.inference import CLASS_NAMES  # local import to avoid cycles

    severity = CLASS_NAMES[pred_idx]
    guide = get_full_guidance(pred_idx, age_months=age, gender=gender)
    rec = build_structured_recommendations(severity, conf, age_months=age)

    return {
        "prediction": severity,
        "confidence": round(conf, 4),
        "confidence_score": round(conf, 4),
        "risk_level": rec["risk_level"],
        "class_probabilities": build_probabilities_dict(probs),
        "hb_estimate_gdl": round(hb_gdl, 2),
        "nutrition": guide["advice"],
        "recommended_foods": guide["foods"],
        "referral_action": guide["referral"],
        "recommendations": rec["recommendations"],
        "fusion_strategy": fusion_strategy,
    }


@router.post(
    "/predict/image",
    response_model=None,
    tags=["Prediction"],
    summary="Quick binary screen (Anemic / Non-Anemic) from image only",
)
async def predict_image_endpoint(request: Request, file: UploadFile = File(...)) -> Any:
    reg = request.app.state.registry

    if reg.get("visual_interp") is None:
        raise HTTPException(503, "Visual model not loaded.")

    try:
        validate_image_content_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(422, "Uploaded file is empty.")
    if len(raw) > request.app.extra.get("max_upload_bytes", 10 * 1024 * 1024):
        raise HTTPException(413, "File too large.")

    try:
        img_arr, _ = preprocess_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(422, f"Image error: {exc}") from exc

    try:
        pred_idx, conf, probs, hb_gdl = predict_visual(img_arr, reg.get("visual_interp"), reg.get("hb_mean"), reg.get("hb_std"))
    except Exception as exc:
        logger.error(f"Visual inference failed: {exc}", exc_info=True)
        raise HTTPException(500, "Visual model inference failed.") from exc

    from app.services.inference import VISUAL_CLASS_NAMES
    label = VISUAL_CLASS_NAMES[pred_idx]
    guide = get_binary_guidance(pred_idx)
    default_age_months = 24.0
    rec = build_structured_recommendations(label, conf, age_months=default_age_months)

    return {
        "prediction": label,
        "confidence": round(conf, 4),
        "confidence_score": round(conf, 4),
        "risk_level": rec["risk_level"],
        "class_probabilities": build_visual_probabilities_dict(probs),
        "hb_estimate_gdl": round(hb_gdl, 2),
        "nutrition": guide["advice"],
        "recommended_foods": guide["foods"],
        "referral_action": guide["referral"],
        "recommendations": rec["recommendations"],
    }
