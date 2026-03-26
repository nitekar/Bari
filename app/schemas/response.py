"""
schemas/response.py
Pydantic response models.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction:          str              = Field(..., description="Predicted class (severity or binary)")
    confidence:          float            = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    hb_estimate_gdl:     Optional[float]  = Field(None, description="Estimated Hb in g/dL")
    nutrition:           str              = Field(..., description="Nutritional advice")
    recommended_foods:   List[str]        = Field(...)
    referral_action:     str              = Field(...)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "prediction":          "Moderate",
                "confidence":          0.82,
                "class_probabilities": {"Normal": 0.05, "Mild": 0.08, "Moderate": 0.82, "Severe": 0.05},
                "hb_estimate_gdl":     8.4,
                "nutrition":           "Iron, folate, and B12 required.",
                "recommended_foods":   ["Liver", "Spinach", "Lentils"],
                "referral_action":     "Medical consultation within 2 weeks.",
            }]
        }
    }


class HealthResponse(BaseModel):
    status:        str
    version:       str
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    detail: str
    code:   int
