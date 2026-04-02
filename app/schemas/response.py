"""
schemas/response.py
Pydantic response models.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Recommendations(BaseModel):
    diet_plan: str = Field(..., description="Plain-language diet plan")
    foods_to_include: List[str] = Field(..., description="Iron-rich and supporting foods")
    foods_to_avoid: List[str] = Field(..., description="Foods that reduce iron absorption")
    urgency_level: str = Field(..., description="routine | elevated | urgent")


class PredictionResponse(BaseModel):
    prediction:          str              = Field(..., description="Predicted class (severity or binary)")
    confidence:          float            = Field(..., ge=0.0, le=1.0, description="Model confidence for prediction")
    class_probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    hb_estimate_gdl:     Optional[float]  = Field(None, description="Estimated Hb in g/dL")

    # Legacy nutrition fields (kept for backward compatibility with existing clients)
    nutrition:           str              = Field(..., description="Short nutritional advice")
    recommended_foods:   List[str]        = Field(..., description="List of suggested foods")
    referral_action:     str              = Field(..., description="Suggested clinical action / referral")

    # New decision-support fields
    risk_level:          str              = Field(..., description="low | moderate | high")
    confidence_score:    float            = Field(..., ge=0.0, le=1.0, description="Alias of confidence for frontend use")
    recommendations:     Recommendations  = Field(..., description="Structured nutritional recommendations")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "prediction":          "Moderate",
                "confidence":          0.82,
                "class_probabilities": {"Non-Anemic": 0.05, "Mild": 0.08, "Moderate": 0.82, "Severe": 0.05},
                "hb_estimate_gdl":     8.4,
                "nutrition":           "Iron, folate, and B12 required.",
                "recommended_foods":   ["Liver", "Spinach", "Lentils"],
                "referral_action":     "Medical consultation within 2 weeks.",
                "risk_level":          "moderate",
                "confidence_score":    0.82,
                "recommendations": {
                    "diet_plan": "Increase iron intake in every main meal and add vitamin C sources.",
                    "foods_to_include": ["Liver", "Red meat", "Beans and lentils", "Spinach", "Citrus fruits", "Tomatoes"],
                    "foods_to_avoid": [
                        "Tea during or 1 hour around iron-rich meals",
                        "Coffee during or 1 hour around iron-rich meals",
                        "Calcium-heavy foods (milk, yoghurt, cheese) at the same time as iron supplements",
                    ],
                    "urgency_level": "elevated",
                },
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
