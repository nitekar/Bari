"""
schemas/response.py
Pydantic response models — every endpoint returns a consistent shape.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Standard prediction response for all endpoints."""

    prediction: str = Field(..., description="Predicted severity class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence [0,1]")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Probability for each class"
    )
    nutrition: str = Field(..., description="Short nutritional advice")
    recommended_foods: List[str] = Field(..., description="List of recommended foods")
    referral_action: str = Field(..., description="Medical referral instruction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "Moderate",
                    "confidence": 0.82,
                    "class_probabilities": {
                        "Non-Anemic": 0.05,
                        "Mild": 0.08,
                        "Moderate": 0.82,
                        "Severe": 0.05,
                    },
                    "nutrition": "Iron, folate, and B12 required.",
                    "recommended_foods": ["Liver", "Spinach", "Lentils"],
                    "referral_action": "Medical consultation within 2 weeks.",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response for GET /."""

    status: str
    version: str
    models_loaded: Dict[str, bool]


class ExplainResponse(BaseModel):
    """Response for POST /explain/tabular."""

    prediction: str
    confidence: float
    top_features: Dict[str, float] = Field(
        ..., description="Feature name → mean |SHAP| contribution"
    )
    nutrition: str
    note: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    code: int
