"""
schemas/request.py
Pydantic request models for all API endpoints.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TabularRequest(BaseModel):
    """
    Legacy tabular request shape reused by older notebook/export flows.

    age      : patient age in months
    gender   : 0 = Male, 1 = Female
    hb_level : optional haemoglobin level in g/dL
    """

    age: float = Field(..., ge=0, le=1200, description="Age in months")
    gender: int = Field(..., ge=0, le=1,   description="0 = Male | 1 = Female")
    hb_level: Optional[float] = Field(
        None, ge=0.0, le=25.0,
        description="Haemoglobin level g/dL (optional)",
    )

    @field_validator("gender")
    @classmethod
    def gender_must_be_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("gender must be 0 (Male) or 1 (Female)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"age": 24, "gender": 1, "hb_level": 9.5},
                {"age": 36, "gender": 0},
            ]
        }
    }


class MultimodalRequest(BaseModel):
    """
    Tabular part of /predict/multimodal (image is sent as UploadFile).
    """

    age: float = Field(..., ge=0, le=1200, description="Age in months")
    gender: int = Field(..., ge=0, le=1,   description="0 = Male | 1 = Female")
    hb_level: Optional[float] = Field(
        None, ge=0.0, le=25.0,
        description="Haemoglobin level g/dL (optional)",
    )

    @field_validator("gender")
    @classmethod
    def gender_must_be_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("gender must be 0 (Male) or 1 (Female)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"age": 18, "gender": 1, "hb_level": 8.0},
                {"age": 48, "gender": 0},
            ]
        }
    }


class ExplainRequest(BaseModel):
    """Input for /explain/tabular."""

    age: float = Field(..., ge=0, le=1200)
    gender: int = Field(..., ge=0, le=1)
    hb_level: Optional[float] = Field(None, ge=0.0, le=25.0)
