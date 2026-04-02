"""API service helpers — thin wrappers around `app.services` implementations.

Move heavy logic from route handlers into functions here when refactoring.
"""

from typing import Any

from app.services import inference


def predict_multimodal(*args, **kwargs) -> Any:
    return inference.predict_fusion(*args, **kwargs)
