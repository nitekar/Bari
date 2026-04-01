"""api package — scaffold for routes, services and schemas.

This package provides a place to move API route definitions (`routes.py`),
business logic (`services.py`) and pydantic models (`schemas.py`).

Currently this is a non-breaking scaffold; migrate endpoints from `app/main.py`
into `api/routes.py` and import the router into the FastAPI app when ready.
"""

from . import routes  # noqa: F401
"""api package — structural wrapper re-exporting the existing FastAPI app.

This non-invasive wrapper lets callers import the API from `api` while the
original `app` package remains functional for backward compatibility.
"""
from __future__ import annotations

from app.main import app as app

__all__ = ["app"]
