"""Pydantic schemas for the API package.

This module currently re-exports the application schemas from `app.schemas` so
the migration can be done incrementally without breaking imports.
"""
from app.schemas.request import *  # noqa: F401,F403
from app.schemas.response import *  # noqa: F401,F403
