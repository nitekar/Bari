# API package

This package re-exports the FastAPI application exposed by `app.main`.

Purpose: gradually migrate `app/` → `api/` while retaining compatibility for tests
and scripts that import `app.main`.
