from __future__ import annotations

import pathlib

import pytest


# Auto-tag tests into categories based on filename.
# This avoids having to sprinkle @pytest.mark.* across many files.
_RULES: list[tuple[str, set[str]]] = [
    (
        "validation",
        {
            "test_dataset_integrity.py",
            "test_feature_schema.py",
        },
    ),
    (
        "integration",
        {
            "test_api.py",
            "test_api_mocked.py",
            "test_api_contract.py",
            "test_pipeline.py",
            "test_split_manifest.py",
        },
    ),
    (
        "unit",
        {
            "test_inference.py",
            "test_model.py",
            "test_fusion_helpers.py",
            "test_preprocessing_and_inference.py",
        },
    ),
]


def _item_path(item) -> str:
    path = getattr(item, "path", None)
    if path is None:
        path = getattr(item, "fspath")
    return str(path)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        basename = pathlib.Path(_item_path(item)).name

        applied = False
        for mark_name, filenames in _RULES:
            if basename in filenames:
                item.add_marker(getattr(pytest.mark, mark_name))
                applied = True
                break

        # Default: treat unknown/uncategorized as unit tests.
        if not applied:
            item.add_marker(pytest.mark.unit)
