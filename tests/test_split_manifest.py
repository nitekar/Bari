from __future__ import annotations

import inspect
import unittest

from app.main import app
from app.schemas.response import PredictionResponse


class ApiContractTests(unittest.TestCase):
    @staticmethod
    def _paths() -> set[str]:
        return {route.path for route in app.routes}

    def test_current_prediction_endpoints_exist(self) -> None:
        paths = self._paths()
        self.assertIn("/predict/image", paths)
        self.assertIn("/predict/multimodal", paths)
        self.assertIn("/health", paths)

    def test_legacy_endpoints_are_not_exposed(self) -> None:
        paths = self._paths()
        self.assertNotIn("/predict/tabular", paths)
        self.assertNotIn("/explain/tabular", paths)

    def test_multimodal_accepts_optional_hb_level(self) -> None:
        route = next(r for r in app.routes if r.path == "/predict/multimodal")
        sig = inspect.signature(route.endpoint)
        self.assertIn("hb_level", sig.parameters)
        # FastAPI wraps form defaults in Form(...), so default is not bare None.
        # The important contract is that a default exists (parameter is optional).
        self.assertIsNot(sig.parameters["hb_level"].default, inspect.Parameter.empty)

    def test_prediction_response_includes_hb_estimate(self) -> None:
        self.assertIn("hb_estimate_gdl", PredictionResponse.model_fields)


if __name__ == "__main__":
    unittest.main()
