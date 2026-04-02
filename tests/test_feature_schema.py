from __future__ import annotations

import unittest

from app.utils.tabular_utils import FEAT_NO_HB, FEAT_WITH_HB, build_nh_vector, build_wh_vector


class FeatureSchemaTests(unittest.TestCase):
    def test_feature_counts_are_stable(self) -> None:
        self.assertEqual(len(FEAT_NO_HB), 16)
        self.assertEqual(len(FEAT_WITH_HB), 17)
        self.assertEqual(FEAT_WITH_HB[-3:], ["HB_LEVEL", "Age(Months)", "Gender_F"])

    def test_nh_vector_shape_and_tail(self) -> None:
        row = build_nh_vector({}, age=24, gender=1)
        self.assertEqual(row.shape, (1, 16))
        self.assertEqual(float(row[0, -2]), 24.0)
        self.assertEqual(float(row[0, -1]), 1.0)

    def test_wh_vector_includes_hb_age_gender(self) -> None:
        row = build_wh_vector({}, age=36, gender=0, hb_estimated=10.5)
        self.assertEqual(row.shape, (1, 17))
        self.assertEqual(float(row[0, -3]), 10.5)
        self.assertEqual(float(row[0, -2]), 36.0)
        self.assertEqual(float(row[0, -1]), 0.0)


if __name__ == "__main__":
    unittest.main()
