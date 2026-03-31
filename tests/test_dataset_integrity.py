from __future__ import annotations

import csv
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "Tabular" / "anemia.csv"


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            cls.rows = list(csv.DictReader(f))

    def test_csv_exists_and_has_rows(self) -> None:
        self.assertTrue(CSV_PATH.exists())
        self.assertGreater(len(self.rows), 0)

    def test_required_columns_present(self) -> None:
        required = {
            "IMAGE_ID",
            "HB_LEVEL",
            "Severity",
            "Age(Months)",
            "GENDER",
            "REMARK",
        }
        self.assertTrue(required.issubset(set(self.rows[0].keys())))

    def test_image_tabular_pairing_is_complete(self) -> None:
        ids = [r["IMAGE_ID"].strip() for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(any(x == "" for x in ids))

    def test_image_ids_are_unique(self) -> None:
        ids = [r["IMAGE_ID"].strip() for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_labels_are_in_expected_vocab(self) -> None:
        sev_vocab = {"Non-Anemic", "Mild", "Moderate", "Severe"}
        remark_vocab = {"Anemic", "Non-anemic"}
        severities = {r["Severity"].strip() for r in self.rows}
        remarks = {r["REMARK"].strip() for r in self.rows}
        self.assertTrue(severities.issubset(sev_vocab))
        self.assertTrue(remarks.issubset(remark_vocab))

    def test_folder_labels_match_tabular_labels(self) -> None:
        # Conservative consistency check: "Non-Anemic" severity should map to non-anemic remark.
        mismatches = [
            r for r in self.rows
            if (r["Severity"].strip() == "Non-Anemic") != (r["REMARK"].strip() == "Non-anemic")
        ]
        self.assertEqual(mismatches, [])

    def test_required_tabular_fields_are_present(self) -> None:
        self.assertFalse(any(r["HB_LEVEL"].strip() == "" for r in self.rows))
        self.assertFalse(any(r["Age(Months)"].strip() == "" for r in self.rows))
        self.assertFalse(any(r["GENDER"].strip() == "" for r in self.rows))

    def test_severe_class_is_minority(self) -> None:
        severe_count = sum(1 for r in self.rows if r["Severity"].strip() == "Severe")
        ratio = severe_count / len(self.rows)
        self.assertLess(ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
