"""
Simple experiment logging and comparison utilities for the Bari notebook.

Usage inside Jupyter:

    from experiment_logging import ExperimentLogger
    logger = ExperimentLogger("Notebook/results/model_comparison.csv")

    logger.log(
        model_name="Visual_MobileNetV3",
        modality="image",
        params={"lr": 1e-4, "batch_size": 32},
        accuracy=acc,
        f1=f1,
    )

    logger.to_dataframe()
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ExperimentRecord:
    model_name: str
    modality: str
    accuracy: float
    f1: float
    params: Dict[str, Any] = field(default_factory=dict)


class ExperimentLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.records: List[ExperimentRecord] = []
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        model_name: str,
        modality: str,
        params: Dict[str, Any],
        accuracy: float,
        f1: float,
    ) -> None:
        rec = ExperimentRecord(
            model_name=model_name,
            modality=modality,
            accuracy=float(accuracy),
            f1=float(f1),
            params=params,
        )
        self.records.append(rec)

    def flush(self) -> None:
        if not self.records:
            return
        fieldnames = ["model_name", "modality", "accuracy", "f1", "params"]
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for rec in self.records:
                row = asdict(rec)
                row["params"] = repr(row["params"])
                writer.writerow(row)
        self.records.clear()

    def to_dataframe(self) -> pd.DataFrame:
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(
            [asdict(r) for r in self.records],
            columns=["model_name", "modality", "accuracy", "f1", "params"],
        )

