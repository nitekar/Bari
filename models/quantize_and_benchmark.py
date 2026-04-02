"""Quantize and benchmark TFLite and tabular models
--------------------------------------------------
This script:
- lists tflite models in `models/saved_models`
- runs inference benchmarks (random input) and records size/latency
- for tabular sklearn models (`.pkl`) it runs predict on random input
- saves a CSV report to `models/saved_models/quant_benchmark.csv`

It will exit gracefully if TensorFlow is not installed, printing instructions.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "saved_models"
REPORT = SAVED / "quant_benchmark.csv"


def benchmark_tflite(path: Path, n_runs: int = 100) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow is required to benchmark TFLite models: pip install tensorflow") from e

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()

    in_det = interp.get_input_details()
    out_det = interp.get_output_details()

    in_shape = tuple(in_det[0]["shape"])
    dtype = in_det[0]["dtype"]

    # random input matching shape
    dummy = np.random.rand(*in_shape).astype(dtype)

    # warmup
    for _ in range(5):
        interp.set_tensor(in_det[0]["index"], dummy)
        interp.invoke()

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_det[0]["index"], dummy)
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    out = interp.get_tensor(out_det[0]["index"]) if out_det else None

    return {
        "model": path.name,
        "type": "tflite",
        "size_mb": round(os.path.getsize(path) / 1024**2, 3),
        "avg_ms": round(float(np.mean(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "input_shape": str(in_shape),
        "output_shape": str(out.shape) if out is not None else "",
    }


def benchmark_tabular(path: Path, n_runs: int = 100) -> dict[str, Any]:
    # lightweight benchmark for sklearn-like models
    import joblib

    model = joblib.load(path)

    # try to infer required input size from a sample if available
    # fallback to 10 features
    try:
        if hasattr(model, "n_features_in_"):
            n_feat = int(model.n_features_in_)
        else:
            n_feat = 10
    except Exception:
        n_feat = 10

    dummy = np.random.rand(1, n_feat).astype("float32")

    # warmup
    for _ in range(5):
        _ = model.predict(dummy)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "model": path.name,
        "type": "tabular",
        "size_mb": round(os.path.getsize(path) / 1024**2, 3),
        "avg_ms": round(float(np.mean(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "input_shape": f"(1,{n_feat})",
        "output_shape": "(1,)",
    }


def main():
    if not SAVED.exists():
        print(f"Saved models folder not found: {SAVED}")
        return

    rows = []

    # tflite benchmarks
    tflites = sorted(SAVED.glob("*.tflite"))
    if tflites:
        print(f"Found {len(tflites)} .tflite models; benchmarking...")
        for p in tflites:
            try:
                print(f"Benchmarking {p.name} ...")
                rows.append(benchmark_tflite(p, n_runs=50))
            except Exception as e:
                print(f"  SKIP {p.name}: {e}")
    else:
        print("No .tflite models found in saved_models/")

    # tabular models
    pickles = sorted(SAVED.glob("*.pkl"))
    if pickles:
        print(f"Found {len(pickles)} tabular models; benchmarking...")
        for p in pickles:
            try:
                print(f"Benchmarking {p.name} ...")
                rows.append(benchmark_tabular(p, n_runs=200))
            except Exception as e:
                print(f"  SKIP {p.name}: {e}")
    else:
        print("No .pkl tabular models found in saved_models/")

    if rows:
        print(f"Writing CSV report to: {REPORT}")
        with open(REPORT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "type", "size_mb", "avg_ms", "p95_ms", "input_shape", "output_shape"])  # type: ignore[arg-type]
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        for r in rows:
            print(r)
    else:
        print("No benchmark data collected.")


if __name__ == "__main__":
    main()
