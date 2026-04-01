"""models/validate_models.py
Small utility to validate saved model artifacts (Keras, joblib, TFLite).
Generates a CSV report with loadability, input/output shapes and dtypes.

Run from the repository root:
    python models/validate_models.py

"""
from __future__ import annotations

import csv
import os
import sys
import traceback
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT_CSV = os.path.join(ROOT, "models", "artifact_validation_report.csv")


def try_load_joblib(path: str) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(path)
    return {"type": "joblib", "keys": list(bundle.keys())}


def try_load_keras(path: str) -> dict[str, Any]:
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    return {
        "type": "keras",
        "inputs": [str(t.shape) + ":" + str(t.dtype.name) for t in model.inputs],
        "outputs": [str(t.shape) + ":" + str(t.dtype.name) for t in model.outputs],
    }


def try_load_tflite(path: str) -> dict[str, Any]:
    # Lazy-import to allow running without TF if not needed
    try:
        import tensorflow as tf
    except Exception:
        # Try tflite_runtime fallback
        try:
            import tflite_runtime.interpreter as tflite
            Interp = tflite.Interpreter
        except Exception:
            raise
    else:
        Interp = tf.lite.Interpreter

    interp = Interp(model_path=path)
    interp.allocate_tensors()
    in_dets = interp.get_input_details()
    out_dets = interp.get_output_details()
    return {
        "type": "tflite",
        "inputs": [f"{d['name']}:{d['shape']}:{str(d['dtype'])}" for d in in_dets],
        "outputs": [f"{d['name']}:{d['shape']}:{str(d['dtype'])}" for d in out_dets],
    }


def discover_artifacts(root: str) -> list[str]:
    exts = (".tflite", ".pkl", ".joblib", ".keras", ".h5")
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                found.append(os.path.join(dirpath, f))
    return found


def main():
    model_dir = os.path.join(ROOT, "models")
    artifacts = discover_artifacts(model_dir)
    rows = []
    for path in artifacts:
        rel = os.path.relpath(path, ROOT)
        print("Checking:", rel)
        entry = {"artifact": rel, "status": "unknown", "notes": ""}
        try:
            if path.lower().endswith(".tflite"):
                info = try_load_tflite(path)
                entry.update({"status": "ok", "type": info.pop("type"), "details": str(info)})
            elif path.lower().endswith(('.pkl', '.joblib')):
                info = try_load_joblib(path)
                entry.update({"status": "ok", "type": info.pop("type"), "details": str(info)})
            elif path.lower().endswith(('.keras', '.h5')):
                info = try_load_keras(path)
                entry.update({"status": "ok", "type": info.pop("type"), "details": str(info)})
            else:
                entry.update({"status": "skipped", "notes": "unhandled extension"})
        except Exception as exc:
            entry.update({"status": "error", "notes": str(exc)})
            tb = traceback.format_exc()
            entry["traceback"] = tb
            print(f"  ERROR loading {rel}: {exc}")
        rows.append(entry)

    # Write CSV
    keys = ["artifact", "status", "type", "details", "notes"]
    os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
    with open(REPORT_CSV, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    print("Report saved to:", REPORT_CSV)


if __name__ == "__main__":
    main()
