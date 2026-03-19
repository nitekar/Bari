"""
tflite_converter.py — Convert Keras models to TFLite for mobile deployment
===========================================================================
Usage:
    python models/tflite_converter.py

Outputs (written to models/saved_models/):
    anemia_visual.tflite              – standard conversion
    anemia_visual_quantized.tflite    – post-training dynamic-range quantization

Also prints a benchmark table: model size, avg latency, P95 latency.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_MODEL_PATHS = [
    PROJECT_ROOT / "Notebook" / "models" / "mobilenetv2_visual_4class.h5",
    PROJECT_ROOT / "Notebook" / "models" / "mobilenetv2_finetuned_visual_model.h5",
]

OUTPUT_DIR = PROJECT_ROOT / "models" / "saved_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Non-Anemic", "Mild", "Moderate", "Severe"]


# ── Utilities ─────────────────────────────────────────────────────────────────
def find_source_model() -> str:
    """Return path to first existing source model."""
    for p in SOURCE_MODEL_PATHS:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No source model found. Checked:\n"
        + "\n".join(f"  {p}" for p in SOURCE_MODEL_PATHS)
        + "\nTrain the visual model first (Notebook Part 6)."
    )


def convert_to_tflite(
    keras_path: str,
    output_path: str,
    quantize: bool = False,
    representative_data_gen=None,
) -> float:
    """
    Convert a Keras .h5 model to TFLite format.

    Args:
        keras_path            : source model path
        output_path           : destination .tflite file
        quantize              : apply post-training quantization
        representative_data_gen: callable → generator of (1,224,224,3) float32
                                  arrays (required for full-integer quantization)

    Returns:
        file size in MB
    """
    import tensorflow as tf  # local import to allow module import without TF

    print(f"\nLoading Keras model from: {keras_path}")
    model     = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_gen is not None:
            # Full-integer quantization (requires representative dataset)
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type  = tf.uint8
            converter.inference_output_type = tf.uint8

    print("Converting …")
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved  : {output_path}")
    print(f"Size   : {size_mb:.2f} MB")
    return size_mb


def benchmark_tflite(tflite_path: str, n_runs: int = 100) -> dict:
    """
    Benchmark TFLite model inference speed using random input.

    Returns:
        dict with model_size_mb, avg_latency_ms, p50_latency_ms, p95_latency_ms
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    in_shape = in_det[0]["shape"]
    dtype    = in_det[0]["dtype"]
    dummy    = np.random.rand(*in_shape).astype(dtype)

    # Warmup
    for _ in range(5):
        interp.set_tensor(in_det[0]["index"], dummy)
        interp.invoke()

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_det[0]["index"], dummy)
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    out = interp.get_tensor(out_det[0]["index"])

    return {
        "model_size_mb" : round(os.path.getsize(tflite_path) / 1024**2, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "input_shape"   : in_shape.tolist(),
        "output_shape"  : out.shape,
        "n_classes"     : int(out.shape[-1]),
    }


def validate_tflite(tflite_path: str, image_path: str | None = None) -> dict:
    """
    Run a single inference with TFLite and return class probabilities.
    Uses a random image if image_path is not provided.
    """
    import tensorflow as tf
    from PIL import Image

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    if image_path and os.path.exists(image_path):
        img = np.array(Image.open(image_path).convert("RGB").resize((224, 224)),
                       dtype="float32") / 255.0
        inp = np.expand_dims(img, 0)
    else:
        inp = np.random.rand(1, 224, 224, 3).astype("float32")
        print("(No valid image path — using random input for validation)")

    inp = inp.astype(in_det[0]["dtype"])
    interp.set_tensor(in_det[0]["index"], inp)
    interp.invoke()
    probs = interp.get_tensor(out_det[0]["index"])[0].astype("float32")

    idx = int(np.argmax(probs))
    return {
        "diagnosis"    : CLASS_NAMES[idx],
        "confidence"   : round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TFLite Conversion & Benchmarking")
    print("=" * 60)

    source_path = find_source_model()
    print(f"\nSource model : {source_path}")
    print(f"Output dir   : {OUTPUT_DIR}")

    standard_path  = str(OUTPUT_DIR / "anemia_visual.tflite")
    quantized_path = str(OUTPUT_DIR / "anemia_visual_quantized.tflite")

    results = {}

    # ── Standard conversion ───────────────────────────────────────────────────
    print("\n[1/2] Standard TFLite (no quantization)")
    try:
        std_mb = convert_to_tflite(source_path, standard_path, quantize=False)
        results["Standard"] = std_mb
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Quantized conversion ──────────────────────────────────────────────────
    print("\n[2/2] Post-training dynamic-range quantization")
    try:
        q_mb = convert_to_tflite(source_path, quantized_path, quantize=True)
        results["Quantized"] = q_mb
        if "Standard" in results:
            reduction = (1 - q_mb / results["Standard"]) * 100
            print(f"  Size reduction vs standard: {reduction:.1f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Inference Speed Benchmark (100 runs, random input)")
    print("=" * 60)

    bench_data = []
    for label, path in [("Standard", standard_path), ("Quantized", quantized_path)]:
        if not os.path.exists(path):
            print(f"{label}: file not found, skipping.")
            continue
        print(f"\nBenchmarking {label} …")
        try:
            stats = benchmark_tflite(path, n_runs=100)
            stats["model"] = label
            bench_data.append(stats)
            print(f"  Size          : {stats['model_size_mb']} MB")
            print(f"  Avg latency   : {stats['avg_latency_ms']} ms")
            print(f"  P50 latency   : {stats['p50_latency_ms']} ms")
            print(f"  P95 latency   : {stats['p95_latency_ms']} ms")
            print(f"  Input shape   : {stats['input_shape']}")
        except Exception as e:
            print(f"  Benchmark error: {e}")

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Output Validation")
    print("=" * 60)
    for label, path in [("Standard", standard_path), ("Quantized", quantized_path)]:
        if os.path.exists(path):
            try:
                res = validate_tflite(path)
                print(f"\n{label} model prediction (random input):")
                print(f"  Diagnosis  : {res['diagnosis']}")
                print(f"  Confidence : {res['confidence']}")
                for cls, prob in res["probabilities"].items():
                    print(f"  {cls:15s}: {prob}")
            except Exception as e:
                print(f"  Validation error: {e}")

    # ── Summary table ─────────────────────────────────────────────────────────
    if bench_data:
        print("\n" + "=" * 60)
        print("  Summary Table")
        print("=" * 60)
        header = f"{'Model':12s}  {'Size(MB)':>10}  {'Avg(ms)':>10}  {'P95(ms)':>10}"
        print(header)
        print("-" * len(header))
        for row in bench_data:
            print(
                f"{row['model']:12s}  {row['model_size_mb']:>10.2f}  "
                f"{row['avg_latency_ms']:>10.2f}  {row['p95_latency_ms']:>10.2f}"
            )

    print(f"\nAll TFLite files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
