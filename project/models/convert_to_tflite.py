"""
TensorFlow Lite Model Conversion
==================================
Convert a trained Keras model to TFLite with post-training quantization and
evaluate model size and latency on a CPU device.

Usage:
    python convert_to_tflite.py [--model_path PATH] [--output_dir DIR]
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path


def convert_model(model_path: str, output_dir: str, quantize: bool = True) -> str:
    """Convert a saved Keras model to TFLite format with optional quantization.

    Args:
        model_path: Path to the .h5 or SavedModel file.
        output_dir: Directory to write the .tflite file.
        quantize: Apply post-training dynamic range quantization.

    Returns:
        Path to the generated .tflite file.
    """
    import tensorflow as tf

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Post-training quantization: ENABLED (dynamic range)")

    print("Converting …")
    tflite_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(model_path).stem
    suffix = "_quantized" if quantize else ""
    output_path = os.path.join(output_dir, f"{stem}{suffix}.tflite")

    with open(output_path, "wb") as fh:
        fh.write(tflite_model)

    size_mb = len(tflite_model) / (1024 ** 2)
    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")
    return output_path


def benchmark_tflite(tflite_path: str, n_runs: int = 20) -> dict:
    """Benchmark TFLite model inference speed on CPU.

    Args:
        tflite_path: Path to a .tflite file.
        n_runs: Number of inference passes to average.

    Returns:
        Dict with keys: mean_ms, min_ms, max_ms, size_mb.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Build dummy inputs matching each input tensor's shape
    dummy_inputs = []
    for detail in input_details:
        shape = list(detail["shape"])
        # Replace dynamic batch dimension
        shape = [1 if s == -1 or s == 0 else s for s in shape]
        dummy_inputs.append(np.random.rand(*shape).astype(np.float32))

    # Warm-up
    for _ in range(3):
        for detail, inp in zip(input_details, dummy_inputs):
            interpreter.set_tensor(detail["index"], inp)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        for detail, inp in zip(input_details, dummy_inputs):
            interpreter.set_tensor(detail["index"], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    size_mb = os.path.getsize(tflite_path) / (1024 ** 2)
    results = {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "min_ms":  round(float(np.min(latencies)),  2),
        "max_ms":  round(float(np.max(latencies)),  2),
        "size_mb": round(size_mb, 3),
    }
    print(f"\nBenchmark results ({n_runs} runs):")
    print(f"  Mean latency : {results['mean_ms']} ms")
    print(f"  Min  latency : {results['min_ms']}  ms")
    print(f"  Max  latency : {results['max_ms']}  ms")
    print(f"  Model size   : {results['size_mb']} MB")
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument(
        "--model_path",
        default=str(Path(__file__).parent / "saved_models" / "fusion_model_best.h5"),
        help="Path to saved Keras model (.h5 or SavedModel directory)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "saved_models"),
        help="Directory for the output .tflite file",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable post-training quantization",
    )
    args = parser.parse_args()

    tflite_path = convert_model(
        args.model_path,
        args.output_dir,
        quantize=not args.no_quantize,
    )
    benchmark_tflite(tflite_path)


if __name__ == "__main__":
    main()
