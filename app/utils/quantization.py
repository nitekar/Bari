"""
Lightweight helpers to export quantized TFLite models for deployment.

Usage (from a notebook or script):

    from app.utils.quantization import export_int8_tflite
    export_int8_tflite(visual_model, "models/saved_models/visual_model_int8.tflite")

"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf


def export_int8_tflite(keras_model: tf.keras.Model, out_path: str, representative_ds: Optional[tf.data.Dataset] = None) -> None:
    """
    Export a Keras model as an INT8 quantized TFLite model.

    If `representative_ds` is provided, it should yield batches of input images
    for calibration. When omitted, dynamic range quantization is used.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if representative_ds is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_gen():
            for batch in representative_ds.take(100):
                # Assume image is the first tensor in the batch
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                yield [x]

        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

