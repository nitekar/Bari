"""
Grad-CAM utilities for the visual CNN (Keras model, not TFLite).

This is intended for offline analysis / notebooks, but is shipped here so that
the same logic can be imported from scripts without duplicating code.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def compute_gradcam(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    last_conv_layer_name: str,
    class_index: int | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model with a convolutional backbone and classification head.
    image_tensor : np.ndarray
        Array of shape (1, H, W, 3) ready for the model.
    last_conv_layer_name : str
        Name of the last convolutional layer in the model.
    class_index : int, optional
        Target class index. If None, uses argmax of model prediction.

    Returns
    -------
    np.ndarray
        Heatmap of shape (H, W) in [0, 1].
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

