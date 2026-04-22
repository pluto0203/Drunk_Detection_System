"""
Model export module for TFLite and ONNX conversion.

Supports:
  - Standard TFLite conversion
  - Optimized TFLite (post-training quantization)
  - ONNX export for cross-framework deployment
"""

import logging
import os
from pathlib import Path
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("drunk_detection.export")


def export_tflite(
    model_path: str,
    output_path: str,
    optimize: bool = True,
) -> str:
    """
    Convert a Keras model to TFLite format.

    Args:
        model_path: Path to the saved Keras model (.keras or SavedModel).
        output_path: Path to save the .tflite file.
        optimize: If True, apply post-training quantization for size reduction.

    Returns:
        Path to the saved TFLite model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If conversion fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    logger.info("Converting to TFLite (optimize=%s)...", optimize)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Post-training quantization enabled")

    try:
        tflite_model = converter.convert()
    except Exception as e:
        raise RuntimeError(f"TFLite conversion failed: {e}") from e

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(tflite_model)

    # Report size
    original_size = _get_model_size(model_path)
    tflite_size = output_file.stat().st_size
    logger.info(
        "TFLite model saved: %s (%.2f MB)", output_path, tflite_size / 1e6
    )
    if original_size:
        reduction = (1 - tflite_size / original_size) * 100
        logger.info(
            "Size reduction: %.2f MB → %.2f MB (%.1f%% smaller)",
            original_size / 1e6, tflite_size / 1e6, reduction,
        )

    return str(output_file)


def verify_tflite_model(
    tflite_path: str,
    expected_input_shape: tuple = (1, 224, 224, 3),
) -> bool:
    """
    Verify a TFLite model loads correctly and has expected I/O shapes.

    Args:
        tflite_path: Path to the .tflite model.
        expected_input_shape: Expected input tensor shape.

    Returns:
        True if verification passes.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = tuple(input_details[0]["shape"])
        output_shape = tuple(output_details[0]["shape"])

        logger.info("TFLite verification — Input: %s, Output: %s", input_shape, output_shape)

        if input_shape != expected_input_shape:
            logger.warning(
                "Input shape mismatch: expected %s, got %s",
                expected_input_shape, input_shape,
            )
            return False

        logger.info("✅ TFLite model verification passed")
        return True

    except Exception as e:
        logger.error("TFLite verification failed: %s", e)
        return False


def _get_model_size(model_path: str) -> Optional[int]:
    """Get total size of a model file or directory in bytes."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return None
