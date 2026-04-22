"""
MobileNetV3Small model architecture for drunk detection.

Implements a two-phase transfer learning approach:
  Phase 1: Train only the classification head (backbone frozen)
  Phase 2: Fine-tune upper layers of the backbone with lower LR

Architecture:
  MobileNetV3Small (ImageNet) → GAP → Dense(128, swish) → BN → Dense(64, swish) → Dropout → Softmax
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger("drunk_detection.model")


def build_model(
    img_size: Tuple[int, int] = (224, 224),
    num_classes: int = 2,
    dense_units: List[int] = None,
    dropout_rate: float = 0.3,
    weights: str = "imagenet",
) -> Tuple[Model, Model]:
    """
    Build MobileNetV3Small with custom classification head.

    Args:
        img_size: Input image dimensions (H, W).
        num_classes: Number of output classes.
        dense_units: List of units for dense layers. Defaults to [128, 64].
        dropout_rate: Dropout rate before final layer.
        weights: Pre-trained weights ('imagenet' or None).

    Returns:
        Tuple of (full_model, base_model) for fine-tuning control.
    """
    if dense_units is None:
        dense_units = [128, 64]

    # Backbone
    base_model = MobileNetV3Small(
        weights=weights,
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False  # Frozen for Phase 1

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for units in dense_units[:-1]:
        x = Dense(units, activation="swish")(x)
        x = BatchNormalization()(x)

    x = Dense(dense_units[-1], activation="swish")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    logger.info(
        "Model built — Trainable: %s, Non-trainable: %s, Total: %s",
        f"{trainable:,}", f"{non_trainable:,}", f"{trainable + non_trainable:,}",
    )

    return model, base_model


def compile_model(
    model: Model,
    learning_rate: float = 1e-4,
    loss: str = "categorical_crossentropy",
) -> Model:
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for Adam optimizer.
        loss: Loss function name.

    Returns:
        Compiled model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    logger.info("Model compiled — LR=%.6f, loss=%s", learning_rate, loss)
    return model


def unfreeze_model(
    model: Model,
    base_model: Model,
    fine_tune_at: int = 100,
    learning_rate: float = 1e-5,
) -> Model:
    """
    Unfreeze upper layers of backbone for Phase 2 fine-tuning.

    Args:
        model: Full model.
        base_model: Backbone model.
        fine_tune_at: Layer index to start unfreezing from.
        learning_rate: Lower learning rate for fine-tuning.

    Returns:
        Re-compiled model with partially unfrozen backbone.
    """
    base_model.trainable = True

    # Freeze layers below fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    unfrozen_count = sum(1 for l in base_model.layers[fine_tune_at:] if l.trainable)
    logger.info(
        "Phase 2: Unfroze %d layers (from layer %d onwards)",
        unfrozen_count, fine_tune_at,
    )

    model = compile_model(model, learning_rate=learning_rate)
    return model


def check_gpu() -> bool:
    """
    Check GPU availability and configure memory growth.

    Returns:
        True if GPU is available, False otherwise.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("No GPU found. Training will run on CPU.")
        return False

    logger.info("Found %d GPU(s): %s", len(gpus), gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("GPU memory growth setting failed: %s", e)

    return True
