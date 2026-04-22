"""
Callback configurations for model training.

Provides standard callback setups for checkpointing,
early stopping, and learning rate scheduling.
"""

import logging
from typing import List

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

logger = logging.getLogger("drunk_detection.callbacks")


def get_callbacks(
    checkpoint_path: str,
    monitor: str = "val_accuracy",
    mode: str = "max",
    patience: int = 25,
    reduce_lr: bool = True,
) -> List[Callback]:
    """
    Create a standard set of training callbacks.

    Args:
        checkpoint_path: Path to save the best model.
        monitor: Metric to monitor for checkpointing.
        mode: 'max' for accuracy, 'min' for loss.
        patience: Number of epochs without improvement before stopping.
        reduce_lr: Whether to include ReduceLROnPlateau callback.

    Returns:
        List of Keras callbacks.
    """
    callbacks: List[Callback] = []

    # Best model checkpoint
    callbacks.append(
        ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            verbose=1,
        )
    )

    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Learning rate reduction on plateau
    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience // 3,
                min_lr=1e-7,
                verbose=1,
            )
        )

    logger.info(
        "Callbacks configured: checkpoint=%s, patience=%d, reduce_lr=%s",
        monitor, patience, reduce_lr,
    )

    return callbacks
