"""
Training orchestrator for the Drunk Detection System.

Implements the two-phase training pipeline:
  Phase 1: Train classification head with frozen MobileNetV3 backbone
  Phase 2: Fine-tune upper backbone layers with reduced learning rate

Integrates with MLflow for experiment tracking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

from src.models.mobilenet_v3 import build_model, compile_model, unfreeze_model
from src.training.callbacks import get_callbacks

logger = logging.getLogger("drunk_detection.trainer")


class Trainer:
    """
    Two-phase training orchestrator.

    Phase 1: Trains the classification head (backbone frozen).
    Phase 2: Fine-tunes upper backbone layers with lower LR.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer.

        Args:
            config: Full configuration dictionary from YAML.
        """
        self.config = config
        self.model: Optional[Model] = None
        self.base_model: Optional[Model] = None
        self.history_phase1: Optional[tf.keras.callbacks.History] = None
        self.history_phase2: Optional[tf.keras.callbacks.History] = None

    def build(self) -> None:
        """Build and compile the model for Phase 1."""
        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("data", {})

        self.model, self.base_model = build_model(
            img_size=tuple(data_cfg.get("img_size", [224, 224])),
            num_classes=data_cfg.get("num_classes", 2),
            dense_units=model_cfg.get("dense_units", [128, 64]),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            weights=model_cfg.get("weights", "imagenet"),
        )

        train_cfg = self.config.get("training", {})
        self.model = compile_model(
            self.model,
            learning_rate=train_cfg.get("learning_rate", 1e-4),
            loss=train_cfg.get("loss", "categorical_crossentropy"),
        )

        logger.info("Model built and compiled for Phase 1")

    def train_phase1(
        self,
        train_gen: Any,
        val_gen: Any,
        checkpoint_path: str,
    ) -> tf.keras.callbacks.History:
        """
        Phase 1: Train classification head with frozen backbone.

        Args:
            train_gen: Training data generator.
            val_gen: Validation data generator.
            checkpoint_path: Path to save best model checkpoint.

        Returns:
            Training history object.
        """
        if self.model is None:
            raise RuntimeError("Call build() before training")

        train_cfg = self.config.get("training", {})
        epochs = train_cfg.get("epochs", 100)

        callbacks = get_callbacks(
            checkpoint_path=checkpoint_path,
            monitor=train_cfg.get("checkpoint_monitor", "val_accuracy"),
            mode=train_cfg.get("checkpoint_mode", "max"),
            patience=train_cfg.get("early_stopping_patience", 25),
        )

        logger.info("=" * 60)
        logger.info("Phase 1: Training classification head (%d epochs)", epochs)
        logger.info("=" * 60)

        self.history_phase1 = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history_phase1

    def train_phase2(
        self,
        train_gen: Any,
        val_gen: Any,
        checkpoint_path: str,
    ) -> tf.keras.callbacks.History:
        """
        Phase 2: Fine-tune upper backbone layers.

        Args:
            train_gen: Training data generator.
            val_gen: Validation data generator.
            checkpoint_path: Path to save best model checkpoint.

        Returns:
            Training history object.
        """
        if self.model is None or self.base_model is None:
            raise RuntimeError("Call build() and train_phase1() first")

        ft_cfg = self.config.get("finetuning", {})
        train_cfg = self.config.get("training", {})
        epochs = ft_cfg.get("epochs", 20)

        self.model = unfreeze_model(
            self.model,
            self.base_model,
            fine_tune_at=ft_cfg.get("fine_tune_at_layer", 100),
            learning_rate=ft_cfg.get("learning_rate", 1e-5),
        )

        callbacks = get_callbacks(
            checkpoint_path=checkpoint_path,
            monitor=train_cfg.get("checkpoint_monitor", "val_accuracy"),
            mode=train_cfg.get("checkpoint_mode", "max"),
            patience=train_cfg.get("early_stopping_patience", 25),
        )

        logger.info("=" * 60)
        logger.info("Phase 2: Fine-tuning backbone (%d epochs)", epochs)
        logger.info("=" * 60)

        self.history_phase2 = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history_phase2

    def save_model(self, output_path: str, save_format: str = "keras") -> None:
        """
        Save the trained model.

        Args:
            output_path: Path to save the model.
            save_format: Format ('keras', 'h5', or 'tf').
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_format == "h5":
            self.model.save(output_path, save_format="h5")
        else:
            self.model.save(output_path)

        logger.info("Model saved to: %s (format=%s)", output_path, save_format)

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves for both phases.

        Args:
            save_path: Path to save the plot. If None, displays interactively.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Collect metrics from both phases
        train_acc = list(self.history_phase1.history["accuracy"])
        val_acc = list(self.history_phase1.history["val_accuracy"])
        train_loss = list(self.history_phase1.history["loss"])
        val_loss = list(self.history_phase1.history["val_loss"])

        phase1_epochs = len(train_acc)

        if self.history_phase2:
            train_acc += self.history_phase2.history["accuracy"]
            val_acc += self.history_phase2.history["val_accuracy"]
            train_loss += self.history_phase2.history["loss"]
            val_loss += self.history_phase2.history["val_loss"]

        # Accuracy plot
        ax1.plot(train_acc, label="Train", color="#2196F3")
        ax1.plot(val_acc, label="Validation", color="#FF5722")
        ax1.axvline(x=phase1_epochs, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
        ax1.set_title("Model Accuracy", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2.plot(train_loss, label="Train", color="#2196F3")
        ax2.plot(val_loss, label="Validation", color="#FF5722")
        ax2.axvline(x=phase1_epochs, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
        ax2.set_title("Model Loss", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Training curves saved to: %s", save_path)
        else:
            plt.show()

        plt.close(fig)
