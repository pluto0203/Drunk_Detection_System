"""
Knowledge Distillation module for the Drunk Detection System.

Trains a smaller "student" model to mimic the predictions of the
larger MobileNetV3 "teacher" model. The student learns from both:
  1. Hard labels (ground truth)
  2. Soft labels (teacher's probability distribution)

This produces a model that is:
  - Smaller in size (fewer parameters)
  - Faster inference on edge devices
  - Retains most of the teacher's accuracy

Interview talking point:
    "I used Knowledge Distillation to compress my model. The student
    learns the teacher's 'dark knowledge' — the relationships between
    classes encoded in soft probabilities. A teacher confident at 0.9
    Drunk / 0.1 Not-Drunk teaches the student more than a hard label."
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    ReLU,
)
from tensorflow.keras.models import Model

logger = logging.getLogger("drunk_detection.distillation")


class DistillationTrainer:
    """
    Knowledge Distillation trainer.

    Uses a trained teacher model to guide training of a smaller
    student model via soft label matching (KL divergence).
    """

    def __init__(
        self,
        teacher_model: tf.keras.Model,
        temperature: float = 5.0,
        alpha: float = 0.3,
        student_type: str = "tiny",
    ) -> None:
        """
        Initialize the distillation trainer.

        Args:
            teacher_model: Trained teacher model (MobileNetV3).
            temperature: Softmax temperature for soft labels.
                Higher = softer probability distributions = more knowledge transfer.
            alpha: Weight for distillation loss vs. hard label loss.
                alpha=0.3 means 30% hard labels + 70% soft labels.
            student_type: Student architecture ('tiny', 'micro').
        """
        self.teacher = teacher_model
        self.teacher.trainable = False  # Freeze teacher
        self.temperature = temperature
        self.alpha = alpha
        self.student_type = student_type

        self.student: Optional[tf.keras.Model] = None
        self.distiller: Optional[tf.keras.Model] = None

        logger.info(
            "Distillation trainer initialized — T=%.1f, α=%.2f, student=%s",
            temperature, alpha, student_type,
        )

    def build_student(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 2,
    ) -> tf.keras.Model:
        """
        Build a lightweight student model.

        The 'tiny' student has ~150K params (vs teacher's ~2.5M).
        The 'micro' student has ~50K params for ultra-edge deployment.

        Args:
            input_shape: Input image shape.
            num_classes: Number of output classes.

        Returns:
            Student Keras model (uncompiled).
        """
        if self.student_type == "tiny":
            self.student = self._build_tiny_student(input_shape, num_classes)
        elif self.student_type == "micro":
            self.student = self._build_micro_student(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown student type: {self.student_type}")

        # Log parameter comparison
        teacher_params = self.teacher.count_params()
        student_params = self.student.count_params()
        compression = teacher_params / student_params

        logger.info("=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("  Teacher: %s params", f"{teacher_params:,}")
        logger.info("  Student: %s params", f"{student_params:,}")
        logger.info("  Compression: %.1fx smaller", compression)
        logger.info("=" * 50)

        return self.student

    def _build_tiny_student(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
    ) -> tf.keras.Model:
        """
        Build a tiny CNN student (~150K params).

        Architecture:
            Conv2D(16) → DW-Conv(16) → Conv2D(32) → DW-Conv(32) →
            Conv2D(64) → GAP → Dense(32) → Dropout → Softmax
        """
        inputs = Input(shape=input_shape, name="student_input")

        # Block 1
        x = Conv2D(16, 3, strides=2, padding="same", use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DepthwiseConv2D(3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Block 2
        x = Conv2D(32, 1, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Block 3
        x = Conv2D(64, 1, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Head
        x = GlobalAveragePooling2D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation="softmax", name="student_output")(x)

        model = Model(inputs, outputs, name="student_tiny")
        logger.info("Built tiny student: %s params", f"{model.count_params():,}")
        return model

    def _build_micro_student(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
    ) -> tf.keras.Model:
        """
        Build a micro CNN student (~50K params).

        Ultra-lightweight for resource-constrained edge devices.
        """
        inputs = Input(shape=input_shape, name="student_input")

        x = Conv2D(8, 3, strides=2, padding="same", use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(16, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation="softmax", name="student_output")(x)

        model = Model(inputs, outputs, name="student_micro")
        logger.info("Built micro student: %s params", f"{model.count_params():,}")
        return model

    def train(
        self,
        train_generator: Any,
        val_generator: Any,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Train the student model via knowledge distillation.

        The loss function is a weighted combination of:
          L = α * CE(y_true, y_student) + (1-α) * T² * KL(σ(z_t/T), σ(z_s/T))

        Where:
          - CE = categorical cross-entropy (hard labels)
          - KL = KL divergence (soft labels from teacher)
          - T = temperature (softens probability distributions)
          - α = weight balancing hard vs soft labels

        Args:
            train_generator: Training data generator.
            val_generator: Validation data generator.
            epochs: Number of training epochs.
            learning_rate: Learning rate for student.
            callbacks: Optional Keras callbacks.

        Returns:
            Dictionary with training results and comparison metrics.
        """
        if self.student is None:
            raise RuntimeError("Call build_student() first")

        logger.info("=" * 60)
        logger.info("Knowledge Distillation Training")
        logger.info("  Temperature: %.1f", self.temperature)
        logger.info("  Alpha (hard label weight): %.2f", self.alpha)
        logger.info("  Epochs: %d", epochs)
        logger.info("=" * 60)

        # Build distillation model
        self.distiller = Distiller(
            student=self.student,
            teacher=self.teacher,
            temperature=self.temperature,
            alpha=self.alpha,
        )
        self.distiller.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        # Train
        history = self.distiller.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks or [],
            verbose=1,
        )

        # Compare teacher vs student on validation
        results = self._compare_models(val_generator)
        results["history"] = history.history

        return results

    def _compare_models(self, val_generator: Any) -> Dict[str, Any]:
        """Compare teacher and student performance on validation set."""
        logger.info("Comparing teacher vs student...")

        # Teacher evaluation
        teacher_metrics = self.teacher.evaluate(val_generator, verbose=0)
        teacher_acc = teacher_metrics[1] if len(teacher_metrics) > 1 else teacher_metrics[0]

        # Student evaluation
        student_metrics = self.student.evaluate(val_generator, verbose=0)
        student_acc = student_metrics[1] if len(student_metrics) > 1 else student_metrics[0]

        teacher_params = self.teacher.count_params()
        student_params = self.student.count_params()

        results = {
            "teacher_accuracy": float(teacher_acc),
            "student_accuracy": float(student_acc),
            "accuracy_retention": float(student_acc / teacher_acc * 100) if teacher_acc > 0 else 0,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "compression_ratio": teacher_params / student_params,
        }

        logger.info("=" * 50)
        logger.info("DISTILLATION RESULTS")
        logger.info("  Teacher accuracy:     %.4f", results["teacher_accuracy"])
        logger.info("  Student accuracy:     %.4f", results["student_accuracy"])
        logger.info("  Accuracy retention:   %.1f%%", results["accuracy_retention"])
        logger.info("  Compression ratio:    %.1fx", results["compression_ratio"])
        logger.info("=" * 50)

        return results

    def save_student(self, path: str) -> None:
        """Save the trained student model."""
        if self.student is None:
            raise RuntimeError("No student model to save")
        self.student.save(path)
        logger.info("Student model saved: %s", path)


class Distiller(tf.keras.Model):
    """
    Custom Keras model that implements knowledge distillation.

    Combines hard label loss (cross-entropy) with soft label loss
    (KL divergence between teacher and student soft predictions).
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 5.0,
        alpha: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(self, optimizer, metrics=None, **kwargs):
        """Compile with custom loss functions."""
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.hard_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.soft_loss_fn = tf.keras.losses.KLDivergence()

    def call(self, inputs, training=False):
        """Forward pass through student."""
        return self.student(inputs, training=training)

    def train_step(self, data):
        """Custom training step with distillation loss."""
        x, y_true = data

        # Teacher predictions (no gradient)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student predictions
            student_preds = self.student(x, training=True)

            # Hard loss: student vs ground truth
            hard_loss = self.hard_loss_fn(y_true, student_preds)

            # Soft loss: student vs teacher (with temperature scaling)
            teacher_soft = tf.nn.softmax(
                tf.math.log(teacher_preds + 1e-10) / self.temperature
            )
            student_soft = tf.nn.softmax(
                tf.math.log(student_preds + 1e-10) / self.temperature
            )
            soft_loss = self.soft_loss_fn(teacher_soft, student_soft)

            # Combined loss
            # T² scaling ensures gradient magnitudes are balanced
            total_loss = (
                self.alpha * hard_loss
                + (1 - self.alpha) * (self.temperature ** 2) * soft_loss
            )

        # Update student weights
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )

        # Update metrics
        self.compiled_metrics.update_state(y_true, student_preds)

        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        """Custom test step."""
        x, y_true = data
        student_preds = self.student(x, training=False)
        loss = self.hard_loss_fn(y_true, student_preds)

        self.compiled_metrics.update_state(y_true, student_preds)

        return {
            "loss": loss,
            **{m.name: m.result() for m in self.metrics},
        }
