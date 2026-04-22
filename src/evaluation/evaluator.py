"""
Comprehensive model evaluation module for the Drunk Detection System.

Provides full evaluation suite including:
  - Accuracy, Precision, Recall, F1-score per class
  - ROC-AUC curve
  - Precision-Recall curve
  - Confusion Matrix
  - Classification Report
  - False Negative Rate (critical for safety)
  - Threshold analysis for optimal operating point
  - Grad-CAM visualization

Interview talking point:
    "In a safety-critical system, False Negative Rate matters more than
    overall accuracy — a drunk driver classified as sober is dangerous.
    I optimized my threshold for high recall on the 'Drunk' class."
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("drunk_detection.evaluator")


class ModelEvaluator:
    """
    Comprehensive model evaluation with safety-critical metrics.

    Designed for binary classification (Drunk vs Not Drunk) with
    emphasis on minimizing False Negatives.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str] = None,
        output_dir: str = "outputs/evaluation",
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            model: Trained Keras model.
            class_names: List of class names. Defaults to ['Drunk', 'Not Drunk'].
            output_dir: Directory to save evaluation artifacts.
        """
        self.model = model
        self.class_names = class_names or ["Drunk", "Not Drunk"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.predictions: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None
        self.predicted_labels: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}

    def evaluate(
        self,
        test_generator: Any,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite on the test set.

        Args:
            test_generator: Test data generator.
            threshold: Classification threshold (default 0.5).

        Returns:
            Dictionary of all computed metrics.
        """
        logger.info("=" * 60)
        logger.info("Running comprehensive evaluation")
        logger.info("=" * 60)

        # Get predictions
        self.predictions = self.model.predict(test_generator, verbose=1)
        self.true_labels = test_generator.classes

        # Apply threshold for binary classification
        if self.predictions.shape[1] == 2:
            # Use probability of class 0 (Drunk) for thresholding
            drunk_probs = self.predictions[:, 0]
            self.predicted_labels = (drunk_probs >= threshold).astype(int)
            # Invert: class 0 = Drunk, class 1 = Not Drunk
            self.predicted_labels = 1 - self.predicted_labels
        else:
            self.predicted_labels = np.argmax(self.predictions, axis=1)

        # Compute metrics
        self._compute_basic_metrics()
        self._compute_per_class_metrics()
        self._compute_safety_metrics()

        # Generate visualizations
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_precision_recall_curve()
        self._plot_threshold_analysis()

        # Print summary
        self._print_report()

        return self.metrics

    def _compute_basic_metrics(self) -> None:
        """Compute accuracy, weighted F1, and overall metrics."""
        self.metrics["accuracy"] = accuracy_score(
            self.true_labels, self.predicted_labels
        )
        self.metrics["f1_weighted"] = f1_score(
            self.true_labels, self.predicted_labels, average="weighted"
        )
        self.metrics["precision_weighted"] = precision_score(
            self.true_labels, self.predicted_labels, average="weighted"
        )
        self.metrics["recall_weighted"] = recall_score(
            self.true_labels, self.predicted_labels, average="weighted"
        )

        logger.info("Accuracy: %.4f", self.metrics["accuracy"])
        logger.info("F1 (weighted): %.4f", self.metrics["f1_weighted"])

    def _compute_per_class_metrics(self) -> None:
        """Compute per-class precision, recall, and F1."""
        report = classification_report(
            self.true_labels,
            self.predicted_labels,
            target_names=self.class_names,
            output_dict=True,
        )
        self.metrics["classification_report"] = report

        for cls_name in self.class_names:
            if cls_name in report:
                logger.info(
                    "  %s — P: %.4f, R: %.4f, F1: %.4f",
                    cls_name,
                    report[cls_name]["precision"],
                    report[cls_name]["recall"],
                    report[cls_name]["f1-score"],
                )

    def _compute_safety_metrics(self) -> None:
        """
        Compute safety-critical metrics (FNR, ROC-AUC).

        FNR = FN / (FN + TP) — the rate of drunk drivers missed.
        """
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        self.metrics["confusion_matrix"] = cm

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            self.metrics["true_positives"] = int(tp)
            self.metrics["true_negatives"] = int(tn)
            self.metrics["false_positives"] = int(fp)
            self.metrics["false_negatives"] = int(fn)
            self.metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            self.metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            logger.info(
                "False Negative Rate (FNR): %.4f — %d drunk drivers missed",
                self.metrics["fnr"], fn,
            )

        # ROC-AUC
        if self.predictions.shape[1] == 2:
            try:
                self.metrics["roc_auc"] = roc_auc_score(
                    self.true_labels, self.predictions[:, 1]
                )
                logger.info("ROC-AUC: %.4f", self.metrics["roc_auc"])
            except ValueError as e:
                logger.warning("Could not compute ROC-AUC: %s", e)

    def _plot_confusion_matrix(self) -> None:
        """Plot and save confusion matrix."""
        cm = self.metrics.get("confusion_matrix")
        if cm is None:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        save_path = self.output_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved: %s", save_path)

    def _plot_roc_curve(self) -> None:
        """Plot and save ROC curve."""
        if self.predictions.shape[1] != 2:
            return

        try:
            fpr, tpr, _ = roc_curve(self.true_labels, self.predictions[:, 1])
            roc_auc = auc(fpr, tpr)
        except ValueError:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curve saved: %s", save_path)

    def _plot_precision_recall_curve(self) -> None:
        """Plot and save Precision-Recall curve."""
        if self.predictions.shape[1] != 2:
            return

        try:
            precision, recall, _ = precision_recall_curve(
                self.true_labels, self.predictions[:, 1]
            )
            pr_auc = auc(recall, precision)
        except ValueError:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#FF5722", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
        ax.fill_between(recall, precision, alpha=0.1, color="#FF5722")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / "precision_recall_curve.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Precision-Recall curve saved: %s", save_path)

    def _plot_threshold_analysis(
        self,
        threshold_range: Tuple[float, float] = (0.1, 0.95),
        num_steps: int = 50,
    ) -> None:
        """
        Plot metrics vs. classification threshold.

        Helps find the optimal operating point for safety-critical
        deployment (maximize recall on Drunk class).
        """
        if self.predictions.shape[1] != 2:
            return

        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
        accuracies, precisions, recalls, f1s, fnrs = [], [], [], [], []

        for t in thresholds:
            pred = (self.predictions[:, 0] >= t).astype(int)
            pred = 1 - pred  # Invert for class indexing

            acc = accuracy_score(self.true_labels, pred)
            prec = precision_score(self.true_labels, pred, average="weighted", zero_division=0)
            rec = recall_score(self.true_labels, pred, average="weighted", zero_division=0)
            f1 = f1_score(self.true_labels, pred, average="weighted", zero_division=0)

            cm = confusion_matrix(self.true_labels, pred)
            if cm.shape == (2, 2):
                _, _, fn, tp = cm.ravel()
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            else:
                fnr = 0

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            fnrs.append(fnr)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, accuracies, label="Accuracy", color="#2196F3", lw=2)
        ax.plot(thresholds, precisions, label="Precision", color="#4CAF50", lw=2)
        ax.plot(thresholds, recalls, label="Recall", color="#FF9800", lw=2)
        ax.plot(thresholds, f1s, label="F1-Score", color="#9C27B0", lw=2)
        ax.plot(thresholds, fnrs, label="FNR (↓ better)", color="#F44336", lw=2, linestyle="--")

        # Find optimal threshold (minimize FNR while F1 > 0.8)
        best_idx = None
        for i, (f1, fnr) in enumerate(zip(f1s, fnrs)):
            if f1 > 0.8:
                if best_idx is None or fnrs[i] < fnrs[best_idx]:
                    best_idx = i

        if best_idx is not None:
            ax.axvline(
                x=thresholds[best_idx], color="red", linestyle=":",
                alpha=0.7, label=f"Optimal: {thresholds[best_idx]:.2f}",
            )
            self.metrics["optimal_threshold"] = float(thresholds[best_idx])
            logger.info(
                "Optimal threshold: %.3f (FNR=%.4f, F1=%.4f)",
                thresholds[best_idx], fnrs[best_idx], f1s[best_idx],
            )

        ax.set_xlabel("Classification Threshold", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(
            "Threshold Analysis — Finding Optimal Operating Point",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / "threshold_analysis.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Threshold analysis saved: %s", save_path)

    def _print_report(self) -> None:
        """Print a formatted evaluation summary."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info("Accuracy:           %.4f", self.metrics.get("accuracy", 0))
        logger.info("F1 (weighted):      %.4f", self.metrics.get("f1_weighted", 0))
        logger.info("Precision (weighted):%.4f", self.metrics.get("precision_weighted", 0))
        logger.info("Recall (weighted):  %.4f", self.metrics.get("recall_weighted", 0))
        logger.info("ROC-AUC:            %.4f", self.metrics.get("roc_auc", 0))
        logger.info("FNR (Drunk missed): %.4f", self.metrics.get("fnr", 0))
        if "optimal_threshold" in self.metrics:
            logger.info("Optimal Threshold:  %.3f", self.metrics["optimal_threshold"])
        logger.info("=" * 60)

        # Full classification report
        report_str = classification_report(
            self.true_labels,
            self.predicted_labels,
            target_names=self.class_names,
        )
        logger.info("\nClassification Report:\n%s", report_str)


def generate_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    target_layer_name: str = "Conv_1",
    class_index: int = 0,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap showing WHERE the model looks.

    Visualizes which regions of the face image contribute most
    to the drunk/not-drunk classification decision.

    Args:
        model: Trained Keras model.
        image: Preprocessed input image (224x224x3, normalized).
        target_layer_name: Name of the convolutional layer for Grad-CAM.
        class_index: Class index to visualize (0=Drunk by default).
        output_path: Path to save the visualization (optional).

    Returns:
        Grad-CAM heatmap as numpy array.
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output],
    )

    # Compute gradient
    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor)
        class_output = predictions[:, class_index]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generate heatmap
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Overlay on original image
    if output_path:
        import cv2

        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        if image.max() <= 1.0:
            display_image = np.uint8(image * 255)
        else:
            display_image = np.uint8(image)

        superimposed = cv2.addWeighted(display_image, 0.6, heatmap_colored, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(display_image[:, :, ::-1])
        axes[0].set_title("Original", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(superimposed[:, :, ::-1])
        axes[2].set_title("Overlay", fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(
            f"Grad-CAM — Where the model looks (class: {class_index})",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grad-CAM saved: %s", output_path)

    return heatmap
