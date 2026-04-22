"""
Evaluation entry point — Drunk Detection System.

Usage:
    python scripts/evaluate.py --model outputs/models/drunk_detection_model.keras --test-dir ./data/test
    python scripts/evaluate.py --model outputs/models/drunk_detection_model.keras --test-dir ./data/test --grad-cam
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_data_generators
from src.evaluation.evaluator import ModelEvaluator, generate_gradcam
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Drunk Detection model with comprehensive metrics.",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model (.keras or SavedModel)",
    )
    parser.add_argument(
        "--test-dir", type=str, required=True,
        help="Test data directory",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--grad-cam", action="store_true",
        help="Generate Grad-CAM visualizations",
    )
    parser.add_argument(
        "--grad-cam-samples", type=int, default=5,
        help="Number of Grad-CAM samples to generate",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log results to MLflow",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation pipeline."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logger(
        "drunk_detection",
        log_level=config.get("logging", {}).get("level", "INFO"),
        log_dir=config.get("logging", {}).get("log_dir"),
    )

    logger.info("=" * 60)
    logger.info("Drunk Detection System — Evaluation Pipeline")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading model from: %s", args.model)
    model = tf.keras.models.load_model(args.model)

    # Prepare test data
    data_cfg = config.get("data", {})
    _, _, test_gen = create_data_generators(
        train_dir=args.test_dir,  # Dummy, won't be used
        val_dir=args.test_dir,    # Dummy, won't be used
        test_dir=args.test_dir,
        img_size=tuple(data_cfg.get("img_size", [224, 224])),
        batch_size=data_cfg.get("batch_size", 32),
    )

    # Run evaluation
    evaluator = ModelEvaluator(
        model=model,
        class_names=data_cfg.get("class_names", ["Drunk", "Not Drunk"]),
        output_dir=args.output_dir,
    )

    metrics = evaluator.evaluate(test_gen, threshold=args.threshold)

    # Generate Grad-CAM visualizations
    if args.grad_cam:
        logger.info("Generating Grad-CAM visualizations...")
        eval_cfg = config.get("evaluation", {})
        grad_cam_layer = eval_cfg.get("grad_cam_layer", "Conv_1")

        # Get sample images
        test_gen.reset()
        batch_images, batch_labels = next(test_gen)
        num_samples = min(args.grad_cam_samples, len(batch_images))

        for i in range(num_samples):
            output_path = f"{args.output_dir}/gradcam_sample_{i}.png"
            try:
                generate_gradcam(
                    model=model,
                    image=batch_images[i],
                    target_layer_name=grad_cam_layer,
                    class_index=np.argmax(batch_labels[i]),
                    output_path=output_path,
                )
            except Exception as e:
                logger.warning("Grad-CAM failed for sample %d: %s", i, e)

    # Log to MLflow
    if args.mlflow:
        try:
            import mlflow

            mlflow_cfg = config.get("mlflow", {})
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "drunk-detection"))

            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics({
                    "test_accuracy": metrics.get("accuracy", 0),
                    "test_f1_weighted": metrics.get("f1_weighted", 0),
                    "test_roc_auc": metrics.get("roc_auc", 0),
                    "test_fnr": metrics.get("fnr", 0),
                })

                # Log all evaluation artifacts
                output_dir = Path(args.output_dir)
                for artifact in output_dir.glob("*.png"):
                    mlflow.log_artifact(str(artifact))

                logger.info("Evaluation results logged to MLflow")
        except ImportError:
            logger.warning("MLflow not installed. Skipping.")

    logger.info("Evaluation pipeline finished! ✅")


if __name__ == "__main__":
    main()
