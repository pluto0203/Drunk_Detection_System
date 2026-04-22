"""
Training entry point — Drunk Detection System.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --train-dir ./data/train --val-dir ./data/val
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_data_generators
from src.models.mobilenet_v3 import check_gpu
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Drunk Detection model (two-phase transfer learning).",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--train-dir", type=str, help="Training data directory")
    parser.add_argument("--val-dir", type=str, help="Validation data directory")
    parser.add_argument("--test-dir", type=str, help="Test data directory")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="outputs/checkpoints/best_model.keras",
        help="Path for model checkpoint",
    )
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip Phase 2 fine-tuning",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training pipeline."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logger(
        "drunk_detection",
        log_level=config.get("logging", {}).get("level", "INFO"),
        log_dir=config.get("logging", {}).get("log_dir"),
    )

    logger.info("=" * 60)
    logger.info("Drunk Detection System — Training Pipeline")
    logger.info("=" * 60)

    # CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    # Check GPU
    check_gpu()

    # Prepare data
    train_dir = args.train_dir or input("Enter training data path: ")
    val_dir = args.val_dir or input("Enter validation data path: ")
    test_dir = args.test_dir

    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})

    train_gen, val_gen, test_gen = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        img_size=tuple(data_cfg.get("img_size", [224, 224])),
        batch_size=data_cfg.get("batch_size", 32),
        augmentation_config=aug_cfg if aug_cfg else None,
    )

    # MLflow tracking (optional)
    if args.mlflow:
        try:
            import mlflow

            mlflow_cfg = config.get("mlflow", {})
            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "drunk-detection"))
            mlflow.start_run()

            # Log parameters
            mlflow.log_params({
                "model": config.get("model", {}).get("backbone", "MobileNetV3Small"),
                "img_size": str(data_cfg.get("img_size")),
                "batch_size": data_cfg.get("batch_size"),
                "epochs_phase1": config.get("training", {}).get("epochs"),
                "lr_phase1": config.get("training", {}).get("learning_rate"),
                "epochs_phase2": config.get("finetuning", {}).get("epochs"),
                "lr_phase2": config.get("finetuning", {}).get("learning_rate"),
                "dropout": config.get("model", {}).get("dropout_rate"),
            })
            logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not installed. Skipping tracking.")
            args.mlflow = False

    # Build and train
    trainer = Trainer(config)
    trainer.build()

    # Phase 1
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.train_phase1(train_gen, val_gen, args.checkpoint_path)

    # Phase 2
    if not args.skip_finetune:
        trainer.train_phase2(train_gen, val_gen, args.checkpoint_path)

    # Save model and plots
    final_path = f"{args.output_dir}/drunk_detection_model.keras"
    trainer.save_model(final_path)
    trainer.plot_history(save_path=f"{args.output_dir}/training_curves.png")

    # Evaluate on test set if available
    if test_gen:
        logger.info("Evaluating on test set...")
        test_loss, test_acc = trainer.model.evaluate(test_gen)
        logger.info("Test Loss: %.4f, Test Accuracy: %.4f", test_loss, test_acc)

        if args.mlflow:
            import mlflow
            mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

    # Log to MLflow
    if args.mlflow:
        import mlflow

        mlflow.log_artifact(f"{args.output_dir}/training_curves.png")
        mlflow.keras.log_model(trainer.model, "model")
        mlflow.end_run()
        logger.info("MLflow run completed")

    logger.info("Training pipeline finished successfully! ✅")


if __name__ == "__main__":
    main()
