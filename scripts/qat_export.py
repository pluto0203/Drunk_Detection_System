"""
Quantization-Aware Training entry point.

Usage:
    python scripts/qat_export.py \
        --model outputs/models/drunk_detection_model.keras \
        --train-dir ./data/train --val-dir ./data/val \
        --output outputs/models/drunk_detection_qat.tflite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.data.dataset import create_data_generators
from src.models.mobilenet_v3 import check_gpu
from src.training.quantization import (
    QuantizationAwareTrainer,
    create_representative_dataset,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization-Aware Training")
    parser.add_argument("--model", required=True, help="Trained model path")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output", default="outputs/models/drunk_detection_qat.tflite"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--full-int8", action="store_true",
        help="Full INT8 quantization with representative dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("drunk_detection", log_dir="logs")

    logger.info("=" * 60)
    logger.info("Quantization-Aware Training Pipeline")
    logger.info("=" * 60)

    check_gpu()

    # Load model
    model = tf.keras.models.load_model(args.model)

    # Data
    data_cfg = config.get("data", {})
    train_gen, val_gen, _ = create_data_generators(
        train_dir=args.train_dir, val_dir=args.val_dir,
        img_size=tuple(data_cfg.get("img_size", [224, 224])),
        batch_size=data_cfg.get("batch_size", 32),
    )

    # QAT
    qat = QuantizationAwareTrainer(model)
    qat.prepare(learning_rate=args.lr)
    results = qat.train(train_gen, val_gen, epochs=args.epochs)

    # Export
    rep_dataset = None
    if args.full_int8:
        rep_dataset = create_representative_dataset(val_gen, num_samples=100)

    export_results = qat.export_tflite(args.output, rep_dataset)

    logger.info("QAT pipeline complete! ✅")
    logger.info("  Accuracy: %.4f → %.4f", results["pre_qat_accuracy"],
                results["post_qat_accuracy"])
    logger.info("  Model: %s (%.3f MB)", args.output, export_results["size_mb"])


if __name__ == "__main__":
    main()
