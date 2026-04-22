"""
Knowledge Distillation entry point.

Usage:
    python scripts/distill.py \
        --teacher outputs/models/drunk_detection_model.keras \
        --train-dir ./data/train --val-dir ./data/val \
        --student-type tiny --temperature 5.0 --alpha 0.3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src.data.dataset import create_data_generators
from src.models.mobilenet_v3 import check_gpu
from src.training.distillation import DistillationTrainer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge Distillation")
    parser.add_argument("--teacher", required=True, help="Teacher model path")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/models/student_model.keras")
    parser.add_argument("--student-type", default="tiny", choices=["tiny", "micro"])
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("drunk_detection", log_dir="logs")

    logger.info("=" * 60)
    logger.info("Knowledge Distillation Pipeline")
    logger.info("=" * 60)

    check_gpu()

    # Load teacher
    logger.info("Loading teacher: %s", args.teacher)
    teacher = tf.keras.models.load_model(args.teacher)
    teacher.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Data
    data_cfg = config.get("data", {})
    train_gen, val_gen, _ = create_data_generators(
        train_dir=args.train_dir, val_dir=args.val_dir,
        img_size=tuple(data_cfg.get("img_size", [224, 224])),
        batch_size=data_cfg.get("batch_size", 32),
    )

    # Distillation
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        student_type=args.student_type,
    )
    trainer.build_student(
        input_shape=(*data_cfg.get("img_size", [224, 224]), 3),
        num_classes=data_cfg.get("num_classes", 2),
    )

    results = trainer.train(
        train_gen, val_gen,
        epochs=args.epochs, learning_rate=args.lr,
    )

    # Save
    trainer.save_student(args.output)

    # Export student to TFLite
    tflite_path = args.output.replace(".keras", ".tflite")
    from src.models.export import export_tflite
    export_tflite(args.output, tflite_path, optimize=True)

    logger.info("Distillation complete! ✅")
    logger.info("  Compression: %.1fx", results.get("compression_ratio", 0))
    logger.info("  Accuracy retained: %.1f%%", results.get("accuracy_retention", 0))


if __name__ == "__main__":
    main()
