"""
Hyperparameter Tuning entry point.

Usage:
    python scripts/tune.py \
        --train-dir ./data/train --val-dir ./data/val \
        --n-trials 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mobilenet_v3 import check_gpu
from src.training.tuning import run_tuning
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning (Optuna)")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("drunk_detection", log_dir="logs")

    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning Pipeline")
    logger.info("=" * 60)

    check_gpu()

    data_cfg = config.get("data", {})
    img_size = tuple(data_cfg.get("img_size", [224, 224]))

    results = run_tuning(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        n_trials=args.n_trials,
        img_size=img_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )

    logger.info("Tuning complete! ✅")
    logger.info("Best params — use these in configs/default.yaml:")
    for k, v in results["best_params"].items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
