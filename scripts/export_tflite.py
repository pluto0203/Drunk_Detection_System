"""
TFLite export entry point — Drunk Detection System.

Usage:
    python scripts/export_tflite.py --model outputs/models/drunk_detection_model.keras
    python scripts/export_tflite.py --model outputs/models/drunk_detection_model.keras --optimize --verify
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.export import export_tflite, verify_tflite_model
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained model to TFLite for edge deployment.",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/models/drunk_detection.tflite",
        help="Output path for TFLite model",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Apply post-training quantization",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the exported model",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Run the TFLite export pipeline."""
    args = parse_args()

    config = load_config(args.config)
    logger = setup_logger(
        "drunk_detection",
        log_level=config.get("logging", {}).get("level", "INFO"),
        log_dir=config.get("logging", {}).get("log_dir"),
    )

    logger.info("=" * 60)
    logger.info("Drunk Detection System — TFLite Export")
    logger.info("=" * 60)

    # Export
    tflite_path = export_tflite(
        model_path=args.model,
        output_path=args.output,
        optimize=args.optimize,
    )

    # Verify
    if args.verify:
        data_cfg = config.get("data", {})
        img_size = tuple(data_cfg.get("img_size", [224, 224]))
        expected_shape = (1, *img_size, 3)

        success = verify_tflite_model(tflite_path, expected_input_shape=expected_shape)
        if not success:
            logger.error("Verification failed!")
            sys.exit(1)

    logger.info("Export completed! ✅")
    logger.info("Deploy '%s' to Raspberry Pi", tflite_path)


if __name__ == "__main__":
    main()
