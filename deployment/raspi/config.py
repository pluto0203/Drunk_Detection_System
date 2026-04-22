"""
Raspberry Pi deployment configuration — Drunk Detection System.

All configuration is loaded from environment variables (.env file)
with sensible defaults. No secrets are hardcoded.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("drunk_detection.deployment")


@dataclass
class DeploymentConfig:
    """Configuration for edge deployment on Raspberry Pi."""

    # Model
    model_path: str = os.getenv(
        "MODEL_PATH", "models/DrunkClass_model.tflite"
    )

    # Telegram (from .env — NEVER hardcoded)
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Serial / MQ3 sensor
    serial_port: str = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
    serial_baudrate: int = int(os.getenv("SERIAL_BAUDRATE", "9600"))
    mq3_threshold: int = int(os.getenv("MQ3_THRESHOLD", "400"))

    # Detection parameters
    drunk_detection_seconds: int = int(os.getenv("DRUNK_DETECTION_SECONDS", "40"))
    frame_interval: int = int(os.getenv("FRAME_INTERVAL", "5"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    # Camera
    camera_width: int = 640
    camera_height: int = 480

    # Drivers
    drivers: List[Dict] = field(default_factory=list)
    current_driver: Dict = field(default_factory=lambda: {
        "id": "UNKNOWN", "name": "Unknown Driver", "plate": "UNKNOWN"
    })

    def __post_init__(self) -> None:
        """Load drivers and validate config after initialization."""
        self._load_drivers()
        self._validate()

    def _load_drivers(self) -> None:
        """Load driver list from JSON file."""
        drivers_path = Path("drivers.json")
        if drivers_path.exists():
            try:
                with open(drivers_path, "r", encoding="utf-8") as f:
                    self.drivers = json.load(f)
                if self.drivers:
                    self.current_driver = self.drivers[0]
                logger.info("Loaded %d drivers from drivers.json", len(self.drivers))
            except Exception as e:
                logger.error("Failed to load drivers.json: %s", e)

    def _validate(self) -> None:
        """Validate critical configuration values."""
        if not self.telegram_token:
            logger.warning(
                "TELEGRAM_TOKEN not set! Notifications will be disabled. "
                "Set it in your .env file."
            )
        if not self.telegram_chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set!")
        if not Path(self.model_path).exists():
            logger.warning("Model file not found: %s", self.model_path)
