"""
Warning logger module for Drunk Detection System.

Logs violation events to CSV with structured data.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("drunk_detection.warning_logger")


class WarningLogger:
    """
    Structured violation logger that writes to CSV.

    Each warning record includes timestamp, driver info,
    sensor reading, and evidence photo path.
    """

    HEADERS = [
        "time", "driver_id", "driver_name",
        "vehicle_plate", "mq3_value", "photo_path",
    ]

    def __init__(self, log_file: str = "warnings.csv") -> None:
        self.log_file = Path(log_file)
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        """Create file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log(
        self,
        driver_id: str,
        driver_name: str,
        vehicle_plate: str,
        mq3_value: Optional[int],
        photo_path: str = "",
    ) -> None:
        """
        Log a drunk driving violation.

        Args:
            driver_id: Driver identifier.
            driver_name: Driver full name.
            vehicle_plate: Vehicle license plate.
            mq3_value: MQ3 sensor reading.
            photo_path: Path to evidence photo.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, driver_id, driver_name,
                vehicle_plate, mq3_value or "N/A", photo_path,
            ])

        logger.info(
            "Warning logged: driver=%s, mq3=%s, time=%s",
            driver_id, mq3_value, timestamp,
        )
