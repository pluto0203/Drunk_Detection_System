"""
Dashboard web application for the Drunk Detection System.

Flask-based monitoring dashboard that displays violation history
with filtering, sorting, and photo evidence viewing.

Usage:
    python deployment/dashboard/app.py
    # or: flask --app deployment/dashboard/app run --host 0.0.0.0
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from flask import Flask, render_template, request

logger = logging.getLogger("drunk_detection.dashboard")

app = Flask(__name__)


@dataclass
class WarningLog:
    """Data class for a single violation record."""

    time: str
    driver_id: str
    driver_name: str
    vehicle_plate: str
    mq3_value: int
    photo_path: str = ""
    photo_url: Optional[str] = None

    def attach_photo_url(self) -> None:
        """Generate URL for the evidence photo if it exists."""
        if self.photo_path and os.path.exists(f"static/images/{self.photo_path}"):
            self.photo_url = f"/static/images/{self.photo_path}"
        else:
            self.photo_url = None


def load_logs_from_csv(file_path: str = "warnings.csv") -> List[WarningLog]:
    """
    Load violation logs from CSV file.

    Args:
        file_path: Path to the warnings CSV file.

    Returns:
        List of WarningLog objects.
    """
    if not os.path.exists(file_path):
        logger.warning("CSV file not found: %s", file_path)
        return []

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        return []

    logs: List[WarningLog] = []
    for _, row in df.iterrows():
        log = WarningLog(
            time=str(row.get("time", "")),
            driver_id=str(row.get("driver_id", "")),
            driver_name=str(row.get("driver_name", "")),
            vehicle_plate=str(row.get("vehicle_plate", "")),
            mq3_value=int(row.get("mq3_value", 0)),
            photo_path=str(row.get("photo_path", "")),
        )
        log.attach_photo_url()
        logs.append(log)

    return logs


@app.route("/", methods=["GET"])
def index():
    """Main dashboard page with optional driver ID filtering."""
    logs = load_logs_from_csv()

    driver_id = request.args.get("driver_id", "")
    if driver_id:
        logs = [log for log in logs if log.driver_id == driver_id]

    return render_template("index.html", logs=logs)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "drunk-detection-dashboard"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
