"""
MQ3 alcohol sensor module for Raspberry Pi deployment.

Handles serial communication with Arduino + MQ3 sensor.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("drunk_detection.mq3_sensor")


class MQ3Sensor:
    """
    MQ3 alcohol sensor interface via serial communication.

    Reads analog values from the MQ3 sensor connected through Arduino
    and provides alcohol detection status.
    """

    def __init__(self, port: str, baudrate: int = 9600, threshold: int = 400) -> None:
        self.port = port
        self.baudrate = baudrate
        self.threshold = threshold
        self._serial = None

    def initialize(self) -> bool:
        """
        Initialize serial connection to Arduino.

        Returns:
            True if connection established successfully.
        """
        try:
            import serial

            self._serial = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # Wait for Arduino to reset
            logger.info(
                "MQ3 sensor connected: %s @ %d baud", self.port, self.baudrate
            )
            return True
        except Exception as e:
            logger.error("Serial initialization failed: %s", e)
            return False

    def read_value(self) -> Optional[int]:
        """
        Read the current MQ3 sensor value.

        Returns:
            Integer sensor value, or None on read failure.
        """
        if self._serial is None:
            return None

        try:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            data = self._serial.readline().decode().strip()
            return int(data) if data.isdigit() else None
        except Exception as e:
            logger.error("MQ3 read error: %s", e)
            return None

    def is_alcohol_detected(self, value: Optional[int] = None) -> bool:
        """
        Check if alcohol is detected above threshold.

        Args:
            value: Sensor value to check. If None, reads a new value.

        Returns:
            True if alcohol concentration exceeds threshold.
        """
        if value is None:
            value = self.read_value()
        return value is not None and value > self.threshold

    def send_command(self, command: str) -> None:
        """
        Send a command to Arduino (e.g., '1' for alarm, '0' for clear).

        Args:
            command: Command string to send.
        """
        if self._serial is None:
            return

        try:
            self._serial.write(command.encode())
        except Exception as e:
            logger.error("Serial write error: %s", e)

    def release(self) -> None:
        """Close serial connection."""
        if self._serial is not None:
            try:
                self._serial.flush()
                self._serial.close()
                logger.info("MQ3 sensor connection closed")
            except Exception as e:
                logger.error("Serial close error: %s", e)

    def __enter__(self) -> "MQ3Sensor":
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.release()
