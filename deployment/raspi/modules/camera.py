"""
Camera module for Raspberry Pi deployment.

Provides camera initialization and frame capture with
proper error handling and resource management.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("drunk_detection.camera")


class Camera:
    """
    Raspberry Pi camera wrapper with graceful error handling.

    Supports both Picamera2 (Raspberry Pi) and OpenCV VideoCapture (fallback).
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._camera = None
        self._use_picamera = False

    def initialize(self) -> bool:
        """
        Initialize the camera. Tries Picamera2 first, falls back to OpenCV.

        Returns:
            True if initialization succeeded.
        """
        # Try Picamera2 (Raspberry Pi)
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": (self.width, self.height)}
            )
            self._camera.configure(config)
            self._camera.start()
            self._use_picamera = True
            logger.info("Camera initialized (Picamera2: %dx%d)", self.width, self.height)
            return True
        except (ImportError, Exception) as e:
            logger.info("Picamera2 not available (%s), trying OpenCV...", e)

        # Fallback to OpenCV
        try:
            self._camera = cv2.VideoCapture(0)
            if self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._use_picamera = False
                logger.info("Camera initialized (OpenCV: %dx%d)", self.width, self.height)
                return True
            else:
                logger.error("OpenCV camera failed to open")
                return False
        except Exception as e:
            logger.error("Camera initialization failed: %s", e)
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns:
            BGR frame as numpy array, or None on failure.
        """
        try:
            if self._use_picamera:
                image = self._camera.capture_array()
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self._camera.read()
                return frame if ret else None
        except Exception as e:
            logger.error("Frame capture failed: %s", e)
            return None

    def release(self) -> None:
        """Release camera resources."""
        try:
            if self._camera is not None:
                if self._use_picamera:
                    self._camera.stop()
                else:
                    self._camera.release()
                logger.info("Camera resources released")
        except Exception as e:
            logger.error("Camera release error: %s", e)

    def __enter__(self) -> "Camera":
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.release()
