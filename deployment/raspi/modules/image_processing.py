"""
Image processing module for edge inference.

Handles face extraction and TFLite model inference
on the Raspberry Pi.
"""

import logging
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

logger = logging.getLogger("drunk_detection.inference")


class EdgeInference:
    """
    TFLite inference engine for edge deployment.

    Combines face extraction (MediaPipe) with model inference (TFLite)
    for real-time drunk detection on Raspberry Pi.
    """

    def __init__(
        self,
        model_path: str,
        target_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.7,
        num_threads: int = 4,
    ) -> None:
        self.model_path = model_path
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.num_threads = num_threads

        self._interpreter = None
        self._face_mesh = None
        self._input_details = None
        self._output_details = None

    def initialize(self) -> bool:
        """
        Initialize TFLite interpreter and MediaPipe FaceMesh.

        Returns:
            True if both components loaded successfully.
        """
        # Load TFLite model
        try:
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(
                    model_path=self.model_path, num_threads=self.num_threads
                )
            except ImportError:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(
                    model_path=self.model_path, num_threads=self.num_threads
                )

            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            logger.info("TFLite model loaded: %s", self.model_path)
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            return False

        # Initialize FaceMesh
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1
            )
            logger.info("MediaPipe FaceMesh initialized")
        except Exception as e:
            logger.error("FaceMesh initialization failed: %s", e)
            return False

        return True

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Run full inference pipeline: face extraction → prediction.

        Args:
            frame: BGR camera frame.

        Returns:
            Tuple of (status_string, confidence).
            status_string is 'Drunk' or 'Not Drunk'.
        """
        # Extract face
        face = self._extract_face(frame)
        if face is None:
            return "Not Drunk", 0.0

        # Preprocess
        face_resized = cv2.resize(face, self.target_size)
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Inference
        try:
            self._interpreter.set_tensor(
                self._input_details[0]["index"], face_input
            )
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )

            prob = float(output[0][0])
            status = "Drunk" if prob > self.confidence_threshold else "Not Drunk"

            return status, prob
        except Exception as e:
            logger.error("Inference error: %s", e)
            return "Not Drunk", 0.0

    def _extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region using MediaPipe FaceMesh."""
        try:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_img)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0]
            face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL

            # Build contour
            df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
            routes = []
            for i in range(len(df)):
                src = landmarks.landmark[df.iloc[i]["p1"]]
                tgt = landmarks.landmark[df.iloc[i]["p2"]]
                routes.append(
                    (int(image.shape[1] * src.x), int(image.shape[0] * src.y))
                )
                routes.append(
                    (int(image.shape[1] * tgt.x), int(image.shape[0] * tgt.y))
                )

            # Crop
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(routes, dtype=np.int32)], 255)
            face_extracted = cv2.bitwise_and(image, image, mask=mask)

            x_min, y_min = np.min(routes, axis=0)
            x_max, y_max = np.max(routes, axis=0)
            cropped = face_extracted[y_min:y_max, x_min:x_max]

            return cropped if cropped.size > 0 else None
        except Exception as e:
            logger.error("Face extraction error: %s", e)
            return None

    def release(self) -> None:
        """Release resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            logger.info("Inference engine released")

    def __enter__(self) -> "EdgeInference":
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.release()
