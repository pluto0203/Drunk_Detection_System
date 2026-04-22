"""
Raspberry Pi main entry point — Drunk Detection System.

Production-ready deployment with:
  - Graceful degradation when components fail
  - Health check endpoint
  - Inference benchmarking (FPS, latency)
  - Structured logging (no print statements)
  - Proper resource cleanup

Usage:
    python deployment/raspi/main.py
    python deployment/raspi/main.py --health-check
    python deployment/raspi/main.py --benchmark
"""

import argparse
import logging
import sys
import time
from typing import Dict

import cv2

from config import DeploymentConfig
from modules.camera import Camera
from modules.image_processing import EdgeInference
from modules.logger import WarningLogger
from modules.mq3_sensor import MQ3Sensor
from modules.telegram_bot import TelegramNotifier

logger = logging.getLogger("drunk_detection.main")


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("deployment.log", encoding="utf-8"),
        ],
    )


def run_benchmark(config: DeploymentConfig) -> Dict[str, float]:
    """
    Benchmark inference performance.

    Returns:
        Dictionary with FPS, avg_latency_ms, and memory_mb.
    """
    logger.info("Running inference benchmark...")

    inference = EdgeInference(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
    )
    if not inference.initialize():
        logger.error("Cannot benchmark — model loading failed")
        return {}

    camera = Camera(config.camera_width, config.camera_height)
    if not camera.initialize():
        logger.error("Cannot benchmark — camera initialization failed")
        inference.release()
        return {}

    # Warmup
    for _ in range(5):
        frame = camera.capture_frame()
        if frame is not None:
            inference.predict(frame)

    # Benchmark
    latencies = []
    num_frames = 50

    for i in range(num_frames):
        frame = camera.capture_frame()
        if frame is None:
            continue

        start = time.perf_counter()
        inference.predict(frame)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    camera.release()
    inference.release()

    if not latencies:
        return {}

    avg_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency

    results = {
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "fps": round(fps, 1),
        "frames_tested": len(latencies),
    }

    logger.info("=" * 50)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 50)
    for key, val in results.items():
        logger.info("  %-20s: %s", key, val)
    logger.info("=" * 50)

    return results


def health_check(config: DeploymentConfig) -> bool:
    """
    Run system health check.

    Verifies: model file, camera, serial port, Telegram config.

    Returns:
        True if all critical components pass.
    """
    logger.info("Running health check...")
    all_ok = True

    # Model
    import os
    if os.path.exists(config.model_path):
        logger.info("  ✅ Model file found: %s", config.model_path)
    else:
        logger.error("  ❌ Model file MISSING: %s", config.model_path)
        all_ok = False

    # Camera
    camera = Camera(config.camera_width, config.camera_height)
    if camera.initialize():
        frame = camera.capture_frame()
        if frame is not None:
            logger.info("  ✅ Camera working (%dx%d)", frame.shape[1], frame.shape[0])
        else:
            logger.error("  ❌ Camera opened but cannot capture")
            all_ok = False
        camera.release()
    else:
        logger.error("  ❌ Camera initialization FAILED")
        all_ok = False

    # MQ3 Sensor
    sensor = MQ3Sensor(config.serial_port, config.serial_baudrate)
    if sensor.initialize():
        value = sensor.read_value()
        logger.info("  ✅ MQ3 sensor connected (value=%s)", value)
        sensor.release()
    else:
        logger.warning("  ⚠️ MQ3 sensor not available (non-critical)")

    # Telegram
    if config.telegram_token and config.telegram_chat_id:
        logger.info("  ✅ Telegram configured")
    else:
        logger.warning("  ⚠️ Telegram not configured")

    status = "PASS ✅" if all_ok else "FAIL ❌"
    logger.info("Health check: %s", status)
    return all_ok


def main() -> None:
    """Main detection loop with graceful degradation."""
    parser = argparse.ArgumentParser(description="Drunk Detection System — Edge Deployment")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    args = parser.parse_args()

    setup_logging()
    config = DeploymentConfig()

    if args.health_check:
        sys.exit(0 if health_check(config) else 1)

    if args.benchmark:
        run_benchmark(config)
        return

    # Initialize components
    logger.info("=" * 60)
    logger.info("Drunk Detection System — Starting")
    logger.info("=" * 60)

    inference = EdgeInference(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
    )
    if not inference.initialize():
        logger.critical("Model loading failed. Exiting.")
        sys.exit(1)

    camera = Camera(config.camera_width, config.camera_height)
    if not camera.initialize():
        logger.critical("Camera initialization failed. Exiting.")
        inference.release()
        sys.exit(1)

    sensor = MQ3Sensor(
        config.serial_port, config.serial_baudrate, config.mq3_threshold
    )
    sensor_available = sensor.initialize()
    if not sensor_available:
        logger.warning("MQ3 sensor unavailable — running with camera-only detection")

    notifier = TelegramNotifier(
        config.telegram_token,
        config.telegram_chat_id,
        config.drunk_detection_seconds,
    )
    warning_logger = WarningLogger()

    # Notify startup
    notifier.send_message("✅ *Drunk Detection System started successfully!*")

    driver = config.current_driver
    logger.info("Current driver: %s — %s", driver["id"], driver["name"])

    drunk_frame_count = 0
    max_drunk_frames = config.drunk_detection_seconds // config.frame_interval
    last_process_time = time.time()

    try:
        while True:
            frame = camera.capture_frame()
            if frame is None:
                continue

            current_time = time.time()
            if current_time - last_process_time < config.frame_interval:
                continue

            last_process_time = current_time

            # Flip frame (camera mounted upside-down)
            frame = frame[::-1].copy()

            # Image-based detection
            image_status, confidence = inference.predict(frame)

            # Sensor-based detection
            mq3_value = sensor.read_value() if sensor_available else None
            alcohol_detected = sensor.is_alcohol_detected(mq3_value) if sensor_available else False

            # Fusion decision
            final_status = "Drunk" if (image_status == "Drunk" or alcohol_detected) else "Not Drunk"

            if final_status == "Drunk":
                drunk_frame_count += 1

                if drunk_frame_count >= max_drunk_frames:
                    # Save evidence photo
                    frame_to_save = cv2.flip(frame, 0)
                    frame_to_save = cv2.resize(
                        frame_to_save, (320, 240), interpolation=cv2.INTER_AREA
                    )
                    photo_path = f"drunk_face_{int(time.time())}.jpg"
                    cv2.imwrite(photo_path, frame_to_save)

                    # Send alert
                    notifier.send_alert(
                        mq3_value, driver["id"], driver["name"], driver["plate"],
                        photo_path=photo_path,
                    )

                    # Log warning
                    warning_logger.log(
                        driver["id"], driver["name"], driver["plate"],
                        mq3_value, photo_path,
                    )

                    drunk_frame_count = 0

                if sensor_available:
                    sensor.send_command("1")  # Activate alarm
            else:
                drunk_frame_count = 0
                if sensor_available:
                    sensor.send_command("0")  # Deactivate alarm

            logger.info(
                "Status: %s | Confidence: %.2f | MQ3: %s | Drunk frames: %d/%d",
                final_status, confidence,
                mq3_value if mq3_value else "N/A",
                drunk_frame_count, max_drunk_frames,
            )

    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C)")
    except Exception as e:
        logger.critical("Unexpected error: %s", e, exc_info=True)
        notifier.send_message(f"❌ *System error*: {e}")
    finally:
        camera.release()
        if sensor_available:
            sensor.release()
        inference.release()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
