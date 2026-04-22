"""
Telegram notification module for Drunk Detection System.

Sends async notifications with photos to a Telegram chat
when drunk driving is detected. Non-blocking design.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("drunk_detection.telegram")


class TelegramNotifier:
    """
    Async Telegram notification sender.

    Uses async API to avoid blocking the main detection loop
    while sending messages and photos.
    """

    def __init__(self, token: str, chat_id: str, detection_seconds: int = 40) -> None:
        self.token = token
        self.chat_id = chat_id
        self.detection_seconds = detection_seconds
        self._enabled = bool(token and chat_id)

        if not self._enabled:
            logger.warning(
                "Telegram notifications DISABLED — "
                "set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    def send_alert(
        self,
        mq3_value: Optional[int],
        driver_id: str,
        driver_name: str,
        vehicle_plate: str,
        photo_path: Optional[str] = None,
    ) -> bool:
        """
        Send a drunk driving alert via Telegram.

        Args:
            mq3_value: MQ3 sensor reading.
            driver_id: Driver identifier.
            driver_name: Driver full name.
            vehicle_plate: Vehicle license plate.
            photo_path: Path to evidence photo (optional).

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            logger.debug("Telegram disabled, skipping alert")
            return False

        message = self._format_message(
            mq3_value, driver_id, driver_name, vehicle_plate
        )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._send_async(message, photo_path)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error("Telegram alert failed: %s", e)
            return False

    def send_message(self, text: str) -> bool:
        """
        Send a plain text message.

        Args:
            text: Message text.

        Returns:
            True if sent successfully.
        """
        if not self._enabled:
            return False

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._send_async(text))
            loop.close()
            return result
        except Exception as e:
            logger.error("Telegram message failed: %s", e)
            return False

    async def _send_async(
        self, message: str, photo_path: Optional[str] = None
    ) -> bool:
        """Send message/photo asynchronously."""
        try:
            import telegram

            bot = telegram.Bot(self.token)
            if photo_path:
                with open(photo_path, "rb") as photo:
                    await bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=message,
                        parse_mode="Markdown",
                    )
            else:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="Markdown",
                )

            logger.info("Telegram notification sent")
            return True
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            return False

    def _format_message(
        self,
        mq3_value: Optional[int],
        driver_id: str,
        driver_name: str,
        vehicle_plate: str,
    ) -> str:
        """Format the alert message."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "🚨 *DRUNK DRIVER ALERT!* 🚨\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ *Time*: {current_time}\n"
            f"🆔 *Driver ID*: {driver_id}\n"
            f"👤 *Name*: {driver_name}\n"
            f"🚗 *Plate*: {vehicle_plate}\n"
            f"📊 *MQ3 Value*: {mq3_value if mq3_value else 'N/A'}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Drunk driving detected continuously "
            f"for {self.detection_seconds} seconds."
        )
