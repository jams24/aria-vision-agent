"""
ARIA Telegram Notifier
──────────────────────
Sends incident snapshot images + captions to a Telegram channel/group
via the Bot API.  One bot handles ALL event types.

Env vars:
    TELEGRAM_BOT_TOKEN   — BotFather token  (e.g. 123456:ABC-DEF...)
    TELEGRAM_CHAT_ID     — Target chat/channel ID (e.g. -1001234567890)
"""
from __future__ import annotations

import asyncio
import os
import time
from io import BytesIO

import httpx
import structlog

log = structlog.get_logger(__name__)

# Emoji map for each incident type
_ICONS = {
    "sudden_collapse":      "\U0001f6a8",   # red rotating light
    "medical_fall":         "\U0001f6a8",
    "fire":                 "\U0001f525",   # fire
    "security_intrusion":   "\U0001f6e1\ufe0f",   # shield
    "unresponsive_person":  "\u26a0\ufe0f",        # warning
}


class TelegramNotifier:
    """Async Telegram Bot API client — sends photos with captions."""

    API = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id   = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._client   = httpx.AsyncClient(timeout=15)
        self.enabled    = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            log.warning("Telegram notifier disabled — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
        else:
            log.info("Telegram notifier ready",
                     chat_id=self.chat_id, bot=self.bot_token[:12] + "...")

    async def send_incident(
        self,
        incident_type: str,
        severity: str,
        location: str,
        description: str,
        camera_id: str,
        timestamp: float,
        frame_jpg: bytes | None = None,
    ) -> bool:
        """
        Send an incident alert to the Telegram channel.
        Includes the snapshot image if available.
        Returns True on success.
        """
        if not self.enabled:
            return False

        icon = _ICONS.get(incident_type, "\U0001f4e2")  # default: megaphone
        ts_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        display_type = incident_type.replace("_", " ").upper()

        caption = (
            f"{icon} <b>ARIA ALERT — {display_type}</b>\n\n"
            f"<b>Severity:</b> {severity}\n"
            f"<b>Location:</b> {location}\n"
            f"<b>Camera:</b> {camera_id}\n"
            f"<b>Time:</b> {ts_str}\n\n"
            f"<i>{description}</i>"
        )

        try:
            if frame_jpg:
                # Send photo with caption
                url = f"{self.API}/bot{self.bot_token}/sendPhoto"
                files = {"photo": ("snapshot.jpg", BytesIO(frame_jpg), "image/jpeg")}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                }
                resp = await self._client.post(url, data=data, files=files)
            else:
                # No image — send text message
                url = f"{self.API}/bot{self.bot_token}/sendMessage"
                data = {
                    "chat_id": self.chat_id,
                    "text": caption,
                    "parse_mode": "HTML",
                }
                resp = await self._client.post(url, data=data)

            if resp.status_code == 200 and resp.json().get("ok"):
                log.info("Telegram alert sent", incident_type=incident_type,
                         chat_id=self.chat_id)
                return True
            else:
                log.error("Telegram API error", status=resp.status_code,
                          body=resp.text[:300])
                return False

        except Exception as e:
            log.error("Telegram send failed", error=str(e))
            return False

    async def close(self):
        await self._client.aclose()
