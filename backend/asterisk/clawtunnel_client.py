"""
ClawtunnelClient — REST client for https://voice.clawtunnel.com
───────────────────────────────────────────────────────────────
Drop-in replacement for ARIClient that routes all telephony
through the existing Node.js ARI platform at voice.clawtunnel.com.

The dispatcher sees the same interface (originate / wait_for_answer /
hangup / bridge / whisper) but calls are made via the REST API instead
of a raw ARI WebSocket.  The Node.js service handles the actual SIP
leg and ARI event loop — we never touch it.

ZIP extensions (5-digit internal extensions) are used for all responder
calls via  POST /v1/internal-call.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(__name__)


class ClawtunnelClient:
    """
    HTTP wrapper around the voice.clawtunnel.com REST API.

    Methods match ARIClient so ARIDispatcher works unchanged, plus an
    extra whisper_text() for zero-overhead TTS injection (clawtunnel
    handles Azure TTS server-side — no ElevenLabs file needed).
    """

    def __init__(
        self,
        base_url: str = "https://voice.clawtunnel.com",
        api_key: str = "",
        aria_extension: str = "10000",      # ARIA's own registered ZIP extension
        voice: str = "en-US-JennyNeural",   # Azure TTS voice for briefings
        callback_url: str = "",             # ARIA's public /ari-callback URL
    ):
        self.base_url      = base_url.rstrip("/")
        self.api_key       = api_key
        self.aria_extension = aria_extension
        self.voice         = voice
        self.callback_url  = callback_url

        # uuid → {"answered": bool, "hangup": bool}
        # populated by on_call_event() when the clawtunnel callback fires
        self._call_status: dict[str, dict[str, bool]] = {}
        self._session: aiohttp.ClientSession | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        log.info("ClawtunnelClient ready", base_url=self.base_url,
                 aria_ext=self.aria_extension)

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("ClawtunnelClient disconnected")

    # ── callback ingestion (called by FastAPI /ari-callback endpoint) ─────────

    def on_call_event(self, uuid: str, event: str) -> None:
        """
        Receive a status event pushed by clawtunnel to our callback URL.
        Expected event strings: "answered", "hangup", "no-answer", "failed".
        """
        entry = self._call_status.setdefault(uuid, {"answered": False, "hangup": False})
        ev = event.lower()
        if ev == "answered":
            entry["answered"] = True
            log.info("Call answered (callback)", uuid=uuid)
        elif ev in ("hangup", "no-answer", "failed", "busy"):
            entry["hangup"] = True
            log.info("Call ended (callback)", uuid=uuid, event=ev)

    # ── internal helpers ──────────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to the clawtunnel API; always injects api key."""
        assert self._session, "call connect() first"
        payload = {"apikey": self.api_key, **payload}
        url = f"{self.base_url}{path}"
        try:
            async with self._session.post(url, json=payload) as resp:
                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    data = {"status": "error", "http": resp.status}
                if data.get("status") not in ("success", None):
                    log.warning("Clawtunnel API warning", path=path,
                                response=data)
                return data
        except Exception as exc:
            log.error("Clawtunnel HTTP error", path=path, error=str(exc))
            return {"status": "error", "error": str(exc)}

    # ── ARIDispatcher-compatible interface ────────────────────────────────────

    async def create_bridge(self, name: str) -> dict:
        """
        Bridges are managed server-side by the Node.js ARI layer.
        We return a virtual bridge ID so the dispatcher can track it.
        """
        bridge_id = f"claw_bridge_{name}_{int(time.time())}"
        log.info("Virtual bridge created", bridge_id=bridge_id)
        return {"id": bridge_id}

    async def originate(
        self,
        endpoint: str,
        caller_id: str = "ARIA Emergency <0000>",
        variables: dict | None = None,
    ) -> dict:
        """
        Originate a call to a ZIP extension via POST /v1/internal-call.

        `endpoint` may be bare "10001" or prefixed "PJSIP/10001" —
        both are handled.  Returns {"id": uuid} matching ARIClient shape.
        """
        ext = endpoint.replace("PJSIP/", "").strip()
        result = await self._post("/v1/internal-call", {
            "from_extension": self.aria_extension,
            "to_extension":   ext,
        })
        uid = result.get("uuid", "")
        self._call_status[uid] = {"answered": False, "hangup": False}
        log.info("Call originated via clawtunnel",
                 to=ext, uuid=uid, status=result.get("status"))
        return {"id": uid, "channelId": uid}

    async def wait_for_answer(
        self, channel_id: str, timeout: float = 8.0
    ) -> bool:
        """
        For internal ZIP extension calls, clawtunnel's ARI layer handles
        bridging automatically and does NOT send HTTP callbacks.
        We wait a fixed settle time for the SIP endpoint to ring and
        answer, then assume success so the dispatcher proceeds to TTS.

        If a callback does arrive (e.g. for future PSTN calls), we
        honour it immediately.
        """
        settle_time = min(timeout, 3.0)  # wait 3s for call to connect
        deadline = time.monotonic() + settle_time
        while time.monotonic() < deadline:
            st = self._call_status.get(channel_id, {})
            if st.get("answered"):
                log.info("Call answered (callback)", uuid=channel_id)
                return True
            if st.get("hangup"):
                log.info("Call hung up before answer", uuid=channel_id)
                return False
            await asyncio.sleep(0.5)

        # No callback received — assume answered (internal calls don't callback)
        self._call_status.setdefault(channel_id, {})["answered"] = True
        log.info("Assuming call answered (internal call, no callback)",
                 uuid=channel_id)
        return True

    async def hangup(self, channel_id: str) -> None:
        """Hang up a call by UUID via POST /v1/hangup."""
        await self._post("/v1/hangup", {"uuid": channel_id})
        self._call_status.pop(channel_id, None)
        log.info("Call hung up", uuid=channel_id)

    async def add_to_bridge(self, bridge_id: str, channel_id: str) -> None:
        """
        /v1/internal-call already bridges both parties server-side.
        Nothing additional required here.
        """
        log.debug("add_to_bridge (managed server-side)",
                  bridge_id=bridge_id, channel_id=channel_id)

    async def create_whisper_channel(self, channel_id: str) -> dict:
        """
        ARI snoop channels are not needed — TTS is injected directly via
        /v1/tts.  Return the call UUID so whisper_text() can route it.
        """
        return {"id": channel_id}

    async def play_audio(self, channel_id: str, media_url: str) -> None:
        """
        Not used when whisper_text() is available.
        Included for interface completeness.
        """
        log.debug("play_audio: use whisper_text() instead for clawtunnel")

    async def destroy_bridge(self, bridge_id: str) -> None:
        """Virtual bridge — nothing to tear down on the API."""
        log.info("Virtual bridge destroyed", bridge_id=bridge_id)

    # ── Extended: TTS injection (no ElevenLabs file needed) ──────────────────

    async def whisper_text(self, channel_id: str, text: str) -> bool:
        """
        Play synthesised speech into an active call via POST /v1/tts.
        The clawtunnel server handles Azure TTS conversion and playback.
        This replaces the ElevenLabs→file→ARI-snoop path used by ARIClient.

        Returns True if TTS succeeded, False if the call appears dead.
        """
        result = await self._post("/v1/tts", {
            "uuid":  channel_id,
            "voice": self.voice,
            "text":  text,
        })
        status = result.get("status", "")
        ok = status in ("success", "ok", None)
        if not ok:
            log.warning("TTS failed — call likely hung up",
                        uuid=channel_id, result=result)
            self._call_status.setdefault(channel_id, {})["hangup"] = True
        else:
            log.info("TTS whispered via clawtunnel",
                     uuid=channel_id, chars=len(text))
        return ok

    def is_call_alive(self, channel_id: str) -> bool:
        """Check if a call is still considered alive."""
        st = self._call_status.get(channel_id, {})
        return not st.get("hangup", False)
