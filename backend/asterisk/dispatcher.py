"""
ARIA Dispatcher
───────────────
Orchestrates the full response to an incident:
  1. Originates calls to responders (by tier)
  2. Creates the incident conference bridge
  3. Whispers a live AI briefing to each responder as they join
  4. Runs the escalation ladder if responders don't answer
  5. Optionally triggers PA announcements
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

import structlog
from elevenlabs import AsyncElevenLabs

from backend.asterisk.client import ARIClient

log = structlog.get_logger(__name__)


@dataclass
class Incident:
    id: str
    type: str
    severity: str          # LOW / MEDIUM / HIGH / CRITICAL
    location: str
    description: str
    camera_id: str
    timestamp: float
    bridge_id: str | None = None
    responder_channels: dict[str, str] = field(default_factory=dict)  # responder_id → channel_id
    whisper_channels:   dict[str, str] = field(default_factory=dict)  # channel_id → snoop_id
    resolved: bool = False
    updates: list[str] = field(default_factory=list)
    _live_update_task: asyncio.Task | None = field(default=None, repr=False)


class ARIDispatcher:
    """
    Core ARIA telephony brain.
    Uses ARIClient for low-level ARI calls and handles
    all the orchestration logic on top.
    """

    def __init__(
        self,
        ari: ARIClient,
        responders_config: dict,
        protocols_config: dict,
        elevenlabs_key: str,
        voice_id: str,
        ws_broadcaster: Any | None = None,
    ):
        self.ari         = ari
        self.responders  = {r["id"]: r for r in responders_config["responders"]}
        self.groups      = responders_config.get("groups", {})
        self.protocols   = protocols_config["protocols"]
        self._tts        = AsyncElevenLabs(api_key=elevenlabs_key)
        self._voice_id   = voice_id
        self._broadcast  = ws_broadcaster  # async callable(event_dict)

        self.active_incidents: dict[str, Incident] = {}

        # Global call lock — asyncio.Lock prevents ALL concurrent dispatches.
        # Only one call can be in progress at a time.
        self._call_lock = asyncio.Lock()
        self._call_active = False  # True while a call is in progress

        # Callback: async (frame_jpg_bytes) -> str
        # Set by vision_aria.py to let the dispatcher ask Gemini
        # to describe the current scene for live TTS updates.
        self.scene_describer: Callable | None = None
        # Latest JPEG frame from the camera (set by vision processor)
        self.latest_frame: bytes | None = None

    # ──────────────────────────────────────────────────────────────
    # Public API (called by Claude tools)
    # ──────────────────────────────────────────────────────────────

    async def initiate_response(
        self,
        incident_id: str,
        incident_type: str,
        severity: str,
        location: str,
        description: str,
        camera_id: str,
        timestamp: float,
    ) -> dict:
        """
        Entry point called by Claude's tool call.
        Creates incident, opens bridge, dispatches first-tier responders.
        """
        # Normalize incident_type: Gemini may return "MEDICAL FALL" (uppercase, spaces)
        # but protocols.yaml keys use "medical_fall" (lowercase, underscores)
        incident_type = incident_type.lower().replace(" ", "_").replace("-", "_")

        # ── Guard: reject if ANY call is already in progress ──
        if self._call_active:
            log.info("Skipping dispatch — call already active", new_id=incident_id)
            return {
                "status": "already_active",
                "incident_id": incident_id,
                "reason": "call_in_progress",
            }

        # Acquire lock to prevent concurrent dispatches
        if self._call_lock.locked():
            log.info("Skipping dispatch — lock held by another coroutine",
                     new_id=incident_id)
            return {
                "status": "already_active",
                "incident_id": incident_id,
                "reason": "lock_held",
            }

        async with self._call_lock:
            # Double-check after acquiring lock
            if self._call_active:
                log.info("Skipping dispatch — call became active while waiting",
                         new_id=incident_id)
                return {
                    "status": "already_active",
                    "incident_id": incident_id,
                    "reason": "call_in_progress",
                }
            self._call_active = True
            log.info("Call lock acquired", incident_id=incident_id)

        incident = Incident(
            id=incident_id,
            type=incident_type,
            severity=severity,
            location=location,
            description=description,
            camera_id=camera_id,
            timestamp=timestamp,
        )
        self.active_incidents[incident_id] = incident

        log.info("ARIA dispatch initiated", incident_id=incident_id,
                 type=incident_type, severity=severity, location=location)

        await self._broadcast_event("incident_created", incident)

        # 1. Create conference bridge
        bridge = await self.ari.create_bridge(name=f"aria-{incident_id}")
        incident.bridge_id = bridge["id"]

        # 2. Get protocol for this incident type + severity
        protocol = self._get_protocol(incident_type, severity)
        if not protocol:
            log.warning("No protocol found", type=incident_type, severity=severity)
            return {"status": "error", "reason": "no_protocol"}

        # 3. Dispatch first-tier responders concurrently
        first_tier = self._resolve_responders(protocol.get("groups", []))
        await asyncio.gather(*[
            self._call_responder(incident, r)
            for r in first_tier
        ])

        # 4. PA announcement (fire it and forget)
        if protocol.get("pa_announcement"):
            asyncio.create_task(
                self._play_pa_announcement(protocol["pa_announcement"])
            )

        # 5. Schedule escalation
        escalate_after = protocol.get("escalate_after_s", 30)
        escalate_to    = protocol.get("escalate_to", [])
        if escalate_to and escalate_after > 0:
            asyncio.create_task(
                self._schedule_escalation(incident, escalate_after, escalate_to)
            )

        return {
            "status": "dispatched",
            "incident_id": incident_id,
            "bridge_id": bridge["id"],
            "responders_called": [r["id"] for r in first_tier],
        }

    async def update_incident(
        self,
        incident_id: str,
        update_text: str,
    ) -> dict:
        """
        Called by Claude when camera feed reveals new information.
        Whispers the update to all active responders on the bridge.
        """
        incident = self.active_incidents.get(incident_id)
        if not incident:
            return {"status": "error", "reason": "incident_not_found"}

        incident.updates.append(update_text)
        await self._broadcast_event("incident_updated", incident)

        # Whisper update to every responder currently on the bridge
        await asyncio.gather(*[
            self._whisper_to_channel(channel_id, update_text)
            for channel_id in incident.responder_channels.values()
        ])

        log.info("Update whispered to all responders", incident_id=incident_id,
                 active_responders=len(incident.responder_channels))

        return {"status": "ok", "whispered_to": len(incident.responder_channels)}

    async def resolve_incident(self, incident_id: str) -> dict:
        """Mark incident resolved and tear down the bridge."""
        incident = self.active_incidents.get(incident_id)
        if not incident:
            return {"status": "error", "reason": "not_found"}

        incident.resolved = True

        # Stop live scene updates
        if incident._live_update_task:
            incident._live_update_task.cancel()
            incident._live_update_task = None

        # Release global call lock
        self._call_active = False

        await self._broadcast_event("incident_resolved", incident)

        if incident.bridge_id:
            await self.ari.destroy_bridge(incident.bridge_id)

        log.info("Incident resolved, call lock released", incident_id=incident_id)
        return {"status": "resolved"}

    # ──────────────────────────────────────────────────────────────
    # Internal dispatch helpers
    # ──────────────────────────────────────────────────────────────

    async def _call_responder(self, incident: Incident, responder: dict) -> None:
        """
        Originate call to a responder, wait for answer,
        add to bridge, then whisper a situation briefing.
        """
        responder_id = responder["id"]
        log.info("Calling responder", responder=responder["name"],
                 ext=responder["extension"])

        await self._broadcast_event("responder_calling", {
            "incident_id": incident.id, "responder_id": responder_id
        })

        try:
            channel = await self.ari.originate(
                endpoint=responder["extension"],
                caller_id="ARIA Emergency <0000>",
                variables={
                    "ARIA_INCIDENT_ID": incident.id,
                    "ARIA_LOCATION": incident.location,
                },
            )
            channel_id = channel.get("id") or channel.get("channelId", "unknown")

            answered = await self.ari.wait_for_answer(
                channel_id, timeout=responder.get("timeout_s", 20)
            )

            if not answered:
                log.warning("Responder did not answer", responder=responder["name"])
                await self._broadcast_event("responder_no_answer", {
                    "incident_id": incident.id, "responder_id": responder_id
                })
                await self.ari.hangup(channel_id)
                return

            # Add to conference bridge
            await self.ari.add_to_bridge(incident.bridge_id, channel_id)
            incident.responder_channels[responder_id] = channel_id

            await self._broadcast_event("responder_joined", {
                "incident_id": incident.id, "responder_id": responder_id
            })

            # Whisper the situation briefing
            briefing = self._build_briefing(incident, responder["name"])
            await self._whisper_to_channel(channel_id, briefing)

            # Start live scene updates if this is the first responder
            if not incident._live_update_task and self.scene_describer:
                incident._live_update_task = asyncio.create_task(
                    self._live_scene_updates(incident)
                )

        except Exception as e:
            log.error("Failed to call responder", responder=responder_id, error=str(e))

    async def _whisper_to_channel(self, channel_id: str, text: str) -> bool:
        """
        Inject a TTS briefing into the target channel.
        Returns True if succeeded, False if call appears dead (hangup).

        Two paths:
        ① ClawtunnelClient — has whisper_text(): POST /v1/tts directly.
           No local TTS file needed; Azure TTS is handled server-side.
        ② ARIClient — no whisper_text(): generate ElevenLabs TTS file,
           create ARI snoop channel, play into it.
        """
        try:
            # Path ①: clawtunnel direct TTS (preferred)
            if hasattr(self.ari, "whisper_text"):
                return await self.ari.whisper_text(channel_id, text)

            # Path ②: legacy ElevenLabs → ARI snoop
            audio_path = await self._tts_to_file(text)
            snoop = await self.ari.create_whisper_channel(channel_id)
            await self.ari.play_audio(snoop["id"], f"sound:{audio_path}")
            return True

        except Exception as e:
            log.error("Whisper failed", channel=channel_id, error=str(e))
            return False

    async def _tts_to_file(self, text: str) -> str:
        """
        Generate TTS via ElevenLabs and save to Asterisk sounds directory.
        Returns the path string suitable for Asterisk `sound:` URI.
        """
        # Try shared docker volume first, fall back to local temp dir
        sounds_dir = Path(os.getenv("ARIA_SOUNDS_DIR", "/tmp/aria_sounds"))
        sounds_dir.mkdir(parents=True, exist_ok=True)

        filename = f"briefing_{id(text) & 0xFFFFFF}.ulaw"
        out_path = sounds_dir / filename

        with open(out_path, "wb") as f:
            # convert() returns an async generator of audio chunks
            async for chunk in self._tts.text_to_speech.convert(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="ulaw_8000",  # Raw µ-law — Asterisk's native format
            ):
                if chunk:
                    f.write(chunk)

        return f"aria/{filename.removesuffix('.ulaw')}"

    async def _live_scene_updates(self, incident: Incident) -> None:
        """
        Background loop: every 8 seconds, ask Gemini to describe the
        current camera scene, then whisper the description to all
        responders on the active call.

        Detects hangup when TTS fails and auto-resolves the incident.
        Also enforces a max call duration of 90s as a safety net.
        """
        await asyncio.sleep(4)  # short wait for initial briefing to start playing
        update_num = 0
        call_start = time.time()
        max_call_duration = 120  # auto-resolve after 120s
        tts_fail_count: dict[str, int] = {}  # resp_id → consecutive failures
        TTS_FAIL_THRESHOLD = 3  # require 3 consecutive failures to consider hangup

        while not incident.resolved and incident.responder_channels:
            # Safety net: auto-resolve if call has been going too long
            if time.time() - call_start > max_call_duration:
                log.info("Max call duration reached, auto-resolving",
                         incident_id=incident.id)
                await self._auto_resolve(incident, "max_duration")
                break

            try:
                if self.scene_describer and self.latest_frame:
                    update_num += 1
                    description = await self.scene_describer(
                        self.latest_frame, incident
                    )
                    if description and description.strip():
                        update_text = f"Update {update_num}: {description.strip()}"
                        incident.updates.append(update_text)
                        await self._broadcast_event("incident_updated", incident)

                        # Whisper to all active responders, track failures
                        dead_channels = []
                        for resp_id, channel_id in list(incident.responder_channels.items()):
                            ok = await self._whisper_to_channel(channel_id, update_text)
                            if not ok:
                                tts_fail_count[resp_id] = tts_fail_count.get(resp_id, 0) + 1
                                log.warning("TTS failed for responder",
                                            responder=resp_id,
                                            consecutive_failures=tts_fail_count[resp_id])
                                if tts_fail_count[resp_id] >= TTS_FAIL_THRESHOLD:
                                    dead_channels.append(resp_id)
                            else:
                                tts_fail_count[resp_id] = 0  # reset on success

                        # Remove dead channels (responder hung up — confirmed by 3 failures)
                        for resp_id in dead_channels:
                            ch = incident.responder_channels.pop(resp_id, None)
                            log.info("Responder confirmed hung up after 3 TTS failures",
                                     responder=resp_id, channel=ch,
                                     incident_id=incident.id)
                            await self._broadcast_event("responder_hangup", {
                                "incident_id": incident.id,
                                "responder_id": resp_id,
                            })

                        # Auto-resolve if all responders disconnected
                        if not incident.responder_channels:
                            log.info("All responders hung up, auto-resolving",
                                     incident_id=incident.id)
                            await self._auto_resolve(incident, "all_hangup")
                            break

                        log.info("Live scene update whispered",
                                 incident_id=incident.id, update=update_num)
            except Exception as e:
                log.error("Live scene update failed", error=str(e))

            await asyncio.sleep(8)  # wait 8s between updates

        log.info("Live scene updates stopped", incident_id=incident.id)

    async def _auto_resolve(self, incident: Incident, reason: str) -> None:
        """Auto-resolve an incident and clear cooldowns so new detections trigger new calls."""
        incident.resolved = True
        incident._live_update_task = None

        # Remove from active incidents so new detections can fire
        self.active_incidents.pop(incident.id, None)

        # Release global call lock
        self._call_active = False

        await self._broadcast_event("incident_resolved", incident)

        if incident.bridge_id:
            try:
                await self.ari.destroy_bridge(incident.bridge_id)
            except Exception:
                pass

        log.info("Incident auto-resolved, call lock released",
                 incident_id=incident.id, reason=reason)

    async def _play_pa_announcement(self, text: str) -> None:
        """Dial the PA system SIP endpoint and play the announcement."""
        try:
            audio_path = await self._tts_to_file(text)
            channel = await self.ari.originate(
                endpoint="PJSIP/pa-system",
                caller_id="ARIA Emergency <0000>",
            )
            await self.ari.wait_for_answer(channel["id"], timeout=5)
            await self.ari.play_audio(channel["id"], f"sound:{audio_path}")
            await asyncio.sleep(10)
            await self.ari.hangup(channel["id"])
        except Exception as e:
            log.error("PA announcement failed", error=str(e))

    async def _schedule_escalation(
        self,
        incident: Incident,
        delay_s: int,
        escalate_to_ids: list[str],
    ) -> None:
        """Wait `delay_s`, then check if incident still active and escalate."""
        await asyncio.sleep(delay_s)

        if incident.resolved:
            return

        answered_count = len(incident.responder_channels)
        if answered_count > 0:
            log.info("Escalation skipped — responders on scene",
                     incident_id=incident.id, on_scene=answered_count)
            return

        log.warning("Escalating incident — no response",
                    incident_id=incident.id, escalate_to=escalate_to_ids)

        await self._broadcast_event("incident_escalated", {
            "incident_id": incident.id,
            "escalating_to": escalate_to_ids,
        })

        next_responders = self._resolve_responders(escalate_to_ids)
        await asyncio.gather(*[
            self._call_responder(incident, r) for r in next_responders
        ])

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_protocol(self, incident_type: str, severity: str) -> dict | None:
        # Normalize: Gemini may return "MEDICAL FALL" → must be "medical_fall"
        normalized = incident_type.lower().replace(" ", "_").replace("-", "_")
        protocol_def = self.protocols.get(normalized)
        if not protocol_def:
            return None
        return protocol_def["severity_map"].get(severity)

    def _resolve_responders(self, group_or_ids: list[str]) -> list[dict]:
        """Resolve a list of responder IDs or group names to responder dicts."""
        result = []
        seen = set()
        for item in group_or_ids:
            if item in self.groups:
                ids = self.groups[item]
            else:
                ids = [item]
            for rid in ids:
                if rid not in seen and rid in self.responders:
                    result.append(self.responders[rid])
                    seen.add(rid)
        return result

    def _build_briefing(self, incident: Incident, responder_name: str) -> str:
        return (
            f"ARIA alert. {incident.type.replace('_', ' ')} at {incident.location}. "
            f"{incident.severity}. {incident.description} "
            f"Stand by for live updates."
        )

    async def _broadcast_event(self, event_type: str, data: Any) -> None:
        if self._broadcast:
            try:
                # Serialize Incident into the shape the React frontend expects:
                #   responders: string[]  (NOT responder_channels: dict)
                if isinstance(data, Incident):
                    payload = {
                        "id":          data.id,
                        "type":        data.type,
                        "severity":    data.severity,
                        "location":    data.location,
                        "description": data.description,
                        "camera_id":   data.camera_id,
                        "timestamp":   data.timestamp,
                        "resolved":    data.resolved,
                        "responders":  list(data.responder_channels.keys()),
                        "updates":     data.updates,
                    }
                else:
                    import dataclasses
                    payload = dataclasses.asdict(data) if dataclasses.is_dataclass(data) else data
                await self._broadcast({"event": event_type, "data": payload})
            except Exception as e:
                log.warning("broadcast failed", error=str(e))
