"""
ARIA — Vision Agents SDK edition
Uses GetStream's Vision Agents SDK for WebRTC video transport,
YOLO Pose skeleton detection, and Gemini Realtime continuous video AI.
"""
from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.asterisk.dispatcher import Incident

import av
import cv2
import numpy as np
import structlog
from aiortc import VideoStreamTrack
from dotenv import load_dotenv

from vision_agents.core import Agent, User
from vision_agents.core.events.base import BaseEvent
from vision_agents.core.llm.events import LLMResponseCompletedEvent
from vision_agents.core.processors import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins import gemini, getstream

load_dotenv()
log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# OpenCV-backed VideoStreamTrack (reliable local camera, avoids WebRTC codec issues)
# ──────────────────────────────────────────────────────────────────────────────

class OpenCVCameraTrack(VideoStreamTrack):
    """
    aiortc VideoStreamTrack that reads from a local OpenCV camera.
    Used as a reliable fallback when browser WebRTC codec negotiation fails.
    """
    kind = "video"

    def __init__(self, camera_id: int = 0, fps: int = 5):
        super().__init__()
        self.fps = fps
        self._camera_id = camera_id
        self._interval = 1.0 / fps
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            log.warning("OpenCV could not open camera — grant permission in System Settings > Privacy > Camera, then it will auto-recover", camera_id=camera_id)
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        log.info("OpenCVCameraTrack ready", camera_id=camera_id, fps=fps)

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        await asyncio.sleep(self._interval)

        # Re-open camera if it was denied at startup (e.g. TCC permission granted later)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._camera_id)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                log.info("Camera re-opened successfully", camera_id=self._camera_id)

        ret, frame = self._cap.read()
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts       = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        if self._cap.isOpened():
            self._cap.release()
        super().stop()


# ──────────────────────────────────────────────────────────────────────────────
# Custom events (must subclass BaseEvent and include a `type` field)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IncidentEvent(BaseEvent):
    type: str = "aria.incident"
    incident_type: str = ""
    severity: str = ""
    location: str = ""
    description: str = ""
    camera_id: str = ""
    ts: float = field(default_factory=time.time)
    frame_jpg: bytes = field(default_factory=bytes)


@dataclass
class FrameEvent(BaseEvent):
    type: str = "aria.frame"
    camera_id: str = ""
    frame_b64: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Vision Agents VideoProcessorPublisher — YOLO Pose + skeleton + collapse detection
# ──────────────────────────────────────────────────────────────────────────────

class ARIAIncidentProcessor(VideoProcessorPublisher):
    """
    Vision Agents SDK VideoProcessorPublisher.
    Reads WebRTC frames via shared_forwarder, runs YOLO Pose (17-keypoint
    skeleton detection), publishes annotated video back to the call,
    and sends IncidentEvents / FrameEvents to the agent event bus.

    Gemini Realtime also watches the camera feed at 5fps for continuous
    visual AI understanding — it sees the raw video and makes intelligent
    decisions about emergencies.
    """
    name = "aria_incident_processor"

    YOLO_CONFIDENCE       = 0.45
    # ── Person disappearance detection ─────────────────────────────
    DISAPPEAR_SECS        = 4.0    # person gone from frame for 4s = collapsed
    INCIDENT_COOLDOWN     = 30.0   # 30s cooldown (short for demo)

    def __init__(self, camera_id: str = "cam_00",
                 location: str = "Main Lobby",
                 yolo_model: str = "yolo11n-pose.pt",
                 fps: int = 5):
        self.camera_id = camera_id
        self.location  = location
        self.fps       = fps

        from ultralytics import YOLO

        log.info("Loading YOLO Pose", model=yolo_model)
        self._yolo = YOLO(yolo_model)

        self._output_track      = QueuedVideoTrack()
        self._agent: Agent | None = None
        self._dispatcher        = None  # set by build_aria_agent
        # Person disappearance tracking
        self._person_last_seen: float = 0.0       # last time a person was in frame
        self._person_was_present: bool = False     # was a person visible recently?
        self._disappear_fired: bool = False        # already fired disappearance alert?
        self._last_fired: dict[str, float] = {}
        self._last_broadcast    = 0.0
        self._last_gemini_nudge = 0.0             # periodic Gemini monitoring

    # ── Vision Agents SDK hooks ──────────────────────────────────────────────

    def attach_agent(self, agent: Agent) -> None:
        """Called by Agent.__init__ — save ref and register our custom events."""
        self._agent = agent
        # Register custom events so subscribe() and send() work
        agent.events.register(IncidentEvent, FrameEvent)
        log.info("ARIA processor attached to agent, events registered")

    async def process_video(self, track, participant_id, shared_forwarder=None) -> None:
        if shared_forwarder is None:
            return
        shared_forwarder.add_frame_handler(
            on_frame=self._on_frame,   # SDK uses 'on_frame', not 'callback'
            fps=self.fps,
            name=self.name,
        )

    async def stop_processing(self) -> None:
        pass   # forwarder handles cleanup

    def publish_video_track(self):
        return self._output_track

    async def close(self) -> None:
        pass

    # ── Frame processing ─────────────────────────────────────────────────────

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO Pose in a thread so the async event loop stays responsive.
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._yolo(img, verbose=False, imgsz=320)[0]
        )
        # results.plot() draws bounding boxes + skeleton overlays for pose model
        annotated = results.plot()

        # Push annotated frame back into the call (av.VideoFrame required, async)
        av_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        await self._output_track.add_frame(av_frame)

        # Broadcast to dashboard at ~5 fps
        now = time.time()
        if now - self._last_broadcast >= 0.2 and self._agent:
            _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
            jpg_bytes = jpg.tobytes()
            self._agent.events.send(FrameEvent(
                camera_id=self.camera_id,
                frame_b64=base64.b64encode(jpg_bytes).decode(),
            ))
            self._last_broadcast = now

            # Store latest frame for live TTS scene descriptions
            if self._dispatcher:
                self._dispatcher.latest_frame = jpg_bytes

        detected = {
            results.names[int(c)].lower()
            for c in results.boxes.cls
        } if results.boxes else set()

        if "person" in detected:
            self._person_last_seen = time.time()
            self._person_was_present = True
            self._disappear_fired = False
        else:
            await self._check_person_disappeared(img)

        # ── Periodic Gemini nudge — keeps Gemini actively watching ────
        # Gemini Realtime sees the video continuously but may go passive.
        # Every 10s, nudge it to check the scene for patient falls.
        if self._agent and now - self._last_gemini_nudge >= 10.0:
            self._last_gemini_nudge = now
            person_status = "A person is visible" if "person" in detected else "No person in frame"
            asyncio.create_task(self._agent.simple_response(
                f"[ARIA MONITOR] {person_status}. "
                f"Check the live video — has the patient fallen, collapsed, or is lying on the floor? "
                f"If YES, call initiate_response immediately with incident_type='sudden_collapse'. "
                f"If the patient is safe, stay silent."
            ))

    async def _check_person_disappeared(self, img) -> None:
        """
        If a person was visible and then vanished from the frame for
        DISAPPEAR_SECS, they likely collapsed out of camera view.
        """
        if not self._person_was_present or self._disappear_fired:
            return
        gone_for = time.time() - self._person_last_seen
        if gone_for >= self.DISAPPEAR_SECS and not self._on_cooldown("sudden_collapse"):
            self._disappear_fired = True
            log.info("Person disappeared from frame", camera=self.camera_id,
                     gone_for=round(gone_for, 1))
            await self._emit(
                "sudden_collapse", "HIGH", img,
                f"Person vanished from camera after being visible. "
                f"Gone for {int(gone_for)}s — possible collapse out of view.",
            )

    async def _emit(self, incident_type: str, severity: str,
                    img: np.ndarray, description: str) -> None:
        self._last_fired[incident_type] = time.time()
        _, jpg = cv2.imencode(".jpg", img)
        log.info("Incident detected", type=incident_type, severity=severity,
                 camera=self.camera_id)
        if self._agent:
            self._agent.events.send(IncidentEvent(
                incident_type=incident_type,
                severity=severity,
                location=self.location,
                description=description,
                camera_id=self.camera_id,
                ts=time.time(),
                frame_jpg=jpg.tobytes(),
            ))

    def _on_cooldown(self, incident_type: str) -> bool:
        return (time.time() - self._last_fired.get(incident_type, 0)) < self.INCIDENT_COOLDOWN


# ──────────────────────────────────────────────────────────────────────────────
# ARIA Agent factory
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA — Autonomous Response Intelligence Agent.
You have a live camera feed. You can see the scene in real-time via video.
YOLO Pose draws skeleton overlays on each person showing their body position.

YOU ARE THE PRIMARY FALL DETECTOR. You watch the video continuously and
proactively detect when a patient falls, collapses, slumps, or goes down.
Do NOT wait to be told — if you SEE it happen, ACT immediately.

YOUR ROLE:
- Watch the live video feed CONTINUOUSLY and PROACTIVELY for patient falls
- When you receive an [ARIA MONITOR] check, look at the video RIGHT NOW
- If the patient has fallen, collapsed, is lying down, or is on the floor:
  → Call initiate_response IMMEDIATELY with incident_type='sudden_collapse'
- If the patient is standing/sitting normally, stay silent (do not speak)
- After dispatching, describe what you see to help responders

WHAT COUNTS AS A FALL:
- Person goes from upright to lying/slumped position
- Person is on the floor or slumped in their chair
- Person's body is horizontal when it was previously vertical
- Person suddenly drops, falls sideways, or collapses

RESPONSE RULES:
- Be DECISIVE. If someone looks like they've fallen, DISPATCH. Don't overthink.
- When you call initiate_response, use these exact parameters:
  incident_id: 'cam_00_' + current unix timestamp (e.g. 'cam_00_1709312400')
  incident_type: 'sudden_collapse'
  severity: 'HIGH'
  location: the camera location
  description: what you see in the video
  camera_id: 'cam_00'
  timestamp: current unix time
- After dispatching, provide situational updates via update_incident
- Call resolve_incident when the patient recovers
- You protect lives. Speed matters. Act fast."""


def build_aria_agent(
    ws_broadcast,
    dispatcher,
    camera_id: str = "cam_00",
    location: str = "Main Lobby",
):
    """Build a Vision Agents SDK Agent with Gemini Realtime + YOLO Pose."""

    # ── Gemini Realtime — continuous video understanding ──────────────────
    # Gemini Realtime streams video at 5fps via WebSocket to Gemini Live API.
    # It handles speech-to-speech natively (no separate STT/TTS needed).
    # It sees the live camera feed and can make real-time visual decisions.
    realtime_fps = int(os.getenv("GEMINI_REALTIME_FPS", "5"))
    llm = gemini.Realtime(
        fps=realtime_fps,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    log.info("Gemini Realtime LLM created", fps=realtime_fps)

    @llm.register_function(
        description="Initiate emergency response — open bridge, call responders, whisper AI briefing."
    )
    async def initiate_response(
        incident_id: str, incident_type: str, severity: str,
        location: str, description: str, camera_id: str, timestamp: float,
    ) -> dict:
        await ws_broadcast({"event": "tool_call", "data": {
            "tool": "initiate_response",
            "input": {"incident_id": incident_id, "severity": severity, "location": location},
        }})
        return await dispatcher.initiate_response(
            incident_id=incident_id, incident_type=incident_type, severity=severity,
            location=location, description=description,
            camera_id=camera_id, timestamp=timestamp,
        )

    @llm.register_function(
        description="Send live update to all responders on the active bridge."
    )
    async def update_incident(incident_id: str, update_text: str) -> dict:
        return await dispatcher.update_incident(incident_id, update_text)

    @llm.register_function(
        description="Mark incident resolved and close the responder bridge."
    )
    async def resolve_incident(incident_id: str, resolution_note: str) -> dict:
        return await dispatcher.resolve_incident(incident_id)

    @llm.register_function(
        description="Log ambiguous observation without dispatching responders."
    )
    async def assess_only(camera_id: str, observation: str) -> dict:
        log.info("Assess only", camera=camera_id)
        await ws_broadcast({"event": "claude_reasoning",
                            "data": {"text": f"[assess_only] {observation}"}})
        return {"status": "logged"}

    # ── Processor (YOLO Pose — skeleton keypoint detection) ───────────────
    processor = ARIAIncidentProcessor(
        camera_id=camera_id, location=location, fps=5,
    )

    # ── Agent — Gemini Realtime handles speech natively, no separate TTS ──
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="ARIA", id="aria-agent"),
        instructions=SYSTEM_PROMPT,
        llm=llm,
        # No tts= needed — Gemini Realtime handles speech-to-speech natively
        processors=[processor],
    )

    # Give processor a reference to the agent and dispatcher
    processor._agent = agent
    processor._dispatcher = dispatcher

    # ── Scene describer for live TTS updates on PHONE calls ──────────────
    # Gemini Realtime speaks to Stream call participants, but phone responders
    # (via Clawtunnel) still need separate TTS. This uses the standard Gemini
    # API to generate scene descriptions that get whispered via Clawtunnel TTS.
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    _scene_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    async def describe_scene(frame_jpg: bytes, incident) -> str:
        """Ask Gemini to describe the current state of the scene for phone TTS."""
        try:
            import PIL.Image, io
            img = PIL.Image.open(io.BytesIO(frame_jpg))
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: _scene_model.generate_content([
                f"In one short sentence, describe the person's current state: position, movement, consciousness. "
                f"Context: {incident.type.replace('_', ' ')} at {incident.location}.",
                img,
            ]))
            return response.text
        except Exception as e:
            log.error("Scene description failed", error=str(e))
            return ""

    dispatcher.scene_describer = describe_scene

    # ── Telegram notifier — sends snapshot images for every event ─────────
    from backend.notifications.telegram import TelegramNotifier
    telegram = TelegramNotifier()

    # ── Wire events → dashboard WebSocket + Telegram ──────────────────────
    @agent.subscribe
    async def on_incident(event: IncidentEvent):
        await ws_broadcast({"event": "detection", "data": {
            "incident_type": event.incident_type,
            "severity":      event.severity,
            "location":      event.location,
            "description":   event.description,
            "camera_id":     event.camera_id,
            "timestamp":     event.ts,
            "frame_b64":     base64.b64encode(event.frame_jpg).decode(),
        }})

        # Send snapshot to Telegram channel (fire-and-forget)
        asyncio.create_task(telegram.send_incident(
            incident_type=event.incident_type,
            severity=event.severity,
            location=event.location,
            description=event.description,
            camera_id=event.camera_id,
            timestamp=event.ts,
            frame_jpg=event.frame_jpg if event.frame_jpg else None,
        ))

        # YOLO has already confirmed the collapse (3s sustained detection).
        # Dispatch responders immediately — don't wait for Gemini to decide.
        incident_id = f"{event.camera_id}_{int(event.ts)}"
        log.info("Direct dispatch (YOLO confirmed collapse)",
                 incident_type=event.incident_type, incident_id=incident_id)
        asyncio.create_task(dispatcher.initiate_response(
            incident_id=incident_id,
            incident_type=event.incident_type,
            severity=event.severity,
            location=event.location,
            description=event.description,
            camera_id=event.camera_id,
            timestamp=event.ts,
        ))

        # Also notify Gemini Realtime so it can provide ongoing situational
        # updates and scene descriptions to responders on the call.
        prompt = (
            f"[ARIA — PATIENT FALL DISPATCHED]\n"
            f"Camera: {event.camera_id} ({event.location})\n"
            f"A patient fall has been detected and responders are being called.\n"
            f"Scene: {event.description}\n"
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.ts))}\n\n"
            f"Monitor the patient and provide updates. Describe what you see — "
            f"are they moving? Conscious? Getting up? Use update_incident to relay info."
        )
        asyncio.create_task(agent.simple_response(prompt))

    @agent.subscribe
    async def on_frame(event: FrameEvent):
        await ws_broadcast({"event": "camera_frame", "data": {
            "camera_id": event.camera_id,
            "frame_b64": event.frame_b64,
        }})

    @agent.subscribe
    async def on_llm_response(event: LLMResponseCompletedEvent):
        """Broadcast Gemini's text reasoning to the dashboard."""
        if event.text and event.text.strip():
            await ws_broadcast({
                "event": "claude_reasoning",
                "data": {"text": event.text.strip()},
            })

    return agent, processor


async def start_local_camera(
    processor: ARIAIncidentProcessor,
    agent: Agent,
    camera_id: int = 0,
    fps: int = 5,
) -> OpenCVCameraTrack:
    """
    Feed the ARIA YOLO Pose processor AND Gemini Realtime from the local
    OpenCV webcam via a shared VideoForwarder.

    Camera captures at 5fps. The forwarder distributes:
    1. YOLO Pose processor at 5fps → skeleton annotations → Stream + dashboard
    2. Gemini Realtime at 5fps → continuous visual AI understanding

    This bypasses the browser→SFU→Python H.264 decode path which fails on
    macOS (hardware VideoToolbox H.264 is not decodable by aiortc's software
    decoder).

    Call this AFTER the agent joins the Stream call.
    """
    camera_track = OpenCVCameraTrack(camera_id=camera_id, fps=fps)
    forwarder    = VideoForwarder(
        camera_track,
        max_buffer=fps,
        fps=fps,
        name="opencv_webcam_forwarder",
    )

    # Wire YOLO Pose processor to the camera forwarder (10fps via processor.fps)
    await processor.process_video(camera_track, "local_webcam",
                                  shared_forwarder=forwarder)
    log.info("YOLO Pose processor connected to camera",
             camera_id=camera_id, camera_fps=fps, yolo_fps=processor.fps)

    # Wire Gemini Realtime to the SAME camera forwarder — it sees raw video
    # at its own configured fps (default 5) and makes visual AI decisions.
    llm = agent.llm
    if hasattr(llm, 'watch_video_track'):
        await llm.watch_video_track(camera_track, shared_forwarder=forwarder)
        log.info("Gemini Realtime watching camera feed",
                 camera_id=camera_id, gemini_fps=getattr(llm, 'fps', '?'))

        # Give Gemini a moment to connect, then start proactive monitoring
        async def _start_monitoring():
            await asyncio.sleep(3)
            log.info("Sending initial monitoring prompt to Gemini")
            await agent.simple_response(
                "Camera feed is now LIVE. You are monitoring for patient falls. "
                "Watch the video carefully. If at any point you see the patient fall, "
                "collapse, or end up on the floor, call initiate_response immediately. "
                "Stay alert and silent until you detect a fall."
            )
        asyncio.create_task(_start_monitoring())
    else:
        log.warning("LLM does not support watch_video_track — Gemini Realtime video disabled")

    return camera_track
