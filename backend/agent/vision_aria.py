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
from collections import deque
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
    # ── Sudden-collapse detection (motion-tracking based) ──────────
    COLLAPSE_DROP_PCT     = 0.10   # center-Y jumps down by ≥10% of frame height
    COLLAPSE_SHRINK_PCT   = 0.25   # bbox height shrinks by ≥25%
    COLLAPSE_TIME_WINDOW  = 3.0    # change must happen within 3 seconds
    COLLAPSE_CONFIRM_SECS = 3.0    # stay "collapsed" for 3s before alerting
    # ── Person disappearance detection ─────────────────────────────
    DISAPPEAR_SECS        = 4.0    # person gone from frame for 4s = collapsed
    INCIDENT_COOLDOWN     = 120.0  # 2-min cooldown between same-type incidents

    # COCO keypoint indices (used for collapse pose detection)
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16

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
        # Collapse motion tracking: deque of (timestamp, center_y_norm, bbox_h_norm)
        self._person_history: deque = deque(maxlen=30)   # ~6s at 5 fps
        self._collapse_since: float | None = None
        # Person disappearance tracking
        self._person_last_seen: float = 0.0       # last time a person was in frame
        self._person_was_present: bool = False     # was a person visible recently?
        self._disappear_fired: bool = False        # already fired disappearance alert?
        self._last_fired: dict[str, float] = {}
        self._last_broadcast    = 0.0

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
            await self._check_sudden_collapse(img, results)
        else:
            await self._check_person_disappeared(img)

    async def _check_sudden_collapse(self, img, results) -> None:
        """
        Hybrid collapse detection using BOTH bbox motion tracking AND
        YOLO Pose keypoints when available.

        Stage 1 (bbox): Tracks center-Y and height across frames.
                         Triggers on rapid downward motion.
        Stage 2 (pose): If keypoints available, checks if the person's
                         skeleton is horizontal (shoulders ≈ hips ≈ ankles Y).
        """
        frame_h = img.shape[0]

        # Find the person box with highest confidence
        best_conf, best_cy, best_bh = 0.0, 0.0, 0.0
        best_idx = -1
        for i, box in enumerate(results.boxes):
            if results.names[int(box.cls)].lower() != "person":
                continue
            conf = float(box.conf)
            if conf > best_conf:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                best_conf = conf
                best_cy   = (y1 + y2) / 2.0 / frame_h   # normalised 0-1
                best_bh   = (y2 - y1) / frame_h          # normalised 0-1
                best_idx  = i

        if best_idx < 0:
            return   # no person in frame — nothing to track

        now = time.time()
        self._person_history.append((now, best_cy, best_bh))

        # ── Stage 1: bbox motion tracking ────────────────────────────────
        collapsed_bbox = False
        for ts, old_cy, old_bh in self._person_history:
            age = now - ts
            if age < 0.5 or age > self.COLLAPSE_TIME_WINDOW:
                continue

            cy_drop = best_cy - old_cy
            if cy_drop >= self.COLLAPSE_DROP_PCT:
                collapsed_bbox = True
                break

            if old_bh > 0.10:
                bh_shrink = (old_bh - best_bh) / old_bh
                if bh_shrink >= self.COLLAPSE_SHRINK_PCT:
                    collapsed_bbox = True
                    break

        # ── Stage 2: pose keypoint check (if available) ──────────────────
        collapsed_pose = False
        if results.keypoints is not None and len(results.keypoints) > best_idx:
            kpts = results.keypoints[best_idx]
            if kpts is not None and kpts.xyn is not None and len(kpts.xyn) > 0:
                kp = kpts.xyn[0]   # normalised keypoints [17, 2]
                if len(kp) >= 17:
                    # Get Y positions of key body parts (normalised 0-1)
                    shoulder_y = float((kp[self.L_SHOULDER][1] + kp[self.R_SHOULDER][1]) / 2)
                    hip_y      = float((kp[self.L_HIP][1] + kp[self.R_HIP][1]) / 2)
                    ankle_y    = float((kp[self.L_ANKLE][1] + kp[self.R_ANKLE][1]) / 2)

                    # If all visible (non-zero) and vertical span is tiny = horizontal person
                    if shoulder_y > 0.01 and hip_y > 0.01 and ankle_y > 0.01:
                        vertical_span = abs(ankle_y - shoulder_y)
                        if vertical_span < 0.15:  # skeleton is very flat / horizontal
                            collapsed_pose = True

        # Either trigger is enough (bbox motion OR skeleton horizontal)
        collapsed = collapsed_bbox or collapsed_pose

        if collapsed:
            if self._collapse_since is None:
                self._collapse_since = now
                self._collapse_snapshot = img.copy()  # capture frame at moment of collapse
                trigger = "pose" if collapsed_pose else "bbox"
                log.info("Sudden collapse detected", camera=self.camera_id,
                         trigger=trigger, cy=round(best_cy, 2), bh=round(best_bh, 2))
            elapsed = now - self._collapse_since
            if elapsed >= self.COLLAPSE_CONFIRM_SECS and not self._on_cooldown("sudden_collapse"):
                severity = "CRITICAL" if elapsed > 15 else "HIGH"
                snap = getattr(self, '_collapse_snapshot', img)
                self._collapse_snapshot = None
                await self._emit(
                    "sudden_collapse", severity, snap,
                    f"Person collapsed suddenly, down for {int(elapsed)}s. Immediate assistance needed.",
                )
        else:
            self._collapse_since = None

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

YOUR ROLE:
- Watch the live video feed continuously for patient falls and collapses
- When you receive a [VISION ALERT], look at the video to confirm or reject it
- If you see a genuine fall or collapse, call initiate_response immediately
- Provide clear, calm spoken updates to the Stream call participants
- Keep monitoring and call update_incident as the situation evolves
- Call resolve_incident when the patient is safe / situation resolved

DETECTABLE EMERGENCY:
PATIENT FALL / SUDDEN COLLAPSE — Person falls down, goes limp, collapses, or disappears from view.
YOLO Pose tracks their skeleton and detects rapid downward motion, horizontal body position,
or sudden disappearance from frame.

RESPONSE RULES:
- Be decisive. If someone has collapsed or fallen, DISPATCH immediately.
- Describe WHAT you see: are they on the floor? Moving? Conscious?
- Use the video to provide real descriptions, not generic text.
- Speak concisely — responders need fast, clear information.
- If the alert looks like a false positive (person just sitting/bending/stretching), use assess_only.
- You protect lives. Act fast on genuine emergencies."""


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

        # Send text alert to Gemini Realtime session — it already sees the
        # video, so this gives it structured context to act on immediately.
        # Gemini visually confirms or rejects the detection before dispatching.
        prompt = (
            f"[ARIA VISION ALERT]\n"
            f"Camera: {event.camera_id} ({event.location})\n"
            f"Detected: PATIENT FALL / SUDDEN COLLAPSE\n"
            f"Incident Type Key: {event.incident_type}\n"
            f"CV Severity: {event.severity}\n"
            f"Scene: {event.description}\n"
            f"Time: {time.strftime('%H:%M:%S', time.localtime(event.ts))}\n\n"
            f"Incident ID: {event.camera_id}_{int(event.ts)}\n"
            f"Use incident_type='{event.incident_type}' when calling initiate_response.\n"
            f"Look at the live video feed to confirm the patient has fallen, then assess and respond."
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
    else:
        log.warning("LLM does not support watch_video_track — Gemini Realtime video disabled")

    return camera_track
