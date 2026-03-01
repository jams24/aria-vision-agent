"""
ARIA — Main Entry Point  (Vision Agents SDK edition)
──────────────────────────────────────────────────────
1. Builds ARIDispatcher (Asterisk ARI phone calls)
2. Builds Vision Agents Agent (GetStream WebRTC + YOLO + Gemini)
3. Starts the agent on a Stream call — React frontend joins with webcam
4. Serves FastAPI WebSocket dashboard + REST endpoints
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import yaml
import uvicorn

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.asterisk.clawtunnel_client import ClawtunnelClient
from backend.asterisk.dispatcher import ARIDispatcher
from backend.agent.vision_aria import build_aria_agent, IncidentEvent, start_local_camera

load_dotenv()
log = structlog.get_logger(__name__)

STREAM_CALL_TYPE = "default"
STREAM_CALL_ID   = "aria-emergency-room"

# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection manager
# ──────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        log.info("Dashboard client connected", total=len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        log.info("Dashboard client disconnected", total=len(self._active))

    async def broadcast(self, data: dict) -> None:
        if not self._active:
            return
        message = json.dumps(data, default=str)
        dead = set()
        for ws in self._active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._active -= dead


ws_manager  = ConnectionManager()
aria_agent  = None   # Vision Agents Agent
dispatcher  = None   # ARIDispatcher
ari_client  = None   # ClawtunnelClient (needed by /ari-callback)


# ──────────────────────────────────────────────────────────────────────────────
# App lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aria_agent, dispatcher, ari_client

    log.info("╔══════════════════════════════════╗")
    log.info("║   ARIA — Booting up...           ║")
    log.info("╚══════════════════════════════════╝")

    # Load YAML configs
    responders_cfg = yaml.safe_load(
        Path(os.getenv("RESPONDERS_CONFIG", "config/responders.yaml")).read_text()
    )
    protocols_cfg = yaml.safe_load(
        Path(os.getenv("PROTOCOLS_CONFIG", "config/protocols.yaml")).read_text()
    )

    # ── Telephony client: voice.clawtunnel.com ────────────────────────────────
    # Uses the existing Node.js ARI platform (ariclient/) via its REST API.
    # ZIP extensions (5-digit internal extensions) are used for responder calls.
    # The Node.js service handles the actual SIP leg — we never touch it.
    dashboard_port = int(os.getenv("DASHBOARD_WS_PORT", "8000"))
    callback_url = os.getenv(
        "CLAWTUNNEL_CALLBACK_URL",
        f"http://localhost:{dashboard_port}/ari-callback",
    )
    ari_client = ClawtunnelClient(
        base_url       = os.getenv("CLAWTUNNEL_BASE_URL", "https://voice.clawtunnel.com"),
        api_key        = os.getenv("CLAWTUNNEL_API_KEY", ""),
        aria_extension = os.getenv("CLAWTUNNEL_ARIA_EXTENSION", "10000"),
        voice          = os.getenv("CLAWTUNNEL_VOICE", "en-US-JennyNeural"),
        callback_url   = callback_url,
    )
    await ari_client.connect()

    # Build dispatcher
    dispatcher = ARIDispatcher(
        ari=ari_client,
        responders_config=responders_cfg,
        protocols_config=protocols_cfg,
        elevenlabs_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        ws_broadcaster=ws_manager.broadcast,
    )

    # Build Vision Agents SDK agent
    location = os.getenv("CAMERA_LOCATIONS", "Main Lobby").split(",")[0].strip()
    aria_agent, processor = build_aria_agent(
        ws_broadcast=ws_manager.broadcast,
        dispatcher=dispatcher,
        camera_id="cam_00",
        location=location,
    )

    # Register the agent user on Stream, create/join the call
    await aria_agent.create_user()
    call = await aria_agent.create_call(STREAM_CALL_TYPE, STREAM_CALL_ID)
    log.info("Stream call ready", call_type=STREAM_CALL_TYPE, call_id=STREAM_CALL_ID)

    # Camera source (0 = default webcam, or path to video file)
    camera_src_str = os.getenv("CAMERA_SOURCES", "0").split(",")[0].strip()
    try:
        camera_src = int(camera_src_str)
    except ValueError:
        camera_src = 0

    # Run the agent. The agent joins the Stream call so the browser operator
    # can also join and see ARIA's annotated video published back.
    # Local OpenCV webcam feeds BOTH the YOLO Pose processor AND Gemini Realtime.
    # This bypasses the browser→SFU→Python H.264 decode path which fails on macOS
    # (hardware VideoToolbox H.264 is not decodable by aiortc's software decoder).
    async def _run_agent():
        try:
            async with aria_agent.join(call):
                log.info("ARIA joined Stream call ✓ — starting local webcam + Gemini Realtime")
                await start_local_camera(processor, agent=aria_agent,
                                         camera_id=camera_src, fps=5)
                await aria_agent.finish()
        except Exception as e:
            log.error("Vision Agents agent error", error=str(e))

    agent_task = asyncio.create_task(_run_agent())

    log.info("ARIA is operational ✓")

    await ws_manager.broadcast({
        "event": "system_ready",
        "data": {
            "cameras": [{"id": "cam_00", "location": location}],
            "call_id": STREAM_CALL_ID,
            "call_type": STREAM_CALL_TYPE,
            "timestamp": time.time(),
        },
    })

    yield  # ── app running ──

    agent_task.cancel()
    try:
        await agent_task
    except asyncio.CancelledError:
        pass
    await ari_client.disconnect()
    log.info("ARIA shut down cleanly.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ARIA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Clawtunnel call-status callback ───────────────────────────────────────────
# The Node.js ARI service (voice.clawtunnel.com) POSTs call events here.
# Configure CLAWTUNNEL_CALLBACK_URL to point to a publicly reachable URL
# (e.g. ngrok tunnel) so clawtunnel can reach this endpoint.
@app.post("/ari-callback")
async def ari_callback(request: Request):
    """
    Receive call status events from voice.clawtunnel.com.
    Expected JSON shape (flexible — handles common field names):
      { "uuid": "...", "event": "answered"|"hangup"|"no-answer"|"failed" }
    """
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid JSON"}

    uuid  = data.get("uuid") or data.get("callid") or data.get("call_id") or ""
    event = (
        data.get("event") or data.get("status") or
        data.get("call_status") or data.get("type") or ""
    )

    log.info("ARI callback received", uuid=uuid, event=event)

    if uuid and event and ari_client:
        ari_client.on_call_event(uuid, str(event))

    return {"ok": True}


# ── Stream token endpoint — React frontend calls this to join the call ─────────
@app.get("/stream-token")
async def stream_token(user_id: str = "operator"):
    """Generate a Stream user token for the React Video SDK client."""
    from getstream import Stream
    client = Stream(
        api_key=os.getenv("STREAM_API_KEY", ""),
        api_secret=os.getenv("STREAM_API_SECRET", ""),
    )
    token = client.create_token(user_id)
    return {
        "token": token,
        "api_key": os.getenv("STREAM_API_KEY", ""),
        "user_id": user_id,
        "call_type": STREAM_CALL_TYPE,
        "call_id": STREAM_CALL_ID,
    }


# ── WebSocket dashboard ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)

    try:
        location = os.getenv("CAMERA_LOCATIONS", "Main Lobby").split(",")[0].strip()
        await ws.send_text(json.dumps({
            "event": "system_ready",
            "data": {
                "cameras": [{"id": "cam_00", "location": location}],
                "call_id": STREAM_CALL_ID,
                "call_type": STREAM_CALL_TYPE,
                "timestamp": time.time(),
            },
        }))
        # Replay active incidents
        if dispatcher:
            for inc in list(dispatcher.active_incidents.values()):
                if inc.resolved:
                    continue
                await ws.send_text(json.dumps({
                    "event": "incident_created",
                    "data": {
                        "id": inc.id, "type": inc.type, "severity": inc.severity,
                        "location": inc.location, "description": inc.description,
                        "camera_id": inc.camera_id, "timestamp": inc.timestamp,
                        "resolved": inc.resolved,
                        "responders": list(inc.responder_channels.keys()),
                        "updates": inc.updates,
                    },
                }))
    except Exception:
        ws_manager.disconnect(ws)
        return

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            action = msg.get("action")

            if action == "resolve_incident" and dispatcher:
                await dispatcher.resolve_incident(msg.get("incident_id"))

            elif action == "simulate" and aria_agent:
                # Emit a synthetic IncidentEvent into the Vision Agents event bus
                import cv2, numpy as np
                fake = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(fake, "SIMULATED", (150, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                _, jpg = cv2.imencode(".jpg", fake)
                descs = {
                    "sudden_collapse":      "Person collapsed suddenly, on floor unresponsive.",
                    "medical_fall":         "Person collapsed suddenly, on floor unresponsive.",
                }
                itype = msg.get("incident_type", "sudden_collapse")
                # Frontend sends "medical_fall" but backend uses "sudden_collapse"
                if itype == "medical_fall":
                    itype = "sudden_collapse"
                aria_agent.events.send(IncidentEvent(
                    incident_type=itype,
                    severity=msg.get("severity", "HIGH"),
                    location=msg.get("location", "Demo Room"),
                    description=descs.get(itype, "Incident detected."),
                    camera_id=msg.get("camera_id", "cam_00"),
                    ts=time.time(),
                    frame_jpg=jpg.tobytes(),
                ))

    except (WebSocketDisconnect, Exception):
        ws_manager.disconnect(ws)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ARIA", "sdk": "vision-agents"}


@app.get("/incidents")
async def get_incidents():
    if not dispatcher:
        return []
    return [
        {
            "id": inc.id, "type": inc.type, "severity": inc.severity,
            "location": inc.location, "description": inc.description,
            "camera_id": inc.camera_id, "timestamp": inc.timestamp,
            "resolved": inc.resolved,
            "responders": list(inc.responder_channels.keys()),
            "updates": inc.updates,
        }
        for inc in dispatcher.active_incidents.values()
    ]


# Serve React build
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


def run():
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DASHBOARD_WS_PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
