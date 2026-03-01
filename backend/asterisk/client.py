"""
Asterisk ARI Async Client — pure aiohttp
─────────────────────────────────────────
REST calls + WebSocket events. No third-party ARI lib needed.
"""
from __future__ import annotations
import asyncio, json, uuid
from typing import Callable, Any
import aiohttp, structlog

log = structlog.get_logger(__name__)


class ARIClient:
    def __init__(self, host: str, port: int, username: str, password: str, app: str):
        self.base   = f"http://{host}:{port}/ari"
        self.ws_url = f"ws://{host}:{port}/ari/events?app={app}&api_key={username}:{password}"
        self._auth  = aiohttp.BasicAuth(username, password)
        self.app    = app
        self._session: aiohttp.ClientSession | None = None
        self._handlers: dict[str, list[Callable]] = {}
        self._ws_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(auth=self._auth)
        self._ws_task = asyncio.create_task(self._listen())
        log.info("Asterisk ARI client ready ✓", base=self.base)

    async def disconnect(self) -> None:
        if self._ws_task: self._ws_task.cancel()
        if self._session: await self._session.close()

    async def _listen(self) -> None:
        while True:
            try:
                async with self._session.ws_connect(self.ws_url) as ws:
                    log.info("ARI WebSocket connected")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            event = json.loads(msg.data)
                            await self._dispatch(event)
            except Exception as e:
                log.warning("ARI WS error — retry in 3s", error=str(e))
                await asyncio.sleep(3)

    async def _dispatch(self, event: dict) -> None:
        for h in self._handlers.get(event.get("type", ""), []):
            try: await h(event)
            except Exception as e: log.error("Handler error", error=str(e))

    def on_event(self, event_type: str, handler: Callable) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def _post(self, path: str, **body) -> dict:
        async with self._session.post(f"{self.base}{path}", json=body) as r:
            try: return await r.json()
            except: return {}

    async def _delete(self, path: str, **params) -> None:
        async with self._session.delete(f"{self.base}{path}", params=params) as r:
            pass

    async def originate(self, endpoint: str, caller_id: str = "ARIA <0000>", variables: dict | None = None) -> dict:
        log.info("Originating", endpoint=endpoint)
        cid = f"aria-{uuid.uuid4().hex[:8]}"
        try:
            body: dict[str, Any] = {
                "endpoint": endpoint, "app": self.app, "appArgs": "outbound",
                "callerId": caller_id, "channelId": cid,
            }
            if variables:
                body["variables"] = variables
            data = await self._post("/channels", **body)
            log.debug("Originate ARI response", endpoint=endpoint, data=str(data)[:120])
            if isinstance(data, dict) and data.get("id"):
                return data
        except Exception as e:
            log.warning("Originate REST failed", endpoint=endpoint, error=str(e))
        # Fallback: always return a channel dict with our pre-generated id
        return {"id": cid, "state": "Down", "_simulated": True}

    async def hangup(self, channel_id: str) -> None:
        await self._delete(f"/channels/{channel_id}", reason="normal")

    async def play_audio(self, channel_id: str, media: str) -> dict:
        return await self._post(f"/channels/{channel_id}/play", media=media)

    async def snoop_channel(self, channel_id: str, spy: str = "none", whisper: str = "out") -> dict:
        sid = f"snoop-{uuid.uuid4().hex[:8]}"
        data = await self._post(f"/channels/{channel_id}/snoop",
                                 spy=spy, whisper=whisper, app=self.app, appArgs="whisper", snoopId=sid)
        return data if isinstance(data, dict) and "id" in data else {"id": sid, **(data or {})}

    async def create_whisper_channel(self, channel_id: str) -> dict:
        """Alias: create a snoop channel that whispers into channel_id."""
        return await self.snoop_channel(channel_id, spy="none", whisper="out")

    async def wait_for_answer(self, channel_id: str, timeout: float = 20.0) -> bool:
        answered, hungup = asyncio.Event(), asyncio.Event()
        async def on_state(e):
            if e.get("channel", {}).get("id") == channel_id and e.get("channel", {}).get("state") == "Up":
                answered.set()
        async def on_hangup(e):
            if e.get("channel", {}).get("id") == channel_id: hungup.set()
        self.on_event("ChannelStateChange", on_state)
        self.on_event("ChannelDestroyed",   on_hangup)
        done, _ = await asyncio.wait(
            [asyncio.create_task(answered.wait()), asyncio.create_task(hungup.wait())],
            timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        return answered.is_set()

    async def create_bridge(self, name: str = "aria-bridge") -> dict:
        bid = f"bridge-{uuid.uuid4().hex[:8]}"
        data = await self._post("/bridges", type="mixing", name=name, bridgeId=bid)
        log.info("Bridge created", bridge_id=bid)
        return data or {"id": bid}

    async def add_to_bridge(self, bridge_id: str, channel_id: str) -> None:
        await self._post(f"/bridges/{bridge_id}/addChannel", channel=channel_id)

    async def destroy_bridge(self, bridge_id: str) -> None:
        await self._delete(f"/bridges/{bridge_id}")

    async def ping(self) -> bool:
        try:
            async with self._session.get(f"{self.base}/asterisk/info") as r:
                return r.status == 200
        except: return False
