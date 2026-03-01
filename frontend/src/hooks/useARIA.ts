import { useState, useEffect, useCallback, useRef } from "react";
import type { ARIAEvent, Incident, Responder, Camera, Detection } from "../types/incident";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";

export function useARIA() {
  const [connected, setConnected] = useState(false);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [incidents, setIncidents] = useState<Map<string, Incident>>(new Map());
  const [responders, setResponders] = useState<Map<string, Responder>>(new Map());
  const [recentDetections, setRecentDetections] = useState<Detection[]>([]);
  const [cameraFrames, setCameraFrames] = useState<Map<string, string>>(new Map()); // camera_id → frame_b64
  const [claudeLog, setClaudeLog] = useState<string[]>([]);
  const [toolLog, setToolLog] = useState<{ tool: string; input: unknown; ts: number }[]>([]);

  const ws = useRef<WebSocket | null>(null);

  const handleEvent = useCallback((ev: ARIAEvent) => {
    switch (ev.event) {
      case "system_ready":
        setCameras(ev.data.cameras);
        break;

      case "detection":
        setRecentDetections((prev) => [ev.data, ...prev].slice(0, 20));
        // Also store the frame thumbnail for live camera view
        if (ev.data.frame_b64) {
          setCameraFrames((prev) => new Map(prev).set(ev.data.camera_id, ev.data.frame_b64));
        }
        break;

      case "camera_frame":
        // Live frame streamed from camera (real or synthetic)
        if (ev.data.frame_b64) {
          setCameraFrames((prev) => new Map(prev).set(ev.data.camera_id, ev.data.frame_b64));
        }
        break;

      case "incident_created":
      case "incident_updated":
        setIncidents((prev) => {
          const next = new Map(prev);
          next.set(ev.data.id, ev.data as unknown as Incident);
          return next;
        });
        break;

      case "incident_resolved":
        setIncidents((prev) => {
          const next = new Map(prev);
          const existing = next.get(ev.data.id);
          if (existing) next.set(ev.data.id, { ...existing, resolved: true });
          return next;
        });
        break;

      case "responder_calling":
        setResponders((prev) => {
          const next = new Map(prev);
          const existing = next.get(ev.data.responder_id) ?? {
            id: ev.data.responder_id,
            name: ev.data.responder_id,
            status: "idle",
          };
          next.set(ev.data.responder_id, {
            ...existing,
            status: "calling",
            incident_id: ev.data.incident_id,
          });
          return next;
        });
        break;

      case "responder_joined":
        setResponders((prev) => {
          const next = new Map(prev);
          const existing = next.get(ev.data.responder_id);
          if (existing) next.set(ev.data.responder_id, { ...existing, status: "answered" });
          return next;
        });
        break;

      case "responder_no_answer":
        setResponders((prev) => {
          const next = new Map(prev);
          const existing = next.get(ev.data.responder_id);
          if (existing) next.set(ev.data.responder_id, { ...existing, status: "no_answer" });
          return next;
        });
        break;

      case "claude_reasoning":
        setClaudeLog((prev) => [ev.data.text, ...prev].slice(0, 50));
        break;

      case "tool_call":
        setToolLog((prev) =>
          [{ tool: ev.data.tool, input: ev.data.input, ts: Date.now() }, ...prev].slice(0, 20)
        );
        break;
    }
  }, []);

  useEffect(() => {
    function connect() {
      const socket = new WebSocket(WS_URL);
      ws.current = socket;

      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // auto-reconnect
      };
      socket.onmessage = (msg) => {
        try {
          handleEvent(JSON.parse(msg.data) as ARIAEvent);
        } catch {}
      };
    }
    connect();
    return () => ws.current?.close();
  }, [handleEvent]);

  const resolveIncident = useCallback((incident_id: string) => {
    ws.current?.send(JSON.stringify({ action: "resolve_incident", incident_id }));
  }, []);

  const simulateIncident = useCallback((
    incident_type: string,
    severity: string,
    location: string,
  ) => {
    ws.current?.send(JSON.stringify({
      action: "simulate",
      incident_type,
      severity,
      location,
      camera_id: "cam_00",
    }));
  }, []);

  return {
    connected,
    cameras,
    incidents: Array.from(incidents.values()),
    responders: Array.from(responders.values()),
    recentDetections,
    cameraFrames,
    claudeLog,
    toolLog,
    resolveIncident,
    simulateIncident,
  };
}
