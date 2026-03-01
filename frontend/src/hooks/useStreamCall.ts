/**
 * useStreamCall — joins a GetStream video call with the local webcam.
 * The Vision Agents Python agent is already in the call waiting for video.
 * Once we join and enable camera, the agent's YOLO processor starts running.
 */
import { useEffect, useRef, useState } from "react";
import {
  StreamVideoClient,
  StreamCall,
  Call,
} from "@stream-io/video-react-sdk";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface StreamCallState {
  client: StreamVideoClient | null;
  call: Call | null;
  joined: boolean;
  error: string | null;
}

export function useStreamCall() {
  const [state, setState] = useState<StreamCallState>({
    client: null,
    call: null,
    joined: false,
    error: null,
  });
  const initialised = useRef(false);

  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;

    async function init() {
      try {
        // Fetch token + call info from ARIA backend
        const res = await fetch(`${API_BASE}/stream-token?user_id=operator`);
        if (!res.ok) throw new Error(`Token fetch failed: ${res.status}`);
        const { token, api_key, user_id, call_type, call_id } = await res.json();

        const client = new StreamVideoClient({
          apiKey: api_key,
          user: { id: user_id, name: "ARIA Operator" },
          token,
        });

        const call = client.call(call_type, call_id);

        // Set VP8 preference defensively (no-op since camera is disabled below,
        // but kept in case mic is ever re-enabled on a non-macOS host).
        call.updatePublishOptions({ preferredCodec: "vp8" });

        // Join the call without publishing a camera track.
        // The ARIA Python agent processes video via local OpenCV (not WebRTC),
        // so the operator's browser camera is not needed. Publishing it triggers
        // repeated WebRTC codec negotiation timeouts (H.264 vs VP8 on macOS)
        // that destabilize the agent's Stream call connection.
        await call.join({ create: true });
        await call.camera.disable();
        await call.microphone.disable();

        setState({ client, call, joined: true, error: null });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("Stream call join failed:", msg);
        setState((s) => ({ ...s, error: msg }));
      }
    }

    init();

    return () => {
      // cleanup on unmount
      setState((s) => {
        s.call?.leave().catch(() => {});
        s.client?.disconnectUser().catch(() => {});
        return { client: null, call: null, joined: false, error: null };
      });
    };
  }, []);

  return state;
}
