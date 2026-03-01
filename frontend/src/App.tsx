import { useState } from "react";
import { useARIA } from "./hooks/useARIA";
import { useStreamCall } from "./hooks/useStreamCall";
import { IncidentCard } from "./components/IncidentCard";
import {
  Wifi, WifiOff, Camera, Phone, Zap, AlertTriangle,
  Activity, Radio, Terminal, CheckCircle, FlaskConical
} from "lucide-react";
import clsx from "clsx";

const SIMULATE_TYPES = [
  { type: "medical_fall",       label: "🏥 Patient Fall",      severity: "HIGH"     },
];

export default function App() {
  const {
    connected, cameras, incidents, responders,
    recentDetections, cameraFrames, claudeLog, toolLog, resolveIncident, simulateIncident,
  } = useARIA();
  const { joined: streamJoined, error: streamError } = useStreamCall();
  const [simLocation, setSimLocation] = useState("Demo Room");

  const activeIncidents = incidents.filter((i) => !i.resolved);
  const resolvedIncidents = incidents.filter((i) => i.resolved);

  const criticalCount = activeIncidents.filter((i) => i.severity === "CRITICAL").length;

  return (
    <div className="min-h-screen bg-slate-950 text-white font-mono flex flex-col">

      {/* ── Top bar ─────────────────────────────────────────── */}
      <header className="border-b border-slate-800 px-6 py-3 flex items-center justify-between bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Radio className="w-5 h-5 text-cyan-400 animate-pulse" />
          <span className="text-lg font-bold tracking-widest text-cyan-400">ARIA</span>
          <span className="text-xs text-slate-500 tracking-wider">
            AUTONOMOUS RESPONSE INTELLIGENCE AGENT
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs">
          {criticalCount > 0 && (
            <span className="flex items-center gap-1.5 bg-red-500 text-white px-3 py-1 rounded-full animate-pulse font-bold">
              <AlertTriangle className="w-3.5 h-3.5" />
              {criticalCount} CRITICAL
            </span>
          )}

          <span className="flex items-center gap-1.5 text-slate-400">
            <Camera className="w-3.5 h-3.5" />
            {cameras.length} cameras
          </span>

          <span className={clsx(
            "flex items-center gap-1.5 px-2 py-1 rounded-full text-xs",
            streamJoined ? "text-cyan-400" : streamError ? "text-red-400" : "text-yellow-400"
          )}>
            <Activity className="w-3 h-3" />
            {streamJoined ? "STREAM LIVE" : streamError ? "STREAM ERR" : "STREAM..."}
          </span>

          <span className={clsx(
            "flex items-center gap-1.5 px-2 py-1 rounded-full",
            connected ? "text-green-400" : "text-red-400"
          )}>
            {connected
              ? <><Wifi className="w-3.5 h-3.5" /> LIVE</>
              : <><WifiOff className="w-3.5 h-3.5" /> OFFLINE</>
            }
          </span>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: Active Incidents ───────────────────────── */}
        <aside className="w-80 border-r border-slate-800 flex flex-col shrink-0">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <span className="text-xs font-bold text-slate-400 tracking-wider">ACTIVE INCIDENTS</span>
            <span className="bg-red-900/50 text-red-300 text-xs px-2 py-0.5 rounded-full border border-red-800">
              {activeIncidents.length}
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {activeIncidents.length === 0 ? (
              <div className="text-center py-12 text-slate-600">
                <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">All clear</p>
              </div>
            ) : (
              activeIncidents.map((inc) => (
                <IncidentCard key={inc.id} incident={inc} onResolve={resolveIncident} />
              ))
            )}
          </div>

          {/* Resolved */}
          {resolvedIncidents.length > 0 && (
            <>
              <div className="px-4 py-2 border-t border-slate-800 text-xs text-slate-600 font-bold tracking-wider">
                RESOLVED ({resolvedIncidents.length})
              </div>
              <div className="max-h-40 overflow-y-auto p-3 space-y-2">
                {resolvedIncidents.slice(0, 5).map((inc) => (
                  <IncidentCard key={inc.id} incident={inc} onResolve={resolveIncident} />
                ))}
              </div>
            </>
          )}
        </aside>

        {/* ── Center: Camera grid ──────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-bold text-slate-400 tracking-wider">CAMERA FEEDS</span>
          </div>

          {/* Camera grid — flex column so cards stretch to fill all available height */}
          <div className={clsx(
            "flex-1 p-4 overflow-hidden min-h-0",
            cameras.length > 1
              ? "grid grid-cols-2 gap-4 overflow-y-auto"
              : "flex flex-col gap-4"
          )}>
            {cameras.map((cam) => {
              const latestDetection = recentDetections.find(
                (d) => d.camera_id === cam.id
              );
              const activeIncident = activeIncidents.find(
                (i) => i.camera_id === cam.id
              );

              // Live frame: prefer streaming frame, fall back to last detection frame
              const liveFrame = cameraFrames.get(cam.id) ?? latestDetection?.frame_b64;

              return (
                <div
                  key={cam.id}
                  className={clsx(
                    "rounded-xl border overflow-hidden bg-slate-900 flex flex-col min-h-0",
                    cameras.length === 1 && "flex-1",
                    activeIncident?.severity === "CRITICAL"
                      ? "border-red-500 shadow-lg shadow-red-500/20"
                      : activeIncident
                      ? "border-orange-500"
                      : "border-slate-800"
                  )}
                >
                  {/* Camera header */}
                  <div className="px-3 py-2 bg-slate-800/60 flex items-center justify-between shrink-0">
                    <div className="flex items-center gap-2">
                      <Camera className="w-3.5 h-3.5 text-slate-400" />
                      <span className="text-xs text-slate-300 font-bold">{cam.id.toUpperCase()}</span>
                      <span className="text-xs text-slate-500">{cam.location}</span>
                    </div>
                    {activeIncident && (
                      <span className={clsx(
                        "text-xs px-2 py-0.5 rounded-full font-bold",
                        activeIncident.severity === "CRITICAL"
                          ? "bg-red-500 text-white animate-pulse"
                          : "bg-orange-500 text-white"
                      )}>
                        {activeIncident.severity}
                      </span>
                    )}
                  </div>

                  {/* Frame — flex-1 fills all remaining card height */}
                  <div className="relative bg-black flex-1 min-h-0">
                    {liveFrame ? (
                      <img
                        src={`data:image/jpeg;base64,${liveFrame}`}
                        alt={`Camera ${cam.id}`}
                        className="absolute inset-0 w-full h-full object-contain"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-slate-700">
                        <Camera className="w-8 h-8" />
                      </div>
                    )}

                    {/* Overlay on active incident */}
                    {activeIncident && (
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                        <p className="text-xs text-red-300 font-bold">
                          ⚠ {activeIncident.type.replace("_", " ").toUpperCase()}
                        </p>
                        <p className="text-xs text-white/70 mt-0.5 line-clamp-2">
                          {activeIncident.description}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Detection footer */}
                  {latestDetection && (
                    <div className="px-3 py-1.5 bg-slate-800/40 text-xs text-slate-500 shrink-0">
                      Last: {latestDetection.incident_type.replace("_", " ")} —{" "}
                      {new Date(latestDetection.timestamp * 1000).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              );
            })}

            {cameras.length === 0 && (
              <div className="flex-1 flex items-center justify-center text-slate-700">
                <div className="text-center">
                  <Camera className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">Waiting for cameras...</p>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ── Right: ARIA brain + responders ───────────────── */}
        <aside className="w-80 border-l border-slate-800 flex flex-col shrink-0">

          {/* ── Simulate panel ─────────────────────────────── */}
          <div className="border-b border-slate-800 shrink-0">
            <div className="px-4 py-3 flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-yellow-400" />
              <span className="text-xs font-bold text-slate-400 tracking-wider">SIMULATE INCIDENT</span>
            </div>
            <div className="px-3 pb-3 space-y-2">
              <input
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-600"
                placeholder="Location (e.g. Room 214)"
                value={simLocation}
                onChange={(e) => setSimLocation(e.target.value)}
              />
              <div className="grid grid-cols-2 gap-1.5">
                {SIMULATE_TYPES.map((s) => (
                  <button
                    key={s.type}
                    disabled={!connected}
                    onClick={() => simulateIncident(s.type, s.severity, simLocation || "Demo Room")}
                    className="text-xs bg-slate-800 hover:bg-yellow-900/50 border border-slate-700 hover:border-yellow-700 text-slate-300 hover:text-yellow-200 px-2 py-2 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-left leading-tight"
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Responder status */}
          <div className="border-b border-slate-800">
            <div className="px-4 py-3 flex items-center gap-2">
              <Phone className="w-4 h-4 text-green-400" />
              <span className="text-xs font-bold text-slate-400 tracking-wider">RESPONDERS</span>
            </div>
            <div className="px-3 pb-3 space-y-1.5 max-h-44 overflow-y-auto">
              {responders.length === 0 ? (
                <p className="text-xs text-slate-600 px-1">No active calls</p>
              ) : (
                responders.map((r) => (
                  <div
                    key={r.id}
                    className={clsx(
                      "flex items-center justify-between px-3 py-2 rounded-lg text-xs",
                      r.status === "on_call"     ? "bg-green-950/50 border border-green-800"
                      : r.status === "answered"  ? "bg-green-950/50 border border-green-800"
                      : r.status === "calling"   ? "bg-blue-950/50 border border-blue-800"
                      : r.status === "ended"     ? "bg-slate-900/50 border border-slate-700"
                      : r.status === "no_answer" ? "bg-red-950/30 border border-red-900"
                      : "bg-slate-900 border border-slate-800"
                    )}
                  >
                    <span className="text-slate-200">{r.name.replace("_", " ")}</span>
                    <span className={clsx(
                      "font-bold uppercase",
                      r.status === "on_call"     ? "text-green-400"
                      : r.status === "answered"  ? "text-green-400"
                      : r.status === "calling"   ? "text-blue-400 animate-pulse"
                      : r.status === "ended"     ? "text-slate-500"
                      : r.status === "no_answer" ? "text-red-400"
                      : "text-slate-500"
                    )}>
                      {r.status === "calling" ? "📞 RINGING..."
                       : r.status === "on_call" ? "🟢 ON CALL"
                       : r.status === "answered" ? "🟢 ON CALL"
                       : r.status === "ended" ? "ENDED"
                       : r.status === "no_answer" ? "NO ANSWER"
                       : r.status}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Claude reasoning */}
          <div className="border-b border-slate-800 flex-1 flex flex-col min-h-0">
            <div className="px-4 py-3 flex items-center gap-2 shrink-0">
              <Zap className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-bold text-slate-400 tracking-wider">ARIA REASONING</span>
            </div>
            <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-2">
              {claudeLog.slice(0, 8).map((text, i) => (
                <div key={i} className="bg-purple-950/30 border border-purple-900/50 rounded-lg px-3 py-2">
                  <p className="text-xs text-purple-200 leading-relaxed">{text}</p>
                </div>
              ))}
              {claudeLog.length === 0 && (
                <p className="text-xs text-slate-600 px-1">Monitoring...</p>
              )}
            </div>
          </div>

          {/* Tool calls */}
          <div className="shrink-0">
            <div className="px-4 py-3 flex items-center gap-2 border-t border-slate-800">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-slate-400 tracking-wider">ACTIONS</span>
            </div>
            <div className="px-3 pb-3 space-y-1.5 max-h-40 overflow-y-auto">
              {toolLog.map((t, i) => (
                <div key={i} className="bg-cyan-950/30 border border-cyan-900/40 rounded-lg px-3 py-2">
                  <p className="text-xs font-bold text-cyan-300">
                    ⚡ {t.tool.replace("_", " ")}
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {new Date(t.ts).toLocaleTimeString()}
                  </p>
                </div>
              ))}
              {toolLog.length === 0 && (
                <p className="text-xs text-slate-600 px-1">No actions yet</p>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
