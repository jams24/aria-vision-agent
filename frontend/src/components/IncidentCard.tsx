import { formatDistanceToNow } from "date-fns";
import { ShieldAlert, Flame, UserX, AlertTriangle, CheckCircle, Phone } from "lucide-react";
import clsx from "clsx";
import type { Incident } from "../types/incident";

const SEVERITY_STYLES: Record<string, string> = {
  CRITICAL: "border-red-500 bg-red-950/40 shadow-red-500/20 shadow-lg",
  HIGH:     "border-orange-500 bg-orange-950/40 shadow-orange-500/10 shadow-md",
  MEDIUM:   "border-yellow-500 bg-yellow-950/30",
  LOW:      "border-slate-600 bg-slate-900/40",
};

const SEVERITY_BADGE: Record<string, string> = {
  CRITICAL: "bg-red-500 text-white animate-pulse",
  HIGH:     "bg-orange-500 text-white",
  MEDIUM:   "bg-yellow-500 text-black",
  LOW:      "bg-slate-600 text-white",
};

const TYPE_ICON: Record<string, React.ReactNode> = {
  medical_fall:        <UserX className="w-5 h-5" />,
  fire:                <Flame className="w-5 h-5" />,
  security_intrusion:  <ShieldAlert className="w-5 h-5" />,
  unresponsive_person: <AlertTriangle className="w-5 h-5" />,
};

const TYPE_LABEL: Record<string, string> = {
  medical_fall:        "Patient Fall",
  fire:                "Fire / Smoke",
  security_intrusion:  "Security Intrusion",
  unresponsive_person: "Unresponsive Person",
};

interface Props {
  incident: Incident;
  onResolve: (id: string) => void;
}

export function IncidentCard({ incident, onResolve }: Props) {
  const timeAgo = formatDistanceToNow(new Date(incident.timestamp * 1000), { addSuffix: true });

  return (
    <div
      className={clsx(
        "rounded-xl border p-4 transition-all duration-300",
        SEVERITY_STYLES[incident.severity],
        incident.resolved && "opacity-40 grayscale"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-white/80 shrink-0">
            {TYPE_ICON[incident.type]}
          </span>
          <div className="min-w-0">
            <p className="font-semibold text-white truncate">
              {TYPE_LABEL[incident.type]}
            </p>
            <p className="text-xs text-slate-400 truncate">{incident.location}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className={clsx("text-xs font-bold px-2 py-0.5 rounded-full", SEVERITY_BADGE[incident.severity])}>
            {incident.severity}
          </span>
          {incident.resolved && (
            <CheckCircle className="w-4 h-4 text-green-400" />
          )}
        </div>
      </div>

      {/* Scene thumbnail */}
      {incident.frame_b64 && (
        <img
          src={`data:image/jpeg;base64,${incident.frame_b64}`}
          alt="Scene snapshot"
          className="w-full h-28 object-cover rounded-lg mb-3 border border-white/10"
        />
      )}

      {/* Description */}
      <p className="text-sm text-slate-300 leading-relaxed mb-3">
        {incident.description}
      </p>

      {/* Responders */}
      {(incident.responders ?? []).length > 0 && (
        <div className="flex items-center gap-1.5 mb-3 flex-wrap">
          <Phone className="w-3.5 h-3.5 text-green-400 shrink-0" />
          {(incident.responders ?? []).map((r) => (
            <span key={r} className="text-xs bg-green-900/50 text-green-300 border border-green-700 px-2 py-0.5 rounded-full">
              {r.replace("_", " ")}
            </span>
          ))}
        </div>
      )}

      {/* Live updates */}
      {(incident.updates ?? []).length > 0 && (
        <div className="border-t border-white/10 pt-2 mb-3 space-y-1 max-h-24 overflow-y-auto">
          {(incident.updates ?? []).map((u, i) => (
            <p key={i} className="text-xs text-slate-400 italic">{u}</p>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">{timeAgo}</span>
        {!incident.resolved && (
          <button
            onClick={() => onResolve(incident.id)}
            className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1 rounded-full transition-colors"
          >
            Mark Resolved
          </button>
        )}
      </div>
    </div>
  );
}
