export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export type IncidentType =
  | "medical_fall"
  | "sudden_collapse";

export interface Incident {
  id: string;
  type: IncidentType;
  severity: Severity;
  location: string;
  description: string;
  camera_id: string;
  timestamp: number;
  resolved: boolean;
  responders: string[];
  updates: string[];
  frame_b64?: string;
}

export interface Responder {
  id: string;
  name: string;
  status: "idle" | "calling" | "on_call" | "answered" | "no_answer" | "ended";
  incident_id?: string;
}

export type ARIAEvent =
  | { event: "system_ready"; data: { cameras: Camera[]; timestamp: number } }
  | { event: "detection"; data: Detection }
  | { event: "incident_created"; data: Incident }
  | { event: "incident_updated"; data: Incident }
  | { event: "incident_resolved"; data: Incident }
  | { event: "incident_escalated"; data: { incident_id: string; escalating_to: string[] } }
  | { event: "responder_calling"; data: { incident_id: string; responder_id: string } }
  | { event: "responder_joined"; data: { incident_id: string; responder_id: string } }
  | { event: "responder_no_answer"; data: { incident_id: string; responder_id: string } }
  | { event: "responder_hangup"; data: { incident_id: string; responder_id: string } }
  | { event: "claude_reasoning"; data: { text: string } }
  | { event: "tool_call"; data: { tool: string; input: Record<string, unknown> } }
  | { event: "camera_frame"; data: { camera_id: string; frame_b64: string } };

export interface Detection {
  incident_type: IncidentType;
  severity: Severity;
  location: string;
  description: string;
  camera_id: string;
  timestamp: number;
  frame_b64: string;
}

export interface Camera {
  id: string;
  location: string;
}
