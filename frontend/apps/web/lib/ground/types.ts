// Contracts of the Artemis-X ground station API v1 (ground_station/artemis_ground/schemas.py, hub.py, ingest.py).
// Clock domains: `t_obs_s` = rover observation time in SIMULATION seconds; `*_utc` = ground reception time (UTC wall
// clock). Freshness is computed from UTC only; the two are never mixed.

export type Pagination = {
  limit: number;
  offset: number;
  total: number;
  next_offset: number | null;
};

export type Meta = {
  api_version: string;
  request_id: string | null;
  generated_utc: string;
  pagination: Pagination | null;
};

export type Envelope<T> = { data: T; meta: Meta };

export type ErrorEnvelope = {
  error: { code: string; message: string; details?: unknown };
  meta: Meta;
};

export type MissionStatus = "active" | "idle";

export type HealthState = { state: string; reasons: string[]; t_obs_s?: number; rx_utc?: string };

export type MissionCounters = Partial<{
  packets: number;
  pose: number;
  obstacles: number;
  event: number;
  duplicates: number;
  map_resets: number;
}>;

export type Mission = {
  mission_id: string;
  session: number;
  status: MissionStatus;
  created_utc: string;
  last_rx_utc: string;
  first_t_obs_s: number | null;
  last_t_obs_s: number | null;
  last_rx_age_s: number;
  counters: MissionCounters;
  health: HealthState | null;
  config: Record<string, unknown>;
  frame?: string;
};

export type Pose = {
  frame: "odom";
  x_m: number;
  y_m: number;
  yaw_rad: number;
  heading_deg: number;
  sigma_xy_m: number;
  sigma_xy_saturated: boolean;
};

// A persisted pose (REST) or a pose stamped with its envelope times (WebSocket).
export type Position = Pose & {
  seq?: number;
  t_obs_s: number;
  rx_utc: string;
};

export type ObstacleType = "rock" | "obstacle" | "negative_hazard";
export type ObstacleSource = "yolo+depth" | "depth";
export type ObstacleStatus = "confirmed" | "lost" | "observed";

export type Obstacle = {
  obstacle_id: string;
  type: ObstacleType;
  frame: "odom";
  x_m: number;
  y_m: number;
  radius_m: number;
  confidence: number;
  uncertainty_m: number;
  uncertainty_saturated?: boolean;
  obs_timestamp_s: number;
  source: ObstacleSource;
  status: ObstacleStatus;
  protected?: boolean;
  linked_track?: string | null;
  persistent_id: boolean;
  present?: boolean;
  last_rx_utc?: string;
  first_rx_utc?: string;
  first_t_obs_s?: number;
};

export type TelemetryRecord = {
  seq: number;
  packet_type: "pose" | "obstacles" | "event" | string;
  kind: string;
  t_obs_s: number;
  rx_utc: string;
  seq_flag: string;
  frame: string;
  event?: { event: string; state: string; reasons: string[]; value?: number };
};

export type DelayStats = { n: number; p50: number | null; p95: number | null; max: number | null };

export type EmulatorSummary = {
  preset: string;
  link_state: "UP" | "BLACKOUT";
  configured_loss: number;
  delivery_ratio: number | null;
  delay_ms: DelayStats;
  blackout_total_s: number;
  received: number;
  forwarded: number;
  dropped_loss: number;
  dropped_blackout: number;
  dropped_queue: number;
  throughput_Bps: number | null;
};

export type ReceiverCounters = Partial<{
  datagrams: number;
  valid: number;
  invalid: number;
  gaps: number;
  missing: number;
  duplicates: number;
  reordered: number;
  acks_sent: number;
  ack_seqs: number;
}>;

export type LinkStatus = {
  receiver: ReceiverCounters;
  packets_per_s: number | null;
  last_rx_age_s: number | null;
  link_up: boolean;
  emulator: EmulatorSummary | null;
  ingest?: { queue: number; dropped: number; db_ok: boolean };
};

export type LinkMetric = LinkStatus & { mission_id: string; rx_utc: string };

export type NetworkStatus = EmulatorSummary & {
  config: {
    loss: number;
    latency_ms: number;
    jitter_ms: number;
    bandwidth_Bps: number | null;
    blackout: boolean;
    queue_max: number;
  };
  elapsed_s: number;
  queued: number;
  in_flight: number;
  blackouts: number;
};

export const PRESETS = ["NORMAL", "DEGRADED", "SEVERE", "BLACKOUT", "RESTORE"] as const;
export type Preset = (typeof PRESETS)[number];

export type MissionOverview = {
  mission: Mission;
  latest_position: Position | null;
  health: HealthState | null;
  obstacles: { present: number; by_type: Record<string, number>; by_source: Record<string, number> };
  link: LinkMetric | null;
  freshness: { last_rx_age_s: number; clock: string };
};

export type MissionStats = {
  mission_id: string;
  counters: MissionCounters;
  documents: Record<string, number>;
  obstacles_present: number;
  receiver: ReceiverCounters | null;
  emulator: EmulatorSummary | null;
  ingestion: Record<string, unknown>;
  observation_span_s: number | null;
};

export type Health = {
  status: "ok" | "degraded";
  api_version: string;
  mongodb: "up" | "down";
  receiver: Record<string, unknown>;
  ingestion: Record<string, unknown>;
  websocket_clients: number;
};

// ---- WebSocket events: {type, api_version, mission_id, t_obs_s | null, rx_utc, data} --------------------------------
type Ev<T extends string, D> = {
  type: T;
  api_version: string;
  mission_id: string;
  t_obs_s: number | null;
  rx_utc: string;
  data: D;
};

export type MissionSnapshot = {
  mission: (Omit<Mission, "status" | "last_rx_age_s"> & Partial<Pick<Mission, "status">>) | null;
  latest_position: Position | null;
  obstacles: Obstacle[];
  source: "mongodb" | "unavailable";
  error?: string;
};

export type GroundEvent =
  | Ev<"mission.snapshot", MissionSnapshot>
  | Ev<"rover.position", Pose>
  | Ev<"obstacle.detected", Obstacle>
  | Ev<"obstacle.updated", Obstacle>
  | Ev<"obstacle.removed", { obstacle_id: string; reason: string }>
  | Ev<"map.reset", { resets: number; state: string }>
  | Ev<"health.changed", { state: string; reasons: string[] }>
  | Ev<"telemetry.received", { seq: number; packet_type: string | null; seq_flag: string }>
  | Ev<"link.status", LinkStatus>
  | Ev<"mission.stats", Record<string, unknown>>;

export type GroundEventType = GroundEvent["type"];
