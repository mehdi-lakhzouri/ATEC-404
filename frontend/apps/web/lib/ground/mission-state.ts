// Pure mission state for the dashboard: built from REST, advanced by WebSocket events, never from anything else.
// Only type imports here, so the module runs under `node --test` (type stripping) without a bundler.
import type {
  GroundEvent,
  HealthState,
  LinkMetric,
  LinkStatus,
  Mission,
  MissionSnapshot,
  Obstacle,
  Pose,
  Position,
  TelemetryRecord,
} from "./types";

export const MAX_TRAJECTORY = 20000;
export const MAX_LINK_SAMPLES = 180;
export const MAX_LOG = 200;

export type LogCategory = "Navigation" | "Detection" | "System" | "Communication";
export type LogTone = "info" | "good" | "warn" | "bad";

export type LogEntry = {
  key: string;
  rx_utc: string;
  t_obs_s: number | null;
  category: LogCategory;
  text: string;
  detail?: string;
  tone: LogTone;
};

export type LinkSample = {
  rx_ms: number;
  packets_per_s: number | null;
  loss_pct: number | null; // emulator drops / emulator receptions over the sample window
  delay_p50_ms: number | null; // emulator cumulative p50 (reported as is)
  preset: string | null;
  link_up: boolean;
};

export type TrackedObstacle = Obstacle & { rx_ms: number };

export type MissionView = Partial<Mission> & { mission_id: string };

export type MissionState = {
  missionId: string;
  mission: MissionView | null;
  trajectory: Position[]; // ascending t_obs_s, unique t_obs_s
  obstacles: Record<string, TrackedObstacle>;
  health: HealthState | null;
  link: (LinkStatus & { rx_utc: string }) | null;
  linkSamples: LinkSample[];
  log: LogEntry[]; // newest first
  lastRxMs: number | null; // last ground reception of a rover packet (UTC ms)
  packetsSeen: number;
  duplicatesSeen: number;
  mapResets: number;
  snapshots: number;
  snapshotSource: MissionSnapshot["source"] | null;
  ignoredForeign: number;
};

export function initialMissionState(missionId: string): MissionState {
  return {
    missionId,
    mission: null,
    trajectory: [],
    obstacles: {},
    health: null,
    link: null,
    linkSamples: [],
    log: [],
    lastRxMs: null,
    packetsSeen: 0,
    duplicatesSeen: 0,
    mapResets: 0,
    snapshots: 0,
    snapshotSource: null,
    ignoredForeign: 0,
  };
}

const ms = (iso: string | null | undefined) => {
  if (!iso) return null;
  const v = Date.parse(iso);
  return Number.isNaN(v) ? null : v;
};

const maxMs = (a: number | null, b: number | null) =>
  a === null ? b : b === null ? a : Math.max(a, b);

/** Inserts poses keeping ascending observation time; a pose with an already known t_obs_s is a duplicate. */
export function mergePositions(trajectory: Position[], incoming: Position[]): Position[] {
  if (!incoming.length) return trajectory;
  let out = trajectory;
  let copied = false;
  for (const p of incoming) {
    if (!Number.isFinite(p.t_obs_s) || !Number.isFinite(p.x_m) || !Number.isFinite(p.y_m)) continue;
    const last = out[out.length - 1];
    if (!last || p.t_obs_s > last.t_obs_s) {
      if (!copied) {
        out = out.slice();
        copied = true;
      }
      out.push(p);
      continue;
    }
    let lo = 0;
    let hi = out.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (out[mid]!.t_obs_s < p.t_obs_s) lo = mid + 1;
      else hi = mid;
    }
    if (out[lo]?.t_obs_s === p.t_obs_s) continue; // duplicate (retransmission, snapshot overlap)
    if (!copied) {
      out = out.slice();
      copied = true;
    }
    out.splice(lo, 0, p); // late packet (store-and-forward): placed by observation time
  }
  if (out.length > MAX_TRAJECTORY) out = out.slice(out.length - MAX_TRAJECTORY);
  return out;
}

export function latestPosition(state: MissionState): Position | null {
  return state.trajectory[state.trajectory.length - 1] ?? null;
}

function pushLog(log: LogEntry[], entry: LogEntry): LogEntry[] {
  if (log.some((e) => e.key === entry.key)) return log;
  const out = [entry, ...log];
  out.sort((a, b) => (ms(b.rx_utc) ?? 0) - (ms(a.rx_utc) ?? 0));
  return out.length > MAX_LOG ? out.slice(0, MAX_LOG) : out;
}

const typeLabel: Record<string, string> = {
  rock: "Rock",
  obstacle: "Obstacle",
  negative_hazard: "Negative hazard",
};

function obstacleLog(o: Obstacle, rx_utc: string, t_obs_s: number | null): LogEntry {
  return {
    key: `obstacle.detected:${o.obstacle_id}`,
    rx_utc,
    t_obs_s,
    category: "Detection",
    text: `${typeLabel[o.type] ?? o.type} detected ${o.obstacle_id}`,
    detail: `(${o.x_m.toFixed(2)}, ${o.y_m.toFixed(2)}) m · conf ${(o.confidence * 100).toFixed(0)}% · ±${o.uncertainty_m.toFixed(2)} m · ${o.source}`,
    tone: o.type === "rock" ? "good" : "bad",
  };
}

function healthTone(state: string): LogTone {
  const s = state.toUpperCase();
  return s === "OK" || s === "NOMINAL" ? "good" : s === "DEGRADED" || s === "WARN" ? "warn" : "bad";
}

function linkSample(prevLink: LinkStatus | null, l: LinkStatus, rx_ms: number): LinkSample {
  let loss_pct: number | null = null;
  const e = l.emulator;
  const pe = prevLink?.emulator;
  if (e && pe) {
    const dRecv = e.received - pe.received;
    const dDrop =
      e.dropped_loss + e.dropped_blackout + e.dropped_queue - (pe.dropped_loss + pe.dropped_blackout + pe.dropped_queue);
    if (dRecv > 0 && dDrop >= 0) loss_pct = Math.min(100, (100 * dDrop) / dRecv);
  }
  return {
    rx_ms,
    packets_per_s: l.packets_per_s,
    loss_pct,
    delay_p50_ms: e?.delay_ms?.p50 ?? null,
    preset: e?.preset ?? null,
    link_up: l.link_up,
  };
}

function applyLink(state: MissionState, l: LinkStatus, rx_utc: string): MissionState {
  const rx_ms = ms(rx_utc);
  if (rx_ms === null) return state;
  const prevSample = state.linkSamples[state.linkSamples.length - 1];
  if (prevSample && rx_ms <= prevSample.rx_ms) return { ...state }; // replayed / older sample
  const sample = linkSample(state.link, l, rx_ms);
  const samples = [...state.linkSamples, sample].slice(-MAX_LINK_SAMPLES);
  let log = state.log;
  if (state.link && state.link.link_up !== l.link_up)
    log = pushLog(log, {
      key: `link:${rx_utc}`,
      rx_utc,
      t_obs_s: null,
      category: "Communication",
      text: l.link_up ? "Telemetry downlink restored" : "Telemetry downlink lost",
      detail: l.last_rx_age_s === null ? "" : `(last packet ${l.last_rx_age_s.toFixed(1)} s ago)`,
      tone: l.link_up ? "good" : "bad",
    });
  const prevPreset = state.link?.emulator?.preset;
  if (prevPreset && l.emulator && prevPreset !== l.emulator.preset)
    log = pushLog(log, {
      key: `preset:${rx_utc}`,
      rx_utc,
      t_obs_s: null,
      category: "Communication",
      text: `Lunar link preset ${prevPreset} → ${l.emulator.preset}`,
      detail: `(link ${l.emulator.link_state})`,
      tone: l.emulator.link_state === "BLACKOUT" ? "bad" : "info",
    });
  return { ...state, link: { ...l, rx_utc }, linkSamples: samples, log };
}

function applySnapshot(state: MissionState, s: MissionSnapshot, rx_utc: string): MissionState {
  const snapRx = ms(s.mission?.last_rx_utc ?? null);
  const obstacles: Record<string, TrackedObstacle> = {};
  for (const o of s.obstacles ?? []) obstacles[o.obstacle_id] = { ...o, rx_ms: ms(o.last_rx_utc) ?? snapRx ?? 0 };
  // The snapshot is the ground station's persisted, reconciled picture. Live detections newer than the snapshot's last
  // reception may not be written yet (asynchronous writer): those are kept, everything else follows the snapshot.
  if (s.source === "mongodb")
    for (const [id, o] of Object.entries(state.obstacles))
      if (!(id in obstacles) && snapRx !== null && o.rx_ms > snapRx) obstacles[id] = o;
  const latest = s.latest_position;
  return {
    ...state,
    mission: s.mission ? { ...state.mission, ...s.mission } : state.mission,
    health: s.mission?.health ?? state.health,
    obstacles: s.source === "mongodb" ? obstacles : state.obstacles,
    trajectory: latest ? mergePositions(state.trajectory, [latest]) : state.trajectory,
    lastRxMs: maxMs(state.lastRxMs, snapRx),
    snapshots: state.snapshots + 1,
    snapshotSource: s.source,
    log:
      s.source === "unavailable"
        ? pushLog(state.log, {
            key: `snapshot-unavailable:${rx_utc}`,
            rx_utc,
            t_obs_s: null,
            category: "System",
            text: "Ground database unavailable during resync",
            detail: s.error ? `(${s.error})` : "",
            tone: "warn",
          })
        : state.log,
  };
}

/** Applies one WebSocket event. Events of another mission are ignored (counted), never mixed in. */
export function applyEvent(state: MissionState, ev: GroundEvent): MissionState {
  if (ev.mission_id !== state.missionId) return { ...state, ignoredForeign: state.ignoredForeign + 1 };
  const rx = ms(ev.rx_utc);
  switch (ev.type) {
    case "mission.snapshot":
      return applySnapshot(state, ev.data, ev.rx_utc);
    case "telemetry.received":
      return {
        ...state,
        lastRxMs: maxMs(state.lastRxMs, rx),
        packetsSeen: state.packetsSeen + 1,
        duplicatesSeen: state.duplicatesSeen + (ev.data.seq_flag === "duplicate" ? 1 : 0),
      };
    case "rover.position": {
      if (ev.t_obs_s === null) return state;
      const p: Position = { ...(ev.data as Pose), t_obs_s: ev.t_obs_s, rx_utc: ev.rx_utc };
      return { ...state, trajectory: mergePositions(state.trajectory, [p]), lastRxMs: maxMs(state.lastRxMs, rx) };
    }
    case "obstacle.detected":
    case "obstacle.updated": {
      const o = ev.data;
      const known = o.obstacle_id in state.obstacles;
      const obstacles = { ...state.obstacles, [o.obstacle_id]: { ...o, last_rx_utc: ev.rx_utc, rx_ms: rx ?? 0 } };
      const log = known ? state.log : pushLog(state.log, obstacleLog(o, ev.rx_utc, ev.t_obs_s));
      return { ...state, obstacles, log, lastRxMs: maxMs(state.lastRxMs, rx) };
    }
    case "obstacle.removed": {
      const id = ev.data.obstacle_id;
      if (!(id in state.obstacles)) return state;
      const obstacles = { ...state.obstacles };
      const gone = obstacles[id]!;
      delete obstacles[id];
      return {
        ...state,
        obstacles,
        log: pushLog(state.log, {
          key: `obstacle.removed:${id}:${ev.rx_utc}`,
          rx_utc: ev.rx_utc,
          t_obs_s: ev.t_obs_s,
          category: "Navigation",
          text: `Obstacle ${id} removed by reconciliation`,
          detail: `(${typeLabel[gone.type] ?? gone.type}, ${ev.data.reason.replace("_", " ")})`,
          tone: "info",
        }),
      };
    }
    case "map.reset":
      return {
        ...state,
        obstacles: {},
        mapResets: state.mapResets + 1,
        log: pushLog(state.log, {
          key: `map.reset:${ev.data.resets}:${ev.rx_utc}`,
          rx_utc: ev.rx_utc,
          t_obs_s: ev.t_obs_s,
          category: "Navigation",
          text: "Rover map reset — obstacle picture cleared",
          detail: `(reset #${ev.data.resets}, health ${ev.data.state})`,
          tone: "warn",
        }),
      };
    case "health.changed":
      return {
        ...state,
        health: { state: ev.data.state, reasons: ev.data.reasons, t_obs_s: ev.t_obs_s ?? undefined, rx_utc: ev.rx_utc },
        log: pushLog(state.log, {
          key: `health:${ev.t_obs_s}:${ev.data.state}`,
          rx_utc: ev.rx_utc,
          t_obs_s: ev.t_obs_s,
          category: "System",
          text: `Rover health ${ev.data.state}`,
          detail: ev.data.reasons.length ? `(${ev.data.reasons.join(", ")})` : "",
          tone: healthTone(ev.data.state),
        }),
      };
    case "link.status":
      return applyLink(state, ev.data, ev.rx_utc);
    case "mission.stats":
      return state;
    default:
      return state;
  }
}

export function applyEvents(state: MissionState, events: GroundEvent[]): MissionState {
  let s = state;
  for (const ev of events) s = applyEvent(s, ev);
  return s;
}

/** Initial REST load (or resync): mission, trajectory, current obstacle picture, link history and stored events. */
export function hydrate(
  state: MissionState,
  rest: {
    mission: Mission;
    positions: Position[];
    obstacles: Obstacle[];
    link: LinkMetric[];
    events: TelemetryRecord[];
  },
): MissionState {
  const obstacles: Record<string, TrackedObstacle> = {};
  for (const o of rest.obstacles) obstacles[o.obstacle_id] = { ...o, rx_ms: ms(o.last_rx_utc) ?? 0 };
  let s: MissionState = {
    ...state,
    mission: { ...state.mission, ...rest.mission },
    health: rest.mission.health ?? state.health,
    trajectory: mergePositions(state.trajectory, rest.positions),
    obstacles: { ...obstacles, ...state.obstacles },
    lastRxMs: maxMs(state.lastRxMs, ms(rest.mission.last_rx_utc)),
  };
  const linkAsc = [...rest.link].sort((a, b) => (ms(a.rx_utc) ?? 0) - (ms(b.rx_utc) ?? 0));
  let linked = s;
  for (const l of linkAsc) linked = applyLink(linked, l, l.rx_utc);
  s = { ...linked, log: s.log }; // history samples feed the chart; transitions are logged live only
  for (const e of rest.events) {
    const ev = e.event;
    if (!ev) continue;
    if (e.kind === "health")
      s = {
        ...s,
        log: pushLog(s.log, {
          key: `health:${e.t_obs_s}:${ev.state}`,
          rx_utc: e.rx_utc,
          t_obs_s: e.t_obs_s,
          category: "System",
          text: `Rover health ${ev.state}`,
          detail: ev.reasons?.length ? `(${ev.reasons.join(", ")})` : "",
          tone: healthTone(ev.state),
        }),
      };
    else if (e.kind === "map_reset")
      s = {
        ...s,
        log: pushLog(s.log, {
          key: `map.reset:${ev.value}:stored`,
          rx_utc: e.rx_utc,
          t_obs_s: e.t_obs_s,
          category: "Navigation",
          text: "Rover map reset — obstacle picture cleared",
          detail: `(reset #${ev.value}, health ${ev.state})`,
          tone: "warn",
        }),
      };
  }
  for (const o of rest.obstacles) {
    const rx = o.first_rx_utc ?? o.last_rx_utc;
    if (rx) s = { ...s, log: pushLog(s.log, obstacleLog(o, rx, o.first_t_obs_s ?? o.obs_timestamp_s)) };
  }
  return s;
}
