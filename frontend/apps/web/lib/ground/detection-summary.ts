import type { LogEntry } from "./mission-state";
import type { Obstacle, Position } from "./types";

export type DetectionEventKind = "detected" | "removed" | "reset";

export function detectionEventKind(entry: LogEntry): DetectionEventKind | null {
  if (entry.key.startsWith("obstacle.detected:")) return "detected";
  if (entry.key.startsWith("obstacle.removed:")) return "removed";
  if (entry.key.startsWith("map.reset:")) return "reset";
  return null;
}

export function eventsForObstacle(entries: LogEntry[], id: string) {
  return entries.filter((entry) => entry.key === `obstacle.detected:${id}` || entry.key.startsWith(`obstacle.removed:${id}:`));
}

/** Counts only events actually retained in the session log, with ground UTC minute buckets. */
export function detectionActivity(entries: LogEntry[]) {
  const buckets = new Map<number, { utc: number; detected: number; removed: number; reset: number }>();
  for (const entry of entries) {
    const kind = detectionEventKind(entry);
    const rx = Date.parse(entry.rx_utc);
    if (!kind || !Number.isFinite(rx)) continue;
    const utc = Math.floor(rx / 60000) * 60000;
    const bucket = buckets.get(utc) ?? { utc, detected: 0, removed: 0, reset: 0 };
    bucket[kind] += 1;
    buckets.set(utc, bucket);
  }
  return [...buckets.values()].sort((a, b) => a.utc - b.utc);
}

export function obstacleGeometry(obstacle: Obstacle, rover: Position | null) {
  if (!rover) return null;
  const dx = obstacle.x_m - rover.x_m;
  const dy = obstacle.y_m - rover.y_m;
  const range = Math.hypot(dx, dy);
  const bearing = ((Math.atan2(dy, dx) * 180 / Math.PI) + 360) % 360;
  const relativeBearing = ((bearing - rover.heading_deg + 540) % 360) - 180;
  return { dx, dy, range, bearing, relativeBearing, envelopeRadius: obstacle.radius_m + obstacle.uncertainty_m };
}
