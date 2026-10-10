// Data freshness from the UTC wall clock only (ground reception time), never from rover observation time.

export type Freshness = "LIVE" | "STALE" | "DISCONNECTED" | "NO_DATA";

export const STALE_AFTER_S = 3;

export type ConnectionState = "connecting" | "open" | "reconnecting" | "closed";

/**
 * LIVE: the dashboard is connected and a rover packet reached the ground within `staleAfterS`.
 * STALE: connected, but the last reception is older (blackout, rover stopped, link loss); the last picture is kept.
 * DISCONNECTED: the dashboard has no live stream from the ground station (whatever the age of the data).
 * NO_DATA: connected but nothing was ever received for this mission.
 */
export function freshness(
  connection: ConnectionState,
  lastRxMs: number | null,
  nowMs: number,
  staleAfterS = STALE_AFTER_S,
): Freshness {
  if (connection !== "open") return "DISCONNECTED";
  if (lastRxMs === null) return "NO_DATA";
  return (nowMs - lastRxMs) / 1000 <= staleAfterS ? "LIVE" : "STALE";
}

export function ageSeconds(lastRxMs: number | null, nowMs: number): number | null {
  return lastRxMs === null ? null : Math.max(0, (nowMs - lastRxMs) / 1000);
}

export function formatAge(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 1) return "<1 s";
  if (seconds < 120) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  if (seconds < 7200) return `${Math.floor(seconds / 60)} min ${Math.floor(seconds % 60)} s`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} h`;
  return `${Math.floor(seconds / 86400)} d`;
}

/** Simulation seconds as HH:MM:SS.s (rover observation clock). */
export function formatObsTime(t: number | null | undefined): string {
  if (t === null || t === undefined || !Number.isFinite(t)) return "—";
  const h = Math.floor(t / 3600);
  const m = Math.floor(t / 60) % 60;
  const s = t % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}
