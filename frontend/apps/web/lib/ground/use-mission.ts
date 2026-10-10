"use client";

// Live mission stream: REST initial load, then the WebSocket (snapshot + events) with reconnect and resync.
// Events are buffered and applied in batches so a burst of telemetry costs one render, not one per packet.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, describeError, getAllPages, groundApi } from "./api";
import { GROUND_WS } from "./config";
import type { ConnectionState } from "./freshness";
import {
  applyEvents,
  hydrate,
  initialMissionState,
  latestPosition,
  mergePositions,
  type MissionState,
} from "./mission-state";
import type { GroundEvent, Position } from "./types";

export const FLUSH_MS = 200;
export const MAX_INITIAL_POSITIONS = 5000;
const BACKOFF_MS = [500, 1000, 2000, 4000, 8000, 10000];

export type MissionStream = {
  state: MissionState | null;
  connection: ConnectionState;
  loading: boolean;
  error: string | null;
  reconnects: number;
  retryAtMs: number | null;
  clockOffsetMs: number; // ground clock - browser clock (UTC), from server timestamps
  reload: () => void;
};

/** Most recent `max` poses in ascending observation time (paginated, newest first, then reversed). */
async function latestPositions(missionId: string, max: number, signal: AbortSignal): Promise<Position[]> {
  const { items } = await getAllPages<Position>(
    `/missions/${encodeURIComponent(missionId)}/positions`,
    { order: "desc" },
    { max, signal },
  );
  return items.reverse();
}

export function useMissionStream(missionId: string | null): MissionStream {
  const [state, setState] = useState<MissionState | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("closed");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnects, setReconnects] = useState(0);
  const [retryAtMs, setRetryAtMs] = useState<number | null>(null);
  const [clockOffsetMs, setClockOffsetMs] = useState(0);
  const [reloadKey, setReloadKey] = useState(0);
  const stateRef = useRef<MissionState | null>(null);
  stateRef.current = state;

  const reload = useCallback(() => setReloadKey((k) => k + 1), []);

  useEffect(() => {
    if (!missionId) {
      setState(null);
      setConnection("closed");
      setLoading(false);
      setError(null);
      return;
    }
    const abort = new AbortController();
    let ws: WebSocket | null = null;
    let disposed = false;
    let attempt = 0;
    let everOpened = false;
    let retryTimer: number | undefined;
    let buffer: GroundEvent[] = [];
    let offset = 0;

    // Mission switch: start from an empty picture of THIS mission only.
    setState(initialMissionState(missionId));
    setLoading(true);
    setError(null);
    setReconnects(0);
    setRetryAtMs(null);

    const update = (fn: (s: MissionState) => MissionState) =>
      setState((s) => (s && s.missionId === missionId && !disposed ? fn(s) : s));

    const flush = window.setInterval(() => {
      if (!buffer.length) return;
      const batch = buffer;
      buffer = [];
      update((s) => applyEvents(s, batch));
      setClockOffsetMs(offset);
    }, FLUSH_MS);

    const backfill = async () => {
      // After a reconnect: poses persisted while we were away (incl. store-and-forward deliveries).
      const last = stateRef.current ? latestPosition(stateRef.current) : null;
      try {
        const { items } = await groundApi.allPositions(
          missionId,
          last ? { since_t_obs_s: last.t_obs_s } : {},
          { max: MAX_INITIAL_POSITIONS, signal: abort.signal },
        );
        update((s) => ({ ...s, trajectory: mergePositions(s.trajectory, items) }));
      } catch (e) {
        if (!abort.signal.aborted) setError(`resync: ${describeError(e)}`);
      }
    };

    const connect = () => {
      if (disposed) return;
      setConnection(everOpened ? "reconnecting" : "connecting");
      setRetryAtMs(null);
      const socket = new WebSocket(`${GROUND_WS}/ws/v1/missions/${encodeURIComponent(missionId)}`);
      ws = socket;
      socket.onopen = () => {
        if (disposed) return;
        attempt = 0;
        setConnection("open");
        if (everOpened) {
          setReconnects((n) => n + 1);
          void backfill();
        }
        everOpened = true;
      };
      socket.onmessage = (message) => {
        if (disposed) return;
        let ev: GroundEvent;
        try {
          ev = JSON.parse(String(message.data)) as GroundEvent;
        } catch {
          return;
        }
        if (!ev || typeof ev.type !== "string" || ev.mission_id !== missionId) return; // never mix missions
        if (ev.type === "link.status" || ev.type === "mission.snapshot") {
          const server = Date.parse(ev.rx_utc);
          if (!Number.isNaN(server)) offset = server - Date.now();
        }
        buffer.push(ev);
      };
      socket.onclose = () => {
        if (disposed || ws !== socket) return;
        ws = null;
        const delay = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)]!;
        attempt += 1;
        setConnection("reconnecting");
        setRetryAtMs(Date.now() + delay);
        retryTimer = window.setTimeout(connect, delay);
      };
      socket.onerror = () => {
        // onclose follows and schedules the retry
      };
    };

    const onOnline = () => {
      if (ws || disposed) return;
      window.clearTimeout(retryTimer);
      attempt = 0;
      connect();
    };
    window.addEventListener("online", onOnline);

    (async () => {
      try {
        const [mission, positions, obstacles, link, events] = await Promise.all([
          groundApi.mission(missionId, abort.signal),
          latestPositions(missionId, MAX_INITIAL_POSITIONS, abort.signal),
          groundApi.obstacles(missionId, { present: true }, { signal: abort.signal }),
          groundApi.link(missionId, { limit: 120 }, abort.signal),
          groundApi.events(missionId, { limit: 100 }, abort.signal),
        ]);
        if (disposed) return;
        const generated = Date.parse(mission.meta.generated_utc);
        if (!Number.isNaN(generated)) {
          offset = generated - Date.now();
          setClockOffsetMs(offset);
        }
        update((s) =>
          hydrate(s, {
            mission: mission.data,
            positions,
            obstacles: obstacles.items,
            link: link.data,
            events: events.data,
          }),
        );
      } catch (e) {
        if (abort.signal.aborted) return;
        setError(describeError(e));
        if (e instanceof ApiError && e.status === 404) {
          setLoading(false); // unknown mission: nothing to stream
          return;
        }
        // otherwise the live stream may still work (e.g. database down): keep going
      } finally {
        if (!disposed) setLoading(false);
      }
      connect();
    })();

    return () => {
      disposed = true;
      abort.abort();
      window.clearInterval(flush);
      window.clearTimeout(retryTimer);
      window.removeEventListener("online", onOnline);
      buffer = [];
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      ws = null;
      setConnection("closed");
    };
  }, [missionId, reloadKey]);

  // Mission metadata (status, counters) refresh: cheap, and catches idle/active transitions.
  useEffect(() => {
    if (!missionId) return;
    const abort = new AbortController();
    const timer = window.setInterval(async () => {
      try {
        const m = await groundApi.mission(missionId, abort.signal);
        setState((s) =>
          s && s.missionId === missionId ? { ...s, mission: { ...s.mission, ...m.data } } : s,
        );
      } catch {
        // connection problems are surfaced by the stream itself
      }
    }, 5000);
    return () => {
      abort.abort();
      window.clearInterval(timer);
    };
  }, [missionId]);

  return { state, connection, loading, error, reconnects, retryAtMs, clockOffsetMs, reload };
}

/** Re-renders every `periodMs` with the current time (freshness ages). */
export function useNow(periodMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), periodMs);
    return () => window.clearInterval(timer);
  }, [periodMs]);
  return now;
}
