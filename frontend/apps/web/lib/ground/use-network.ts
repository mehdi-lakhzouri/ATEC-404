"use client";

// Lunar link emulator control (R5.3) through the ground station. A preset is only reported as applied once a
// subsequent GET /network/status confirms it: the UI never assumes a network state.
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, describeError, groundApi } from "./api";
import type { Mission, NetworkStatus, Preset } from "./types";

export const NETWORK_POLL_MS = 2000;

export type PresetRequestState =
  | { phase: "idle" }
  | { phase: "applying"; preset: Preset }
  | { phase: "confirmed"; preset: Preset; atMs: number }
  | { phase: "failed"; preset: Preset; message: string };

export type NetworkControl = {
  status: NetworkStatus | null;
  error: string | null;
  available: boolean;
  request: PresetRequestState;
  blackoutSinceMs: number | null; // browser time when a BLACKOUT was first seen
  restoredAtMs: number | null; // browser time when the link left BLACKOUT
  lastBlackoutS: number | null; // duration of the last completed blackout, from the emulator's blackout_total_s
  apply: (preset: Preset) => Promise<void>;
};

export function useNetworkControl(): NetworkControl {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [request, setRequest] = useState<PresetRequestState>({ phase: "idle" });
  const [blackoutSinceMs, setBlackoutSinceMs] = useState<number | null>(null);
  const [restoredAtMs, setRestoredAtMs] = useState<number | null>(null);
  const [lastBlackoutS, setLastBlackoutS] = useState<number | null>(null);
  const prev = useRef<{ state: string; total: number } | null>(null);

  const accept = useCallback((st: NetworkStatus) => {
    const before = prev.current;
    if (st.link_state === "BLACKOUT") {
      if (before?.state !== "BLACKOUT") {
        setBlackoutSinceMs(Date.now());
        setRestoredAtMs(null);
        // emulator blackout total before this blackout (unknown if the page opened during the blackout)
        prev.current = { state: "BLACKOUT", total: before ? before.total : Number.NaN };
      }
    } else {
      if (before?.state === "BLACKOUT") {
        setBlackoutSinceMs(null);
        setRestoredAtMs(Date.now());
        setLastBlackoutS(Number.isNaN(before.total) ? null : Math.max(0, st.blackout_total_s - before.total));
      }
      prev.current = { state: st.link_state, total: st.blackout_total_s };
    }
    setStatus(st);
    setError(null);
  }, []);

  const poll = useCallback(
    async (signal?: AbortSignal) => {
      try {
        accept((await groundApi.networkStatus(signal)).data);
      } catch (e) {
        if (signal?.aborted) return;
        setError(describeError(e));
        if (e instanceof ApiError && (e.status === 503 || e.status === 0)) setStatus(null);
      }
    },
    [accept],
  );

  useEffect(() => {
    const abort = new AbortController();
    void poll(abort.signal);
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void poll(abort.signal);
    }, NETWORK_POLL_MS);
    return () => {
      abort.abort();
      window.clearInterval(timer);
    };
  }, [poll]);

  const apply = useCallback(
    async (preset: Preset) => {
      setRequest({ phase: "applying", preset });
      try {
        await groundApi.applyPreset(preset);
        const confirmed = (await groundApi.networkStatus()).data; // confirmation comes from a fresh GET
        accept(confirmed);
        if (confirmed.preset !== preset)
          throw new Error(`emulator reports ${confirmed.preset} after requesting ${preset}`);
        setRequest({ phase: "confirmed", preset, atMs: Date.now() });
      } catch (e) {
        setRequest({ phase: "failed", preset, message: describeError(e) });
      }
    },
    [accept],
  );

  return {
    status,
    error,
    available: status !== null,
    request,
    blackoutSinceMs,
    restoredAtMs,
    lastBlackoutS,
    apply,
  };
}

export type MissionList = {
  missions: Mission[];
  total: number;
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  loadMore: () => void;
  refresh: () => void;
};

export const MISSIONS_PAGE = 10;

/** Mission history (persisted in MongoDB): paginated, refreshed periodically. */
export function useMissionList(): MissionList {
  const [missions, setMissions] = useState<Mission[]>([]);
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(MISSIONS_PAGE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const abort = new AbortController();
    groundApi
      .missions({ limit, offset: 0 }, abort.signal)
      .then((env) => {
        setMissions(env.data);
        setTotal(env.meta.pagination?.total ?? env.data.length);
        setError(null);
      })
      .catch((e: unknown) => {
        if (!abort.signal.aborted) setError(describeError(e));
      })
      .finally(() => {
        if (!abort.signal.aborted) setLoading(false);
      });
    return () => abort.abort();
  }, [limit, tick]);

  useEffect(() => {
    const timer = window.setInterval(() => setTick((t) => t + 1), 5000);
    return () => window.clearInterval(timer);
  }, []);

  return {
    missions,
    total,
    loading,
    error,
    hasMore: missions.length < total,
    loadMore: () => setLimit((l) => l + MISSIONS_PAGE),
    refresh: () => setTick((t) => t + 1),
  };
}
