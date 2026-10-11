"use client";

import { useEffect, useState } from "react";
import { describeError, groundApi } from "./api";
import type { MissionStats } from "./types";

/** Stats are tagged by mission so an old request can never populate a new inspector. */
export function useMissionStats(id: string | null) {
  const [loaded, setLoaded] = useState<{ id: string; data: MissionStats } | null>(null);
  const [failed, setFailed] = useState<{ id: string; message: string } | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!id) return;
    const abort = new AbortController();
    void groundApi.stats(id, abort.signal).then((env) => {
      if (abort.signal.aborted) return;
      setLoaded({ id, data: env.data });
      setFailed(null);
    }).catch((error: unknown) => {
      if (!abort.signal.aborted) setFailed({ id, message: describeError(error) });
    });
    return () => abort.abort();
  }, [id, tick]);
  useEffect(() => {
    if (!id) return;
    const timer = window.setInterval(() => document.visibilityState === "visible" && setTick((value) => value + 1), 5000);
    return () => window.clearInterval(timer);
  }, [id]);
  const data = loaded?.id === id ? loaded.data : null;
  const error = failed?.id === id ? failed.message : null;
  return { data, error, loading: Boolean(id && !data && !error), refresh: () => setTick((value) => value + 1) };
}
