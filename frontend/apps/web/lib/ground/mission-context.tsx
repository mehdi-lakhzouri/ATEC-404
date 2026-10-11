"use client";

// One mission session for the whole console: selection (?mission= / follow latest), the single ground-station
// WebSocket, network control, the shared clock and the workspace preference. Mounted in the root layout so that page
// navigation (clean workspace) never opens a second WebSocket or loses the selected mission.
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { ageSeconds, formatAge, freshness, type Freshness } from "./freshness";
import { latestPosition, type MissionState } from "./mission-state";
import type { Position } from "./types";
import { useMissionStream, useNow, type MissionStream } from "./use-mission";
import { useMissionList, useNetworkControl, type MissionList, type NetworkControl } from "./use-network";

// ---- workspace preference (localStorage, hydration-safe) -----------------------------------------------------------
export type Workspace = "clean" | "unified";
export const WORKSPACE_KEY = "artemis.workspace";
const WORKSPACE_EVENT = "artemis-workspace";

function readWorkspace(): Workspace {
  try {
    return window.localStorage.getItem(WORKSPACE_KEY) === "unified" ? "unified" : "clean";
  } catch {
    return "clean"; // storage blocked: default mode
  }
}

function subscribeWorkspace(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener(WORKSPACE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(WORKSPACE_EVENT, onChange);
  };
}

/** null during server render and hydration, then the stored mode: nothing mode-specific renders until it is known. */
export function useWorkspacePreference(): [Workspace | null, (w: Workspace) => void] {
  const value = useSyncExternalStore<Workspace | null>(subscribeWorkspace, readWorkspace, () => null);
  const set = useCallback((w: Workspace) => {
    try {
      window.localStorage.setItem(WORKSPACE_KEY, w);
    } catch {
      // storage blocked: the switch still applies for this page view
    }
    window.dispatchEvent(new Event(WORKSPACE_EVENT));
  }, []);
  return [value, set];
}

// ---- mission session ----------------------------------------------------------------------------------------------
export type MissionSession = {
  list: MissionList;
  stream: MissionStream;
  network: NetworkControl;
  now: number;
  groundNow: number;
  missionId: string | null;
  followLatest: boolean;
  selectMission: (id: string) => void;
  followNewest: () => void;
  stopFollowing: () => void;
  state: MissionState | null;
  latest: Position | null;
  fresh: Freshness;
  freshLabel: string;
  ageS: number | null;
  stale: boolean;
  hasMission: boolean;
  unreachable: boolean;
  motionEnabled: boolean;
  setMotionEnabled: (v: boolean) => void;
  workspace: Workspace | null;
  setWorkspace: (w: Workspace) => void;
  /** Link target that keeps a pinned mission across pages. */
  href: (path: string) => string;
};

export const freshnessLabel: Record<Freshness, string> = {
  LIVE: "Live",
  STALE: "Stale",
  DISCONNECTED: "Disconnected",
  NO_DATA: "No data",
};

const Ctx = createContext<MissionSession | null>(null);

function setMissionParam(id: string | null) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("mission", id);
  else url.searchParams.delete("mission");
  window.history.replaceState(window.history.state, "", url);
}

export function MissionProvider({ children }: { children: ReactNode }) {
  const [missionId, setMissionId] = useState<string | null>(null);
  const [followLatest, setFollowLatest] = useState(true);
  const [motionEnabled, setMotionEnabled] = useState(true);
  const [workspace, setWorkspace] = useWorkspacePreference();

  const list = useMissionList();
  const stream = useMissionStream(missionId);
  const network = useNetworkControl();
  const now = useNow(1000);

  // ?mission=<id> pins a mission (any page); otherwise follow the newest active (else most recent) mission.
  useEffect(() => {
    const pinned = new URLSearchParams(window.location.search).get("mission");
    if (pinned) {
      setMissionId(pinned);
      setFollowLatest(false);
    }
  }, []);
  useEffect(() => {
    if (!followLatest || !list.missions.length) return;
    const newest = list.missions.find((m) => m.status === "active") ?? list.missions[0]!;
    if (newest.mission_id !== missionId) setMissionId(newest.mission_id);
  }, [followLatest, list.missions, missionId]);

  const selectMission = useCallback((id: string) => {
    setFollowLatest(false);
    setMissionId(id);
    setMissionParam(id);
  }, []);
  const followNewest = useCallback(() => {
    setFollowLatest(true);
    setMissionParam(null);
  }, []);
  const stopFollowing = useCallback(() => setFollowLatest(false), []);

  const state = stream.state && stream.state.missionId === missionId ? stream.state : null;
  const groundNow = now + stream.clockOffsetMs;
  const fresh: Freshness = missionId
    ? freshness(stream.connection, state?.lastRxMs ?? null, groundNow)
    : list.error
      ? "DISCONNECTED"
      : "NO_DATA";
  const ageS = ageSeconds(state?.lastRxMs ?? null, groundNow);
  const pinned = !followLatest && missionId ? missionId : null;
  const href = useCallback(
    (path: string) => (pinned ? `${path}?mission=${encodeURIComponent(pinned)}` : path),
    [pinned],
  );

  const value = useMemo<MissionSession>(
    () => ({
      list,
      stream,
      network,
      now,
      groundNow,
      missionId,
      followLatest,
      selectMission,
      followNewest,
      stopFollowing,
      state,
      latest: state ? latestPosition(state) : null,
      fresh,
      freshLabel: `${freshnessLabel[fresh]} · last ground reception ${formatAge(ageS)} ago (UTC)`,
      ageS,
      stale: fresh === "STALE" || fresh === "DISCONNECTED",
      hasMission: missionId !== null,
      unreachable: missionId === null && list.error !== null,
      motionEnabled,
      setMotionEnabled,
      workspace,
      setWorkspace,
      href,
    }),
    [
      list,
      stream,
      network,
      now,
      groundNow,
      missionId,
      followLatest,
      selectMission,
      followNewest,
      stopFollowing,
      state,
      fresh,
      ageS,
      motionEnabled,
      workspace,
      setWorkspace,
      href,
    ],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useMission(): MissionSession {
  const v = useContext(Ctx);
  if (!v) throw new Error("useMission() outside <MissionProvider>");
  return v;
}
