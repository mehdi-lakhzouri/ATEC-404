// Minimal Next.js (React client component) integration with the Artemis-X ground station API v1 (R5.5).
// Drop into app/mission/[id]/LiveMission.tsx of the future dashboard; no extra dependency needed.
"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_GROUND_API ?? "http://127.0.0.1:8000";
const WS = API.replace(/^http/, "ws");

// Every REST response: { data, meta: { api_version, request_id, generated_utc, pagination } }
type Envelope<T> = { data: T; meta: { api_version: string; request_id: string; pagination: null | { total: number; next_offset: number | null } } };
// Every WebSocket event: { type, api_version, mission_id, t_obs_s (simulation s) | null, rx_utc (UTC ISO), data }
type GroundEvent = { type: string; mission_id: string; t_obs_s: number | null; rx_utc: string; data: any };
type Obstacle = { obstacle_id: string; type: string; x_m: number; y_m: number; radius_m: number; confidence: number; uncertainty_m: number; source: string; status: string };

export async function getJSON<T>(path: string): Promise<Envelope<T>> {
  const r = await fetch(`${API}/api/v1${path}`, { cache: "no-store" });
  const body = await r.json();
  if (!r.ok) throw new Error(`${body.error?.code}: ${body.error?.message}`);  // errors: { error: {code, message}, meta }
  return body;
}

export function LiveMission({ missionId }: { missionId: string }) {
  const [pose, setPose] = useState<any>(null);
  const [obstacles, setObstacles] = useState<Record<string, Obstacle>>({});
  const [link, setLink] = useState<any>(null);
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${WS}/ws/v1/missions/${missionId}`);
      ws.onopen = () => { retry.current = 0; };
      ws.onmessage = (m) => {
        const ev: GroundEvent = JSON.parse(m.data);
        switch (ev.type) {
          case "mission.snapshot":                       // first message after every (re)connect: full current state
            setPose(ev.data.latest_position);
            setObstacles(Object.fromEntries((ev.data.obstacles ?? []).map((o: Obstacle) => [o.obstacle_id, o])));
            break;
          case "rover.position": setPose(ev.data); break;
          case "obstacle.detected":
          case "obstacle.updated": setObstacles((s) => ({ ...s, [ev.data.obstacle_id]: ev.data })); break;
          case "obstacle.removed": setObstacles((s) => { const n = { ...s }; delete n[ev.data.obstacle_id]; return n; }); break;
          case "map.reset": setObstacles({}); break;
          case "link.status": setLink(ev.data); break;
        }
      };
      ws.onclose = () => { if (!closed) setTimeout(connect, Math.min(10000, 500 * 2 ** retry.current++)); };  // backoff
    };
    connect();
    return () => { closed = true; ws?.close(); };
  }, [missionId]);

  return (
    <pre>{JSON.stringify({ pose, obstacles: Object.keys(obstacles).length, link_up: link?.link_up }, null, 2)}</pre>
  );
}

// Network control (local dev only): await fetch(`${API}/api/v1/network/preset`, { method: "POST",
//   headers: { "Content-Type": "application/json" }, body: JSON.stringify({ preset: "BLACKOUT" }) });
