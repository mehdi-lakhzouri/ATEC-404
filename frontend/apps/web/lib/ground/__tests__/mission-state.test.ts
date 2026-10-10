// node --test (Node >= 24 strips the types). Run: pnpm --filter web test
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ageSeconds, freshness } from "../freshness.ts";
import {
  MAX_LOG,
  applyEvent,
  applyEvents,
  hydrate,
  initialMissionState,
  latestPosition,
  mergePositions,
} from "../mission-state.ts";
import { trajectorySegments } from "../trajectory.ts";
import type { GroundEvent, LinkStatus, Mission, Obstacle, Position } from "../types.ts";

const M = "1234";
const iso = (s: number) => new Date(Date.UTC(2026, 9, 10, 12, 0, 0) + s * 1000).toISOString();

const pose = (t: number, x = t, y = 0): Position => ({
  frame: "odom",
  x_m: x,
  y_m: y,
  yaw_rad: 0,
  heading_deg: 0,
  sigma_xy_m: 0.1,
  sigma_xy_saturated: false,
  t_obs_s: t,
  rx_utc: iso(t),
});

const obstacle = (id: string, extra: Partial<Obstacle> = {}): Obstacle => ({
  obstacle_id: id,
  type: "rock",
  frame: "odom",
  x_m: 1,
  y_m: 2,
  radius_m: 0.3,
  confidence: 0.9,
  uncertainty_m: 0.2,
  obs_timestamp_s: 10,
  source: "yolo+depth",
  status: "confirmed",
  persistent_id: true,
  ...extra,
});

function ev<T extends GroundEvent["type"]>(
  type: T,
  data: Extract<GroundEvent, { type: T }>["data"],
  t: number | null,
  rxS: number,
  mission = M,
): GroundEvent {
  return { type, api_version: "v1", mission_id: mission, t_obs_s: t, rx_utc: iso(rxS), data } as GroundEvent;
}

const posEv = (t: number, mission = M) => {
  const { t_obs_s: _t, rx_utc: _r, ...p } = pose(t);
  return ev("rover.position", p, t, t, mission);
};

const link = (over: Partial<LinkStatus> = {}, emu: Partial<NonNullable<LinkStatus["emulator"]>> = {}): LinkStatus => ({
  receiver: { valid: 10, missing: 0, duplicates: 0, reordered: 0 },
  packets_per_s: 2,
  last_rx_age_s: 0.2,
  link_up: true,
  emulator: {
    preset: "NORMAL",
    link_state: "UP",
    configured_loss: 0,
    delivery_ratio: 1,
    delay_ms: { n: 1, p50: 1, p95: 1, max: 1 },
    blackout_total_s: 0,
    received: 100,
    forwarded: 100,
    dropped_loss: 0,
    dropped_blackout: 0,
    dropped_queue: 0,
    throughput_Bps: 50,
    ...emu,
  },
  ...over,
});

describe("trajectory", () => {
  it("keeps ascending observation time and drops duplicate poses", () => {
    const t = mergePositions([], [pose(1), pose(2), pose(2), pose(3)]);
    assert.deepEqual(
      t.map((p) => p.t_obs_s),
      [1, 2, 3],
    );
  });

  it("places late (store-and-forward) poses by observation time", () => {
    const t = mergePositions([pose(1), pose(5)], [pose(3), pose(4), pose(1)]);
    assert.deepEqual(
      t.map((p) => p.t_obs_s),
      [1, 3, 4, 5],
    );
  });

  it("latest position is the newest observation, not the last received packet", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [posEv(10), posEv(4)]);
    assert.equal(latestPosition(s)?.t_obs_s, 10);
  });

  it("never bridges a reception gap with a line", () => {
    const pts = [pose(0), pose(0.5), pose(1), pose(9), pose(9.5)];
    const segs = trajectorySegments(pts);
    assert.equal(segs.length, 2);
    assert.deepEqual(
      segs.map((s) => s.map((p) => p.t_obs_s)),
      [
        [0, 0.5, 1],
        [9, 9.5],
      ],
    );
  });

  it("decimates long runs but keeps run end points", () => {
    const pts = Array.from({ length: 1000 }, (_, i) => pose(i * 0.5));
    const segs = trajectorySegments(pts, 3, 100);
    assert.equal(segs.length, 1);
    assert.ok(segs[0]!.length <= 101);
    assert.equal(segs[0]![0]!.t_obs_s, 0);
    assert.equal(segs[0]!.at(-1)!.t_obs_s, 499.5);
  });
});

describe("mission isolation", () => {
  it("ignores events of another mission", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [posEv(1, "9999"), ev("obstacle.detected", obstacle("trk-1"), 1, 1, "9999")]);
    assert.equal(s.trajectory.length, 0);
    assert.equal(Object.keys(s.obstacles).length, 0);
    assert.equal(s.ignoredForeign, 2);
  });
});

describe("obstacles", () => {
  it("detected twice is one obstacle and one log line", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [
      ev("obstacle.detected", obstacle("trk-1"), 1, 1),
      ev("obstacle.detected", obstacle("trk-1"), 1, 1),
      ev("obstacle.updated", obstacle("trk-1", { confidence: 0.95 }), 2, 2),
    ]);
    assert.deepEqual(Object.keys(s.obstacles), ["trk-1"]);
    assert.equal(s.obstacles["trk-1"]!.confidence, 0.95);
    assert.equal(s.log.filter((l) => l.category === "Detection").length, 1);
  });

  it("removes only on a reconciliation event", () => {
    let s = initialMissionState(M);
    s = applyEvent(s, ev("obstacle.detected", obstacle("trk-1"), 1, 1));
    s = applyEvents(s, [posEv(2), ev("link.status", link({ link_up: false }), null, 30)]);
    assert.ok("trk-1" in s.obstacles, "a stale link must not erase obstacles");
    s = applyEvent(s, ev("obstacle.removed", { obstacle_id: "trk-1", reason: "full_snapshot" }, 3, 31));
    assert.ok(!("trk-1" in s.obstacles));
  });

  it("map.reset clears the picture", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [
      ev("obstacle.detected", obstacle("a"), 1, 1),
      ev("obstacle.detected", obstacle("b"), 1, 1),
      ev("map.reset", { resets: 1, state: "DEGRADED" }, 5, 5),
    ]);
    assert.equal(Object.keys(s.obstacles).length, 0);
    assert.equal(s.mapResets, 1);
  });

  it("snapshot resync replaces the picture but keeps detections newer than the snapshot", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [
      ev("obstacle.detected", obstacle("old"), 1, 1),
      ev("obstacle.detected", obstacle("fresh"), 9, 9),
    ]);
    const mission = { mission_id: M, session: 1234, last_rx_utc: iso(5), created_utc: iso(0) } as Mission;
    s = applyEvent(
      s,
      ev(
        "mission.snapshot",
        { mission, latest_position: pose(5), obstacles: [obstacle("kept", { last_rx_utc: iso(4) })], source: "mongodb" },
        null,
        10,
      ),
    );
    assert.deepEqual(Object.keys(s.obstacles).sort(), ["fresh", "kept"]);
    assert.equal(s.snapshots, 1);
  });

  it("an unavailable-database snapshot does not wipe the picture", () => {
    let s = initialMissionState(M);
    s = applyEvent(s, ev("obstacle.detected", obstacle("a"), 1, 1));
    s = applyEvent(
      s,
      ev("mission.snapshot", { mission: null, latest_position: null, obstacles: [], source: "unavailable", error: "X" }, null, 2),
    );
    assert.ok("a" in s.obstacles);
    assert.equal(s.snapshotSource, "unavailable");
  });
});

describe("freshness (UTC wall clock only)", () => {
  it("LIVE / STALE / DISCONNECTED / NO_DATA", () => {
    const now = Date.parse(iso(100));
    assert.equal(freshness("open", Date.parse(iso(99)), now), "LIVE");
    assert.equal(freshness("open", Date.parse(iso(90)), now), "STALE");
    assert.equal(freshness("reconnecting", Date.parse(iso(99.9)), now), "DISCONNECTED");
    assert.equal(freshness("open", null, now), "NO_DATA");
    assert.equal(ageSeconds(Date.parse(iso(90)), now), 10);
  });

  it("observation time never refreshes data: only receptions do", () => {
    let s = initialMissionState(M);
    s = applyEvent(s, posEv(1));
    const rx = s.lastRxMs;
    s = applyEvent(s, ev("link.status", link(), null, 50)); // ground-side sample, not a rover packet
    assert.equal(s.lastRxMs, rx);
    s = applyEvent(s, ev("telemetry.received", { seq: 7, packet_type: "pose", seq_flag: "new" }, 60, 61));
    assert.equal(s.lastRxMs, Date.parse(iso(61)));
  });
});

describe("link samples", () => {
  it("window loss from emulator counters, blackout transitions logged", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [
      ev("link.status", link(), null, 0),
      ev(
        "link.status",
        link({ link_up: false }, { preset: "BLACKOUT", link_state: "BLACKOUT", received: 120, dropped_blackout: 20 }),
        null,
        2,
      ),
    ]);
    assert.equal(s.linkSamples.length, 2);
    assert.equal(s.linkSamples[1]!.loss_pct, 100);
    assert.ok(s.log.some((l) => l.text.includes("downlink lost")));
    assert.ok(s.log.some((l) => l.text.includes("NORMAL → BLACKOUT")));
  });

  it("replayed samples are ignored", () => {
    let s = initialMissionState(M);
    s = applyEvents(s, [ev("link.status", link(), null, 5), ev("link.status", link(), null, 5), ev("link.status", link(), null, 4)]);
    assert.equal(s.linkSamples.length, 1);
  });
});

describe("hydrate (REST initial load)", () => {
  it("builds trajectory, obstacles and stored events; log stays bounded", () => {
    const mission = {
      mission_id: M,
      session: 1234,
      status: "active",
      created_utc: iso(0),
      last_rx_utc: iso(30),
      first_t_obs_s: 0,
      last_t_obs_s: 30,
      last_rx_age_s: 1,
      counters: { packets: 10 },
      health: { state: "OK", reasons: [] },
      config: {},
    } satisfies Mission;
    const events = Array.from({ length: MAX_LOG + 50 }, (_, i) => ({
      seq: i,
      packet_type: "event",
      kind: "health",
      t_obs_s: i,
      rx_utc: iso(i),
      seq_flag: "new",
      frame: "odom",
      event: { event: "health", state: i % 2 ? "OK" : "DEGRADED", reasons: [] },
    }));
    const s = hydrate(initialMissionState(M), {
      mission,
      positions: [pose(1), pose(2)],
      obstacles: [obstacle("trk-1", { last_rx_utc: iso(3) })],
      link: [],
      events,
    });
    assert.equal(s.trajectory.length, 2);
    assert.ok("trk-1" in s.obstacles);
    assert.equal(s.lastRxMs, Date.parse(iso(30)));
    assert.ok(s.log.length <= MAX_LOG);
    assert.equal(s.health?.state, "OK");
  });
});
