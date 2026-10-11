import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { detectionActivity, detectionEventKind, eventsForObstacle, obstacleGeometry } from "../detection-summary.ts";
import type { LogEntry } from "../mission-state";
import type { Obstacle, Position } from "../types";

const entry = (key: string, rx_utc = "2026-10-11T00:01:02Z"): LogEntry => ({ key, rx_utc, t_obs_s: 99999, category: "Detection", text: key, tone: "info" });

describe("detection dashboard summaries", () => {
  it("classifies only known detection/reset events", () => {
    assert.equal(detectionEventKind(entry("obstacle.detected:trk-1")), "detected");
    assert.equal(detectionEventKind(entry("obstacle.removed:trk-1:rx")), "removed");
    assert.equal(detectionEventKind(entry("map.reset:1:rx")), "reset");
    assert.equal(detectionEventKind(entry("health:STOP")), null);
  });
  it("uses UTC reception buckets, sorts late arrivals and ignores invalid times", () => {
    const rows = detectionActivity([entry("obstacle.detected:2"), entry("obstacle.removed:2:rx"), entry("map.reset:1", "2026-10-11T00:00:59Z"), entry("obstacle.detected:3", "invalid")]);
    assert.deepEqual(rows, [
      { utc: Date.parse("2026-10-11T00:00:00Z"), detected: 0, removed: 0, reset: 1 },
      { utc: Date.parse("2026-10-11T00:01:00Z"), detected: 1, removed: 1, reset: 0 },
    ]);
  });
  it("never confuses track IDs with shared prefixes", () => {
    assert.equal(eventsForObstacle([entry("obstacle.detected:trk-1"), entry("obstacle.detected:trk-10"), entry("obstacle.removed:trk-10:rx"), entry("obstacle.removed:trk-1:rx")], "trk-1").length, 2);
  });
  it("does not invent geometry without a received rover pose", () => {
    const o = { x_m: 3, y_m: 4, radius_m: 0.2, uncertainty_m: 0.1 } as Obstacle;
    assert.equal(obstacleGeometry(o, null), null);
    const g = obstacleGeometry(o, { x_m: 0, y_m: 0, heading_deg: 90 } as Position)!;
    assert.equal(g.range, 5);
    assert.ok(Math.abs(g.relativeBearing + 36.86989765) < 0.0001);
    assert.ok(Math.abs(g.envelopeRadius - 0.3) < 0.0001);
  });
});
