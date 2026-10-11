import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { eventActivity, filterEvents, linkWindow } from "../dashboard-summary.ts";
import type { LinkSample, LogEntry } from "../mission-state";

const event = (key: string, rx_utc: string, tone: LogEntry["tone"] = "info"): LogEntry => ({ key, rx_utc, tone, t_obs_s: 999999, category: "System", text: "Rover health", detail: "stale corridor" });
const sample = (rx_ms: number): LinkSample => ({ rx_ms, packets_per_s: null, loss_pct: null, delay_p50_ms: null, preset: null, link_up: false });

describe("workspace dashboard summaries", () => {
  it("buckets retained severity events by UTC and ignores invalid clocks", () => {
    assert.deepEqual(eventActivity([event("late", "2026-10-10T12:01:59Z", "bad"), event("early", "2026-10-10T12:00:01Z", "good"), event("bad-clock", "invalid")]), [
      { utc: Date.parse("2026-10-10T12:00:00Z"), info: 0, good: 1, warn: 0, bad: 0 },
      { utc: Date.parse("2026-10-10T12:01:00Z"), info: 0, good: 0, warn: 0, bad: 1 },
    ]);
  });
  it("windows old missions relative to their last sample and keeps unknown metrics unknown", () => {
    const input = [sample(180000), sample(1000), sample(120000), sample(NaN)];
    assert.deepEqual(linkWindow(input, 60).map((entry) => entry.rx_ms), [120000, 180000]);
    assert.equal(linkWindow(input, 60)[0]!.loss_pct, null);
    assert.equal(input[0]!.rx_ms, 180000);
    assert.equal(linkWindow(input, null).length, 3);
  });
  it("combines event category, severity and searchable details without inventing entries", () => {
    const entries = [event("health:1", "2026-10-10T12:00:01Z", "bad"), { ...event("detected:2", "2026-10-10T12:00:02Z", "good"), category: "Detection" as const }];
    assert.equal(filterEvents(entries, "System", "bad", "STALE").length, 1);
    assert.equal(filterEvents(entries, "Detection", "bad", "").length, 0);
    assert.equal(filterEvents(entries, "all", "all", "detected:2")[0]!.key, "detected:2");
  });
});
