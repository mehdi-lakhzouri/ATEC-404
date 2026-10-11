import type { LinkSample, LogEntry } from "./mission-state";

/** Bounded, retained events, grouped by ground reception time; never simulation time. */
export function eventActivity(entries: LogEntry[]) {
  const buckets = new Map<number, { utc: number; info: number; good: number; warn: number; bad: number }>();
  for (const event of entries) {
    const rx = Date.parse(event.rx_utc);
    if (!Number.isFinite(rx)) continue;
    const utc = Math.floor(rx / 60000) * 60000;
    const bucket = buckets.get(utc) ?? { utc, info: 0, good: 0, warn: 0, bad: 0 };
    bucket[event.tone]++;
    buckets.set(utc, bucket);
  }
  return [...buckets.values()].sort((a, b) => a.utc - b.utc);
}

/** Historical windows end at the last received sample, not at the browser's current wall clock. */
export function linkWindow(samples: LinkSample[], seconds: number | null) {
  const sorted = samples.filter((sample) => Number.isFinite(sample.rx_ms)).slice().sort((a, b) => a.rx_ms - b.rx_ms);
  if (seconds === null || !sorted.length) return sorted;
  const end = sorted[sorted.length - 1]!.rx_ms;
  return sorted.filter((sample) => sample.rx_ms >= end - seconds * 1000);
}

export function filterEvents(entries: LogEntry[], category: string, tone: string, query: string) {
  const needle = query.trim().toLowerCase();
  return entries.filter((entry) => (category === "all" || entry.category === category) && (tone === "all" || entry.tone === tone)
    && (!needle || `${entry.key} ${entry.text} ${entry.detail ?? ""} ${entry.category} ${entry.rx_utc}`.toLowerCase().includes(needle)));
}
