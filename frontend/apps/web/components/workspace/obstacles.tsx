"use client";

import { MousePointerClick } from "lucide-react";
import { fmtUtc, typeName } from "@/components/mission/shared";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import type { TrackedObstacle } from "@/lib/ground/mission-state";
import type { Position } from "@/lib/ground/types";
import { cn } from "@/lib/utils";
import { Empty, Facts, Meter, type Tone } from "./primitives";

export const typeTone: Record<string, Tone> = { rock: "ok", obstacle: "warn", negative_hazard: "bad" };
export const statusTone: Record<string, Tone> = { confirmed: "ok", observed: "signal", lost: "idle" };

export const provenance = (o: TrackedObstacle) =>
  o.source === "yolo+depth" ? "YOLO detection fused with depth" : "Depth evidence only (no YOLO class)";

export const rangeFrom = (o: TrackedObstacle, p: Position | null) =>
  p ? Math.hypot(o.x_m - p.x_m, o.y_m - p.y_m) : null;

export function TypeTag({ type }: { type: string }) {
  return <span className={cn("cw-type", `tone-${typeTone[type] ?? "warn"}`)}>{typeName[type] ?? type}</span>;
}

export function ObstacleDetail({
  o,
  rover,
  groundNow,
}: {
  o: TrackedObstacle | null | undefined;
  rover: Position | null;
  groundNow: number;
}) {
  if (!o)
    return (
      <Empty icon={MousePointerClick} title="No obstacle selected">
        Pick one on the map or in the list to see its confidence, uncertainty and provenance.
      </Empty>
    );
  const range = rangeFrom(o, rover);
  const rx = o.last_rx_utc ? Date.parse(o.last_rx_utc) : null;
  return (
    <div className="cw-detail">
      <div className="cw-detail-head">
        <TypeTag type={o.type} />
        <p className="cw-detail-id cw-mono">{o.obstacle_id}</p>
        <span className={cn("cw-status", `tone-${statusTone[o.status] ?? "idle"}`)}>{o.status}</span>
      </div>
      <div className="cw-detail-meter">
        <div>
          <span>Confidence</span>
          <strong className="cw-mono">{(o.confidence * 100).toFixed(0)}%</strong>
        </div>
        <Meter value={o.confidence} tone={o.confidence >= 0.6 ? "ok" : o.confidence >= 0.35 ? "warn" : "bad"} label="Confidence" />
      </div>
      <Facts
        items={[
          [
            "Position uncertainty",
            <span key="u" className="cw-mono">
              ±{o.uncertainty_m.toFixed(2)} m{o.uncertainty_saturated ? " (saturated)" : ""}
            </span>,
          ],
          ["Radius", <span key="r" className="cw-mono">{o.radius_m.toFixed(2)} m</span>],
          ["Position (odom)", <span key="p" className="cw-mono">{o.x_m.toFixed(2)}, {o.y_m.toFixed(2)} m</span>],
          ["Range from rover", <span key="g" className="cw-mono">{range === null ? "—" : `${range.toFixed(1)} m`}</span>],
          ["Provenance", provenance(o)],
          ["Track", o.persistent_id ? "persistent track id" : "single observation id"],
          ["Observed at", <span key="o" className="cw-mono">t {formatObsTime(o.obs_timestamp_s)} sim</span>],
          [
            "Last received",
            <span key="l" className="cw-mono">
              {rx ? `${fmtUtc(rx)} UTC · ${formatAge((groundNow - rx) / 1000)} ago` : "—"}
            </span>,
          ],
        ]}
      />
    </div>
  );
}
