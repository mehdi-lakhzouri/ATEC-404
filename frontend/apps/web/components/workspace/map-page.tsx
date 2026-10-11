"use client";

import { useMemo, useState } from "react";
import { LocateFixed, Plus, Timer, WifiOff, X } from "lucide-react";
import { LunarMap } from "@/components/mission/lunar-map";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { cn } from "@/lib/utils";
import { ObstacleDetail, TypeTag, rangeFrom } from "./obstacles";
import { Empty, Facts, Page, PageHeader, Section, useMissionGate } from "./primitives";

export function MapLegend() {
  const [open, setOpen] = useState(true);
  return (
    <div className={cn("cw-legend", !open && "is-collapsed")}>
      <button onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        Legend {open ? <X aria-hidden="true" /> : <Plus aria-hidden="true" />}
      </button>
      {open && (
        <ul>
          <li><span className="legend-rover" />Rover (last pose)</li>
          <li><span className="legend-trail" />Received path</li>
          <li><span className="legend-badge rock" />Rock</li>
          <li><span className="legend-badge hazard" />Negative hazard</li>
          <li><span className="legend-badge obstacle" />Depth-only obstacle</li>
          <li><span className="legend-ring" />Uncertainty</li>
          <li><span className="legend-explored" />Explored (camera)</li>
          <li><span className="legend-area" />Traversed (free)</li>
          <li><span className="legend-unknown" />Unknown</li>
        </ul>
      )}
    </div>
  );
}

export function MapPage() {
  const m = useMission();
  const s = m.state;
  const gate = useMissionGate();
  const [selected, setSelected] = useState<string | null>(null);
  const latest = m.latest;
  const obstacles = useMemo(() => Object.values(s?.obstacles ?? {}), [s?.obstacles]);
  const byRange = useMemo(
    () => [...obstacles].sort((a, b) => (rangeFrom(a, latest) ?? 0) - (rangeFrom(b, latest) ?? 0)),
    [obstacles, latest],
  );
  const sel = selected ? s?.obstacles[selected] : null;

  return (
    <Page id="map">
      <PageHeader
        title="Live map"
        description="Received odometry and the reconciled obstacle picture, in the rover's odom frame. Drag to pan, scroll to zoom."
      />
      {gate ?? (
        <div className="cw-grid cw-map-layout">
          <section className="cw-surface is-flush span-9 cw-map" id="surface">
            {s && (s.trajectory.length > 0 || obstacles.length > 0) ? (
              <LunarMap
                trajectory={s.trajectory}
                obstacles={obstacles}
                initialMode="fit"
                interactive
                stale={m.stale}
                selectedId={selected}
                onSelect={setSelected}
                label={`Lunar map: ${s.trajectory.length} received poses, ${obstacles.length} obstacles`}
              />
            ) : (
              <Empty icon={LocateFixed} title={m.stream.loading ? "Loading the mission picture…" : "No pose received yet"}>
                The rover appears with the first POSE packet that reaches the ground.
              </Empty>
            )}
            {m.stale && s && s.trajectory.length > 0 && (
              <p className={cn("cw-stale", m.fresh === "DISCONNECTED" && "is-disconnected")} role="status">
                {m.fresh === "DISCONNECTED" ? <WifiOff aria-hidden="true" /> : <Timer aria-hidden="true" />}
                {m.fresh === "DISCONNECTED"
                  ? "Ground stream disconnected — last received picture shown"
                  : `Stale — no telemetry for ${formatAge(m.ageS)}. Last received picture kept.`}
              </p>
            )}
            <MapLegend />
          </section>

          <aside className="span-3 cw-stack">
            <Section title="Rover">
              <Facts
                items={[
                  ["Position", latest ? <span className="cw-mono">{latest.x_m.toFixed(2)}, {latest.y_m.toFixed(2)} m</span> : "—"],
                  ["Heading", latest ? <span className="cw-mono">{latest.heading_deg.toFixed(1)}°</span> : "—"],
                  ["Uncertainty", latest ? <span className="cw-mono">±{latest.sigma_xy_m.toFixed(2)} m</span> : "—"],
                  ["Observed", latest ? <span className="cw-mono">t {formatObsTime(latest.t_obs_s)}</span> : "—"],
                  ["Data age", <span key="a" className="cw-mono">{formatAge(m.ageS)}</span>],
                ]}
              />
            </Section>
            <Section title="Selected obstacle" aside={sel ? <button className="cw-ghost" onClick={() => setSelected(null)}>Clear</button> : null}>
              <ObstacleDetail o={sel} rover={latest} groundNow={m.groundNow} />
            </Section>
            <Section title="Nearest obstacles" aside={<span className="cw-mono">{obstacles.length}</span>} flush>
              {byRange.length ? (
                <ul className="cw-list">
                  {byRange.map((o) => {
                    const r = rangeFrom(o, latest);
                    return (
                      <li key={o.obstacle_id}>
                        <button
                          className={cn(selected === o.obstacle_id && "is-selected", o.status === "lost" && "is-lost")}
                          onClick={() => setSelected(selected === o.obstacle_id ? null : o.obstacle_id)}
                          aria-pressed={selected === o.obstacle_id}
                        >
                          <TypeTag type={o.type} />
                          <span className="cw-mono">{o.obstacle_id}</span>
                          <span className="cw-mono cw-list-end">{r === null ? "—" : `${r.toFixed(1)} m`}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="cw-footnote">No obstacle reported yet.</p>
              )}
            </Section>
          </aside>
        </div>
      )}
    </Page>
  );
}
