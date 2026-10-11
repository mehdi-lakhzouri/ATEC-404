"use client";

import { useMemo } from "react";
import { FileText, LocateFixed } from "lucide-react";
import { LunarMap } from "@/components/mission/lunar-map";
import { fmtUtc } from "@/components/mission/shared";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { trajectorySegments } from "@/lib/ground/trajectory";
import type { Position } from "@/lib/ground/types";
import { LogFeed } from "./logs";
import {
  Empty,
  Facts,
  Kpi,
  Page,
  PageHeader,
  Section,
  SignalRail,
  TextLink,
  healthTone,
  useMissionGate,
} from "./primitives";

/** Distance along received poses only: a reception gap is not counted (no invented path). */
export function receivedDistance(trajectory: Position[]): number {
  let d = 0;
  for (const seg of trajectorySegments(trajectory, undefined, Number.MAX_SAFE_INTEGER))
    for (let i = 1; i < seg.length; i++) d += Math.hypot(seg[i]!.x_m - seg[i - 1]!.x_m, seg[i]!.y_m - seg[i - 1]!.y_m);
  return d;
}

export function OverviewPage() {
  const m = useMission();
  const s = m.state;
  const obstacles = useMemo(() => Object.values(s?.obstacles ?? {}), [s?.obstacles]);
  const distance = useMemo(() => receivedDistance(s?.trajectory ?? []), [s?.trajectory]);
  const gate = useMissionGate();
  const rocks = obstacles.filter((o) => o.type === "rock").length;
  const hazards = obstacles.length - rocks;
  const health = s?.health;
  const delivery = s?.link?.emulator?.delivery_ratio;
  const counters = s?.mission?.counters ?? {};
  const latest = m.latest;

  return (
    <Page id="overview">
      <PageHeader
        title="Mission overview"
        description="What the ground station has received from the rover, and how fresh it is."
      />
      {gate ?? (
        <>
          <SignalRail />

          <div className="cw-grid cw-kpis">
            <Kpi
              label="Obstacles on the map"
              value={s ? obstacles.length : "—"}
              context={s ? `${rocks} rocks · ${hazards} hazards or obstacles` : "no mission loaded"}
              tone={hazards > 0 ? "warn" : "signal"}
            />
            <Kpi
              label="Rover health"
              value={health?.state ?? "—"}
              context={health?.reasons?.length ? health.reasons.join(", ").replaceAll("_", " ") : "no fault reported"}
              tone={healthTone(health?.state)}
            />
            <Kpi
              label="Distance along received path"
              value={s ? distance.toFixed(1) : "—"}
              unit="m"
              context={`${s?.trajectory.length ?? 0} poses received`}
            />
            <Kpi
              label="Link delivery"
              value={delivery === null || delivery === undefined ? "—" : (delivery * 100).toFixed(0)}
              unit={delivery === null || delivery === undefined ? undefined : "%"}
              context={s?.link?.emulator ? `emulator preset ${s.link.emulator.preset}` : "emulator not reporting"}
              tone={delivery === null || delivery === undefined ? "idle" : delivery > 0.9 ? "ok" : delivery > 0.6 ? "warn" : "bad"}
            />
          </div>

          <div className="cw-grid">
            <Section title="Trajectory" aside={<TextLink href="/map">Open live map</TextLink>} className="span-8" flush>
              <div className="cw-preview">
                {s && (s.trajectory.length > 0 || obstacles.length > 0) ? (
                  <LunarMap
                    trajectory={s.trajectory}
                    obstacles={obstacles}
                    stale={m.stale}
                    label={`Trajectory preview: ${s.trajectory.length} received poses, ${obstacles.length} obstacles`}
                  />
                ) : (
                  <Empty icon={LocateFixed} title="No pose received yet">
                    The track appears with the first POSE packet that reaches the ground.
                  </Empty>
                )}
              </div>
            </Section>
            <Section title="Rover" className="span-4">
              <Facts
                items={[
                  ["Position (odom)", latest ? <span className="cw-mono">{latest.x_m.toFixed(2)}, {latest.y_m.toFixed(2)} m</span> : "—"],
                  ["Heading", latest ? <span className="cw-mono">{latest.heading_deg.toFixed(1)}°</span> : "—"],
                  [
                    "Position uncertainty",
                    latest ? (
                      <span className="cw-mono">
                        ±{latest.sigma_xy_m.toFixed(2)} m{latest.sigma_xy_saturated ? " (saturated)" : ""}
                      </span>
                    ) : (
                      "—"
                    ),
                  ],
                  ["Observed at", latest ? <span className="cw-mono">t {formatObsTime(latest.t_obs_s)} sim</span> : "—"],
                  ["Received", latest ? <span className="cw-mono">{fmtUtc(latest.rx_utc)} UTC</span> : "—"],
                  ["Mission status", s?.mission?.status ?? "—"],
                ]}
              />
            </Section>
          </div>

          <div className="cw-grid">
            <Section title="Recent activity" aside={<TextLink href="/logs">All events</TextLink>} className="span-8">
              {s?.log.length ? (
                <LogFeed entries={s.log.slice(0, 6)} compact />
              ) : (
                <Empty icon={FileText} title="No mission event yet" />
              )}
            </Section>
            <Section title="Mission record" aside={<TextLink href="/missions">Mission archive</TextLink>} className="span-4">
              <Facts
                items={[
                  ["Started", s?.mission?.created_utc ? <span className="cw-mono">{fmtUtc(s.mission.created_utc, true)}</span> : "—"],
                  ["Packets stored", <span key="p" className="cw-mono">{counters.packets ?? "—"}</span>],
                  ["Duplicates ignored", <span key="d" className="cw-mono">{counters.duplicates ?? 0}</span>],
                  ["Map resets", <span key="r" className="cw-mono">{counters.map_resets ?? 0}</span>],
                  [
                    "Observation span",
                    s?.mission?.last_t_obs_s !== undefined && s?.mission?.last_t_obs_s !== null ? (
                      <span className="cw-mono">
                        {formatAge((s.mission.last_t_obs_s ?? 0) - (s.mission.first_t_obs_s ?? 0))} sim
                      </span>
                    ) : (
                      "—"
                    ),
                  ],
                ]}
              />
            </Section>
          </div>
        </>
      )}
    </Page>
  );
}
