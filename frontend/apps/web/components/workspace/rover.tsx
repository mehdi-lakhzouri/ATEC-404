"use client";

import Link from "next/link";
import { Compass, Crosshair, History, LocateFixed, ShieldCheck, Signal, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Scene, scenes, fmtUtc } from "@/components/mission/shared";
import { freshnessLabel, useMission } from "@/lib/ground/mission-context";
import { formatObsTime } from "@/lib/ground/freshness";
import { DashboardBadge, DashboardCard, DashboardMetric } from "./dashboard";
import { Facts, Page, PageHeader, useMissionGate } from "./primitives";

export function RoverPage() {
  const m = useMission();
  const p = m.latest;
  const health = m.state?.health;
  const gate = useMissionGate();
  return <Page id="rover"><PageHeader title="Rover operations" description="The Artemis-X platform and its last received operating state. No onboard readings are inferred." aside={<Button nativeButton={false} variant="outline" render={<Link href={m.href("/missions")} />}><History /> Mission history</Button>} />
    {gate ?? <div className="dashboard-stack">
      <div className="dashboard-metrics">
        <DashboardMetric label="Reported health" value={health?.state ?? "—"} detail="Last received rover health state" icon={ShieldCheck} tone={health?.state === "OK" ? "nominal" : health ? "caution" : "muted"} />
        <DashboardMetric label="Position · odom" value={p ? p.x_m.toFixed(2) + ", " + p.y_m.toFixed(2) : "—"} detail="Coordinates in meters" icon={LocateFixed} />
        <DashboardMetric label="Heading" value={p ? p.heading_deg.toFixed(1) + "°" : "—"} detail="0° = odom +x, counter-clockwise" icon={Compass} />
        <DashboardMetric label="Pose uncertainty" value={p ? "±" + p.sigma_xy_m.toFixed(2) + " m" : "—"} detail={p?.sigma_xy_saturated ? "Saturated; not an upper bound" : "Reported odometry uncertainty"} icon={Signal} tone={p?.sigma_xy_saturated ? "caution" : "signal"} />
      </div>
      <div className="dashboard-split">
        <DashboardCard title="Artemis-X rover" description="6-wheel research platform · lunar autonomy" className="rover-profile-card"><Scene crop={scenes.rover} label="Decorative illustration of the Artemis-X rover" className="rover-profile-art" /><p className="dashboard-footnote">Platform artwork is illustrative. Operating values below come only from the ground station.</p><div className="dashboard-actions"><Button nativeButton={false} render={<Link href={m.href("/telemetry")} />}><Signal /> Engineering telemetry</Button><Button nativeButton={false} variant="outline" render={<Link href={m.href("/map")} />}><Crosshair /> Mission map</Button></div></DashboardCard>
        <DashboardCard title="Received operating record" action={<span title={m.freshLabel}><DashboardBadge>{freshnessLabel[m.fresh]}</DashboardBadge></span>}><Facts items={[["Mission", m.missionId ?? "—"], ["Frame", "odom"], ["Pose observed", p ? formatObsTime(p.t_obs_s) + " sim" : "—"], ["Pose received", p ? fmtUtc(p.rx_utc, true) + " UTC" : "—"], ["Health reasons", health?.reasons.length ? health.reasons.join(", ").replaceAll("_", " ") : health ? "No reason supplied" : "—"], ["Dashboard stream", m.stream.connection]]} /><Alert className="dashboard-alert"><TriangleAlert /><AlertTitle>Received state, not live ground truth</AlertTitle><AlertDescription>Battery, wheel loads, commanded speed, and video are not downlinked by the current API.</AlertDescription></Alert></DashboardCard>
      </div>
    </div>}
  </Page>;
}
