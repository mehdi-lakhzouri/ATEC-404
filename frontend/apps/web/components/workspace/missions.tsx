"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import type { ColumnDef } from "@tanstack/react-table";
import { Activity, ArrowUpRight, Database, History, Layers3, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { DataTable } from "@/components/ui/data-table";
import { Separator } from "@/components/ui/separator";
import { fmtUtc } from "@/components/mission/shared";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { useMissionStats } from "@/lib/ground/use-mission-stats";
import type { Mission } from "@/lib/ground/types";
import { DashboardBadge, DashboardCard, DashboardMetric, DashboardSelect, DashboardSheet } from "./dashboard";
import { Empty, ErrorNote, Facts, Page, PageHeader } from "./primitives";

const chartConfig = { packets: { label: "Stored packets", color: "#4cb4ff" }, poses: { label: "Stored poses", color: "#2fd69b" } };

function MissionInspector({ mission, onClose }: { mission: Mission | null; onClose: () => void }) {
  const m = useMission();
  const stats = useMissionStats(mission?.mission_id ?? null);
  const s = stats.data;
  const receiver = s?.receiver;
  return <DashboardSheet open={Boolean(mission)} onClose={onClose} eyebrow={<><History /> MISSION DOSSIER</>}
    title={mission ? "#" + mission.mission_id : "Mission inspection"}
    description="Persisted session metadata and ground-station statistics."
    tags={mission && <><DashboardBadge tone={mission.status === "active" ? "nominal" : "muted"}>{mission.status}</DashboardBadge><DashboardBadge>{mission.mission_id === m.missionId ? "Active in this console" : "Inspection only"}</DashboardBadge></>}
    footer={mission && <Button onClick={() => { m.selectMission(mission.mission_id); onClose(); }}><ArrowUpRight /> {mission.mission_id === m.missionId ? "Keep this mission selected" : "Use this mission"}</Button>}>
    {mission && <>
      <Alert className="dashboard-alert"><ShieldCheck /><AlertTitle>Inspect before switching</AlertTitle><AlertDescription>Opening this dossier does not switch your dashboard mission. “Use this mission” pins its ID across every page.</AlertDescription></Alert>
      {stats.error && <ErrorNote message={stats.error} onRetry={stats.refresh} />}
      <Tabs defaultValue="overview"><TabsList className="detection-sheet-tabs"><TabsTrigger value="overview">Overview</TabsTrigger><TabsTrigger value="receiver">Receiver</TabsTrigger><TabsTrigger value="record">Record</TabsTrigger></TabsList>
        <TabsContent value="overview">
          <div className="detection-sheet-metrics">
            <div><span>Stored packets</span><strong>{(s?.counters ?? mission.counters).packets ?? "—"}</strong><small>validated packet counter</small></div>
            <div><span>Stored poses</span><strong>{s?.documents.positions ?? mission.counters.pose ?? "—"}</strong><small>odometry observations</small></div>
            <div><span>Present obstacles</span><strong>{s?.obstacles_present ?? "—"}</strong><small>reconciled picture</small></div>
            <div><span>Observation span</span><strong>{s?.observation_span_s === null || s?.observation_span_s === undefined ? "—" : formatAge(s.observation_span_s)}</strong><small>rover simulation time</small></div>
          </div>
          {stats.loading && <p className="dashboard-footnote" role="status">Loading stored mission statistics…</p>}
          <h3>Session identity</h3><Facts items={[["Mission ID", mission.mission_id], ["Session", mission.session], ["Status", mission.status], ["Coordinate frame", mission.frame ?? "Not supplied"], ["Created", fmtUtc(mission.created_utc, true) + " UTC"], ["Last reception", fmtUtc(mission.last_rx_utc, true) + " UTC"], ["First observation", formatObsTime(mission.first_t_obs_s) + " sim"], ["Latest observation", formatObsTime(mission.last_t_obs_s) + " sim"], ["Reported health", mission.health?.state ?? "Not supplied"], ["Health reasons", mission.health?.reasons.length ? mission.health.reasons.join(", ").replaceAll("_", " ") : "None supplied"]]} />
          <Separator /><h3>Stored document counts</h3>
          {s ? <Facts items={Object.entries(s.documents).map(([key, value]) => [key.replaceAll("_", " "), value])} /> : <p className="dashboard-footnote">Statistics are not available yet.</p>}
        </TabsContent>
        <TabsContent value="receiver"><h3>Ground reception counters</h3><Facts items={[["Valid / rejected", receiver ? (receiver.valid ?? 0) + " / " + (receiver.invalid ?? 0) : "Not supplied"], ["Missing / reordered", receiver ? (receiver.missing ?? 0) + " / " + (receiver.reordered ?? 0) : "Not supplied"], ["Duplicate packets", s?.counters.duplicates ?? mission.counters.duplicates ?? "Not supplied"], ["ACKs sent", receiver?.acks_sent ?? "Not supplied"], ["Map resets", s?.counters.map_resets ?? mission.counters.map_resets ?? "Not supplied"], ["Recorded emulator preset", s?.emulator?.preset ?? "Not supplied"], ["Recorded delivery ratio", s?.emulator?.delivery_ratio === null || s?.emulator?.delivery_ratio === undefined ? "Not supplied" : (s.emulator.delivery_ratio * 100).toFixed(1) + "%"]]} /><p className="dashboard-footnote">Receiver and emulator counters are recorded snapshots, not the current state of the shared network controller.</p></TabsContent>
        <TabsContent value="record"><h3>Mission metadata</h3><pre className="detection-raw">{JSON.stringify(mission, null, 2)}</pre><h3>Persisted statistics</h3><pre className="detection-raw">{s ? JSON.stringify(s, null, 2) : "No statistics received."}</pre></TabsContent>
      </Tabs>
    </>}
  </DashboardSheet>;
}

export function MissionsPage() {
  const m = useMission();
  const list = m.list;
  const stats = useMissionStats(m.missionId);
  const [status, setStatus] = useState("all");
  const [inspected, setInspected] = useState<Mission | null>(null);
  const selected = list.missions.find((mission) => mission.mission_id === m.missionId);
  const rows = useMemo(() => list.missions.filter((mission) => status === "all" || mission.status === status), [list.missions, status]);
  const totals = useMemo(() => list.missions.reduce((acc, mission) => ({ packets: acc.packets + (mission.counters.packets ?? 0), poses: acc.poses + (mission.counters.pose ?? 0), active: acc.active + Number(mission.status === "active") }), { packets: 0, poses: 0, active: 0 }), [list.missions]);
  const comparison = useMemo(() => [...list.missions].sort((a, b) => Date.parse(a.created_utc) - Date.parse(b.created_utc)).slice(-8).map((mission) => ({ id: mission.mission_id, label: "#" + mission.mission_id.slice(-4), packets: mission.counters.packets, poses: mission.counters.pose })), [list.missions]);
  const columns = useMemo<ColumnDef<Mission>[]>(() => [
    { accessorKey: "mission_id", header: "Mission ID", enableHiding: false, cell: ({ row }) => <Button variant="link" className="dashboard-record-id cw-mono" onClick={() => setInspected(row.original)}>#{row.original.mission_id}<ArrowUpRight /></Button> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <DashboardBadge tone={row.original.status === "active" ? "nominal" : "muted"}>{row.original.status}</DashboardBadge> },
    { accessorKey: "created_utc", header: "Started · UTC", cell: ({ row }) => <span className="cw-mono">{fmtUtc(row.original.created_utc, true)}</span> },
    { accessorKey: "last_rx_utc", header: "Last reception", cell: ({ row }) => <span className="cw-mono">{formatAge((m.groundNow - Date.parse(row.original.last_rx_utc)) / 1000)} ago</span> },
    { id: "packets", header: "Packets", accessorFn: (mission) => mission.counters.packets, cell: ({ getValue }) => <span className="cw-mono">{getValue<number | undefined>() ?? "—"}</span> },
    { id: "poses", header: "Poses", accessorFn: (mission) => mission.counters.pose, cell: ({ getValue }) => <span className="cw-mono">{getValue<number | undefined>() ?? "—"}</span> },
    { id: "health", header: "Rover health", accessorFn: (mission) => mission.health?.state, cell: ({ row }) => <DashboardBadge tone={row.original.health?.state === "OK" ? "nominal" : row.original.health ? "caution" : "muted"}>{row.original.health?.state ?? "—"}</DashboardBadge> },
    { id: "selection", header: "Console", enableSorting: false, cell: ({ row }) => row.original.mission_id === m.missionId ? <DashboardBadge>Selected</DashboardBadge> : <Button variant="outline" size="sm" onClick={() => m.selectMission(row.original.mission_id)}>Use mission</Button> },
  ], [m.missionId, m.selectMission, m.groundNow]);
  const counters = stats.data?.counters ?? selected?.counters;
  return <Page id="missions">
    <PageHeader title="Mission archive" description="Compare stored rover sessions, inspect their evidence, and pin the mission your console follows."
      aside={<div className="dashboard-actions"><Button variant="outline" size="sm" onClick={() => { list.refresh(); stats.refresh(); }}><RefreshCw /> Refresh</Button><Button variant={m.followLatest ? "default" : "outline"} size="sm" aria-pressed={m.followLatest} onClick={m.followLatest ? m.stopFollowing : m.followNewest}><Activity /> {m.followLatest ? "Following newest" : "Follow newest"}</Button></div>} />
    <div className="dashboard-stack">
      {list.error && <ErrorNote message={list.error} onRetry={list.refresh} />}
      <div className="dashboard-metrics">
        <DashboardMetric label="Stored sessions" value={!list.missions.length && (list.loading || list.error) ? "—" : list.total} detail={list.missions.length + " loaded for comparison"} icon={History} />
        <DashboardMetric label="Active sessions · loaded" value={list.missions.length ? totals.active : "—"} detail="Backend status, not a freshness guarantee" icon={Activity} tone={totals.active ? "nominal" : "muted"} />
        <DashboardMetric label="Stored packets · loaded" value={list.missions.length ? totals.packets.toLocaleString("en-US") : "—"} detail="Counters across loaded sessions" icon={Database} />
        <DashboardMetric label="Stored poses · loaded" value={list.missions.length ? totals.poses.toLocaleString("en-US") : "—"} detail="Received odometry observations" icon={Layers3} />
      </div>
      <div className="dashboard-split">
        <DashboardCard title="Session comparison" description="Packet and pose counters for up to eight of the loaded sessions." action={<DashboardBadge><Database /> Persisted records</DashboardBadge>}>
          {comparison.length ? <ChartContainer config={chartConfig} className="dashboard-chart"><BarChart data={comparison} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}><CartesianGrid vertical={false} stroke="#1b2e45" /><XAxis dataKey="label" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} allowDecimals={false} /><ChartTooltip content={<ChartTooltipContent labelFormatter={(_, payload) => "Mission #" + (payload?.[0]?.payload as { id?: string } | undefined)?.id} />} /><Bar dataKey="packets" fill="var(--color-packets)" radius={[3, 3, 0, 0]} maxBarSize={32} isAnimationActive={m.motionEnabled} /><Bar dataKey="poses" fill="var(--color-poses)" radius={[3, 3, 0, 0]} maxBarSize={32} isAnimationActive={m.motionEnabled} /></BarChart></ChartContainer> : <Empty icon={History} title={list.loading ? "Loading persisted sessions…" : "No stored mission to compare"} />}
          <div className="dashboard-legend"><span><i style={{ background: "#4cb4ff" }} />Packets</span><span><i style={{ background: "#2fd69b" }} />Poses</span></div>
        </DashboardCard>
        <DashboardCard title="Current console mission" description={m.missionId ? "#" + m.missionId : "No mission selected"} action={<DashboardBadge>{m.followLatest ? "Following" : "Pinned"}</DashboardBadge>}>
          {stats.error && <ErrorNote message={stats.error} onRetry={stats.refresh} />}
          <Facts items={[["Status", selected?.status ?? m.state?.mission?.status ?? "—"], ["Stored packets", counters?.packets ?? "—"], ["Present obstacles", stats.data?.obstacles_present ?? "—"], ["Map resets", counters?.map_resets ?? "—"], ["Duplicates", counters?.duplicates ?? "—"], ["Last reception", fmtUtc(selected?.last_rx_utc ?? m.state?.mission?.last_rx_utc) + " UTC"]]} />
          <div className="dashboard-actions"><Button nativeButton={false} variant="outline" size="sm" render={<Link href={m.href("/map")} />}><ArrowUpRight /> Mapping</Button><Button nativeButton={false} variant="outline" size="sm" render={<Link href={m.href("/logs")} />}><History /> Events</Button></div>
        </DashboardCard>
      </div>
      <DashboardCard title="Mission directory" description="Click a row to inspect its dossier. Switching missions is an explicit action." flush action={<DashboardBadge>{rows.length} / {list.total} sessions</DashboardBadge>}>
        <DataTable key={status} label="mission history" data={rows} columns={columns} getRowId={(mission) => mission.mission_id} selectedRowId={m.missionId}
          onRowClick={setInspected} initialSorting={[{ id: "created_utc", desc: true }]} searchPlaceholder="Search loaded mission IDs, health, or status…"
          searchText={(mission) => [mission.mission_id, mission.status, mission.created_utc, mission.health?.state, ...(mission.health?.reasons ?? [])].join(" ")}
          emptyMessage={list.loading ? "Loading mission records…" : list.missions.length ? "No loaded mission matches these filters." : "No mission is stored yet."}
          toolbar={<DashboardSelect label="Filter mission status" value={status} onChange={setStatus} options={[["all", "All statuses"], ["active", "Active"], ["idle", "Idle"]]} />} />
        {list.hasMore && <div className="data-table-load-more"><Button variant="outline" onClick={list.loadMore}>Load more sessions · {list.missions.length} of {list.total} loaded</Button></div>}
      </DashboardCard>
      <Alert className="dashboard-alert"><TriangleAlert /><AlertTitle>An archive, not a live-state guarantee</AlertTitle><AlertDescription>Charts summarize loaded records only. “Active” is backend session status; UTC reception age determines whether the data is fresh. No rover observation clock is subtracted from UTC.</AlertDescription></Alert>
    </div>
    <MissionInspector mission={inspected ? list.missions.find((mission) => mission.mission_id === inspected.mission_id) ?? inspected : null} onClose={() => setInspected(null)} />
  </Page>;
}
