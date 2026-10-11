"use client";

import { useMemo, useState, type CSSProperties } from "react";
import Link from "next/link";
import type { ColumnDef } from "@tanstack/react-table";
import { Activity, ArrowUpRight, Boxes, ChevronLeft, ChevronRight, Crosshair, Database, Eye, Fingerprint, Radar, ScanEye, ShieldCheck, TriangleAlert } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { DataTable } from "@/components/ui/data-table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fmtUtc, typeName } from "@/components/mission/shared";
import { detectionActivity, detectionEventKind, eventsForObstacle, obstacleGeometry, type DetectionEventKind } from "@/lib/ground/detection-summary";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import type { LogEntry, TrackedObstacle } from "@/lib/ground/mission-state";
import type { Position } from "@/lib/ground/types";
import { cn } from "@/lib/utils";
import { rangeFrom } from "./obstacles";
import { Empty, Facts, Page, PageHeader, useMissionGate } from "./primitives";

const colors = { detected: "#4cb4ff", removed: "#f2b13d", reset: "#b99aff", fused: "#4cb4ff", depth: "#2fd69b" };
const eventLabels = { detected: "Detection", removed: "Reconciled removal", reset: "Map reset" };
const activityConfig = { detected: { label: "Detected", color: colors.detected }, removed: { label: "Removed", color: colors.removed }, reset: { label: "Map reset", color: colors.reset } };
const sourceConfig = { fused: { label: "YOLO + depth", color: colors.fused }, depth: { label: "Depth only", color: colors.depth } };
const confidenceTone = (value: number) => value >= 0.6 ? "nominal" : value >= 0.35 ? "caution" : "fault";

function StatusBadge({ status }: { status: string }) {
  return <Badge variant="outline" className={cn("detection-badge", status === "confirmed" ? "nominal" : status === "observed" ? "signal" : "muted")}>{status}</Badge>;
}
function TypeBadge({ type }: { type: string }) {
  return <Badge variant="outline" className={cn("detection-badge", type === "rock" ? "signal" : type === "negative_hazard" ? "fault" : "caution")}>{typeName[type] ?? type}</Badge>;
}
function FilterSelect({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: [string, string][] }) {
  return <Select value={value} onValueChange={(next) => next && onChange(next)}>
    <SelectTrigger size="sm" aria-label={label}><SelectValue>{options.find(([key]) => key === value)?.[1]}</SelectValue></SelectTrigger>
    <SelectContent>{options.map(([key, text]) => <SelectItem key={key} value={key}>{text}</SelectItem>)}</SelectContent>
  </Select>;
}
function EventTimeline({ entries, compact = false }: { entries: LogEntry[]; compact?: boolean }) {
  return <ol className={cn("detection-timeline", compact && "is-compact")}>
    {entries.map((entry) => {
      const kind = detectionEventKind(entry) ?? "detected";
      return <li key={entry.key}>
        <div className="detection-event-marker" style={{ color: colors[kind] }}>{kind === "reset" ? <Radar /> : kind === "removed" ? <Database /> : <ScanEye />}</div>
        <div className="detection-event-body">
          <div><Badge variant="outline" className="detection-event-kind" style={{ color: colors[kind] }}>{eventLabels[kind]}</Badge><time className="cw-mono" dateTime={entry.rx_utc}>{fmtUtc(entry.rx_utc)} UTC</time></div>
          <p>{entry.text}</p>
          {entry.detail && <span>{entry.detail}</span>}
          <small>Rover observation: {entry.t_obs_s === null ? "not supplied" : formatObsTime(entry.t_obs_s) + " sim"}</small>
        </div>
      </li>;
    })}
  </ol>;
}
function DetectionHistory({ events }: { events: LogEntry[] }) {
  const [kind, setKind] = useState<"all" | DetectionEventKind>("all");
  const [page, setPage] = useState(0);
  const filtered = useMemo(() => events.filter((event) => kind === "all" || detectionEventKind(event) === kind), [events, kind]);
  const pages = Math.max(1, Math.ceil(filtered.length / 6));
  const current = Math.min(page, pages - 1);
  return <Card className="detection-card detection-history">
    <CardHeader><CardTitle>Detection history</CardTitle><CardDescription>A reception timeline, not a video feed or a complete mission archive.</CardDescription><CardAction><Badge variant="outline">{events.length} retained events</Badge></CardAction></CardHeader>
    <CardContent>
      <Tabs value={kind} onValueChange={(value) => { setKind(value as typeof kind); setPage(0); }}>
        <TabsList aria-label="Detection history filters"><TabsTrigger value="all">All events</TabsTrigger><TabsTrigger value="detected">Detections</TabsTrigger><TabsTrigger value="removed">Removals</TabsTrigger><TabsTrigger value="reset">Resets</TabsTrigger></TabsList>
        {filtered.length ? <EventTimeline entries={filtered.slice(current * 6, current * 6 + 6)} /> : <Empty icon={Activity} title="No received event in this category" />}
      </Tabs>
      <div className="detection-history-pagination"><span>{filtered.length ? current * 6 + 1 : 0}–{Math.min(filtered.length, (current + 1) * 6)} of {filtered.length} events</span><span>Page {current + 1} of {pages}</span>
        <Button variant="outline" size="icon-sm" aria-label="Previous history page" disabled={current === 0} onClick={() => setPage(current - 1)}><ChevronLeft /></Button>
        <Button variant="outline" size="icon-sm" aria-label="Next history page" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}><ChevronRight /></Button>
      </div>
    </CardContent>
  </Card>;
}
function RelativePlot({ obstacle, rover }: { obstacle: TrackedObstacle; rover: Position | null }) {
  const g = obstacleGeometry(obstacle, rover);
  if (!g) return <Empty icon={Crosshair} title="No received rover pose for relative geometry" />;
  const extent = Math.max(Math.abs(g.dx), Math.abs(g.dy), g.envelopeRadius, 1) + g.envelopeRadius + 0.5;
  const scale = 78 / extent;
  const x = 110 + g.dx * scale;
  const y = 100 - g.dy * scale;
  return <div className="detection-relative-plot">
    <svg viewBox="0 0 220 200" role="img" aria-label={"Obstacle " + g.range.toFixed(2) + " metres from the last received rover pose; odom axes"}>
      <path d="M20 100H200M110 18V182" stroke="#24405f" strokeDasharray="3 4" />
      <circle cx="110" cy="100" r="45" fill="none" stroke="#1b2e45" /><circle cx="110" cy="100" r="78" fill="none" stroke="#1b2e45" />
      <path d={"M110 100L" + x + " " + y} stroke="#4cb4ff" strokeDasharray="4 4" />
      <circle cx={x} cy={y} r={Math.max(4, g.envelopeRadius * scale)} fill="#f2b13d18" stroke="#f2b13d" strokeDasharray="3 3" />
      <circle cx={x} cy={y} r={Math.max(3, obstacle.radius_m * scale)} fill="#4cb4ff88" stroke="#4cb4ff" />
      <circle cx="110" cy="100" r="5" fill="#2fd69b" />
      <text x="185" y="94" fill="#93a9c4" fontSize="10">+x</text><text x="116" y="25" fill="#93a9c4" fontSize="10">+y</text>
      <text x="110" y="194" textAnchor="middle" fill="#93a9c4" fontSize="10">odom · dashed ring = radius + uncertainty</text>
    </svg>
  </div>;
}
export function DetectionSheet({ obstacle, rover, events, present, onClose }: { obstacle: TrackedObstacle | null; rover: Position | null; events: LogEntry[]; present: boolean; onClose: () => void }) {
  const m = useMission();
  const geometry = obstacle ? obstacleGeometry(obstacle, rover) : null;
  const related = obstacle ? eventsForObstacle(events, obstacle.obstacle_id) : [];
  const rx = obstacle?.last_rx_utc ? Date.parse(obstacle.last_rx_utc) : null;
  return <Sheet open={Boolean(obstacle)} onOpenChange={(open) => !open && onClose()}>
    <SheetContent className="detection-sheet" aria-label="Detection details">
      {obstacle && <>
        <SheetHeader>
          <div className="detection-sheet-eyebrow"><ScanEye /> DETECTION INTELLIGENCE <Badge variant="outline">odom frame</Badge></div>
          <SheetTitle className="detection-sheet-title">{obstacle.obstacle_id}</SheetTitle>
          <SheetDescription>Mission #{m.missionId} · {obstacle.persistent_id ? "persistent track" : "observation-local identity"}</SheetDescription>
          <div className="detection-sheet-tags"><TypeBadge type={obstacle.type} /><StatusBadge status={obstacle.status} /><Badge variant="outline">{present ? "In current picture" : "No longer in picture"}</Badge></div>
        </SheetHeader>
        <ScrollArea className="detection-sheet-scroll">
          <div className="detection-sheet-body">
            {(!present || m.fresh !== "LIVE" || obstacle.status === "lost" || obstacle.uncertainty_saturated) && <Alert className="detection-alert">
              <TriangleAlert /><AlertTitle>Inspect with caution</AlertTitle><AlertDescription>{!present ? "This obstacle was removed or cleared. The last inspected record is retained here, not presented as a current obstacle." : obstacle.uncertainty_saturated ? "Uncertainty is saturated: the displayed value is not a reliable upper bound." : obstacle.status === "lost" ? "The rover reports this track as lost; do not assume it is freshly observed." : m.freshLabel + ". These are the last received readings, not live ground truth."}</AlertDescription>
            </Alert>}
            <Tabs defaultValue="overview">
              <TabsList className="detection-sheet-tabs"><TabsTrigger value="overview">Overview</TabsTrigger><TabsTrigger value="evidence">Evidence</TabsTrigger><TabsTrigger value="events">Events ({related.length})</TabsTrigger></TabsList>
              <TabsContent value="overview">
                <div className="detection-sheet-metrics">
                  <div><span>Confidence</span><strong>{Math.round(obstacle.confidence * 100)}<small>%</small></strong><Progress value={obstacle.confidence * 100} aria-label="Obstacle confidence" /></div>
                  <div><span>Range from rover</span><strong>{geometry ? geometry.range.toFixed(2) : "—"}<small>{geometry && " m"}</small></strong><small>last received rover pose</small></div>
                  <div><span>Position uncertainty</span><strong>±{obstacle.uncertainty_m.toFixed(2)}<small> m</small></strong><small>{obstacle.uncertainty_saturated ? "saturated" : "reported by rover"}</small></div>
                  <div><span>Estimated radius</span><strong>{obstacle.radius_m.toFixed(2)}<small> m</small></strong><small>circular obstacle model</small></div>
                </div>
                <h3>Spatial context</h3><RelativePlot obstacle={obstacle} rover={rover} />
                <Facts items={[
                  ["Position x / y", obstacle.x_m.toFixed(3) + " / " + obstacle.y_m.toFixed(3) + " m"],
                  ["Offset from rover", geometry ? geometry.dx.toFixed(2) + " / " + geometry.dy.toFixed(2) + " m" : "No rover pose received"],
                  ["Odom bearing (+x = 0°)", geometry ? geometry.bearing.toFixed(1) + "°" : "—"],
                  ["Relative to rover heading", geometry ? geometry.relativeBearing.toFixed(1) + "°" : "—"],
                  ["Radius + uncertainty", (obstacle.radius_m + obstacle.uncertainty_m).toFixed(2) + " m" + (obstacle.uncertainty_saturated ? " (saturated)" : "")],
                  ["Rover pose observed", rover ? formatObsTime(rover.t_obs_s) + " sim" : "—"],
                ]} />
                <p className="detection-footnote">Geometry is derived from received odometry. Range is center-to-center distance, not collision clearance or proof that a route is safe.</p>
              </TabsContent>
              <TabsContent value="evidence">
                <h3>Identity & provenance</h3>
                <Facts items={[
                  ["Detection source", obstacle.source === "yolo+depth" ? "YOLO class fused with depth" : "Depth evidence only; no YOLO class"],
                  ["Tracking status", obstacle.status], ["Persistent identity", obstacle.persistent_id ? "Yes" : "No"],
                  ["Linked track", obstacle.linked_track ?? "Not supplied"],
                  ["Mapping protection", obstacle.protected === undefined ? "Not supplied" : obstacle.protected ? "Protected evidence" : "Not protected"],
                  ["Frame", obstacle.frame], ["Mission", m.missionId ?? "—"],
                ]} />
                <Separator /><h3>Two clocks, kept separate</h3>
                <Facts items={[
                  ["First observed (rover)", obstacle.first_t_obs_s === undefined ? "Not supplied" : formatObsTime(obstacle.first_t_obs_s) + " sim"],
                  ["Latest observed (rover)", formatObsTime(obstacle.obs_timestamp_s) + " sim"],
                  ["First received (ground)", obstacle.first_rx_utc ? fmtUtc(obstacle.first_rx_utc, true) + " UTC" : "Not supplied"],
                  ["Last received (ground)", obstacle.last_rx_utc ? fmtUtc(obstacle.last_rx_utc, true) + " UTC" : "Not supplied"],
                  ["Record reception age", rx !== null && Number.isFinite(rx) ? formatAge((m.groundNow - rx) / 1000) + " ago" : "—"],
                ]} />
                <p className="detection-footnote">Simulation time cannot be subtracted from UTC to claim network latency. No image, bounding box, or per-detection inference latency is downlinked.</p>
                <h3>Received record</h3><pre className="detection-raw">{JSON.stringify(obstacle, null, 2)}</pre>
              </TabsContent>
              <TabsContent value="events">{related.length ? <EventTimeline compact entries={related} /> : <Empty icon={Activity} title="No related event retained in this session log" />}</TabsContent>
            </Tabs>
          </div>
        </ScrollArea>
        <div className="detection-sheet-footer"><Button nativeButton={false} render={<Link href={m.href("/map")} />}><Crosshair /> Open mission map <ArrowUpRight /></Button><Button variant="outline" onClick={onClose}>Close inspection</Button></div>
      </>}
    </SheetContent>
  </Sheet>;
}

export function DetectionPage() {
  const m = useMission();
  const gate = useMissionGate();
  const [type, setType] = useState("all");
  const [status, setStatus] = useState("all");
  const [source, setSource] = useState("all");
  const [selection, setSelection] = useState<{ missionId: string | null; obstacle: TrackedObstacle } | null>(null);
  const all = useMemo(() => Object.values(m.state?.obstacles ?? {}), [m.state?.obstacles]);
  const rows = useMemo(() => all.filter((o) => (type === "all" || o.type === type) && (status === "all" || o.status === status) && (source === "all" || o.source === source)), [all, type, status, source]);
  const events = useMemo(() => (m.state?.log ?? []).filter((event) => detectionEventKind(event) !== null), [m.state?.log]);
  const activity = useMemo(() => detectionActivity(events), [events]);
  const fused = all.filter((o) => o.source === "yolo+depth").length;
  const depth = all.filter((o) => o.source === "depth").length;
  const confirmed = all.filter((o) => o.status === "confirmed").length;
  const hazards = all.filter((o) => o.type === "negative_hazard").length;
  const meanConfidence = all.length ? all.reduce((sum, o) => sum + o.confidence, 0) / all.length : null;
  const picked = selection?.missionId === m.missionId ? m.state?.obstacles[selection.obstacle.obstacle_id] ?? selection.obstacle : null;
  const sources = [{ name: "fused", value: fused, fill: colors.fused }, { name: "depth", value: depth, fill: colors.depth }].filter((s) => s.value > 0);
  const bands = [
    { label: "High · 80–100%", count: all.filter((o) => o.confidence >= 0.8).length, color: "#2fd69b" },
    { label: "Moderate · 60–79%", count: all.filter((o) => o.confidence >= 0.6 && o.confidence < 0.8).length, color: "#4cb4ff" },
    { label: "Low · 35–59%", count: all.filter((o) => o.confidence >= 0.35 && o.confidence < 0.6).length, color: "#f2b13d" },
    { label: "Very low · below 35%", count: all.filter((o) => o.confidence < 0.35).length, color: "#ff5a6e" },
  ];
  const columns = useMemo<ColumnDef<TrackedObstacle>[]>(() => [
    { accessorKey: "obstacle_id", header: "Track ID", enableHiding: false, cell: ({ row }) => <Button variant="link" className="detection-id cw-mono" onClick={() => setSelection({ missionId: m.missionId, obstacle: row.original })}>{row.original.obstacle_id}<ArrowUpRight /></Button> },
    { accessorKey: "type", header: "Type", cell: ({ row }) => <TypeBadge type={row.original.type} /> },
    { accessorKey: "confidence", header: "Confidence", cell: ({ row }) => <div className={cn("detection-confidence", confidenceTone(row.original.confidence))}><Progress value={row.original.confidence * 100} aria-label={"Confidence of " + row.original.obstacle_id} /><span className="cw-mono">{Math.round(row.original.confidence * 100)}%</span></div> },
    { id: "range", header: "Range", accessorFn: (o) => rangeFrom(o, m.latest) ?? undefined, sortUndefined: "last", cell: ({ row }) => { const r = rangeFrom(row.original, m.latest); return <span className="cw-mono">{r === null ? "—" : r.toFixed(2) + " m"}</span>; } },
    { accessorKey: "uncertainty_m", header: "Uncertainty", cell: ({ row }) => <span className="cw-mono">±{row.original.uncertainty_m.toFixed(2)} m{row.original.uncertainty_saturated && " *"}</span> },
    { accessorKey: "source", header: "Source", cell: ({ row }) => <span className="detection-source"><span style={{ background: row.original.source === "depth" ? colors.depth : colors.fused }} />{row.original.source === "depth" ? "Depth only" : "YOLO + depth"}</span> },
    { accessorKey: "status", header: "Status", cell: ({ row }) => <StatusBadge status={row.original.status} /> },
    { accessorKey: "last_rx_utc", header: "Last reception", cell: ({ row }) => { const rx = row.original.last_rx_utc ? Date.parse(row.original.last_rx_utc) : null; return <span className="cw-mono cw-dim">{rx !== null && Number.isFinite(rx) ? formatAge((m.groundNow - rx) / 1000) + " ago" : "—"}</span>; } },
  ], [m.missionId, m.latest, m.groundNow]);

  return <Page id="detection">
    <PageHeader title="Detection intelligence" description="Inspect the rover’s received obstacle picture, trace its evidence, and understand how detections evolve." aside={<Button nativeButton={false} variant="outline" render={<Link href={m.href("/map")} />}><Crosshair /> Mission map <ArrowUpRight /></Button>} />
    {gate ?? <div className="detection-dashboard">
      <div className="detection-kpis">
        {[
          { title: "Current obstacle picture", value: all.length, detail: all.filter((o) => o.type === "rock").length + " rocks · " + all.filter((o) => o.type === "obstacle").length + " depth-only obstacles", Icon: Boxes, tone: "signal" },
          { title: "Confirmed tracks", value: confirmed, detail: all.filter((o) => o.status === "observed").length + " observed · " + all.filter((o) => o.status === "lost").length + " lost", Icon: ShieldCheck, tone: "nominal" },
          { title: "Mean confidence", value: meanConfidence === null ? "—" : Math.round(meanConfidence * 100) + "%", detail: "Across the current received picture", Icon: Radar, tone: "signal" },
          { title: "Negative hazards", value: hazards, detail: "Reported drops or negative terrain", Icon: TriangleAlert, tone: hazards ? "fault" : "muted" },
        ].map(({ title, value, detail, Icon, tone }) => <Card className={cn("detection-card detection-kpi", tone)} key={title}><CardHeader><CardDescription>{title}</CardDescription><CardAction><Icon /></CardAction></CardHeader><CardContent><strong>{value}</strong><p>{detail}</p></CardContent></Card>)}
      </div>
      <div className="detection-analysis-grid">
        <Card className="detection-card">
          <CardHeader><CardTitle>Received detection activity</CardTitle><CardDescription>Retained detection, removal, and reset events grouped by UTC reception minute.</CardDescription><CardAction><Badge variant="outline"><Activity className="size-3" /> Ground clock</Badge></CardAction></CardHeader>
          <CardContent>
            {activity.length ? <ChartContainer config={activityConfig} className="detection-activity-chart">
              <BarChart data={activity} margin={{ top: 12, right: 12, bottom: 0, left: -24 }}>
                <CartesianGrid vertical={false} stroke="#1b2e45" /><XAxis dataKey="utc" tickFormatter={(value: number) => fmtUtc(value).slice(0, 5)} tickLine={false} axisLine={false} minTickGap={36} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => fmtUtc(Number(value)) + " UTC"} />} />
                <Bar dataKey="detected" stackId="events" fill="var(--color-detected)" maxBarSize={38} isAnimationActive={m.motionEnabled} /><Bar dataKey="removed" stackId="events" fill="var(--color-removed)" maxBarSize={38} isAnimationActive={m.motionEnabled} /><Bar dataKey="reset" stackId="events" fill="var(--color-reset)" radius={[3, 3, 0, 0]} maxBarSize={38} isAnimationActive={m.motionEnabled} />
              </BarChart>
            </ChartContainer> : <Empty icon={Activity} title="Waiting for received detection events" />}
            <div className="detection-chart-legend">{Object.entries(activityConfig).map(([key, item]) => <span key={key}><i style={{ background: item.color }} />{item.label}</span>)}</div>
          </CardContent>
        </Card>
        <Card className="detection-card">
          <CardHeader><CardTitle>Evidence sources</CardTitle><CardDescription>Composition of the current obstacle picture.</CardDescription></CardHeader>
          <CardContent>
            {sources.length ? <div className="detection-source-chart-wrap"><ChartContainer config={sourceConfig} className="detection-source-chart"><PieChart><ChartTooltip content={<ChartTooltipContent nameKey="name" hideLabel />} /><Pie data={sources} dataKey="value" nameKey="name" innerRadius={56} outerRadius={76} paddingAngle={3} stroke="none" isAnimationActive={m.motionEnabled}>{sources.map((item) => <Cell key={item.name} fill={item.fill} />)}</Pie></PieChart></ChartContainer><div className="detection-donut-label"><strong>{all.length}</strong><span>received</span></div></div> : <Empty icon={Fingerprint} title="No obstacle evidence received" />}
            <div className="detection-source-stats"><span><i style={{ background: colors.fused }} />YOLO + depth <strong>{fused}</strong></span><span><i style={{ background: colors.depth }} />Depth only <strong>{depth}</strong></span></div>
          </CardContent>
        </Card>
      </div>
      <Card className="detection-card detection-catalog">
        <CardHeader><CardTitle>Obstacle catalog</CardTitle><CardDescription>Search, sort, and inspect a track. Click any row to open its evidence sheet.</CardDescription><CardAction><Badge variant="outline">{rows.length} / {all.length} in picture</Badge></CardAction></CardHeader>
        <DataTable key={m.missionId + ":" + type + ":" + status + ":" + source} label="obstacle catalog" columns={columns} data={rows} getRowId={(o) => o.obstacle_id} initialSorting={[{ id: "range", desc: false }]} searchPlaceholder="Search track ID, type, or source…" onRowClick={(o) => setSelection({ missionId: m.missionId, obstacle: o })} selectedRowId={picked?.obstacle_id} emptyMessage={all.length ? "No obstacles match your filters." : "No OBSTACLES packet has reported an obstacle yet."}
          toolbar={<><FilterSelect label="Filter obstacle type" value={type} onChange={setType} options={[["all", "All types"], ["rock", "Rocks"], ["obstacle", "Depth obstacles"], ["negative_hazard", "Negative hazards"]]} /><FilterSelect label="Filter tracking status" value={status} onChange={setStatus} options={[["all", "All statuses"], ["confirmed", "Confirmed"], ["observed", "Observed"], ["lost", "Lost"]]} /><FilterSelect label="Filter evidence source" value={source} onChange={setSource} options={[["all", "All sources"], ["yolo+depth", "YOLO + depth"], ["depth", "Depth only"]]} />{(type !== "all" || status !== "all" || source !== "all") && <Button variant="ghost" size="sm" onClick={() => { setType("all"); setStatus("all"); setSource("all"); }}>Reset filters</Button>}</>} />
      </Card>
      <div className="detection-analysis-grid">
        <DetectionHistory key={m.missionId} events={events} />
        <div className="detection-side-stack">
          <Card className="detection-card"><CardHeader><CardTitle>Confidence distribution</CardTitle><CardDescription>Evidence strength, not a safety score.</CardDescription></CardHeader><CardContent className="detection-confidence-bands">{bands.map((band) => <div key={band.label}><div><span>{band.label}</span><strong>{band.count}</strong></div><Progress value={all.length ? band.count * 100 / all.length : 0} aria-label={band.label} style={{ "--band-color": band.color } as CSSProperties} /></div>)}</CardContent></Card>
          <Alert className="detection-integrity-note"><Eye /><AlertTitle>Received evidence only</AlertTitle><AlertDescription>No video or full occupancy grid is downlinked. Removed tracks are reconciled out of the catalog; history is limited to the session’s retained events. A protected or confident detection does not certify safe navigation.</AlertDescription></Alert>
        </div>
      </div>
    </div>}
    <DetectionSheet key={m.missionId} obstacle={picked} rover={m.latest} events={events} present={Boolean(picked && m.state?.obstacles[picked.obstacle_id])} onClose={() => setSelection(null)} />
  </Page>;
}
