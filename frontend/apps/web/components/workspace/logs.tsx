"use client";

import { useMemo, useState, type ReactNode } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Activity, ArrowUpRight, ChevronLeft, ChevronRight, Clock3, FileText, List, RadioTower, ScanEye, Search, Settings2, ShieldAlert, Waypoints } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DataTable } from "@/components/ui/data-table";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { fmtUtc } from "@/components/mission/shared";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { MAX_LOG, type LogEntry, type LogTone } from "@/lib/ground/mission-state";
import { eventActivity, filterEvents } from "@/lib/ground/dashboard-summary";
import { cn } from "@/lib/utils";
import { DashboardBadge, DashboardCard, DashboardMetric, DashboardSelect, DashboardSheet, type DashboardTone } from "./dashboard";
import { Empty, Facts, Page, PageHeader, useMissionGate } from "./primitives";

const severity: Record<LogTone, { label: string; tone: string; badge: DashboardTone; color: string }> = {
  good: { label: "Nominal", tone: "ok", badge: "nominal", color: "#2fd69b" },
  info: { label: "Info", tone: "signal", badge: "signal", color: "#4cb4ff" },
  warn: { label: "Caution", tone: "warn", badge: "caution", color: "#f2b13d" },
  bad: { label: "Fault", tone: "bad", badge: "fault", color: "#ff5a6e" },
};
const categories = [{ key: "Navigation", Icon: Waypoints }, { key: "Detection", Icon: ScanEye }, { key: "System", Icon: Settings2 }, { key: "Communication", Icon: RadioTower }] as const;
const activityConfig = Object.fromEntries(Object.entries(severity).map(([key, value]) => [key, { label: value.label, color: value.color }]));
const filterOptions = { categories: [["all", "All categories"], ...categories.map(({ key }) => [key, key])] as [string, string][], severity: [["all", "Any severity"], ["bad", "Fault"], ["warn", "Caution"], ["good", "Nominal"], ["info", "Info"]] as [string, string][] };

export function LogFeed({ entries, compact }: { entries: LogEntry[]; compact?: boolean }) {
  return (
    <ol className={cn("cw-feed", compact && "is-compact")}>
      {entries.map((e) => (
        <li key={e.key} className={`tone-${severity[e.tone].tone}`}>
          <time className="cw-mono" dateTime={e.rx_utc} title={`received ${fmtUtc(e.rx_utc, true)} UTC`}>
            {fmtUtc(e.rx_utc)}
          </time>
          <span className="cw-dot" aria-label={severity[e.tone].label} />
          <div className="cw-feed-body">
            <p>
              {e.text}
              {!compact && <span className="cw-tag">{e.category}</span>}
            </p>
            {e.detail && <p className="cw-feed-detail">{e.detail.replace(/^\(|\)$/g, "")}</p>}
          </div>
          {!compact && (
            <span className="cw-feed-obs cw-mono" title="rover observation time (simulation)">
              {e.t_obs_s !== null ? `t ${formatObsTime(e.t_obs_s)}` : "ground"}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}


function EventInspector({ entry, onClose }: { entry: LogEntry | null; onClose: () => void }) {
  const m = useMission();
  const retained = Boolean(entry && m.state?.log.some((candidate) => candidate.key === entry.key));
  const related = entry ? (m.state?.log ?? []).filter((candidate) => candidate.category === entry.category && candidate.key !== entry.key).slice(0, 6) : [];
  return <DashboardSheet open={Boolean(entry)} onClose={onClose} eyebrow={<><FileText /> MISSION EVENT INSPECTION</>} title={entry ? severity[entry.tone].label + " · " + entry.category : "Event inspection"}
    description={"Retained event evidence for mission #" + m.missionId} tags={entry && <><DashboardBadge tone={severity[entry.tone].badge}>{severity[entry.tone].label}</DashboardBadge><DashboardBadge>{entry.category}</DashboardBadge><DashboardBadge tone="muted">{retained ? "Retained in session" : "Last inspected record"}</DashboardBadge></>}>
    {entry && <>
      {!retained && <Alert className="dashboard-alert"><ShieldAlert /><AlertTitle>Outside the retained window</AlertTitle><AlertDescription>The session log has moved on. This inspector keeps the last clicked record; it is not evidence of a new event.</AlertDescription></Alert>}
      <Tabs defaultValue="overview"><TabsList className="detection-sheet-tabs"><TabsTrigger value="overview">Overview</TabsTrigger><TabsTrigger value="context">Context</TabsTrigger><TabsTrigger value="record">Record</TabsTrigger></TabsList>
        <TabsContent value="overview"><div className="event-inspector-message"><DashboardBadge tone={severity[entry.tone].badge}>{severity[entry.tone].label}</DashboardBadge><h3>{entry.text}</h3>{entry.detail && <p>{entry.detail}</p>}</div>
          <h3>Two independent clocks</h3><Facts items={[["Ground reception", fmtUtc(entry.rx_utc, true) + " UTC"], ["Rover observation", entry.t_obs_s === null ? "Ground-generated; not supplied" : formatObsTime(entry.t_obs_s) + " sim"], ["Reception age", Number.isFinite(Date.parse(entry.rx_utc)) ? formatAge((m.groundNow - Date.parse(entry.rx_utc)) / 1000) + " ago" : "Unknown"], ["Mission", m.missionId ?? "—"], ["Category", entry.category], ["Severity label", severity[entry.tone].label], ["Deduplication key", entry.key]]} />
          <Alert className="dashboard-alert"><Clock3 /><AlertTitle>Clock domains stay separate</AlertTitle><AlertDescription>Reception age uses UTC only. Rover simulation time does not measure Earth–Moon network latency.</AlertDescription></Alert>
        </TabsContent>
        <TabsContent value="context"><h3>Other retained {entry.category.toLowerCase()} events</h3><p className="dashboard-footnote">Same-category context only; this does not prove a causal relationship.</p>{related.length ? <LogFeed entries={related} /> : <Empty icon={FileText} title="No other same-category event retained" />}</TabsContent>
        <TabsContent value="record"><h3>Normalized event record</h3><pre className="detection-raw">{JSON.stringify(entry, null, 2)}</pre><p className="dashboard-footnote">This is the console’s normalized event entry, not a raw video frame or an original UDP datagram.</p></TabsContent>
      </Tabs>
    </>}
  </DashboardSheet>;
}

function EventTimeline({ entries, onInspect, query, onQueryChange, filters }: { entries: LogEntry[]; onInspect: (entry: LogEntry) => void; query: string; onQueryChange: (query: string) => void; filters: ReactNode }) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(entries.length / 8));
  const current = Math.min(page, pages - 1);
  return <div className="logs-timeline-panel">
    <div className="data-table-toolbar"><div className="data-table-search"><Search aria-hidden="true" /><Input aria-label="Search mission event timeline" type="search" placeholder="Search event text, key, or details…" value={query} onChange={(event) => { onQueryChange(event.target.value); setPage(0); }} /></div>{filters}</div>
    {entries.length ? <ol className="logs-interactive-timeline">{entries.slice(current * 8, current * 8 + 8).map((entry) => <li key={entry.key}><Button variant="ghost" className="logs-timeline-entry" onClick={() => onInspect(entry)}><span className={cn("logs-severity-marker", "is-" + severity[entry.tone].badge)}><Activity /></span><span className="logs-timeline-content"><span><DashboardBadge tone={severity[entry.tone].badge}>{severity[entry.tone].label}</DashboardBadge><DashboardBadge tone="muted">{entry.category}</DashboardBadge><time dateTime={entry.rx_utc}>{fmtUtc(entry.rx_utc)} UTC</time></span><strong>{entry.text}</strong>{entry.detail && <small>{entry.detail}</small>}<small className="logs-observation">{entry.t_obs_s === null ? "Ground event" : "Observed " + formatObsTime(entry.t_obs_s) + " sim"}</small></span><ArrowUpRight /></Button></li>)}</ol> : <Empty icon={FileText} title="No retained event matches your filters" />}
    <div className="dashboard-timeline-pagination"><span>{entries.length ? current * 8 + 1 : 0}–{Math.min(entries.length, (current + 1) * 8)} of {entries.length} events</span><span>Page {current + 1} of {pages}</span><Button variant="outline" size="icon-sm" aria-label="Previous event timeline page" disabled={current === 0} onClick={() => setPage(current - 1)}><ChevronLeft /></Button><Button variant="outline" size="icon-sm" aria-label="Next event timeline page" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}><ChevronRight /></Button></div>
  </div>;
}

export function LogsPage() {
  const m = useMission();
  const gate = useMissionGate();
  const [category, setCategory] = useState("all");
  const [tone, setTone] = useState("all");
  const [query, setQuery] = useState("");
  const [view, setView] = useState("ledger");
  const [selection, setSelection] = useState<{ missionId: string | null; entry: LogEntry } | null>(null);
  const log = useMemo(() => m.state?.log ?? [], [m.state?.log]);
  const scoped = useMemo(() => filterEvents(log, category, tone, ""), [log, category, tone]);
  const shown = useMemo(() => filterEvents(log, category, tone, query), [log, category, tone, query]);
  const activity = useMemo(() => eventActivity(log), [log]);
  const faults = log.filter((entry) => entry.tone === "bad").length;
  const cautions = log.filter((entry) => entry.tone === "warn").length;
  const latest = log.length ? Date.parse(log[0]!.rx_utc) : null;
  const picked = selection?.missionId === m.missionId ? log.find((entry) => entry.key === selection.entry.key) ?? selection.entry : null;
  const inspect = (entry: LogEntry) => setSelection({ missionId: m.missionId, entry });
  const columns = useMemo<ColumnDef<LogEntry>[]>(() => [
    { accessorKey: "rx_utc", header: "Received · UTC", cell: ({ row }) => <time className="cw-mono" dateTime={row.original.rx_utc}>{fmtUtc(row.original.rx_utc)}</time> },
    { accessorKey: "tone", header: "Severity", cell: ({ row }) => <DashboardBadge tone={severity[row.original.tone].badge}>{severity[row.original.tone].label}</DashboardBadge> },
    { accessorKey: "category", header: "Category", cell: ({ row }) => <DashboardBadge tone="muted">{row.original.category}</DashboardBadge> },
    { accessorKey: "text", header: "Event", enableHiding: false, cell: ({ row }) => <div className="logs-event-cell"><Button variant="link" className="dashboard-record-id" onClick={() => setSelection({ missionId: m.missionId, entry: row.original })}>{row.original.text}<ArrowUpRight /></Button>{row.original.detail && <small>{row.original.detail}</small>}</div> },
    { accessorKey: "t_obs_s", header: "Observed · sim", cell: ({ row }) => <span className="cw-mono">{row.original.t_obs_s === null ? "Ground event" : formatObsTime(row.original.t_obs_s)}</span> },
  ], [m.missionId]);
  const filters = <><DashboardSelect label="Filter event category" value={category} onChange={setCategory} options={filterOptions.categories} /><DashboardSelect label="Filter event severity" value={tone} onChange={setTone} options={filterOptions.severity} />{(category !== "all" || tone !== "all" || query) && <Button variant="ghost" size="sm" onClick={() => { setCategory("all"); setTone("all"); setQuery(""); }}>Reset filters</Button>}</>;
  return <Page id="logs"><PageHeader title="Mission event intelligence" description="Trace reported health, detections, navigation, and link changes through the retained ground-station evidence." />
    {gate ?? <div className="dashboard-stack">
      <div className="dashboard-metrics"><DashboardMetric label="Retained events" value={log.length} detail={"Bounded console log · maximum " + MAX_LOG} icon={FileText} /><DashboardMetric label="Reported faults" value={faults} detail="Fault labels, not unique incidents" icon={ShieldAlert} tone={faults ? "fault" : "muted"} /><DashboardMetric label="Caution events" value={cautions} detail="Warnings in the retained window" icon={Activity} tone={cautions ? "caution" : "muted"} /><DashboardMetric label="Latest event age" value={latest !== null && Number.isFinite(latest) ? formatAge((m.groundNow - latest) / 1000) : "—"} detail="Ground reception time only" icon={Clock3} /></div>
      <div className="dashboard-split">
        <DashboardCard title="Event activity by severity" description="Retained events grouped by UTC reception minute." action={<DashboardBadge><Clock3 /> Ground clock</DashboardBadge>}>
          {activity.length ? <ChartContainer config={activityConfig} className="dashboard-chart"><BarChart data={activity} margin={{ top: 12, right: 10, bottom: 0, left: -24 }}><CartesianGrid vertical={false} stroke="#1b2e45" /><XAxis dataKey="utc" tickFormatter={(value: number) => fmtUtc(value).slice(0, 5)} tickLine={false} axisLine={false} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} /><ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => fmtUtc(Number(value)) + " UTC"} />} />{(["info", "good", "warn", "bad"] as LogTone[]).map((key) => <Bar key={key} dataKey={key} stackId="events" fill={severity[key].color} maxBarSize={40} isAnimationActive={m.motionEnabled} />)}</BarChart></ChartContainer> : <Empty icon={Activity} title="No retained activity to chart" />}
          <div className="dashboard-legend">{Object.entries(severity).map(([key, value]) => <span key={key}><i style={{ background: value.color }} />{value.label}</span>)}</div>
        </DashboardCard>
        <DashboardCard title="Subsystem composition" description="Choose a subsystem to filter the event ledger." action={category !== "all" && <Button size="sm" variant="ghost" onClick={() => setCategory("all")}>Show all</Button>}>
          <div className="logs-category-breakdown">{categories.map(({ key, Icon }) => { const count = log.filter((entry) => entry.category === key).length; return <Button variant="ghost" key={key} aria-pressed={category === key} className="logs-category-button" onClick={() => setCategory(category === key ? "all" : key)}><Icon /><span><span>{key}<strong>{count}</strong></span><Progress value={log.length ? count * 100 / log.length : 0} aria-label={key + " share of retained events"} /></span></Button>; })}</div>
        </DashboardCard>
      </div>
      <DashboardCard title="Retained event history" description="Search, filter, sort, or switch to a chronological timeline. Click an event for its inspection sheet." flush action={<DashboardBadge>{shown.length} / {log.length} events</DashboardBadge>}>
        <Tabs value={view} onValueChange={(value) => setView(value as string)}><div className="dashboard-view-switch"><TabsList aria-label="Event history view"><TabsTrigger value="ledger"><List /> Data ledger</TabsTrigger><TabsTrigger value="timeline"><Activity /> Timeline</TabsTrigger></TabsList><span>Newest reception first by default</span></div>
          <TabsContent value="ledger"><DataTable key={m.missionId + ":" + category + ":" + tone} label="mission events" data={scoped} columns={columns} getRowId={(entry) => entry.key} onRowClick={inspect} selectedRowId={picked?.key} initialSorting={[{ id: "rx_utc", desc: true }]} searchPlaceholder="Search event text, key, or details…" searchText={(entry) => [entry.key, entry.text, entry.detail, entry.category, entry.rx_utc].join(" ")} searchQuery={query} onSearchQueryChange={setQuery} toolbar={filters} emptyMessage="No retained event matches your filters." /></TabsContent>
          <TabsContent value="timeline"><EventTimeline key={m.missionId + ":" + category + ":" + tone} entries={shown} onInspect={inspect} query={query} onQueryChange={setQuery} filters={filters} /></TabsContent>
        </Tabs>
      </DashboardCard>
      <Alert className="dashboard-alert"><FileText /><AlertTitle>Retained evidence, not a complete audit archive</AlertTitle><AlertDescription>The console keeps at most {MAX_LOG} normalized events for this mission. Counts and charts describe this retained window only. Older removed tracks and events may not be recoverable from this view.</AlertDescription></Alert>
    </div>}
    <EventInspector key={m.missionId} entry={picked} onClose={() => setSelection(null)} />
  </Page>;
}
