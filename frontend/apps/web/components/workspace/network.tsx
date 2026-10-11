"use client";

import { useMemo, useState } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDownToLine, ArrowUpRight, Clock3, Gauge, RadioTower, Signal, Timer, TriangleAlert, Wifi, WifiOff } from "lucide-react";
import { Area, CartesianGrid, Cell, ComposedChart, Line, Pie, PieChart, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { fmt, fmtUtc } from "@/components/mission/shared";
import { formatAge } from "@/lib/ground/freshness";
import { freshnessLabel, useMission } from "@/lib/ground/mission-context";
import { linkWindow } from "@/lib/ground/dashboard-summary";
import type { LinkSample } from "@/lib/ground/mission-state";
import { PRESETS, type Preset } from "@/lib/ground/types";
import { cn } from "@/lib/utils";
import { DashboardBadge, DashboardCard, DashboardMetric, DashboardSelect, DashboardSheet } from "./dashboard";
import { Empty, ErrorNote, Facts, Page, PageHeader } from "./primitives";

const PRESET_INFO: Record<Preset, { description: string; detail: string; tone: "signal" | "caution" | "fault" | "nominal" }> = {
  NORMAL: { description: "Nominal delivery", detail: "0% loss · no added delay", tone: "signal" },
  DEGRADED: { description: "Impaired radio link", detail: "30% loss · 200 ± 50 ms", tone: "caution" },
  SEVERE: { description: "Constrained lunar link", detail: "50% loss · 1.3 ± 0.3 s · 1 kB/s", tone: "caution" },
  BLACKOUT: { description: "Complete interruption", detail: "Link cut · no telemetry delivered", tone: "fault" },
  RESTORE: { description: "Resume nominal delivery", detail: "0% loss · restoration recorded", tone: "nominal" },
};
const chartConfig = { rate: { label: "Packets/s", color: "#4cb4ff" }, loss: { label: "Window loss %", color: "#f2b13d" }, delay: { label: "Reported median · ms", color: "#b99aff" } };
const outcomeConfig = { forwarded: { label: "Forwarded", color: "#2fd69b" }, loss: { label: "Random loss", color: "#f2b13d" }, blackout: { label: "Blackout drop", color: "#ff5a6e" }, queue: { label: "Queue drop", color: "#b99aff" } };

function SampleInspector({ sample, onClose }: { sample: LinkSample | null; onClose: () => void }) {
  const m = useMission();
  return <DashboardSheet open={Boolean(sample)} onClose={onClose} eyebrow={<><RadioTower /> LINK SAMPLE INSPECTION</>} title={sample ? fmtUtc(sample.rx_ms) + " UTC" : "Link sample"}
    description={"Recorded network evidence for mission #" + m.missionId} tags={sample && <><DashboardBadge>{sample.preset ?? "Preset unknown"}</DashboardBadge><DashboardBadge tone={sample.link_up ? "nominal" : "caution"}>{sample.link_up ? "Receiver link up" : "Receiver link down"}</DashboardBadge></>}>
    {sample && <>
      <Alert className="dashboard-alert"><HistoryNoteIcon /><AlertTitle>Historical sample</AlertTitle><AlertDescription>This is a retained ground-clock snapshot, not the live emulator state. Opening it does not change the network.</AlertDescription></Alert>
      <div className="detection-sheet-metrics">
        <div><span>Reception rate</span><strong>{fmt(sample.packets_per_s, 1)}{sample.packets_per_s !== null && <small> pkts/s</small>}</strong><small>measured by the ground receiver</small></div>
        <div><span>Window packet loss</span><strong>{fmt(sample.loss_pct, 1)}{sample.loss_pct !== null && <small>%</small>}</strong><small>counter deltas in the sample window</small></div>
        <div><span>Reported median delay</span><strong>{fmt(sample.delay_p50_ms, 0)}{sample.delay_p50_ms !== null && <small> ms</small>}</strong><small>emulator cumulative median</small></div>
        <div><span>Recorded preset</span><strong className="dashboard-small-readout">{sample.preset ?? "—"}</strong><small>reported in this sample</small></div>
      </div>
      <h3>Source and clock</h3><Facts items={[["Mission", m.missionId ?? "—"], ["Reception timestamp", fmtUtc(sample.rx_ms, true) + " UTC"], ["Receiver link up", sample.link_up ? "Yes" : "No"], ["Clock domain", "Ground UTC"], ["Unknown values", "Shown as —, never replaced with zero"]]} />
      <Separator /><h3>How to interpret this evidence</h3><p className="dashboard-footnote">Window loss is derived from successive emulator reception and drop counters. Median delay is reported by the emulator; it is not computed by subtracting simulation time from UTC. A missing loss value can mean there was no valid comparison window.</p>
      <h3>Normalized sample record</h3><pre className="detection-raw">{JSON.stringify(sample, null, 2)}</pre>
    </>}
  </DashboardSheet>;
}
const HistoryNoteIcon = Clock3;

export function NetworkPage() {
  const m = useMission();
  const n = m.network;
  // A cached successful response is not a current control-plane answer after a polling failure.
  const st = n.error ? null : n.status;
  const liveAvailable = st !== null;
  const s = m.state;
  const req = n.request;
  const recorded = s?.link?.emulator ?? null;
  const outcomes = st ?? recorded;
  const rx = s?.link?.receiver;
  const [historyWindow, setHistoryWindow] = useState("all");
  const [metric, setMetric] = useState("traffic");
  const [selection, setSelection] = useState<{ missionId: string | null; sample: LinkSample } | null>(null);
  const samples = useMemo(() => linkWindow(s?.linkSamples ?? [], historyWindow === "all" ? null : Number(historyWindow)), [s?.linkSamples, historyWindow]);
  const data = useMemo(() => samples.map((sample) => ({ utc: sample.rx_ms, rate: sample.packets_per_s, loss: sample.loss_pct, delay: sample.delay_p50_ms })), [samples]);
  const last = samples[samples.length - 1];
  const blackout = st?.link_state !== "BLACKOUT" || n.blackoutSinceMs === null ? null : Math.max(0, (m.now - n.blackoutSinceMs) / 1000);
  const restoredGroundMs = n.restoredAtMs === null ? null : n.restoredAtMs + m.stream.clockOffsetMs;
  const recovery = restoredGroundMs !== null && s?.lastRxMs !== null && s?.lastRxMs !== undefined && s.lastRxMs >= restoredGroundMs ? (s.lastRxMs - restoredGroundMs) / 1000 : null;
  const tone = !st ? "muted" : st.link_state === "BLACKOUT" ? "fault" : st.configured_loss > 0 ? "caution" : "nominal";
  const distribution = outcomes ? [
    { name: "forwarded", value: outcomes.forwarded, fill: "#2fd69b" }, { name: "loss", value: outcomes.dropped_loss, fill: "#f2b13d" },
    { name: "blackout", value: outcomes.dropped_blackout, fill: "#ff5a6e" }, { name: "queue", value: outcomes.dropped_queue, fill: "#b99aff" },
  ] : [];
  const accounted = distribution.reduce((sum, entry) => sum + entry.value, 0);
  const picked = selection?.missionId === m.missionId ? selection.sample : null;
  const columns = useMemo<ColumnDef<LinkSample>[]>(() => [
    { accessorKey: "rx_ms", header: "Received · UTC", enableHiding: false, cell: ({ row }) => <Button variant="link" className="dashboard-record-id cw-mono" onClick={() => setSelection({ missionId: m.missionId, sample: row.original })}>{fmtUtc(row.original.rx_ms)}<ArrowUpRight /></Button> },
    { accessorKey: "preset", header: "Recorded preset", cell: ({ row }) => <DashboardBadge tone={row.original.preset === "BLACKOUT" ? "fault" : row.original.preset === "NORMAL" || row.original.preset === "RESTORE" ? "nominal" : "caution"}>{row.original.preset ?? "Unknown"}</DashboardBadge> },
    { accessorKey: "packets_per_s", header: "Packets/s", cell: ({ row }) => <span className="cw-mono">{fmt(row.original.packets_per_s, 1)}</span> },
    { accessorKey: "loss_pct", header: "Window loss", cell: ({ row }) => <span className="cw-mono">{fmt(row.original.loss_pct, 1, row.original.loss_pct === null ? "" : "%")}</span> },
    { accessorKey: "delay_p50_ms", header: "Median delay", cell: ({ row }) => <span className="cw-mono">{fmt(row.original.delay_p50_ms, 0, row.original.delay_p50_ms === null ? "" : " ms")}</span> },
    { accessorKey: "link_up", header: "Receiver", cell: ({ row }) => <DashboardBadge tone={row.original.link_up ? "nominal" : "muted"}>{row.original.link_up ? "Link up" : "Link down"}</DashboardBadge> },
  ], [m.missionId]);
  const feedback = !liveAvailable ? "Live emulator control unavailable. Historical mission evidence remains readable."
    : req.phase === "applying" ? "Applying " + req.preset + "… awaiting the emulator"
    : req.phase === "confirmed" ? req.preset + " confirmed by the emulator at " + fmtUtc(req.atMs) + " UTC"
    : req.phase === "failed" ? req.preset + " not applied: " + req.message
    : "The highlighted preset is the emulator’s confirmed current preset.";

  return <Page id="network"><PageHeader title="Communication & link health" description="Operate the shared lunar-link emulator and inspect the selected mission’s recorded network evidence." />
    <div className="dashboard-stack">
      <div className="dashboard-split">
        <DashboardCard title="Current emulator condition" description="Live control-plane response · shared across rover missions" action={<DashboardBadge tone={tone}>{st ? "Emulator responding" : "Control unavailable"}</DashboardBadge>} className={cn("network-condition", "is-" + tone)}>
          <div className="network-condition-hero"><div className="network-antenna">{st?.link_state === "BLACKOUT" ? <WifiOff /> : <RadioTower />}</div><div><span className="dashboard-overline">EARTH ↔ MOON</span><strong>{st?.link_state ?? "Unavailable"}</strong><p>{st ? "Confirmed preset " + st.preset : recorded ? "Last mission snapshot: " + recorded.link_state + " · " + recorded.preset : "Waiting for a live emulator response"}</p></div></div>
          <div className="network-session-strip"><span><i />Dashboard stream: {m.stream.connection}</span><span>Selected mission <b>#{m.missionId ?? "—"}</b></span>{!st && s?.link?.rx_utc && <span>Snapshot {fmtUtc(s.link.rx_utc)} UTC</span>}</div>
          {blackout !== null && <Alert className="dashboard-alert"><Timer /><AlertTitle>Blackout in progress · {blackout.toFixed(0)} s observed</AlertTitle><AlertDescription>Keep the last received picture. No new rover data should be assumed.</AlertDescription></Alert>}
          {n.restoredAtMs !== null && blackout === null && <p className="dashboard-footnote">{recovery === null ? "Restoration observed; waiting for this mission’s first newly received packet." : "This mission’s latest packet arrived " + recovery.toFixed(1) + " s after restoration was observed in this browser."}</p>}
        </DashboardCard>
        <DashboardCard title="Ground receiver" description="Selected mission’s latest received snapshot" action={<span title={m.freshLabel}><DashboardBadge>{freshnessLabel[m.fresh]}</DashboardBadge></span>}>
          {m.stream.error && <ErrorNote message={m.stream.error} onRetry={m.stream.reload} />}
          <Facts items={[["Valid / rejected", rx ? (rx.valid ?? 0) + " / " + (rx.invalid ?? 0) : "—"], ["Missing / duplicate / late", rx ? (rx.missing ?? 0) + " / " + (rx.duplicates ?? 0) + " / " + (rx.reordered ?? 0) : "—"], ["ACKs sent", rx?.acks_sent ?? "—"], ["Reception rate", fmt(s?.link?.packets_per_s, 1, " pkts/s")], ["Last rover packet", formatAge(m.ageS) + " ago"], ["Stream reconnects", m.stream.reconnects]]} />
        </DashboardCard>
      </div>
      <div className="dashboard-metrics">
        <DashboardMetric label="Measured window loss" value={fmt(last?.loss_pct, 1, last?.loss_pct === null || last?.loss_pct === undefined ? "" : "%")} detail="Last retained mission sample · not configured loss" icon={Signal} tone={(last?.loss_pct ?? 0) > 30 ? "caution" : "signal"} />
        <DashboardMetric label="Reported median delay" value={fmt(outcomes?.delay_ms.p50, 0, outcomes?.delay_ms.p50 === null || outcomes?.delay_ms.p50 === undefined ? "" : " ms")} detail={st ? "Live emulator · cumulative median" : "Last recorded emulator snapshot"} icon={Clock3} />
        <DashboardMetric label="Live throughput" value={fmt(st?.throughput_Bps, 0, st?.throughput_Bps === null || st?.throughput_Bps === undefined ? "" : " B/s")} detail={!st ? "Live emulator unavailable" : st.config.bandwidth_Bps === null ? "No configured bandwidth cap" : "Cap " + st.config.bandwidth_Bps + " B/s"} icon={Gauge} />
        <DashboardMetric label="Blackout duration" value={blackout !== null ? blackout.toFixed(0) + " s" : n.lastBlackoutS !== null ? n.lastBlackoutS.toFixed(1) + " s" : "—"} detail={blackout !== null ? "Ongoing · observed in this browser" : "Last completed blackout observed this session"} icon={Timer} tone={blackout !== null ? "fault" : "muted"} />
      </div>
      <DashboardCard title="Link scenario controls" description="These controls affect the shared live emulator, not just the mission pinned in the URL." action={<DashboardBadge tone={liveAvailable ? "nominal" : "muted"}>{liveAvailable ? "Control ready" : "Read-only history"}</DashboardBadge>}>
        <div className="network-preset-grid" role="group" aria-label="Lunar link presets">{PRESETS.map((preset) => <Button key={preset} variant="outline" className={cn("network-preset", "is-" + PRESET_INFO[preset].tone, st?.preset === preset && "is-active")} aria-pressed={st?.preset === preset} disabled={req.phase === "applying" || !liveAvailable} onClick={() => void n.apply(preset)} title={"Apply " + preset + " to the shared emulator"}><span><b>{req.phase === "applying" && req.preset === preset ? "Applying…" : preset}</b><span className="network-preset-dot" /></span><strong>{PRESET_INFO[preset].description}</strong><small>{PRESET_INFO[preset].detail}</small></Button>)}</div>
        <p className={cn("network-control-feedback", (req.phase === "failed" || !liveAvailable) && "is-error")} role="status">{feedback}</p>
        {!liveAvailable && n.error && <details className="dashboard-technical"><summary>Control connection details</summary><p>{n.error}</p></details>}
      </DashboardCard>
      <div className="dashboard-split">
        <DashboardCard title="Recorded link performance" description="Ground UTC · the window ends at the last retained sample, including for archived missions." action={<DashboardSelect label="Link history window" value={historyWindow} onChange={setHistoryWindow} options={[["all", "All retained"], ["60", "Last sample minute"], ["300", "Last 5 sample minutes"]]} />}>
          <Tabs value={metric} onValueChange={(value) => setMetric(value as string)}><TabsList aria-label="Link chart metrics"><TabsTrigger value="traffic">Traffic & loss</TabsTrigger><TabsTrigger value="delay">Median delay</TabsTrigger></TabsList></Tabs>
          {data.length > 1 ? <ChartContainer config={chartConfig} className="dashboard-chart network-performance-chart"><ComposedChart data={data} margin={{ top: 15, right: 6, bottom: 0, left: -16 }}><CartesianGrid vertical={false} stroke="#1b2e45" /><XAxis dataKey="utc" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(value: number) => fmtUtc(value).slice(0, 5)} tickLine={false} axisLine={false} minTickGap={40} /><YAxis yAxisId="primary" tickLine={false} axisLine={false} />{metric === "traffic" && <YAxis yAxisId="loss" orientation="right" domain={[0, 100]} tickFormatter={(value: number) => value + "%"} tickLine={false} axisLine={false} width={42} />}<ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => fmtUtc(Number(value)) + " UTC"} />} />{metric === "traffic" ? <><Area yAxisId="primary" dataKey="rate" type="linear" fill="var(--color-rate)" fillOpacity={.15} stroke="var(--color-rate)" strokeWidth={2} connectNulls={false} isAnimationActive={m.motionEnabled} /><Line yAxisId="loss" dataKey="loss" type="stepAfter" stroke="var(--color-loss)" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={m.motionEnabled} /></> : <Line yAxisId="primary" dataKey="delay" type="linear" stroke="var(--color-delay)" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={m.motionEnabled} />}</ComposedChart></ChartContainer> : <Empty icon={Wifi} title="Not enough received link samples to chart" />}
          <div className="dashboard-legend">{metric === "traffic" ? <><span><i style={{ background: "#4cb4ff" }} />Received packets/s</span><span><i style={{ background: "#f2b13d" }} />Window loss %</span></> : <span><i style={{ background: "#b99aff" }} />Emulator cumulative median · ms</span>}</div>
        </DashboardCard>
        <DashboardCard title="Emulator outcome counters" description={st ? "Live shared emulator counters" : "Last emulator snapshot received for this mission"} action={<DashboardBadge tone={st ? "nominal" : "muted"}>{st ? "Live control" : "Recorded"}</DashboardBadge>}>
          {accounted ? <div className="dashboard-donut-wrap"><ChartContainer config={outcomeConfig} className="dashboard-donut"><PieChart><ChartTooltip content={<ChartTooltipContent nameKey="name" hideLabel />} /><Pie data={distribution.filter((entry) => entry.value > 0)} dataKey="value" nameKey="name" innerRadius={55} outerRadius={74} paddingAngle={2} stroke="none" isAnimationActive={m.motionEnabled}>{distribution.filter((entry) => entry.value > 0).map((entry) => <Cell key={entry.name} fill={entry.fill} />)}</Pie></PieChart></ChartContainer><div className="dashboard-donut-label"><strong>{outcomes?.delivery_ratio === null || outcomes?.delivery_ratio === undefined ? "—" : Math.round(outcomes.delivery_ratio * 100) + "%"}</strong><span>reported delivery</span></div></div> : <Empty icon={ArrowDownToLine} title="No emulator outcome counters received" />}
          <div className="dashboard-distribution">{distribution.map((entry) => <span key={entry.name}><i style={{ background: entry.fill }} />{outcomeConfig[entry.name as keyof typeof outcomeConfig].label}<strong>{entry.value}</strong></span>)}</div>
          <p className="dashboard-footnote">Reported outcomes exclude in-flight or pending packets. These counters belong to the shared emulator.</p>
          {st && <><Separator /><div className="network-queue"><span>Queued <b>{st.queued} / {st.config.queue_max}</b></span><Progress value={st.config.queue_max ? Math.min(100, st.queued * 100 / st.config.queue_max) : 0} aria-label="Emulator queue occupancy" /><small>{st.in_flight} packets in flight</small></div></>}
        </DashboardCard>
      </div>
      <DashboardCard title="Network evidence ledger" description="Click a sample to inspect the clock, recorded preset, and measurement definitions." flush action={<DashboardBadge>{samples.length} retained samples</DashboardBadge>}>
        <DataTable key={m.missionId + ":" + historyWindow} label="network samples" columns={columns} data={samples} getRowId={(sample) => String(sample.rx_ms)} initialSorting={[{ id: "rx_ms", desc: true }]}
          onRowClick={(sample) => setSelection({ missionId: m.missionId, sample })} selectedRowId={picked ? String(picked.rx_ms) : null} searchPlaceholder="Search recorded presets or reception timestamps…" emptyMessage="No retained link sample for this mission." searchText={(sample) => (sample.preset ?? "") + " " + fmtUtc(sample.rx_ms, true)} />
      </DashboardCard>
      <Alert className="dashboard-alert"><TriangleAlert /><AlertTitle>Control state and mission history are separate</AlertTitle><AlertDescription>The preset is considered applied only after a fresh emulator status confirms it. During stale data or blackout, recorded rows remain historical; no new telemetry is invented.</AlertDescription></Alert>
    </div>
    <SampleInspector key={m.missionId} sample={picked} onClose={() => setSelection(null)} />
  </Page>;
}
