"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtUtc } from "@/components/mission/shared";
import { describeError, groundApi } from "@/lib/ground/api";
import { formatAge, formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { TRAJECTORY_GAP_S } from "@/lib/ground/trajectory";
import type { TelemetryRecord } from "@/lib/ground/types";
import { cn } from "@/lib/utils";
import { Empty, ErrorNote, Facts, Kpi, Page, PageHeader, Section, freshTone, useMissionGate } from "./primitives";

export const chartTheme = {
  grid: "#16283d",
  axis: "#3b5672",
  tick: { fill: "#93a9c4", fontSize: 11 },
  tooltip: { background: "#0a1626", border: "1px solid #1b2e45", borderRadius: 8, fontSize: 12 },
};

const POLL_MS = 3000;

/** Latest stored packets (REST, validated packets only), refreshed while the page is open. */
function useRecentPackets(missionId: string | null) {
  const [loaded, setLoaded] = useState<{ missionId: string; rows: TelemetryRecord[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!missionId) return;
    const abort = new AbortController();
    const load = () =>
      groundApi
        .telemetry(missionId, { limit: 100 }, abort.signal)
        .then((env) => {
          if (abort.signal.aborted) return;
          setLoaded({ missionId, rows: env.data });
          setError(null);
        })
        .catch((e: unknown) => {
          if (!abort.signal.aborted) setError(describeError(e));
        });
    void load();
    const t = window.setInterval(() => document.visibilityState === "visible" && void load(), POLL_MS);
    return () => {
      abort.abort();
      window.clearInterval(t);
    };
  }, [missionId]);
  return { rows: loaded?.missionId === missionId ? loaded.rows : [], error };
}

export function TelemetryPage() {
  const m = useMission();
  const gate = useMissionGate();
  const s = m.state;
  const p = m.latest;
  const rx = s?.link?.receiver;
  const packets = useRecentPackets(m.missionId);
  const packetColumns = useMemo<ColumnDef<TelemetryRecord>[]>(() => [
    { accessorKey: "seq", header: "Seq", enableHiding: false, cell: ({ row }) => <span className="cw-mono">{row.original.seq}</span> },
    { accessorKey: "packet_type", header: "Type" },
    { accessorKey: "t_obs_s", header: "Observed (sim)", cell: ({ row }) => <span className="cw-mono">{formatObsTime(row.original.t_obs_s)}</span> },
    { accessorKey: "rx_utc", header: "Received (UTC)", cell: ({ row }) => <span className="cw-mono">{fmtUtc(row.original.rx_utc)}</span> },
    { accessorKey: "seq_flag", header: "Flag", cell: ({ row }) => <span className={cn("cw-mono", row.original.seq_flag !== "new" && "cw-warn")}>{row.original.seq_flag}</span> },
  ], []);

  // pose over observation time; a reception gap becomes a break in the line (null point), never an interpolation
  const poseSeries = useMemo(() => {
    const t = s?.trajectory ?? [];
    const step = Math.max(1, Math.ceil(t.length / 600));
    const out: { t: number; x: number | null; y: number | null; sigma: number | null }[] = [];
    t.forEach((q, i) => {
      if (i % step && i !== t.length - 1) return;
      const prev = out[out.length - 1];
      if (prev && prev.x !== null && q.t_obs_s - prev.t > TRAJECTORY_GAP_S * step)
        out.push({ t: (prev.t + q.t_obs_s) / 2, x: null, y: null, sigma: null });
      out.push({ t: q.t_obs_s, x: q.x_m, y: q.y_m, sigma: q.sigma_xy_saturated ? null : q.sigma_xy_m });
    });
    return out;
  }, [s?.trajectory]);
  const rateSeries = useMemo(
    () => (s?.linkSamples ?? []).map((l) => ({ t: l.rx_ms, rate: l.packets_per_s })),
    [s?.linkSamples],
  );

  return (
    <Page id="telemetry">
      <PageHeader
        title="Telemetry"
        description="Engineering view of what the rover sent and what reached the ground. Observation time is the rover's simulation clock; reception time is UTC."
      />
      {gate ?? (
        <>
          <div className="cw-grid cw-kpis">
            <Kpi label="Position (odom)" value={p ? `${p.x_m.toFixed(2)}, ${p.y_m.toFixed(2)}` : "—"} unit={p ? "m" : undefined} context={p ? `observed t ${formatObsTime(p.t_obs_s)}` : "no pose yet"} />
            <Kpi label="Heading" value={p ? p.heading_deg.toFixed(1) : "—"} unit={p ? "°" : undefined} context="yaw in the odom frame" />
            <Kpi
              label="Position uncertainty"
              value={p ? `±${p.sigma_xy_m.toFixed(2)}` : "—"}
              unit={p ? "m" : undefined}
              context={p?.sigma_xy_saturated ? "saturated: odometry drift beyond the encodable range" : "1σ, odometry estimate"}
              tone={p?.sigma_xy_saturated ? "warn" : "signal"}
            />
            <Kpi label="Data age" value={formatAge(m.ageS)} context="since the last rover packet (UTC)" tone={freshTone[m.fresh]} />
          </div>

          <div className="cw-grid">
            <Section title="Position over observation time" aside={<span className="cw-mono">{s?.trajectory.length ?? 0} poses</span>} className="span-8">
              <div className="cw-chart">
                {poseSeries.length > 1 ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <LineChart data={poseSeries} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                      <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                      <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v: number) => formatObsTime(v).slice(3, 8)} tick={chartTheme.tick} stroke={chartTheme.axis} minTickGap={48} />
                      <YAxis tick={chartTheme.tick} stroke={chartTheme.axis} width={40} unit=" m" />
                      <Tooltip contentStyle={chartTheme.tooltip} labelFormatter={(v) => `t ${formatObsTime(Number(v))} sim`} formatter={(v) => (typeof v === "number" ? `${v.toFixed(2)} m` : "—")} />
                      <Legend wrapperStyle={{ fontSize: 12, color: "#93a9c4" }} />
                      <Line dataKey="x" name="x" stroke="#4cb4ff" strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls={false} />
                      <Line dataKey="y" name="y" stroke="#2fd69b" strokeWidth={1.8} dot={false} isAnimationActive={false} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Empty icon={Activity} title="Waiting for poses" />
                )}
              </div>
            </Section>
            <Section title="Packet reception" className="span-4">
              <Facts
                items={[
                  ["Packets / s", <span key="r" className="cw-mono">{s?.link?.packets_per_s?.toFixed(1) ?? "—"}</span>],
                  ["Valid packets", <span key="v" className="cw-mono">{rx?.valid ?? "—"}</span>],
                  ["Rejected (CRC / decode)", <span key="i" className="cw-mono">{rx?.invalid ?? "—"}</span>],
                  ["Missing sequence numbers", <span key="m" className="cw-mono">{rx?.missing ?? "—"}</span>],
                  ["Duplicates", <span key="d" className="cw-mono">{rx?.duplicates ?? "—"}</span>],
                  ["Late (reordered)", <span key="l" className="cw-mono">{rx?.reordered ?? "—"}</span>],
                  ["ACKs sent to the rover", <span key="a" className="cw-mono">{rx?.acks_sent ?? "—"}</span>],
                ]}
              />
            </Section>
          </div>

          <div className="cw-grid">
            <Section title="Reception rate" aside={<span className="cw-mono">ground clock · UTC</span>} className="span-7">
              <div className="cw-chart is-short">
                {rateSeries.length > 1 ? (
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <LineChart data={rateSeries} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                      <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                      <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v: number) => fmtUtc(v).slice(0, 5)} tick={chartTheme.tick} stroke={chartTheme.axis} minTickGap={48} />
                      <YAxis tick={chartTheme.tick} stroke={chartTheme.axis} width={32} allowDecimals={false} />
                      <Tooltip contentStyle={chartTheme.tooltip} labelFormatter={(v) => `${fmtUtc(Number(v))} UTC`} />
                      <Line dataKey="rate" name="packets/s" type="stepAfter" stroke="#4cb4ff" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Empty icon={Activity} title="Waiting for link samples">
                    The ground station samples reception every 1–2 s while the mission is receiving.
                  </Empty>
                )}
              </div>
            </Section>
            <Section title="Latest stored packets" aside={<span className="cw-mono">latest 100 · every 3 s</span>} className="span-5" flush>
              {packets.error && <ErrorNote message={packets.error} />}
              {packets.rows.length ? (
                <DataTable key={m.missionId} label="stored telemetry packets" data={packets.rows} columns={packetColumns}
                  getRowId={(r) => String(r.seq)} initialSorting={[{ id: "seq", desc: true }]} searchPlaceholder="Search sequence, type, or flag…" />
              ) : (
                !packets.error && <Empty icon={Activity} title="No stored packet yet" />
              )}
            </Section>
          </div>

          <details className="cw-surface cw-details">
            <summary>Technical details</summary>
            <Facts
              items={[
                ["WebSocket stream", `${m.stream.connection}${m.stream.reconnects ? ` · ${m.stream.reconnects} reconnect(s)` : ""}`],
                ["Last resync source", s?.snapshotSource ?? "—"],
                ["Snapshots received", <span key="sn" className="cw-mono">{s?.snapshots ?? 0}</span>],
                ["Events of other missions ignored", <span key="ig" className="cw-mono">{s?.ignoredForeign ?? 0}</span>],
                ["Ground ↔ browser clock offset", <span key="co" className="cw-mono">{(m.stream.clockOffsetMs / 1000).toFixed(2)} s</span>],
                ["Ingest queue", <span key="q" className="cw-mono">{s?.link?.ingest ? `${s.link.ingest.queue} queued · ${s.link.ingest.dropped} dropped` : "—"}</span>],
                ["Database writes", s?.link?.ingest ? (s.link.ingest.db_ok ? "healthy" : "retrying") : "—"],
                ["Receiver", <span key="rc" className="cw-mono">{String(s?.mission?.config?.receiver ?? "—")}</span>],
                ["Protocol", <span key="pr" className="cw-mono">{String(s?.mission?.config?.protocol ?? "—")}</span>],
              ]}
            />
          </details>
        </>
      )}
    </Page>
  );
}
