"use client";

// Unified Mission Control: the all-in-one operator command center (R6), unchanged in layout, bindings and styling.
// Shown on "/" when Settings > Unified Mission Control is on.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  CheckCircle2,
  Compass,
  Database,
  FileText,
  Flag,
  Gauge,
  History,
  LocateFixed,
  Map as MapIcon,
  Maximize2,
  Navigation,
  Plus,
  RadioTower,
  RefreshCw,
  Route,
  ScanEye,
  Signal,
  Timer,
  TriangleAlert,
  Wifi,
  WifiOff,
  X,
  type LucideIcon,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { LunarMap } from "@/components/mission/lunar-map";
import {
  ClockGlyph,
  EmptyState,
  FreshnessPill,
  Panel,
  RockGlyph,
  Scene,
  StatusPill,
  WarningGlyph,
  compassPoint,
  connectionText,
  fmt,
  fmtUtc,
  scenes,
  typeName,
  typeShort,
} from "@/components/mission/shared";
import { cn } from "@/lib/utils";
import { formatAge, formatObsTime, type Freshness } from "@/lib/ground/freshness";
import { freshnessLabel, useMission } from "@/lib/ground/mission-context";
import {
  latestPosition,
  type LogCategory,
  type LogEntry,
  type MissionState,
  type TrackedObstacle,
} from "@/lib/ground/mission-state";
import type { MissionStream } from "@/lib/ground/use-mission";
import type { MissionList, NetworkControl } from "@/lib/ground/use-network";
import { PRESETS, type Mission } from "@/lib/ground/types";
import { UNIFIED_TARGETS, useConsoleNav } from "./console-nav";

type DialogName = "map" | "mini" | null;


function SurfaceMap({
  stream,
  fresh,
  ageS,
  hasMission,
  unreachable,
  onExpand,
  highlighted,
  animate,
}: {
  stream: MissionStream;
  fresh: Freshness;
  ageS: number | null;
  hasMission: boolean;
  unreachable: boolean;
  onExpand: () => void;
  highlighted: string | null;
  animate: boolean;
}) {
  const [legendOpen, setLegendOpen] = useState(true);
  useEffect(() => {
    // on laptop-size screens the legend would hide part of the map: start collapsed there
    if (window.innerWidth < 1600) setLegendOpen(false);
  }, []);
  const [selected, setSelected] = useState<string | null>(null);
  const state = stream.state;
  const obstacles = useMemo(() => Object.values(state?.obstacles ?? {}), [state?.obstacles]);
  const trajectory = state?.trajectory ?? [];
  const sel = selected ? state?.obstacles[selected] : undefined;
  const stale = fresh === "STALE" || fresh === "DISCONNECTED";
  return (
    <Panel
      id="surface"
      title="Lunar Surface Map"
      subtitle="Received odometry & reconciled obstacles (odom frame)"
      highlighted={highlighted}
    >
      <div className={cn("surface-scene live-surface", animate && fresh === "LIVE" && "is-animated")}>
        {unreachable ? (
          <EmptyState icon={WifiOff} title="Ground station unreachable">
            No map is shown without the ground station: nothing is drawn from local data.
          </EmptyState>
        ) : !hasMission ? (
          <EmptyState icon={MapIcon} title="No mission recorded">
            Start the rover with ARTEMIS_TELEMETRY=1 and the ground station; the map draws only received telemetry.
          </EmptyState>
        ) : stream.error && !stream.loading && !state?.mission ? (
          <EmptyState icon={TriangleAlert} title="Mission unavailable">
            {stream.error}
          </EmptyState>
        ) : stream.loading && !trajectory.length ? (
          <EmptyState icon={RefreshCw} title="Loading mission picture…" />
        ) : !trajectory.length && !obstacles.length ? (
          <EmptyState icon={LocateFixed} title="No pose received yet">
            The rover position appears with the first POSE packet that reaches the ground.
          </EmptyState>
        ) : (
          <LunarMap
            trajectory={trajectory}
            obstacles={obstacles}
            initialMode="fit"
            interactive
            stale={stale}
            selectedId={selected}
            onSelect={setSelected}
            label={`Lunar map: ${trajectory.length} received poses, ${obstacles.length} obstacles`}
          />
        )}
        {stale && hasMission && (trajectory.length > 0 || obstacles.length > 0) && (
          <div className={cn("stale-banner", fresh === "DISCONNECTED" && "is-disconnected")} role="status">
            {fresh === "DISCONNECTED" ? <WifiOff /> : <Timer />}
            {fresh === "DISCONNECTED"
              ? "Ground station stream disconnected — showing the last received picture"
              : `STALE — no telemetry for ${formatAge(ageS)} · last received picture kept`}
          </div>
        )}
        <AnimatePresence>
          {sel && (
            <motion.div
              className="map-detail"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <Flag />
              <div>
                <strong>
                  {typeName[sel.type] ?? sel.type} {sel.obstacle_id}
                </strong>
                <span>
                  ({sel.x_m.toFixed(2)}, {sel.y_m.toFixed(2)}) m · r {sel.radius_m.toFixed(2)} m · confidence{" "}
                  {(sel.confidence * 100).toFixed(0)}% · uncertainty ±{sel.uncertainty_m.toFixed(2)} m
                  {sel.uncertainty_saturated ? " (saturated)" : ""} · {sel.source} · {sel.status}
                  {sel.persistent_id ? " · persistent track" : ""} · obs t {formatObsTime(sel.obs_timestamp_s)}
                </span>
              </div>
              <button
                className="icon-button"
                onClick={() => setSelected(null)}
                aria-label="Close obstacle details"
              >
                <X />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
        <button
          className="map-expand icon-button"
          onClick={onExpand}
          aria-label="Expand surface map"
        >
          <Maximize2 />
        </button>
      </div>
      <div className={cn("map-legend", !legendOpen && "is-collapsed")}>
        <button
          className="legend-toggle"
          onClick={() => setLegendOpen((v) => !v)}
          aria-expanded={legendOpen}
          title={legendOpen ? "Hide legend" : "Show legend"}
        >
          Legend
          {legendOpen ? <X /> : <Plus />}
        </button>
        {legendOpen && (
          <>
            <div>
              <span className="legend-rover" />
              Rover (last pose)
            </div>
            <div>
              <span className="legend-trail" />
              Received path
            </div>
            <div>
              <span className="legend-badge rock" />
              Detected rock
            </div>
            <div>
              <span className="legend-badge hazard" />
              Hazard (negative)
            </div>
            <div>
              <span className="legend-badge obstacle" />
              Obstacle (depth)
            </div>
            <div>
              <span className="legend-ring" />
              Uncertainty
            </div>
            <div>
              <span className="legend-explored" />
              Explored (camera)
            </div>
            <div>
              <span className="legend-area" />
              Traversed (free)
            </div>
            <div>
              <span className="legend-unknown" />
              Unknown
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}

// ---- Mission Overview (status) ------------------------------------------------------------------------------------------


function MissionStatusPanel({
  stream,
  fresh,
  ageS,
  now,
  missionsError,
  hasMission,
  highlighted,
}: {
  stream: MissionStream;
  fresh: Freshness;
  ageS: number | null;
  now: number;
  missionsError: string | null;
  hasMission: boolean;
  highlighted: string | null;
}) {
  const s = stream.state;
  const m = s?.mission;
  const health = s?.health;
  const latest = s ? latestPosition(s) : null;
  const rows: [string, ReactNode][] = [
    ["Mission", m ? `#${m.mission_id} · session ${m.session ?? "—"}` : "—"],
    ["Status", m?.status ?? "—"],
    ["Last ground reception", s?.lastRxMs ? `${fmtUtc(s.lastRxMs)} UTC · ${formatAge(ageS)} ago` : "—"],
    ["Last observation (rover)", latest ? `t = ${formatObsTime(latest.t_obs_s)} (sim)` : "—"],
    [
      "Rover health",
      health ? (
        <span className={cn("health-value", `is-${health.state.toLowerCase()}`)}>
          {health.state}
          {health.reasons?.length ? ` · ${health.reasons.join(", ")}` : ""}
        </span>
      ) : (
        "not reported"
      ),
    ],
    [
      "Ground stream",
      `${connectionText(stream.connection, stream.retryAtMs, now)}${stream.reconnects ? ` · ${stream.reconnects} reconnect(s)` : ""}`,
    ],
  ];
  const error = stream.error ?? missionsError;
  return (
    <Panel
      id="overview-status"
      slot="camera"
      title={
        <>
          Mission Overview <span>(Ground station)</span>
        </>
      }
      action={<FreshnessPill value={fresh} />}
      highlighted={highlighted}
    >
      <div className="status-body">
        {!hasMission && !error ? (
          <EmptyState icon={Database} title="Waiting for a mission">
            Missions appear when the first rover packet reaches the ground station.
          </EmptyState>
        ) : (
          <dl className="status-grid">
            {rows.map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        )}
        {error && (
          <p className="api-error" role="alert">
            <TriangleAlert />
            {error}
            <button onClick={stream.reload}>Retry</button>
          </p>
        )}
        {s?.snapshotSource === "unavailable" && (
          <p className="api-error" role="status">
            <Database />
            Ground database unavailable: live events only until it recovers.
          </p>
        )}
      </div>
    </Panel>
  );
}

// ---- Obstacle picture -------------------------------------------------------------------------------------------------

function DetectionFeed({
  state,
  highlighted,
}: {
  state: MissionState | null;
  highlighted: string | null;
}) {
  const latest = state ? latestPosition(state) : null;
  const rows = useMemo(() => {
    const list = Object.values(state?.obstacles ?? {});
    const d = (o: TrackedObstacle) => (latest ? Math.hypot(o.x_m - latest.x_m, o.y_m - latest.y_m) : 0);
    return list.map((o) => ({ o, dist: latest ? d(o) : null })).sort((a, b) => (a.dist ?? 0) - (b.dist ?? 0));
  }, [state?.obstacles, latest]);
  return (
    <Panel
      id="detection"
      title={
        <>
          Obstacle Picture <span>(YOLO + depth, reconciled)</span>
        </>
      }
      action={<span className="inference-pill">{rows.length} present</span>}
      highlighted={highlighted}
    >
      <div className="obstacle-table" role="table" aria-label="Present obstacles">
        {rows.length === 0 ? (
          <EmptyState icon={ScanEye} title="No obstacle reported">
            Detections appear when an OBSTACLES packet reaches the ground.
          </EmptyState>
        ) : (
          <>
            <div className="obstacle-row is-head" role="row">
              <span role="columnheader">ID</span>
              <span role="columnheader">Type</span>
              <span role="columnheader">Range</span>
              <span role="columnheader">Conf.</span>
              <span role="columnheader">±Unc.</span>
              <span role="columnheader">Source</span>
              <span role="columnheader">Track</span>
            </div>
            {rows.map(({ o, dist }) => (
              <div className={cn("obstacle-row", `type-${o.type}`)} role="row" key={o.obstacle_id}>
                <span role="cell" title={o.obstacle_id}>
                  {o.obstacle_id}
                </span>
                <span role="cell">{typeShort[o.type] ?? o.type}</span>
                <span role="cell">{dist === null ? "—" : `${dist.toFixed(1)} m`}</span>
                <span role="cell">{(o.confidence * 100).toFixed(0)}%</span>
                <span role="cell">
                  {o.uncertainty_m.toFixed(2)}
                  {o.uncertainty_saturated ? "+" : ""} m
                </span>
                <span role="cell">{o.source}</span>
                <span role="cell" className={`track-${o.status}`}>
                  {o.status}
                </span>
              </div>
            ))}
          </>
        )}
      </div>
    </Panel>
  );
}

// ---- Analytics ---------------------------------------------------------------------------------------------------------

function AnalyticsPanel({
  state,
  highlighted,
}: {
  state: MissionState | null;
  highlighted: string | null;
}) {
  const obstacles = Object.values(state?.obstacles ?? {});
  const rocks = obstacles.filter((o) => o.type === "rock");
  const hazards = obstacles.filter((o) => o.type !== "rock");
  const confirmed = (list: TrackedObstacle[]) => list.filter((o) => o.status === "confirmed").length;
  const counters = state?.mission?.counters ?? {};
  const rx = state?.link?.receiver;
  const valid = rx?.valid ?? null;
  const missing = rx?.missing ?? null;
  const completeness = valid !== null && missing !== null && valid + missing > 0 ? (100 * valid) / (valid + missing) : null;
  const delivery = state?.link?.emulator?.delivery_ratio;
  const pct = (a: number, b: number) => (b > 0 ? (100 * a) / b : null);
  const metrics: {
    label: string;
    value: string;
    caption: string;
    icon: LucideIcon;
    color: string;
    percent: number | null;
    ringLabel: string;
  }[] = [
    {
      label: "Rocks Present",
      value: state ? String(rocks.length) : "—",
      caption: state ? `${confirmed(rocks)} confirmed` : "no mission loaded",
      icon: Route,
      color: "#9ac3ff",
      percent: pct(confirmed(rocks), rocks.length),
      ringLabel: "confirmed share",
    },
    {
      label: "Hazards Present",
      value: state ? String(hazards.length) : "—",
      caption: state ? `${confirmed(hazards)} confirmed` : "no mission loaded",
      icon: TriangleAlert,
      color: "#ff5276",
      percent: pct(confirmed(hazards), hazards.length),
      ringLabel: "confirmed share",
    },
    {
      label: "Packets Stored",
      value: counters.packets !== undefined ? String(counters.packets) : "—",
      caption:
        !state
          ? "no mission loaded"
          : completeness === null
            ? `${counters.duplicates ?? 0} duplicates ignored`
          : `${completeness.toFixed(1)}% seq complete · ${counters.duplicates ?? 0} dup`,
      icon: Database,
      color: "#8be2ff",
      percent: completeness,
      ringLabel: "sequence completeness",
    },
    {
      label: "Link Delivery",
      value: delivery === null || delivery === undefined ? "—" : `${(delivery * 100).toFixed(0)}%`,
      caption: state?.link?.emulator ? `emulator ${state.link.emulator.preset}` : "emulator not reporting",
      icon: RadioTower,
      color: "#42efb7",
      percent: delivery === null || delivery === undefined ? null : delivery * 100,
      ringLabel: "delivered by the lunar link",
    },
  ];
  return (
    <Panel
      id="analytics"
      title="Mission Analytics"
      action={<span className="live-metrics">From received telemetry</span>}
      highlighted={highlighted}
    >
      <div className="analytics-cards">
        {metrics.map(({ label, value, caption, icon: Icon, color, percent, ringLabel }, i) => (
          <div
            className={cn("metric-card", `metric-${i}`)}
            key={label}
            style={{ "--metric-color": color } as CSSProperties}
          >
            <span className="metric-icon">
              {i === 0 ? <RockGlyph /> : i === 1 ? <WarningGlyph /> : <Icon strokeWidth={1.9} />}
            </span>
            <div className="metric-copy">
              <p>{label}</p>
              <div>
                <strong>{value}</strong>
              </div>
              <span className="metric-caption">{caption}</span>
            </div>
            <svg
              className="metric-ring"
              viewBox="0 0 64 64"
              aria-label={percent === null ? `${ringLabel}: no data` : `${ringLabel}: ${percent.toFixed(0)} percent`}
            >
              <circle cx="32" cy="32" r="26" />
              {percent !== null && (
                <circle
                  className="ring-value"
                  cx="32"
                  cy="32"
                  r="26"
                  strokeDasharray={`${percent * 1.634} 163.4`}
                />
              )}
            </svg>
          </div>
        ))}
      </div>
    </Panel>
  );
}

// ---- Mission History --------------------------------------------------------------------------------------------------

function MissionHistory({
  list,
  selected,
  following,
  onSelect,
  onFollowLatest,
  now,
  clockOffsetMs,
  highlighted,
}: {
  list: MissionList;
  selected: string | null;
  following: boolean;
  onSelect: (id: string) => void;
  onFollowLatest: () => void;
  now: number;
  clockOffsetMs: number;
  highlighted: string | null;
}) {
  return (
    <Panel
      id="history"
      slot="depth"
      title="Mission History"
      highlighted={highlighted}
      action={
        <button
          className={cn("inference-pill follow-latest", following && "is-active")}
          onClick={onFollowLatest}
          aria-pressed={following}
          title="Automatically switch to the newest active mission"
        >
          {following ? "Following latest" : "Follow latest"}
        </button>
      }
    >
      <div className="history-rows">
        {list.error && (
          <p className="api-error" role="alert">
            <TriangleAlert />
            {list.error}
            <button onClick={list.refresh}>Retry</button>
          </p>
        )}
        {!list.error && list.loading && !list.missions.length && <EmptyState icon={RefreshCw} title="Loading missions…" />}
        {!list.loading && !list.error && !list.missions.length && (
          <EmptyState icon={History} title="No mission stored" />
        )}
        {list.missions.map((m: Mission) => {
          const age = (now + clockOffsetMs - Date.parse(m.last_rx_utc)) / 1000;
          return (
            <button
              key={m.mission_id}
              className={cn("history-row", m.mission_id === selected && "is-selected")}
              onClick={() => onSelect(m.mission_id)}
              aria-pressed={m.mission_id === selected}
            >
              <span className={cn("history-dot", m.status === "active" && "is-active")} />
              <span className="history-id">#{m.mission_id}</span>
              <span className="history-meta">
                {fmtUtc(m.created_utc, true)} · {m.counters?.packets ?? 0} pkts
              </span>
              <span className="history-age">{m.status === "active" ? "active" : `${formatAge(age)} ago`}</span>
            </button>
          );
        })}
        {list.hasMore && (
          <button className="history-more" onClick={list.loadMore}>
            Load more ({list.missions.length}/{list.total})
          </button>
        )}
      </div>
    </Panel>
  );
}

// ---- Telemetry Monitor ------------------------------------------------------------------------------------------------

function TelemetryPanel({
  state,
  fresh,
  ageS,
  highlighted,
}: {
  state: MissionState | null;
  fresh: Freshness;
  ageS: number | null;
  highlighted: string | null;
}) {
  const p = state ? latestPosition(state) : null;
  const rx = state?.link?.receiver;
  const tiles = [
    {
      label: "Position x / y",
      value: p ? `${p.x_m.toFixed(2)} / ${p.y_m.toFixed(2)}` : "—",
      unit: "m (odom)",
      icon: Navigation,
      color: "#40ece1",
    },
    {
      label: "Heading",
      value: p ? `${p.heading_deg.toFixed(1)}°` : "—",
      unit: p ? compassPoint(p.heading_deg) : "",
      icon: Compass,
      color: "#9ec2ff",
    },
    {
      label: "Pose σxy",
      value: p ? `${p.sigma_xy_m.toFixed(2)}${p.sigma_xy_saturated ? "+" : ""}` : "—",
      unit: p?.sigma_xy_saturated ? "m (saturated)" : "m",
      icon: LocateFixed,
      color: "#40e6d3",
    },
    {
      label: "Packets / s",
      value: fmt(state?.link?.packets_per_s, 1),
      unit: "",
      icon: Gauge,
      color: "#5ab6fc",
      compact: true,
    },
    {
      label: "Gaps · dup · late",
      value: rx ? `${rx.missing ?? 0}·${rx.duplicates ?? 0}·${rx.reordered ?? 0}` : "—",
      unit: "",
      icon: Signal,
      color: "#579fff",
      compact: true,
    },
    {
      label: "Data age (UTC)",
      value: formatAge(ageS),
      unit: "",
      icon: Timer,
      color: "#779dff",
      compact: true,
    },
  ];
  const health = state?.health?.state;
  return (
    <Panel
      id="telemetry"
      title="Telemetry"
      action={
        <StatusPill tone={fresh === "LIVE" ? (health && health.toUpperCase() !== "OK" ? "warn" : undefined) : fresh === "STALE" ? "warn" : "bad"}>
          {fresh === "LIVE" ? (health ?? "Receiving") : freshnessLabel[fresh]}
        </StatusPill>
      }
      highlighted={highlighted}
    >
      <div className={cn("telemetry-tiles", (fresh === "STALE" || fresh === "DISCONNECTED") && "is-stale")}>
        {tiles.map(({ label, value, unit, icon: Icon, color, compact }) => (
          <div
            key={label}
            className={cn(
              "telemetry-tile",
              compact && "is-compact",
              label === "Data age (UTC)" && "mission-time",
            )}
          >
            {label === "Data age (UTC)" ? (
              <ClockGlyph color={color} />
            ) : (
              <Icon style={{ color }} strokeWidth={1.8} />
            )}
            <div>
              <p>{label}</p>
              <strong>
                {value}
                {compact && unit ? <span> {unit}</span> : null}
              </strong>
              {!compact && <span className="telemetry-unit">{unit}</span>}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

// ---- Lunar Network Control ---------------------------------------------------------------------------------------------

function NetworkPanel({
  state,
  network,
  now,
  highlighted,
}: {
  state: MissionState | null;
  network: NetworkControl;
  now: number;
  highlighted: string | null;
}) {
  const st = network.status;
  const samples = state?.linkSamples ?? [];
  const data = useMemo(
    () =>
      samples.map((s) => ({
        t: s.rx_ms,
        rate: s.packets_per_s,
        loss: s.loss_pct,
      })),
    [samples],
  );
  const blackoutNowS = network.blackoutSinceMs ? (now - network.blackoutSinceMs) / 1000 : null;
  // Recovery = first rover packet received after the link left BLACKOUT (both on the browser clock).
  const restored = network.restoredAtMs;
  const lastRx = state?.lastRxMs ?? null;
  const recoveryS =
    restored !== null && lastRx !== null && lastRx >= restored ? Math.max(0, (lastRx - restored) / 1000) : null;
  const linkState = st?.link_state ?? (state?.link?.emulator?.link_state ?? null);
  const req = network.request;
  const feedback = !network.available
    ? `Emulator control unavailable: ${network.error ?? "no answer"}`
    : req.phase === "applying"
      ? `Applying ${req.preset}…`
      : req.phase === "confirmed"
        ? `${req.preset} confirmed by emulator status at ${fmtUtc(req.atMs)} UTC`
        : req.phase === "failed"
          ? `${req.preset} not applied: ${req.message}`
          : `Emulator: ${st?.received ?? 0} rx · ${st?.forwarded ?? 0} fwd · ${(st?.dropped_loss ?? 0) + (st?.dropped_blackout ?? 0) + (st?.dropped_queue ?? 0)} dropped`;
  return (
    <Panel
      id="network"
      slot="communication"
      title="Lunar Network"
      highlighted={highlighted}
    >
      <svg
        className="communication-outline"
        viewBox="0 0 608 195"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <path d="M10 .5H273Q283 .5 283 10.5V31Q283 40.5 293 40.5H598Q607.5 40.5 607.5 50.5V184Q607.5 194.5 598 194.5H10Q.5 194.5 .5 184V11Q.5 .5 10 .5Z" />
      </svg>
      <div className="network-controls">
        <div className="log-filters preset-buttons" role="group" aria-label="Lunar link presets">
          {PRESETS.map((p) => (
            <button
              key={p}
              className={cn(st?.preset === p && "is-active", p === "BLACKOUT" && "is-danger")}
              aria-pressed={st?.preset === p}
              disabled={req.phase === "applying"}
              onClick={() => void network.apply(p)}
              title={`Apply the ${p} preset on the lunar link emulator`}
            >
              {req.phase === "applying" && req.preset === p ? "…" : p}
            </button>
          ))}
        </div>
        <p
          className={cn("network-feedback", (req.phase === "failed" || !network.available) && "is-error")}
          role="status"
        >
          {feedback}
        </p>
      </div>
      <div className="communication-stats network-stats">
        <div>
          <span>Link · preset</span>
          <strong className={cn(linkState === "BLACKOUT" ? "is-bad" : linkState === "UP" && "stable")}>
            {linkState ?? "—"} <small>{st?.preset ?? ""}</small>
          </strong>
        </div>
        <div>
          <span>Loss (cfg · window)</span>
          <strong>
            {st ? `${(st.configured_loss * 100).toFixed(0)}%` : "—"}{" "}
            <small>{fmt(samples[samples.length - 1]?.loss_pct, 0, "%")}</small>
          </strong>
        </div>
        <div>
          <span>Latency (cfg · p50)</span>
          <strong>
            {st ? `${st.config.latency_ms.toFixed(0)} ms` : "—"} <small>{fmt(st?.delay_ms?.p50, 0, " ms")}</small>
          </strong>
        </div>
        <div>
          <span>Throughput</span>
          <strong>{st?.throughput_Bps !== null && st?.throughput_Bps !== undefined ? `${st.throughput_Bps.toFixed(0)} B/s` : "—"}</strong>
        </div>
        <div>
          <span>Blackout · recovery</span>
          <strong className={cn(blackoutNowS !== null && "is-bad")}>
            {blackoutNowS !== null
              ? `${blackoutNowS.toFixed(0)} s`
              : network.lastBlackoutS !== null
                ? `${network.lastBlackoutS.toFixed(1)} s`
                : "—"}{" "}
            <small>
              {blackoutNowS !== null
                ? "ongoing"
                : recoveryS !== null
                  ? `rx +${recoveryS.toFixed(1)} s`
                  : restored !== null
                    ? "awaiting rx"
                    : ""}
            </small>
          </strong>
        </div>
      </div>
      <div className="chart-legend">
        <span>
          <i />
          Received packets/s
        </span>
        <span>
          <i />
          Link loss (%)
        </span>
      </div>
      <div className="communication-chart network-chart">
        {data.length < 2 ? (
          <EmptyState icon={Wifi} title="Waiting for link metrics">
            The ground station samples the link every 2 s while a mission is receiving.
          </EmptyState>
        ) : (
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <ComposedChart data={data} margin={{ top: 3, right: 12, bottom: 0, left: 3 }}>
              <defs>
                <linearGradient id="latency-area" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stopColor="#2378bd" stopOpacity={0.15} />
                  <stop offset="1" stopColor="#2378bd" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#17324b" strokeDasharray="2 3" vertical={false} />
              <XAxis
                dataKey="t"
                type="number"
                scale="time"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(value: number) => fmtUtc(value).slice(0, 5)}
                tick={{ fill: "#bdd1e9", fontSize: 11 }}
                stroke="#40638a"
                tickSize={3}
                height={20}
                minTickGap={40}
              />
              <YAxis
                yAxisId="rate"
                tick={{ fill: "#bdd1e9", fontSize: 11 }}
                stroke="#40638a"
                tickSize={3}
                width={34}
                allowDecimals={false}
              />
              <YAxis
                yAxisId="loss"
                orientation="right"
                domain={[0, 100]}
                ticks={[0, 50, 100]}
                tickFormatter={(value: number) => `${value}%`}
                tick={{ fill: "#e6abfa", fontSize: 11 }}
                stroke="#dc99ee"
                tickSize={3}
                width={40}
              />
              <ChartTooltip
                contentStyle={{
                  background: "#061321",
                  border: "1px solid #2b526e",
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelFormatter={(value) => `${fmtUtc(Number(value))} UTC`}
              />
              <Area
                yAxisId="rate"
                dataKey="rate"
                name="Packets/s"
                type="linear"
                stroke="#64b9ff"
                strokeWidth={1.5}
                fill="url(#latency-area)"
                isAnimationActive={false}
                connectNulls={false}
              />
              <Line
                yAxisId="loss"
                dataKey="loss"
                name="Loss (%)"
                type="stepAfter"
                stroke="#d890ee"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </Panel>
  );
}

// ---- Mission Log --------------------------------------------------------------------------------------------------------

const logIcon: Record<LogCategory, LucideIcon> = {
  Navigation: MapIcon,
  Detection: TriangleAlert,
  System: CheckCircle2,
  Communication: Wifi,
};
const toneColor = { info: "#e2edf8", good: "#8bec5e", warn: "#ffc35c", bad: "#ff5267" } as const;

function MissionLog({ log, highlighted }: { log: LogEntry[]; highlighted: string | null }) {
  const [filter, setFilter] = useState<"All" | LogCategory>("All");
  const filtered = log.filter((entry) => filter === "All" || entry.category === filter);
  return (
    <Panel
      id="logs"
      title="Mission Log"
      highlighted={highlighted}
      action={
        <div
          className="log-filters"
          role="group"
          aria-label="Filter mission logs"
        >
          {(["All", "Navigation", "Detection", "System", "Communication"] as const).map(
            (name) => (
              <button
                key={name}
                className={filter === name ? "is-active" : undefined}
                onClick={() => setFilter(name)}
                aria-pressed={filter === name}
              >
                {name}
              </button>
            ),
          )}
        </div>
      }
    >
      <div className="log-rows" aria-live="polite">
        {filtered.length === 0 ? (
          <EmptyState icon={FileText} title="No mission event yet" />
        ) : (
          filtered.map(({ key, rx_utc, t_obs_s, text, detail, category, tone }) => {
            const Icon = category === "Communication" && tone === "bad" ? WifiOff : logIcon[category];
            const color = toneColor[tone];
            return (
              <div
                className="log-row"
                key={key}
                title={`received ${fmtUtc(rx_utc, true)} UTC${t_obs_s !== null ? ` · observed t=${formatObsTime(t_obs_s)} (sim)` : ""}`}
              >
                <time dateTime={rx_utc}>{fmtUtc(rx_utc)}</time>
                <Icon style={{ color }} />
                <p>
                  {text} {detail && <span>{detail}</span>}
                </p>
              </div>
            );
          })
        )}
      </div>
    </Panel>
  );
}

// ---- Mini map & rover ---------------------------------------------------------------------------------------------------

function MiniMap({
  state,
  stale,
  onExpand,
  highlighted,
}: {
  state: MissionState | null;
  stale: boolean;
  onExpand: () => void;
  highlighted: string | null;
}) {
  const obstacles = useMemo(() => Object.values(state?.obstacles ?? {}), [state?.obstacles]);
  return (
    <Panel
      id="mini"
      title="Mission Extent"
      highlighted={highlighted}
      action={
        <button
          className="icon-button"
          onClick={onExpand}
          aria-label="Expand mission extent map"
        >
          <Maximize2 />
        </button>
      }
    >
      <div className="mini-scene live-mini">
        {state && (state.trajectory.length > 0 || obstacles.length > 0) ? (
          <LunarMap
            trajectory={state.trajectory}
            obstacles={obstacles}
            stale={stale}
            compact
            label="Whole received trajectory and obstacles"
          />
        ) : (
          <EmptyState icon={MapIcon} title="No data" />
        )}
      </div>
    </Panel>
  );
}

function RoverCard({ highlighted }: { highlighted: string | null }) {
  return (
    <Panel
      id="rover"
      title="Artemis-X Rover"
      subtitle="6-Wheel Research Platform"
      highlighted={highlighted}
    >
      <Scene
        crop={scenes.rover}
        label="Artemis-X six-wheel lunar rover illustration"
        className="rover-scene"
      />
      <svg className="rover-antennas" viewBox="0 0 28 90" aria-hidden="true">
        <path d="M6 87V9M22 80V5" stroke="#c4d1e1" strokeWidth=".8" />
        <circle cx="6" cy="9" r="1.3" fill="#c8d8ea" />
        <circle cx="22" cy="5" r="1.3" fill="#c8d8ea" />
      </svg>
      <div className="rover-features">
        {[
          { icon: Route, text: "Autonomous Navigation" },
          { icon: MapIcon, text: "Real-time Object Detection" },
          { icon: ScanEye, text: "Stereo Depth Perception" },
          { icon: RadioTower, text: "Lunar Terrain Mapping" },
          { icon: Wifi, text: "Delay-Tolerant Telemetry" },
        ].map(({ icon: Icon, text }) => (
          <div key={text}>
            <Icon />
            {text}
          </div>
        ))}
      </div>
      <p className="rover-motto">DESIGNED FOR A HIGHER TOMORROW</p>
    </Panel>
  );
}

// ---- dialogs ------------------------------------------------------------------------------------------------------------

function DashboardDialog({
  dialog,
  onClose,
  state,
  stale,
}: {
  dialog: DialogName;
  onClose: () => void;
  state: MissionState | null;
  stale: boolean;
}) {
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!dialog) return;
    const previous = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const elements = Array.from(
          document.querySelectorAll<HTMLElement>(
            ".dashboard-dialog button, .dashboard-dialog input",
          ),
        );
        const first = elements[0],
          last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
      previous?.focus();
    };
  }, [dialog, onClose]);
  const title = dialog === "mini" ? "Mission Extent" : "Lunar Surface Map";
  const obstacles = Object.values(state?.obstacles ?? {});
  return (
    <AnimatePresence>
      {dialog && (
        <motion.div
          className="dialog-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.section
            role="dialog"
            aria-modal="true"
            aria-labelledby="dialog-title"
            className="dashboard-dialog"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="dialog-heading">
              <h2 id="dialog-title">{title}</h2>
              <button
                ref={closeButton}
                className="icon-button"
                onClick={onClose}
                aria-label="Close dialog"
              >
                <X />
              </button>
            </div>
            {(
              <div className="expanded-scene live-expanded">
                {state && (state.trajectory.length > 0 || obstacles.length > 0) ? (
                  <LunarMap
                    trajectory={state.trajectory}
                    obstacles={obstacles}
                    interactive
                    stale={stale}
                    label={title}
                  />
                ) : (
                  <EmptyState icon={MapIcon} title="No data received for this mission" />
                )}
              </div>
            )}
          </motion.section>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/**
 * The R6 all-in-one command center. Panels, bindings and styling are unchanged; the session (selection, WebSocket,
 * network control) now comes from the console-wide MissionProvider instead of page-local hooks.
 */
export function UnifiedDashboard() {
  const m = useMission();
  const { highlighted, setSpy: onSpy } = useConsoleNav();
  const reducedMotion = useReducedMotion();
  const [dialog, setDialog] = useState<DialogName>(null);
  const closeDialog = useCallback(() => setDialog(null), []);
  const { list, stream, network, now, state, fresh, ageS, stale, hasMission } = m;

  // Scroll spy: the sidebar's active item follows the panel in view.
  useEffect(() => {
    const targets = new Set(UNIFIED_TARGETS);
    const visible = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0);
        if (window.scrollY < 40) {
          onSpy(null);
          return;
        }
        let best: string | null = null;
        let ratio = 0.35;
        for (const [id, r] of visible)
          if (r > ratio && targets.has(id)) {
            best = id;
            ratio = r;
          }
        if (best) onSpy(best);
      },
      { threshold: [0, 0.35, 0.6, 0.9], rootMargin: "-90px 0px 0px 0px" },
    );
    for (const id of targets) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [onSpy]);

  return (
    <>
      <div className="dashboard-layout">
        <SurfaceMap
          stream={stream}
          fresh={fresh}
          ageS={ageS}
          hasMission={hasMission}
          unreachable={m.unreachable}
          onExpand={() => setDialog("map")}
          highlighted={highlighted}
          animate={m.motionEnabled && !reducedMotion}
        />
        <MissionStatusPanel
          stream={stream}
          fresh={fresh}
          ageS={ageS}
          now={now}
          missionsError={list.error}
          hasMission={hasMission}
          highlighted={highlighted}
        />
        <DetectionFeed state={state} highlighted={highlighted} />
        <AnalyticsPanel state={state} highlighted={highlighted} />
        <MissionHistory
          list={list}
          selected={m.missionId}
          following={m.followLatest}
          onSelect={m.selectMission}
          onFollowLatest={m.followNewest}
          now={now}
          clockOffsetMs={stream.clockOffsetMs}
          highlighted={highlighted}
        />
        <TelemetryPanel state={state} fresh={fresh} ageS={ageS} highlighted={highlighted} />
        <NetworkPanel state={state} network={network} now={now} highlighted={highlighted} />
        <MissionLog log={state?.log ?? []} highlighted={highlighted} />
        <MiniMap state={state} stale={stale} onExpand={() => setDialog("mini")} highlighted={highlighted} />
        <RoverCard highlighted={highlighted} />
      </div>
      <DashboardDialog dialog={dialog} onClose={closeDialog} state={state} stale={stale} />
    </>
  );
}
