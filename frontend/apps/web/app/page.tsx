"use client";

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
  Crosshair,
  Database,
  FileText,
  Flag,
  Gauge,
  History,
  Home as HomeIcon,
  LocateFixed,
  Map as MapIcon,
  Maximize2,
  Menu,
  Navigation,
  RadioTower,
  RefreshCw,
  Route,
  ScanEye,
  Settings,
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
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LunarMap } from "@/components/mission/lunar-map";
import { cn } from "@/lib/utils";
import { GROUND_API, GROUND_WS } from "@/lib/ground/config";
import {
  ageSeconds,
  formatAge,
  formatObsTime,
  freshness,
  type ConnectionState,
  type Freshness,
} from "@/lib/ground/freshness";
import {
  latestPosition,
  type LogCategory,
  type LogEntry,
  type MissionState,
  type TrackedObstacle,
} from "@/lib/ground/mission-state";
import { useMissionStream, useNow, type MissionStream } from "@/lib/ground/use-mission";
import {
  useMissionList,
  useNetworkControl,
  type MissionList,
  type NetworkControl,
} from "@/lib/ground/use-network";
import { PRESETS, type Mission, type Position } from "@/lib/ground/types";

type Crop = readonly [number, number, number, number];
type ViewName =
  | "Overview"
  | "Live Map"
  | "Detection"
  | "Telemetry"
  | "Network"
  | "Missions"
  | "Logs"
  | "Settings";
type DialogName = "map" | "mini" | "settings" | null;

// Only decorative artwork (sidebar terrain, header moon, rover illustration) is sampled from the reference image.
// Every mission value on this page comes from the ground station API: nothing is simulated in the browser.
const scenes = {
  rover: [1243, 797, 267, 111],
  landscape: [0, 651, 205, 205],
  moon: [1458, 0, 214, 85],
} satisfies Record<string, Crop>;

function Scene({
  crop,
  label,
  className,
  children,
}: {
  crop: Crop;
  label: string;
  className?: string;
  children?: ReactNode;
}) {
  const [x, y, width, height] = crop;
  const style: CSSProperties = {
    backgroundImage: "url('/artemis-reference.png')",
    backgroundSize: `${(1672 / width) * 100}% ${(941 / height) * 100}%`,
    backgroundPosition: `${(x / (1672 - width)) * 100}% ${(y / (941 - height)) * 100}%`,
  };
  return (
    <div
      className={cn("reference-scene", className)}
      style={style}
      role="img"
      aria-label={label}
    >
      {children}
    </div>
  );
}

function ArtemisLogo() {
  return (
    <svg
      className="artemis-symbol"
      viewBox="0 0 76 66"
      fill="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient
          id="logo-metal"
          x1="38"
          y1="2"
          x2="38"
          y2="44"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#fff" />
          <stop offset="1" stopColor="#95bee8" />
        </linearGradient>
        <linearGradient
          id="logo-orbit"
          x1="2"
          y1="39"
          x2="73"
          y2="64"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#9cedff" />
          <stop offset=".55" stopColor="#65b8ff" />
          <stop offset="1" stopColor="#316cf5" />
        </linearGradient>
      </defs>
      <path
        d="M14 43C3 14 29 6 40 10M45 10C70 12 73 31 64 42"
        stroke="#1745a0"
        strokeWidth="1.2"
      />
      <path
        d="M38 1 61 40 49 34 38 17 27 35 15 41 38 1Z"
        fill="url(#logo-metal)"
      />
      <path d="M38 20 43 34 38 31 33 35 38 20Z" fill="#091321" />
      <path
        d="M2 61C20 35 54 34 74 61"
        stroke="url(#logo-orbit)"
        strokeWidth="3.5"
      />
      <path d="M11 62C25 45 51 44 66 62" stroke="#68b9ff" strokeWidth="2.6" />
      <path d="M21 62C30 53 46 52 55 62" stroke="#225df2" strokeWidth="1.7" />
    </svg>
  );
}

function RockGlyph({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 42 42"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="m7 18 10-4 9 6 1 12-7 7-13-2-6-9 6-10Zm20-17 10 5 5 10-5 9-7 2-1-10-9-5 7-11Z" />
      <circle cx="11" cy="29" r="2" fill="#19314e" />
      <circle cx="23" cy="29" r="1.7" fill="#19314e" />
      <circle cx="35" cy="15" r="1.8" fill="#19314e" />
    </svg>
  );
}

function WarningGlyph() {
  return (
    <svg viewBox="0 0 40 40" fill="currentColor" aria-hidden="true">
      <path d="M16.7 5.4a3.8 3.8 0 0 1 6.6 0l15.1 26.3a3.8 3.8 0 0 1-3.3 5.7H4.9a3.8 3.8 0 0 1-3.3-5.7L16.7 5.4Z" />
      <path
        d="M20 14v10"
        stroke="#40142a"
        strokeWidth="3.6"
        strokeLinecap="round"
      />
      <circle cx="20" cy="30" r="2" fill="#40142a" />
    </svg>
  );
}

function ClockGlyph({ color }: { color: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="10.5" fill={color} />
      <path
        d="M12 5.8v6.8l4.7 1.5"
        stroke="#142b64"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function HomeGlyph() {
  return (
    <svg
      className="navigation-icon"
      viewBox="0 0 32 32"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M16 2 30 14H26V28H20V19H12V28H6V14H2L16 2Z" />
    </svg>
  );
}

// ---- formatting ------------------------------------------------------------------------------------------------------

const utcTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});
const utcDate = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  month: "short",
  day: "2-digit",
  year: "numeric",
});

function fmtUtc(iso: string | number | null | undefined, withDate = false) {
  if (iso === null || iso === undefined) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return withDate ? `${utcDate.format(d)} ${utcTime.format(d)}` : utcTime.format(d);
}

const fmt = (v: number | null | undefined, digits = 2, unit = "") =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}${unit}`;

function compassPoint(deg: number) {
  const points = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]; // odom yaw: 0 = +x, counter-clockwise
  return points[Math.round((((deg % 360) + 360) % 360) / 45) % 8]!;
}

const freshnessLabel: Record<Freshness, string> = {
  LIVE: "Live",
  STALE: "Stale",
  DISCONNECTED: "Disconnected",
  NO_DATA: "No data",
};

function FreshnessPill({
  value,
  className,
  title,
}: {
  value: Freshness;
  className?: string;
  title?: string;
}) {
  return (
    <span className={cn("freshness-pill", `is-${value.toLowerCase()}`, className)} title={title}>
      <span className="status-dot" />
      {freshnessLabel[value]}
    </span>
  );
}

// ---- shell -----------------------------------------------------------------------------------------------------------

const navigation: { label: ViewName; icon: LucideIcon; target?: string }[] = [
  { label: "Overview", icon: HomeIcon },
  { label: "Live Map", icon: MapIcon, target: "surface" },
  { label: "Detection", icon: Crosshair, target: "detection" },
  { label: "Telemetry", icon: Signal, target: "telemetry" },
  { label: "Network", icon: RadioTower, target: "network" },
  { label: "Missions", icon: History, target: "history" },
  { label: "Logs", icon: FileText, target: "logs" },
  { label: "Settings", icon: Settings },
];

function Sidebar({
  open,
  active,
  onSelect,
  onClose,
}: {
  open: boolean;
  active: ViewName;
  onSelect: (view: ViewName, target?: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.button
            className="navigation-backdrop"
            aria-label="Close navigation"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
        )}
      </AnimatePresence>
      <aside
        id="mission-navigation"
        className={cn("mission-sidebar", open && "is-open")}
        aria-label="Mission navigation"
      >
        <button
          className="sidebar-close icon-button"
          aria-label="Close navigation"
          onClick={onClose}
        >
          <X />
        </button>
        <nav className="sidebar-navigation">
          {navigation.map(({ label, icon: Icon, target }) => (
            <button
              key={label}
              className={cn("navigation-item", active === label && "is-active")}
              aria-current={active === label ? "page" : undefined}
              onClick={() => onSelect(label, target)}
            >
              {label === "Overview" ? (
                <HomeGlyph />
              ) : (
                <Icon className="navigation-icon" strokeWidth={1.8} />
              )}
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <Scene
          crop={scenes.landscape}
          label="Lunar terrain"
          className="sidebar-landscape"
        />
        <p className="sidebar-motto">
          EXPLORE
          <br />
          AUTONOMOUSLY
          <br />
          BUILD FURTHER
        </p>
      </aside>
    </>
  );
}

function MissionHeader({
  fresh,
  freshTitle,
  mission,
  latest,
  onMenu,
  menuOpen,
}: {
  fresh: Freshness;
  freshTitle: string;
  mission: MissionState["mission"];
  latest: Position | null;
  onMenu: () => void;
  menuOpen: boolean;
}) {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    const update = () => setNow(new Date());
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <header className="mission-header">
      <a
        className="brand-lockup"
        href="#overview"
        aria-label="Artemis-X Mission Control home"
      >
        <ArtemisLogo />
        <span className="brand-type">
          <strong>ARTEMIS-X</strong>
          <span>MISSION CONTROL</span>
        </span>
      </a>
      <button
        className="mobile-menu icon-button"
        aria-label="Open navigation"
        aria-controls="mission-navigation"
        aria-expanded={menuOpen}
        onClick={onMenu}
      >
        <Menu />
      </button>
      <div className="header-heading">
        <div className="header-title-row">
          <h1>Artemis-X Mission Control</h1>
          <FreshnessPill value={fresh} className="simulation-status" title={freshTitle} />
        </div>
        <p className="mission-disciplines">
          LUNAR ROBOTICS <i>·</i> AUTONOMY <i>·</i> MAPPING <i>·</i> DETECTION{" "}
          <i>·</i> EXPLORATION
        </p>
      </div>
      <div className="mission-context">
        <div className="context-utc">
          <span>UTC</span>
          <p>
            {now ? utcDate.format(now) : "—"}
            <br />
            <time dateTime={now?.toISOString()}>{now ? utcTime.format(now) : "—"}</time>
          </p>
        </div>
        <div className="context-day">
          <span>Mission</span>
          <p title={mission?.mission_id}>
            {mission ? `#${mission.mission_id}` : "—"}
            <br />
            <small>{mission?.status ?? "none"}</small>
          </p>
        </div>
        <div className="context-location">
          <span>Rover clock (sim, last pose)</span>
          <p>
            {formatObsTime(latest?.t_obs_s)}
            <br />
            <small>Webots lunar field · odom frame</small>
          </p>
        </div>
      </div>
      <Scene crop={scenes.moon} label="The Moon" className="header-moon" />
    </header>
  );
}

function Panel({
  id,
  slot,
  title,
  subtitle,
  action,
  className,
  children,
  highlighted,
}: {
  id: string;
  slot?: string;
  title: ReactNode;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
  highlighted?: string | null;
}) {
  return (
    <Card
      id={id}
      className={cn(
        "mission-panel",
        `${slot ?? id}-panel`,
        className,
        highlighted === id && "panel-highlighted",
      )}
    >
      <div className="panel-heading">
        <div className="panel-title-block">
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </Card>
  );
}

function StatusPill({ children, tone }: { children: ReactNode; tone?: "warn" | "bad" | "idle" }) {
  return (
    <span className={cn("status-pill", tone && `is-${tone}`)}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

function EmptyState({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="empty-state" role="status">
      <Icon />
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  );
}

// ---- Live Lunar Map ----------------------------------------------------------------------------------------------------

const typeName: Record<string, string> = {
  rock: "Rock",
  obstacle: "Obstacle",
  negative_hazard: "Negative hazard",
};
const typeShort: Record<string, string> = { rock: "Rock", obstacle: "Obstacle", negative_hazard: "Hazard" };

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
  const [mode, setMode] = useState<"follow" | "fit">("fit");
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
        <div className="log-filters map-mode" role="group" aria-label="Map view">
          {(["fit", "follow"] as const).map((m) => (
            <button
              key={m}
              className={mode === m ? "is-active" : undefined}
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
            >
              {m === "fit" ? "Fit mission" : "Follow rover"}
            </button>
          ))}
        </div>
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
            mode={mode}
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
      <div className="map-legend">
        <div>
          <span className="legend-rover" />
          Rover (last pose)
        </div>
        <div>
          <span className="legend-trail" />
          Received path
        </div>
        <div>
          <span className="legend-dot rock" />
          Rock
        </div>
        <div>
          <span className="legend-dot obstacle" />
          Obstacle
        </div>
        <div>
          <span className="legend-dot hazard" />
          Negative hazard
        </div>
        <div>
          <span className="legend-ring" />
          Uncertainty
        </div>
        <div>
          <span className="legend-area" />
          Traversed (free)
        </div>
        <div>
          <span className="legend-unknown" />
          Unknown
        </div>
      </div>
    </Panel>
  );
}

// ---- Mission Overview (status) ------------------------------------------------------------------------------------------

function connectionText(c: ConnectionState, retryAtMs: number | null, now: number) {
  if (c === "open") return "WebSocket open";
  if (c === "connecting") return "Connecting…";
  if (c === "reconnecting")
    return retryAtMs ? `Reconnecting in ${Math.max(0, Math.ceil((retryAtMs - now) / 1000))} s` : "Reconnecting…";
  return "Closed";
}

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
            mode="fit"
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
  motionEnabled,
  onMotionChange,
  followLatest,
  onFollowChange,
}: {
  dialog: DialogName;
  onClose: () => void;
  state: MissionState | null;
  stale: boolean;
  motionEnabled: boolean;
  onMotionChange: (enabled: boolean) => void;
  followLatest: boolean;
  onFollowChange: (enabled: boolean) => void;
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
  const title =
    dialog === "settings"
      ? "Mission Control Settings"
      : dialog === "mini"
        ? "Mission Extent"
        : "Lunar Surface Map";
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
            className={cn(
              "dashboard-dialog",
              dialog === "settings" && "settings-dialog",
            )}
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
            {dialog === "settings" ? (
              <div className="settings-content">
                <label>
                  <div>
                    <strong>Follow the newest mission</strong>
                    <span>
                      Switch automatically when a new rover session starts
                      sending telemetry.
                    </span>
                  </div>
                  <input
                    type="checkbox"
                    checked={followLatest}
                    onChange={(event) => onFollowChange(event.target.checked)}
                  />
                </label>
                <label>
                  <div>
                    <strong>Motion effects</strong>
                    <span>
                      Show the rover locator pulse and animated interface
                      transitions.
                    </span>
                  </div>
                  <input
                    type="checkbox"
                    checked={motionEnabled}
                    onChange={(event) => onMotionChange(event.target.checked)}
                  />
                </label>
                <div className="settings-endpoints">
                  <strong>Ground station</strong>
                  <span>REST {GROUND_API}/api/v1</span>
                  <span>WebSocket {GROUND_WS}/ws/v1/missions/&#123;id&#125;</span>
                </div>
                <Button onClick={onClose}>Done</Button>
              </div>
            ) : (
              <div className="expanded-scene live-expanded">
                {state && (state.trajectory.length > 0 || obstacles.length > 0) ? (
                  <LunarMap
                    trajectory={state.trajectory}
                    obstacles={obstacles}
                    mode="fit"
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

// ---- page -------------------------------------------------------------------------------------------------------------

function readMissionParam(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("mission");
}

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [active, setActive] = useState<ViewName>("Overview");
  const [motionEnabled, setMotionEnabled] = useState(true);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [dialog, setDialog] = useState<DialogName>(null);
  const [missionId, setMissionId] = useState<string | null>(null);
  const [followLatest, setFollowLatest] = useState(true);
  const reducedMotion = useReducedMotion();
  const closeDialog = useCallback(() => setDialog(null), []);

  const list = useMissionList();
  const stream = useMissionStream(missionId);
  const network = useNetworkControl();
  const now = useNow(1000);

  // Mission selection: ?mission=<id> pins a mission; otherwise follow the newest active (else most recent) mission.
  useEffect(() => {
    const pinned = readMissionParam();
    if (pinned) {
      setMissionId(pinned);
      setFollowLatest(false);
    }
  }, []);
  useEffect(() => {
    if (!followLatest || !list.missions.length) return;
    const newest = list.missions.find((m) => m.status === "active") ?? list.missions[0]!;
    if (newest.mission_id !== missionId) setMissionId(newest.mission_id);
  }, [followLatest, list.missions, missionId]);

  const selectMission = useCallback((id: string) => {
    setFollowLatest(false);
    setMissionId(id);
    const url = new URL(window.location.href);
    url.searchParams.set("mission", id);
    window.history.replaceState(null, "", url);
  }, []);
  const followNewest = useCallback(() => {
    setFollowLatest(true);
    const url = new URL(window.location.href);
    url.searchParams.delete("mission");
    window.history.replaceState(null, "", url);
  }, []);

  const state = stream.state && stream.state.missionId === missionId ? stream.state : null;
  const groundNow = now + stream.clockOffsetMs;
  const fresh: Freshness = missionId
    ? freshness(stream.connection, state?.lastRxMs ?? null, groundNow)
    : list.error
      ? "DISCONNECTED"
      : "NO_DATA";
  const ageS = ageSeconds(state?.lastRxMs ?? null, groundNow);
  const latest = state ? latestPosition(state) : null;
  const stale = fresh === "STALE" || fresh === "DISCONNECTED";
  const hasMission = missionId !== null;
  const freshTitle = `${freshnessLabel[fresh]} · last ground reception ${formatAge(ageS)} ago (UTC) · ${connectionText(stream.connection, stream.retryAtMs, now)}`;

  const selectView = (view: ViewName, target?: string) => {
    setActive(view);
    setSidebarOpen(false);
    if (view === "Settings") {
      setDialog("settings");
      return;
    }
    setHighlighted(target ?? null);
    if (target)
      document.getElementById(target)?.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: "nearest",
      });
    else
      window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
  };
  return (
    <div
      id="overview"
      className={cn(
        "mission-app",
        (!motionEnabled || reducedMotion) && "reduce-motion",
      )}
      data-freshness={fresh}
    >
      <MissionHeader
        fresh={fresh}
        freshTitle={freshTitle}
        mission={state?.mission ?? null}
        latest={latest}
        onMenu={() => setSidebarOpen(true)}
        menuOpen={sidebarOpen}
      />
      <Sidebar
        open={sidebarOpen}
        active={active}
        onSelect={selectView}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="mission-main" aria-label="Artemis-X mission dashboard">
        <div className="dashboard-layout">
          <SurfaceMap
            stream={stream}
            fresh={fresh}
            ageS={ageS}
            hasMission={hasMission}
            unreachable={!hasMission && list.error !== null}
            onExpand={() => setDialog("map")}
            highlighted={highlighted}
            animate={motionEnabled && !reducedMotion}
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
            selected={missionId}
            following={followLatest}
            onSelect={selectMission}
            onFollowLatest={followNewest}
            now={now}
            clockOffsetMs={stream.clockOffsetMs}
            highlighted={highlighted}
          />
          <TelemetryPanel state={state} fresh={fresh} ageS={ageS} highlighted={highlighted} />
          <NetworkPanel state={state} network={network} now={now} highlighted={highlighted} />
          <MissionLog log={state?.log ?? []} highlighted={highlighted} />
          <MiniMap
            state={state}
            stale={stale}
            onExpand={() => setDialog("mini")}
            highlighted={highlighted}
          />
          <RoverCard highlighted={highlighted} />
        </div>
      </main>
      <DashboardDialog
        dialog={dialog}
        onClose={closeDialog}
        state={state}
        stale={stale}
        motionEnabled={motionEnabled}
        onMotionChange={setMotionEnabled}
        followLatest={followLatest}
        onFollowChange={(v) => (v ? followNewest() : setFollowLatest(false))}
      />
    </div>
  );
}
