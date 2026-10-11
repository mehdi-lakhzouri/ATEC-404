"use client";

// Shared Mission Control primitives: artwork crops, brand glyphs, formatting, panel / pill / empty-state components.
// Used by the console shell, the unified command center and the clean workspace pages.
import { type CSSProperties, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ConnectionState, Freshness } from "@/lib/ground/freshness";
import { freshnessLabel } from "@/lib/ground/mission-context";

export type Crop = readonly [number, number, number, number];

// Only decorative artwork (header moon, rover illustration) is sampled from the reference image; the sidebar terrain is
// /sidebar-terrain.png, the reference's terrain strip (y 611-941) with the baked-in motto removed so it stays live text.
// Every mission value on this page comes from the ground station API: nothing is simulated in the browser.
export const scenes = {
  rover: [1243, 797, 267, 111],
  moon: [1458, 0, 214, 85],
} satisfies Record<string, Crop>;

export function Scene({
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

export function ArtemisLogo() {
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

export function RockGlyph({ className }: { className?: string }) {
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

/** Reference-matched rover navigation glyph, kept vector-sharp at every display scale. */
export function RoverNavigationGlyph() {
  return <svg className="navigation-icon" viewBox="0 0 28 28" fill="none" stroke="currentColor" strokeWidth="1.35" aria-hidden="true">
    <path d="M8 15h12l2 3M8 15l-3 3M9 15v-5h8v5M11 10V6h4M13 6V3m-3 3h6M17 12h4v3M10 17l3 3h5l3-3" />
    <circle cx="5" cy="21" r="2.5" /><circle cx="15" cy="22" r="2.5" /><circle cx="24" cy="20" r="2.5" />
    <path d="M5 18.5V17m10 2.5V17m9 .5v-2M4 12h2M20 6h3m-1.5-1.5v3" />
  </svg>;
}

export function WarningGlyph() {
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

export function ClockGlyph({ color }: { color: string }) {
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

export function HomeGlyph() {
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

export const utcTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});
export const utcDate = new Intl.DateTimeFormat("en-US", {
  timeZone: "UTC",
  month: "short",
  day: "2-digit",
  year: "numeric",
});

export function fmtUtc(iso: string | number | null | undefined, withDate = false) {
  if (iso === null || iso === undefined) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return withDate ? `${utcDate.format(d)} ${utcTime.format(d)}` : utcTime.format(d);
}

export const fmt = (v: number | null | undefined, digits = 2, unit = "") =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}${unit}`;

export function compassPoint(deg: number) {
  const points = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]; // odom yaw: 0 = +x, counter-clockwise
  return points[Math.round((((deg % 360) + 360) % 360) / 45) % 8]!;
}


export function FreshnessPill({
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

export function Panel({
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

export function StatusPill({ children, tone }: { children: ReactNode; tone?: "warn" | "bad" | "idle" }) {
  return (
    <span className={cn("status-pill", tone && `is-${tone}`)}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

export function EmptyState({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="empty-state" role="status">
      <Icon />
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  );
}

export const typeName: Record<string, string> = {
  rock: "Rock",
  obstacle: "Obstacle",
  negative_hazard: "Negative hazard",
};
export const typeShort: Record<string, string> = { rock: "Rock", obstacle: "Obstacle", negative_hazard: "Hazard" };

export function connectionText(c: ConnectionState, retryAtMs: number | null, now: number) {
  if (c === "open") return "WebSocket open";
  if (c === "connecting") return "Connecting…";
  if (c === "reconnecting")
    return retryAtMs ? `Reconnecting in ${Math.max(0, Math.ceil((retryAtMs - now) / 1000))} s` : "Reconnecting…";
  return "Closed";
}
