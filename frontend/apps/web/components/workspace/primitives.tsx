"use client";

// Clean workspace building blocks. Every value shown comes from the mission session; these only lay it out.
import type { ReactNode } from "react";
import Link from "next/link";
import { RefreshCw, TriangleAlert, WifiOff, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatAge, formatObsTime, type Freshness } from "@/lib/ground/freshness";
import { freshnessLabel, useMission } from "@/lib/ground/mission-context";
import { fmtUtc } from "@/components/mission/shared";

export type Tone = "ok" | "warn" | "bad" | "idle" | "signal";

export const freshTone: Record<Freshness, Tone> = { LIVE: "ok", STALE: "warn", DISCONNECTED: "bad", NO_DATA: "idle" };

export function healthTone(state: string | null | undefined): Tone {
  if (!state) return "idle";
  const s = state.toUpperCase();
  return s === "OK" || s === "NOMINAL" ? "ok" : s === "DEGRADED" || s === "WARN" ? "warn" : "bad";
}

export function Page({ id, children }: { id: string; children: ReactNode }) {
  return (
    <div className="cw" data-page={id}>
      {children}
    </div>
  );
}

/** Eyebrow = the mission this page is about (a fact, not a decoration). */
export function PageHeader({ title, description, aside }: { title: string; description?: string; aside?: ReactNode }) {
  const m = useMission();
  const mission = m.state?.mission;
  return (
    <header className="cw-head">
      <div>
        <p className="cw-eyebrow">
          {mission ? (
            <>
              Mission <span className="cw-mono">#{mission.mission_id}</span>
              <i>·</i>
              {m.followLatest ? "following the newest session" : "pinned"}
            </>
          ) : m.unreachable ? (
            "Ground station unreachable"
          ) : (
            "No mission selected"
          )}
        </p>
        <h1>{title}</h1>
        {description && <p className="cw-lede">{description}</p>}
      </div>
      <div className="cw-head-aside">
        {aside}
        <FreshBadge />
      </div>
    </header>
  );
}

export function FreshBadge({ value }: { value?: Freshness }) {
  const m = useMission();
  const f = value ?? m.fresh;
  return (
    <span className={cn("cw-badge", `tone-${freshTone[f]}`)} title={m.freshLabel}>
      <span className="cw-dot" />
      {freshnessLabel[f]}
      {f !== "NO_DATA" && m.ageS !== null && <span className="cw-mono cw-badge-age">{formatAge(m.ageS)}</span>}
    </span>
  );
}

export function Section({
  title,
  aside,
  className,
  children,
  flush,
}: {
  title?: string;
  aside?: ReactNode;
  className?: string;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className={cn("cw-surface", flush && "is-flush", className)}>
      {(title || aside) && (
        <div className="cw-section-head">
          {title && <h2>{title}</h2>}
          {aside && <div className="cw-section-aside">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Kpi({
  label,
  value,
  unit,
  context,
  tone = "signal",
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  context?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className={cn("cw-kpi", `tone-${tone}`)}>
      <p className="cw-kpi-label">{label}</p>
      <p className="cw-kpi-value cw-mono">
        {value}
        {unit && <small>{unit}</small>}
      </p>
      {context && <p className="cw-kpi-context">{context}</p>}
    </div>
  );
}

export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="cw-facts">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Meter({ value, tone = "signal", label }: { value: number | null; tone?: Tone; label: string }) {
  const pct = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className={cn("cw-meter", `tone-${tone}`)} role="meter" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <span style={{ width: `${pct}%` }} />
    </span>
  );
}

export function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="cw-empty" role="status">
      <Icon aria-hidden="true" />
      <strong>{title}</strong>
      {children && <span>{children}</span>}
    </div>
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <p className="cw-error" role="alert">
      <TriangleAlert aria-hidden="true" />
      <span>{message}</span>
      {onRetry && (
        <button onClick={onRetry}>
          <RefreshCw aria-hidden="true" />
          Retry
        </button>
      )}
    </p>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string; count?: number }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="cw-segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} className={cn(value === o.value && "is-active")} aria-pressed={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
          {o.count !== undefined && <span className="cw-mono">{o.count}</span>}
        </button>
      ))}
    </div>
  );
}

export function TextLink({ href, children }: { href: string; children: ReactNode }) {
  const m = useMission();
  return (
    <Link className="cw-link" href={m.href(href)}>
      {children}
      <span aria-hidden="true">→</span>
    </Link>
  );
}

/**
 * Signature: the two clocks of the mission, never mixed. Left, the rover's observation clock (simulation time of the
 * last received pose); right, the ground's reception clock (UTC). The line between them carries the reception age:
 * a travelling pulse while LIVE, frozen and hatched when STALE, broken when the ground stream is DISCONNECTED.
 */
export function SignalRail({ compact }: { compact?: boolean }) {
  const m = useMission();
  const f = m.fresh;
  const link = m.state?.link?.emulator;
  const mid =
    f === "LIVE"
      ? `${formatAge(m.ageS)} since the last packet`
      : f === "STALE"
        ? `no telemetry for ${formatAge(m.ageS)} · last picture kept`
        : f === "DISCONNECTED"
          ? "ground stream disconnected"
          : "waiting for the first packet";
  return (
    <div className={cn("cw-rail", `tone-${freshTone[f]}`, compact && "is-compact")} aria-label={`Signal: ${mid}`}>
      <div className="cw-rail-clock">
        <p className="cw-eyebrow">Rover clock · sim</p>
        <p className="cw-rail-time cw-mono">{formatObsTime(m.latest?.t_obs_s)}</p>
        {!compact && <p className="cw-rail-sub">last observation received</p>}
      </div>
      <div className="cw-rail-line" aria-hidden="true">
        <span className="cw-rail-node" />
        <span className="cw-rail-track">
          {f === "DISCONNECTED" ? <WifiOff className="cw-rail-break" /> : <span className="cw-rail-pulse" />}
        </span>
        <span className="cw-rail-node" />
      </div>
      <div className="cw-rail-clock is-right">
        <p className="cw-eyebrow">Ground reception · UTC</p>
        <p className="cw-rail-time cw-mono">{m.state?.lastRxMs ? fmtUtc(m.state.lastRxMs) : "—"}</p>
        {!compact && <p className="cw-rail-sub">last packet on the ground</p>}
      </div>
      <p className="cw-rail-mid">
        <span>{mid}</span>
        {link && (
          <span className="cw-mono">
            link {link.link_state} · {link.preset}
          </span>
        )}
      </p>
    </div>
  );
}

/** Shared guard: what a page shows before a mission is available. Returns null when the page can render. */
export function useMissionGate(): ReactNode {
  const m = useMission();
  if (m.unreachable)
    return (
      <Section>
        <Empty icon={WifiOff} title="Ground station unreachable">
          Nothing is drawn from local data. Start the ground backend, then retry.
        </Empty>
        <ErrorNote message={m.list.error ?? "unreachable"} onRetry={m.list.refresh} />
      </Section>
    );
  if (!m.hasMission)
    return (
      <Section>
        <Empty icon={RefreshCw} title={m.list.loading ? "Loading missions…" : "No mission recorded yet"}>
          {m.list.loading ? null : "A mission appears when the first rover packet reaches the ground station."}
        </Empty>
      </Section>
    );
  if (m.stream.error && !m.state?.mission && !m.stream.loading)
    return (
      <Section>
        <Empty icon={TriangleAlert} title="Mission unavailable" />
        <ErrorNote message={m.stream.error} onRetry={m.stream.reload} />
      </Section>
    );
  return null;
}

export const fmtWhen = (iso: string | number | null | undefined, groundNow: number) => {
  if (iso === null || iso === undefined) return "—";
  const t = typeof iso === "number" ? iso : Date.parse(iso);
  return `${fmtUtc(t)} · ${formatAge((groundNow - t) / 1000)} ago`;
};
