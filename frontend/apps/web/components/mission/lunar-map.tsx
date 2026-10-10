"use client";

// Live lunar map in the rover's odom frame (x right, y up), drawn ONLY from received telemetry:
//  - background = unknown (hatched): the occupancy grid is not downlinked, so nothing is shown as free by default;
//  - traversed corridor = known free (the rover itself occupied it);
//  - obstacles = reconciled ground picture (confidence -> opacity, uncertainty ring, tracking status style);
//  - trajectory = received poses only; a gap in observation time is NOT bridged (no invented positions).
import { useEffect, useMemo, useRef, useState } from "react";
import type { TrackedObstacle } from "@/lib/ground/mission-state";
import { trajectorySegments } from "@/lib/ground/trajectory";
import type { Position } from "@/lib/ground/types";
import { cn } from "@/lib/utils";

const ROVER_WIDTH_M = 0.6;

type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

function useSize<T extends Element>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 400, height: 260 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

function niceStep(span: number) {
  const target = span / 6;
  for (const s of [0.5, 1, 2, 5, 10, 20, 50, 100]) if (s >= target) return s;
  return 200;
}

const typeColor: Record<string, string> = {
  rock: "#8ff363",
  obstacle: "#ffc35c",
  negative_hazard: "#ff5a70",
};

export function LunarMap({
  trajectory,
  obstacles,
  mode,
  stale,
  compact,
  selectedId,
  onSelect,
  label,
}: {
  trajectory: Position[];
  obstacles: TrackedObstacle[];
  mode: "follow" | "fit";
  stale: boolean;
  compact?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  label: string;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const latest = trajectory[trajectory.length - 1] ?? null;
  const segments = useMemo(() => trajectorySegments(trajectory), [trajectory]);

  const view = useMemo(() => {
    let b: Bounds;
    if (mode === "follow" && latest) {
      const half = 6;
      b = { minX: latest.x_m - half, maxX: latest.x_m + half, minY: latest.y_m - half, maxY: latest.y_m + half };
    } else {
      b = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
      const grow = (x: number, y: number, r = 0) => {
        b.minX = Math.min(b.minX, x - r);
        b.maxX = Math.max(b.maxX, x + r);
        b.minY = Math.min(b.minY, y - r);
        b.maxY = Math.max(b.maxY, y + r);
      };
      for (const p of trajectory) grow(p.x_m, p.y_m);
      for (const o of obstacles) grow(o.x_m, o.y_m, o.radius_m + o.uncertainty_m);
      if (!Number.isFinite(b.minX)) b = { minX: -5, maxX: 5, minY: -5, maxY: 5 };
      const pad = 1.2;
      b = { minX: b.minX - pad, maxX: b.maxX + pad, minY: b.minY - pad, maxY: b.maxY + pad };
      const minSpan = 6;
      const cx = (b.minX + b.maxX) / 2;
      const cy = (b.minY + b.maxY) / 2;
      const w = Math.max(minSpan, b.maxX - b.minX);
      const h = Math.max(minSpan, b.maxY - b.minY);
      b = { minX: cx - w / 2, maxX: cx + w / 2, minY: cy - h / 2, maxY: cy + h / 2 };
    }
    // match the container aspect ratio so 1 m is the same length on both axes
    const aspect = size.width / size.height;
    let w = b.maxX - b.minX;
    let h = b.maxY - b.minY;
    if (w / h < aspect) w = h * aspect;
    else h = w / aspect;
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;
    return { x: cx - w / 2, y: -(cy + h / 2), w, h, cx, cy };
  }, [mode, latest, trajectory, obstacles, size.width, size.height]);

  const px = view.w / size.width; // metres per CSS pixel
  const grid = niceStep(view.w);
  const gridLines = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let x = Math.ceil(view.x / grid) * grid; x <= view.x + view.w; x += grid) xs.push(x);
    for (let y = Math.ceil(view.y / grid) * grid; y <= view.y + view.h; y += grid) ys.push(y);
    return { xs, ys };
  }, [view, grid]);

  const pts = (seg: Position[]) => seg.map((p) => `${p.x_m.toFixed(3)},${(-p.y_m).toFixed(3)}`).join(" ");
  const scaleM = niceStep(view.w * 0.9);

  return (
    <div ref={ref} className={cn("lunar-map", stale && "is-stale", compact && "is-compact")}>
      <svg
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={label}
      >
        <defs>
          <pattern
            id={compact ? "unknown-hatch-mini" : "unknown-hatch"}
            width={8 * px}
            height={8 * px}
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2={8 * px} stroke="#16283b" strokeWidth={2 * px} />
          </pattern>
        </defs>
        <rect
          x={view.x}
          y={view.y}
          width={view.w}
          height={view.h}
          fill={`url(#${compact ? "unknown-hatch-mini" : "unknown-hatch"})`}
        />
        {!compact &&
          gridLines.xs.map((x) => (
            <line key={`gx${x}`} x1={x} x2={x} y1={view.y} y2={view.y + view.h} className="map-grid" strokeWidth={px} />
          ))}
        {!compact &&
          gridLines.ys.map((y) => (
            <line key={`gy${y}`} y1={y} y2={y} x1={view.x} x2={view.x + view.w} className="map-grid" strokeWidth={px} />
          ))}
        {segments.map((seg, i) =>
          seg.length > 1 ? (
            <polyline key={`free${i}`} points={pts(seg)} className="map-free" strokeWidth={ROVER_WIDTH_M} />
          ) : (
            <circle key={`free${i}`} cx={seg[0]!.x_m} cy={-seg[0]!.y_m} r={ROVER_WIDTH_M / 2} className="map-free-dot" />
          ),
        )}
        {segments.map((seg, i) =>
          seg.length > 1 ? (
            <polyline
              key={`trail${i}`}
              points={pts(seg)}
              className="map-trail"
              strokeWidth={(compact ? 1.4 : 1.8) * px}
              strokeDasharray={compact ? undefined : `${5 * px} ${3 * px}`}
            />
          ) : null,
        )}
        {obstacles.map((o) => {
          const color = typeColor[o.type] ?? "#ffc35c";
          const selected = selectedId === o.obstacle_id;
          return (
            <g
              key={o.obstacle_id}
              className={cn("map-obstacle", `status-${o.status}`, selected && "is-selected")}
              onClick={onSelect ? () => onSelect(selected ? null : o.obstacle_id) : undefined}
              role={onSelect ? "button" : undefined}
              aria-label={`${o.type} ${o.obstacle_id}, confidence ${(o.confidence * 100).toFixed(0)} percent, uncertainty ${o.uncertainty_m.toFixed(2)} metres, ${o.status}`}
            >
              <circle
                cx={o.x_m}
                cy={-o.y_m}
                r={o.radius_m + o.uncertainty_m}
                fill="none"
                stroke={color}
                strokeOpacity={0.55}
                strokeWidth={px * 1.2}
                strokeDasharray={`${3 * px} ${3 * px}`}
              />
              <circle
                cx={o.x_m}
                cy={-o.y_m}
                r={Math.max(o.radius_m, 2.5 * px)}
                fill={color}
                fillOpacity={o.status === "lost" ? 0.08 : 0.2 + 0.7 * o.confidence}
                stroke={color}
                strokeWidth={px * (selected ? 3 : o.status === "confirmed" ? 1.6 : 1)}
                strokeDasharray={o.status === "observed" ? `${2 * px} ${2 * px}` : undefined}
              />
              {!compact && (o.persistent_id || selected) && (
                <text x={o.x_m + o.radius_m + 3 * px} y={-o.y_m - 3 * px} fontSize={10 * px} className="map-label">
                  {o.obstacle_id}
                </text>
              )}
            </g>
          );
        })}
        {latest && (
          <g className="map-rover" transform={`translate(${latest.x_m} ${-latest.y_m})`}>
            {!compact && !latest.sigma_xy_saturated && latest.sigma_xy_m > 0 && (
              <circle r={latest.sigma_xy_m} className="map-rover-sigma" strokeWidth={px} />
            )}
            <circle r={(compact ? 4 : 7) * px} className="map-rover-body" strokeWidth={(compact ? 2 : 3) * px} />
            <line
              x1={0}
              y1={0}
              x2={Math.cos(latest.yaw_rad) * (compact ? 9 : 16) * px}
              y2={-Math.sin(latest.yaw_rad) * (compact ? 9 : 16) * px}
              className="map-rover-heading"
              strokeWidth={2 * px}
            />
          </g>
        )}
      </svg>
      {!compact && (
        <div className="map-scale" aria-hidden="true">
          <span style={{ width: `${scaleM / px}px` }} />
          {scaleM} m · odom frame
        </div>
      )}
    </div>
  );
}
