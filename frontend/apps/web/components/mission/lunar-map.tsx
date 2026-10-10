"use client";

// Live lunar map in the rover's odom frame (x right, y up), drawn ONLY from received telemetry, in the Mission Control
// style of the design reference (glowing explored area, dashed path, badge markers, rover halo):
//  - unknown (dark, hatched): the occupancy grid is not downlinked, nothing is free by default;
//  - explored = camera coverage of RECEIVED poses (FOV 1.0 rad x 6 m mapping range): what the sensors looked at, not free;
//  - traversed (free) = the rover's own footprint along received poses;
//  - path = received poses only; an observation-time gap is never bridged (no invented positions);
//  - obstacles = reconciled ground picture: confidence -> opacity, uncertainty ring, tracking status.
import { useCallback, useEffect, useId, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import { Crosshair, LocateFixed, Maximize, Minus, Plus } from "lucide-react";
import type { TrackedObstacle } from "@/lib/ground/mission-state";
import { trajectorySegments } from "@/lib/ground/trajectory";
import type { Position } from "@/lib/ground/types";
import { cn } from "@/lib/utils";

const ROVER_WIDTH_M = 0.6;
export const CAMERA_FOV_RAD = 1.0; // artemis-webots course camera (fov_rad)
export const SENSOR_RANGE_M = 6.0; // R4 mapping max_range_m: depth beyond is not used
const CAMERA_OFFSET_M = 0.16; // camera mount ahead of the rover centre
const MAX_WEDGES = 360;
const FOLLOW_SPAN_M = 14;

type View = { cx: number; cy: number; span: number }; // span = visible width in metres before aspect fit

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

function niceStep(span: number, parts = 6) {
  const target = span / parts;
  for (const s of [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]) if (s >= target) return s;
  return 500;
}

/** Camera wedges for a thinned subset of received poses (new wedge every 0.5 m or 0.25 rad). */
export function coverageWedges(points: Position[]): string[] {
  const picked: Position[] = [];
  let last: Position | null = null;
  for (const p of points) {
    if (
      !last ||
      Math.hypot(p.x_m - last.x_m, p.y_m - last.y_m) >= 0.5 ||
      Math.abs(Math.atan2(Math.sin(p.yaw_rad - last.yaw_rad), Math.cos(p.yaw_rad - last.yaw_rad))) >= 0.25
    ) {
      picked.push(p);
      last = p;
    }
  }
  const tail = points[points.length - 1];
  if (tail && picked[picked.length - 1] !== tail) picked.push(tail);
  const step = Math.max(1, Math.ceil(picked.length / MAX_WEDGES));
  const half = CAMERA_FOV_RAD / 2;
  return picked
    .filter((_, i) => i % step === 0 || i === picked.length - 1)
    .map((p) => {
      const ax = p.x_m + Math.cos(p.yaw_rad) * CAMERA_OFFSET_M;
      const ay = p.y_m + Math.sin(p.yaw_rad) * CAMERA_OFFSET_M;
      const pt = (a: number) => `${(ax + Math.cos(a) * SENSOR_RANGE_M).toFixed(2)},${(-(ay + Math.sin(a) * SENSOR_RANGE_M)).toFixed(2)}`;
      // SVG y is flipped: a counter-clockwise odom arc is clockwise on screen (sweep flag 0)
      return `M${ax.toFixed(2)},${(-ay).toFixed(2)}L${pt(p.yaw_rad - half)}A${SENSOR_RANGE_M},${SENSOR_RANGE_M} 0 0 0 ${pt(p.yaw_rad + half)}Z`;
    });
}

const typeColor: Record<string, string> = {
  rock: "#5cf07a",
  obstacle: "#ffc35c",
  negative_hazard: "#ff4d6a",
};

function Badge({
  o,
  r,
  selected,
  compact,
}: {
  o: TrackedObstacle;
  r: number;
  selected: boolean;
  compact?: boolean;
}) {
  const lost = o.status === "lost";
  const color = lost ? "#8b98a8" : (typeColor[o.type] ?? "#ffc35c");
  const opacity = lost ? 0.55 : 0.45 + 0.55 * o.confidence;
  if (compact) return <circle r={r * 0.55} fill={color} fillOpacity={opacity} />;
  const t = r * 0.55;
  return (
    <g opacity={opacity}>
      <circle r={r * 1.55} fill={`url(#glow-${o.type})`} />
      <circle
        r={r}
        fill="#06121f"
        fillOpacity={0.85}
        stroke={color}
        strokeWidth={r * (selected ? 0.24 : o.status === "confirmed" ? 0.16 : 0.1)}
        strokeDasharray={o.status === "observed" ? `${r * 0.35} ${r * 0.25}` : undefined}
      />
      {o.type === "obstacle" ? (
        <path d={`M0,${-t}L${t},0L0,${t}L${-t},0Z`} fill={color} />
      ) : (
        <>
          <path d={`M0,${-t}L${t * 1.05},${t * 0.8}L${-t * 1.05},${t * 0.8}Z`} fill={color} />
          <path d={`M0,${-t * 0.35}V${t * 0.3}`} stroke="#06121f" strokeWidth={t * 0.22} strokeLinecap="round" />
          <circle cy={t * 0.58} r={t * 0.12} fill="#06121f" />
        </>
      )}
    </g>
  );
}

export function LunarMap({
  trajectory,
  obstacles,
  initialMode = "fit",
  stale,
  compact,
  interactive,
  selectedId,
  onSelect,
  label,
}: {
  trajectory: Position[];
  obstacles: TrackedObstacle[];
  initialMode?: "follow" | "fit";
  stale: boolean;
  compact?: boolean;
  interactive?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  label: string;
}) {
  const uid = useId().replace(/:/g, "");
  const [ref, size] = useSize<HTMLDivElement>();
  const [mode, setMode] = useState<"follow" | "fit" | "manual">(initialMode);
  const [manual, setManual] = useState<View | null>(null);
  const [followSpan, setFollowSpan] = useState(FOLLOW_SPAN_M);
  const [hover, setHover] = useState<string | null>(null);
  const drag = useRef<{ x: number; y: number; view: View; moved: boolean } | null>(null);
  const latest = trajectory[trajectory.length - 1] ?? null;
  const first = trajectory[0] ?? null;
  const segments = useMemo(() => trajectorySegments(trajectory), [trajectory]);
  const wedges = useMemo(() => coverageWedges(trajectory), [trajectory]);

  const fitView = useMemo<View>(() => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const grow = (x: number, y: number, r = 0) => {
      minX = Math.min(minX, x - r);
      maxX = Math.max(maxX, x + r);
      minY = Math.min(minY, y - r);
      maxY = Math.max(maxY, y + r);
    };
    for (const p of trajectory) grow(p.x_m, p.y_m, 1.5);
    for (const o of obstacles) grow(o.x_m, o.y_m, o.radius_m + o.uncertainty_m + 0.6);
    if (!Number.isFinite(minX)) return { cx: 0, cy: 0, span: 12 };
    const aspect = size.width / size.height;
    const w = Math.max(8, maxX - minX, (maxY - minY) * aspect);
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, span: w * 1.06 };
  }, [trajectory, obstacles, size.width, size.height]);

  const view: View =
    mode === "manual" && manual
      ? manual
      : mode === "follow" && latest
        ? { cx: latest.x_m, cy: latest.y_m, span: followSpan }
        : fitView;
  const aspect = size.width / size.height;
  const vw = view.span;
  const vh = vw / aspect;
  const box = { x: view.cx - vw / 2, y: -view.cy - vh / 2, w: vw, h: vh };
  const px = vw / size.width; // metres per CSS pixel

  const minor = niceStep(vw, compact ? 4 : 10);
  const major = minor * 5;
  const grid = useMemo(() => {
    if (compact) return { minor: [] as number[][], major: [] as number[][] };
    const lines = (step: number) => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let x = Math.ceil(box.x / step) * step; x <= box.x + box.w; x += step) xs.push(x);
      for (let y = Math.ceil(box.y / step) * step; y <= box.y + box.h; y += step) ys.push(y);
      return [xs, ys];
    };
    return { minor: lines(minor), major: lines(major) };
  }, [compact, box.x, box.y, box.w, box.h, minor, major]);

  const pts = (seg: Position[]) => seg.map((p) => `${p.x_m.toFixed(3)},${(-p.y_m).toFixed(3)}`).join(" ");
  const scaleM = niceStep(vw * 0.9, 6);
  const badgeR = Math.max(4.5, Math.min(compact ? 5 : 11, 9 / Math.sqrt(Math.max(1, obstacles.length / 25)))) * px;
  const showLabels = !compact && obstacles.length <= 14;

  const zoom = useCallback(
    (factor: number, at?: { x: number; y: number }) => {
      const base = view;
      const span = Math.min(400, Math.max(2, base.span * factor));
      if (mode === "follow") {
        setFollowSpan(span);
        return;
      }
      let cx = base.cx;
      let cy = base.cy;
      if (at) {
        // keep the point under the cursor fixed
        cx = at.x + (base.cx - at.x) * (span / base.span);
        cy = at.y + (base.cy - at.y) * (span / base.span);
      }
      setManual({ cx, cy, span });
      setMode("manual");
    },
    [view, mode],
  );

  const toWorld = (clientX: number, clientY: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return { x: view.cx, y: view.cy };
    return {
      x: box.x + ((clientX - rect.left) / rect.width) * box.w,
      y: -(box.y + ((clientY - rect.top) / rect.height) * box.h),
    };
  };

  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    if (!interactive) return;
    zoom(e.deltaY > 0 ? 1.18 : 1 / 1.18, toWorld(e.clientX, e.clientY));
  };
  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!interactive || e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, view, moved: false };
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.hypot(dx, dy) < 4) return;
    if (!d.moved) (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    d.moved = true;
    setManual({ cx: d.view.cx - dx * px, cy: d.view.cy + dy * px, span: d.view.span });
    setMode("manual");
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  // wheel listeners must be non-passive to prevent page scroll
  useEffect(() => {
    const el = ref.current;
    if (!el || !interactive) return;
    const stop = (e: globalThis.WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", stop, { passive: false });
    return () => el.removeEventListener("wheel", stop);
  }, [ref, interactive]);

  const g = (name: string) => `${name}-${uid}`;
  const halfFov = CAMERA_FOV_RAD / 2;
  const fovPath = latest
    ? (() => {
        const ax = latest.x_m + Math.cos(latest.yaw_rad) * CAMERA_OFFSET_M;
        const ay = latest.y_m + Math.sin(latest.yaw_rad) * CAMERA_OFFSET_M;
        const pt = (a: number) => `${ax + Math.cos(a) * SENSOR_RANGE_M},${-(ay + Math.sin(a) * SENSOR_RANGE_M)}`;
        return `M${ax},${-ay}L${pt(latest.yaw_rad - halfFov)}A${SENSOR_RANGE_M},${SENSOR_RANGE_M} 0 0 0 ${pt(latest.yaw_rad + halfFov)}Z`;
      })()
    : null;
  const roverPx = compact ? 7 : 13;

  return (
    <div
      ref={ref}
      className={cn("lunar-map", stale && "is-stale", compact && "is-compact", interactive && "is-interactive")}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <svg viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}>
        <defs>
          <pattern id={g("hatch")} width={9 * px} height={9 * px} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2={9 * px} stroke="#0f2132" strokeWidth={2 * px} />
          </pattern>
          <filter id={g("regolith")} x={box.x} y={box.y} width={box.w} height={box.h} filterUnits="userSpaceOnUse">
            <feTurbulence type="fractalNoise" baseFrequency={compact ? 1.2 : 3.4} numOctaves={4} seed={7} result="n" />
            <feColorMatrix
              in="n"
              type="matrix"
              values="0 0 0 0 0.62  0 0 0 0 0.72  0 0 0 0 0.84  0 0 0 0.55 -0.12"
              result="tint"
            />
            <feComposite in="tint" in2="SourceGraphic" operator="in" result="tex" />
            <feMerge>
              <feMergeNode in="SourceGraphic" />
              <feMergeNode in="tex" />
            </feMerge>
          </filter>
          <filter id={g("edge")} x={box.x} y={box.y} width={box.w} height={box.h} filterUnits="userSpaceOnUse">
            <feMorphology in="SourceAlpha" operator="dilate" radius={1.4 * px} result="d" />
            <feComposite in="d" in2="SourceAlpha" operator="out" result="edge" />
            <feFlood floodColor="#55c8ff" />
            <feComposite in2="edge" operator="in" result="line" />
            <feGaussianBlur in="line" stdDeviation={3 * px} result="glow" />
            <feMerge>
              <feMergeNode in="glow" />
              <feMergeNode in="line" />
            </feMerge>
          </filter>
          <filter id={g("blur")} x={box.x} y={box.y} width={box.w} height={box.h} filterUnits="userSpaceOnUse">
            <feGaussianBlur stdDeviation={2.5 * px} />
          </filter>
          <radialGradient id={g("halo")}>
            <stop offset="0" stopColor="#3aa0ff" stopOpacity="0.55" />
            <stop offset="0.6" stopColor="#2a7bff" stopOpacity="0.18" />
            <stop offset="1" stopColor="#2a7bff" stopOpacity="0" />
          </radialGradient>
          <linearGradient id={g("fov")} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#7fd0ff" stopOpacity="0.35" />
            <stop offset="1" stopColor="#7fd0ff" stopOpacity="0" />
          </linearGradient>
          {Object.entries(typeColor).map(([t, c]) => (
            <radialGradient key={t} id={`glow-${t}`}>
              <stop offset="0.45" stopColor={c} stopOpacity="0.35" />
              <stop offset="1" stopColor={c} stopOpacity="0" />
            </radialGradient>
          ))}
        </defs>

        {/* unknown: everything not covered below */}
        <rect x={box.x} y={box.y} width={box.w} height={box.h} className="map-unknown-bg" />
        <rect x={box.x} y={box.y} width={box.w} height={box.h} fill={`url(#${g("hatch")})`} />
        {grid.minor[0]?.map((x) => (
          <line key={`mx${x}`} x1={x} x2={x} y1={box.y} y2={box.y + box.h} className="map-grid" strokeWidth={px} />
        ))}
        {grid.minor[1]?.map((y) => (
          <line key={`my${y}`} y1={y} y2={y} x1={box.x} x2={box.x + box.w} className="map-grid" strokeWidth={px} />
        ))}
        {grid.major[0]?.map((x) => (
          <line key={`Mx${x}`} x1={x} x2={x} y1={box.y} y2={box.y + box.h} className="map-grid-major" strokeWidth={px} />
        ))}
        {grid.major[1]?.map((y) => (
          <line key={`My${y}`} y1={y} y2={y} x1={box.x} x2={box.x + box.w} className="map-grid-major" strokeWidth={px} />
        ))}

        {/* explored: camera coverage of received poses (uniform union via group opacity), regolith texture, glowing rim */}
        {wedges.length > 0 && (
          <>
            <g className="map-explored" filter={compact ? undefined : `url(#${g("regolith")})`}>
              {wedges.map((d, i) => (
                <path key={i} d={d} />
              ))}
            </g>
            {!compact && (
              <g filter={`url(#${g("edge")})`} opacity={0.9}>
                {wedges.map((d, i) => (
                  <path key={i} d={d} fill="#fff" />
                ))}
              </g>
            )}
          </>
        )}

        {/* traversed: the rover's own footprint = known free */}
        {segments.map((seg, i) =>
          seg.length > 1 ? (
            <polyline key={`free${i}`} points={pts(seg)} className="map-free" strokeWidth={ROVER_WIDTH_M} />
          ) : (
            <circle key={`free${i}`} cx={seg[0]!.x_m} cy={-seg[0]!.y_m} r={ROVER_WIDTH_M / 2} className="map-free-dot" />
          ),
        )}
        {!compact &&
          segments.map((seg, i) =>
            seg.length > 1 ? (
              <polyline
                key={`glow${i}`}
                points={pts(seg)}
                className="map-trail-glow"
                strokeWidth={5 * px}
                filter={`url(#${g("blur")})`}
              />
            ) : null,
          )}
        {segments.map((seg, i) =>
          seg.length > 1 ? (
            <polyline
              key={`trail${i}`}
              points={pts(seg)}
              className="map-trail"
              strokeWidth={(compact ? 1.4 : 2.2) * px}
              strokeDasharray={compact ? undefined : `${7 * px} ${4 * px}`}
            />
          ) : null,
        )}

        {/* start of the received track */}
        {first && !compact && (
          <g transform={`translate(${first.x_m} ${-first.y_m})`} className="map-start" strokeWidth={1.5 * px}>
            <path d={`M0,0V${-18 * px}`} />
            <path d={`M0,${-18 * px}L${11 * px},${-14 * px}L0,${-10 * px}Z`} />
            <text x={4 * px} y={10 * px} fontSize={9 * px}>
              START
            </text>
          </g>
        )}

        {/* obstacles: true footprint + uncertainty, then a constant-size badge */}
        {obstacles.map((o) => {
          const color = typeColor[o.type] ?? "#ffc35c";
          const selected = selectedId === o.obstacle_id;
          const hovered = hover === o.obstacle_id;
          return (
            <g
              key={o.obstacle_id}
              className={cn("map-obstacle", `status-${o.status}`, selected && "is-selected")}
              onClick={onSelect && !compact ? () => onSelect(selected ? null : o.obstacle_id) : undefined}
              onPointerEnter={compact ? undefined : () => setHover(o.obstacle_id)}
              onPointerLeave={compact ? undefined : () => setHover((h) => (h === o.obstacle_id ? null : h))}
              role={onSelect && !compact ? "button" : undefined}
              aria-label={`${o.type} ${o.obstacle_id}, confidence ${(o.confidence * 100).toFixed(0)} percent, uncertainty ${o.uncertainty_m.toFixed(2)} metres, ${o.status}`}
            >
              {!compact && (
                <>
                  <circle
                    cx={o.x_m}
                    cy={-o.y_m}
                    r={o.radius_m + o.uncertainty_m}
                    fill="none"
                    stroke={color}
                    strokeOpacity={0.5}
                    strokeWidth={px}
                    strokeDasharray={`${3 * px} ${3 * px}`}
                  />
                  <circle cx={o.x_m} cy={-o.y_m} r={o.radius_m} fill={color} fillOpacity={0.16} />
                </>
              )}
              <g transform={`translate(${o.x_m} ${-o.y_m})`}>
                <Badge o={o} r={badgeR} selected={selected || hovered} compact={compact} />
              </g>
              {(showLabels || selected || hovered) && !compact && (
                <text
                  x={o.x_m + badgeR * 1.3}
                  y={-o.y_m - badgeR * 0.9}
                  fontSize={10 * px}
                  className={cn("map-label", (selected || hovered) && "is-strong")}
                >
                  {o.obstacle_id}
                  {(selected || hovered) && ` · ${(o.confidence * 100).toFixed(0)}% · ±${o.uncertainty_m.toFixed(2)} m`}
                </text>
              )}
            </g>
          );
        })}

        {/* rover: halo, current camera footprint, glyph rotated by yaw */}
        {latest && (
          <>
            {fovPath && !compact && <path d={fovPath} className="map-fov" />}
            <g transform={`translate(${latest.x_m} ${-latest.y_m})`} className="map-rover">
              <circle r={roverPx * 2.6 * px} fill={`url(#${g("halo")})`} />
              {!compact && !latest.sigma_xy_saturated && latest.sigma_xy_m > 0 && (
                <circle r={latest.sigma_xy_m} className="map-rover-sigma" strokeWidth={px} />
              )}
              {!compact && <circle r={roverPx * 1.7 * px} className="map-rover-ring" strokeWidth={1.4 * px} />}
              <g transform={`rotate(${(-latest.yaw_rad * 180) / Math.PI}) scale(${px * roverPx})`}>
                <rect x={-0.95} y={-0.75} width={1.9} height={1.5} rx={0.3} className="map-rover-body" strokeWidth={0.14} />
                <rect x={-0.85} y={-1.02} width={0.5} height={0.26} rx={0.08} className="map-rover-wheel" />
                <rect x={0.35} y={-1.02} width={0.5} height={0.26} rx={0.08} className="map-rover-wheel" />
                <rect x={-0.85} y={0.76} width={0.5} height={0.26} rx={0.08} className="map-rover-wheel" />
                <rect x={0.35} y={0.76} width={0.5} height={0.26} rx={0.08} className="map-rover-wheel" />
                <path d="M0.35,-0.45L1.05,0L0.35,0.45Z" className="map-rover-nose" />
              </g>
            </g>
          </>
        )}
      </svg>

      {!compact && (
        <>
          <div className="map-axes" aria-hidden="true" title="odom frame axes (no absolute north on the rover's odometry)">
            <svg viewBox="-24 -24 48 48">
              <circle r="21" className="axes-ring" />
              <path d="M0,4V-17M-4,-12L0,-18L4,-12" className="axes-y" />
              <path d="M-4,0H17M12,-4L18,0L12,4" className="axes-x" />
              <text x="-3" y="22" className="axes-label">odom</text>
              <text x="3" y="-14" className="axes-label">+y</text>
              <text x="11" y="-5" className="axes-label">+x</text>
            </svg>
          </div>
          <div className="map-scale" aria-hidden="true">
            <span style={{ width: `${scaleM / px}px` }} />
            <em>{scaleM} m</em>
          </div>
        </>
      )}
      {interactive && (
        <div className="map-zoom" role="group" aria-label="Map zoom">
          <button onClick={() => zoom(1 / 1.4)} aria-label="Zoom in" title="Zoom in (or mouse wheel)">
            <Plus />
          </button>
          <button onClick={() => zoom(1.4)} aria-label="Zoom out" title="Zoom out">
            <Minus />
          </button>
          <button
            className={cn(mode === "fit" && "is-active")}
            onClick={() => setMode("fit")}
            aria-pressed={mode === "fit"}
            title="Fit the whole mission"
          >
            <Maximize />
          </button>
          <button
            className={cn(mode === "follow" && "is-active")}
            onClick={() => setMode("follow")}
            aria-pressed={mode === "follow"}
            title="Follow the rover"
            disabled={!latest}
          >
            {mode === "follow" ? <LocateFixed /> : <Crosshair />}
          </button>
        </div>
      )}
    </div>
  );
}
