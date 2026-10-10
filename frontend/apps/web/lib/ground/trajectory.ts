// Trajectory rendering helpers (pure). Poses are sent every 0.5 s (R5.1 adapter): a larger observation-time gap means
// packets never reached the ground, and the map must not draw a line through positions nobody received.
import type { Position } from "./types";

export const TRAJECTORY_GAP_S = 3;
export const MAX_RENDER_POINTS = 2500;

/** Splits the trajectory where observation time jumps and decimates long runs (first/last of a run always kept). */
export function trajectorySegments(
  points: Position[],
  gapS = TRAJECTORY_GAP_S,
  maxPoints = MAX_RENDER_POINTS,
): Position[][] {
  const step = Math.max(1, Math.ceil(points.length / maxPoints));
  const segments: Position[][] = [];
  let current: Position[] = [];
  points.forEach((p, i) => {
    const prev = points[i - 1];
    const next = points[i + 1];
    if (prev && p.t_obs_s - prev.t_obs_s > gapS) {
      if (current.length) segments.push(current);
      current = [];
    }
    const endOfRun = !next || next.t_obs_s - p.t_obs_s > gapS;
    if (current.length === 0 || i % step === 0 || endOfRun) current.push(p);
  });
  if (current.length) segments.push(current);
  return segments;
}
