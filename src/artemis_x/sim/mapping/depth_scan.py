"""Depth image -> classified 3-D points in the rover base frame -> per-column free/hit ranges (R4 plan, sections 4 and 6.3).

For a pixel (u, v) at optical-axis depth d, the camera point is (d, -(u - cx) d / fx, -(v - cy) d / fy) and the base point is
linear in d:  p_base = t_mount + d * a(u, v). ``a`` is precomputed once per sampled pixel, so a frame costs a few
multiplications per sample. Height above ground is z_b because the rover is kinematic at z = 0 and never tilts.

Classification of a valid point (thresholds in ``MappingConfig``):
    ground    |z_b| <= ground_band            -> evidence that the line of sight to it is clear
    positive  z_b >= obstacle_height          -> obstacle (rocks, objects, crater rims)
    negative  z_b <= -negative_depth          -> hazard below the ground plane (crater bowls)
    otherwise "uncertain"                     -> ignored for occupancy
Invalid depth (inf / NaN / out of range / beyond max_range) is never turned into free or occupied space.
"""

from __future__ import annotations

import array
import ctypes
import math
from dataclasses import dataclass, field

from ..projection import CameraModel
from .config import MappingConfig
from .frames import Pose2D

KIND_GROUND, KIND_POSITIVE, KIND_NEGATIVE, KIND_UNCERTAIN = "ground", "positive", "negative", "uncertain"


def read_depth(range_finder) -> array.array | list | None:
    """Fast read of a Webots RangeFinder frame as ``array('f')`` (raw float32 copy of the C buffer, no per-pixel Python
    objects). Falls back to the list API if the buffer path is unavailable."""
    w, h = range_finder.getWidth(), range_finder.getHeight()
    try:
        ptr = range_finder.getRangeImage(data_type="buffer")
        if not ptr:
            return None
        out = array.array("f")
        out.frombytes(ctypes.string_at(ptr, w * h * 4))
        return out
    except (TypeError, ValueError, AttributeError):
        img = range_finder.getRangeImage()
        return list(img) if img else None


@dataclass(frozen=True)
class Column:
    bearing_rad: float                 # horizontal direction in the base frame (0 = straight ahead, + = left)
    free_range_m: float | None         # line of sight known clear up to here (from the sensor's ground projection)
    hit_range_m: float | None          # nearest obstacle / hazard in this column
    hit_kind: str | None               # "positive" | "negative" | None
    free_segments: tuple = ()          # R4.2: ((xa, ya), (xb, yb)) base-frame pieces of OBSERVED free ground
    free_chains: tuple = ()            # R4.4 perf: chains of joined ground points (top row first); scan() fills this

    def segments(self) -> tuple:
        """The R4.2 segment form. A chain p0 (top) .. pk (bottom) is: the point pk, then (p[i+1], p[i]) for each link."""
        if self.free_segments or not self.free_chains:
            return self.free_segments
        out = []
        for ch in self.free_chains:
            out.append((ch[-1], ch[-1]))
            out.extend((ch[i + 1], ch[i]) for i in range(len(ch) - 1))
        return tuple(out)


@dataclass
class DepthObservation:
    stamp_s: float                     # sensor sample time (not the time the controller read it)
    frame_index: int
    pose: Pose2D                       # estimated pose at stamp_s (odom)
    columns: list
    obstacle_points: list              # [(x_odom, y_odom, z_b, kind)]
    n_sampled: int
    n_valid: int
    counts: dict = field(default_factory=dict)
    lower_valid_fraction: float = 1.0  # R4.3: valid fraction of samples in the lower half of the image (the ground region)

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_sampled if self.n_sampled else 0.0


class DepthScanner:
    def __init__(self, cam: CameraModel, cfg: MappingConfig | None = None):
        self.cam, self.cfg = cam, cfg or MappingConfig()
        c, s = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
        self.origin = (cam.mount_t[0], cam.mount_t[1])          # sensor ground projection in the base frame
        self.tz = cam.mount_t[2]
        cols = list(range(self.cfg.col_stride // 2, cam.width, self.cfg.col_stride))
        rows = list(range(int(cam.height * self.cfg.row_start_frac), cam.height, self.cfg.row_stride))
        self.n_cols = len(cols)
        self.dtheta_row = self.cfg.row_stride / cam.fy            # angular step between sampled rows (for the gap rule)
        self.bearings = []
        self.samples = []                                        # (pixel index, column, ax, ay, az, horizontal range per metre, lower half)
        for ci, u in enumerate(cols):
            ay = -(u - cam.cx) / cam.fx
            self.bearings.append(math.atan2(ay, c))
            for v in rows:                                             # top row first (farthest first) - see scan()
                zc = -(v - cam.cy) / cam.fy
                self.samples.append((v * cam.width + u, ci, c + s * zc, ay, -s + c * zc, math.hypot(c + s * zc, ay), 1 if v >= cam.height // 2 else 0))
        self.n_lower = sum(sm[6] for sm in self.samples)

    def scan(self, depth, pose: Pose2D, stamp_s: float, frame_index: int = -1) -> DepthObservation:
        """One pass over the samples, each column TOP ROW FIRST (farthest first). Classification as documented above;
        the free-space rules of R4.2 are applied on the fly:

        1. Overhang rule: on open ground, higher pixels see farther. A ground sample with a NEARER positive obstacle point
           (lower than ``overhang_max_height_m``) ABOVE it in the image lies under that obstacle (a bulging boulder, a
           lying tank, a raised panel): the rover could not drive there, so it is not free evidence.
        2. Gap rule: two pixel-consecutive valid ground samples are joined only if the range grows by about what flat
           ground gives between neighbouring sampled rows (x factor, within [min, max]). Anything between them (an
           obstacle, an uncertain point) or a larger jump (a hidden depression, an occluded area) breaks the chain, so
           that space stays UNKNOWN. The blind zone before the nearest ground sample is never included.
        Free space is returned as CHAINS of joined ground points per column (each point converted to a grid cell once);
        ``Column.segments()`` gives the equivalent R4.2 segment list."""
        cfg, cam = self.cfg, self.cam
        lo, hi = cam.min_range, min(cam.max_range, cfg.max_range_m)
        g_band, h_obs, h_neg = cfg.ground_band_m, cfg.obstacle_height_m, -cfg.negative_depth_m
        h_over, margin = cfg.overhang_max_height_m, cfg.overhang_margin_m
        gmax, gmin, gk = cfg.free_gap_max_m, cfg.free_gap_min_m, cfg.free_gap_factor / self.tz * self.dtheta_row
        ox, oy, tz = self.origin[0], self.origin[1], self.tz
        n = self.n_cols
        ground_far = [0.0] * n
        hit_near = [math.inf] * n
        hit_kind = [None] * n
        chains = [[] for _ in range(n)]
        pts = []
        n_ground = n_pos = n_neg = n_unc = n_valid = n_low = 0
        cur_ci, min_above, chain, pr = -1, math.inf, None, 0.0   # chain: points joined so far (top first); pr: its last range
        for idx, ci, ax, ay, az, rh, low in self.samples:     # hot loop: locals only
            if ci != cur_ci:
                cur_ci, min_above, chain = ci, math.inf, None
            d = depth[idx]
            if not (lo <= d < hi):                              # also rejects NaN and inf; invalid samples break nothing
                continue
            n_valid += 1
            n_low += low
            zb = tz + d * az
            r = d * rh
            if -g_band <= zb <= g_band:
                n_ground += 1
                if r > ground_far[ci]:
                    ground_far[ci] = r
                if r < min_above - margin:                      # not under a nearer obstacle seen above
                    x, y = ox + d * ax, oy + d * ay
                    if chain is not None and pr >= r and pr - r <= min(gmax, max(gmin, gk * pr * pr)):
                        chain.append((x, y))
                    else:
                        chain = [(x, y)]
                        chains[ci].append(chain)
                    pr = r
                else:
                    chain = None
            elif zb >= h_obs:
                n_pos += 1
                xb, yb = ox + d * ax, oy + d * ay
                pts.append((xb, yb, zb, KIND_POSITIVE))
                if r < hit_near[ci]:
                    hit_near[ci], hit_kind[ci] = r, KIND_POSITIVE
                if zb <= h_over and r < min_above:
                    min_above = r
                chain = None
            elif zb <= h_neg:
                n_neg += 1
                xb, yb = ox + d * ax, oy + d * ay
                pts.append((xb, yb, zb, KIND_NEGATIVE))
                if r < hit_near[ci]:
                    hit_near[ci], hit_kind[ci] = r, KIND_NEGATIVE
                chain = None
            else:
                n_unc += 1
                chain = None
        counts = {KIND_GROUND: n_ground, KIND_POSITIVE: n_pos, KIND_NEGATIVE: n_neg, KIND_UNCERTAIN: n_unc,
                  "invalid": len(self.samples) - n_valid}
        columns = []
        for ci in range(n):
            hit = hit_near[ci] if hit_near[ci] < math.inf else None
            free = hit if hit is not None else (ground_far[ci] or None)
            columns.append(Column(self.bearings[ci], free, hit, hit_kind[ci], (), tuple(tuple(c) for c in chains[ci])))
        c, s, px, py = math.cos(pose.yaw), math.sin(pose.yaw), pose.x, pose.y      # base -> odom (inlined base_to_odom)
        pts_odom = [(px + c * x - s * y, py + s * x + c * y, z, k) for x, y, z, k in pts]
        return DepthObservation(stamp_s, frame_index, pose, columns, pts_odom, len(self.samples), n_valid, counts,
                                n_low / self.n_lower if self.n_lower else 0.0)
