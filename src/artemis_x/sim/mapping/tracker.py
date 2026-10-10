"""Obstacle tracker (R4.4, plan section 6.5). Stdlib only. Rocks are static; tracks exist to de-duplicate detections
across frames, give each obstacle a persistent ID, and accumulate a confident position, size and label.

  association  greedy nearest neighbour on centre distance, gate = gate0 + gate_per_m * range + the larger of the two
               radii (fragments of one large rock associate); Hungarian / JPDA would add nothing at this density; each
               track takes at most one measurement per frame
  depth support  a hit counts toward confirmation only if its box held >= track_min_support depth points >= 2 cm high
               (plan 6.4: a YOLO box on bare terrain with no depth support never confirms)
  refinement   ``refine(grid)`` snaps confirmed tracks to their connected OCCUPIED blob in the depth grid: circle fit for
               blobs >= 0.5 m (centre / radius of a boulder from its visible arc); tracks sharing a boulder-sized blob
               (>= 0.8 m) are merged as fragments of one object; smaller clusters are left alone (touching small rocks);
               terrain blobs are never fitted or merged: wider than refine_max_extent_m, or with NEGATIVE (crater bowl)
               cells inside their bounding box - a crater rim is not a boulder, and rocks on a rim stay separate
  update       inverse-variance weighted mean of the centre; radius and confidence by exponential moving average
  lifecycle    tentative -> confirmed after ``confirm_hits`` hits within ``confirm_window_s``; tentative tracks that are
               not confirmed in time are deleted (false-positive hygiene); a miss is counted only when the track's centre
               should have been visible (inside the FOV, within miss_range_m, not occluded by nearer depth) and no
               measurement associated; confirmed -> lost after ``lost_misses`` consecutive misses (kept, flagged)
  merging      two confirmed tracks whose discs overlap by more than ``merge_overlap`` of the smaller disc are merged
               into the one with more hits (the merge count is a KPI: duplicates that association alone did not prevent)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import MappingConfig
from .frames import Pose2D, odom_to_base

TENTATIVE, CONFIRMED, LOST = "tentative", "confirmed", "lost"


@dataclass
class Track:
    id: int
    cls: str
    x: float
    y: float
    sigma: float
    radius_m: float
    height_m: float
    confidence: float
    hits: int
    misses: int
    first_seen_s: float
    last_seen_s: float
    status: str = TENTATIVE
    bimodal_hits: int = 0
    hit_times: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "cls": self.cls, "x": round(self.x, 4), "y": round(self.y, 4), "sigma": round(self.sigma, 4),
                "radius_m": round(self.radius_m, 3), "height_m": round(self.height_m, 3), "confidence": round(self.confidence, 3),
                "hits": self.hits, "misses": self.misses, "first_seen_s": round(self.first_seen_s, 3),
                "last_seen_s": round(self.last_seen_s, 3), "status": self.status, "bimodal_hits": self.bimodal_hits}


class ObstacleTracker:
    def __init__(self, cfg: MappingConfig | None = None):
        self.cfg = cfg or MappingConfig()
        self.tracks: list = []
        self.next_id = 1
        self.stats = {"measurements": 0, "associated": 0, "new_tracks": 0, "confirmed": 0, "deleted_tentative": 0,
                      "merged": 0, "misses": 0, "lost": 0}

    def update(self, meas: list, pose: Pose2D, t: float, depth=None, cam=None) -> list:
        """Associate one frame's measurements (taken at pose/t). Returns the track id per measurement (same order)."""
        cfg = self.cfg
        self.stats["measurements"] += len(meas)
        pairs = []
        for mi, m in enumerate(meas):
            gate = cfg.track_gate0_m + cfg.track_gate_per_m * m.range_m
            for tr in self.tracks:
                d = math.hypot(tr.x - m.centre_odom[0], tr.y - m.centre_odom[1])
                if d <= gate + max(tr.radius_m, m.radius_m):
                    pairs.append((d, mi, tr))
        pairs.sort(key=lambda p: p[0])
        used_m, used_t, ids = set(), set(), [None] * len(meas)
        for d, mi, tr in pairs:
            if mi in used_m or tr.id in used_t:
                continue
            used_m.add(mi); used_t.add(tr.id)
            self._absorb(tr, meas[mi], t)
            ids[mi] = tr.id
            self.stats["associated"] += 1
        for mi, m in enumerate(meas):
            if ids[mi] is None:
                sup = getattr(m, "support", cfg.track_min_support) >= cfg.track_min_support
                tr = Track(self.next_id, "rock", m.centre_odom[0], m.centre_odom[1], m.sigma_m, m.radius_m, m.height_m, m.conf,
                           1, 0, t, t, TENTATIVE, int(m.depth_bimodal), [t] if sup else [])
                self.next_id += 1
                self.tracks.append(tr)
                ids[mi] = tr.id
                self.stats["new_tracks"] += 1
        self._misses(pose, t, used_t, depth, cam)
        self._housekeeping(t)
        return ids

    def _absorb(self, tr: Track, m, t: float) -> None:
        cfg = self.cfg
        w0, w1 = 1.0 / (tr.sigma ** 2), 1.0 / (m.sigma_m ** 2)
        tr.x = (w0 * tr.x + w1 * m.centre_odom[0]) / (w0 + w1)
        tr.y = (w0 * tr.y + w1 * m.centre_odom[1]) / (w0 + w1)
        tr.sigma = math.sqrt(1.0 / (w0 + w1))
        a = cfg.track_ema
        tr.radius_m = (1 - a) * tr.radius_m + a * m.radius_m
        tr.height_m = (1 - a) * tr.height_m + a * m.height_m
        tr.confidence = (1 - a) * tr.confidence + a * m.conf
        tr.hits += 1
        tr.misses = 0
        tr.last_seen_s = t
        tr.bimodal_hits += int(m.depth_bimodal)
        if getattr(m, "support", cfg.track_min_support) >= cfg.track_min_support:
            tr.hit_times.append(t)                            # only depth-supported hits confirm a track
        tr.hit_times = [h for h in tr.hit_times if t - h <= cfg.confirm_window_s]
        if tr.status != CONFIRMED and len(tr.hit_times) >= cfg.confirm_hits:
            tr.status = CONFIRMED
            self.stats["confirmed"] += 1

    def _misses(self, pose: Pose2D, t: float, seen: set, depth, cam) -> None:
        cfg = self.cfg
        for tr in self.tracks:
            if tr.id in seen or tr.status == TENTATIVE:
                continue
            bx, by = odom_to_base((tr.x, tr.y), pose)
            cx, cy = bx - (cam.mount_t[0] if cam else 0.16), by
            rng = math.hypot(cx, cy)
            if rng > cfg.track_miss_range_m or cx <= 0 or abs(math.atan2(cy, cx)) > cfg.track_fov_half_rad:
                continue
            if depth is not None and cam is not None and not self._visible(bx, by, tr, depth, cam):
                continue
            tr.misses += 1
            self.stats["misses"] += 1
            if tr.status == CONFIRMED and tr.misses >= cfg.lost_misses:
                tr.status = LOST
                self.stats["lost"] += 1

    def _visible(self, bx: float, by: float, tr: Track, depth, cam) -> bool:
        """Should the camera see the track now? Project its centre at mid-height into the image and look at the depth in
        a small window there: no valid return (sky / out of range) or mostly NEARER surfaces (an occluder) -> not
        visible, so no miss is charged."""
        px, py, pz = bx - cam.mount_t[0], by - cam.mount_t[1], 0.5 * max(tr.height_m, 0.04) - cam.mount_t[2]
        c, s = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
        xc, zc = c * px - s * pz, s * px + c * pz             # base -> camera (inverse of camera_to_rover's pitch)
        if xc <= cam.min_range:
            return False
        u, v = cam.cx - cam.fx * py / xc, cam.cy - cam.fy * zc / xc
        valid = near = 0
        for dv in (-8, 0, 8):
            for du in (-8, 0, 8):
                a, b = int(round(u + du)), int(round(v + dv))
                if 0 <= a < cam.width and 0 <= b < cam.height:
                    d = depth[b * cam.width + a]
                    if cam.min_range <= d < cam.max_range:
                        valid += 1
                        near += d < xc - tr.radius_m - 0.3
        return valid >= 5 and near < valid / 2

    def _housekeeping(self, t: float) -> None:
        cfg = self.cfg
        keep = []
        for tr in self.tracks:
            if tr.status == TENTATIVE and t - tr.first_seen_s > cfg.tentative_timeout_s:
                self.stats["deleted_tentative"] += 1
                continue
            keep.append(tr)
        self.tracks = keep
        conf = [tr for tr in self.tracks if tr.status != TENTATIVE]
        gone = set()
        for i, a in enumerate(conf):
            if a.id in gone:
                continue
            for b in conf[i + 1:]:
                if b.id in gone:
                    continue
                d = math.hypot(a.x - b.x, a.y - b.y)
                r = min(a.radius_m, b.radius_m)
                if d < (a.radius_m + b.radius_m) - 2 * r * cfg.merge_overlap:   # overlap of the smaller disc > merge_overlap
                    win, lose = (a, b) if a.hits >= b.hits else (b, a)
                    w0, w1 = win.hits, lose.hits
                    win.x = (w0 * win.x + w1 * lose.x) / (w0 + w1); win.y = (w0 * win.y + w1 * lose.y) / (w0 + w1)
                    win.hits += lose.hits
                    win.radius_m = max(win.radius_m, lose.radius_m)
                    gone.add(lose.id)
                    self.stats["merged"] += 1
        if gone:
            self.tracks = [tr for tr in self.tracks if tr.id not in gone]

    def reset(self) -> int:
        """Explicit map / mission reset: drop every track (confirmed and lost included). Ids keep increasing, so an id is
        never reused within a mission. Returns the number of tracks dropped."""
        n = len(self.tracks)
        self.tracks = []
        self.stats["resets"] = self.stats.get("resets", 0) + 1
        return n

    def refine(self, grid) -> dict:
        """Snap confirmed tracks to the depth grid (no ground truth): find the connected OCCUPIED blob under each track,
        circle-fit large blobs for centre / radius, merge tracks that sit on the same boulder-sized blob."""
        from .grid import NEGATIVE, OCCUPIED
        cfg = self.cfg
        n, res = grid.n, grid.res
        blob_of: dict = {}
        blobs: list = []
        out = {"refined": 0, "merged": 0}
        by_blob: dict = {}
        for tr in [t for t in self.tracks if t.status != TENTATIVE]:
            k0 = grid.index(tr.x, tr.y)
            if k0 < 0:
                continue
            j0, i0 = divmod(k0, n)
            reach = int(math.ceil((cfg.refine_search_m + tr.radius_m) / res))
            seed, best = None, None
            for dj in range(-reach, reach + 1):
                for di in range(-reach, reach + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < n and 0 <= j < n and grid.state_at(j * n + i) == OCCUPIED:
                        d = di * di + dj * dj
                        if best is None or d < best:
                            best, seed = d, j * n + i
            if seed is None:
                continue
            if seed not in blob_of:                           # flood fill (8-connected), capped
                bid = len(blobs)
                cells, stack = [], [seed]
                blob_of[seed] = bid
                while stack and len(cells) < 3000:
                    k = stack.pop()
                    cells.append(k)
                    j, i = divmod(k, n)
                    for dj in (-1, 0, 1):
                        for di in (-1, 0, 1):
                            a, b = i + di, j + dj
                            if 0 <= a < n and 0 <= b < n:
                                q = b * n + a
                                if q not in blob_of and grid.state_at(q) == OCCUPIED:
                                    blob_of[q] = bid
                                    stack.append(q)
                blobs.append(cells)
            bid = blob_of[seed]
            by_blob.setdefault(bid, []).append(tr)
        for bid, trs in by_blob.items():
            pts = [grid.cell_centre(k) for k in blobs[bid]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + res
            if extent > cfg.refine_max_extent_m or _has_negative(grid, blobs[bid]):
                out.setdefault("terrain_blobs_skipped", 0)
                out["terrain_blobs_skipped"] += 1
                continue                                      # a crater rim / ridge: not an object
            fit = _circle_fit(pts) if extent >= cfg.refine_fit_min_extent_m and len(pts) >= 6 else None
            if fit is not None:
                cx, cy, r = fit
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                if 0.25 * extent <= r <= extent and math.hypot(cx - mx, cy - my) <= extent:
                    for tr in trs:
                        tr.x, tr.y, tr.radius_m = cx, cy, r
                        out["refined"] += 1
            if len(trs) > 1 and extent >= cfg.refine_merge_min_extent_m:
                trs.sort(key=lambda t: -t.hits)
                win = trs[0]
                for lose in trs[1:]:
                    win.hits += lose.hits
                    win.radius_m = max(win.radius_m, lose.radius_m)
                    self.tracks.remove(lose)
                    self.stats["merged"] += 1
                    out["merged"] += 1
        return out

    def confirmed(self) -> list:
        return [tr for tr in self.tracks if tr.status != TENTATIVE]


def _circle_fit(pts: list):
    """Algebraic (Kasa) least-squares circle through 2-D points -> (cx, cy, r), or None if degenerate."""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    suu = svv = suv = suuu = svvv = suvv = svuu = 0.0
    for x, y in pts:
        u, v = x - mx, y - my
        suu += u * u; svv += v * v; suv += u * v
        suuu += u * u * u; svvv += v * v * v; suvv += u * v * v; svuu += v * u * u
    det = suu * svv - suv * suv
    if abs(det) < 1e-12:
        return None
    b1, b2 = 0.5 * (suuu + suvv), 0.5 * (svvv + svuu)
    uc = (b1 * svv - b2 * suv) / det
    vc = (b2 * suu - b1 * suv) / det
    r = math.sqrt(uc * uc + vc * vc + (suu + svv) / n)
    return mx + uc, my + vc, r


def _has_negative(grid, cells: list) -> bool:
    """Any NEGATIVE (below-ground) cell inside the blob's bounding box - the blob surrounds or borders a depression."""
    from .grid import NEGATIVE
    n = grid.n
    js = [k // n for k in cells]; is_ = [k % n for k in cells]
    for j in range(min(js), max(js) + 1):
        for i in range(min(is_), max(is_) + 1):
            if grid.state_at(j * n + i) == NEGATIVE:
                return True
    return False
