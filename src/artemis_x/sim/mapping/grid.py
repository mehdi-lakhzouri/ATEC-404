"""Rolling local log-odds occupancy grid (R4 plan section 6.2, decision D2: 0.10 m cells, 12 x 12 m window).

Cell (i, j) covers [origin_x + i res, origin_x + (i+1) res) x [origin_y + j res, ...) in the ``odom`` frame; flat index
k = j * n + i. The origin is always a whole multiple of the resolution, so recentring is an integer shift and a cell keeps
exactly the same odom footprint for as long as it stays inside the window (no resampling, no blur).

Per cycle each cell receives at most ONE update: obstacle hits first (``l_hit``), then observed-free evidence (``l_miss``)
for cells not hit in this cycle. Values are clamped to [l_min, l_max], so evidence can always be revised.

States: UNKNOWN (never observed, or |l| inside the thresholds), FREE (l < l_free), OCCUPIED (l > l_occupied); an occupied
cell whose latest hit was below the ground plane is reported as NEGATIVE (hazard such as a crater bowl).
Only what the depth layer observed changes a cell; the camera blind zone, occluded areas and anything never seen stay UNKNOWN.

Provenance (release review): every cell carries bit flags of the evidence sources that raised it since the last reset -
SRC_DEPTH (a depth obstacle / hazard hit), SRC_YOLO_POINTS (depth points 2-5 cm high selected by a confirmed YOLO track),
SRC_YOLO_FOOTPRINT (a confirmed small rock's footprint) - and, in the exported layer, SRC_PROTECTED for cells currently
withheld from FREE evidence. Depth-based and YOLO-protected occupancy therefore stay distinguishable.
"""

from __future__ import annotations

import array
import math

from .config import MappingConfig
from .depth_scan import KIND_NEGATIVE, DepthObservation

UNKNOWN, FREE, OCCUPIED, NEGATIVE = 0, 1, 2, 3
NEVER = -1.0
SRC_DEPTH, SRC_YOLO_POINTS, SRC_YOLO_FOOTPRINT, SRC_PROTECTED = 1, 2, 4, 8     # provenance bit flags


class LogOddsGrid:
    def __init__(self, cfg: MappingConfig | None = None, centre: tuple = (0.0, 0.0)):
        self.cfg = cfg or MappingConfig()
        self.res, self.n = self.cfg.resolution_m, self.cfg.grid_cells
        self.inv = 1.0 / self.res                            # ONE indexing expression everywhere: floor((x - origin) * inv)
        self.occ_th, self.free_th = self.cfg.l_occupied + 1e-6, self.cfg.l_free - 1e-6   # float32 storage: 0.85f > 0.85
        N = self.n * self.n
        self.logodds = array.array("f", bytes(4 * N))
        self.last_obs = array.array("f", [NEVER]) * N       # sim time of the last update (NEVER = never observed)
        self.neg = array.array("b", bytes(N))                # 1 if the latest hit in the cell was a negative hazard
        self.src = array.array("B", bytes(N))                # provenance flags (SRC_*) of the evidence that raised the cell
        self._mark = array.array("i", [-1]) * N              # cycle id of the last update (one update per cell per cycle)
        self.cycle_id = 0
        self.touched: list = []                              # cells updated since the cost layer last consumed them (R4.3)
        self.protected: set = set()                          # cells that never receive FREE evidence (confirmed small rocks)
        self.origin_x = self.origin_y = 0.0
        self.recenter(*centre, force=True)
        self.recenters = 0
        self.resets = 0

    def reset(self, centre: tuple) -> None:
        """Explicit map / mission reset: every cell back to UNKNOWN with no provenance, nothing protected, window centred
        on ``centre``. Only called on request (Mapper.reset_map); nothing in normal mapping clears the grid."""
        N = self.n * self.n
        self.logodds = array.array("f", bytes(4 * N))
        self.last_obs = array.array("f", [NEVER]) * N
        self.neg = array.array("b", bytes(N))
        self.src = array.array("B", bytes(N))
        self._mark = array.array("i", [-1]) * N
        self.touched, self.protected = [], set()
        self.recenter(*centre, force=True)
        self.resets += 1

    # ---- geometry --------------------------------------------------------------------------------------------------
    def _snap(self, v: float) -> float:
        return round(round(v / self.res) * self.res, 9)

    def centre(self) -> tuple:
        h = self.n * self.res / 2
        return self.origin_x + h, self.origin_y + h

    def index(self, x: float, y: float) -> int:
        i = math.floor((x - self.origin_x) * self.inv)
        j = math.floor((y - self.origin_y) * self.inv)
        return j * self.n + i if 0 <= i < self.n and 0 <= j < self.n else -1

    def cell_centre(self, k: int) -> tuple:
        j, i = divmod(k, self.n)
        return self.origin_x + (i + 0.5) * self.res, self.origin_y + (j + 0.5) * self.res

    def recenter(self, x: float, y: float, force: bool = False) -> bool:
        """Shift the window by whole cells so (x, y) is near its centre, once (x, y) is ``recenter_m`` off-centre.
        Cells that stay inside keep their values; cells that enter start UNKNOWN; cells that leave are dropped."""
        cx, cy = self.centre()
        if not force and max(abs(x - cx), abs(y - cy)) < self.cfg.recenter_m:
            return False
        h = self.n * self.res / 2
        nox, noy = self._snap(x - h), self._snap(y - h)
        di, dj = round((nox - self.origin_x) / self.res), round((noy - self.origin_y) / self.res)
        if force:
            self.origin_x, self.origin_y = nox, noy
            return True
        n, N = self.n, self.n * self.n
        lo, lob, ng, sr = array.array("f", bytes(4 * N)), array.array("f", [NEVER]) * N, array.array("b", bytes(N)), array.array("B", bytes(N))
        for j in range(max(0, -dj), min(n, n - dj)):              # new row j <- old row j + dj
            oj = j + dj
            i0, i1 = max(0, -di), min(n, n - di)
            if i1 <= i0:
                continue
            a, b = j * n + i0, oj * n + i0 + di
            lo[a:a + i1 - i0] = self.logodds[b:b + i1 - i0]
            lob[a:a + i1 - i0] = self.last_obs[b:b + i1 - i0]
            ng[a:a + i1 - i0] = self.neg[b:b + i1 - i0]
            sr[a:a + i1 - i0] = self.src[b:b + i1 - i0]
        self.logodds, self.last_obs, self.neg, self.src = lo, lob, ng, sr
        self._mark = array.array("i", [-1]) * N
        self.origin_x, self.origin_y = nox, noy
        self.recenters += 1
        self.last_shift = (di, dj)                            # read by the cost layer to shift instead of rebuilding
        return True

    # ---- updates ---------------------------------------------------------------------------------------------------
    def integrate(self, obs: DepthObservation) -> dict:
        """One depth observation (R4.1) -> one grid update. Returns per-cycle counters."""
        cfg = self.cfg
        self.cycle_id += 1
        cyc, t = self.cycle_id, obs.stamp_s
        lo, mark, last, neg, src = self.logodds, self._mark, self.last_obs, self.neg, self.src
        add = self.touched.append
        l_hit, l_miss, l_min, l_max = cfg.l_hit, cfg.l_miss, cfg.l_min, cfg.l_max
        ox, oy, res, n, inv = self.origin_x, self.origin_y, self.res, self.n, self.inv
        floor = math.floor
        hits = frees = 0
        # 1. obstacle / hazard hits (odom points)
        for x, y, _z, kind in obs.obstacle_points:
            i, j = floor((x - ox) * inv), floor((y - oy) * inv)
            if 0 <= i < n and 0 <= j < n:
                k = j * n + i
                if mark[k] != cyc:
                    mark[k] = cyc
                    v = lo[k] + l_hit
                    lo[k] = v if v < l_max else l_max
                    last[k] = t
                    neg[k] = 1 if kind == KIND_NEGATIVE else 0
                    src[k] |= SRC_DEPTH
                    hits += 1
                    add(k)
                elif kind == KIND_NEGATIVE:
                    neg[k] = 1
        # 2. observed free ground (base-frame chains / segments -> odom), cells not already hit this cycle
        p = obs.pose
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        bx, by = p.x - ox, p.y - oy                           # rover position relative to the grid origin
        step = res * 0.5
        prot = self.protected
        for col in obs.columns:
            for ch in col.free_chains:                        # each point converted once; neighbours share endpoints
                pi = pj = None
                pax = pay = 0.0
                for (xa, ya) in ch:
                    ax, ay = bx + c * xa - s * ya, by + s * xa + c * ya
                    fi, fj = ax * inv, ay * inv
                    i, j = (int(fi) if fi >= 0 else -1), (int(fj) if fj >= 0 else -1)
                    if pi is not None and (abs(i - pi) > 1 or abs(j - pj) > 1):   # long link: sample along it
                        m = int(math.hypot(ax - pax, ay - pay) / step) + 1
                        cells = [(floor((pax + q / m * (ax - pax)) * inv), floor((pay + q / m * (ay - pay)) * inv)) for q in range(1, m)]
                        cells.append((i, j))
                    elif pi == i and pj == j:
                        cells = ()
                    else:
                        cells = ((i, j),)
                    for ci_, cj_ in cells:
                        if 0 <= ci_ < n and 0 <= cj_ < n:
                            k = cj_ * n + ci_
                            if mark[k] != cyc and k not in prot:
                                mark[k] = cyc
                                v = lo[k] + l_miss
                                lo[k] = v if v > l_min else l_min
                                last[k] = t
                                frees += 1
                                add(k)
                    pi, pj, pax, pay = i, j, ax, ay
            for (xa, ya), (xb, yb) in col.free_segments:      # R4.2 segment form (tests, older producers)
                ax, ay = bx + c * xa - s * ya, by + s * xa + c * ya
                ex, ey = bx + c * xb - s * yb, by + s * xb + c * yb
                i0, j0, i1, j1 = floor(ax * inv), floor(ay * inv), floor(ex * inv), floor(ey * inv)
                if abs(i1 - i0) <= 1 and abs(j1 - j0) <= 1:            # short segment: its end cells cover it
                    cells = ((i0, j0),) if (i0, j0) == (i1, j1) else ((i0, j0), (i1, j1))
                else:
                    m = int(math.hypot(ex - ax, ey - ay) / step) + 1
                    cells = [(floor((ax + q / m * (ex - ax)) * inv), floor((ay + q / m * (ey - ay)) * inv)) for q in range(m + 1)]
                for i, j in cells:
                    if 0 <= i < n and 0 <= j < n:
                        k = j * n + i
                        if mark[k] != cyc and k not in prot:
                            mark[k] = cyc
                            v = lo[k] + l_miss
                            lo[k] = v if v > l_min else l_min
                            last[k] = t
                            frees += 1
                            add(k)
        return {"cells_hit": hits, "cells_free_updated": frees}

    def add_hits(self, points: list, stamp_s: float, l_add: float | None = None, source: int = SRC_YOLO_POINTS) -> int:
        """R4.4 small-rock rule: depth points (odom x, y, z) that YOLO marked as a rock, below the generic obstacle height
        (``source`` SRC_YOLO_POINTS), or a confirmed small rock's footprint (SRC_YOLO_FOOTPRINT, with ``l_add``).
        At most one hit per cell per call; never a miss (YOLO cannot clear space)."""
        self.cycle_id += 1
        cyc, n, inv, floor = self.cycle_id, self.n, self.inv, math.floor
        lo, mark, l_hit, l_max = self.logodds, self._mark, (self.cfg.l_hit if l_add is None else l_add), self.cfg.l_max
        hits = 0
        for x, y, _z in points:
            i, j = floor((x - self.origin_x) * inv), floor((y - self.origin_y) * inv)
            if 0 <= i < n and 0 <= j < n:
                k = j * n + i
                if mark[k] != cyc:
                    mark[k] = cyc
                    v = lo[k] + l_hit
                    lo[k] = v if v < l_max else l_max
                    self.last_obs[k] = stamp_s
                    self.neg[k] = 0
                    self.src[k] |= source
                    self.touched.append(k)
                    hits += 1
        return hits

    # ---- read-out --------------------------------------------------------------------------------------------------
    def state_at(self, k: int) -> int:
        v = self.logodds[k]
        if v > self.occ_th:
            return NEGATIVE if self.neg[k] else OCCUPIED
        if v < self.free_th and self.last_obs[k] != NEVER:
            return FREE
        return UNKNOWN

    def state_layer(self) -> bytes:
        lo, neg, last = self.logodds, self.neg, self.last_obs
        occ, fr = self.occ_th, self.free_th
        return bytes((NEGATIVE if neg[k] else OCCUPIED) if lo[k] > occ else (FREE if lo[k] < fr and last[k] != NEVER else UNKNOWN)
                     for k in range(len(lo)))

    def provenance_layer(self) -> bytes:
        """Exported provenance: the SRC_* flags, plus SRC_PROTECTED on cells currently withheld from FREE evidence."""
        out = bytearray(self.src.tobytes())
        for k in self.protected:
            out[k] |= SRC_PROTECTED
        return bytes(out)

    def occupied_by_source(self) -> dict:
        """OCCUPIED / NEGATIVE cells split by provenance: depth evidence alone, YOLO-derived evidence alone, or both."""
        lo, src, occ = self.logodds, self.src, self.occ_th
        yolo = SRC_YOLO_POINTS | SRC_YOLO_FOOTPRINT
        out = {"depth_only": 0, "yolo_only": 0, "depth_and_yolo": 0, "protected_occupied": 0}
        for k in range(len(lo)):
            if lo[k] > occ:
                d, y = src[k] & SRC_DEPTH, src[k] & yolo
                out["depth_and_yolo" if d and y else "depth_only" if d else "yolo_only"] += 1
        out["protected_occupied"] = sum(1 for k in self.protected if lo[k] > occ)
        return out

    def counts(self) -> dict:
        st = self.state_layer()
        return {"unknown": st.count(UNKNOWN), "free": st.count(FREE), "occupied": st.count(OCCUPIED), "negative": st.count(NEGATIVE)}

    def nbytes(self) -> int:
        return sum(a.itemsize * len(a) for a in (self.logodds, self.last_obs, self.neg, self.src, self._mark))
