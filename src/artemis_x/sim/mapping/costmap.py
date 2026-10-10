"""Navigation cost layer over the R4.2 occupancy grid (R4.3, plan section 6.6). Stdlib only.

What the values mean - they are an INFLATED NAVIGATION COST for the rover CENTRE, not raw occupancy:
      0      FREE cell, farther than collision radius + safety margin + uncertainty margin from any hazard
      1-252  FREE cell inside the graded safety band (252 * exp(-k (d - r_inflated)); higher = closer)
      253    INSCRIBED: a rover centre here could touch a hazard (d <= collision radius + uncertainty margin)
      254    LETHAL: the cell itself is OCCUPIED or a NEGATIVE hazard (crater bowl)
      255    UNKNOWN: never observed or inconclusive - NEVER traversable and never lowered by inflation
Only FREE cells receive inflation; UNKNOWN stays 255 and LETHAL stays 254 whatever surrounds them.

Footprint (Koala.proto R2025a, measured): circumscribed radius 0.210 m about the rover origin (wheel corners); the
collision radius used here is 0.25 m (+0.04 m allowance, plan decision D3) and the safety margin 0.15 m.

Distances: obstacle and query cells are 0.10 m squares, so the cell-centre distance is reduced by half a cell diagonal
(the hazard may lie anywhere in its cell) - conservative by up to 7 cm.

Uncertainty: each hazard cell is inflated by its own 2 sigma_rel (uncertainty.py: odometry drift since the cell was last
observed, with the heading lever arm to the rover), rounded up to 5 cm and capped at 0.20 m. Kernels are precomputed per
level. Updates are incremental: only 1 m tiles whose cells changed state, or whose hazards changed uncertainty level, are
recomputed. A window recentre shifts the cost arrays by the same whole-cell offset as the grid and recomputes only the
edge tiles (cells that entered are UNKNOWN; hazards that left no longer inflate their neighbours).
"""

from __future__ import annotations

import array
import math

from .config import MappingConfig
from .grid import FREE, NEGATIVE, OCCUPIED, UNKNOWN, LogOddsGrid
from .uncertainty import OdometerHistory, UncertaintyModel

COST_FREE, COST_INSCRIBED, COST_LETHAL, COST_UNKNOWN = 0, 253, 254, 255
_BASE = {UNKNOWN: COST_UNKNOWN, FREE: COST_FREE, OCCUPIED: COST_LETHAL, NEGATIVE: COST_LETHAL}


def build_kernel(cfg: MappingConfig, margin_m: float) -> list:
    """[(di, dj, cost)] around a hazard cell for one uncertainty margin (cost > 0 only, centre excluded)."""
    res = cfg.resolution_m
    r_ins = cfg.collision_radius_m + margin_m
    r_out = r_ins + cfg.safety_margin_m
    half_diag = res * math.sqrt(2) / 2
    n = int(math.ceil((r_out + half_diag) / res))
    out = []
    for dj in range(-n, n + 1):
        for di in range(-n, n + 1):
            if di == 0 and dj == 0:
                continue
            d = max(0.0, math.hypot(di, dj) * res - half_diag)
            if d <= r_ins:
                out.append((di, dj, COST_INSCRIBED))
            elif d <= r_out:
                out.append((di, dj, max(1, min(252, int(252 * math.exp(-cfg.cost_decay_per_m * (d - r_ins)))))))
    return out


def disc_offsets(radius_m: float, res: float) -> list:
    """Cells whose centre lies within ``radius_m`` of a point's cell centre (footprint queries)."""
    n = int(math.ceil(radius_m / res))
    return [(di, dj) for dj in range(-n, n + 1) for di in range(-n, n + 1) if math.hypot(di, dj) * res <= radius_m + 1e-9]


class CostLayer:
    def __init__(self, cfg: MappingConfig, grid: LogOddsGrid, model: UncertaintyModel, history: OdometerHistory):
        self.cfg, self.grid, self.model, self.history = cfg, grid, model, history
        self.n = grid.n
        N = self.n * self.n
        self.T = cfg.cost_tile_cells
        self.nt = (self.n + self.T - 1) // self.T
        self.levels = [i * cfg.uncertainty_margin_step_m for i in range(model.n_levels())]
        self.kernels = [build_kernel(cfg, m) for m in self.levels]
        self.cost = array.array("B", [COST_UNKNOWN]) * N
        self.state = array.array("B", bytes(N))                # state cache (UNKNOWN = 0)
        self.level = array.array("b", [-1]) * N                # uncertainty level of hazard cells, -1 elsewhere
        self.tile_hazards: dict = {}                            # tile -> set of hazard cell indices
        self._recenters = grid.recenters
        self._last_pass = None                                  # (D, A, x, y) at the last uncertainty-level pass
        self.stats = {"updates": 0, "full_rebuilds": 0, "tiles_recomputed": 0}
        self.rebuild(0.0, (0.0, 0.0), (0.0, 0.0))

    # ---- helpers ---------------------------------------------------------------------------------------------------
    def _tile(self, k: int) -> int:
        j, i = divmod(k, self.n)
        return (j // self.T) * self.nt + i // self.T

    def _reach(self, t: int) -> set:
        """A tile and its 8 neighbours: every tile a hazard inside ``t`` can inflate (kernel radius < one tile)."""
        tj, ti = divmod(t, self.nt)
        return {a * self.nt + b for a in (tj - 1, tj, tj + 1) for b in (ti - 1, ti, ti + 1) if 0 <= a < self.nt and 0 <= b < self.nt}

    def _sigma_rel(self, k: int, now_DA: tuple, rover_xy: tuple, cache: dict) -> float:
        t_obs = self.grid.last_obs[k]
        DA = cache.get(t_obs)
        if DA is None:
            DA = cache[t_obs] = self.history.at(t_obs)
        cx, cy = self.grid.cell_centre(k)
        rho = math.hypot(cx - rover_xy[0], cy - rover_xy[1])
        return self.model.sigma(max(0.0, now_DA[0] - DA[0]), max(0.0, now_DA[1] - DA[1]), rho)

    def _level_of(self, k: int, now_DA: tuple, rover_xy: tuple, cache: dict) -> int:
        """Inflation level of a hazard cell, computed with headroom (distance, turn and lever arm each increased by what the
        rover may still move before the next level pass) - between passes a level can only be too large, never too small."""
        hd, ha = self.cfg.cost_level_headroom_m, self.cfg.cost_level_headroom_rad
        t_obs = self.grid.last_obs[k]
        DA = cache.get(t_obs)
        if DA is None:
            DA = cache[t_obs] = self.history.at(t_obs)
        cx, cy = self.grid.cell_centre(k)
        rho = math.hypot(cx - rover_xy[0], cy - rover_xy[1]) + hd
        return self.model.level(self.model.sigma(max(0.0, now_DA[0] - DA[0]) + hd, max(0.0, now_DA[1] - DA[1]) + ha, rho))

    def sigma_rel(self, k: int, now_DA: tuple, rover_xy: tuple) -> float:
        return self._sigma_rel(k, now_DA, rover_xy, {})

    def _set_hazard(self, k: int, is_hazard: bool, lvl: int) -> None:
        t = self._tile(k)
        if is_hazard:
            self.tile_hazards.setdefault(t, set()).add(k)
            self.level[k] = lvl
        else:
            s = self.tile_hazards.get(t)
            if s is not None:
                s.discard(k)
            self.level[k] = -1

    # ---- updates ---------------------------------------------------------------------------------------------------
    def rebuild(self, now_t: float, now_DA: tuple, rover_xy: tuple) -> None:
        """Everything from scratch (start, window recentre, tests)."""
        g = self.grid
        N = self.n * self.n
        self.tile_hazards = {}
        self.level = array.array("b", [-1]) * N
        cache: dict = {}
        for k in range(N):
            s = g.state_at(k)
            self.state[k] = s
            if s == OCCUPIED or s == NEGATIVE:
                self._set_hazard(k, True, self._level_of(k, now_DA, rover_xy, cache))
        g.touched = []
        self._recompute(set(range(self.nt * self.nt)))
        self._recenters = g.recenters
        self.stats["full_rebuilds"] += 1

    def update(self, now_t: float, now_DA: tuple, rover_xy: tuple) -> dict:
        """Incremental update after a grid integration: cells touched in that integration, plus uncertainty-level changes."""
        g = self.grid
        dirty = set()
        shifted = False
        if g.recenters == self._recenters + 1 and getattr(g, "last_shift", None) is not None:
            dirty = self._shift(*g.last_shift)
            self._recenters = g.recenters
            shifted = True
        elif g.recenters != self._recenters:
            self.rebuild(now_t, now_DA, rover_xy)
            return {"full": True, "shift": False, "tiles": self.nt * self.nt}
        cache: dict = {}
        state_at, state = g.state_at, self.state
        touched, g.touched = g.touched, []
        for k in touched:
            s = state_at(k)
            if s != state[k]:
                was = state[k] == OCCUPIED or state[k] == NEGATIVE
                state[k] = s
                hz = s == OCCUPIED or s == NEGATIVE
                if hz or was:                                 # a hazard appeared / vanished: its whole reach is dirty
                    self._set_hazard(k, hz, self._level_of(k, now_DA, rover_xy, cache) if hz else -1)
                    dirty |= self._reach(self._tile(k))
                else:
                    dirty.add(self._tile(k))                  # only the cell's own base cost changed
        lp = self._last_pass
        moved = (lp is None or now_DA[0] - lp[0] >= self.cfg.cost_level_headroom_m or now_DA[1] - lp[1] >= self.cfg.cost_level_headroom_rad
                 or math.hypot(rover_xy[0] - lp[2], rover_xy[1] - lp[3]) >= self.cfg.cost_level_headroom_m)
        if moved:
            self._last_pass = (now_DA[0], now_DA[1], rover_xy[0], rover_xy[1])
        for t, cells in (self.tile_hazards.items() if moved else ()):   # uncertainty grows as the rover drives / turns
            for k in cells:
                lv = self._level_of(k, now_DA, rover_xy, cache)
                if lv != self.level[k]:
                    self.level[k] = lv
                    dirty |= self._reach(t)
        if dirty:
            self._recompute(dirty)
        self.stats["updates"] += 1
        return {"full": False, "shift": shifted, "tiles": len(dirty)}

    def _shift(self, di: int, dj: int) -> set:
        """Follow a grid recentre by (di, dj) cells (new cell (i, j) = old cell (i + di, j + dj)). Returns the dirty tiles:
        every tile within one tile of an edge the shift moved (entering cells, and cells that lost a departed hazard)."""
        n, N = self.n, self.n * self.n
        cost, state, level = array.array("B", [COST_UNKNOWN]) * N, array.array("B", bytes(N)), array.array("b", [-1]) * N
        i0, i1 = max(0, -di), min(n, n - di)
        for j in range(max(0, -dj), min(n, n - dj)):
            if i1 <= i0:
                break
            a, b = j * n + i0, (j + dj) * n + i0 + di
            cost[a:a + i1 - i0] = self.cost[b:b + i1 - i0]
            state[a:a + i1 - i0] = self.state[b:b + i1 - i0]
            level[a:a + i1 - i0] = self.level[b:b + i1 - i0]
        self.cost, self.state, self.level = cost, state, level
        self.tile_hazards = {}
        for k in range(N):
            if level[k] >= 0:
                self.tile_hazards.setdefault(self._tile(k), set()).add(k)
        nt, band = self.nt, 1 + (max(abs(di), abs(dj)) + self.T - 1) // self.T
        dirty = set()
        for tj in range(nt):
            for ti in range(nt):
                if (di and (ti < band or ti >= nt - band)) or (dj and (tj < band or tj >= nt - band)):
                    dirty.add(tj * nt + ti)
        self.stats["shifts"] = self.stats.get("shifts", 0) + 1
        return dirty

    def _recompute(self, dirty: set) -> None:
        n, T, nt = self.n, self.T, self.nt
        cost, state = self.cost, self.state
        for t in dirty:                                       # 1. base costs from the cell states
            tj, ti = divmod(t, nt)
            for j in range(tj * T, min(n, tj * T + T)):
                row = j * n
                for i in range(ti * T, min(n, ti * T + T)):
                    cost[row + i] = _BASE[state[row + i]]
        sources = set()                                       # 2. hazards that can reach a dirty tile (kernel < 1 tile)
        for t in dirty:
            tj, ti = divmod(t, nt)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    a, b = tj + dj, ti + di
                    if 0 <= a < nt and 0 <= b < nt:
                        s = self.tile_hazards.get(a * nt + b)
                        if s:
                            sources |= s
        level, kernels = self.level, self.kernels
        for k in sources:
            j0, i0 = divmod(k, n)
            for di, dj, c in kernels[level[k]]:
                i, j = i0 + di, j0 + dj
                if 0 <= i < n and 0 <= j < n and (j // T) * nt + i // T in dirty:
                    q = j * n + i
                    if cost[q] < c:                           # FREE cells only: 254 / 255 are never below a kernel value
                        cost[q] = c
        self.stats["tiles_recomputed"] += len(dirty)

    # ---- queries ---------------------------------------------------------------------------------------------------
    def cost_at(self, x: float, y: float) -> int:
        k = self.grid.index(x, y)
        return COST_UNKNOWN if k < 0 else self.cost[k]
