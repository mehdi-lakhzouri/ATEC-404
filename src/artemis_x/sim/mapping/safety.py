"""Navigation-facing safety interface (R4.3, Feature E; the plan's ``MapProvider``, section 12). Read-only: it answers
questions about the latest map and health; it never commands motion. A*, path following and replanning belong to R5.

Rules for consumers (plan section 12): never enter cost >= 253; UNKNOWN (255) is not traversable; any STOP reason stops
motion; inputs older than 0.5 s are invalid.
"""

from __future__ import annotations

import math

from .config import MappingConfig
from .costmap import COST_INSCRIBED, COST_UNKNOWN, CostLayer, disc_offsets
from .frames import Pose2D
from .grid import FREE, NEVER, UNKNOWN, LogOddsGrid
from .health import HealthMonitor, MapHealth


class SafetyInterface:
    def __init__(self, cfg: MappingConfig, grid: LogOddsGrid, costmap: CostLayer, health: HealthMonitor, estimator):
        self.cfg, self.grid, self.costmap, self._health, self.estimator = cfg, grid, costmap, health, estimator
        self.footprint = disc_offsets(cfg.footprint_radius_measured_m, cfg.resolution_m)

    # ---- plan section 12 API ---------------------------------------------------------------------------------------
    def health(self) -> MapHealth | None:
        return self._health.latest

    def get_stop_reasons(self) -> list:
        return self._health.stop_reasons()

    def cost_at(self, x: float, y: float) -> int:
        return self.costmap.cost_at(x, y)

    def _footprint_cells(self, x: float, y: float) -> list:
        g = self.grid
        k0 = g.index(x, y)
        if k0 < 0:
            return [-1]
        j0, i0 = divmod(k0, g.n)
        return [(j0 + dj) * g.n + i0 + di if 0 <= i0 + di < g.n and 0 <= j0 + dj < g.n else -1 for di, dj in self.footprint]

    def footprint_cost(self, pose: Pose2D) -> int:
        """Collision cost of the rover at ``pose``: the inflated cost of its centre cell (inflation already encodes the
        footprint), but UNKNOWN (255) if any cell under the measured footprint is unknown or outside the window."""
        cells = self._footprint_cells(pose.x, pose.y)
        if any(k < 0 or self.costmap.cost[k] == COST_UNKNOWN for k in cells):
            return COST_UNKNOWN
        return self.costmap.cost_at(pose.x, pose.y)

    def corridor_ahead(self, pose: Pose2D, length_m: float | None = None) -> list:
        L = self.cfg.corridor_length_m if length_m is None else length_m
        n = int(round(L / self.cfg.corridor_step_m))
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return [Pose2D(pose.x + c * q * self.cfg.corridor_step_m, pose.y + s * q * self.cfg.corridor_step_m, pose.yaw, pose.stamp_s)
                for q in range(n + 1)]

    def corridor_clear(self, path: list, max_age_s: float | None = None, now_t: float | None = None) -> tuple:
        """(ok, reasons, details) for a list of poses. Reasons: obstacle_in_corridor (centre cost >= INSCRIBED),
        unobserved_corridor (an UNKNOWN cell under the footprint - includes the camera blind zone if never seen before),
        stale_corridor (FREE cells observed longer ago than max_age_s, or whose relative position uncertainty exceeds
        corridor_max_sigma_m)."""
        cfg, g, cm = self.cfg, self.grid, self.costmap
        max_age = cfg.corridor_max_age_s if max_age_s is None else max_age_s
        now = self.estimator.stamp_s if now_t is None else now_t
        odo = self.estimator.odo
        now_DA = (odo.dist, odo.turned)
        rx, ry = odo.x, odo.y
        first_obstacle = None
        cells = set()
        unknown = stale = 0
        for q, p in enumerate(path):
            k = g.index(p.x, p.y)
            if k >= 0 and COST_INSCRIBED <= cm.cost[k] < COST_UNKNOWN and first_obstacle is None:
                first_obstacle = round(q * cfg.corridor_step_m, 3)
            cells.update(self._footprint_cells(p.x, p.y))
        cache: dict = {}
        worst_age = worst_sigma = 0.0
        for k in cells:
            if k < 0 or cm.state[k] == UNKNOWN:
                unknown += 1
                continue
            if cm.state[k] == FREE:
                t_obs = g.last_obs[k]
                age = math.inf if t_obs == NEVER else now - t_obs
                sig = cm._sigma_rel(k, now_DA, (rx, ry), cache)
                worst_age, worst_sigma = max(worst_age, age), max(worst_sigma, sig)
                if age > max_age or sig > cfg.corridor_max_sigma_m:
                    stale += 1
        reasons = []
        if first_obstacle is not None:
            reasons.append("obstacle_in_corridor")
        if unknown:
            reasons.append("unobserved_corridor")
        if stale:
            reasons.append("stale_corridor")
        return (not reasons, reasons, {"cells": len(cells), "unknown_cells": unknown, "stale_cells": stale,
                                       "first_obstacle_m": first_obstacle, "worst_free_age_s": round(min(worst_age, 1e6), 3),
                                       "worst_free_sigma_m": round(worst_sigma, 4)})
