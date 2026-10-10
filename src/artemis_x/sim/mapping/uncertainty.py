"""Odometry uncertainty for a LOCAL map (R4.3, Feature C).

The R4.1 model grew sigma_xy linearly with the distance since the START (2 %/m): 1.33 m after the 66 m course tour, which
would have tripped the planned 0.3 m STOP after ~15 m although the local map stayed exact. The flaw is the reference
point, not the coefficients: a local map only needs how wrong the rover can be about a cell RELATIVE to when that cell
was last observed.

Model (1 sigma, a design envelope - see MappingConfig):
    dD   = odometer distance driven since the cell was last observed
    dA   = absolute heading change accumulated since then (sum of |turns|)
    rho  = current distance from the rover to the cell (the lever arm of a heading error)
    dpsi = k_yaw_turn * dA + k_yaw_dist * dD
    sigma_rel = sqrt((k_scale * dD)^2 + (rho * dpsi)^2)

The absolute (start-relative) bound uses the same terms with dD = D, dA = A and rho = D. It is reported for R5's global
goals; local safety decisions use sigma_rel only.

Limits: errors are treated as accumulating in the worst direction (a +a then -a turn pair is charged 2a even though a
pure turn-scale error cancels); wheel slip, sinkage and terrain-dependent bias on real regolith are not modelled - they
are what the envelope is meant to cover, and must be re-measured on real terrain.
"""

from __future__ import annotations

import bisect
import math

from .config import MappingConfig


class OdometerHistory:
    """(sim time, cumulative distance D, cumulative |turn| A), recorded once per mapping cycle, for sigma_rel lookups."""

    def __init__(self, horizon: int = 4096):
        self.t: list = []
        self.D: list = []
        self.A: list = []
        self.horizon = horizon

    def add(self, t: float, D: float, A: float) -> None:
        if self.t and t <= self.t[-1]:
            self.D[-1], self.A[-1] = D, A
            return
        self.t.append(t); self.D.append(D); self.A.append(A)
        if len(self.t) > self.horizon:                      # ~17 min of cycles; older cells are long stale anyway
            del self.t[:1024], self.D[:1024], self.A[:1024]

    def at(self, t: float) -> tuple:
        """(D, A) at the latest record <= t. For t older than every record (or an empty history) the odometer at the
        start, (0, 0): the largest possible drift since the observation, i.e. the conservative answer."""
        k = bisect.bisect_right(self.t, t) - 1
        if k < 0:
            return 0.0, 0.0
        return self.D[k], self.A[k]


class UncertaintyModel:
    def __init__(self, cfg: MappingConfig | None = None):
        self.cfg = cfg or MappingConfig()

    def sigma(self, dD: float, dA: float, rho: float) -> float:
        c = self.cfg
        dpsi = c.odo_k_yaw_turn * dA + c.odo_k_yaw_dist * dD
        return math.sqrt((c.odo_k_scale * dD) ** 2 + (rho * dpsi) ** 2)

    def heading_sigma(self, dD: float, dA: float) -> float:
        return self.cfg.odo_k_yaw_turn * dA + self.cfg.odo_k_yaw_dist * dD

    def absolute(self, D: float, A: float) -> float:
        return self.sigma(D, A, D)

    def margin(self, sigma_rel: float) -> float:
        """2-sigma uncertainty margin added to obstacles, capped (plan section 6.6)."""
        return min(self.cfg.uncertainty_margin_cap_m, 2.0 * sigma_rel)

    def level(self, sigma_rel: float) -> int:
        """Index of the precomputed inflation kernel: margins rounded UP to the next 5 cm step (conservative)."""
        step = self.cfg.uncertainty_margin_step_m
        return min(int(math.ceil(self.margin(sigma_rel) / step - 1e-9)), self.n_levels() - 1)

    def n_levels(self) -> int:
        return int(round(self.cfg.uncertainty_margin_cap_m / self.cfg.uncertainty_margin_step_m)) + 1
