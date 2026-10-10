"""Estimated rover pose: wheel-encoder odometry with a growing uncertainty, and a timestamped history for
"pose at capture time" lookups (camera frames and YOLO results arrive later than the pose they were taken at)."""

from __future__ import annotations

import collections
import math

from .. import drive
from .config import MappingConfig
from .frames import Pose2D, wrap


class PoseEstimator:
    """Wraps ``drive.Odometry`` (the same integrator the scripted tour drives on). ``sigma_xy`` / ``sigma_yaw`` are the
    ABSOLUTE (start-relative) 1-sigma bounds of the R4.3 uncertainty model; local safety decisions use the relative
    model in ``uncertainty.py`` instead (see there for why). No ground truth, no correction."""

    def __init__(self, cfg: MappingConfig | None = None):
        from .uncertainty import UncertaintyModel
        self.cfg = cfg or MappingConfig()
        self.model = UncertaintyModel(self.cfg)
        self.odo = drive.Odometry()
        self.sigma_xy = 0.0
        self.sigma_yaw = 0.0
        self.stamp_s = 0.0

    def update(self, enc_left: float | None, enc_right: float | None, stamp_s: float) -> Pose2D:
        if enc_left is not None and enc_right is not None:
            self.odo.update(enc_left, enc_right)
        self.sigma_xy = self.model.absolute(self.odo.dist, self.odo.turned)
        self.sigma_yaw = self.model.heading_sigma(self.odo.dist, self.odo.turned)
        self.stamp_s = stamp_s
        return self.pose()

    def pose(self) -> Pose2D:
        return Pose2D(self.odo.x, self.odo.y, wrap(self.odo.yaw), self.stamp_s, self.sigma_xy, self.sigma_yaw)


class PoseBuffer:
    """Bounded, time-ordered pose history with linear interpolation (yaw interpolated along the short arc)."""

    def __init__(self, horizon_s: float = 2.0):
        self.horizon_s = horizon_s
        self._q: collections.deque = collections.deque()

    def add(self, p: Pose2D) -> None:
        if self._q and p.stamp_s <= self._q[-1].stamp_s:
            if p.stamp_s == self._q[-1].stamp_s:
                self._q[-1] = p
            return                                     # out-of-order samples are ignored (simulation time is monotonic)
        self._q.append(p)
        while self._q and self._q[-1].stamp_s - self._q[0].stamp_s > self.horizon_s:
            self._q.popleft()

    def latest(self) -> Pose2D | None:
        return self._q[-1] if self._q else None

    def at(self, t: float, tol_s: float = 1e-6) -> Pose2D | None:
        """Pose at time t, or None if t is outside the buffered window (too old, or in the future)."""
        q = self._q
        if not q or t < q[0].stamp_s - tol_s or t > q[-1].stamp_s + tol_s:
            return None
        lo, hi = 0, len(q) - 1
        while hi - lo > 1:                             # binary search on the deque (O(log n) indexing is fine for ~70 items)
            mid = (lo + hi) // 2
            if q[mid].stamp_s <= t:
                lo = mid
            else:
                hi = mid
        a, b = q[lo], q[hi]
        if b.stamp_s <= a.stamp_s or t <= a.stamp_s:
            return a if abs(t - a.stamp_s) <= abs(t - b.stamp_s) else b
        if t >= b.stamp_s:
            return b
        k = (t - a.stamp_s) / (b.stamp_s - a.stamp_s)
        return Pose2D(a.x + k * (b.x - a.x), a.y + k * (b.y - a.y), wrap(a.yaw + k * wrap(b.yaw - a.yaw)), t,
                      a.sigma_xy + k * (b.sigma_xy - a.sigma_xy), a.sigma_yaw + k * (b.sigma_yaw - a.sigma_yaw), a.source)

    def __len__(self) -> int:
        return len(self._q)


def sample_time(t: float, period_s: float, eps: float = 1e-9) -> float:
    """Simulation time at which a Webots sensor with sampling period ``period_s`` (enabled at t = 0) last refreshed:
    the image read at time t was rendered at the largest multiple of the period <= t."""
    return math.floor(t / period_s + eps) * period_s
