"""Mapping / safety health (R4.3, Feature B; plan section 10). ADVISORY ONLY: nothing here commands the motors - the
scripted tour keeps running whatever the state; R5 will be the first consumer allowed to act on it.

States: OK (all inputs healthy), DEGRADED (mapping continues with reduced confidence), STOP (observations do not justify
moving, or a hazard is too close). The state is the worst severity among the ACTIVE reasons.

Hysteresis: a reason becomes active on the first cycle its condition holds (fast detection: one mapping cycle, 0.256 s)
and is cleared only after ``clear_cycles`` consecutive cycles without it (no flapping on a single noisy cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import MappingConfig

OK, DEGRADED, STOP = "OK", "DEGRADED", "STOP"
_RANK = {OK: 0, DEGRADED: 1, STOP: 2}

REASONS = {
    # depth / mapping inputs
    "depth_stale": STOP,             # no accepted depth frame for depth_stale_s
    "depth_frozen": STOP,            # byte-identical depth frame while the rover moved: frozen sensor
    "depth_invalid": STOP,           # ground-region valid fraction below depth_low_valid_stop
    "depth_degraded": DEGRADED,      # ground-region valid fraction below depth_low_valid_degraded
    "no_pose": STOP,                 # no pose estimate for the frame's sample time
    # safety around the rover
    "obstacle_in_footprint": STOP,   # inflated cost >= INSCRIBED at the rover's own position
    "obstacle_in_corridor": STOP,    # ... somewhere on the prospective corridor ahead
    "unobserved_corridor": STOP,     # UNKNOWN cells under the rover footprint along the corridor (incl. the blind zone)
    "stale_corridor": STOP,          # FREE evidence on the corridor older than corridor_max_age_s or too uncertain
    # optional / degraded inputs
    "inference_disabled": DEGRADED,  # YOLO not configured: depth-only mapping (plan decision D9)
    "inference_unavailable": DEGRADED,
    "inference_stale": DEGRADED,
    "cycle_overrun": DEGRADED,       # mapping work exceeded cycle_overrun_ms (wall clock)
}
# The absolute (start-relative) position uncertainty is reported in every MapHealth (``pose_sigma_xy`` and the
# ``global_pose_uncertain`` measurement) but is NOT a reason: with odometry only it grows without bound (32 m envelope
# after the 66 m tour, turns charged worst-case) while the LOCAL map stays consistent. It concerns R5's global goals,
# which need absolute localisation first; local safety uses sigma_rel (corridor staleness, inflation).


@dataclass(frozen=True)
class MapHealth:
    """Plan section 7 contract (unchanged fields) + timestamp and the measurements behind each decision."""
    state: str
    reasons: tuple
    depth_age_s: float
    inference_age_s: float | None
    inference_connected: bool
    depth_valid_fraction: float
    pose_sigma_xy: float
    stamp_s: float = 0.0
    measurements: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"state": self.state, "reasons": list(self.reasons), "stamp_s": round(self.stamp_s, 4),
                "depth_age_s": round(self.depth_age_s, 3), "inference_age_s": None if self.inference_age_s is None else round(self.inference_age_s, 3),
                "inference_connected": self.inference_connected, "depth_valid_fraction": round(self.depth_valid_fraction, 3),
                "pose_sigma_xy": round(self.pose_sigma_xy, 4), "measurements": self.measurements}


class HealthMonitor:
    def __init__(self, cfg: MappingConfig | None = None):
        self.cfg = cfg or MappingConfig()
        self.active: dict = {}                               # reason -> sim time it became active
        self.clean: dict = {}                                # reason -> consecutive cycles without the condition
        self.state = STOP                                    # nothing observed yet: not justified to move
        self.transitions: list = []                          # (stamp, from, to, reasons)
        self.reason_events: list = []                        # (stamp, reason, "raised" | "cleared")
        self.time_in_state = {OK: 0.0, DEGRADED: 0.0, STOP: 0.0}
        self._last_t = None
        self.latest: MapHealth | None = None

    def evaluate(self, t: float, conditions: dict, info: dict) -> MapHealth:
        """``conditions``: reason -> bool (condition holds this cycle). ``info``: the measurements for MapHealth."""
        for r in REASONS:
            hold = bool(conditions.get(r, False))
            if hold:
                self.clean[r] = 0
                if r not in self.active:
                    self.active[r] = t
                    self.reason_events.append((round(t, 4), r, "raised"))
            elif r in self.active:
                self.clean[r] = self.clean.get(r, 0) + 1
                if self.clean[r] >= self.cfg.clear_cycles:
                    del self.active[r]
                    self.reason_events.append((round(t, 4), r, "cleared"))
        new = max((REASONS[r] for r in self.active), key=_RANK.get, default=OK)
        if self._last_t is not None:
            self.time_in_state[self.state] += max(0.0, t - self._last_t)
        self._last_t = t
        if new != self.state:
            self.transitions.append((round(t, 4), self.state, new, sorted(self.active)))
            self.state = new
        self.latest = MapHealth(new, tuple(sorted(self.active)), info.get("depth_age_s", 0.0), info.get("inference_age_s"),
                                bool(info.get("inference_connected", False)), info.get("depth_valid_fraction", 0.0),
                                info.get("pose_sigma_xy", 0.0), t, info.get("measurements", {}))
        return self.latest

    def stop_reasons(self) -> list:
        return sorted(r for r in self.active if REASONS[r] == STOP)

    def summary(self) -> dict:
        return {"final_state": self.state, "time_in_state_s": {k: round(v, 2) for k, v in self.time_in_state.items()},
                "transitions": len(self.transitions), "reason_events": len(self.reason_events)}
