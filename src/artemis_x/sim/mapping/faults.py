"""Deterministic fault injection for R4 evaluation (R4.3, Feature D). OFF unless explicitly enabled.

Enabled only by the environment variable ARTEMIS_FAULTS holding a JSON object; without it ``from_env()`` returns None and
no code path changes. The active specification is written to the run's mapping_config.json and every telemetry line
lists the faults active in that cycle, so a fault run can never be mistaken for a normal one.

    {"odometry":      {"scale": 1.03, "yaw_scale": 1.03, "yaw_drift_per_m": 0.0},       # whole run
     "depth_dropout": {"t0": 40, "t1": 60, "fraction": 0.3, "seed": 7},                  # sampled pixels -> NaN
     "depth_freeze":  {"t0": 80, "t1": 82},                                               # sensor repeats the last frame
     "depth_invalid": {"t0": 120, "t1": 125, "value": "inf"},                             # lower half of the image
     "cycle_overrun": {"t0": 160, "t1": 165, "ms": 60}}                                   # extra wall-clock work

Depth faults act on the frame the mapper captured (the same frame object the inference link submits), on the pixels the
mapper samples (dropout) or on the whole lower half of the image (invalid). Inference faults (server absent, killed, slow)
are produced by the evaluation runner, not here.
"""

from __future__ import annotations

import array
import json
import math
import os
import random
import time


class FaultInjector:
    def __init__(self, spec: dict):
        self.spec = spec
        self._last_depth = None
        self.counts = {k: 0 for k in spec}

    @classmethod
    def from_env(cls, var: str = "ARTEMIS_FAULTS"):
        raw = os.environ.get(var)
        if not raw:
            return None
        spec = json.loads(raw)
        if not isinstance(spec, dict) or not spec:
            return None
        return cls(spec)

    def _on(self, name: str, t: float) -> bool:
        f = self.spec.get(name)
        return f is not None and f.get("t0", -math.inf) <= t < f.get("t1", math.inf)

    def active(self, t: float) -> list:
        return [k for k in self.spec if k != "odometry" and self._on(k, t)] + (["odometry"] if "odometry" in self.spec else [])

    def configure_odometry(self, odo) -> None:
        o = self.spec.get("odometry")
        if not o:
            return
        odo.scale = float(o.get("scale", 1.0))
        odo.yaw_scale = odo.yaw_scale * float(o.get("yaw_scale", 1.0))
        odo.yaw_drift_per_m = float(o.get("yaw_drift_per_m", 0.0))

    def apply_depth(self, frame, t: float, sample_indices: list, width: int, height: int) -> list:
        """Mutate ``frame.depth`` for the depth faults active at t; returns the names applied."""
        applied = []
        if frame.depth is None:
            return applied
        if self._on("depth_freeze", t) and self._last_depth is not None:
            frame.depth = array.array("f", self._last_depth)       # the sensor keeps delivering the old image
            applied.append("depth_freeze")
        else:
            self._last_depth = bytes(frame.depth) if isinstance(frame.depth, array.array) else array.array("f", frame.depth).tobytes()
        if self._on("depth_dropout", t):
            f = self.spec["depth_dropout"]
            rng = random.Random(int(f.get("seed", 0)) * 1_000_003 + frame.index)
            nan = float("nan")
            for idx in rng.sample(sample_indices, int(len(sample_indices) * float(f.get("fraction", 0.3)))):
                frame.depth[idx] = nan
            applied.append("depth_dropout")
        if self._on("depth_invalid", t):
            v = float(self.spec["depth_invalid"].get("value", "inf"))
            start = (height // 2) * width
            if isinstance(frame.depth, array.array):
                frame.depth[start:] = array.array("f", [v]) * (len(frame.depth) - start)
            else:
                for i in range(start, len(frame.depth)):
                    frame.depth[i] = v
            applied.append("depth_invalid")
        for a in applied:
            self.counts[a] += 1
        return applied

    def overrun(self, t: float) -> bool:
        if self._on("cycle_overrun", t):
            time.sleep(float(self.spec["cycle_overrun"].get("ms", 60)) / 1000.0)
            self.counts["cycle_overrun"] += 1
            return True
        return False
