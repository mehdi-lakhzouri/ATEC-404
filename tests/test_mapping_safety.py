"""R4.3: footprint, inflation kernels, cost-layer semantics (UNKNOWN / LETHAL preserved, incremental == full rebuild,
recentre, reset), relative odometry uncertainty, health hysteresis, R5 safety interface, fault injection."""

import array
import json
import math
import os
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim import drive
from artemis_x.sim.mapping.config import MappingConfig
from artemis_x.sim.mapping.costmap import COST_INSCRIBED, COST_LETHAL, COST_UNKNOWN, CostLayer, build_kernel
from artemis_x.sim.mapping.depth_scan import Column, DepthObservation
from artemis_x.sim.mapping.faults import FaultInjector
from artemis_x.sim.mapping.frames import Pose2D
from artemis_x.sim.mapping.grid import FREE, NEGATIVE, OCCUPIED, UNKNOWN, LogOddsGrid
from artemis_x.sim.mapping.health import DEGRADED, OK, STOP, HealthMonitor
from artemis_x.sim.mapping.pose import PoseEstimator
from artemis_x.sim.mapping.safety import SafetyInterface
from artemis_x.sim.mapping.uncertainty import OdometerHistory, UncertaintyModel

CFG = MappingConfig()
P0 = Pose2D(0.0, 0.0, 0.0, 0.0)


def obs(points=(), segments=(), pose=P0, t=0.0):
    return DepthObservation(t, 0, pose, [Column(0.0, None, None, None, tuple(segments))], list(points), 1, 1)


def occupy(g, xy_kind, t0=0.0, n=4):                          # enough hits to override earlier free evidence
    for q in range(n):
        g.integrate(obs([(x, y, 0.2 if k == "positive" else -0.2, k) for x, y, k in xy_kind], t=t0 + q))


def free_box(g, x0, x1, y0, y1, t0=0.0, n=3):
    segs = [((x0, y), (x1, y)) for y in [y0 + 0.05 + 0.1 * j for j in range(int(round((y1 - y0) / 0.1)))]]
    for q in range(n):
        g.integrate(obs(segments=segs, t=t0 + q))


class Stack:
    """grid + history + cost layer + estimator, driven like the mapper does."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.g = LogOddsGrid(cfg)
        self.h = OdometerHistory()
        self.m = UncertaintyModel(cfg)
        self.c = CostLayer(cfg, self.g, self.m, self.h)
        self.est = PoseEstimator(cfg)
        self.hm = HealthMonitor(cfg)
        self.safety = SafetyInterface(cfg, self.g, self.c, self.hm, self.est)

    def step(self, t, D=0.0, A=0.0, x=0.0, y=0.0):
        self.h.add(t, D, A)
        self.est.odo.dist, self.est.odo.turned, self.est.odo.x, self.est.odo.y = D, A, x, y
        self.est.stamp_s = t
        return self.c.update(t, (D, A), (x, y))


class TestFootprint(unittest.TestCase):
    def test_measured_radius_from_koala_geometry(self):
        """Koala.proto R2025a: boundingObject boxes/cylinders + six wheels (r 0.0425, 0.04 wide) at x +-0.09/0, y +-0.1425."""
        pts = [(wx + dx, wy + dy) for wx in (0.09, 0.0, -0.09) for wy in (0.1425, -0.1425) for dx in (-0.0425, 0.0425) for dy in (-0.02, 0.02)]
        pts += [(0.175, 0.075), (-0.125, 0.14), (0.19, 0.0)]                    # body boxes, visual front
        pts += [(0.105 + 0.07 * math.cos(a), 0.07 + 0.07 * math.sin(a)) for a in [k * math.pi / 18 for k in range(36)]]
        r = max(math.hypot(x, y) for x, y in pts)
        self.assertAlmostEqual(r, 0.2097, places=3)
        self.assertGreaterEqual(CFG.footprint_radius_measured_m, r - 1e-3)
        self.assertGreaterEqual(CFG.collision_radius_m, CFG.footprint_radius_measured_m)


class TestKernel(unittest.TestCase):
    def test_radii_and_grading(self):
        hd = 0.1 * math.sqrt(2) / 2
        for lvl, margin in ((0, 0.0), (4, 0.2)):
            K = {(di, dj): c for di, dj, c in build_kernel(CFG, margin)}
            r_ins, r_out = CFG.collision_radius_m + margin, CFG.collision_radius_m + margin + CFG.safety_margin_m
            for (di, dj), c in K.items():
                d = max(0.0, math.hypot(di, dj) * 0.1 - hd)
                self.assertEqual(c == COST_INSCRIBED, d <= r_ins + 1e-12)
                self.assertLessEqual(d, r_out + 1e-12)
            far = max(math.hypot(di, dj) * 0.1 - hd for di, dj in K)
            self.assertGreater(far, r_out - 0.1)                               # the band reaches the outer radius
            graded = sorted(((math.hypot(di, dj), c) for (di, dj), c in K.items() if c < COST_INSCRIBED))
            self.assertTrue(all(a[1] >= b[1] for a, b in zip(graded, graded[1:]) if b[0] > a[0] + 1e-9))   # decreasing outward


class TestCostLayer(unittest.TestCase):
    def test_semantics_unknown_and_lethal_preserved(self):
        s = Stack()
        free_box(s.g, 0.0, 3.0, -1.0, 1.0)
        occupy(s.g, [(1.55, 0.05, "positive")], t0=5)
        s.step(6.0)
        g, c = s.g, s.c
        self.assertEqual(c.cost_at(1.55, 0.05), COST_LETHAL)
        self.assertEqual(c.cost_at(1.75, 0.05), COST_INSCRIBED)                 # 0.2 m from the hazard cell centre
        self.assertGreater(c.cost_at(1.95, 0.05), 0); self.assertLess(c.cost_at(1.95, 0.05), COST_INSCRIBED)
        self.assertEqual(c.cost_at(2.35, 0.05), 0)                              # beyond 0.25 + 0.15 (+ half cell)
        self.assertEqual(c.cost_at(-0.45, 0.05), COST_UNKNOWN)                 # never observed
        self.assertEqual(c.cost_at(1.55, 1.25), COST_UNKNOWN)                  # unknown right next to the inflation
        free_cells = sum(1 for k in range(len(c.cost)) if g.state_at(k) == FREE)
        self.assertEqual(sum(1 for k in range(len(c.cost)) if c.cost[k] < COST_INSCRIBED), free_cells - sum(
            1 for k in range(len(c.cost)) if g.state_at(k) == FREE and c.cost[k] >= COST_INSCRIBED))
        for k in range(len(c.cost)):                                            # inflation never lowers / never touches unknown
            if g.state_at(k) == UNKNOWN:
                self.assertEqual(c.cost[k], COST_UNKNOWN)
            if g.state_at(k) in (OCCUPIED, NEGATIVE):
                self.assertEqual(c.cost[k], COST_LETHAL)

    def test_negative_hazard_is_lethal_and_inflated(self):
        s = Stack()
        free_box(s.g, 0.0, 3.0, -1.0, 1.0)
        occupy(s.g, [(2.05, 0.05, "negative")], t0=5)
        s.step(6.0)
        self.assertEqual(s.g.state_at(s.g.index(2.05, 0.05)), NEGATIVE)
        self.assertEqual(s.c.cost_at(2.05, 0.05), COST_LETHAL)
        self.assertEqual(s.c.cost_at(1.85, 0.05), COST_INSCRIBED)

    def test_uncertainty_grows_inflation_and_is_capped(self):
        s = Stack()
        free_box(s.g, 0.0, 4.0, -1.5, 1.5)
        occupy(s.g, [(2.05, 0.05, "positive")], t0=5)
        s.step(6.0, D=0.0, A=0.0)
        k = s.g.index(2.05, 0.05)
        self.assertLessEqual(s.c.level[k], 1)                                 # just seen: at most the headroom's level
        r0 = sum(1 for q in s.c.cost if q == COST_INSCRIBED)
        s.step(20.0, D=5.0, A=0.0, x=0.0)                                       # drove 5 m since the hazard was seen
        hd, ha = CFG.cost_level_headroom_m, CFG.cost_level_headroom_rad      # levels carry the re-evaluation headroom
        self.assertEqual(s.c.level[k], s.m.level(s.m.sigma(5.0 + hd, ha, math.hypot(2.05, 0.05) + hd)))
        self.assertGreaterEqual(s.c.level[k], s.m.level(s.m.sigma(5.0, 0.0, 2.05)))   # never below the exact level
        self.assertGreater(s.c.level[k], 0)
        self.assertGreater(sum(1 for q in s.c.cost if q == COST_INSCRIBED), r0)
        s.step(40.0, D=500.0, A=50.0)
        self.assertEqual(s.c.level[k], s.m.n_levels() - 1)                      # capped at 0.20 m

    def test_incremental_equals_full_rebuild(self):
        rnd = random.Random(3)
        s = Stack()
        for t in range(1, 40):
            pts = [(rnd.uniform(-4, 4), rnd.uniform(-4, 4), 0.2, rnd.choice(["positive", "negative"])) for _ in range(6)]
            segs = [((rnd.uniform(-4, 4), y), (rnd.uniform(-4, 4), y)) for y in [rnd.uniform(-4, 4) for _ in range(8)]]
            s.g.integrate(obs(pts, segs, t=float(t)))
            s.step(float(t), D=0.3 * t, A=0.05 * t)
        inc = bytes(s.c.cost)
        s.c.rebuild(39.0, (0.3 * 39, 0.05 * 39), (0.0, 0.0))
        self.assertEqual(inc, bytes(s.c.cost))

    def test_recentre_shifts_and_equals_full_rebuild(self):
        rnd = random.Random(11)
        s = Stack()
        free_box(s.g, -5.0, 5.0, -5.0, 5.0)
        occupy(s.g, [(rnd.uniform(-5, 5), rnd.uniform(-5, 5), rnd.choice(["positive", "negative"])) for _ in range(40)], t0=5)
        s.step(6.0)
        before = {xy: s.c.cost_at(*xy) for xy in [(rnd.uniform(-2, 2), rnd.uniform(-2, 2)) for _ in range(50)]}
        for t, (x, y) in enumerate([(1.3, 0.2), (2.6, -1.1), (2.6, -2.4), (0.0, 0.0)]):
            s.g.recenter(x, y)
            st = s.step(7.0 + t, x=x, y=y)
            self.assertTrue(st["shift"]); self.assertFalse(st["full"])
            inc = bytes(s.c.cost)
            s.c.rebuild(7.0 + t, (0.0, 0.0), (x, y))
            self.assertEqual(inc, bytes(s.c.cost), (x, y))                      # shifted == rebuilt from scratch
        for xy, c in before.items():
            self.assertEqual(s.c.cost_at(*xy), c)                               # back at the start: same costs in odom

    def test_reset(self):
        s = Stack()
        self.assertTrue(all(q == COST_UNKNOWN for q in s.c.cost))                # a new map is entirely UNKNOWN


class TestUncertainty(unittest.TestCase):
    def test_relative_model(self):
        m = UncertaintyModel(CFG)
        self.assertEqual(m.sigma(0, 0, 3.0), 0.0)                              # just observed: no drift yet
        self.assertAlmostEqual(m.sigma(2.0, 0, 0.0), CFG.odo_k_scale * 2.0)
        self.assertAlmostEqual(m.sigma(0.0, math.pi / 2, 2.0), 2.0 * CFG.odo_k_yaw_turn * math.pi / 2)   # lever arm
        self.assertEqual(m.level(0.0), 0); self.assertEqual(m.level(0.0126), 1)                         # rounded UP
        self.assertEqual(m.level(10.0), m.n_levels() - 1)
        self.assertGreaterEqual(2 * CFG.odo_k_scale, 0.03); self.assertGreaterEqual(2 * CFG.odo_k_yaw_turn, 0.03)  # 2 sigma covers +-3 %

    def test_history_lookup(self):
        h = OdometerHistory()
        for t, D in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)):
            h.add(t, D, 0.0)
        self.assertEqual(h.at(1.5), (0.5, 0.0)); self.assertEqual(h.at(-5), (0.0, 0.0)); self.assertEqual(h.at(9), (1.0, 0.0))

    def test_default_odometry_unchanged(self):
        o = drive.Odometry()
        self.assertEqual((o.scale, o.yaw_scale, o.yaw_drift_per_m), (1.0, drive.YAW_SCALE, 0.0))
        o.update(0, 0); o.update(-2, 2)
        self.assertAlmostEqual(o.yaw, drive.YAW_SCALE * 4 * drive.WHEEL_RADIUS_M / drive.TRACK_M)


class TestHealth(unittest.TestCase):
    def test_hysteresis_and_severity(self):
        hm = HealthMonitor(CFG)
        self.assertEqual(hm.evaluate(0.0, {}, {}).state, OK)
        self.assertEqual(hm.evaluate(0.25, {"inference_unavailable": True}, {}).state, DEGRADED)   # raised at once
        h = hm.evaluate(0.5, {"inference_unavailable": True, "depth_frozen": True}, {})
        self.assertEqual(h.state, STOP); self.assertEqual(h.reasons, ("depth_frozen", "inference_unavailable"))
        self.assertEqual(hm.evaluate(0.75, {}, {}).state, STOP)                 # one clean cycle: not cleared yet
        self.assertEqual(hm.evaluate(1.0, {}, {}).state, OK)                    # second clean cycle: cleared
        self.assertEqual([tr[2] for tr in hm.transitions], [OK, DEGRADED, STOP, OK])   # starts in STOP: nothing observed yet
        hm.evaluate(1.25, {"cycle_overrun": True}, {}); hm.evaluate(1.5, {}, {}); hm.evaluate(1.75, {"cycle_overrun": True}, {})
        self.assertEqual(hm.state, DEGRADED)                                    # an intermittent fault does not flap
        self.assertEqual(hm.stop_reasons(), [])


class TestSafetyInterface(unittest.TestCase):
    def test_unknown_ahead_is_not_clear(self):
        s = Stack(); s.step(1.0)
        ok, reasons, d = s.safety.corridor_clear(s.safety.corridor_ahead(P0))
        self.assertFalse(ok); self.assertEqual(reasons, ["unobserved_corridor"])
        self.assertEqual(s.safety.footprint_cost(P0), COST_UNKNOWN)

    def test_observed_corridor_is_clear_then_blocked_then_stale(self):
        s = Stack()
        free_box(s.g, -0.5, 2.0, -0.6, 0.6, t0=1.0)
        s.step(3.0)
        ok, reasons, d = s.safety.corridor_clear(s.safety.corridor_ahead(P0), now_t=3.0)
        self.assertTrue(ok, (reasons, d))
        self.assertEqual(s.safety.footprint_cost(P0), 0)
        occupy(s.g, [(1.05, 0.05, "positive")], t0=4.0)
        s.step(5.0)
        ok, reasons, d = s.safety.corridor_clear(s.safety.corridor_ahead(P0), now_t=5.0)
        self.assertIn("obstacle_in_corridor", reasons); self.assertLessEqual(d["first_obstacle_m"], 0.85)
        ok, reasons, _ = s.safety.corridor_clear(s.safety.corridor_ahead(P0), now_t=30.0)
        self.assertIn("stale_corridor", reasons)                                # evidence older than 10 s
        s2 = Stack()
        free_box(s2.g, -0.5, 2.0, -0.6, 0.6, t0=1.0)
        s2.step(3.5, D=8.0, A=0.0)                                              # drove 8 m since: sigma_rel 0.16 m > 0.10
        ok, reasons, d = s2.safety.corridor_clear(s2.safety.corridor_ahead(P0), now_t=3.5)
        self.assertIn("stale_corridor", reasons); self.assertGreater(d["worst_free_sigma_m"], CFG.corridor_max_sigma_m)


class FakeFrame:
    def __init__(self, depth, index=0):
        self.depth, self.index = depth, index


class TestFaultInjector(unittest.TestCase):
    def test_off_unless_enabled(self):
        os.environ.pop("ARTEMIS_FAULTS", None)
        self.assertIsNone(FaultInjector.from_env())
        os.environ["ARTEMIS_FAULTS"] = ""
        self.assertIsNone(FaultInjector.from_env())
        os.environ.pop("ARTEMIS_FAULTS")

    def test_faults_are_deterministic_and_windowed(self):
        spec = {"odometry": {"scale": 1.03, "yaw_scale": 1.03}, "depth_dropout": {"t0": 1, "t1": 2, "fraction": 0.3, "seed": 7},
                "depth_freeze": {"t0": 3, "t1": 4}, "depth_invalid": {"t0": 5, "t1": 6}}
        os.environ["ARTEMIS_FAULTS"] = json.dumps(spec)
        try:
            fi = FaultInjector.from_env()
        finally:
            os.environ.pop("ARTEMIS_FAULTS")
        o = drive.Odometry(); fi.configure_odometry(o)
        self.assertEqual(o.scale, 1.03); self.assertAlmostEqual(o.yaw_scale, drive.YAW_SCALE * 1.03)
        idx = list(range(0, 20 * 30, 3))
        base = [2.0] * 600
        f0 = FakeFrame(array.array("f", base), 0)
        self.assertEqual(fi.apply_depth(f0, 0.5, idx, 20, 30), [])              # outside every window: untouched
        self.assertEqual(list(f0.depth), base)
        f1 = FakeFrame(array.array("f", base), 1)
        self.assertEqual(fi.apply_depth(f1, 1.5, idx, 20, 30), ["depth_dropout"])
        nan_idx = [i for i, v in enumerate(f1.depth) if v != v]
        self.assertEqual(len(nan_idx), int(len(idx) * 0.3)); self.assertTrue(set(nan_idx) <= set(idx))
        f1b = FakeFrame(array.array("f", base), 1)
        fi.apply_depth(f1b, 1.5, idx, 20, 30)
        self.assertEqual(nan_idx, [i for i, v in enumerate(f1b.depth) if v != v])   # same frame -> same pixels
        fi.apply_depth(FakeFrame(array.array("f", [7.0] * 600), 2), 2.5, idx, 20, 30)   # last good frame
        f3 = FakeFrame(array.array("f", [9.0] * 600), 3)
        self.assertEqual(fi.apply_depth(f3, 3.5, idx, 20, 30), ["depth_freeze"])
        self.assertEqual(set(f3.depth), {7.0})
        f5 = FakeFrame(array.array("f", base), 5)
        fi.apply_depth(f5, 5.5, idx, 20, 30)
        self.assertTrue(all(math.isinf(v) for v in f5.depth[300:])); self.assertEqual(set(f5.depth[:300]), {2.0})


class TestMapperFaultHandling(unittest.TestCase):
    """Frozen frames are rejected; invalid depth never creates FREE space (full Mapper with fake sensors)."""

    def run_mapper(self, depth_fn, moving=True, cycles=12, spec=None):
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import FakeSensors, FLAT
        sens = FakeSensors(FLAT)
        sens.capture = lambda depth_buffer=False, with_rgb=True, bgra=False: _frame(sens, depth_fn)
        os.environ.pop("ARTEMIS_FAULTS", None)
        fi = FaultInjector(spec) if spec else None
        with tempfile.TemporaryDirectory() as d:
            m = Mapper(sens, Path(d), faults=fi)
            states = []
            for k in range(1, cycles * 8 + 1):
                t = round(0.032 * k, 6)
                sens.t = t
                w = (t / 0.032) * 0.075 if moving else 0.0
                m.update_pose(w, w, t)
                if m.due(t):
                    m.cycle(t)
                else:
                    m.integrate_pending()
                if m.health_monitor.latest is not None:
                    states.append((t, m.health_monitor.latest.state, m.health_monitor.latest.reasons))
            s = m.finish()
            return m, s, states

    def test_frozen_frame_rejected_while_moving(self):
        from tests.test_mapping_depth import FLAT
        fixed = array.array("f", FLAT)
        m, s, states = self.run_mapper(lambda k: fixed, moving=True)
        self.assertGreater(s["frozen_frames_rejected"], 5)                      # identical image, rover moving: frozen
        self.assertTrue(any("depth_frozen" in r for _, _, r in states))
        m2, s2, _ = self.run_mapper(lambda k: fixed, moving=False)
        self.assertEqual(s2["frozen_frames_rejected"], 0)                       # stationary: identical images are expected

    def test_invalid_depth_creates_no_free_space(self):
        inf = [float("inf")] * (720 * 480)
        m, s, states = self.run_mapper(lambda k: array.array("f", inf), moving=True)
        self.assertEqual(m.grid.counts()["free"], 0); self.assertEqual(m.grid.counts()["occupied"], 0)
        self.assertTrue(all(q == COST_UNKNOWN for q in m.costmap.cost))
        self.assertTrue(any("depth_invalid" in r for _, _, r in states))
        self.assertEqual(states[-1][1], STOP)


def _frame(sens, depth_fn):
    from artemis_x.sim.sensors import Frame
    f = Frame(sens.count, sens.t, 720, 480, None, depth_fn(sens.count))
    sens.count += 1
    return f


if __name__ == "__main__":
    unittest.main()
