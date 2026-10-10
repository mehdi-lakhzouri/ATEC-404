"""R4.1: depth sampling, back-projection, ground / obstacle / negative classification, column scan, fast depth read,
and one mapper cycle with fake Webots sensors. Synthetic depth images are ray-cast analytically from the real camera model."""

import array
import ctypes
import json
import math
import tempfile
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim.mapping.config import MappingConfig
from artemis_x.sim.mapping.depth_scan import DepthScanner, read_depth
from artemis_x.sim.mapping.frames import Pose2D
from artemis_x.sim.mapping.mapper import Mapper
from artemis_x.sim.projection import CameraModel
from artemis_x.sim.sensors import Frame, intrinsics

I = intrinsics(720, 480, 1.0)
CAM = CameraModel(I["fx"], I["fy"], I["cx"], I["cy"], 720, 480, 0.05, 15.0)
INF = float("inf")


def ray(u, v):
    """Base-frame direction per metre of optical-axis depth (same algebra as DepthScanner, written out independently)."""
    c, s = math.cos(CAM.mount_pitch), math.sin(CAM.mount_pitch)
    zc = -(v - CAM.cy) / CAM.fy
    return c + s * zc, -(u - CAM.cx) / CAM.fx, -s + c * zc


def render(box=None, pit=None):
    """Depth image of flat ground (z = 0), optionally a box [x0, x1, y0, y1, height] and a pit [x0, x1, y0, y1, depth]."""
    img = []
    for v in range(480):
        for u in range(720):
            (ax, ay, az), (tx, ty, tz) = ray(u, v), CAM.mount_t
            d = -tz / az if az < 0 else INF                     # ground
            if pit and d < INF:
                x, y = tx + d * ax, ty + d * ay
                if pit[0] <= x <= pit[1] and pit[2] <= y <= pit[3]:
                    d = (-pit[4] - tz) / az                     # pit floor
            if box and ax > 0:
                dw = (box[0] - tx) / ax                         # front face x = box[0]
                y, z = ty + dw * ay, tz + dw * az
                if dw > 0 and box[2] <= y <= box[3] and 0 <= z <= box[4] and dw < d:
                    d = dw
            img.append(d if d < 15 else INF)
    return img


FLAT = render()
BOX = render(box=[2.0, 2.4, -0.3, 0.3, 0.2])
PIT = render(pit=[2.5, 3.5, -1.0, 1.0, 0.2])
POSE0 = Pose2D(0.0, 0.0, 0.0, 1.0)


class TestDepthScan(unittest.TestCase):
    def setUp(self):
        self.sc = DepthScanner(CAM, MappingConfig())

    def test_sampling_layout(self):
        self.assertEqual(self.sc.n_cols, 90)
        self.assertEqual(len(self.sc.samples), 90 * len(range(144, 480, 6)))

    def test_flat_ground_is_all_ground_and_free(self):
        obs = self.sc.scan(FLAT, POSE0, 1.0)
        self.assertEqual(obs.counts["positive"] + obs.counts["negative"], 0)
        self.assertEqual(obs.obstacle_points, [])
        self.assertTrue(all(c.hit_range_m is None for c in obs.columns))
        far = [c.free_range_m for c in obs.columns if c.free_range_m]
        self.assertEqual(len(far), 90)
        self.assertGreater(min(far), 4.0); self.assertLessEqual(max(far), 6.0 + 1e-9)   # mapping range cap

    def test_box_is_positive_obstacle_at_the_right_place(self):
        obs = self.sc.scan(BOX, POSE0, 1.0)
        pos = [p for p in obs.obstacle_points if p[3] == "positive"]
        self.assertGreater(len(pos), 20)
        for x, y, z, _ in pos:
            self.assertAlmostEqual(x, 2.0, delta=1e-6); self.assertLessEqual(abs(y), 0.3 + 1e-9); self.assertGreaterEqual(z, 0.05)
        hit = [c for c in obs.columns if c.hit_kind == "positive"]
        self.assertTrue(hit)
        for c in hit:
            self.assertAlmostEqual(c.free_range_m, c.hit_range_m)                 # free space ends at the obstacle
            self.assertLess(abs(c.bearing_rad), math.atan2(0.3, 1.84) + 0.02)
        self.assertTrue(any(c.hit_range_m is None for c in obs.columns))         # columns beside the box stay clear

    def test_pit_is_negative_hazard(self):
        obs = self.sc.scan(PIT, POSE0, 1.0)
        neg = [p for p in obs.obstacle_points if p[3] == "negative"]
        self.assertGreater(len(neg), 20)
        self.assertTrue(all(abs(z + 0.2) < 1e-6 and 2.5 <= x for x, _, z, _ in neg))
        self.assertTrue(any(c.hit_kind == "negative" for c in obs.columns))

    def test_pose_moves_points_into_odom(self):
        obs = self.sc.scan(BOX, Pose2D(1.0, 2.0, math.pi / 2, 1.0), 1.0)
        for x, y, z, _ in obs.obstacle_points:                                    # rover at (1, 2) facing +y: box 2 m ahead
            self.assertAlmostEqual(y, 4.0, delta=1e-6); self.assertLessEqual(abs(x - 1.0), 0.3 + 1e-9)

    def test_invalid_depth_never_creates_space(self):
        bad = [INF] * (720 * 480)
        obs = self.sc.scan(bad, POSE0, 1.0)
        self.assertEqual(obs.n_valid, 0); self.assertEqual(obs.valid_fraction, 0.0)
        self.assertTrue(all(c.free_range_m is None and c.hit_range_m is None for c in obs.columns))
        nan = [float("nan")] * (720 * 480)
        self.assertEqual(self.sc.scan(nan, POSE0, 1.0).n_valid, 0)
        near = [0.01] * (720 * 480)                                               # below min range
        self.assertEqual(self.sc.scan(near, POSE0, 1.0).counts["invalid"], len(self.sc.samples))


class FakeRangeFinder:
    def __init__(self, values):
        self.buf = (ctypes.c_float * len(values))(*values)
        self.values = values

    def getWidth(self): return 720
    def getHeight(self): return 480

    def getRangeImage(self, data_type="list"):
        return ctypes.cast(self.buf, ctypes.POINTER(ctypes.c_float)) if data_type == "buffer" else list(self.values)


class TestReadDepth(unittest.TestCase):
    def test_buffer_read_equals_list_read(self):
        vals = [float(i % 997) / 7 for i in range(720 * 480)]
        vals[5] = INF
        arr = read_depth(FakeRangeFinder(vals))
        self.assertIsInstance(arr, array.array)
        self.assertEqual(len(arr), 720 * 480)
        self.assertTrue(math.isinf(arr[5]))
        for i in (0, 1, 777, 345599):
            self.assertAlmostEqual(arr[i], vals[i], places=4)


class TestSensorCapture(unittest.TestCase):
    """The real ``SensorSuite.capture`` on fake Webots devices (regression: RGB must be returned unless explicitly skipped)."""

    def suite(self):
        from artemis_x.sim.sensors import SensorSuite

        class Cam:
            def enable(self, p): pass
            def getWidth(self): return 720
            def getHeight(self): return 480
            def getImage(self): return bytes([10, 20, 30, 255]) * (720 * 480)       # BGRA

        rf = FakeRangeFinder([2.5] * (720 * 480))
        rf.enable = lambda p: None

        class Robot:
            def getDevice(self, name): return {"front_camera": Cam(), "front_depth": rf}[name]
            def getTime(self): return 1.024

        return SensorSuite(Robot(), 64)

    def test_capture_variants(self):
        s = self.suite()
        f = s.capture()
        self.assertEqual(f.rgb[:3], bytes([30, 20, 10])); self.assertEqual((f.width, f.height), (720, 480))
        self.assertIsInstance(f.depth, list)
        g = s.capture(depth_buffer=True)
        self.assertEqual(len(g.rgb), 720 * 480 * 3); self.assertIsInstance(g.depth, array.array)
        self.assertAlmostEqual(g.depth[100], 2.5)
        h = s.capture(depth_buffer=True, with_rgb=False)
        self.assertIsNone(h.rgb); self.assertEqual((h.width, h.height), (720, 480)); self.assertEqual(len(h.depth), 720 * 480)


class FakeSensors:
    period_ms = 64

    def __init__(self, depth):
        self.depth_img, self.t, self.count = depth, 0.0, 0
        self.camera = self.depth = object()

    def describe(self):
        return {"depth": {**I, "min_range_m": 0.05, "max_range_m": 15.0}}

    def capture(self, depth_buffer=False, with_rgb=True, bgra=False):
        f = Frame(self.count, self.t, 720, 480, b"\0" * 3, self.depth_img)
        self.count += 1
        return f


class TestMapperCycle(unittest.TestCase):
    def test_cycle_alignment_pose_and_telemetry(self):
        with tempfile.TemporaryDirectory() as d:
            sens = FakeSensors(BOX)
            m = Mapper(sens, Path(d))
            cycles = []
            for k in range(1, 40):                                                # 32 ms steps, driving +x at 0.1 m/s
                t = round(0.032 * k, 6)
                sens.t = t
                m.update_pose(t / 0.032 * 0.075, t / 0.032 * 0.075, t)
                if m.due(t):
                    frame, pose = m.cycle(t, gt={"x": 9, "y": 9, "yaw": 0})
                    cycles.append((t, pose))
            s = m.finish()
            self.assertTrue(all(abs(t / 0.064 - round(t / 0.064)) < 1e-6 for t, _ in cycles))   # on sensor sample instants
            self.assertTrue(all(b[0] - a[0] >= 0.256 - 1e-9 for a, b in zip(cycles, cycles[1:])))
            self.assertTrue(all(p.stamp_s == t for t, p in cycles))
            lines = [json.loads(x) for x in (Path(d) / "mapping.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), s["cycles"]); self.assertGreater(s["cycles"], 3)
            self.assertEqual(lines[0]["gt_pose_eval_only"], {"x": 9, "y": 9, "yaw": 0})
            self.assertGreater(lines[0]["obstacle_points"], 0)
            pts = [json.loads(x) for x in (Path(d) / "obstacle_points.jsonl").read_text().splitlines()]
            self.assertLessEqual(len(pts[0]["points"]), MappingConfig().log_points_per_cycle)
            self.assertIn("p95", s["timing_ms"]["scan_ms"])


if __name__ == "__main__":
    unittest.main()
