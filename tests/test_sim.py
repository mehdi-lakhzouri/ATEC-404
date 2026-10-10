"""Webots simulation support (artemis_x.sim): pure logic tests + an opt-in end-to-end Webots run."""

import math
import os
import subprocess
import sys
import unittest

from tests._common import ROOT

from artemis_x.sim import drive  # noqa: E402
from artemis_x.sim.sensors import bgra_to_rgb, depth_stats, intrinsics  # noqa: E402


class TestDrive(unittest.TestCase):
    def test_primitives(self):
        self.assertEqual(drive.forward(3), drive.WheelCommand(3, 3))
        self.assertEqual(drive.backward(3), drive.WheelCommand(-3, -3))
        self.assertEqual(drive.turn_left(2), drive.WheelCommand(-2, 2))
        self.assertEqual(drive.turn_right(2), drive.WheelCommand(2, -2))
        self.assertEqual(drive.STOP, drive.WheelCommand(0, 0))

    def test_clamp_keeps_curvature(self):
        c = drive.WheelCommand(20, 10).clamped(10)
        self.assertEqual((c.left, c.right), (10, 5))
        self.assertEqual(drive.WheelCommand(3, -4).clamped(10), drive.WheelCommand(3, -4))

    def test_twist_round_trip(self):
        cmd = drive.twist_to_wheels(0.2, 0.5)
        v, w = drive.wheels_to_twist(cmd)
        self.assertAlmostEqual(v, 0.2); self.assertAlmostEqual(w, 0.5)
        self.assertGreater(drive.wheels_to_twist(drive.turn_left(1))[1], 0)       # left turn = +yaw

    def test_demo_sequence(self):
        steps = drive.demo_sequence()
        self.assertEqual([s.name for s in steps], ["start_stationary", "forward_1", "stop_1", "turn_left", "forward_2", "stop_2"])
        self.assertEqual(steps[0].command, drive.STOP); self.assertEqual(steps[-1].command, drive.STOP)
        turn = steps[3]
        _, w = drive.wheels_to_twist(turn.command)
        self.assertAlmostEqual(w * turn.duration_s, math.pi / 2)
        p = drive.SequencePlayer(steps)
        self.assertEqual(p.index_at(0.0), 0); self.assertEqual(p.index_at(1.5), 1); self.assertIsNone(p.index_at(p.total_s + 0.01))


class TestSensors(unittest.TestCase):
    def test_bgra_to_rgb(self):
        self.assertEqual(bgra_to_rgb(bytes([1, 2, 3, 255, 10, 20, 30, 255]), 2, 1), bytes([3, 2, 1, 30, 20, 10]))
        with self.assertRaises(ValueError):
            bgra_to_rgb(bytes(7), 2, 1)

    def test_depth_stats_excludes_no_return(self):
        s = depth_stats([0.5, 1.0, float("inf"), 15.0, 0.01], 0.05, 15.0)
        self.assertEqual((s["valid"], s["non_finite"], s["no_return"]), (2, 1, 1))
        self.assertEqual((s["min_m"], s["max_m"]), (0.5, 1.0))

    def test_intrinsics(self):
        k = intrinsics(720, 480, 1.0)
        self.assertAlmostEqual(k["fx"], 360 / math.tan(0.5))
        self.assertEqual((k["cx"], k["cy"]), (359.5, 239.5))


@unittest.skipUnless(os.environ.get("ARTEMIS_RUN_WEBOTS") == "1", "set ARTEMIS_RUN_WEBOTS=1 to launch Webots (about 5-20 s)")
class TestWebotsEndToEnd(unittest.TestCase):
    def test_smoke(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/run_webots_smoke.py")], capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
