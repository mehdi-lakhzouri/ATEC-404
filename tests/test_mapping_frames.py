"""R4.1: SE(2) frames, pose estimate + uncertainty, pose buffer, and the no-ground-truth rule for the mapping package."""

import ast
import math
import unittest

from tests._common import ROOT
from artemis_x.sim import drive
from artemis_x.sim.mapping.config import MappingConfig
from artemis_x.sim.mapping.frames import Pose2D, base_to_odom, compose, inverse, odom_to_base, wrap
from artemis_x.sim.mapping.pose import PoseBuffer, PoseEstimator, sample_time

MAPPING = ROOT / "src" / "artemis_x" / "sim" / "mapping"


class TestFrames(unittest.TestCase):
    def test_compose_inverse_round_trip(self):
        a = Pose2D(1.0, -2.0, 0.7, 0.0)
        b = Pose2D(0.3, 0.4, -1.2, 0.0)
        ab = compose(a, b)
        back = compose(inverse(a), ab)
        for u, v in ((back.x, b.x), (back.y, b.y), (back.yaw, b.yaw)):
            self.assertAlmostEqual(u, v, places=9)
        ident = compose(a, inverse(a))
        self.assertAlmostEqual(ident.x, 0, places=9); self.assertAlmostEqual(ident.y, 0, places=9); self.assertAlmostEqual(ident.yaw, 0)

    def test_base_odom_points(self):
        pose = Pose2D(1.0, 2.0, math.pi / 2, 0.0)
        p = base_to_odom((2.0, 0.5, 0.1), pose)                  # 2 m ahead, 0.5 m left of a rover facing +y
        self.assertAlmostEqual(p[0], 0.5); self.assertAlmostEqual(p[1], 4.0); self.assertEqual(p[2], 0.1)
        q = odom_to_base(p, pose)
        self.assertAlmostEqual(q[0], 2.0); self.assertAlmostEqual(q[1], 0.5)

    def test_wrap(self):
        self.assertAlmostEqual(wrap(3 * math.pi), math.pi, places=9)
        self.assertAlmostEqual(wrap(-3.5 * math.pi), 0.5 * math.pi, places=9)


class TestPoseEstimator(unittest.TestCase):
    def test_straight_then_turn_sigma_grows(self):
        cfg = MappingConfig()
        est = PoseEstimator(cfg)
        est.update(0, 0, 0.0)
        p = est.update(10, 10, 1.0)                               # 10 rad of wheel = 0.425 m straight
        self.assertAlmostEqual(p.x, 10 * drive.WHEEL_RADIUS_M)
        d = p.x                                                   # absolute bound of the R4.3 model (rho = D for a straight run)
        self.assertAlmostEqual(p.sigma_xy, ((cfg.odo_k_scale * d) ** 2 + (d * cfg.odo_k_yaw_dist * d) ** 2) ** 0.5)
        p2 = est.update(8, 12, 2.0)                               # spin in place
        self.assertGreater(p2.yaw, 0); self.assertGreater(p2.sigma_xy, p.sigma_xy)       # heading error x lever arm (R4.3 model)
        self.assertGreater(p2.sigma_yaw, p.sigma_yaw)
        self.assertEqual(p2.stamp_s, 2.0)

    def test_missing_encoders_keep_pose(self):
        est = PoseEstimator()
        est.update(0, 0, 0.0); est.update(4, 4, 0.032)
        p = est.update(None, None, 0.064)
        self.assertAlmostEqual(p.x, 4 * drive.WHEEL_RADIUS_M); self.assertEqual(p.stamp_s, 0.064)


class TestPoseBuffer(unittest.TestCase):
    def _buf(self):
        b = PoseBuffer(horizon_s=1.0)
        for k in range(50):                                       # 0.032 s steps, moving +x at 1 m/s, yaw crossing +-pi
            t = 0.032 * k
            b.add(Pose2D(t, 0.0, wrap(math.pi - 0.2 + 0.4 * t), t))
        return b

    def test_interpolation_and_yaw_short_arc(self):
        b = self._buf()
        p = b.at(1.2 + 0.016)
        self.assertAlmostEqual(p.x, 1.216, places=6)
        self.assertAlmostEqual(math.cos(p.yaw), math.cos(math.pi - 0.2 + 0.4 * 1.216), places=6)   # no flip across +-pi

    def test_horizon_and_out_of_range(self):
        b = self._buf()
        self.assertLessEqual(b.latest().stamp_s - b._q[0].stamp_s, 1.0)
        self.assertIsNone(b.at(0.0))                              # dropped (older than the horizon)
        self.assertIsNone(b.at(b.latest().stamp_s + 0.1))         # future
        self.assertEqual(b.at(b.latest().stamp_s).x, b.latest().x)

    def test_sample_time(self):
        self.assertAlmostEqual(sample_time(0.288, 0.064), 0.256)
        self.assertAlmostEqual(sample_time(0.256, 0.064), 0.256)  # float noise must not drop a whole period
        self.assertAlmostEqual(sample_time(0.064 * 3, 0.064), 0.192)


class TestNoGroundTruthInMapping(unittest.TestCase):
    """Plan rule: the live mapping package never uses the Webots supervisor ground truth."""
    FORBIDDEN = {"getSelf", "getPosition", "getOrientation", "getFromDef", "Supervisor", "getRecognitionObjects",
                 "getRecognitionSegmentationImage", "self_node"}

    def test_no_supervisor_api(self):
        for f in sorted(MAPPING.glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            self.assertFalse(names & self.FORBIDDEN, f"{f.name} uses {names & self.FORBIDDEN}")

    def test_gt_argument_is_only_logged(self):
        """Any ``gt`` value passed in may only be tested for None or stored under ``gt_pose_eval_only`` (by subscript
        assignment or in a dict literal used to carry it to the log)."""
        for f in sorted(MAPPING.glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            for n in ast.walk(tree):
                if isinstance(n, ast.Name) and n.id == "gt" and isinstance(n.ctx, ast.Load):
                    par = parents[n]
                    ok = isinstance(par, ast.Compare) or (
                        isinstance(par, ast.Assign) and any(isinstance(tg, ast.Subscript) and isinstance(tg.slice, ast.Constant)
                                                            and tg.slice.value == "gt_pose_eval_only" for tg in par.targets)) or (
                        isinstance(par, ast.Dict) and any(v is n and isinstance(k, ast.Constant) and k.value == "gt_pose_eval_only"
                                                          for k, v in zip(par.keys, par.values)))
                    self.assertTrue(ok, f"{f.name}:{n.lineno} uses ground truth beyond logging")


if __name__ == "__main__":
    unittest.main()
