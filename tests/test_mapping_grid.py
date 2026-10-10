"""R4.2: log-odds grid updates, rolling-window recentring, UNKNOWN preservation (blind zone, occlusion, hidden pits) and
snapshot serialisation. Scene tests reuse the analytically ray-cast depth images of ``test_mapping_depth``."""

import math
import tempfile
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim.mapping.config import MappingConfig
from artemis_x.sim.mapping.depth_scan import Column, DepthObservation, DepthScanner
from artemis_x.sim.mapping.frames import Pose2D
from artemis_x.sim.mapping.grid import FREE, NEGATIVE, NEVER, OCCUPIED, UNKNOWN, LogOddsGrid
from artemis_x.sim.mapping.snapshot import SCHEMA, read_snapshot, write_snapshot
from tests.test_mapping_depth import BOX, CAM, FLAT, PIT

CFG = MappingConfig()
P0 = Pose2D(0.0, 0.0, 0.0, 0.0)


def obs(points=(), segments=(), pose=P0, t=0.0):
    col = Column(0.0, None, None, None, tuple(segments))
    return DepthObservation(t, 0, pose, [col], list(points), 1, 1)


class TestLogOdds(unittest.TestCase):
    def setUp(self):
        self.g = LogOddsGrid(CFG)

    def st(self, x, y):
        return self.g.state_at(self.g.index(x, y))

    def test_free_segment_then_hit(self):
        for t in range(3):
            self.g.integrate(obs([(2.05, 0.05, 0.2, "positive")], [((1.0, 0.05), (1.95, 0.05))], t=t))
        for x in (1.05, 1.45, 1.95):
            self.assertEqual(self.st(x, 0.05), FREE)
        self.assertEqual(self.st(2.05, 0.05), OCCUPIED)
        self.assertEqual(self.st(2.55, 0.05), UNKNOWN)                     # beyond the hit: never observed
        self.assertEqual(self.st(0.45, 0.05), UNKNOWN)                     # before the first ground sample (blind zone)

    def test_thresholds_need_repeated_evidence(self):
        self.g.integrate(obs([(2.05, 0.05, 0.2, "positive")]))
        self.assertEqual(self.st(2.05, 0.05), UNKNOWN)                     # 0.85 is not > 0.85: one hit is not enough
        self.g.integrate(obs([(2.05, 0.05, 0.2, "positive")], t=1))
        self.assertEqual(self.st(2.05, 0.05), OCCUPIED)
        for t in range(2):
            self.g.integrate(obs(segments=[((1.05, 0.05), (1.05, 0.05))], t=t))
        self.assertEqual(self.st(1.05, 0.05), UNKNOWN)                     # -0.8 is not < -0.85
        self.g.integrate(obs(segments=[((1.05, 0.05), (1.05, 0.05))], t=3))
        self.assertEqual(self.st(1.05, 0.05), FREE)

    def test_saturation_and_reversal(self):
        k = self.g.index(2.05, 0.05)
        for t in range(20):
            self.g.integrate(obs([(2.05, 0.05, 0.2, "positive")], t=t))
        self.assertAlmostEqual(self.g.logodds[k], CFG.l_max)
        need = math.floor((CFG.l_max - CFG.l_free) / -CFG.l_miss) + 1        # misses to cross into FREE from the clamp
        for t in range(need - 1):
            self.g.integrate(obs(segments=[((2.05, 0.05), (2.05, 0.05))], t=t))
        self.assertNotEqual(self.st(2.05, 0.05), FREE)
        self.g.integrate(obs(segments=[((2.05, 0.05), (2.05, 0.05))], t=99))
        self.assertEqual(self.st(2.05, 0.05), FREE)
        for t in range(40):
            self.g.integrate(obs(segments=[((2.05, 0.05), (2.05, 0.05))], t=t))
        self.assertAlmostEqual(self.g.logodds[k], CFG.l_min)

    def test_one_update_per_cell_per_cycle_and_hit_wins(self):
        k = self.g.index(2.05, 0.05)
        pts = [(2.01 + 0.01 * i, 0.05, 0.2, "positive") for i in range(8)]   # 8 points, one cell
        self.g.integrate(obs(pts, [((1.0, 0.05), (2.09, 0.05))]))          # the free segment also crosses that cell
        self.assertAlmostEqual(self.g.logodds[k], CFG.l_hit)

    def test_negative_hazard_kind(self):
        for t in range(2):
            self.g.integrate(obs([(3.05, 0.05, -0.2, "negative")], t=t))
        self.assertEqual(self.st(3.05, 0.05), NEGATIVE)
        for t in range(2):
            self.g.integrate(obs([(3.05, 0.05, 0.2, "positive")], t=5 + t))
        self.assertEqual(self.st(3.05, 0.05), OCCUPIED)                    # latest hit decides the kind

    def test_pose_transform_of_segments(self):
        p = Pose2D(1.0, 2.0, math.pi / 2, 0.0)                             # facing +y: base x -> odom +y
        for t in range(3):
            self.g.integrate(obs(segments=[((1.0, 0.0), (1.5, 0.0))], pose=p, t=t))
        self.assertEqual(self.st(1.0, 3.25), FREE)
        self.assertEqual(self.st(2.25, 2.0), UNKNOWN)


class TestRecenter(unittest.TestCase):
    def test_cells_keep_their_odom_position(self):
        g = LogOddsGrid(CFG)
        for t in range(3):
            g.integrate(obs([(3.05, -1.95, 0.2, "positive"), (-5.95, 0.05, 0.2, "positive")], [((1.0, 1.0), (2.0, 1.0))], t=t))
        before = {xy: g.state_at(g.index(*xy)) for xy in ((3.05, -1.95), (1.55, 1.05), (-5.95, 0.05))}
        self.assertFalse(g.recenter(0.5, 0.5))                             # inside the dead band: no shift
        self.assertTrue(g.recenter(4.04, -1.53))
        self.assertEqual(g.origin_x, round(round((4.04 - 6) / 0.1) * 0.1, 9))
        self.assertEqual(g.state_at(g.index(3.05, -1.95)), before[(3.05, -1.95)])
        self.assertEqual(g.state_at(g.index(1.55, 1.05)), FREE)
        self.assertEqual(g.index(-5.95, 0.05), -1)                         # left the window: dropped
        self.assertEqual(g.state_at(g.index(9.95, -7.45)), UNKNOWN)        # entered the window: unknown
        self.assertEqual(g.last_obs[g.index(9.95, -7.45)], NEVER)
        self.assertEqual(g.recenters, 1)

    def test_round_trip_shift(self):
        g = LogOddsGrid(CFG)
        for t in range(3):
            g.integrate(obs([(0.55, 0.55, 0.2, "positive")], t=t))
        g.recenter(3.0, 2.0); g.recenter(0.0, 0.0)
        self.assertEqual(g.state_at(g.index(0.55, 0.55)), OCCUPIED)


class TestSceneSemantics(unittest.TestCase):
    """Full chain on rendered depth: scanner -> grid. The rover sits at the origin facing +x."""

    def run_scene(self, depth, cycles=4):
        sc, g = DepthScanner(CAM, CFG), LogOddsGrid(CFG)
        for t in range(cycles):
            g.integrate(sc.scan(depth, P0, float(t)))
        return g

    def test_blind_zone_stays_unknown(self):
        g = self.run_scene(FLAT)
        for x in (0.05, 0.25, 0.45, 0.55):                                  # ground nearer than ~0.64 m is never seen
            self.assertEqual(g.state_at(g.index(x, 0.05)), UNKNOWN, x)
        for x in (1.05, 2.05):
            self.assertEqual(g.state_at(g.index(x, 0.05)), FREE, x)
        self.assertEqual(g.state_at(g.index(-1.05, 0.05)), UNKNOWN)         # behind the rover
        self.assertEqual(g.state_at(g.index(1.05, 2.05)), UNKNOWN)          # outside the field of view

    def test_box_front_occupied_shadow_unknown(self):
        g = self.run_scene(BOX)
        self.assertIn(OCCUPIED, (g.state_at(g.index(1.95, 0.05)), g.state_at(g.index(2.05, 0.05))))   # face at x = 2.0
        for x in (2.55, 3.05, 3.55):                                         # hidden behind the box
            self.assertNotEqual(g.state_at(g.index(x, 0.05)), FREE, x)
        self.assertEqual(g.state_at(g.index(1.55, 0.05)), FREE)

    def test_hidden_pit_is_never_free(self):
        g = self.run_scene(PIT)
        kinds = {UNKNOWN: 0, FREE: 0, OCCUPIED: 0, NEGATIVE: 0}
        for k in range(len(g.logodds)):
            x, y = g.cell_centre(k)
            if 2.6 < x < 3.4 and -0.9 < y < 0.9:                            # cells entirely inside the pit footprint
                st = g.state_at(k)
                kinds[st] += 1
                self.assertNotEqual(st, FREE, (x, y))
        # the simple renderer has no far pit wall, so floor returns land beyond the footprint: check they exist at all
        self.assertTrue(any(g.state_at(k) == NEGATIVE for k in range(len(g.logodds))))


def render_slab(x0=2.0, x1=2.6, half_w=0.3, z_bottom=0.10, z_top=0.30):
    """Flat ground plus a raised slab (an overhang: a bulging boulder, a lying tank, a tilted panel) whose underside is
    z_bottom above the ground; the camera can see ground under it."""
    from tests.test_mapping_depth import INF, ray
    img = []
    tx, ty, tz = CAM.mount_t
    for v in range(480):
        for u in range(720):
            ax, ay, az = ray(u, v)
            d = -tz / az if az < 0 else INF
            hits = []
            if ax > 0:
                dw = (x0 - tx) / ax                                          # front face
                y, z = ty + dw * ay, tz + dw * az
                if abs(y) <= half_w and z_bottom <= z <= z_top:
                    hits.append(dw)
            if az != 0:
                for zz in (z_bottom, z_top):                                 # underside and top
                    dp = (zz - tz) / az
                    x, y = tx + dp * ax, ty + dp * ay
                    if dp > 0 and x0 <= x <= x1 and abs(y) <= half_w:
                        hits.append(dp)
            d = min([d] + hits)
            img.append(d if d < 15 else INF)
    return img


class TestOverhang(unittest.TestCase):
    def test_ground_seen_under_an_overhang_is_not_free(self):
        sc, g = DepthScanner(CAM, CFG), LogOddsGrid(CFG)
        img = render_slab()
        for t in range(4):
            g.integrate(sc.scan(img, P0, float(t)))
        self.assertEqual(g.state_at(g.index(1.55, 0.05)), FREE)               # open ground in front
        for x in (2.15, 2.25, 2.35, 2.45):                                    # under the slab: never FREE
            self.assertNotEqual(g.state_at(g.index(x, 0.05)), FREE, x)
        self.assertIn(OCCUPIED, (g.state_at(g.index(1.95, 0.05)), g.state_at(g.index(2.05, 0.05))))

    def test_rule_can_be_disabled_for_comparison(self):
        from dataclasses import replace
        cfg = replace(CFG, overhang_max_height_m=-1.0)                         # no obstacle counts as overhead
        sc, g = DepthScanner(CAM, cfg), LogOddsGrid(cfg)
        img = render_slab()
        for t in range(4):
            g.integrate(sc.scan(img, P0, float(t)))
        self.assertTrue(any(g.state_at(g.index(x, 0.05)) == FREE for x in (2.15, 2.25, 2.35, 2.45)))   # the bug it fixes


class TestChainsEqualSegments(unittest.TestCase):
    """R4 performance review: the scanner emits free space as chains; the grid's chain path must update exactly the same
    cells as the R4.2 segment path (Column.segments() gives that form)."""

    def test_identical_grids(self):
        import random
        from tests.test_mapping_depth import INF
        rnd = random.Random(9)
        noisy = [rnd.choice([INF, float("nan"), rnd.uniform(0.3, 7.0)]) if rnd.random() < 0.3 else v * rnd.uniform(0.9, 1.1) for v in FLAT]
        sc = DepthScanner(CAM, CFG)
        for img in (FLAT, BOX, PIT, render_slab(), noisy):
            for pose in (P0, Pose2D(1.3, -0.7, 0.4, 0.0), Pose2D(-4.0, 5.0, 2.9, 0.0)):
                o = sc.scan(img, pose, 1.0)
                seg_cols = [Column(c.bearing_rad, c.free_range_m, c.hit_range_m, c.hit_kind, c.segments(), ()) for c in o.columns]
                o2 = DepthObservation(o.stamp_s, o.frame_index, o.pose, seg_cols, o.obstacle_points, o.n_sampled, o.n_valid, o.counts)
                a, b = LogOddsGrid(CFG), LogOddsGrid(CFG)
                a.integrate(o); b.integrate(o2)
                self.assertEqual(bytes(a.logodds), bytes(b.logodds))
                self.assertEqual(bytes(a.last_obs), bytes(b.last_obs))

    def test_protected_cells_never_receive_free_evidence(self):
        g = LogOddsGrid(CFG)
        k = g.index(1.45, 0.05)
        g.protected = {k}
        for t in range(5):
            g.integrate(obs(segments=[((1.0, 0.05), (1.95, 0.05))], t=t))
        self.assertEqual(g.state_at(k), UNKNOWN); self.assertEqual(g.state_at(g.index(1.25, 0.05)), FREE)


class TestSnapshot(unittest.TestCase):
    def test_round_trip(self):
        g = LogOddsGrid(CFG)
        for t in range(3):
            g.integrate(obs([(2.05, 0.05, 0.2, "positive")], [((1.0, 0.05), (1.95, 0.05))], t=float(t)))
        g.recenter(1.6, 0.0)
        with tempfile.TemporaryDirectory() as d:
            hdr = write_snapshot(g, 3.0, P0.to_dict(), Path(d) / "map_3.000")             # dots in the name are kept
            h2, L = read_snapshot(Path(d) / "map_3.000.json")
        self.assertEqual(h2["schema"], SCHEMA); self.assertEqual(h2, hdr)
        self.assertEqual(L["state"], g.state_layer())                         # derived by the reader = live state
        self.assertEqual(h2["origin_xy"], [g.origin_x, g.origin_y])
        k = g.index(2.05, 0.05)
        self.assertEqual(L["log_odds"][k], g.logodds[k])
        self.assertEqual(L["last_obs_s"][k], 2.0)                            # last updated at t = 2.0
        self.assertEqual(L["last_obs_s"][g.index(-3.0, -3.0)], NEVER)       # never observed
        self.assertEqual(L["state"].count(OCCUPIED), 1)


if __name__ == "__main__":
    unittest.main()
