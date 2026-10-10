"""R4.4: YOLO + depth fusion (measurement geometry, occlusion / bimodal depth, small-rock rule) and the obstacle tracker
(association, confirmation, tentative expiry, merging, visibility-gated misses). Depth images are ray-cast analytically."""

import math
import random
import unittest

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim.mapping.config import MappingConfig
from artemis_x.sim.mapping.depth_scan import DepthScanner
from artemis_x.sim.mapping.frames import Pose2D
from artemis_x.sim.mapping.fusion import measure
from artemis_x.sim.mapping.tracker import CONFIRMED, LOST, TENTATIVE, ObstacleTracker
from tests.test_mapping_depth import CAM, INF, ray

CFG = MappingConfig()
P0 = Pose2D(0.0, 0.0, 0.0, 0.0)


def render_boxes(boxes):
    """Flat ground + axis-aligned boxes [x0, x1, y0, y1, z0, z1] (front faces and tops)."""
    img = []
    tx, ty, tz = CAM.mount_t
    for v in range(480):
        for u in range(720):
            ax, ay, az = ray(u, v)
            d = -tz / az if az < 0 else INF
            for x0, x1, y0, y1, z0, z1 in boxes:
                if ax > 0:
                    dw = (x0 - tx) / ax
                    y, z = ty + dw * ay, tz + dw * az
                    if y0 <= y <= y1 and z0 <= z <= z1 and 0 < dw < d:
                        d = dw
                if az < 0:
                    dt = (z1 - tz) / az
                    x, y = tx + dt * ax, ty + dt * ay
                    if x0 <= x <= x1 and y0 <= y <= y1 and 0 < dt < d:
                        d = dt
            img.append(d if d < 15 else INF)
    return img


def box_px(x0, y0, y1, z0, z1):
    """Image box of a front face at x0 (camera model)."""
    pts = []
    for y in (y0, y1):
        for z in (z0, z1):
            c, s = math.cos(CAM.mount_pitch), math.sin(CAM.mount_pitch)
            px, py, pz = x0 - CAM.mount_t[0], y - CAM.mount_t[1], z - CAM.mount_t[2]
            xc, zc = c * px - s * pz, s * px + c * pz                      # base -> camera (inverse pitch)
            pts.append((CAM.cx - CAM.fx * py / xc, CAM.cy - CAM.fy * zc / xc))
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return [min(us), min(vs), max(us), max(vs)]


ROCK = [2.0, 2.4, -0.2, 0.2, 0.0, 0.25]
IMG_ROCK = render_boxes([ROCK])
IMG_OCCL = render_boxes([[3.0, 3.4, -0.3, 0.3, 0.0, 0.3], [2.0, 2.05, 0.05, 0.4, 0.0, 0.6]])   # far rock, nearer post covering part
IMG_LOW = render_boxes([[2.0, 2.2, -0.1, 0.1, 0.0, 0.035]])                                     # 3.5 cm high pebble


class TestMeasure(unittest.TestCase):
    def test_geometry(self):
        det = {"xyxy": box_px(2.0, -0.2, 0.2, 0.0, 0.25), "conf": 0.8}
        m, why = measure(det, IMG_ROCK, P0, CAM, CFG, 1.0, 3)
        self.assertIsNone(why)
        self.assertAlmostEqual(m.surface_odom[0], 2.0, delta=0.03)          # front face
        self.assertAlmostEqual(m.surface_odom[1], 0.0, delta=0.05)
        self.assertAlmostEqual(m.radius_m, 0.2, delta=0.04)                 # half the 0.4 m width
        self.assertAlmostEqual(m.centre_odom[0], 2.0 + m.radius_m, delta=0.03)
        self.assertAlmostEqual(m.height_m, 0.25, delta=0.05)
        self.assertFalse(m.depth_bimodal); self.assertEqual(m.low_points, [])
        self.assertAlmostEqual(m.sigma_m, CFG.fusion_sigma0_m + CFG.fusion_sigma_per_m * m.range_m)

    def test_pose_at_capture_is_used(self):
        det = {"xyxy": box_px(2.0, -0.2, 0.2, 0.0, 0.25), "conf": 0.8}
        m, _ = measure(det, IMG_ROCK, Pose2D(5.0, 1.0, math.pi / 2, 1.0), CAM, CFG, 1.0, 3)
        self.assertAlmostEqual(m.surface_odom[0], 5.0, delta=0.05); self.assertAlmostEqual(m.surface_odom[1], 3.0, delta=0.03)

    def test_occluded_box_is_bimodal_and_uses_the_dominant_surface(self):
        det = {"xyxy": box_px(3.0, -0.3, 0.3, 0.0, 0.3), "conf": 0.7}
        m, _ = measure(det, IMG_OCCL, P0, CAM, CFG, 1.0, 3)
        self.assertTrue(m.depth_bimodal)
        self.assertAlmostEqual(m.surface_odom[0], 3.0, delta=0.06)          # the far rock, not the post at 2.0 m
        self.assertAlmostEqual(m.sigma_m, 2 * (CFG.fusion_sigma0_m + CFG.fusion_sigma_per_m * m.range_m))

    def test_small_rock_rule_points(self):
        det = {"xyxy": box_px(2.0, -0.1, 0.1, 0.0, 0.035), "conf": 0.5}
        m, _ = measure(det, IMG_LOW, P0, CAM, CFG, 1.0, 3)
        self.assertGreater(len(m.low_points), 2)
        self.assertTrue(all(CFG.fusion_low_obstacle_m <= z < CFG.obstacle_height_m for _, _, z in m.low_points))
        obs = DepthScanner(CAM, CFG).scan(IMG_LOW, P0, 1.0)                    # depth alone: no obstacle at all
        self.assertEqual(obs.counts["positive"], 0)

    def test_no_depth(self):
        det = {"xyxy": [300, 100, 340, 130], "conf": 0.6}                     # sky: no valid depth
        m, why = measure(det, IMG_ROCK, P0, CAM, CFG, 1.0, 3)
        self.assertIsNone(m); self.assertEqual(why, "insufficient valid depth in box")


class M:
    def __init__(self, x, y, r=0.2, rng=2.0, sigma=0.07, bimodal=False, conf=0.6):
        self.centre_odom, self.radius_m, self.range_m, self.sigma_m = (x, y), r, rng, sigma
        self.height_m, self.conf, self.depth_bimodal = 0.2, conf, bimodal


class TestTracker(unittest.TestCase):
    def test_noisy_repeats_make_one_confirmed_track(self):
        rnd = random.Random(1)
        tr = ObstacleTracker(CFG)
        for k in range(20):
            tr.update([M(3.0 + rnd.gauss(0, 0.05), 1.0 + rnd.gauss(0, 0.05))], P0, 0.25 * k)
        self.assertEqual(len(tr.tracks), 1)
        t = tr.tracks[0]
        self.assertEqual(t.status, CONFIRMED); self.assertEqual(t.hits, 20)
        self.assertLess(math.hypot(t.x - 3.0, t.y - 1.0), 0.04)

    def test_two_rocks_two_tracks(self):
        tr = ObstacleTracker(CFG)
        for k in range(4):
            tr.update([M(3.0, 1.0), M(3.0, 0.2)], P0, 0.25 * k)
        self.assertEqual(sorted(len([t for t in tr.tracks if t.status == CONFIRMED]) for _ in [0]), [2])

    def test_unconfirmed_detection_expires(self):
        tr = ObstacleTracker(CFG)
        tr.update([M(2.0, 0.0)], P0, 0.0)                                      # a one-off (false positive)
        self.assertEqual(tr.tracks[0].status, TENTATIVE)
        tr.update([], P0, CFG.tentative_timeout_s + 0.1)
        self.assertEqual(tr.tracks, []); self.assertEqual(tr.stats["deleted_tentative"], 1)

    def test_overlapping_confirmed_tracks_merge(self):
        tr = ObstacleTracker(CFG)
        for k in range(3):                                                     # two tracks born apart, then grow together
            tr.update([M(3.0, 1.0, r=0.1, rng=0.5), M(3.0, 1.2, r=0.1, rng=0.5)], P0, 0.25 * k)   # one rock seen as two
        self.assertEqual(len(tr.confirmed()), 2)
        for k in range(3, 6):
            tr.update([M(3.0, 1.0, r=0.4, rng=0.5), M(3.0, 1.2, r=0.4, rng=0.5)], P0, 0.25 * k)   # closer: its true size
        self.assertEqual(len(tr.confirmed()), 1); self.assertEqual(tr.stats["merged"], 1)

    def test_misses_only_when_visible(self):
        tr = ObstacleTracker(CFG)
        for k in range(3):
            tr.update([M(2.5, 0.0)], P0, 0.25 * k)
        behind = Pose2D(5.0, 0.0, 0.0, 1.0)                                    # the rock is behind the rover: no misses
        for k in range(6):
            tr.update([], behind, 1.0 + 0.25 * k)
        self.assertEqual(tr.tracks[0].misses, 0)
        open_ground = render_boxes([])
        for k in range(CFG.lost_misses):                                       # in view, nothing in front, not detected
            tr.update([], P0, 3.0 + 0.25 * k, depth=open_ground, cam=CAM)
        self.assertEqual(tr.tracks[0].status, LOST)
        tr3 = ObstacleTracker(CFG)
        for k in range(3):
            tr3.update([M(2.5, 0.0)], P0, 0.25 * k)
        for k in range(6):                                                     # no depth return there (sky): not visible
            tr3.update([], P0, 1.0 + 0.25 * k, depth=[INF] * (720 * 480), cam=CAM)
        self.assertEqual(tr3.tracks[0].misses, 0)
        tr2 = ObstacleTracker(CFG)
        for k in range(3):
            tr2.update([M(3.2, 0.0)], P0, 0.25 * k)
        wall = render_boxes([[1.5, 1.6, -1.0, 1.0, 0.0, 1.0]])               # a nearer wall hides it: no misses
        for k in range(6):
            tr2.update([], P0, 1.0 + 0.25 * k, depth=wall, cam=CAM)
        self.assertEqual(tr2.tracks[0].misses, 0)


class TestDepthSupportAndRefine(unittest.TestCase):
    def test_unsupported_hits_never_confirm(self):
        tr = ObstacleTracker(CFG)
        for k in range(6):
            m = M(2.0, 0.0); m.support = 0                                       # a YOLO box on bare terrain
            tr.update([m], P0, 0.25 * k)
        self.assertEqual([t.status for t in tr.tracks], [TENTATIVE])
        tr.update([], P0, 10.0)
        self.assertEqual(tr.tracks, [])                                          # expired, never confirmed

    def _arc_grid(self):
        from artemis_x.sim.mapping.grid import LogOddsGrid
        g = LogOddsGrid(CFG)
        pts = [(3.0 - 0.9 * math.cos(a), 0.9 * math.sin(a), 0.3, "positive") for a in [k * 0.05 - 1.0 for k in range(41)]]
        from tests.test_mapping_safety import obs
        for t in range(3):
            g.integrate(obs(pts, t=float(t)))                                    # visible arc of a 1.8 m boulder
        return g

    def test_fragments_on_a_boulder_merge_and_fit_the_centre(self):
        g = self._arc_grid()
        tr = ObstacleTracker(CFG)
        for k in range(3):                                                       # two fragment boxes, small radii
            tr.update([M(2.15, -0.35, r=0.05), M(2.2, 0.45, r=0.05)], P0, 0.25 * k)
        self.assertEqual(len(tr.confirmed()), 2)
        out = tr.refine(g)
        self.assertEqual(len(tr.confirmed()), 1); self.assertEqual(out["merged"], 1)
        t = tr.confirmed()[0]
        self.assertLess(math.hypot(t.x - 3.0, t.y - 0.0), 0.15); self.assertAlmostEqual(t.radius_m, 0.9, delta=0.15)

    def test_rocks_on_a_crater_rim_stay_separate(self):
        from artemis_x.sim.mapping.grid import LogOddsGrid
        from tests.test_mapping_safety import obs
        g = LogOddsGrid(CFG)
        rim = [(3.0 + 1.3 * math.cos(a), 1.3 * math.sin(a), 0.1, "positive") for a in [k * 0.08 for k in range(79)]]
        bowl = [(3.0 + 0.1 * dx, 0.1 * dy, -0.2, "negative") for dx in range(-4, 5) for dy in range(-4, 5)]
        for t in range(3):
            g.integrate(obs(rim + bowl, t=float(t)))                            # a 2.6 m crater: rim ring + bowl
        tr = ObstacleTracker(CFG)
        rocks = [(3.0 - 1.3, 0.0), (3.0, 1.3), (3.0 + 1.3, 0.0)]                # three rocks sitting on the rim
        for k in range(3):
            tr.update([M(x, y, r=0.15, rng=1.5) for x, y in rocks], P0, 0.25 * k)
        out = tr.refine(g)
        self.assertEqual(len(tr.confirmed()), 3); self.assertEqual(out.get("terrain_blobs_skipped", 0) >= 1, True)
        self.assertTrue(all(t.radius_m < 0.3 for t in tr.confirmed()))         # not inflated to the crater size

    def test_small_touching_rocks_are_not_merged(self):
        from artemis_x.sim.mapping.grid import LogOddsGrid
        from tests.test_mapping_safety import obs
        g = LogOddsGrid(CFG)
        pts = [(2.0 + 0.1 * i, 0.05, 0.2, "positive") for i in range(5)]          # one 0.5 m blob: two touching rocks
        for t in range(3):
            g.integrate(obs(pts, t=float(t)))
        tr = ObstacleTracker(CFG)
        for k in range(3):
            tr.update([M(2.05, 0.05, r=0.12), M(2.45, 0.05, r=0.12)], P0, 0.25 * k)
        tr.refine(g)
        self.assertEqual(len(tr.confirmed()), 2)


class TestMapperFuse(unittest.TestCase):
    def test_fuse_rules(self):
        import tempfile
        from pathlib import Path
        from artemis_x.sim.mapping.grid import OCCUPIED
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import FakeSensors
        with tempfile.TemporaryDirectory() as d:
            m = Mapper(FakeSensors(IMG_LOW), Path(d))
            det = [{"xyxy": box_px(2.0, -0.1, 0.1, 0.0, 0.035), "conf": 0.5}]
            pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
            self.assertEqual(m.fuse(det, IMG_LOW, pose, 1.0, 1, 2.5), [None])          # 1.5 s old: stale, rejected
            self.assertEqual(m.fusion_stats["rejected"].get("stale"), 1)
            before = list(m.grid.logodds)
            ids = [m.fuse(det, IMG_LOW, pose, 2.0 + 0.25 * k, 2 + k, 2.1 + 0.25 * k) for k in range(5)]
            self.assertEqual(len({i[0] for i in ids}), 1)                              # one persistent track id
            self.assertEqual(m.fusion_stats["small_rock_hits"] > 0, True)
            k = m.grid.index(2.05, 0.0)
            self.assertEqual(m.grid.state_at(k), OCCUPIED)                             # the 3.5 cm pebble is now mapped
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(m.grid.logodds, before)))   # fusion never lowers a cell
            m.finish()
        cfg_off = MappingConfig(fusion_enabled=False)
        with tempfile.TemporaryDirectory() as d:
            m = Mapper(FakeSensors(IMG_LOW), Path(d), cfg=cfg_off)
            self.assertEqual(m.fuse(det, IMG_LOW, pose, 2.0, 2, 2.1), [None])                 # one id slot per detection
            m.finish()

    def test_confirmed_small_rock_is_protected_even_when_lost(self):
        import tempfile
        from pathlib import Path
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import FakeSensors
        det = [{"xyxy": box_px(2.0, -0.1, 0.1, 0.0, 0.035), "conf": 0.5}]
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        with tempfile.TemporaryDirectory() as d:
            m = Mapper(FakeSensors(IMG_LOW), Path(d))
            m.fuse(det, IMG_LOW, pose, 2.0, 1, 2.1)
            self.assertEqual(m.tracker.tracks[0].status, TENTATIVE)
            self.assertEqual(m._protected_cells(), set())                                # tentative: nothing withheld
            for k in range(1, 5):
                m.fuse(det, IMG_LOW, pose, 2.0 + 0.25 * k, 1 + k, 2.1 + 0.25 * k)
            tr = m.tracker.tracks[0]
            self.assertEqual(tr.status, CONFIRMED)
            self.assertGreater(m.fusion_stats.get("footprint_hits", 0), 0)              # footprint evidence added
            prot = m._protected_cells()
            self.assertIn(m.grid.index(tr.x, tr.y), prot)
            tr.status = LOST                                                             # out of view / missed by YOLO
            self.assertEqual(m._protected_cells(), prot)
            m.finish()
        with tempfile.TemporaryDirectory() as d:
            m = Mapper(FakeSensors(IMG_LOW), Path(d), cfg=MappingConfig(fusion_protect_small_rocks=False))
            for k in range(5):
                m.fuse(det, IMG_LOW, pose, 2.0 + 0.25 * k, 1 + k, 2.1 + 0.25 * k)
            self.assertNotIn("footprint_hits", m.fusion_stats)
            m.finish()


def run_cycles(m, sens, t, n):
    """Stationary rover: ``n`` mapping cycles (capture + grid update) on the fake sensors, from sim time ``t``."""
    done = 0
    while done < n:
        t = round(t + 0.032, 6)
        sens.t = t
        m.update_pose(0.0, 0.0, t)
        if m.due(t):
            m.cycle(t)
            m.integrate_pending()
            done += 1
    return t


class TestReleaseSafety(unittest.TestCase):
    """Release review: small-rock hazards persist within a mission (false, lost and re-detected tracks), only an explicit
    map reset clears them, and depth-based vs YOLO-derived occupancy stays distinguishable (provenance)."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import FLAT, FakeSensors
        self.tmp = tempfile.TemporaryDirectory()
        self.sens = FakeSensors(FLAT)                                  # depth sees only flat ground (no rock above 3 cm)
        self.m = Mapper(self.sens, Path(self.tmp.name))
        self.det = [{"xyxy": box_px(2.0, -0.1, 0.1, 0.0, 0.035), "conf": 0.5}]
        self.pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def tearDown(self):
        if not self.m.tele.closed:
            self.m.finish()
        self.tmp.cleanup()

    def _fuse(self, t, n=1, det=None, img=None):
        for k in range(n):
            ts = t + 0.25 * k
            self.m.fuse(self.det if det is None else det, IMG_LOW if img is None else img, self.pose, ts, int(ts * 100), ts + 0.1)
        return t + 0.25 * n

    def test_persistent_false_track_is_withheld_from_free_until_reset(self):
        from artemis_x.sim.mapping.grid import FREE, SRC_DEPTH, SRC_PROTECTED, UNKNOWN
        from artemis_x.sim.mapping.tracker import Track
        g = self.m.grid
        # a false confirmed small track on bare ground (e.g. a YOLO false positive that once had depth support)
        self.m.tracker.tracks.append(Track(99, "rock", 2.05, 0.0, 0.05, 0.05, 0.03, 0.4, 2, 0, 0.0, 0.0, CONFIRMED))
        t = run_cycles(self.m, self.sens, 0.0, 6)
        k, ctrl = g.index(2.05, 0.0), g.index(2.05, 0.8)
        self.assertEqual(g.state_at(ctrl), FREE)                                    # flat ground beside it is FREE
        self.assertEqual(g.state_at(k), UNKNOWN)                                    # never cleared: not traversable
        self.assertGreater(self.m.costmap.cost_at(2.05, 0.0), 252)
        prov = g.provenance_layer()
        self.assertTrue(prov[k] & SRC_PROTECTED)
        self.assertFalse(prov[k] & SRC_DEPTH)                                       # visibly NOT depth evidence
        self.m.tracker.tracks[0].status = LOST                                      # lost: still persists in the mission
        t = run_cycles(self.m, self.sens, t, 4)
        self.assertEqual(g.state_at(k), UNKNOWN)
        ev = self.m.reset_map("operator: false track")
        self.assertEqual(ev["tracks_dropped"], 1)
        run_cycles(self.m, self.sens, t, 6)
        self.assertEqual(g.state_at(k), FREE)                                       # after the reset, depth clears it
        self.assertFalse(g.provenance_layer()[k] & SRC_PROTECTED)

    def test_lost_then_redetected_rock_persists_and_keeps_its_id(self):
        from artemis_x.sim.mapping.grid import OCCUPIED, SRC_DEPTH, SRC_PROTECTED, SRC_YOLO_FOOTPRINT, SRC_YOLO_POINTS
        g, tr = self.m.grid, self.m.tracker
        t = self._fuse(2.0, 5)
        self.assertEqual(len(tr.tracks), 1)
        rock = tr.tracks[0]
        self.assertEqual(rock.status, CONFIRMED)
        k = g.index(rock.x, rock.y)
        self.assertEqual(g.state_at(k), OCCUPIED)
        t = self._fuse(t, CFG.lost_misses, det=[], img=render_boxes([]))         # in view, not detected -> lost
        self.assertEqual(rock.status, LOST)
        run_cycles(self.m, self.sens, 0.0, 8)                                      # depth (3 cm band) sees only ground there
        self.assertEqual(g.state_at(k), OCCUPIED)                                  # the lost rock is still an obstacle
        prov = g.provenance_layer()
        self.assertTrue(prov[k] & (SRC_YOLO_POINTS | SRC_YOLO_FOOTPRINT))
        self.assertTrue(prov[k] & SRC_PROTECTED)
        self.assertFalse(prov[k] & SRC_DEPTH)
        self.assertEqual(g.occupied_by_source()["depth_only"], 0)
        self.assertGreater(g.occupied_by_source()["yolo_only"], 0)
        self._fuse(t, CFG.confirm_hits)                                            # re-detected
        self.assertEqual(len(tr.tracks), 1)                                        # same track, no duplicate
        self.assertIs(tr.tracks[0], rock)
        self.assertEqual(rock.status, CONFIRMED)

    def test_map_reset_clears_everything_and_mapping_continues(self):
        import json
        from pathlib import Path
        from artemis_x.sim.mapping.costmap import COST_UNKNOWN
        g = self.m.grid
        t = self._fuse(2.0, 5)
        old_id = self.m.tracker.tracks[0].id
        tc = run_cycles(self.m, self.sens, 0.0, 4)
        self.assertGreater(g.counts()["free"], 0)
        self.assertGreater(g.counts()["occupied"], 0)
        self.m.reset_map("new mission")
        self.assertEqual(g.counts()["unknown"], g.n * g.n)
        self.assertEqual(self.m.tracker.tracks, [])
        self.assertEqual(g.protected, set())
        self.assertEqual(set(g.provenance_layer()), {0})
        self.assertEqual(set(self.m.costmap.cost), {COST_UNKNOWN})               # nothing traversable right after a reset
        self.assertIsNone(self.m.pending)
        run_cycles(self.m, self.sens, tc, 3)                                       # mapping resumes from depth only
        self.assertGreater(g.counts()["free"], 0)
        self.assertEqual(g.counts()["occupied"], 0)
        self._fuse(t, 1)
        self.assertGreater(self.m.tracker.tracks[0].id, old_id)                   # ids are never reused
        s = self.m.finish()
        self.assertEqual(s["grid"]["resets"], 1)
        self.assertEqual(s["map_resets"][0]["reason"], "new mission")
        self.assertEqual(s["fusion"]["tracker"]["resets"], 1)
        lines = [json.loads(x) for x in (Path(self.tmp.name) / "mapping.jsonl").read_text().splitlines()]
        self.assertEqual(sum(1 for x in lines if x.get("event") == "map_reset"), 1)

    def test_snapshot_provenance_layer_separates_depth_from_yolo(self):
        from pathlib import Path
        from artemis_x.sim.mapping.grid import SRC_DEPTH, SRC_YOLO_FOOTPRINT, SRC_YOLO_POINTS
        from artemis_x.sim.mapping.snapshot import read_snapshot
        from tests.test_mapping_depth import BOX
        self.sens.depth_img = BOX                                                  # a 20 cm box at x = 2.0: depth obstacle
        self._fuse(2.0, 5)                                                         # and the YOLO pebble at the same range
        run_cycles(self.m, self.sens, 0.0, 3)
        self.m.finish()
        hdr, L = read_snapshot(Path(self.tmp.name) / "snapshots" / "map_final.json")
        prov, st = L["provenance"], L["state"]
        occ = [k for k in range(len(st)) if st[k] >= 2]
        yolo = SRC_YOLO_POINTS | SRC_YOLO_FOOTPRINT
        self.assertTrue([k for k in occ if prov[k] & SRC_DEPTH])
        self.assertTrue([k for k in occ if prov[k] & yolo])
        self.assertTrue(all(prov[k] & (SRC_DEPTH | yolo) for k in occ))           # every hazard cell is attributed
        self.assertIn("provenance", [x["name"] for x in hdr["layers"]])


if __name__ == "__main__":
    unittest.main()
