"""Scene tooling: glTF -> Webots conversion, Poly Haven placement transform, test-course layout invariants."""

import base64
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tests._common import ROOT

WB = ROOT / "artemis-webots"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


fetch = _load("fetch_polyhaven", WB / "tools" / "fetch_polyhaven.py")
mw = _load("make_world", WB / "tools" / "make_world.py")


class TestGltfConversion(unittest.TestCase):
    def test_axes_units_and_resting_on_ground(self):
        # one triangle in glTF Y-up metres: points at y = 0.0 .. 0.2 (height), z = -0.1 .. 0.1 (depth)
        pos = [(0.0, 0.0, 0.1), (0.2, 0.0, -0.1), (0.1, 0.2, 0.0)]
        uv = [(0, 0), (1, 0), (0.5, 1)]
        blob = b"".join(struct.pack("<3f", *p) for p in pos) + b"".join(struct.pack("<2f", *t) for t in uv) + struct.pack("<3H", 0, 1, 2) + b"\0\0"
        g = {"asset": {"version": "2.0"}, "buffers": [{"uri": "t.bin", "byteLength": len(blob)}],
             "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 24},
                             {"buffer": 0, "byteOffset": 60, "byteLength": 6}],
             "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                           {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
                           {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
             "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2}]}],
             "nodes": [{"mesh": 0}]}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.bin").write_bytes(blob)
            (Path(d) / "t.gltf").write_text(json.dumps(g))
            m = fetch.gltf_to_webots(Path(d) / "t.gltf")
            fetch.write_obj(m, Path(d) / "t.obj", "test")
            obj = (Path(d) / "t.obj").read_text()
        P = m["P"]
        self.assertAlmostEqual(P[:, 2].min(), 0.0)                          # rests on z = 0
        self.assertAlmostEqual(P[:, 2].max(), 0.2, places=6)                # glTF +y (up) became Webots +z
        self.assertAlmostEqual(P[:, 1].max() - P[:, 1].min(), 0.2, places=6)  # glTF z extent became Webots y
        self.assertEqual(obj.count("\nf "), 1)
        self.assertIn("vt 0.500000 0.000000", obj)                           # V flipped for OBJ


class TestDecimation(unittest.TestCase):
    def test_vertex_clustering_keeps_corner_uvs(self):
        import numpy as np
        n = 60                                                               # 60 x 60 grid -> 6962 triangles
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        P = np.stack([u.ravel(), v.ravel(), 0.1 * np.sin(3 * u.ravel())], 1)
        T = np.stack([u.ravel(), v.ravel()], 1); N = np.tile([0, 0, 1.0], (len(P), 1))
        q = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)]
        F = np.array([t for a, b, c, d in q for t in ((a, b, c), (a, c, d))])
        m = fetch.decimate({"P": P, "N": N, "T": T, "F": F}, 1000)
        self.assertLessEqual(len(m["F"]), 1000); self.assertGreater(len(m["F"]), 500)
        self.assertEqual(m["decimated_from"], len(F))
        self.assertTrue(all(any((F == fa).all(1)) for fa in m["FA"][:50]))    # surviving triangles keep original corners (UV/normal)
        self.assertAlmostEqual(float(m["P"][:, 2].min()), 0.0)                 # still rests on z = 0
        self.assertEqual(fetch.decimate({"P": P, "N": N, "T": T, "F": F[:10]}, 1000)["F"].shape[0], 10)   # small mesh untouched


class TestTourDriving(unittest.TestCase):
    def test_odometry_straight_and_turn(self):
        from artemis_x.sim import drive
        o = drive.Odometry(); o.update(0, 0); o.update(10, 10)
        self.assertAlmostEqual(o.x, 10 * drive.WHEEL_RADIUS_M); self.assertAlmostEqual(o.yaw, 0)
        o = drive.Odometry(); o.update(0, 0); o.update(-2, 2)
        self.assertAlmostEqual(o.yaw, drive.YAW_SCALE * 4 * drive.WHEEL_RADIUS_M / drive.TRACK_M); self.assertAlmostEqual(o.dist, 0)

    def test_tour_moves_absolute_and_slowdown(self):
        from artemis_x.sim import drive
        mv = drive.tour_moves([{"zone": "A", "x": 1.0, "look_rad": 0.5}, {"zone": "B", "x": 3.0, "look_rad": 0.7}])
        self.assertEqual([m.kind for m in mv], ["drive_to", "turn", "pause", "turn"] * 2 + ["pause"])
        self.assertEqual([m.value for m in mv if m.kind == "drive_to"], [1.0, 3.0])          # absolute lane positions
        self.assertEqual(sum(m.value for m in mv if m.kind == "turn"), 0)                    # always turns back onto the lane
        self.assertTrue(all(m.capture for m in mv if m.label.startswith("observe_")))
        fast, slow = drive.move_command(mv[0], 1.0), drive.move_command(mv[0], 0.001)
        self.assertEqual(fast.left, 4.0); self.assertAlmostEqual(slow.left, 0.6)            # 15 % floor near the target
        right = drive.move_command(mv[3], -0.2)                                              # negative turn = right
        self.assertGreater(right.left, 0); self.assertLess(right.right, 0)
        mv = drive.tour_moves([{"zone": "A", "x": 1.0, "look_rad": 0.5, "approach_m": 0.8}])
        self.assertEqual([(m.kind, m.value) for m in mv], [("drive_to", 1.0), ("turn", 0.5), ("drive", 0.8), ("pause", 2.0),
                                                          ("drive", -0.8), ("turn", -0.5), ("pause", 1.0)])   # out and back


@unittest.skipUnless(mw.polyhaven_available(), "Poly Haven assets not fetched (artemis-webots/tools/fetch_polyhaven.py)")
class TestPolyHavenScenes(unittest.TestCase):
    def test_placement_transform(self):
        m = mw.ph_mesh("moon_rock_02")
        placed = mw.ph_place("moon_rock_02", 2.0, -1.0, math.pi / 2, 4.0)
        xs = [v[0] for v in placed["vertices"]]; ys = [v[1] for v in placed["vertices"]]
        # 90 deg yaw swaps the footprint axes; scale 4
        self.assertAlmostEqual(max(xs) - min(xs), 4 * m["extent_m"][1], places=3)
        self.assertAlmostEqual(max(ys) - min(ys), 4 * m["extent_m"][0], places=3)
        self.assertAlmostEqual(min(v[2] for v in placed["vertices"]), -placed["sink"], places=4)

    def test_course_layout(self):
        lay = mw.course_layout()
        cols = [tuple(o["color"]) for o in lay["rocks"] + lay["distractors"]]
        self.assertEqual(len(cols), len(set(cols)))                          # unique segmentation colours
        self.assertTrue(all(o["class"] == "rock" for o in lay["rocks"]))
        self.assertTrue(all(o["class"] == "non_rock" for o in lay["distractors"]))
        zones = {o["zone"] for o in lay["rocks"] + lay["distractors"]}
        self.assertEqual(len(zones), 9)
        self.assertTrue(zones <= {v["zone"] for v in lay["viewpoints"]})      # every zone has a camera viewpoint
        for a in lay["rocks"]:                                               # no interpenetrating rock footprints except the cluster
            for b in lay["rocks"]:
                if a is not b and a["zone"] != "Z4_occlusion_cluster":
                    self.assertGreater(math.hypot(a["x"] - b["x"], a["y"] - b["y"]), 0.3 * (a["width_m"] + b["width_m"]))

    def test_course_lane_and_tour_plan(self):
        lay = mw.course_layout(); tour = lay["tour"]; lane = tour["lane_y"]
        for r in lay["rocks"]:                                               # nothing within 0.6 m of the lane centre line
            self.assertGreater(abs(r["y"] - lane) - r["width_m"] / 2, 0.6)
        for d in lay["distractors"]:
            self.assertGreater(abs(d["y"] - lane) - 0.5, 0.6)
        obs = mw.course_obstacles(lay["rocks"], lay["distractors"])
        for s in tour["stops"]:                                              # every point of every approach keeps the clearance
            for k in range(int(s["approach_m"] / 0.02) + 1):
                px, py = s["x"] + 0.02 * k * math.cos(s["look_rad"]), lane + 0.02 * k * math.sin(s["look_rad"])
                self.assertGreaterEqual(min(math.hypot(px - x, py - y) - r for x, y, r in obs), mw.ROVER_CLEARANCE_M - 1e-9)
            self.assertLess(s["observe_to_zone_centre_m"], 2.6)              # observes each zone from close by
        xs = [s["x"] for s in tour["stops"]]
        self.assertEqual(xs, sorted(xs)); self.assertEqual(len(xs), 9); self.assertGreater(xs[0], 0)
        self.assertTrue(all(0 < s["look_rad"] <= 1.55 for s in tour["stops"]))
        cd = json.dumps(tour, separators=(",", ":")).replace('"', '\\"')
        node = mw.rover(True, (0, lane, 0), custom_data=cd)
        line = next(l for l in node.splitlines() if l.strip().startswith("customData"))
        self.assertEqual(json.loads(line.strip()[len('customData "'):-1].replace('\\"', '"')), tour)   # Webots string unescape

    def test_terrain_craters_and_rock_heights(self):
        for c in mw.CRATERS:
            self.assertLess(mw.terrain_height(c["x"], c["y"]), -0.8 * c["depth"])            # bowl
            self.assertGreater(mw.terrain_height(c["x"] + c["radius"], c["y"]), 0.6 * c["rim"])  # raised rim
        self.assertLess(abs(mw.terrain_height(5.0, -2.6)), 0.015)                             # gentle relief on the lane
        self.assertAlmostEqual(mw.terrain_height(mw.TERRAIN_EXTENT[1], 0), -(mw.GROUND_DROP_M + 0.02), places=6)
        lay = mw.course_layout()
        for r in lay["rocks"]:
            self.assertAlmostEqual(r["z0"], mw.terrain_height(r["x"], r["y"]), places=3)
            placed = mw.ph_place(r["asset"], r["x"], r["y"], r["yaw"], r["scale"], r["z0"], r["sink_frac"])
            self.assertAlmostEqual(min(v[2] for v in placed["vertices"]), r["z0"] - placed["sink"], places=3)
        self.assertTrue(all(r["sink_frac"] >= 0.45 for r in lay["rocks"] if r["zone"] == "Z9_half_buried"))
        self.assertTrue(all(mw.ph_mesh(a)["triangles"] and len(mw.ph_mesh(a)["triangles"]) <= 21845 for a in mw.COURSE_ASSETS))  # shadows

    def test_r1r2_world_unchanged_by_generator(self):
        import hashlib
        before = hashlib.sha256((WB / "worlds" / "artemis_moon.wbt").read_bytes()).hexdigest()
        info = ('    "Artemis-X simulator v0.1: one Koala rover (Python controller) with a forward RGB camera and an aligned RangeFinder."\n'
                '    "Generated from the official Webots R2025a moon.wbt sample by artemis-webots/tools/make_world.py."')
        import os
        home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
        if not (Path(home) / mw.SAMPLE).exists():
            self.skipTest("Webots sample world not available")
        world = mw.sample_head(home, "Artemis-X lunar rover (v0.1)", info) + mw.ROVER + "".join(mw.rock(*r) for r in mw.ROCKS)
        self.assertEqual(hashlib.sha256(world.encode()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
