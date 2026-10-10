"""R4.6: the stable obstacle output contract (artemis.obstacles.v1) - fields, types, sources, ids, reset, no ground truth."""

import json
import tempfile
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim.mapping.contract import SCHEMA, export_obstacles
from artemis_x.sim.mapping.mapper import Mapper
from artemis_x.sim.mapping.tracker import CONFIRMED, LOST, TENTATIVE, Track
from tests.test_mapping_depth import BOX, PIT, FakeSensors
from tests.test_mapping_fusion import run_cycles

FIELDS = {"id", "type", "x_m", "y_m", "radius_m", "confidence", "uncertainty_m", "timestamp_s", "source", "status", "protected"}


class TestObstacleContract(unittest.TestCase):
    def mapper(self, depth):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sens = FakeSensors(depth)
        m = Mapper(sens, Path(tmp.name))
        self.addCleanup(lambda: None if m.tele.closed else m.finish())
        return m, sens, Path(tmp.name)

    def test_depth_box_is_one_obstacle_with_all_fields(self):
        m, sens, _ = self.mapper(BOX)
        run_cycles(m, sens, 0.0, 4)
        doc = export_obstacles(m)
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(doc["frame"], "odom")
        json.dumps(doc)                                                    # serialisable as is
        dep = [o for o in doc["obstacles"] if o["source"] == "depth"]
        self.assertEqual(len(dep), 1)                                      # the box face is one connected cluster
        o = dep[0]
        self.assertTrue(FIELDS <= set(o))
        self.assertEqual(o["type"], "obstacle")
        self.assertTrue(o["id"].startswith("dep-"))
        self.assertAlmostEqual(o["x_m"], 2.05, delta=0.1)                  # box front face at x = 2.0 (cells 2.0-2.1)
        self.assertAlmostEqual(o["y_m"], 0.0, delta=0.1)
        self.assertGreater(o["radius_m"], 0.25)                            # 0.6 m wide face
        self.assertTrue(0.5 < o["confidence"] <= 1.0)
        self.assertGreater(o["uncertainty_m"], 0.0)
        self.assertGreater(o["timestamp_s"], 0.0)
        self.assertEqual(export_obstacles(m)["obstacles"][0]["id"], o["id"])   # deterministic for the same map

    def test_pit_is_a_negative_hazard(self):
        m, sens, _ = self.mapper(PIT)
        run_cycles(m, sens, 0.0, 4)
        types = {o["type"] for o in export_obstacles(m)["obstacles"]}
        self.assertEqual(types, {"negative_hazard"})

    def test_tracks_exported_with_status_and_tentative_excluded(self):
        m, sens, _ = self.mapper(BOX)
        run_cycles(m, sens, 0.0, 4)
        m.tracker.tracks += [Track(7, "rock", 2.05, 0.0, 0.04, 0.3, 0.2, 0.81, 9, 0, 0.0, 1.5, CONFIRMED),
                             Track(8, "rock", 3.0, 1.0, 0.06, 0.05, 0.03, 0.40, 9, 4, 0.0, 1.2, LOST),
                             Track(9, "rock", 4.0, 0.0, 0.09, 0.05, 0.03, 0.30, 1, 0, 0.0, 1.0, TENTATIVE)]
        obs = {o["id"]: o for o in export_obstacles(m)["obstacles"]}
        self.assertNotIn("trk-9", obs)
        t7, t8 = obs["trk-7"], obs["trk-8"]
        self.assertEqual((t7["source"], t7["status"], t7["type"]), ("yolo+depth", "confirmed", "rock"))
        self.assertEqual(t8["status"], "lost")
        self.assertEqual((t7["confidence"], t7["uncertainty_m"], t7["timestamp_s"]), (0.81, 0.04, 1.5))
        self.assertFalse(t7["protected"])                                  # 0.3 m radius > small-rock limit
        self.assertTrue(t8["protected"])
        dep = [o for o in obs.values() if o["source"] == "depth"][0]
        self.assertIn("trk-7", dep["track_ids"])                           # same physical object, linked

    def test_reset_empties_the_list_and_finish_writes_the_file(self):
        m, sens, out = self.mapper(BOX)
        run_cycles(m, sens, 0.0, 4)
        self.assertTrue(export_obstacles(m)["obstacles"])
        m.reset_map("test")
        doc = export_obstacles(m)
        self.assertEqual(doc["obstacles"], [])
        self.assertEqual(doc["map_resets"], 1)
        s = m.finish()
        f = json.loads((out / "obstacles_final.json").read_text())
        self.assertEqual(f["schema"], SCHEMA)
        self.assertIn("ms", s["obstacles_export"])

    def test_no_ground_truth_in_the_contract(self):
        import inspect
        from artemis_x.sim.mapping import contract
        src = inspect.getsource(contract)
        for word in ("gt_pose", "Supervisor", "getSelf", "ground_truth"):
            self.assertNotIn(word, src.replace("ground truth is never used", ""))


if __name__ == "__main__":
    unittest.main()
