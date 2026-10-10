"""R4.5 display part: the live monitoring panels (LOCAL MAP, DEPTH SENSOR, RGB / AI headers) on fake Webots displays.
They must show the live mapping state truthfully: UNKNOWN never as FREE, inflation and negative hazards, depth vs YOLO
provenance, rover at the ESTIMATED pose, a map reset clears the picture, and missing YOLO never stops depth mapping."""

import math
import tempfile
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.sim import monitor as mon
from artemis_x.sim.mapping.grid import FREE, OCCUPIED, SRC_DEPTH, SRC_YOLO_FOOTPRINT
from artemis_x.sim.mapping.mapper import Mapper
from artemis_x.sim.mapping.tracker import CONFIRMED, LOST, Track
from tests.test_mapping_depth import BOX, FLAT, PIT, FakeSensors
from tests.test_mapping_fusion import run_cycles


class FakeDisplay:
    RGB, BGRA = 3, 6

    def __init__(self, name="d"):
        self.name, self.ops, self.images, self.attached, self.saved = name, [], {}, None, []

    def getName(self):
        return self.name

    def imageNew(self, data, fmt, w, h):
        assert len(data) == w * h * 3 and fmt == self.RGB, (len(data), w, h)
        k = len(self.images) + 1
        self.images[k] = (bytes(data), w, h)
        return k

    def imagePaste(self, k, x, y, blend):
        self.ops.append(("paste", k, x, y))

    def imageDelete(self, k):
        pass

    def imageSave(self, k, path):
        self.saved.append(path)

    def attachCamera(self, cam):
        self.attached = cam

    def texts(self):
        return [o[1] for o in self.ops if o[0] == "text"]

    def __getattr__(self, name):          # setColor, fillRectangle, drawLine, drawText, ... are recorded
        def rec(*a):
            self.ops.append(("text", a[0]) if name == "drawText" else (name, *a))
        return rec


class FakeRobot:
    def __init__(self, devices):
        self.devices = devices

    def getNumberOfDevices(self):
        return len(self.devices)

    def getDeviceByIndex(self, i):
        return self.devices[i]


def pixel(disp, img_id, px, py):
    data, w, _ = disp.images[img_id]
    o = (py * w + px) * 3
    return (data[o] << 16) | (data[o + 1] << 8) | data[o + 2]


def last_map_image(d):
    return [o for o in d.ops if o[0] == "paste"][-1][1]


class MonitorCase(unittest.TestCase):
    def make(self, depth=FLAT, inference=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sens = FakeSensors(depth)
        self.sens.camera = object()
        self.m = Mapper(self.sens, Path(self.tmp.name))
        self.addCleanup(lambda: None if self.m.tele.closed else self.m.finish())
        if inference is not None:
            self.m.set_inference_status(inference)
        self.disp = {n: FakeDisplay(n) for n in ("rgb_view", "yolo_view", "depth_view", "map_view")}
        self.mon = mon.Monitor(FakeRobot(list(self.disp.values())), self.sens, self.m)
        return self.m

    def render_map(self, t=10.0):
        self.mon.map.begin(self.m, t)
        while not self.mon.map.step():
            pass
        return self.mon.map.last

    def cell_px(self, jb, x, y):
        cx, cy = self.mon.map.to_px(jb, x, y)
        return int(cx) - self.mon.map.MAP_X, int(cy) - self.mon.map.MAP_Y


class TestMapPanel(MonitorCase):
    def test_unknown_is_never_drawn_as_free(self):
        m = self.make(FLAT)
        run_cycles(m, self.sens, 0.0, 4)
        jb = self.render_map()
        img = last_map_image(self.disp["map_view"])
        free_c, unk_c = mon.CELL_RGB[mon.C_FREE], mon.CELL_RGB[mon.C_UNKNOWN]
        g = m.grid
        seen_free = seen_unknown = 0
        for j in range(0, g.n, 3):
            for i in range(0, g.n, 3):
                k = j * g.n + i
                x, y = g.cell_centre(k)
                c = pixel(self.disp["map_view"], img, *self.cell_px(jb, x, y))
                st = g.state_at(k)
                if st == FREE:
                    seen_free += 1
                    self.assertIn(c, (free_c, mon.CELL_RGB[mon.C_BAND], mon.CELL_RGB[mon.C_NOGO]))
                elif st != OCCUPIED:
                    seen_unknown += 1
                    self.assertNotEqual(c, free_c)                          # UNKNOWN cells never look FREE
                    self.assertEqual(c, unk_c)
        self.assertGreater(seen_free, 20)
        self.assertGreater(seen_unknown, 100)

    def test_obstacles_inflation_negative_and_provenance(self):
        m = self.make(BOX)
        run_cycles(m, self.sens, 0.0, 4)
        g = m.grid
        k_neg = g.index(-3.0, -3.0)                                         # a crater-bowl cell (negative hit), synthetic
        g.logodds[k_neg], g.neg[k_neg], g.src[k_neg] = 3.0, 1, SRC_DEPTH
        k_y = g.index(-3.0, 3.0)                                            # a YOLO-protected small rock, synthetic
        g.logodds[k_y], g.src[k_y] = 2.0, SRC_YOLO_FOOTPRINT
        k_dy = g.index(2.05, 0.0)                                           # box face: depth AND YOLO evidence
        g.src[k_dy] |= SRC_YOLO_FOOTPRINT
        m.costmap.rebuild(1.0, (0.0, 0.0), (0.0, 0.0))
        jb = self.render_map()
        d, img = self.disp["map_view"], last_map_image(self.disp["map_view"])
        col = lambda x, y: pixel(d, img, *self.cell_px(jb, *g.cell_centre(g.index(x, y))))   # sample the cell's centre
        self.assertEqual(col(-3.0, -3.0), mon.CELL_RGB[mon.C_NEG])
        self.assertEqual(col(-3.0, 3.0), mon.CELL_RGB[mon.C_OCC_Y])
        self.assertEqual(col(2.05, 0.0), mon.CELL_RGB[mon.C_OCC_DY])
        self.assertEqual(col(2.05, 0.15), mon.CELL_RGB[mon.C_OCC_D])     # box face: depth only
        self.assertEqual(col(1.85, 0.0), mon.CELL_RGB[mon.C_NOGO])       # free ground inside the collision radius
        band = [col(1.6 - 0.1 * s, 0.0) for s in range(4)]               # graded band further out
        self.assertIn(mon.CELL_RGB[mon.C_BAND], band)

    def test_rover_tracks_and_sidebar_follow_the_estimated_pose(self):
        m = self.make(FLAT)
        run_cycles(m, self.sens, 0.0, 2)
        m.tracker.tracks += [Track(7, "rock", 2.0, 0.5, 0.05, 0.1, 0.05, 0.63, 5, 0, 0.0, 0.0, CONFIRMED),
                             Track(9, "rock", 3.0, -0.5, 0.05, 0.1, 0.05, 0.41, 5, 4, 0.0, 0.0, LOST)]
        jb = self.render_map()
        d = self.disp["map_view"]
        p = m.estimator.pose()
        ovals = [o for o in d.ops if o[0] == "drawOval"]
        rx, ry = self.mon.map.to_px(jb, p.x, p.y)
        self.assertIn((int(rx), int(ry)), [(o[1], o[2]) for o in ovals])  # rover footprint at the odometry estimate
        t = d.texts()
        self.assertIn(f"x {p.x:+6.2f}  y {p.y:+6.2f} m", t)
        self.assertIn("7:0.63", t)
        self.assertNotIn("9:0.41", t)                                       # lost: grey circle, no label
        self.assertTrue(any("cycle" in s and "Hz" in s for s in t))
        self.assertTrue(any(s.startswith("depth age") for s in t))
        self.assertIn("LOCAL MAP", t)

    def test_fov_lines_stay_inside_the_map(self):
        m = self.make(FLAT)
        run_cycles(m, self.sens, 0.0, 2)
        self.render_map()
        mp = self.mon.map
        for o in self.disp["map_view"].ops:
            if o[0] == "drawLine" and o[2] > mp.MAP_Y and o[1] > mp.MAP_X and o[1] < mp.MAP_X + mp.MAP_S - 1:
                self.assertLessEqual(max(o[1], o[3]), mp.MAP_X + mp.MAP_S)

    def test_map_reset_clears_the_picture(self):
        m = self.make(BOX)
        run_cycles(m, self.sens, 0.0, 4)
        m.tracker.tracks.append(Track(3, "rock", 2.0, 0.0, 0.05, 0.1, 0.05, 0.5, 5, 0, 0.0, 0.0, CONFIRMED))
        self.render_map()
        m.reset_map("test")
        self.mon.tick(20.0)                                                 # notices the reset: new frame due now
        for k in range(20):
            self.mon.tick(20.0 + 0.032 * k)
        jb = self.mon.map.last
        self.assertEqual(jb["resets"], 1)
        self.assertEqual(jb["tracks"], [])
        data, w, h = self.disp["map_view"].images[last_map_image(self.disp["map_view"])]
        unk = mon._rgb(mon.CELL_RGB[mon.C_UNKNOWN])
        self.assertEqual(set(data[i:i + 3] for i in range(0, len(data), 3 * 97)), {unk})
        self.assertIn("MAP RESET", self.disp["map_view"].texts())

    def test_work_is_chunked(self):
        m = self.make(FLAT)
        run_cycles(m, self.sens, 0.0, 2)
        mp = self.mon.map
        mp.begin(m, 1.0)
        steps = 0
        while not mp.step():
            steps += 1
        self.assertEqual(steps + 1, math.ceil(m.grid.n / mp.rows_per_chunk))


class TestOtherPanels(MonitorCase):
    def test_depth_panel_colours(self):
        m = self.make(PIT)
        frame, _ = m.cycle(0.256)
        self.mon.on_frame(frame)
        dr = self.mon.depth
        dr.begin(*self.mon._depth_src)
        while not dr.step():
            pass
        d = self.disp["depth_view"]
        data, w, h = d.images[[o for o in d.ops if o[0] == "paste"][-1][1]]
        self.assertEqual((w, h), (720, 480))
        self.assertEqual(data[0:3], b"\x00\x00\x00")                       # top-left: sky, no return -> black
        near = data[(470 * 720 + 360) * 3:(470 * 720 + 360) * 3 + 3]        # bottom centre: ground about 0.5 m away
        lut = mon.depth_lut()
        self.assertIn(near, lut[:40])
        self.assertIn("DEPTH SENSOR", d.texts())

    def test_rgb_panel_attaches_the_camera_and_titles(self):
        self.make(FLAT)
        self.assertIs(self.disp["rgb_view"].attached, self.sens.camera)
        self.mon._headers(1.0)
        self.assertIn("RGB CAMERA", self.disp["rgb_view"].texts())
        for n, title in (("yolo_view", "AI DETECTION"), ("depth_view", "DEPTH SENSOR"), ("map_view", "LOCAL MAP")):
            self.assertIn(title, self.disp[n].texts())

    def test_missing_yolo_does_not_stop_depth_mapping(self):
        m = self.make(BOX, inference={"enabled": True, "connected": False, "last_ok_sim_s": None})
        t = 0.0
        for k in range(40):                                                 # the controller's step loop, YOLO absent
            t = round(t + 0.032, 6)
            self.sens.t = t
            m.update_pose(0.0, 0.0, t)
            if m.due(t):
                frame, _ = m.cycle(t)
                self.mon.on_frame(frame)
            elif m.pending is not None:
                m.integrate_pending()
            else:
                m.idle_step(t)
                self.mon.tick(t)
        self.assertGreater(m.stats["cycles"], 3)
        self.assertGreater(m.grid.counts()["occupied"], 0)
        self.assertGreater(self.mon.stats["map_frames"], 0)
        self.assertIn("YOLO unavailable", self.disp["yolo_view"].texts())
        self.assertIn("YOLO OFFLINE - depth only", self.disp["map_view"].texts())

    def test_font_is_set_at_most_once_per_display(self):
        """Regression (R4.5 memory investigation): every font CHANGE on a Webots R2025a Display leaks native memory in the
        simulator process (>1 GB/s with a few changes per frame). Many frames of every panel -> at most one setFont each."""
        from artemis_x.sim.perception import draw_detections
        m = self.make(BOX, inference={"enabled": True, "connected": False})
        run_cycles(m, self.sens, 0.0, 3)
        frame, _ = m.cycle(5.12)
        self.mon.on_frame(frame)
        det = [{"xyxy": [100, 200, 160, 260], "conf": 0.8, "valid": True, "range_m": 2.0},
               {"xyxy": [300, 40, 340, 80], "conf": 0.3, "valid": False}]
        for k in range(200):
            self.mon.tick(10.0 + 0.1 * k)
            if k % 5 == 0:
                draw_detections(self.disp["yolo_view"], b"\0" * (720 * 480 * 3), 720, 480, det, k, 20.0)
        self.assertGreater(self.mon.stats["map_frames"], 3)
        self.assertGreater(self.mon.stats["depth_frames"], 3)
        for d in self.disp.values():
            self.assertLessEqual(sum(1 for o in d.ops if o[0] == "setFont"), 1, d.name)

    def test_create_only_with_panels_and_opt_out(self):
        import os
        self.make(FLAT)
        self.assertIsNone(mon.Monitor.create(FakeRobot([FakeDisplay("yolo_view")]), self.sens, self.m))   # R3 course world
        os.environ["ARTEMIS_MONITOR"] = "0"
        try:
            self.assertIsNone(mon.Monitor.create(FakeRobot(list(self.disp.values())), self.sens, self.m))
        finally:
            del os.environ["ARTEMIS_MONITOR"]
        self.assertIsNotNone(mon.Monitor.create(FakeRobot(list(self.disp.values())), self.sens, self.m))


if __name__ == "__main__":
    unittest.main()
