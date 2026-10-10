"""R3: protocol, async client failure modes, depth association/projection, and the real inference server."""

import json
import math
import socket
import struct
import threading
import time
import unittest
from pathlib import Path

from tests._common import ROOT

from artemis_x.sim import protocol  # noqa: E402
from artemis_x.sim.client import DetectionClient  # noqa: E402
from artemis_x.sim.projection import CameraModel, camera_to_rover, localize, pixel_to_camera  # noqa: E402

CAM = CameraModel(fx=658.9756, fy=658.9756, cx=359.5, cy=239.5, width=720, height=480, min_range=0.05, max_range=15.0)
SHA = "413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb"


class FakeServer:
    """Scripted local server: behaviour(conn) runs after the hello."""

    def __init__(self, behaviour, sha=SHA):
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.behaviour, self.sha = behaviour, sha
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            conn.sendall(protocol.encode({"type": "hello", "model_sha256": self.sha}))
            try:
                self.behaviour(conn)
            except (OSError, protocol.ConnectionClosed):
                pass

    def close(self):
        self.sock.close()


def _wait(client, n=1, timeout=8.0):
    out, end = [], time.time() + timeout
    while len(out) < n and time.time() < end:
        out += client.poll(); time.sleep(0.01)
    return out


class TestProtocol(unittest.TestCase):
    def test_round_trip_lossless(self):
        a, b = socket.socketpair()
        rgb = bytes(range(256)) * (720 * 480 * 3 // 256)
        threading.Thread(target=lambda: a.sendall(protocol.frame_message(7, 1.25, 720, 480, rgb)), daemon=True).start()
        h, p = protocol.read_message(b)
        protocol.validate_frame(h, p)
        self.assertEqual((h["frame_id"], h["width"], h["height"], p), (7, 720, 480, rgb))
        a.close(); b.close()

    def test_malformed_and_oversized_rejected_before_payload(self):
        for raw in (b"XXXX" + struct.pack("!HII", 1, 2, 0) + b"{}",                                   # bad magic
                    protocol.MAGIC + struct.pack("!HII", 9, 2, 0) + b"{}",                              # bad version
                    protocol.MAGIC + struct.pack("!HII", 1, protocol.MAX_HEADER + 1, 0),                # oversized header
                    protocol.MAGIC + struct.pack("!HII", 1, 2, protocol.MAX_PAYLOAD + 1) + b"{}",       # oversized payload
                    protocol.MAGIC + struct.pack("!HII", 1, 3, 0) + b"{x}"):                            # bad JSON
            a, b = socket.socketpair()
            a.sendall(raw)
            with self.assertRaises(protocol.ProtocolError):
                protocol.read_message(b)
            a.close(); b.close()

    def test_validate_frame(self):
        ok = {"type": "frame", "frame_id": 1, "sim_time_s": 0.0, "width": 2, "height": 2, "encoding": "rgb8"}
        protocol.validate_frame(ok, bytes(12))
        for bad, payload in (({**ok, "encoding": "bgra"}, bytes(12)), ({**ok, "frame_id": -1}, bytes(12)), (ok, bytes(11)),
                             ({**ok, "width": 99999}, bytes(12)), ({k: v for k, v in ok.items() if k != "height"}, bytes(12))):
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_frame(bad, payload)


class TestClientFailureModes(unittest.TestCase):
    def test_server_not_started(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        c = DetectionClient(port=port, connect_timeout_s=0.3, retry_s=0.1).start()
        t0 = time.perf_counter(); c.submit(1, 0.0, 2, 2, bytes(12)); self.assertLess(time.perf_counter() - t0, 0.05)   # non-blocking
        r = _wait(c)
        self.assertFalse(r[0]["ok"]); self.assertEqual(r[0]["error"], "server unavailable")
        self.assertGreaterEqual(c.snapshot()["connect_failures"], 1)
        c.close()

    def test_server_terminates_during_inference(self):
        def die(conn):
            protocol.read_message(conn)          # read the frame, then vanish
        srv = FakeServer(die)
        c = DetectionClient(port=srv.port, retry_s=0.1).start()
        c.submit(3, 0.0, 2, 2, bytes(12))
        r = _wait(c)
        self.assertFalse(r[0]["ok"]); self.assertIn("ConnectionClosed", r[0]["error"])
        c.close(); srv.close()

    def test_timeout(self):
        hold = threading.Event()

        def silent(conn):
            protocol.read_message(conn); hold.wait(5)
        srv = FakeServer(silent)
        c = DetectionClient(port=srv.port, request_timeout_s=0.3).start()
        c.submit(4, 0.0, 2, 2, bytes(12))
        r = _wait(c)
        self.assertEqual(r[0]["error"], "timeout"); self.assertEqual(c.snapshot()["timeouts"], 1)
        hold.set(); c.close(); srv.close()

    def test_wrong_frame_id_is_never_attached(self):
        def wrong_then_right(conn):
            h, _ = protocol.read_message(conn)
            conn.sendall(protocol.encode({"type": "detections", "frame_id": h["frame_id"] + 100, "detections": [{"xyxy": [0, 0, 1, 1], "conf": .9}]}))
            conn.sendall(protocol.encode({"type": "detections", "frame_id": h["frame_id"], "detections": []}))
            time.sleep(0.5)
        srv = FakeServer(wrong_then_right)
        c = DetectionClient(port=srv.port).start()
        c.submit(5, 0.0, 2, 2, bytes(12))
        r = _wait(c)
        self.assertEqual((r[0]["frame_id"], r[0]["detections"]), (5, []))
        self.assertEqual(c.snapshot()["mismatched"], 1)
        c.close(); srv.close()

    def test_unexpected_model_sha_refused(self):
        srv = FakeServer(lambda conn: time.sleep(0.5), sha="0" * 64)
        c = DetectionClient(port=srv.port, expected_sha256=SHA, retry_s=0.1).start()
        c.submit(1, 0.0, 2, 2, bytes(12))
        r = _wait(c)
        self.assertFalse(r[0]["ok"]); self.assertIn("sha256", c.snapshot()["last_error"])
        c.close(); srv.close()

    def test_backpressure_drops_old_frames(self):
        def slow(conn):
            while True:
                h, _ = protocol.read_message(conn)
                if h["type"] != "frame":                 # e.g. the client's clean 'bye'
                    return
                time.sleep(0.3)
                conn.sendall(protocol.encode({"type": "detections", "frame_id": h["frame_id"], "detections": []}))
        srv = FakeServer(slow)
        c = DetectionClient(port=srv.port).start()
        for i in range(10):
            c.submit(i, i * 0.01, 2, 2, bytes(12)); time.sleep(0.02)
        r = _wait(c, n=2, timeout=3)
        s = c.snapshot()
        self.assertGreater(s["dropped"], 0)
        self.assertLessEqual(s["sent"], 3)                       # never an unbounded queue
        self.assertIn(9, [x["frame_id"] for x in r] + [x["frame_id"] for x in _wait(c, 1, 2)])   # newest frame wins
        c.close(); srv.close()

    def test_clean_shutdown(self):
        got_bye = threading.Event()

        def wait_bye(conn):
            h, _ = protocol.read_message(conn)
            if h["type"] == "frame":
                conn.sendall(protocol.encode({"type": "detections", "frame_id": h["frame_id"], "detections": []}))
                h, _ = protocol.read_message(conn)
            if h["type"] == "bye":
                got_bye.set()
        srv = FakeServer(wait_bye)
        c = DetectionClient(port=srv.port).start()
        c.submit(1, 0.0, 2, 2, bytes(12)); _wait(c)
        c.close()
        self.assertFalse(c._thread.is_alive()); self.assertTrue(got_bye.wait(2))
        srv.close()


class TestAssociationAndProjection(unittest.TestCase):
    def test_transforms(self):
        x, y, z = pixel_to_camera(CAM.cx, CAM.cy, 2.0, CAM)
        self.assertEqual((x, y, z), (2.0, 0.0, 0.0))
        self.assertGreater(pixel_to_camera(0, CAM.cy, 2.0, CAM)[1], 0)            # left of centre -> +y
        r = camera_to_rover((1.0, 0.0, 0.0), CAM)
        self.assertAlmostEqual(r[0], 0.16 + math.cos(0.12)); self.assertAlmostEqual(r[2], 0.24 - math.sin(0.12))   # pitched down

    def test_invalid_depth_is_rejected_not_invented(self):
        depth = [float("inf")] * (720 * 480)
        loc = localize({"xyxy": [300, 200, 400, 280], "conf": 0.9}, depth, CAM)
        self.assertFalse(loc["valid"]); self.assertEqual(loc["reason"], "insufficient valid depth in box")
        self.assertFalse(localize({"xyxy": [0, 0, 10, 10], "conf": .5}, None, CAM)["valid"])
        depth = [float("nan")] * (720 * 480)
        self.assertFalse(localize({"xyxy": [300, 200, 400, 280], "conf": 0.9}, depth, CAM)["valid"])

    def test_foreground_surface_point(self):
        depth = [5.0] * (720 * 480)                                              # background plane at 5 m
        for v in range(220, 261):
            for u in range(340, 381):
                depth[v * 720 + u] = 2.0                                         # object at 2 m
        loc = localize({"xyxy": [330, 210, 390, 270], "conf": 0.8}, depth, CAM)
        self.assertTrue(loc["valid"]); self.assertAlmostEqual(loc["surface_point_camera_m"][0], 2.0, places=3)

    def test_stale_and_evicted_results(self):
        from artemis_x.sim.perception import InferenceLink

        class FakeSensors:
            camera = depth = object()
            def describe(self):
                return {"depth": {"fx": CAM.fx, "fy": CAM.fy, "cx": CAM.cx, "cy": CAM.cy, "width": 720, "height": 480,
                                  "min_range_m": .05, "max_range_m": 15}}

        class FakeClient:
            def snapshot(self):
                return {}
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            link = InferenceLink(FakeSensors(), FakeClient(), Path(d), max_age_s=1.0)
            link.ring[1] = (0.5, [3.0] * (720 * 480), {"x": 0, "y": 0, "z": 0, "yaw": 0}, bytes(720 * 480 * 3), 720, 480)
            fresh = link.handle({"frame_id": 1, "sim_time_s": 0.5, "ok": True, "detections": [{"xyxy": [10, 10, 60, 60], "conf": .7}], "latency_ms": 5}, 0.8)
            stale = link.handle({"frame_id": 1, "sim_time_s": 0.5, "ok": True, "detections": [], "latency_ms": 5}, 3.0)
            evicted = link.handle({"frame_id": 99, "sim_time_s": 2.9, "ok": True, "detections": [], "latency_ms": 5}, 3.0)
            link.finish()
            self.assertFalse(fresh["stale"]); self.assertTrue(fresh["detections"][0]["valid"])
            self.assertTrue(stale["stale"]); self.assertIn("evicted", evicted["association"])


class TestRealServer(unittest.TestCase):
    """V2 parity + server-side error handling (needs the ML environment; validation images only)."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            from artemis_x.sim.inference_server import InferenceServer
        except ImportError as e:
            raise unittest.SkipTest(f"ML environment not available: {e}")
        cls.srv = InferenceServer(port=0)
        cls.th = threading.Thread(target=cls.srv.serve_forever, daemon=True); cls.th.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop(); cls.th.join(3)

    def _conn(self):
        s = socket.create_connection(("127.0.0.1", self.srv.port), timeout=30)
        hello, _ = protocol.read_message(s)
        return s, hello

    def test_hello_reports_frozen_model(self):
        s, hello = self._conn()
        self.assertEqual(hello["model_sha256"], SHA)
        self.assertEqual(hello["inference"], {"imgsz": 736, "conf": 0.2, "nms_iou": 0.6, "max_det": 300, "min_box_area_px2": 50.0})
        s.close()

    def test_parity_with_rockdetector_on_validation_images(self):
        import numpy as np
        from PIL import Image
        from artemis_x.config import paths
        if not paths.local_dataset_available():
            self.skipTest("validation images not present")
        s, _ = self._conn()
        for i, name in enumerate(("render0115.png", "render5589.png", "render0575.png")):
            p = paths.YOLO_DATASET / "images" / "val" / name
            rgb = np.asarray(Image.open(p).convert("RGB"))
            s.sendall(protocol.frame_message(i, 0.0, rgb.shape[1], rgb.shape[0], rgb.tobytes()))
            h, _ = protocol.read_message(s)
            ref = self.srv.det.predict(p)                                     # file path -> Ultralytics' own loader
            got = sorted((tuple(d["xyxy"]), d["conf"]) for d in h["detections"])
            exp = sorted((tuple(round(v, 2) for v in (d.x0, d.y0, d.x1, d.y1)), round(d.confidence, 5)) for d in ref.detections)
            self.assertEqual(len(got), len(exp), name)
            self.assertEqual(got, exp, name)
        s.close()

    def test_malformed_frame_gets_error_and_server_survives(self):
        s, _ = self._conn()
        s.sendall(protocol.encode({"type": "frame", "frame_id": 1, "sim_time_s": 0, "width": 4, "height": 4, "encoding": "rgb8"}, bytes(5)))
        h, _ = protocol.read_message(s)
        self.assertEqual(h["type"], "error")
        s.sendall(protocol.frame_message(2, 0.0, 4, 4, bytes(48)))
        h, _ = protocol.read_message(s)
        self.assertEqual((h["type"], h["frame_id"]), ("detections", 2))
        s.sendall(protocol.encode({"type": "bye"})); s.close()


if __name__ == "__main__":
    unittest.main()
