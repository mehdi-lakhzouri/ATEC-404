"""R5.2: UDP transport over real localhost sockets - delivery and decoding, receiver absent / restarting (never fatal),
gap / duplicate / late / invalid accounting, change-only snapshots, and the controller glue (LiveTelemetry)."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.telemetry import protocol as P
from artemis_x.telemetry.transport import GroundReceiver, LiveTelemetry, UdpSender
from tests.test_telemetry import FIX


def free_receiver(out=None):
    rx = GroundReceiver("127.0.0.1", 0, out, timeout_s=0.05)
    return rx, rx.sock.getsockname()[1]


def pump(rx, until, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not until():
        rx.receive_once()


class TestUdp(unittest.TestCase):
    def test_delivery_and_decoding_over_udp(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rx.jsonl"
            rx, port = free_receiver(out)
            tx = UdpSender("127.0.0.1", port)
            pkts = [P.encode_pose(9, 0, FIX["pose"])] + P.encode_obstacles(9, 1, FIX) + [P.encode_event(9, 3, 1.0, "health", "STOP", ["stale_corridor"])]
            for p in pkts:
                self.assertTrue(tx.send(p))
            pump(rx, lambda: rx.stats["valid"] >= len(pkts))
            st = rx.close()
            tx.close()
            self.assertEqual((st["valid"], st["invalid"], st["gaps"], st["duplicates"]), (4, 0, 0, 0))
            lines = [json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual([x["seq"] for x in lines], [0, 1, 2, 3])
            self.assertTrue(all(x["rx_wall_s"] > 1e9 for x in lines))      # reception time recorded (wall clock)
            self.assertAlmostEqual(lines[0]["body"]["x"], FIX["pose"]["x"], delta=0.0005)
            ids = [o["id"] for x in lines if x["type"] == P.T_OBSTACLES for o in x["body"]["obstacles"]]
            self.assertEqual(ids, [o["id"] for o in FIX["obstacles"]])

    def test_receiver_absent_is_never_fatal(self):
        _, port = free_receiver()                                       # bind then close: nothing listens on this port
        GroundReceiver.close(_)
        tx = UdpSender("127.0.0.1", port)
        for k in range(30):
            tx.send(P.encode_pose(1, k, FIX["pose"]))                    # must not raise
        self.assertEqual(tx.stats["sent"] + tx.stats["errors"], 30)
        tx.close()

    def test_receiver_restart(self):
        rx1, port = free_receiver()
        tx = UdpSender("127.0.0.1", port)
        for k in range(5):
            tx.send(P.encode_pose(1, k, FIX["pose"]))
        pump(rx1, lambda: rx1.stats["valid"] >= 5)
        rx1.close()
        for k in range(5, 10):                                            # receiver down: lost, not fatal
            tx.send(P.encode_pose(1, k, FIX["pose"]))
        rx2 = GroundReceiver("127.0.0.1", port, timeout_s=0.05)          # restarted on the same port
        for k in range(10, 15):
            tx.send(P.encode_pose(1, k, FIX["pose"]))
        pump(rx2, lambda: rx2.stats["valid"] >= 5)
        st = rx2.close()
        tx.close()
        self.assertEqual(st["valid"], 5)
        self.assertEqual(st["sessions"]["1"]["last_seq"], 14)

    def test_gap_duplicate_late_invalid_accounting(self):
        rx, _ = free_receiver()
        for seq in (0, 1, 4, 4, 3, 5):
            rx.handle(P.encode_pose(2, seq, FIX["pose"]), 100.0)
        good = P.encode_pose(2, 6, FIX["pose"])
        rx.handle(good[:-1] + bytes([good[-1] ^ 1]), 100.0)              # CRC error
        rx.handle(b"garbage", 100.0)                                     # truncated
        st = rx.close()
        self.assertEqual((st["valid"], st["gaps"], st["duplicates"], st["reordered"], st["invalid"]), (6, 1, 1, 1, 2))   # a duplicate is valid, flagged
        self.assertEqual(st["missing"], 1)                                # seq 2 never arrived (3 came late)


class TestLiveGlue(unittest.TestCase):
    def test_live_telemetry_with_mapper(self):
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import BOX, FakeSensors
        from tests.test_mapping_fusion import run_cycles
        with tempfile.TemporaryDirectory() as d:
            rx, port = free_receiver(Path(d) / "rx.jsonl")
            sens = FakeSensors(BOX)
            m = Mapper(sens, Path(d) / "map")
            t = run_cycles(m, sens, 0.0, 4)
            lt = LiveTelemetry(m, "127.0.0.1", port, session_id=77, pose_period_s=0.5, obstacle_period_s=1.0, full_refresh_s=5.0)
            stop = threading.Event()
            th = threading.Thread(target=rx.run, kwargs={"stop": stop}, daemon=True)
            th.start()
            for k in range(60):                                          # 6 s of idle ticks, map unchanged
                lt.tick(t + 0.1 * k)
            s = lt.finish(t + 6.0, Path(d))
            time.sleep(0.3)
            stop.set()
            th.join()
            st = rx.stats
            a = s["adapter"]
            self.assertEqual(s["sender"]["errors"], 0)
            self.assertEqual(st["valid"], s["sender"]["sent"])            # every datagram arrived and decoded
            self.assertEqual((st["gaps"], st["duplicates"], st["invalid"]), (0, 0, 0))
            self.assertGreaterEqual(st["by_type"]["pose"], 10)
            self.assertGreater(a["unchanged_skipped"], 0)                  # unchanged map -> obstacles not re-sent ...
            self.assertGreaterEqual(a["full_snapshots"], 3)               # ... except first, 5 s refresh and the final one
            log = json.loads((Path(d) / "telemetry_send_log.json").read_text())
            self.assertEqual(len(log["rows"]), s["sender"]["sent"])
            m.finish()

    def test_from_env_is_opt_in(self):
        import os
        os.environ.pop("ARTEMIS_TELEMETRY", None)
        self.assertIsNone(LiveTelemetry.from_env(object()))


if __name__ == "__main__":
    unittest.main()
