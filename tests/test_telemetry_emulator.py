"""R5.3: lunar network emulator - integrity at 0 % loss, statistical loss rates, total loss, latency / jitter, bandwidth,
queue limits, live preset switching over the control port, blackout and restoration, determinism and clean shutdown."""

import math
import socket
import threading
import time
import unittest

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.telemetry import protocol as P
from artemis_x.telemetry.emulator import PRESETS, LinkConfig, LunarLinkEmulator, send_control
from tests.test_telemetry import FIX

POSE = [P.encode_pose(5, k, {**FIX["pose"], "x": 0.001 * k}) for k in range(4000)]
BIG = P.encode_obstacles(5, 0, FIX)[0]                     # 1,066 B (40 records)


class Rig:
    """Emulator on ephemeral ports + a raw ground socket; the emulator loop runs in a thread."""

    def __init__(self, cfg=None, preset="NORMAL", seed=1, control=True):
        self.ground = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ground.bind(("127.0.0.1", 0))
        self.ground.settimeout(0.05)
        self.em = LunarLinkEmulator(("127.0.0.1", 0), self.ground.getsockname(), ("127.0.0.1", 0) if control else None,
                                    config=cfg, preset=preset, seed=seed)
        self.port = self.em.rx.getsockname()[1]
        self.ctl = self.em.ctl.getsockname() if control else None
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.em.run, args=(self.stop,), daemon=True)
        self.th.start()
        self.got = []

    def send(self, pkts, gap_s=0.0):
        for p in pkts:
            self.tx.sendto(p, ("127.0.0.1", self.port))
            if gap_s:
                time.sleep(gap_s)

    def collect(self, n=None, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end and (n is None or len(self.got) < n):
            try:
                d, _ = self.ground.recvfrom(65535)
                self.got.append((time.perf_counter(), d))
            except (socket.timeout, TimeoutError, ConnectionResetError):
                pass
        return self.got

    def close(self):
        self.stop.set()
        self.th.join()
        self.ground.close()
        self.tx.close()
        return self.em.stats()


class TestForwarding(unittest.TestCase):
    def test_zero_loss_forwards_every_packet_unchanged(self):
        r = Rig()
        pkts = POSE[:150] + [BIG]
        r.send(pkts, gap_s=0.001)
        got = r.collect(len(pkts))
        st = r.close()
        self.assertEqual(sorted(d for _, d in got), sorted(pkts))          # bytes (payload + CRC) unchanged
        self.assertTrue(all(P.decode(d) for _, d in got))
        self.assertEqual((st["received"], st["forwarded"], st["dropped_loss"], st["delivery_ratio"]), (151, 151, 0, 1.0))

    def test_loss_rates_are_statistically_consistent(self):
        for p in (0.10, 0.30, 0.50):
            em = LunarLinkEmulator(("127.0.0.1", 0), ("127.0.0.1", 9), None, config=LinkConfig(loss=p), seed=42)
            for d in POSE:
                em._ingest(d, time.perf_counter())
            st = em.close()
            n = st["received"]
            obs = st["dropped_loss"] / n
            sigma = math.sqrt(p * (1 - p) / n)
            self.assertLess(abs(obs - p), 4 * sigma, (p, obs))             # binomial, 4 sigma
            self.assertEqual(st["dropped_loss"] + st["dropped_shutdown"] + st["forwarded"], n)

    def test_total_loss_and_determinism(self):
        r = Rig(cfg=LinkConfig(loss=1.0))
        r.send(POSE[:100], gap_s=0.0005)
        got = r.collect(timeout=0.8)
        st = r.close()
        self.assertEqual((len(got), st["forwarded"], st["dropped_loss"]), (0, 0, 100))
        runs = []
        for _ in range(2):                                                 # same seed -> same drop pattern
            em = LunarLinkEmulator(("127.0.0.1", 0), ("127.0.0.1", 9), None, config=LinkConfig(loss=0.3), seed=7)
            pattern = []
            for d in POSE[:500]:
                before = em.st["dropped_loss"]
                em._ingest(d, time.perf_counter())
                pattern.append(em.st["dropped_loss"] > before)
            em.close()
            runs.append(pattern)
        self.assertEqual(runs[0], runs[1])


class TestTiming(unittest.TestCase):
    def test_latency_and_jitter(self):
        r = Rig(cfg=LinkConfig(latency_ms=120.0, jitter_ms=30.0))
        r.send(POSE[:60], gap_s=0.005)
        r.collect(60)
        st = r.close()
        d = st["delay_ms"]
        self.assertEqual(d["n"], 60)
        self.assertGreaterEqual(min(r.em.delays_ms), 90.0 - 1.0)
        self.assertLessEqual(max(r.em.delays_ms), 150.0 + 25.0)            # + scheduling slack of the select loop
        self.assertGreater(max(r.em.delays_ms) - min(r.em.delays_ms), 20.0)   # jitter really varies the delay

    def test_bandwidth_enforced(self):
        r = Rig(cfg=LinkConfig(bandwidth_Bps=4000.0, queue_max=64))
        r.send([BIG] * 12)                                                 # 12,792 B at once
        got = r.collect(12, timeout=6.0)
        st = r.close()
        self.assertEqual(len(got), 12)
        span = got[-1][0] - got[0][0]
        rate = sum(len(d) for _, d in got[1:]) / span                      # the bucket starts with one packet of burst
        self.assertLess(abs(rate - 4000.0) / 4000.0, 0.15, rate)
        self.assertEqual(st["dropped_queue"], 0)

    def test_queue_limit_drops_new_packets(self):
        r = Rig(cfg=LinkConfig(bandwidth_Bps=500.0, queue_max=5))
        r.send([BIG] * 30)
        time.sleep(0.3)
        st = r.close()
        self.assertGreaterEqual(st["dropped_queue"], 20)
        self.assertEqual(st["received"], 30)
        self.assertEqual(st["received"], st["forwarded"] + st["dropped_queue"] + st["dropped_shutdown"])


class TestLiveControl(unittest.TestCase):
    def test_presets_switch_live_blackout_and_restore(self):
        r = Rig()
        r.send(POSE[:20], gap_s=0.002)
        r.collect(20)
        rep = send_control("SEVERE", r.ctl)
        self.assertEqual((rep["preset"], rep["configured_loss"]), ("SEVERE", 0.5))
        rep = send_control("BLACKOUT", r.ctl)
        self.assertEqual(rep["link_state"], "BLACKOUT")
        before = len(r.got)
        r.send(POSE[20:60], gap_s=0.002)
        r.collect(timeout=0.5)
        self.assertEqual(len(r.got), before)                               # nothing delivered during blackout
        time.sleep(0.2)
        rep = send_control("RESTORE", r.ctl)
        self.assertEqual(rep["link_state"], "UP")
        self.assertGreater(rep["blackout_total_s"], 0.5)
        r.send(POSE[60:80], gap_s=0.002)
        r.collect(before + 20)
        st = r.close()
        self.assertEqual(len(r.got), before + 20)                          # communication resumes immediately
        self.assertEqual(st["dropped_blackout"], 40)
        self.assertEqual(st["by_preset"]["BLACKOUT"]["dropped"], 40)
        rep_err = None
        r2 = Rig()
        rep_err = send_control("SET loss=2", r2.ctl)
        r2.close()
        self.assertIn("error", rep_err)                                    # invalid settings rejected, link unchanged

    def test_blackout_discards_in_flight_and_clean_shutdown(self):
        r = Rig(cfg=LinkConfig(latency_ms=800.0))
        r.send(POSE[:10], gap_s=0.001)
        time.sleep(0.1)
        r.em.apply("BLACKOUT")                                             # packets in flight are not delivered later
        time.sleep(0.9)
        r.em.apply("RESTORE", latency_ms=800.0)                           # link back, still 800 ms of propagation
        r.send(POSE[10:15], gap_s=0.001)
        time.sleep(0.2)
        st = r.close()                                                     # in flight at shutdown -> dropped_shutdown
        self.assertEqual(st["forwarded"], 0)
        self.assertEqual(st["dropped_blackout"], 10)
        self.assertEqual(st["dropped_shutdown"], 5)
        self.assertEqual(st["received"], 15)
        self.assertTrue(r.em.rx._closed)

    def test_presets_defined(self):
        self.assertEqual(set(PRESETS), {"NORMAL", "DEGRADED", "SEVERE", "BLACKOUT", "RESTORE"})
        self.assertEqual((PRESETS["DEGRADED"].loss, PRESETS["SEVERE"].loss, PRESETS["BLACKOUT"].loss), (0.3, 0.5, 1.0))
        self.assertEqual(PRESETS["RESTORE"], PRESETS["NORMAL"])


if __name__ == "__main__":
    unittest.main()
