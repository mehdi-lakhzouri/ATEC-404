"""R5.4 store-and-forward: queue logic on a fake clock (priorities, coalescing, superseding, retries with adaptive RTO and
Karn's rule, expiry, memory limits, probes), ground reconciliation (duplicates, out of order, full snapshots, resets), the
ACK packet, and end-to-end over real localhost UDP: rover DTN -> lunar link emulator (forward + reverse) -> ground with
ACKs, through blackout / restore, receiver restart and queue overflow."""

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.telemetry import protocol as P
from artemis_x.telemetry.adapter import CRITICAL, LOW, NORMAL
from artemis_x.telemetry.dtn import DtnLink, QueueConfig, StoreForwardQueue
from artemis_x.telemetry.emulator import LinkConfig, LunarLinkEmulator
from artemis_x.telemetry.ground_map import GroundMap
from artemis_x.telemetry.transport import GroundReceiver
from tests.test_telemetry import FIX


def pkt(seq, kind="obstacles", prio=NORMAL, **meta):
    if kind == "pose":
        b = P.encode_pose(1, seq, FIX["pose"])
    elif kind == "event":
        b = P.encode_event(1, seq, 1.0, "map_reset", value=1)
    else:
        b = P.encode_obstacles(1, seq, FIX, FIX["obstacles"][:3], full=meta.get("full", False))[0]
    return b, {"kind": kind, "prio": prio, "seq": seq, "t_obs_s": 1.0, **meta}


class TestQueue(unittest.TestCase):
    def q(self, **kw):
        return StoreForwardQueue(QueueConfig(**kw))

    def test_priority_order_and_pose_coalescing(self):
        q = self.q()
        for s in range(5):
            q.offer(*pkt(s, "pose", LOW), now=0.0)
        q.offer(*pkt(10, prio=NORMAL), now=0.0)
        q.offer(*pkt(11, "event", CRITICAL), now=0.0)
        self.assertEqual(q.st["coalesced"], 4)                              # only the newest pose waits
        self.assertEqual([e.seq for e in q.due(0.0, True)], [11, 10, 4])   # CRITICAL, NORMAL, LOW
        for e in q.due(0.0, True):
            q.on_sent(e, 0.0)
        self.assertNotIn(4, q.pending)                                      # poses are fire-and-forget
        self.assertEqual(q.st["fire_and_forget"], 1)

    def test_backoff_karn_and_adaptive_rto(self):
        q = self.q()
        e = q.offer(*pkt(1), now=0.0)
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 16.0]                         # rto 1 s initial, doubling, capped at 16 s
        t = 0.0
        for gap in expected:
            q.on_sent(e, t)
            self.assertAlmostEqual(e.next_tx - t, gap)
            t = e.next_tx
        q.on_ack(1, t + 0.5)
        self.assertEqual(q.rtt.samples, 0)                                  # Karn: retransmitted -> no RTT sample
        for s in range(2, 12):                                              # clean 1.3 s RTT samples (lunar light time)
            e = q.offer(*pkt(s), now=100.0 + s)
            q.on_sent(e, 100.0 + s)
            q.on_ack(s, 101.3 + s)
        self.assertGreaterEqual(q.rtt.rto, 1.3)
        self.assertLess(q.rtt.rto, 2.0)
        self.assertEqual(q.st["acked"]["normal"], 11)

    def test_expiry_by_ttl_and_attempts_but_critical_survives_a_long_blackout(self):
        q = self.q()
        n = q.offer(*pkt(1), now=0.0)                                       # NORMAL: TTL 60 s
        c = q.offer(*pkt(2, "event", CRITICAL), now=0.0)                    # CRITICAL: TTL 600 s
        q.expire(59.0)
        self.assertIn(1, q.pending)
        q.expire(61.0)
        self.assertNotIn(1, q.pending)
        self.assertEqual(q.st["expired"]["normal"], 1)
        q.expire(300.0)                                                     # 5-minute blackout: still kept
        self.assertIn(2, q.pending)
        a = self.q()
        e = a.offer(*pkt(3), now=0.0)
        for k in range(a.cfg.max_attempts[NORMAL]):                         # attempt cap
            a.on_sent(e, e.next_tx)
        a.expire(e.next_tx)
        self.assertEqual(a.st["expired"]["normal"], 1)
        self.assertIsNotNone(n)
        self.assertIsNotNone(c)

    def test_memory_limits_evict_low_then_normal_then_critical(self):
        q = self.q(max_entries=4)
        q.offer(*pkt(1, "event", CRITICAL), now=0.0)
        q.offer(*pkt(2, "event", CRITICAL), now=0.1)
        q.offer(*pkt(3, prio=NORMAL), now=0.2)
        q.offer(*pkt(4, "pose", LOW), now=0.3)
        q.offer(*pkt(5, prio=NORMAL), now=0.4)                              # 5 entries > 4: the pose goes
        self.assertEqual(q.st["evicted"], {"critical": 0, "normal": 0, "low": 1})
        q.offer(*pkt(6, "event", CRITICAL), now=0.5)                        # then the OLDEST normal
        self.assertNotIn(3, q.pending)
        q.offer(*pkt(7, "event", CRITICAL), now=0.6)
        q.offer(*pkt(8, "event", CRITICAL), now=0.7)                        # only criticals left: the oldest critical goes
        self.assertEqual(q.st["evicted"], {"critical": 1, "normal": 2, "low": 1})
        self.assertEqual(sorted(q.pending), [2, 6, 7, 8])
        size = len(pkt(9)[0])
        b = self.q(max_bytes=3 * size)
        for s in range(5):
            b.offer(*pkt(20 + s), now=float(s))
        self.assertLessEqual(b.bytes, 3 * size)
        self.assertEqual(b.st["peak_bytes"], 3 * size)

    def test_full_snapshot_supersedes_older_refreshes_but_not_critical_hazards(self):
        q = self.q()
        q.offer(*pkt(1, prio=NORMAL, snapshot_t_obs=10.0), now=0.0)
        q.offer(*pkt(2, prio=CRITICAL, snapshot_t_obs=10.0, new_hazards=["trk-1"]), now=0.0)
        q.offer(*pkt(3, prio=NORMAL, full=True, snapshot_t_obs=12.0), now=0.1)
        q.offer(*pkt(4, prio=NORMAL, full=True, snapshot_t_obs=12.0), now=0.1)   # second part of the same snapshot stays
        self.assertEqual(sorted(q.pending), [2, 3, 4])
        self.assertEqual(q.st["superseded"], 1)

    def test_link_down_sends_one_probe_per_interval(self):
        q = self.q(probe_interval_s=1.0)
        for s in range(20):
            q.offer(*pkt(s, "event", CRITICAL), now=0.0)
        self.assertEqual(len(q.due(0.0, False)), 1)
        self.assertEqual(len(q.due(0.5, False)), 0)
        self.assertEqual(len(q.due(1.0, False)), 1)
        self.assertEqual(len(q.due(1.0, True)), q.cfg.budget_per_tick)       # link up again: bounded drain
        st = q.stats()
        for k in ("pending_critical", "queue_bytes", "retries", "expired", "evicted", "superseded", "coalesced", "rto_s",
                  "delivery_latency_s", "peak_entries", "probes"):
            self.assertIn(k, st)                                            # exported for Mission Control

    def test_karn_rto_backoff_converges_on_a_long_lunar_rtt(self):
        """RTO starts at 1 s, the real round trip is 2.6 s: every first try times out, so without backing off the RTO
        itself Karn's rule would never get a sample. With it, the RTO grows past 2.6 s and clean samples follow."""
        q = self.q(backoff_cap_s=64.0)
        rtt, t = 2.6, 0.0
        inflight = []
        for s in range(60):
            e = q.offer(*pkt(s), now=t)
            q.on_sent(e, t)
            inflight.append((t + rtt, s))
            t += 1.0
            for e2 in [x for x in list(q.pending.values()) if x.next_tx <= t]:
                q.on_sent(e2, t)                                            # timeout retransmission
            for ta, sq in [x for x in inflight if x[0] <= t]:
                inflight.remove((ta, sq))
                q.on_ack(sq, ta)
        self.assertGreater(q.rtt.samples, 10)
        self.assertGreaterEqual(q.rtt.rto, rtt)
        self.assertGreater(q.stats()["rto_backoffs"], 0)

    def test_redundant_critical_evicted_first_and_covered_by_full_ack(self):
        q = self.q(max_entries=3)
        q.offer(*pkt(1, prio=CRITICAL, new_hazards=["trk-1"], ids=["trk-1"], snapshot_t_obs=10.0), now=0.0)
        q.offer(*pkt(2, "event", CRITICAL, event="map_reset"), now=0.1)
        q.offer(*pkt(3, prio=CRITICAL, new_hazards=["trk-2"], ids=["trk-2"], snapshot_t_obs=12.0), now=0.2)
        q.offer(*pkt(4, prio=CRITICAL, new_hazards=["trk-3"], ids=["trk-1", "trk-3"], snapshot_t_obs=14.0), now=0.3)
        self.assertNotIn(1, q.pending)                                      # trk-1 also in the newer packet 4
        self.assertEqual(q.st["redundant_evicted"], 1)
        self.assertIn(2, q.pending)                                         # the map reset is kept
        c = self.q()
        c.offer(*pkt(5, prio=CRITICAL, new_hazards=["trk-5"], ids=["trk-5"], snapshot_t_obs=20.0), now=0.0)
        f = c.offer(*pkt(6, prio=NORMAL, full=True, ids=["trk-5", "trk-6"], snapshot_t_obs=22.0), now=0.0)
        c.on_sent(f, 0.0)
        c.on_ack(6, 0.1)                                                    # the ground holds trk-5 as of t = 22
        self.assertNotIn(5, c.pending)
        self.assertEqual(c.st["covered_by_full"], 1)

    def test_unknown_and_duplicate_acks(self):
        q = self.q()
        e = q.offer(*pkt(1), now=0.0)
        q.on_sent(e, 0.0)
        self.assertIsNotNone(q.on_ack(1, 0.2))
        self.assertIsNone(q.on_ack(1, 0.3))                                  # duplicate ACK
        self.assertIsNone(q.on_ack(99, 0.3))
        self.assertEqual(q.st["acks_unknown"], 2)


def obs_pkt(seq, t, ids, full=False, part=0, parts=1):
    doc = {**FIX, "stamp_s": t}
    dep = {"source": "depth", "status": "observed", "type": "obstacle"}
    recs = [{**FIX["obstacles"][0], "id": i, "x_m": float(k), **(dep if i.startswith("dep-") else {})} for k, i in enumerate(ids)]
    b = P.encode_obstacles(1, seq, doc, recs, full=full)[0]
    pk = P.decode(b)
    pk.body["part"], pk.body["parts"] = part, parts
    return pk


class TestGroundMap(unittest.TestCase):
    def test_duplicates_and_out_of_order(self):
        g = GroundMap()
        new = obs_pkt(5, 20.0, ["trk-1"])
        old = obs_pkt(3, 10.0, ["trk-1"])
        old.body["obstacles"][0]["x_m"] = 99.0
        g.apply(new)
        g.apply(new)                                                        # retransmission whose ACK was lost
        g.apply(old)                                                        # late retry of an older observation
        self.assertEqual(g.st["duplicates_ignored"], 1)
        self.assertEqual(g.st["stale_ignored"], 1)
        self.assertEqual(g.obs["trk-1"][0]["x_m"], 0.0)                     # newest observation kept

    def test_full_snapshot_reconciles_only_when_complete_and_newer(self):
        g = GroundMap()
        g.apply(obs_pkt(1, 10.0, ["trk-1", "trk-2", "dep-1_1"]))
        g.apply(obs_pkt(2, 20.0, ["trk-1"], full=True, part=0, parts=2))
        self.assertIn("trk-2", g.obs)                                       # incomplete: nothing removed yet
        g.apply(obs_pkt(3, 20.0, ["dep-1_1"], full=True, part=1, parts=2))
        self.assertEqual(sorted(g.obs), ["dep-1_1", "trk-1"])               # trk-2 no longer exists on the rover
        g.apply(obs_pkt(9, 30.0, ["trk-7"]))
        g.apply(obs_pkt(4, 15.0, ["trk-1"], full=True))                     # a late, OLDER full snapshot
        self.assertIn("trk-7", g.obs)                                       # never removes newer knowledge
        self.assertEqual(g.st["full_snapshots_late"], 1)

    def test_map_reset_and_pre_reset_late_packets(self):
        g = GroundMap()
        g.apply(obs_pkt(1, 10.0, ["trk-1"]))
        g.apply(P.decode(P.encode_event(1, 2, 15.0, "map_reset", value=1)))
        self.assertEqual(g.obs, {})
        g.apply(obs_pkt(3, 12.0, ["trk-1"]))                                # observed before the reset, arrived after it
        self.assertEqual(g.obs, {})
        self.assertEqual(g.st["pre_reset_ignored"], 1)
        g.apply(obs_pkt(4, 16.0, ["trk-9"]))
        self.assertEqual(list(g.obs), ["trk-9"])
        g.apply(P.decode(P.encode_pose(1, 6, {**FIX["pose"], "stamp_s": 40.0})))
        g.apply(P.decode(P.encode_pose(1, 5, {**FIX["pose"], "stamp_s": 30.0, "x": 1.0})))
        self.assertEqual(g.snapshot()["pose"]["t_obs_s"], 40.0)             # newest pose wins


class TestAckPacket(unittest.TestCase):
    def test_round_trip_limits_and_full_flag(self):
        pk = P.decode(P.encode_ack(7, 1234, [5, 6, 2 ** 32 - 1], 12.5))
        self.assertEqual((pk.type, pk.session, pk.body["seqs"]), (P.T_ACK, 1234, [5, 6, 2 ** 32 - 1]))
        for bad in ([], list(range(P.MAX_ACK + 1))):
            with self.assertRaises(P.ProtocolError):
                P.encode_ack(1, 1, bad)
        raw = bytearray(P.encode_ack(1, 1, [1, 2]))
        raw[19] ^= 1
        with self.assertRaises(P.ProtocolError):
            P.decode(bytes(raw))
        full = P.decode(P.encode_obstacles(1, 0, FIX, full=True)[0]).body
        self.assertTrue(full["full"])
        self.assertFalse(P.decode(P.encode_obstacles(1, 0, FIX)[0]).body["full"])


class FakeAdapter:
    """Feeds real protocol packets with R5.4 metadata; session 1."""

    def __init__(self):
        self.session, self.items, self.seq = 1, [], 0

    def add(self, kind, prio, **meta):
        b, m = pkt(self.seq, kind, prio, **meta)
        self.items.append((b, m))
        self.seq += 1
        return m["seq"]

    def drain_meta(self, n=None):
        out, self.items = self.items, []
        return out


class TestEndToEnd(unittest.TestCase):
    """rover DtnLink -> emulator (forward + reverse) -> GroundReceiver with ACKs, real localhost UDP."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ground_port = self._free()
        self.em = LunarLinkEmulator(("127.0.0.1", 0), ("127.0.0.1", self.ground_port), ("127.0.0.1", 0), seed=3,
                                    reverse_listen=("127.0.0.1", 0), rover_ack=("127.0.0.1", 1))
        self.ad = FakeAdapter()
        cfg = QueueConfig(down_after_s=0.6, probe_interval_s=0.2)
        self.dtn = DtnLink(self.ad, self.em.rx.getsockname(), ("127.0.0.1", 0), cfg)
        self.em.rover_ack = self.dtn.ack.getsockname()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.em.run, args=(self.stop,), daemon=True)
        self.th.start()
        self.rx = self._receiver()

    def _free(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def _receiver(self):
        rx = GroundReceiver("127.0.0.1", self.ground_port, Path(self.tmp.name) / "g.jsonl", timeout_s=0.02,
                            ack_to=self.em.rev.getsockname(), ack_every_s=0.05, reconcile=True)
        ev = threading.Event()
        th = threading.Thread(target=rx.run, kwargs={"stop": ev}, daemon=True)
        th.start()
        rx._stop, rx._th = ev, th
        return rx

    def _stop_receiver(self, rx):
        rx._stop.set()
        rx._th.join()

    def pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.dtn.tick(0.0)
            time.sleep(0.01)

    def tearDown(self):
        self._stop_receiver(self.rx)
        self.stop.set()
        self.th.join()
        self.dtn.finish(grace_s=0.0)

    def received(self, rx):
        return {k[1] for k in rx.map.seen}

    def test_blackout_critical_survives_and_is_recovered_after_restore(self):
        for _ in range(5):
            self.ad.add("obstacles", NORMAL)
        self.pump(0.5)
        self.assertEqual(self.dtn.q.stats()["pending_total"], 0)            # all ACKed over the reverse link
        self.em.apply("BLACKOUT")
        crit = [self.ad.add("event", CRITICAL) for _ in range(3)] + [self.ad.add("obstacles", CRITICAL, new_hazards=["trk-9"])]
        for _ in range(20):
            self.ad.add("pose", LOW)
        self.pump(1.5)
        st = self.dtn.stats()
        self.assertEqual(st["link_state"], "DOWN")                          # no ACKs -> detected
        self.assertEqual(st["pending_critical"], 4)                         # kept during the blackout
        self.assertGreater(st["probes"], 0)
        self.assertGreaterEqual(st["coalesced"], 19)                        # stale poses discarded
        self.assertFalse(set(crit) & self.received(self.rx))                # nothing got through the blackout
        self.em.apply("RESTORE")
        t0 = time.perf_counter()
        while self.dtn.q.stats()["pending_critical"] and time.perf_counter() - t0 < 5.0:
            self.pump(0.05)
        recovery = time.perf_counter() - t0
        self.assertEqual(self.dtn.q.stats()["pending_critical"], 0)
        self.assertTrue(set(crit) <= self.received(self.rx))                # recovered after restoration
        self.assertEqual(self.dtn.stats()["link_state"], "UP")
        self.assertLess(recovery, 3.0)
        self.assertEqual(self.dtn.stats()["outages_detected"][0][1] is not None, True)

    def test_loss_both_directions_retries_dedup_and_honest_accounting(self):
        """50 % loss on data AND on ACKs: a round trip succeeds 25 % of the time. Nothing is guaranteed: each NORMAL packet
        is retried until ACKed or its attempt cap (8) expires it. The ground still gets almost everything (each packet
        has up to 8 independent 50 % chances), lost ACKs cause duplicates that the ground ignores, and the rover's books
        close: offered = ACKed + expired."""
        self.dtn.q.cfg.backoff_cap_s = 1.0                                  # keep the test short; same mechanism
        self.em.apply(None, loss=0.5, latency_ms=30.0, jitter_ms=20.0)
        sent = [self.ad.add("obstacles", NORMAL) for _ in range(40)]
        self.pump(0.05)                                                     # first tick moves them into the queue
        t0 = time.perf_counter()
        while self.dtn.q.stats()["pending_total"] and time.perf_counter() - t0 < 30.0:
            self.pump(0.1)
        st = self.dtn.q.stats()
        self.assertEqual(st["pending_total"], 0)
        self.assertEqual(st["acked"]["normal"] + st["expired"]["normal"], 40)
        time.sleep(0.2)
        got = set(sent) & self.received(self.rx)
        self.assertGreaterEqual(len(got), 38)                               # expected 39.8 of 40 (1 - 0.5^8 each)
        self.assertGreaterEqual(len(got), st["acked"]["normal"])           # an ACK implies reception, not the reverse
        self.assertGreater(st["retries"], 0)
        self.assertGreater(self.rx.stats["duplicates"], 0)                  # lost ACKs -> retransmissions -> duplicates
        self.assertEqual(self.rx.map.st["duplicates_ignored"], self.rx.stats["duplicates"])

    def test_receiver_restart(self):
        self._stop_receiver(self.rx)
        self.rx.close()
        crit = [self.ad.add("event", CRITICAL) for _ in range(3)]
        self.pump(1.0)                                                      # ground down: sent, never ACKed
        self.assertEqual(self.dtn.q.stats()["pending_critical"], 3)
        self.rx = self._receiver()                                          # restarted on the same port
        t0 = time.perf_counter()
        while self.dtn.q.stats()["pending_critical"] and time.perf_counter() - t0 < 6.0:
            self.pump(0.05)
        self.assertEqual(self.dtn.q.stats()["pending_critical"], 0)
        self.assertTrue(set(crit) <= self.received(self.rx))

    def test_queue_overflow_during_blackout_keeps_critical(self):
        self.dtn.q.cfg.max_entries = 8
        self.em.apply("BLACKOUT")
        crit = [self.ad.add("event", CRITICAL) for _ in range(3)]
        for _ in range(30):
            self.ad.add("obstacles", NORMAL)
        self.pump(0.5)
        st = self.dtn.q.stats()
        self.assertLessEqual(st["pending_total"], 8)                        # memory bounded
        self.assertEqual(st["evicted"]["critical"], 0)
        self.assertGreater(st["evicted"]["normal"], 0)
        self.em.apply("RESTORE")
        t0 = time.perf_counter()
        while self.dtn.q.stats()["pending_critical"] and time.perf_counter() - t0 < 5.0:
            self.pump(0.05)
        self.assertTrue(set(crit) <= self.received(self.rx))


if __name__ == "__main__":
    unittest.main()
