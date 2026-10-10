"""R5.1: binary lunar telemetry protocol v1 and the R5.2 adapter. Round trips on a REAL R4.6 artemis.obstacles.v1 export
(tests/fixtures/obstacles_v1_fusion.json, the nominal fusion run of r4-validate-20261010-204718), quantisation bounds,
CRC / truncation / malformed-packet rejection, limits, id namespaces and duplicate-observation handling."""

import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from tests import _common  # noqa: F401  (puts src/ on sys.path)
from artemis_x.telemetry import protocol as P
from artemis_x.telemetry.adapter import TelemetryAdapter, dedupe

FIX = json.loads((Path(__file__).parent / "fixtures" / "obstacles_v1_fusion.json").read_text(encoding="utf-8"))


def packets_obstacles(pkts):
    out = []
    for p in pkts:
        out += P.decode(p).body["obstacles"]
    return out


class TestRoundTrip(unittest.TestCase):
    def test_crc_check_value(self):
        self.assertEqual(P.crc16(b"123456789"), 0x29B1)                 # CRC-16/CCITT-FALSE reference

    def test_sizes(self):
        self.assertEqual((P.HEADER.size, P.POSE.size, P.OBS_HEAD.size, P.RECORD.size, P.EVENT.size), (18, 13, 6, 26, 8))
        self.assertEqual(P.MAX_PACKET, 1066)

    def test_real_export_round_trip_and_precision(self):
        pkts = P.encode_obstacles(7, 100, FIX)
        self.assertEqual(len(pkts), 2)                                     # 48 obstacles -> 40 + 8
        self.assertTrue(all(len(p) <= P.MAX_PACKET for p in pkts))
        got = packets_obstacles(pkts)
        self.assertEqual(len(got), len(FIX["obstacles"]))
        for a, b in zip(FIX["obstacles"], got):
            self.assertEqual(a["id"], b["id"])                             # ids exactly preserved (both namespaces)
            for k in ("type", "source", "status"):
                self.assertEqual(a[k], b[k])
            self.assertEqual(bool(a["protected"]), b["protected"])
            for k in ("x_m", "y_m", "radius_m", "uncertainty_m", "timestamp_s"):
                self.assertLessEqual(abs(a[k] - b[k]), 0.0005 + 1e-9, (a["id"], k))   # 1 mm / 1 ms quantisation
            self.assertLessEqual(abs(a["confidence"] - b["confidence"]), 0.5 / 255 + 1e-9)
            if a.get("track_ids"):
                self.assertEqual(b["linked_track"], a["track_ids"][0])
        p0 = P.decode(pkts[0], rx_time_s=600.0)
        self.assertEqual((p0.type, p0.session, p0.seq, p0.body["part"], p0.body["parts"]), (P.T_OBSTACLES, 7, 100, 0, 2))
        self.assertEqual(P.decode(pkts[1]).seq, 101)
        self.assertAlmostEqual(p0.t_obs_s, FIX["stamp_s"], places=3)     # observation time from the rover
        self.assertEqual(p0.rx_time_s, 600.0)                              # reception time: receiver-side only
        self.assertEqual(p0.body["frame"], "odom")

    def test_pose_round_trip(self):
        pose = dict(FIX["pose"])
        pk = P.decode(P.encode_pose(1, 5, pose))
        self.assertEqual(len(P.encode_pose(1, 5, pose)), 18 + 13 + 2)
        self.assertAlmostEqual(pk.body["x"], pose["x"], delta=0.0005)
        self.assertAlmostEqual(pk.body["yaw"], pose["yaw"], delta=0.5e-4)
        self.assertAlmostEqual(pk.body["sigma_xy"], pose["sigma_xy"], delta=0.005)
        self.assertFalse(pk.body["sigma_xy_saturated"])
        wrapped = P.decode(P.encode_pose(1, 6, {**pose, "yaw": 3 * math.pi / 2})).body["yaw"]
        self.assertAlmostEqual(wrapped, -math.pi / 2, delta=1e-4)          # wrapped to [-pi, pi]
        sat = P.decode(P.encode_pose(1, 7, {**pose, "sigma_xy": 1e6})).body
        self.assertTrue(sat["sigma_xy_saturated"])                         # explicit saturation, flagged

    def test_event_round_trip(self):
        pk = P.decode(P.encode_event(3, 9, 12.5, "health", "STOP", ["stale_corridor", "inference_unavailable"]))
        self.assertEqual(pk.body, {"event": "health", "state": "STOP", "reasons": ["stale_corridor", "inference_unavailable"], "value": 0})
        self.assertEqual(len(P.encode_event(3, 9, 12.5, "map_reset", value=2)), 18 + 8 + 2)

    def test_empty_snapshot_is_one_packet(self):
        doc = {**FIX, "obstacles": []}
        pkts = P.encode_obstacles(1, 0, doc)
        self.assertEqual(len(pkts), 1)
        self.assertEqual(P.decode(pkts[0]).body["obstacles"], [])


class TestRejection(unittest.TestCase):
    def setUp(self):
        self.pkt = P.encode_obstacles(7, 1, FIX)[1]

    def test_crc_corruption_every_byte(self):
        for i in range(len(self.pkt)):
            bad = bytearray(self.pkt)
            bad[i] ^= 0x01
            with self.assertRaises(P.ProtocolError):
                P.decode(bytes(bad))

    def test_truncation_and_extension(self):
        for n in range(len(self.pkt)):
            with self.assertRaises(P.ProtocolError):
                P.decode(self.pkt[:n])
        with self.assertRaises(P.ProtocolError):
            P.decode(self.pkt + b"\x00")

    def _forge(self, ptype=P.T_OBSTACLES, version=P.VERSION, magic=P.MAGIC, payload=None):
        payload = self.pkt[P.HEADER.size:-2] if payload is None else payload
        head = P.HEADER.pack(magic, version, ptype, 1, 1, 0, len(payload))
        return head + payload + P.CRC.pack(P.crc16(head + payload))       # valid CRC: the check must be semantic

    def test_unknown_version_type_magic(self):
        for kw in ({"version": 2}, {"ptype": 9}, {"magic": b"XX"}):
            with self.assertRaises(P.ProtocolError):
                P.decode(self._forge(**kw))

    def test_invalid_fields_with_valid_crc(self):
        body = bytearray(self.pkt[P.HEADER.size:-2])
        bad_frame = bytearray(body); bad_frame[0] = 3
        bad_count = bytearray(body); bad_count[3] = 41                     # count > MAX_RECORDS
        bad_flags = bytearray(body); bad_flags[P.OBS_HEAD.size] = 0xFF     # reserved bit / bad enum
        bad_type = bytearray(body); bad_type[P.OBS_HEAD.size] = (body[P.OBS_HEAD.size] & ~0x06) | 0x06   # type index 3
        for b in (bad_frame, bad_count, bad_flags, bad_type):
            with self.assertRaises(P.ProtocolError):
                P.decode(self._forge(payload=bytes(b)))
        with self.assertRaises(P.ProtocolError):
            P.decode(self._forge(ptype=P.T_EVENT, payload=struct.pack(">BBIH", 9, 0, 0, 0)))
        with self.assertRaises(P.ProtocolError):
            P.decode(self._forge(ptype=P.T_EVENT, payload=struct.pack(">BBIH", 1, 0, 1 << 20, 0)))
        with self.assertRaises(P.ProtocolError):
            P.decode(b"\x00" * (P.MAX_PACKET + 1))

    def test_encode_rejects_invalid_values(self):
        o = copy.deepcopy(FIX["obstacles"][0])
        cases = [("x_m", 3e6), ("x_m", float("nan")), ("radius_m", -0.1), ("radius_m", 70.0), ("confidence", 1.2),
                 ("uncertainty_m", -1.0), ("timestamp_s", -1.0), ("type", "crater"), ("status", "gone"),
                 ("id", "obj-3"), ("id", "trk-x"), ("id", "dep-40000_1"), ("source", "depth")]   # last: namespace mismatch
        for k, v in cases:
            with self.assertRaises(P.ProtocolError, msg=(k, v)):
                P.encode_obstacles(1, 0, {**FIX, "obstacles": [{**o, k: v}]})
        with self.assertRaises(P.ProtocolError):
            P.encode_obstacles(1, 0, {**FIX, "frame": "map"})
        with self.assertRaises(P.ProtocolError):
            P.encode_pose(1, 0, {**FIX["pose"], "frame": "world"})
        with self.assertRaises(P.ProtocolError):
            P.encode_event(1, 0, 1.0, "health", "PANIC")
        with self.assertRaises(P.ProtocolError):
            P.encode_pose(2 ** 32, 0, FIX["pose"])                          # session overflow


class TestLimitsAndIds(unittest.TestCase):
    def test_extreme_values_round_trip(self):
        o = {"id": "dep--32768_32767", "type": "negative_hazard", "x_m": -2147483.648, "y_m": 2147483.647, "radius_m": 65.535,
             "confidence": 1.0, "uncertainty_m": 1e9, "timestamp_s": 4294967.295, "source": "depth", "status": "observed",
             "protected": True}
        got = packets_obstacles(P.encode_obstacles(2 ** 32 - 1, 2 ** 32 - 1, {**FIX, "stamp_s": 4294967.295, "obstacles": [o]}))[0]
        self.assertEqual(got["id"], "dep--32768_32767")
        self.assertEqual((got["x_m"], got["y_m"], got["radius_m"]), (-2147483.648, 2147483.647, 65.535))
        self.assertTrue(got["uncertainty_saturated"])
        p = P.decode(P.encode_obstacles(1, 2 ** 32 - 1, {**FIX, "obstacles": FIX["obstacles"][:41]})[1])
        self.assertEqual(p.seq, 0)                                          # sequence wraps modulo 2^32

    def test_max_records_and_packet_split(self):
        objs = [{**FIX["obstacles"][0], "id": f"trk-{k}"} for k in range(P.MAX_RECORDS * 3 + 1)]
        pkts = P.encode_obstacles(1, 0, {**FIX, "obstacles": objs})
        self.assertEqual(len(pkts), 4)
        self.assertEqual(max(len(p) for p in pkts), P.MAX_PACKET)
        self.assertEqual([o["id"] for o in packets_obstacles(pkts)], [o["id"] for o in objs])

    def test_track_and_cluster_ids_never_collide(self):
        a, b = P.encode_id("trk-65537"), P.encode_id("dep-1_1")          # same 32-bit value 0x00010001
        self.assertEqual(a[1], b[1])
        self.assertNotEqual(a[0], b[0])                                    # different namespace bit
        self.assertEqual(P.decode_id(*a), "trk-65537")
        self.assertEqual(P.decode_id(*b), "dep-1_1")
        ids = [o["id"] for o in FIX["obstacles"]]
        self.assertEqual(len(set(P.encode_id(i) for i in ids)), len(ids))


class TestDuplicatesAndAdapter(unittest.TestCase):
    def test_dedupe_on_real_export(self):
        kept, dropped = dedupe(FIX)
        linked = [o for o in FIX["obstacles"] if o["source"] == "depth" and o.get("track_ids")]
        self.assertEqual(len(kept) + dropped, len(FIX["obstacles"]))
        tracks = {o["id"]: o for o in FIX["obstacles"] if o["source"] == "yolo+depth"}
        for o in linked:                                                   # dropped exactly when the track covers it
            same = o["radius_m"] <= max(tracks[t]["radius_m"] for t in o["track_ids"]) + 0.2
            self.assertEqual(o not in kept, same, o["id"])
        self.assertTrue(all(o in kept for o in FIX["obstacles"] if o["source"] == "yolo+depth"))   # tracks always sent

    def test_adapter_live_mapper(self):
        from artemis_x.sim.mapping.mapper import Mapper
        from tests.test_mapping_depth import BOX, FakeSensors
        from tests.test_mapping_fusion import run_cycles
        with tempfile.TemporaryDirectory() as d:
            sens = FakeSensors(BOX)
            m = Mapper(sens, Path(d))
            run_cycles(m, sens, 0.0, 4)
            ad = TelemetryAdapter(m, session_id=42, max_queue=8)
            for k in range(12):
                ad.tick(1.0 + 0.1 * k)
            got = [P.decode(p) for p in ad.drain()]
            kinds = {g.type for g in got}
            self.assertTrue({P.T_EVENT, P.T_POSE, P.T_OBSTACLES} <= kinds)
            self.assertEqual([g.seq for g in got], sorted(g.seq for g in got))       # monotonic sequence
            self.assertTrue(all(g.session == 42 for g in got))
            obs = [o for g in got if g.type == P.T_OBSTACLES for o in g.body["obstacles"]]
            self.assertTrue(any(o["source"] == "depth" for o in obs))
            m.reset_map("test")
            ad.tick(5.0)
            ev = P.decode(ad.drain()[-1])
            self.assertEqual((ev.body["event"], ev.body["value"]), ("map_reset", 1))
            for k in range(40):                                            # bounded queue: oldest dropped, counted
                ad.tick(10.0 + k)
            self.assertLessEqual(len(ad.queue), 8)
            self.assertGreater(ad.stats["dropped_queue_full"], 0)
            m.finish()


if __name__ == "__main__":
    unittest.main()
