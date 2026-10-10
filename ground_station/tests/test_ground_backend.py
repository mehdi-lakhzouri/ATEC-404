"""R5.5 ground backend against a REAL MongoDB (docker/compose.yml) and real localhost UDP into the R5.4 GroundReceiver.

The database tests are SKIPPED (and say so) when MongoDB is unreachable - they never pass without a real database. Each
test class uses its own throw-away database, dropped afterwards.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ground_station"))
sys.path.insert(0, str(ROOT / "src"))

from artemis_ground.config import Settings  # noqa: E402
from artemis_ground.main import create_app  # noqa: E402
from artemis_x.telemetry import protocol as P  # noqa: E402
from artemis_x.telemetry.emulator import LunarLinkEmulator  # noqa: E402
from pymongo import MongoClient  # noqa: E402
import json  # noqa: E402

FIX = json.loads((ROOT / "tests" / "fixtures" / "obstacles_v1_fusion.json").read_text(encoding="utf-8"))   # real R4.6 export

BASE = Settings()


def mongo_up() -> bool:
    try:
        with MongoClient(BASE.mongo_uri, serverSelectionTimeoutMS=1500) as c:
            c.admin.command("ping")
        return True
    except Exception:
        return False


MONGO = mongo_up()
SESSION = 4242


def wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def obstacles_pkt(seq, t, ids, full=True, x=0.0, session=SESSION):
    dep = {"source": "depth", "status": "observed", "type": "obstacle"}
    recs = [{**FIX["obstacles"][0], "id": i, "x_m": x + k, **(dep if i.startswith("dep-") else {})} for k, i in enumerate(ids)]
    return P.encode_obstacles(session, seq, {**FIX, "stamp_s": t}, recs, full=full)[0]


def pose_pkt(seq, t, x=1.0):
    return P.encode_pose(SESSION, seq, {**FIX["pose"], "stamp_s": t, "x": x, "sigma_xy": 0.25})


class Backend:
    """The real app (TestClient runs its lifespan) with the receiver on an ephemeral UDP port."""

    def __init__(self, **over):
        self.dbname = f"artemis_test_{uuid.uuid4().hex[:8]}"
        self.settings = Settings(mongo_db=self.dbname, udp_port=0, ack_to=over.pop("ack_to", ""), emulator_control=over.pop(
            "emulator_control", "127.0.0.1:1"), link_metrics_period_s=0.5, log_json=False, log_level="WARNING", **over)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.svc = self.app.state.ingest
        self.port = self.svc.receiver.sock.getsockname()[1]
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, *pkts, gap=0.002):
        for p in pkts:
            self.tx.sendto(p, ("127.0.0.1", self.port))
            time.sleep(gap)

    def settle(self, n_valid, timeout=5.0):
        ok = wait(lambda: self.svc.receiver.stats["valid"] >= n_valid and self.svc.queue.empty(), timeout)
        time.sleep(0.15)                                    # the writer finishes its current item
        return ok

    def get(self, path, **params):
        return self.client.get(path, params=params)

    def close(self):
        self.client.__exit__(None, None, None)
        self.tx.close()
        if MONGO and self.dbname:
            with MongoClient(BASE.mongo_uri) as c:
                c.drop_database(self.dbname)


@unittest.skipUnless(MONGO, "MongoDB not reachable - start it with: docker compose -f docker/compose.yml up -d")
class TestIngestionAndApi(unittest.TestCase):
    def setUp(self):
        self.b = Backend()
        self.mid = str(SESSION)

    def tearDown(self):
        self.b.close()

    def test_indexes_mission_pose_obstacles_and_rest(self):
        b = self.b
        b.send(pose_pkt(0, 1.0), pose_pkt(1, 1.5, 2.0), P.encode_event(SESSION, 2, 1.6, "health", "STOP", ["stale_corridor"]),
               *P.encode_obstacles(SESSION, 3, FIX, full=True))
        self.assertTrue(b.settle(5))
        with MongoClient(BASE.mongo_uri) as c:
            names = {i["name"] for i in c[b.dbname]["telemetry_events"].list_indexes()}
        self.assertIn("mission_seq_unique", names)
        m = b.get(f"/api/v1/missions/{self.mid}").json()
        self.assertEqual(m["meta"]["api_version"], "v1")
        self.assertEqual((m["data"]["session"], m["data"]["status"]), (SESSION, "active"))
        self.assertEqual(m["data"]["health"]["state"], "STOP")
        pos = b.get(f"/api/v1/missions/{self.mid}/positions").json()
        self.assertEqual(pos["meta"]["pagination"]["total"], 2)
        self.assertEqual(pos["data"][0]["seq"], 1)                     # newest observation first
        self.assertAlmostEqual(pos["data"][0]["x_m"], 2.0)
        self.assertEqual(pos["data"][0]["frame"], "odom")
        self.assertIn("rx_utc", pos["data"][0])
        obs = b.get(f"/api/v1/missions/{self.mid}/obstacles", limit=1000).json()
        self.assertEqual(obs["meta"]["pagination"]["total"], len(FIX["obstacles"]))
        o = next(x for x in obs["data"] if x["obstacle_id"] == FIX["obstacles"][0]["id"])
        src = FIX["obstacles"][0]
        self.assertAlmostEqual(o["x_m"], src["x_m"], delta=0.0005)
        self.assertAlmostEqual(o["uncertainty_m"], src["uncertainty_m"], delta=0.0005)
        self.assertEqual((o["source"], o["status"], o["present"], o["persistent_id"]), (src["source"], src["status"], True, True))
        page = b.get(f"/api/v1/missions/{self.mid}/obstacles", limit=10, offset=10).json()
        self.assertEqual((len(page["data"]), page["meta"]["pagination"]["next_offset"]), (10, 20))
        ev = b.get(f"/api/v1/missions/{self.mid}/events", kind="health").json()
        self.assertEqual(ev["data"][0]["event"]["reasons"], ["stale_corridor"])
        tel = b.get(f"/api/v1/missions/{self.mid}/telemetry").json()
        self.assertEqual(tel["meta"]["pagination"]["total"], 5)
        ov = b.get(f"/api/v1/missions/{self.mid}/overview").json()["data"]
        self.assertEqual(ov["obstacles"]["present"], len(FIX["obstacles"]))
        self.assertEqual(ov["latest_position"]["seq"], 1)
        st = b.get(f"/api/v1/missions/{self.mid}/stats").json()["data"]
        self.assertEqual(st["documents"]["rover_positions"], 2)
        self.assertEqual(b.get("/api/v1/missions", status="active").json()["meta"]["pagination"]["total"], 1)
        self.assertEqual(b.get("/api/v1/missions/999").status_code, 404)
        self.assertEqual(b.get(f"/api/v1/missions/{self.mid}/positions", limit=5000).json()["error"]["code"], "validation_error")

    def test_duplicates_are_idempotent(self):
        b = self.b
        p = pose_pkt(7, 3.0)
        b.send(p, p, p)                                                # retransmissions (same bytes, same seq)
        self.assertTrue(b.settle(3))
        st = b.get(f"/api/v1/missions/{self.mid}/stats").json()["data"]
        self.assertEqual(st["documents"]["rover_positions"], 1)
        self.assertEqual(st["documents"]["telemetry_events"], 1)
        self.assertEqual(st["counters"]["duplicates"], 2)
        self.assertGreaterEqual(b.svc.stats["duplicates_ignored"], 2)

    def test_out_of_order_full_snapshot_and_map_reset(self):
        b = self.b
        b.send(obstacles_pkt(10, 20.0, ["trk-1"], full=False, x=5.0), obstacles_pkt(9, 10.0, ["trk-1"], full=False, x=99.0))
        self.assertTrue(b.settle(2))
        o = b.get(f"/api/v1/missions/{self.mid}/obstacles").json()["data"][0]
        self.assertEqual(o["x_m"], 5.0)                                # the late, older observation does not win
        b.send(obstacles_pkt(11, 30.0, ["trk-1", "trk-2", "dep-3_4"]), obstacles_pkt(12, 40.0, ["trk-1", "dep-3_4"]))
        self.assertTrue(b.settle(4))
        gone = b.get(f"/api/v1/missions/{self.mid}/obstacles", present=False).json()["data"]
        self.assertEqual([(g["obstacle_id"], g["removed"]["reason"]) for g in gone], [("trk-2", "full_snapshot")])
        b.send(P.encode_event(SESSION, 13, 50.0, "map_reset", value=1), obstacles_pkt(14, 45.0, ["trk-9"]))   # late, pre-reset
        self.assertTrue(b.settle(6))
        self.assertEqual(b.get(f"/api/v1/missions/{self.mid}/obstacles").json()["data"], [])
        reasons = {g["obstacle_id"]: g["removed"]["reason"] for g in b.get(f"/api/v1/missions/{self.mid}/obstacles", present=False).json()["data"]}
        self.assertEqual(reasons, {"trk-1": "map_reset", "dep-3_4": "map_reset", "trk-2": "full_snapshot"})
        self.assertEqual(b.get(f"/api/v1/missions/{self.mid}").json()["data"]["counters"]["map_resets"], 1)

    def test_two_missions_never_share_a_picture(self):
        """Regression (found in the live run): a backend receiving two missions in a row must reconcile each one on its
        own GroundMap - identical obstacle ids in mission B must not update or remove mission A's obstacles."""
        b = self.b
        b.send(obstacles_pkt(0, 10.0, ["trk-1", "trk-2"], session=111), obstacles_pkt(0, 5.0, ["trk-1"], session=222, x=50.0))
        self.assertTrue(b.settle(2))
        a = {o["obstacle_id"]: o for o in b.get("/api/v1/missions/111/obstacles").json()["data"]}
        c = {o["obstacle_id"]: o for o in b.get("/api/v1/missions/222/obstacles").json()["data"]}
        self.assertEqual(sorted(a), ["trk-1", "trk-2"])              # B's full snapshot did not remove A's trk-2
        self.assertEqual(sorted(c), ["trk-1"])
        self.assertEqual((a["trk-1"]["x_m"], c["trk-1"]["x_m"]), (0.0, 50.0))
        self.assertEqual(b.get("/api/v1/missions").json()["meta"]["pagination"]["total"], 2)

    def test_invalid_datagrams_never_persisted(self):
        b = self.b
        good = pose_pkt(20, 5.0)
        b.send(good[:-1] + bytes([good[-1] ^ 1]), b"garbage")          # CRC error, truncated
        b.send(good)
        self.assertTrue(b.settle(1))
        self.assertEqual(b.svc.receiver.stats["invalid"], 2)
        self.assertEqual(b.get(f"/api/v1/missions/{self.mid}/telemetry").json()["meta"]["pagination"]["total"], 1)

    def test_websocket_live_events_with_envelopes(self):
        b = self.b
        with b.client.websocket_connect(f"/ws/v1/missions/{self.mid}") as ws:
            snap = ws.receive_json()
            self.assertEqual(snap["type"], "mission.snapshot")
            b.send(pose_pkt(30, 6.0), obstacles_pkt(31, 6.1, ["trk-5"]), P.encode_event(SESSION, 32, 6.2, "health", "DEGRADED", []),
                   obstacles_pkt(33, 7.0, ["trk-6"]), P.encode_event(SESSION, 34, 8.0, "map_reset", value=1))
            got, end = [], time.time() + 5
            while time.time() < end and not {"map.reset", "obstacle.removed"} <= {g["type"] for g in got}:
                got.append(ws.receive_json())
        types = [g["type"] for g in got]
        for t in ("telemetry.received", "rover.position", "obstacle.detected", "obstacle.removed", "health.changed", "map.reset"):
            self.assertIn(t, types)
        pos = next(g for g in got if g["type"] == "rover.position")
        self.assertEqual((pos["api_version"], pos["mission_id"], pos["t_obs_s"]), ("v1", self.mid, 6.0))
        self.assertTrue(pos["rx_utc"].endswith("Z"))
        self.assertEqual(pos["data"]["frame"], "odom")
        rem = [g["data"] for g in got if g["type"] == "obstacle.removed"]
        self.assertIn({"obstacle_id": "trk-5", "reason": "full_snapshot"}, rem)    # trk-6's full snapshot replaced trk-5


@unittest.skipUnless(MONGO, "MongoDB not reachable")
class TestNetworkControl(unittest.TestCase):
    def test_presets_drive_the_real_emulator_and_are_local_only(self):
        em = LunarLinkEmulator(("127.0.0.1", 0), ("127.0.0.1", 9), ("127.0.0.1", 0), seed=1)
        stop = threading.Event()
        th = threading.Thread(target=em.run, args=(stop,), daemon=True)
        th.start()
        ctl = em.ctl.getsockname()
        b = None
        try:
            b = Backend(emulator_control=f"{ctl[0]}:{ctl[1]}")
            r = b.client.post("/api/v1/network/preset", json={"preset": "SEVERE"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["data"]["preset"], "SEVERE")
            self.assertEqual(em.cfg.loss, 0.5)                          # the emulator itself changed
            self.assertEqual(b.client.get("/api/v1/network/status").json()["data"]["preset"], "SEVERE")
            b.client.post("/api/v1/network/preset", json={"preset": "BLACKOUT"})
            self.assertTrue(em.cfg.blackout)
            b.client.post("/api/v1/network/preset", json={"preset": "RESTORE"})
            self.assertFalse(em.cfg.blackout)
            self.assertEqual(b.client.post("/api/v1/network/preset", json={"preset": "WARP"}).status_code, 422)
            remote = TestClient(b.app, client=("10.1.2.3", 5555))
            self.assertEqual(remote.post("/api/v1/network/preset", json={"preset": "NORMAL"}).json()["error"]["code"], "local_only")
        finally:
            if b:
                b.close()
            stop.set()
            th.join()

    def test_emulator_absent_is_503(self):
        b = Backend(emulator_control="127.0.0.1:9")
        try:
            r = b.client.get("/api/v1/network/status")
            self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "emulator_unavailable"))
        finally:
            b.close()


class TestDatabaseOutage(unittest.TestCase):
    """No MongoDB at all (unreachable URI): reception, ACKs and live WebSocket events continue; failures are observable."""

    def test_receiver_keeps_acking_and_streaming_without_database(self):
        ack = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ack.bind(("127.0.0.1", 0))
        ack.settimeout(2.0)
        a = ack.getsockname()
        b = Backend(mongo_uri="mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=300", mongo_timeout_ms=300, ack_to=f"{a[0]}:{a[1]}")
        b.dbname = None
        try:
            h = b.get("/health").json()["data"]
            self.assertEqual((h["status"], h["mongodb"]), ("degraded", "down"))
            with b.client.websocket_connect(f"/ws/v1/missions/{SESSION}") as ws:
                self.assertEqual(ws.receive_json()["data"]["source"], "unavailable")
                b.send(pose_pkt(1, 1.0), pose_pkt(2, 1.5))
                live = ws.receive_json()
                self.assertEqual(live["type"], "telemetry.received")    # still streamed live
            pk = P.decode(ack.recvfrom(4096)[0])
            self.assertEqual(pk.type, P.T_ACK)                           # still acknowledged
            self.assertTrue(wait(lambda: b.svc.stats["db_errors"] > 0, 5))
            self.assertFalse(b.svc.db_ok)
            self.assertIsNotNone(b.svc.stats["last_db_error"])
        finally:
            b.client.__exit__(None, None, None)
            b.tx.close()
            ack.close()


if __name__ == "__main__":
    unittest.main()
