"""Ground-station ingestion: R5.4 GroundReceiver (thread) -> bounded asyncio queue -> MongoDB writer (task).

    receiver thread      UDP + CRC + strict decode + ACKs (unchanged R5.4 code). For each VALID packet the
                         ``on_packet`` hook applies it to the mission's own R5.4 ``GroundMap`` (one per session: two
                         missions never share a picture, even with identical obstacle ids), computes the delta
                         (detected / updated / removed ids) and hands one item to the loop with ``call_soon_threadsafe``
                         - it never touches MongoDB, so reception and ACKs never wait on it.
    event loop           publishes the live WebSocket events at once, then enqueues the item (bounded; a full queue drops
                         the item and counts it - observable in /health and mission stats).
    writer task          one sequential writer (preserves arrival order): idempotent inserts (``_id = mission:seq``,
                         unique indexes), latest reconciled obstacle state, mission counters. On a database error it
                         retries the same item with backoff (reception continues; the queue absorbs the outage).
    link task            every ``link_metrics_period_s``: receiver counters + the emulator's live status (real UDP
                         control port) -> ``link_metrics`` and ``link.status`` / ``mission.stats`` events.

Only packets the receiver actually decoded are ingested: a packet the emulator dropped never reaches this code.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo.errors import DuplicateKeyError, PyMongoError

from collections import OrderedDict

from artemis_x.telemetry import protocol as P
from artemis_x.telemetry.ground_map import GroundMap
from artemis_x.telemetry.transport import GroundReceiver

MAX_SESSIONS = 16                               # reconciled pictures kept in memory (LRU); MongoDB keeps every mission

from . import db as D
from . import logging as L
from .hub import Hub, envelope

TYPE_NAMES = {P.T_POSE: "pose", P.T_OBSTACLES: "obstacles", P.T_EVENT: "event", P.T_ACK: "ack"}
log = L.get("artemis.ground.ingest")


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class IngestionService:
    def __init__(self, settings, database: D.Database, hub: Hub):
        self.s, self.db, self.hub = settings, database, hub
        self.queue: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.receiver: GroundReceiver | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tasks: list = []
        self.maps: OrderedDict = OrderedDict()   # session -> GroundMap (receiver thread only)
        self.db_ok = False
        self.current_mission: str | None = None
        self.stats = {"enqueued": 0, "dropped_queue_full": 0, "persisted": 0, "duplicates_ignored": 0, "db_errors": 0,
                      "db_retries": 0, "last_db_error": None, "events_published": 0}

    # ---- lifecycle ---------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(maxsize=self.s.ingest_queue_max)
        self.db_ok = await self.db.ping()
        if self.db_ok:
            try:
                await self.db.ensure_indexes()
            except PyMongoError as e:
                self.db_ok = False
                log.warning("index_creation_failed", error=str(e))
        else:
            log.warning("mongodb_unavailable_at_startup", uri_host=self.s.mongo_uri.split("@")[-1])
        self._tasks = [asyncio.create_task(self._writer(), name="ingest-writer"),
                       asyncio.create_task(self._link_loop(), name="link-metrics")]
        if self.s.receiver_enabled:
            self.receiver = GroundReceiver(self.s.udp_host, self.s.udp_port, Path(self.s.jsonl_out) if self.s.jsonl_out else None,
                                           timeout_s=0.2, ack_to=self.s.ack_addr, reconcile=False, on_packet=self._on_packet)
            self._thread = threading.Thread(target=self._receive_loop, name="ground-receiver", daemon=True)
            self._thread.start()
            log.info("receiver_started", udp=f"{self.s.udp_host}:{self.s.udp_port}", ack_to=self.s.ack_to)

    async def stop(self, drain_s: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 3.0)
        end = time.monotonic() + drain_s                    # let the writer persist what is queued, bounded
        while self.queue is not None and not self.queue.empty() and self.db_ok and time.monotonic() < end:
            await asyncio.sleep(0.05)
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("ingestion_stopped", left_in_queue=self.queue.qsize() if self.queue else 0, **self.stats_flat())

    def _receive_loop(self) -> None:
        rx = self.receiver
        try:
            while not self._stop.is_set():
                rx.receive_once()
        finally:
            rx.close()

    # ---- receiver thread -> loop ------------------------------------------------------------------------------------
    def _on_packet(self, pk: P.Packet, flag: str, rx_wall: float) -> None:
        """Runs in the receiver thread, after ACK bookkeeping and GroundMap.apply(). Never blocks."""
        delta = None
        gmap = self.maps.get(pk.session)
        if gmap is None:
            gmap = self.maps[pk.session] = GroundMap()
            while len(self.maps) > MAX_SESSIONS:
                self.maps.popitem(last=False)
        self.maps.move_to_end(pk.session)
        prev = dict(gmap.obs)
        gmap.apply(pk)                                      # dedup + observation-time ordering + full / reset rules
        if pk.type in (P.T_OBSTACLES, P.T_EVENT):
            now_obs = gmap.obs
            detected = [now_obs[i][0] for i in now_obs if i not in prev]
            updated = [now_obs[i][0] for i in now_obs if i in prev and now_obs[i][0] != prev[i][0]]   # content changed only
            removed = [i for i in prev if i not in now_obs]
            if detected or updated or removed:
                reason = "map_reset" if pk.type == P.T_EVENT else "full_snapshot"
                delta = {"detected": detected, "updated": updated, "removed": removed, "removed_reason": reason}
        item = {"session": pk.session, "seq": pk.seq, "type": pk.type, "t_obs_s": pk.t_obs_s, "flag": flag,
                "rx_utc": utc(rx_wall), "body": pk.body, "delta": delta, "size": None}
        self.loop.call_soon_threadsafe(self._accept, item)

    def _accept(self, item: dict) -> None:
        """Event loop: live events first (they are genuine receptions even if the database is down), then the queue."""
        mid = str(item["session"])
        self.current_mission = mid
        self._publish_live(mid, item)
        try:
            self.queue.put_nowait(item)
            self.stats["enqueued"] += 1
        except asyncio.QueueFull:
            self.stats["dropped_queue_full"] += 1

    def _publish_live(self, mid: str, it: dict) -> None:
        t, rx, b = it["t_obs_s"], it["rx_utc"], it["body"]
        pub = self.hub.publish
        pub(envelope("telemetry.received", mid, {"seq": it["seq"], "packet_type": TYPE_NAMES.get(it["type"]), "seq_flag": it["flag"]}, t, rx))
        if it["flag"] == "duplicate":
            return
        if it["type"] == P.T_POSE:
            pub(envelope("rover.position", mid, pose_doc(b), t, rx))
        elif it["type"] == P.T_EVENT:
            if b["event"] == "map_reset":
                pub(envelope("map.reset", mid, {"resets": b["value"], "state": b["state"]}, t, rx))
            elif b["event"] == "health":
                pub(envelope("health.changed", mid, {"state": b["state"], "reasons": b["reasons"]}, t, rx))
        d = it["delta"]
        if d:
            for o in d["detected"]:
                pub(envelope("obstacle.detected", mid, obstacle_view(o), t, rx))
            for o in d["updated"]:
                pub(envelope("obstacle.updated", mid, obstacle_view(o), t, rx))
            for i in d["removed"]:
                pub(envelope("obstacle.removed", mid, {"obstacle_id": i, "reason": d["removed_reason"]}, t, rx))
        self.stats["events_published"] = self.hub.stats["published"]

    # ---- writer ------------------------------------------------------------------------------------------------------
    async def _writer(self) -> None:
        backoff = 0.25
        while True:
            item = await self.queue.get()
            while True:
                try:
                    await self._persist(item)
                    if not self.db_ok:
                        log.info("mongodb_recovered")
                    self.db_ok, backoff = True, 0.25
                    break
                except PyMongoError as e:
                    self.stats["db_errors"] += 1
                    self.stats["last_db_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    if self.db_ok:
                        log.warning("mongodb_write_failed", error=self.stats["last_db_error"])
                    self.db_ok = False
                    self.stats["db_retries"] += 1
                    await asyncio.sleep(backoff)
                    backoff = min(5.0, backoff * 2)

    async def _persist(self, it: dict) -> None:
        mid, seq, t, rx, b = str(it["session"]), it["seq"], it["t_obs_s"], it["rx_utc"], it["body"]
        tname = TYPE_NAMES.get(it["type"], "unknown")
        doc = {"_id": f"{mid}:{seq}", "mission_id": mid, "seq": seq, "packet_type": tname, "kind": tname, "t_obs_s": t,
               "rx_utc": rx, "seq_flag": it["flag"], "frame": "odom",
               "provenance": {"transport": "udp", "receiver": f"{self.s.udp_host}:{self.s.udp_port}", "validated": "crc16+strict_decode"}}
        if it["type"] == P.T_EVENT:
            doc.update({"kind": b["event"], "event": b})
        elif it["type"] == P.T_OBSTACLES:
            doc.update({"snapshot": {"full": b.get("full", False), "part": b["part"], "parts": b["parts"],
                                     "map_resets": b["map_resets"], "records": len(b["obstacles"]),
                                     "ids": [o["id"] for o in b["obstacles"]]}})
        elif it["type"] == P.T_POSE:
            doc["pose"] = pose_doc(b)
        missions = self.db[D.MISSIONS]
        await missions.update_one({"_id": mid}, {"$setOnInsert": {"session": it["session"], "created_utc": rx, "frame": "odom",
                                                                  "first_t_obs_s": t, "config": self._mission_config()},
                                                 "$set": {"status": "active", "last_rx_utc": rx},
                                                 "$max": {"last_t_obs_s": t}}, upsert=True)
        try:
            await self.db[D.EVENTS].insert_one(doc)
        except DuplicateKeyError:                           # retransmission already stored: idempotent
            self.stats["duplicates_ignored"] += 1
            await missions.update_one({"_id": mid}, {"$inc": {"counters.duplicates": 1}})
            return
        inc = {"counters.packets": 1, f"counters.{tname}": 1}
        if it["type"] == P.T_POSE:
            try:
                await self.db[D.POSITIONS].insert_one({"_id": f"{mid}:{seq}", "mission_id": mid, "seq": seq, "t_obs_s": t,
                                                       "rx_utc": rx, **pose_doc(b)})
            except DuplicateKeyError:
                pass
        elif it["type"] == P.T_EVENT:
            if b["event"] == "map_reset":
                inc["counters.map_resets"] = 1
            if b["event"] in ("health", "map_reset"):
                await missions.update_one({"_id": mid}, {"$set": {"health": {"state": b["state"], "reasons": b["reasons"],
                                                                             "t_obs_s": t, "rx_utc": rx}}})
        d = it["delta"]
        if d:
            col = self.db[D.OBSTACLES]
            for o in d["detected"] + d["updated"]:
                v = obstacle_view(o)
                await col.update_one({"mission_id": mid, "obstacle_id": v["obstacle_id"]},
                                     {"$set": {**v, "present": True, "packet_t_obs_s": t, "last_rx_utc": rx, "last_seq": seq,
                                               "removed": None},
                                      "$setOnInsert": {"first_t_obs_s": t, "first_rx_utc": rx}}, upsert=True)
            for i in d["removed"]:
                await col.update_one({"mission_id": mid, "obstacle_id": i},
                                     {"$set": {"present": False, "removed": {"reason": d["removed_reason"], "packet_t_obs_s": t,
                                                                             "rx_utc": rx, "seq": seq}}})
        await missions.update_one({"_id": mid}, {"$inc": inc})
        self.stats["persisted"] += 1

    def _mission_config(self) -> dict:
        return {"frame": "odom", "receiver": f"{self.s.udp_host}:{self.s.udp_port}", "ack_to": self.s.ack_to,
                "emulator_control": self.s.emulator_control, "protocol": f"artemis telemetry v{P.VERSION}"}

    # ---- link metrics -------------------------------------------------------------------------------------------------
    async def _link_loop(self) -> None:
        from . import network
        last = None
        while True:
            await asyncio.sleep(self.s.link_metrics_period_s)
            mid = self.current_mission
            if mid is None or self.receiver is None:
                continue
            now = datetime.now(timezone.utc)
            rs = dict(self.receiver.stats)
            emu = None
            if self.s.emulator_addr:
                try:
                    st = await network.status(self.s.emulator_addr)
                    emu = {k: st.get(k) for k in ("preset", "link_state", "configured_loss", "delivery_ratio", "delay_ms",
                                                  "blackout_total_s", "received", "forwarded", "dropped_loss", "dropped_blackout",
                                                  "dropped_queue", "throughput_Bps")}
                except network.EmulatorUnavailable:
                    emu = None
            rate = None
            if last is not None:
                dt = (now - last[0]).total_seconds()
                rate = round((rs["valid"] - last[1]) / dt, 3) if dt > 0 else None
            last = (now, rs["valid"])
            age = (now - utc(rs["last_rx_wall_s"])).total_seconds() if rs.get("last_rx_wall_s") else None
            doc = {"mission_id": mid, "rx_utc": now,
                   "receiver": {k: rs.get(k) for k in ("datagrams", "valid", "invalid", "gaps", "missing", "duplicates",
                                                       "reordered", "acks_sent", "ack_seqs")},
                   "packets_per_s": rate, "last_rx_age_s": None if age is None else round(age, 3),
                   "link_up": age is not None and age < 2 * self.s.link_metrics_period_s,
                   "emulator": emu, "ingest": {"queue": self.queue.qsize(), "dropped": self.stats["dropped_queue_full"], "db_ok": self.db_ok}}
            self.hub.publish(envelope("link.status", mid, {k: v for k, v in doc.items() if k not in ("mission_id", "rx_utc")}, None, now))
            self.hub.publish(envelope("mission.stats", mid, self.stats_flat(), None, now))
            try:
                await self.db[D.LINK].insert_one(doc)
            except PyMongoError as e:
                self.stats["db_errors"] += 1
                self.stats["last_db_error"] = f"{type(e).__name__}: {str(e)[:160]}"

    def stats_flat(self) -> dict:
        r = self.receiver.stats if self.receiver else {}
        return {**{k: v for k, v in self.stats.items()}, "queue_size": self.queue.qsize() if self.queue else 0,
                "db_ok": self.db_ok, "receiver_valid": r.get("valid"), "receiver_invalid": r.get("invalid"),
                "acks_sent": r.get("acks_sent"), "ws_clients": self.hub.clients()}


def pose_doc(b: dict) -> dict:
    return {"frame": b["frame"], "x_m": b["x"], "y_m": b["y"], "yaw_rad": b["yaw"],
            "heading_deg": round(math.degrees(b["yaw"]), 3), "sigma_xy_m": b["sigma_xy"],
            "sigma_xy_saturated": b["sigma_xy_saturated"]}


def obstacle_view(o: dict) -> dict:
    return {"obstacle_id": o["id"], "type": o["type"], "frame": "odom", "x_m": o["x_m"], "y_m": o["y_m"],
            "radius_m": o["radius_m"], "confidence": round(o["confidence"], 4), "uncertainty_m": o["uncertainty_m"],
            "uncertainty_saturated": o.get("uncertainty_saturated", False), "obs_timestamp_s": o["timestamp_s"],
            "source": o["source"], "status": o["status"], "protected": o["protected"], "linked_track": o.get("linked_track"),
            "persistent_id": o["id"].startswith("trk-")}
