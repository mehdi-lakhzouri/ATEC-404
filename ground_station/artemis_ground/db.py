"""MongoDB access (PyMongo async API): the five mission collections and their indexes.

Idempotency: every packet-derived document has ``_id = "<mission_id>:<seq>"`` plus a unique (mission_id, seq) index, so
a retransmitted packet can never be stored twice; obstacles are unique per (mission_id, obstacle_id) and hold the latest
RECONCILED state (GroundMap), not raw packets.
Clock domains are kept apart: ``t_obs_s`` = rover observation time (simulation seconds), ``rx_utc`` = ground reception
time (UTC wall clock). Freshness is computed from ``rx_utc`` only.
"""

from __future__ import annotations

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, IndexModel

MISSIONS, POSITIONS, OBSTACLES, EVENTS, LINK = "missions", "rover_positions", "obstacles", "telemetry_events", "link_metrics"

INDEXES = {
    MISSIONS: [IndexModel([("status", ASCENDING), ("last_rx_utc", DESCENDING)], name="status_recent"),
               IndexModel([("session", ASCENDING)], name="session", unique=True)],
    POSITIONS: [IndexModel([("mission_id", ASCENDING), ("seq", ASCENDING)], name="mission_seq_unique", unique=True),
                IndexModel([("mission_id", ASCENDING), ("t_obs_s", DESCENDING)], name="mission_tobs"),
                IndexModel([("mission_id", ASCENDING), ("rx_utc", DESCENDING)], name="mission_rx")],
    OBSTACLES: [IndexModel([("mission_id", ASCENDING), ("obstacle_id", ASCENDING)], name="mission_obstacle_unique", unique=True),
                IndexModel([("mission_id", ASCENDING), ("present", ASCENDING), ("type", ASCENDING)], name="mission_present"),
                IndexModel([("mission_id", ASCENDING), ("last_rx_utc", DESCENDING)], name="mission_recent")],
    EVENTS: [IndexModel([("mission_id", ASCENDING), ("seq", ASCENDING)], name="mission_seq_unique", unique=True),
             IndexModel([("mission_id", ASCENDING), ("kind", ASCENDING), ("rx_utc", DESCENDING)], name="mission_kind_recent"),
             IndexModel([("mission_id", ASCENDING), ("t_obs_s", DESCENDING)], name="mission_tobs")],
    LINK: [IndexModel([("mission_id", ASCENDING), ("rx_utc", DESCENDING)], name="mission_recent")],
}


class Database:
    def __init__(self, uri: str, name: str, timeout_ms: int = 2000):
        self.client = AsyncMongoClient(uri, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms, tz_aware=True,
                                       appname="artemis-ground")
        self.db = self.client[name]
        self.name = name

    def __getitem__(self, col: str):
        return self.db[col]

    async def ping(self) -> bool:
        try:
            await self.client.admin.command("ping")
            return True
        except Exception:
            return False

    async def ensure_indexes(self) -> None:
        for col, idx in INDEXES.items():
            await self.db[col].create_indexes(idx)

    async def close(self) -> None:
        await self.client.close()
