"""Bounded WebSocket fan-out. Every event uses one envelope:

    {"type": "rover.position", "api_version": "v1", "mission_id": "...", "t_obs_s": 12.3 | null, "rx_utc": "ISO-8601",
     "data": {...}}

``t_obs_s`` is the rover's observation time (simulation seconds) and ``rx_utc`` the ground reception time; events that
are not tied to one observation (link.status, mission.stats) carry ``t_obs_s: null``. Each subscriber has its own bounded
queue: a slow client loses its OLDEST events (counted), never blocks ingestion or other clients.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from . import API_VERSION

EVENT_TYPES = ("rover.position", "obstacle.detected", "obstacle.updated", "obstacle.removed", "map.reset", "health.changed",
               "telemetry.received", "link.status", "mission.stats", "mission.snapshot")


def envelope(etype: str, mission_id: str, data: dict, t_obs_s: float | None = None, rx_utc: datetime | None = None) -> dict:
    assert etype in EVENT_TYPES, etype
    rx = rx_utc or datetime.now(timezone.utc)
    return {"type": etype, "api_version": API_VERSION, "mission_id": mission_id, "t_obs_s": t_obs_s,
            "rx_utc": rx.isoformat().replace("+00:00", "Z"), "data": data}


class Subscriber:
    def __init__(self, mission_id: str, maxsize: int):
        self.mission_id = mission_id
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0


class Hub:
    def __init__(self, queue_max: int = 512):
        self.queue_max = queue_max
        self.subs: dict[str, set[Subscriber]] = {}
        self.stats = {"published": 0, "delivered_to_queues": 0, "dropped_slow_clients": 0}

    def subscribe(self, mission_id: str) -> Subscriber:
        s = Subscriber(mission_id, self.queue_max)
        self.subs.setdefault(mission_id, set()).add(s)
        return s

    def unsubscribe(self, s: Subscriber) -> None:
        self.subs.get(s.mission_id, set()).discard(s)

    def publish(self, event: dict) -> None:
        """Called on the event loop; never awaits."""
        self.stats["published"] += 1
        for s in list(self.subs.get(event["mission_id"], ())) + list(self.subs.get("*", ())):
            if s.queue.full():
                try:
                    s.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                s.dropped += 1
                self.stats["dropped_slow_clients"] += 1
            s.queue.put_nowait(event)
            self.stats["delivered_to_queues"] += 1

    def clients(self) -> int:
        return sum(len(v) for v in self.subs.values())
