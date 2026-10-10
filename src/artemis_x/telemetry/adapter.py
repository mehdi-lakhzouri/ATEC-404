"""Telemetry event adapter (R5.1 -> R5.2 interface). Turns the live mapping state into protocol-v1 packets and queues
them for a future transport. No network here.

Non-blocking by design: ``tick`` is called by the controller on IDLE simulation steps (like the R4.5 monitor) and does at
most one job per call; packets go into a bounded queue; when the queue is full the OLDEST packet is dropped and counted
(fresh state is worth more than stale state on a lossy lunar link). The transport (R5.2) calls ``drain()``.

Duplicate physical observations: a depth cluster whose ``track_ids`` link it to a tracked object, and whose extent is
no larger than that track (radius + 2 cells), is the same physical object as the track -> it is not sent. Larger linked
clusters (e.g. a boulder field holding one small tracked rock) carry geometry the track does not -> they are sent, with
their ``linked_track`` so the receiver can still associate them.
"""

from __future__ import annotations

import collections

from . import protocol as P

DEDUP_MARGIN_M = 0.2                    # 2 cells of the 0.10 m grid
CRITICAL, NORMAL, LOW = 0, 1, 2         # R5.4 store-and-forward priorities (lower = more important)


def signature(o: dict) -> tuple:
    """What makes an observation worth re-sending: geometry, status, protection, confidence (0.01), uncertainty (1 cm).
    The timestamp is deliberately excluded - a re-detection that changes nothing else is not re-sent."""
    return (round(o["x_m"], 3), round(o["y_m"], 3), round(o["radius_m"], 3), o["status"], bool(o.get("protected")),
            round(o["confidence"], 2), round(o["uncertainty_m"], 2), tuple(o.get("track_ids") or ()))


def dedupe(doc: dict) -> tuple[list, int]:
    """Obstacles to send: all tracks, plus depth clusters that are NOT the same object as a track. Returns (list, dropped)."""
    tracks = {o["id"]: o for o in doc["obstacles"] if o["source"] == "yolo+depth"}
    out, dropped = [], 0
    for o in doc["obstacles"]:
        if o["source"] == "depth" and o.get("track_ids"):
            linked = [tracks[t] for t in o["track_ids"] if t in tracks]
            if linked and o["radius_m"] <= max(t["radius_m"] for t in linked) + DEDUP_MARGIN_M:
                dropped += 1
                continue
        out.append(o)
    return out, dropped


class TelemetryAdapter:
    def __init__(self, mapper, session_id: int, pose_period_s: float = 0.5, obstacle_period_s: float = 2.0,
                 max_queue: int = 256, dedup: bool = True, full_refresh_s: float | None = None):
        self.mapper, self.session = mapper, P._u32("session", session_id)
        self.pose_period, self.obs_period, self.dedup = pose_period_s, obstacle_period_s, dedup
        self.queue: collections.deque = collections.deque()
        self.max_queue = max_queue
        self.seq = 0
        self.next_pose = self.next_obs = 0.0
        self.full_refresh_s = full_refresh_s     # None: every snapshot is full (R5.1 behaviour); else changes only + periodic full
        self._sent: dict = {}                    # obstacle id -> signature last sent
        self._next_full = 0.0
        self._health = None                      # (state, reasons) last sent
        self._resets = mapper.grid.resets
        self.stats = {"packets": 0, "bytes": 0, "dropped_queue_full": 0, "deduplicated": 0,
                      "by_type": {"pose": 0, "obstacles": 0, "event": 0}, "encode_errors": 0,
                      "snapshots": 0, "full_snapshots": 0, "unchanged_skipped": 0}

    # ---- producer side (controller, idle steps) ------------------------------------------------------------------------
    def tick(self, t: float) -> None:
        """At most one job per call: an event if the health state / map changed, else a due pose, else a due snapshot."""
        m = self.mapper
        if m.grid.resets != self._resets:
            self._resets = m.grid.resets
            self._push([P.encode_event(self.session, self._next(), t, "map_reset", self._state(), value=min(0xFFFF, m.grid.resets))], "event",
                       CRITICAL, {"event": "map_reset"})
            self.next_obs = t                   # the map is empty now: tell the receiver at once
            self._sent, self._next_full = {}, t                       # ... with a full (empty) snapshot
            return
        h = m.health_monitor.latest
        if h is not None and (h.state, tuple(h.reasons)) != self._health:
            state_changed = self._health is None or self._health[0] != h.state
            self._health = (h.state, tuple(h.reasons))
            self._push([P.encode_event(self.session, self._next(), h.stamp_s, "health",
                                       h.state, [r for r in h.reasons if r in P.REASONS])], "event",
                       CRITICAL if state_changed else NORMAL, {"event": "health", "state": h.state})
            return
        if t >= self.next_pose:
            self.next_pose = t + self.pose_period
            self._push([P.encode_pose(self.session, self._next(), m.estimator.pose().to_dict())], "pose", LOW, {})
            return
        if t >= self.next_obs:
            self.next_obs = t + self.obs_period
            full = self.full_refresh_s is None or t >= self._next_full
            if full and self.full_refresh_s is not None:
                self._next_full = t + self.full_refresh_s
            self.push_snapshot(self._export(), full=full)

    def push_snapshot(self, doc: dict, full: bool = True) -> None:
        """full: every (deduplicated) obstacle; otherwise only obstacles that are new or changed since they were last sent.
        A changes-only snapshot with no change sends nothing. A full snapshot is always sent, even when empty (it is how
        a receiver learns the map is empty, e.g. after a reset)."""
        obs, dropped = dedupe(doc) if self.dedup else (doc["obstacles"], 0)
        self.stats["deduplicated"] += dropped
        if not full:
            changed = [o for o in obs if self._sent.get(o["id"]) != signature(o)]
            self.stats["unchanged_skipped"] += len(obs) - len(changed)
            if not changed:
                return
            obs = changed
        else:
            self.stats["full_snapshots"] += 1
        # R5.4 priority, per packet: a part that carries a hazard never sent before in this map epoch (a newly confirmed
        # track, or a new depth obstacle / negative hazard) is CRITICAL; anything else is a NORMAL refresh
        new = {o["id"] for o in obs if o["id"] not in self._sent and (o["source"] == "depth" or o["status"] == "confirmed")}
        for o in obs:
            self._sent[o["id"]] = signature(o)
        self.stats["snapshots"] += 1
        pkts = P.encode_obstacles(self.session, self.seq, doc, obs, full=full)
        self.seq = (self.seq + len(pkts)) & 0xFFFFFFFF
        for k, pkt in enumerate(pkts):
            part_new = [o["id"] for o in obs[k * P.MAX_RECORDS:(k + 1) * P.MAX_RECORDS] if o["id"] in new]
            part_ids = [o["id"] for o in obs[k * P.MAX_RECORDS:(k + 1) * P.MAX_RECORDS]]
            self._push([pkt], "obstacles", CRITICAL if part_new else NORMAL,
                       {"full": full, "new_hazards": part_new, "ids": part_ids, "snapshot_t_obs": doc["stamp_s"]})

    # ---- consumer side (R5.2 transport) ---------------------------------------------------------------------------------
    def drain(self, max_packets: int | None = None) -> list:
        """Packets only (R5.2 direct transport)."""
        return [pkt for pkt, _ in self.drain_meta(max_packets)]

    def drain_meta(self, max_packets: int | None = None) -> list:
        """(packet, meta) pairs; meta: kind, prio (CRITICAL / NORMAL / LOW), seq, t_obs_s and kind-specific fields (R5.4)."""
        n = len(self.queue) if max_packets is None else min(max_packets, len(self.queue))
        return [self.queue.popleft() for _ in range(n)]

    # ---- internals ---------------------------------------------------------------------------------------------------------
    def _export(self) -> dict:
        from ..sim.mapping.contract import export_obstacles
        return export_obstacles(self.mapper)

    def _state(self) -> str:
        h = self.mapper.health_monitor.latest
        return h.state if h is not None else "OK"

    def _next(self) -> int:
        s = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        return s

    def _push(self, pkts: list, kind: str, prio: int = NORMAL, extra: dict | None = None) -> None:
        for p in pkts:
            if len(self.queue) >= self.max_queue:
                self.queue.popleft()
                self.stats["dropped_queue_full"] += 1
            _, _, _, _, seq, t_ms, _ = P.HEADER.unpack_from(p)
            self.queue.append((p, {"kind": kind, "prio": prio, "seq": seq, "t_obs_s": t_ms / 1000.0, **(extra or {})}))
            self.stats["packets"] += 1
            self.stats["bytes"] += len(p)
        self.stats["by_type"][kind] += len(pkts)
