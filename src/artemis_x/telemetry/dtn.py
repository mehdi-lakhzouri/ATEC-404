"""R5.4 DTN-inspired store-and-forward for rover telemetry (standard library only).

Delivery is confirmed ONLY by acknowledgements that come back from the ground over the (simulated) reverse link: a
successful ``sendto`` proves nothing. Without ACKs nothing here claims delivery.

``StoreForwardQueue`` - pure logic, explicit clock (``now`` arguments), fully unit-testable:
    priorities   CRITICAL (map resets, health STATE changes, obstacle parts with a newly confirmed hazard)
                 NORMAL   (obstacle refreshes, health reason changes)       LOW (poses)
    retention    CRITICAL and NORMAL are kept until ACKed, expired (TTL / attempt cap) or evicted by the memory limits;
                 LOW is fire-and-forget and COALESCED: only the newest pose waits, older ones are discarded as stale
    superseding  a FULL obstacle snapshot makes older pending NORMAL obstacle packets obsolete (it carries the current
                 state); CRITICAL ones are kept until ACKed
    retries      adaptive RTO (RFC 6298 smoothing) with Karn's algorithm: RTT samples only from first transmissions,
                 AND the RTO itself backs off on a timeout retransmission (rate-limited to once per RTO) - otherwise an
                 RTO below the real RTT (e.g. 1 s vs a 2.6 s lunar round trip) could never grow; exponential per-entry
                 backoff, capped
    coverage     a CRITICAL hazard packet whose hazards are delivered by an ACKed newer FULL snapshot is redundant and
                 is purged (``covered_by_full``)
    memory       bounded entries and bytes; eviction order LOW, then NORMAL, then CRITICAL - and within CRITICAL,
                 redundant hazard packets (every hazard also carried by a newer pending obstacle packet) before
                 anything else, then the oldest; map-reset events last. All counted
    link         when the link is believed DOWN (no ACK for ``down_after_s`` after a send) only one probe per
                 ``probe_interval_s`` is sent, highest priority first; the first ACK brings the link back UP and the
                 backlog drains in priority order, bounded per tick
``DtnLink`` - rover glue: adapter -> queue -> non-blocking UDP; reads ACKs from its own non-blocking socket. Every call
is bounded; nothing waits on the network, so mapping and motor control are never blocked.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field

from . import protocol as P
from .adapter import CRITICAL, LOW, NORMAL

PRIO_NAMES = {CRITICAL: "critical", NORMAL: "normal", LOW: "low"}


def _q(xs, p):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 3) if s else None


@dataclass
class Entry:
    seq: int
    pkt: bytes
    prio: int
    kind: str
    meta: dict
    created: float
    expires: float
    next_tx: float = 0.0
    attempts: int = 0
    first_tx: float | None = None
    last_tx: float | None = None
    rto0: float | None = None                   # RTO at the first transmission: this entry's backoff base


class RttEstimator:
    """RFC 6298: srtt / rttvar smoothing, rto = srtt + 4 rttvar, clamped to [min_rto, max_rto]."""

    def __init__(self, initial_rto: float = 1.0, min_rto: float = 0.3, max_rto: float = 10.0):
        self.srtt = self.rttvar = None
        self.rto, self.min_rto, self.max_rto = initial_rto, min_rto, max_rto
        self.samples = 0

    def sample(self, rtt: float) -> None:
        if self.srtt is None:
            self.srtt, self.rttvar = rtt, rtt / 2
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - rtt)
            self.srtt = 0.875 * self.srtt + 0.125 * rtt
        self.rto = min(self.max_rto, max(self.min_rto, self.srtt + 4 * self.rttvar))
        self.samples += 1

    def backoff(self, now: float) -> bool:
        """Karn: a timeout doubles the RTO (kept until a clean sample); at most once per current RTO."""
        if now - getattr(self, "_last_backoff", float("-inf")) < self.rto:
            return False
        self._last_backoff = now
        self.rto = min(self.max_rto, self.rto * 2)
        self.backoffs = getattr(self, "backoffs", 0) + 1
        return True


@dataclass
class QueueConfig:
    max_entries: int = 400
    max_bytes: int = 192 * 1024
    ttl_s: dict = field(default_factory=lambda: {CRITICAL: 600.0, NORMAL: 60.0, LOW: 3.0})
    max_attempts: dict = field(default_factory=lambda: {CRITICAL: 60, NORMAL: 8, LOW: 1})
    backoff_cap_s: float = 16.0
    down_after_s: float = 2.0
    probe_interval_s: float = 1.0
    budget_per_tick: int = 16


class StoreForwardQueue:
    def __init__(self, cfg: QueueConfig | None = None, rtt: RttEstimator | None = None):
        self.cfg = cfg or QueueConfig()
        self.rtt = rtt or RttEstimator()
        self.pending: dict = {}                 # seq -> Entry
        self.bytes = 0
        self.next_probe = 0.0
        self.lat: dict = {p: [] for p in PRIO_NAMES}
        z = {n: 0 for n in PRIO_NAMES.values()}
        self.st = {"offered": dict(z), "sent": 0, "retries": 0, "probes": 0, "acked": dict(z), "expired": dict(z),
                   "evicted": dict(z), "superseded": 0, "coalesced": 0, "acks_unknown": 0, "fire_and_forget": 0,
                   "covered_by_full": 0, "redundant_evicted": 0,
                   "peak_entries": 0, "peak_bytes": 0}

    # ---- producer ---------------------------------------------------------------------------------------------------------
    def offer(self, pkt: bytes, meta: dict, now: float) -> Entry:
        prio = meta["prio"]
        e = Entry(meta["seq"], pkt, prio, meta["kind"], meta, now, now + self.cfg.ttl_s[prio], next_tx=now)
        self.st["offered"][PRIO_NAMES[prio]] += 1
        if prio == LOW:                         # newest pose wins: older unsent poses are stale
            for old in [x for x in self.pending.values() if x.prio == LOW]:
                self._remove(old.seq)
                self.st["coalesced"] += 1
        if e.kind == "obstacles" and meta.get("full"):     # a full snapshot (any of its parts) obsoletes older refreshes
            t_new = meta.get("snapshot_t_obs", 0.0)
            for old in [x for x in self.pending.values()
                        if x.kind == "obstacles" and x.prio == NORMAL and x.meta.get("snapshot_t_obs", 0.0) < t_new]:
                self._remove(old.seq)
                self.st["superseded"] += 1
        self.pending[e.seq] = e
        self.bytes += len(pkt)
        self._enforce_limits()
        self.st["peak_entries"] = max(self.st["peak_entries"], len(self.pending))
        self.st["peak_bytes"] = max(self.st["peak_bytes"], self.bytes)
        return e

    def _enforce_limits(self) -> None:
        c = self.cfg
        while self.pending and (len(self.pending) > c.max_entries or self.bytes > c.max_bytes):
            worst = max(x.prio for x in self.pending.values())
            cands = [x for x in self.pending.values() if x.prio == worst]
            if worst == CRITICAL:
                red = [x for x in cands if self._redundant(x)]
                if red:
                    victim = min(red, key=lambda x: (x.created, x.seq))
                    self.st["redundant_evicted"] += 1
                else:                            # oldest health / hazard first, map-reset events last
                    victim = min(cands, key=lambda x: (x.meta.get("event") == "map_reset", x.created, x.seq))
            else:
                victim = min(cands, key=lambda x: (x.created, x.seq))
            self._remove(victim.seq)
            self.st["evicted"][PRIO_NAMES[victim.prio]] += 1

    def _redundant(self, e: Entry) -> bool:
        """A CRITICAL hazard packet is redundant when every hazard it announces is also in a NEWER pending obstacle packet."""
        hz = e.meta.get("new_hazards")
        if e.kind != "obstacles" or not hz:
            return False
        newer = set()
        for x in self.pending.values():
            if x.kind == "obstacles" and x.seq != e.seq and x.meta.get("snapshot_t_obs", 0.0) > e.meta.get("snapshot_t_obs", 0.0):
                newer.update(x.meta.get("ids", ()))
        return set(hz) <= newer

    def _remove(self, seq: int) -> Entry | None:
        e = self.pending.pop(seq, None)
        if e is not None:
            self.bytes -= len(e.pkt)
        return e

    # ---- scheduling -------------------------------------------------------------------------------------------------------
    def expire(self, now: float) -> None:
        for e in list(self.pending.values()):
            if now >= e.expires or (e.attempts >= self.cfg.max_attempts[e.prio] and now >= e.next_tx):
                self._remove(e.seq)
                self.st["expired"][PRIO_NAMES[e.prio]] += 1

    def due(self, now: float, link_up: bool) -> list:
        if link_up:
            ready = sorted((e for e in self.pending.values() if e.next_tx <= now), key=lambda e: (e.prio, e.seq))
            return ready[:self.cfg.budget_per_tick]
        if self.pending and now >= self.next_probe:   # link believed down: one probe per interval, most important first,
            self.next_probe = now + self.cfg.probe_interval_s   # regardless of its own backoff (detect restoration fast)
            self.st["probes"] += 1
            return [min(self.pending.values(), key=lambda e: (e.prio, e.seq))]
        return []

    def on_sent(self, e: Entry, now: float) -> None:
        e.attempts += 1
        self.st["sent"] += 1
        if e.attempts > 1:
            self.st["retries"] += 1
            self.rtt.backoff(now)                # timeout retransmission: grow the RTO (Karn)
        e.first_tx = e.first_tx if e.first_tx is not None else now
        e.rto0 = e.rto0 if e.rto0 is not None else self.rtt.rto
        e.last_tx = now
        if e.prio == LOW:                       # best effort: never retried
            self._remove(e.seq)
            self.st["fire_and_forget"] += 1
            return
        backoff = min(self.cfg.backoff_cap_s, e.rto0 * (2 ** (e.attempts - 1)))
        e.next_tx = now + backoff

    def on_ack(self, seq: int, now: float) -> Entry | None:
        e = self._remove(seq)
        if e is None:
            self.st["acks_unknown"] += 1        # duplicate ACK, or an ACK for a fire-and-forget pose
            return None
        if e.attempts == 1 and e.first_tx is not None:   # Karn: only unambiguous samples
            self.rtt.sample(now - e.first_tx)
        self.st["acked"][PRIO_NAMES[e.prio]] += 1
        if e.kind == "obstacles" and e.meta.get("full"):          # the ground now holds these ids as of this snapshot
            ids, t_snap = set(e.meta.get("ids", ())), e.meta.get("snapshot_t_obs", 0.0)
            for x in [x for x in self.pending.values() if x.kind == "obstacles" and x.prio == CRITICAL
                      and x.meta.get("snapshot_t_obs", 0.0) <= t_snap and set(x.meta.get("new_hazards", ())) <= ids]:
                self._remove(x.seq)
                self.st["covered_by_full"] += 1
        lat = self.lat[e.prio]
        if len(lat) < 20000:
            lat.append(now - e.created)         # storage -> confirmed delivery (includes outage time)
        return e

    def stats(self) -> dict:
        by = {n: 0 for n in PRIO_NAMES.values()}
        for e in self.pending.values():
            by[PRIO_NAMES[e.prio]] += 1
        return {**self.st, "pending": by, "pending_total": len(self.pending), "pending_critical": by["critical"],
                "queue_bytes": self.bytes, "rto_s": round(self.rtt.rto, 3), "rto_backoffs": getattr(self.rtt, "backoffs", 0),
                "srtt_s": None if self.rtt.srtt is None else round(self.rtt.srtt, 4), "rtt_samples": self.rtt.samples,
                "delivery_latency_s": {PRIO_NAMES[p]: {"n": len(v), "p50": _q(v, 0.5), "p95": _q(v, 0.95),
                                                       "max": round(max(v), 3) if v else None} for p, v in self.lat.items()}}


class DtnLink:
    """Rover side: adapter packets -> StoreForwardQueue -> UDP (to the emulator or the ground); ACKs on ``ack_listen``."""

    def __init__(self, adapter, dest=("127.0.0.1", 5791), ack_listen=("127.0.0.1", 5793), cfg: QueueConfig | None = None,
                 clock=time.perf_counter, log_max: int = 40000):
        from .transport import UdpSender
        self.adapter, self.clock = adapter, clock
        self.q = StoreForwardQueue(cfg)
        self.sender = UdpSender(*dest)
        self.ack = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack.bind(ack_listen)
        self.ack.setblocking(False)
        self.last_ack = None
        self.first_unacked_send = None
        self.link_up = True
        self.outages: list = []                 # [down_at, up_at] in clock seconds since start
        self.t0 = clock()
        self.log: list = []
        self.log_max = log_max
        self.st = {"acks_received": 0, "acks_invalid": 0, "acks_other_session": 0, "acked_seqs": 0}
        self.hazards: dict = {}                 # seq -> hazards announced (CRITICAL obstacle packets), bounded by log_max

    def tick(self, t_sim: float) -> None:
        now = self.clock()
        self._read_acks(now)
        for pkt, meta in self.adapter.drain_meta():
            if meta.get("new_hazards") and len(self.hazards) < self.log_max:
                self.hazards[meta["seq"]] = list(meta["new_hazards"])
            self.q.offer(pkt, meta, now)
        self.q.expire(now)
        self._update_link(now)
        for e in self.q.due(now, self.link_up):
            ok = self.sender.send(e.pkt)
            attempt = e.attempts + 1
            self.q.on_sent(e, now)
            if ok and self.first_unacked_send is None:
                self.first_unacked_send = now
            if len(self.log) < self.log_max:
                self.log.append((e.seq, P.HEADER.unpack_from(e.pkt)[2], round(e.meta.get("t_obs_s", 0.0), 3), round(t_sim, 3),
                                 round(time.time(), 6), ok, attempt, e.prio))

    def _read_acks(self, now: float) -> None:
        for _ in range(64):
            try:
                data, _ = self.ack.recvfrom(4096)
            except (BlockingIOError, ConnectionResetError):
                return
            except OSError:
                return
            try:
                pk = P.decode(data)
            except P.ProtocolError:
                self.st["acks_invalid"] += 1
                continue
            if pk.type != P.T_ACK:
                self.st["acks_invalid"] += 1
                continue
            if pk.body["acked_session"] != self.adapter.session:
                self.st["acks_other_session"] += 1
                continue
            self.st["acks_received"] += 1
            for s in pk.body["seqs"]:
                self.st["acked_seqs"] += 1
                self.q.on_ack(s, now)
            self.last_ack, self.first_unacked_send = now, None   # any valid ACK is evidence the link works

    def _update_link(self, now: float) -> None:
        down = self.first_unacked_send is not None and now - self.first_unacked_send > self.q.cfg.down_after_s
        if self.link_up and down:
            self.link_up = False
            self.outages.append([round(now - self.t0, 3), None])
        elif not self.link_up and not down:
            self.link_up = True
            self.outages[-1][1] = round(now - self.t0, 3)

    def finish(self, grace_s: float = 3.0) -> dict:
        """End of mission: keep draining and reading ACKs for up to ``grace_s`` (wall clock) while CRITICAL or NORMAL
        entries are pending; whatever is still pending afterwards is reported as undelivered, never as delivered."""
        end = self.clock() + grace_s
        while self.clock() < end and any(e.prio != LOW for e in self.q.pending.values()):
            self.tick(-1.0)
            time.sleep(0.02)
        st = self.stats()
        for sk in (self.sender, ):
            sk.close()
        try:
            self.ack.close()
        except OSError:
            pass
        return st

    def stats(self) -> dict:
        return {**self.q.stats(), **self.st, "link_state": "UP" if self.link_up else "DOWN", "outages_detected": self.outages,
                "sender": self.sender.stats}
