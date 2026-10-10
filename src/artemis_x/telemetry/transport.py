"""R5.2 UDP transport for protocol-v1 packets (standard library only). No retransmission, no acknowledgement, by design.

``UdpSender``   rover side. Non-blocking socket; never raises into the controller. A missing or restarting receiver costs
                nothing but counted send errors (on Windows a closed localhost port answers with ICMP "port unreachable",
                surfaced as ConnectionResetError on a later call - caught and counted). Independent of the YOLO TCP link.
``GroundReceiver``  ground side. Validates and decodes every datagram (CRC first), stamps the RECEPTION time (wall clock),
                tracks gaps / duplicates / reordering per session, counts invalid datagrams, and writes one JSON line per
                datagram. Never crashes on bad input.
``LiveTelemetry``   controller glue: TelemetryAdapter + UdpSender + a bounded send log (for latency on a shared clock).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

from . import protocol as P

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 5790


class UdpSender:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, max_per_flush: int = 32):
        self.addr = (host, port)
        self.max_per_flush = max_per_flush
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.stats = {"sent": 0, "bytes": 0, "errors": 0, "error_kinds": {}}

    def send(self, pkt: bytes) -> bool:
        try:
            self.sock.sendto(pkt, self.addr)
        except OSError as e:                     # receiver absent (ICMP reset), buffer full, ... -> drop, count, continue
            self.stats["errors"] += 1
            k = type(e).__name__
            self.stats["error_kinds"][k] = self.stats["error_kinds"].get(k, 0) + 1
            return False
        self.stats["sent"] += 1
        self.stats["bytes"] += len(pkt)
        return True

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class LiveTelemetry:
    """Controller-side glue. ``tick(t)`` on idle simulation steps: one adapter job, then send what is queued (bounded)."""

    def __init__(self, mapper, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, session_id: int | None = None,
                 pose_period_s: float = 0.5, obstacle_period_s: float = 2.0, full_refresh_s: float = 10.0, log_max: int = 20000,
                 dtn: dict | None = None):
        from .adapter import TelemetryAdapter
        session = session_id if session_id is not None else int(time.time()) & 0xFFFFFFFF
        self.adapter = TelemetryAdapter(mapper, session, pose_period_s, obstacle_period_s, full_refresh_s=full_refresh_s,
                                        max_queue=4096 if dtn is not None else 256)
        self.dtn = None
        if dtn is not None:                       # R5.4: store-and-forward with ACKs over the reverse link
            from .dtn import DtnLink, QueueConfig
            cfg = QueueConfig(**{k: v for k, v in dtn.items() if k in ("max_entries", "max_bytes")})
            self.dtn = DtnLink(self.adapter, (host, port), ("127.0.0.1", dtn.get("ack_port", 5793)), cfg)
            self.sender = self.dtn.sender
        else:
            self.sender = UdpSender(host, port)
        self.log: list = []                      # (seq, type, t_obs_s, t_sim_send_s, wall_send_s, ok) - bounded
        self.log_max = log_max
        self.first_send_wall = None

    @classmethod
    def from_env(cls, mapper):
        if mapper is None or os.environ.get("ARTEMIS_TELEMETRY", "0") != "1":
            return None
        s = os.environ.get("ARTEMIS_TELEMETRY_SESSION")
        dtn = None
        if os.environ.get("ARTEMIS_TELEMETRY_DTN", "0") == "1":
            dtn = {"ack_port": int(os.environ.get("ARTEMIS_TELEMETRY_ACK_PORT", 5793))}
            for k, env in (("max_entries", "ARTEMIS_DTN_MAX_ENTRIES"), ("max_bytes", "ARTEMIS_DTN_MAX_BYTES")):
                if os.environ.get(env):
                    dtn[k] = int(os.environ[env])
        return cls(mapper, os.environ.get("ARTEMIS_TELEMETRY_HOST", DEFAULT_HOST), int(os.environ.get("ARTEMIS_TELEMETRY_PORT", DEFAULT_PORT)),
                   session_id=int(s) if s else None, dtn=dtn)

    def tick(self, t: float) -> None:
        try:
            self.adapter.tick(t)
        except P.ProtocolError:                  # a value outside the protocol's range: drop that job, keep the mission
            self.adapter.stats["encode_errors"] += 1
        if self.dtn is not None:
            self.dtn.tick(t)
        else:
            self._flush(t)

    def _flush(self, t: float, limit: int | None = None) -> None:
        for pkt in self.adapter.drain(limit or self.sender.max_per_flush):
            wall = time.time()
            ok = self.sender.send(pkt)
            if self.first_send_wall is None and ok:
                self.first_send_wall = wall
            if len(self.log) < self.log_max:
                _, _, ptype, _, seq, t_ms, _ = P.HEADER.unpack_from(pkt)
                self.log.append((seq, ptype, t_ms / 1000.0, round(t, 3), round(wall, 6), ok))

    def finish(self, t: float, out_dir: Path) -> dict:
        """Final snapshot + flush of everything queued, then the send log and stats (never raises)."""
        try:
            self.adapter.push_snapshot(self.adapter._export(), full=True)
        except P.ProtocolError:
            self.adapter.stats["encode_errors"] += 1
        dtn_stats = None
        if self.dtn is not None:
            self.dtn.tick(t)
            dtn_stats = self.dtn.finish(grace_s=3.0)
            self.log = self.dtn.log
            self.first_send_wall = self.log[0][4] if self.log else None
        else:
            while self.adapter.queue:
                self._flush(t, 256)
            self.sender.close()
        s = {"session": self.adapter.session, "addr": list(self.sender.addr), "adapter": self.adapter.stats,
             "sender": self.sender.stats, "first_send_wall_s": self.first_send_wall, "log_entries": len(self.log), "dtn": dtn_stats}
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            cols = ["seq", "type", "t_obs_s", "t_sim_send_s", "wall_send_s", "ok"] + (["attempt", "prio"] if self.dtn is not None else [])
            doc = {"columns": cols, "rows": self.log}
            if self.dtn is not None:
                doc["critical_hazards"] = {str(k): v for k, v in self.dtn.hazards.items()}
            (out_dir / "telemetry_send_log.json").write_text(json.dumps(doc), encoding="utf-8")
        except OSError:
            pass
        return s


class GroundReceiver:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, out: Path | None = None, timeout_s: float = 0.5,
                 ack_to: tuple | None = None, ack_every_s: float = 0.1, reconcile: bool = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(min(timeout_s, ack_every_s) if ack_to else timeout_s)
        self.ack_to, self.ack_every = ack_to, ack_every_s       # R5.4: selective ACKs back over the reverse link
        self.ack_pending: dict = {}              # session -> [seq, ...]
        self.ack_seq = 0
        self.ack_last = time.perf_counter()
        self.ack_t_obs = 0.0
        from .ground_map import GroundMap
        self.map = GroundMap() if reconcile else None
        self.out = open(out, "a", encoding="utf-8") if out else None
        self.sessions: dict = {}                 # session -> {"last": seq, "seen": set, ...}
        self.stats = {"datagrams": 0, "bytes": 0, "valid": 0, "invalid": 0, "invalid_reasons": {}, "gaps": 0, "missing": 0,
                      "duplicates": 0, "reordered": 0, "by_type": {"pose": 0, "obstacles": 0, "event": 0, "ack": 0},
                      "acks_sent": 0, "ack_seqs": 0, "ack_send_errors": 0,
                      "first_rx_wall_s": None, "last_rx_wall_s": None}

    def receive_once(self) -> P.Packet | None:
        """Wait up to the socket timeout for one datagram; returns the decoded packet, or None (timeout or invalid)."""
        try:
            data, _ = self.sock.recvfrom(65535)
        except (socket.timeout, TimeoutError):
            self._flush_acks(False)
            return None
        except ConnectionResetError:             # Windows: ICMP from an earlier datagram - ignore
            return None
        pk = self.handle(data, time.time())
        self._flush_acks(False)
        return pk

    def _flush_acks(self, force: bool) -> None:
        if not self.ack_to or not self.ack_pending:
            return
        if not force and time.perf_counter() - self.ack_last < self.ack_every and all(len(v) < P.MAX_ACK for v in self.ack_pending.values()):
            return
        for session, seqs in self.ack_pending.items():
            for i in range(0, len(seqs), P.MAX_ACK):
                pkt = P.encode_ack(self.ack_seq, session, seqs[i:i + P.MAX_ACK], self.ack_t_obs)
                self.ack_seq = (self.ack_seq + 1) & 0xFFFFFFFF
                try:
                    self.sock.sendto(pkt, self.ack_to)
                    self.stats["acks_sent"] += 1
                    self.stats["ack_seqs"] += len(seqs[i:i + P.MAX_ACK])
                except OSError:
                    self.stats["ack_send_errors"] += 1
        self.ack_pending = {}
        self.ack_last = time.perf_counter()

    def handle(self, data: bytes, rx_wall: float) -> P.Packet | None:
        st = self.stats
        st["datagrams"] += 1
        st["bytes"] += len(data)
        st["first_rx_wall_s"] = st["first_rx_wall_s"] or rx_wall
        st["last_rx_wall_s"] = rx_wall
        try:
            pk = P.decode(data, rx_time_s=rx_wall)
        except P.ProtocolError as e:
            st["invalid"] += 1
            reason = str(e).split(":")[0][:40]
            st["invalid_reasons"][reason] = st["invalid_reasons"].get(reason, 0) + 1
            self._write({"rx_wall_s": round(rx_wall, 6), "invalid": str(e), "len": len(data)})
            return None
        st["valid"] += 1
        st["by_type"][{P.T_POSE: "pose", P.T_OBSTACLES: "obstacles", P.T_EVENT: "event", P.T_ACK: "ack"}[pk.type]] += 1
        if pk.type == P.T_ACK:                   # not for the ground; ignore (never acknowledged)
            return pk
        if self.ack_to:                          # acknowledge duplicates too: their first ACK may have been lost
            self.ack_pending.setdefault(pk.session, []).append(pk.seq)
            self.ack_t_obs = max(self.ack_t_obs, pk.t_obs_s)
        if self.map is not None:
            self.map.apply(pk)
        flag = self._sequence(pk.session, pk.seq)
        self._write({"rx_wall_s": round(rx_wall, 6), "session": pk.session, "seq": pk.seq, "type": pk.type,
                     "t_obs_s": pk.t_obs_s, "seq_flag": flag, "body": pk.body})
        return pk

    def _sequence(self, session: int, seq: int) -> str:
        s = self.sessions.setdefault(session, {"last": None, "seen": set()})
        if seq in s["seen"]:
            self.stats["duplicates"] += 1
            return "duplicate"
        s["seen"].add(seq)
        if len(s["seen"]) > 100000:              # bounded memory: keep a recent window only
            s["seen"] = {x for x in s["seen"] if (seq - x) % 2 ** 32 < 50000}
        last, s_ = s["last"], s
        if last is None:
            s_["last"] = seq
            return "first"
        d = (seq - last) % 2 ** 32
        if d == 1:
            s_["last"] = seq
            return "ok"
        if d < 2 ** 31:                          # forward jump: packets lost (or not yet arrived)
            self.stats["gaps"] += 1
            self.stats["missing"] += d - 1
            s_["last"] = seq
            return f"gap:{d - 1}"
        self.stats["reordered"] += 1             # older than the newest seen: late arrival
        self.stats["missing"] = max(0, self.stats["missing"] - 1)
        return "late"

    def _write(self, rec: dict) -> None:
        if self.out:
            self.out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.out.flush()

    def run(self, duration_s: float | None = None, stop=None) -> dict:
        t_end = None if duration_s is None else time.time() + duration_s
        while (t_end is None or time.time() < t_end) and not (stop is not None and stop.is_set()):
            self.receive_once()
        return self.close()

    def close(self) -> dict:
        try:
            self._flush_acks(True)
            self.sock.close()
        finally:
            if self.out:
                self.out.close()
                self.out = None
        out = {**self.stats, "sessions": {str(k): {"last_seq": v["last"], "received": len(v["seen"])} for k, v in self.sessions.items()}}
        if self.map is not None:
            out["ground_map"] = self.map.snapshot()
        return out
