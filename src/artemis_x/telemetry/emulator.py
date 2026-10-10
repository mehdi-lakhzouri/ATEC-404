"""R5.3 lunar network emulator: rover UDP -> impairments -> ground UDP. Standard library only.

    rover (R5.2 UdpSender) --udp:5791--> LunarLinkEmulator --udp:5790--> GroundReceiver
                                              ^ control udp:5792 (text commands: PRESET <name> | SET k=v ... | STATUS)

Pipeline for every datagram from the rover, in arrival order:
    1. blackout          -> dropped (reason "blackout"); nothing is buffered (store-and-forward is R5.4)
    2. random loss       -> dropped with probability ``loss`` (seeded RNG: reproducible)
    3. bandwidth         -> token bucket (``bandwidth_Bps``, burst = one maximum packet); packets wait in a bounded FIFO;
                            a full FIFO drops the NEW packet (reason "queue")
    4. propagation       -> delivered ``latency_ms`` + uniform(+-``jitter_ms``) after it leaves the bucket (may reorder)
Bytes are forwarded unchanged (payload and CRC16 untouched). Scheduling uses ``time.perf_counter`` (monotonic,
high-resolution); simulated rover time is only read from packet headers for the log, never used for scheduling.
Entering a blackout discards packets already queued or in flight (counted "blackout"): no hidden buffering.

R5.4 reverse link (optional, ``reverse_listen`` + ``rover_ack``): datagrams from the ground (ACKs) go through the SAME
blackout and the same loss / latency / jitter settings, with an independent random stream (so the forward pattern of a
given seed is unchanged), and no bandwidth limit (the uplink is not the bottleneck). Counted separately as ``rev_*``.
"""

from __future__ import annotations

import heapq
import json
import random
import select
import socket
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import protocol as P

ROVER_PORT, GROUND_PORT, CONTROL_PORT = 5791, 5790, 5792
ROVER_ACK_PORT, REVERSE_PORT = 5793, 5794      # R5.4 reverse link: ground ACKs -> udp 5794 -> emulator -> rover udp 5793


@dataclass(frozen=True)
class LinkConfig:
    loss: float = 0.0                      # probability 0..1
    latency_ms: float = 0.0                # one-way propagation added after the bucket
    jitter_ms: float = 0.0                 # uniform +- jitter (delay never negative)
    bandwidth_Bps: float | None = None     # None = unlimited
    blackout: bool = False
    queue_max: int = 64                    # packets waiting for bandwidth

    def check(self) -> "LinkConfig":
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError(f"loss {self.loss} outside [0, 1]")
        if self.latency_ms < 0 or self.jitter_ms < 0 or (self.bandwidth_Bps is not None and self.bandwidth_Bps <= 0) or self.queue_max < 1:
            raise ValueError(f"invalid link config {self}")
        return self


PRESETS = {
    "NORMAL": LinkConfig(),
    "DEGRADED": LinkConfig(loss=0.30, latency_ms=200.0, jitter_ms=50.0),
    "SEVERE": LinkConfig(loss=0.50, latency_ms=1300.0, jitter_ms=300.0, bandwidth_Bps=1000.0),   # ~Earth-Moon light time
    "BLACKOUT": LinkConfig(loss=1.0, blackout=True),
    "RESTORE": LinkConfig(),               # back to NORMAL, logged as a restoration
}


def _q(xs, p):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 3) if s else None


class LunarLinkEmulator:
    def __init__(self, listen=("127.0.0.1", ROVER_PORT), ground=("127.0.0.1", GROUND_PORT), control=("127.0.0.1", CONTROL_PORT),
                 config: LinkConfig | None = None, preset: str = "NORMAL", seed: int = 0, log_path: Path | None = None,
                 reverse_listen: tuple | None = None, rover_ack: tuple | None = None):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(listen)
        self.rx.setblocking(False)
        self.ctl = None
        if control is not None:
            self.ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.ctl.bind(control)
            self.ctl.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ground = ground
        self.rng = random.Random(seed)
        self.rev = None
        self.rover_ack = rover_ack
        if reverse_listen is not None and rover_ack is not None:
            self.rev = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rev.bind(reverse_listen)
            self.rev.setblocking(False)
        self.rng_rev = random.Random(seed ^ 0x5EED)
        self.lock = threading.Lock()
        self.cfg = (config or PRESETS[preset]).check()
        self.preset = preset if config is None else "CUSTOM"
        self.t0 = time.perf_counter()
        self.tokens = float(P.MAX_PACKET)
        self.t_tokens = self.t0
        self.waiting: list = []            # FIFO waiting for bandwidth: (t_arrival, data, meta)
        self.flight: list = []             # heap: (t_deliver, n, t_arrival, data, meta)
        self.n = 0
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.blackout_since = self.t0 if self.cfg.blackout else None
        self.blackout_total = 0.0
        self.delays_ms: list = []
        self.st = {"received": 0, "bytes_received": 0, "forwarded": 0, "bytes_forwarded": 0, "dropped_loss": 0,
                   "dropped_blackout": 0, "dropped_queue": 0, "dropped_shutdown": 0, "invalid_crc_seen": 0,
                   "send_errors": 0, "config_changes": 0, "blackouts": 0, "by_preset": {},
                   "rev_received": 0, "rev_forwarded": 0, "rev_dropped_loss": 0, "rev_dropped_blackout": 0, "rev_dropped_shutdown": 0}
        self._event("start", preset=self.preset, config=asdict(self.cfg), seed=seed)

    # ---- control ---------------------------------------------------------------------------------------------------------
    def apply(self, preset: str | None = None, **overrides) -> LinkConfig:
        """Switch to a preset and/or override fields; takes effect for the very next datagram."""
        with self.lock:
            base = PRESETS[preset.upper()] if preset else self.cfg
            new = replace(base, **overrides).check()
            now = time.perf_counter()
            if new.blackout and not self.cfg.blackout:
                self.blackout_since = now
                self.st["blackouts"] += 1
                for item in self.waiting + [(f[2], f[3], f[4]) for f in self.flight]:   # nothing hidden in buffers
                    self._drop("blackout", item[1], item[2], now)
                self.waiting, self.flight = [], []
            if self.cfg.blackout and not new.blackout and self.blackout_since is not None:
                self.blackout_total += now - self.blackout_since
                self.blackout_since = None
            self.cfg = new
            self.preset = preset.upper() if preset and not overrides else ("CUSTOM" if overrides else self.preset)
            self.st["config_changes"] += 1
            self._event("config", preset=self.preset, config=asdict(new))
            return new

    def _control(self) -> None:
        try:
            data, addr = self.ctl.recvfrom(4096)
        except OSError:
            return
        words = data.decode("utf-8", "replace").strip().split()
        reply = {}
        try:
            if not words or words[0].upper() == "STATUS":
                pass
            elif words[0].upper() == "PRESET" and len(words) == 2:
                self.apply(words[1])
            elif words[0].upper() in PRESETS and len(words) == 1:
                self.apply(words[0])
            elif words[0].upper() == "SET":
                kv = dict(w.split("=", 1) for w in words[1:])
                conv = {k: (None if v.lower() == "none" else (v.lower() == "true") if k == "blackout"
                            else int(v) if k == "queue_max" else float(v)) for k, v in kv.items()}
                self.apply(None, **conv)
            else:
                raise ValueError(f"unknown command {' '.join(words)!r}")
        except (ValueError, TypeError, KeyError) as e:
            reply["error"] = str(e)
        reply.update(self.stats())
        try:
            self.ctl.sendto(json.dumps(reply).encode(), addr)
        except OSError:
            pass

    # ---- data path ----------------------------------------------------------------------------------------------------
    def _meta(self, data: bytes) -> dict:
        if len(data) >= P.HEADER.size + P.CRC.size and data[:2] == P.MAGIC:
            _, _, ptype, session, seq, t_ms, _ = P.HEADER.unpack_from(data)
            ok = P.CRC.unpack_from(data, len(data) - 2)[0] == P.crc16(data[:-2])
            return {"type": ptype, "session": session, "seq": seq, "t_obs_s": t_ms / 1000.0, "crc_ok": ok}
        return {"type": None, "crc_ok": False}

    def _ingest(self, data: bytes, now: float) -> None:
        meta = self._meta(data)
        st = self.st
        st["received"] += 1
        st["bytes_received"] += len(data)
        st["invalid_crc_seen"] += 0 if meta["crc_ok"] else 1
        bp = st["by_preset"].setdefault(self.preset, {"received": 0, "forwarded": 0, "dropped": 0})
        bp["received"] += 1
        meta["preset"] = self.preset
        cfg = self.cfg
        if cfg.blackout:
            return self._drop("blackout", data, meta, now)
        if cfg.loss > 0 and self.rng.random() < cfg.loss:
            return self._drop("loss", data, meta, now)
        if cfg.bandwidth_Bps is None:
            self._launch(now, now, data, meta)
        elif len(self.waiting) >= cfg.queue_max:
            self._drop("queue", data, meta, now)
        else:
            self.waiting.append((now, data, meta))

    def _ingest_rev(self, data: bytes, now: float) -> None:
        meta = {**self._meta(data), "dir": "rev", "preset": self.preset}
        self.st["rev_received"] += 1
        cfg = self.cfg
        if cfg.blackout:
            return self._drop("blackout", data, meta, now)
        if cfg.loss > 0 and self.rng_rev.random() < cfg.loss:
            return self._drop("loss", data, meta, now)
        self._launch(now, now, data, meta)

    def _launch(self, t_leave: float, t_arrival: float, data: bytes, meta: dict) -> None:
        cfg = self.cfg
        rng = self.rng_rev if meta.get("dir") == "rev" else self.rng
        delay = cfg.latency_ms + (rng.uniform(-cfg.jitter_ms, cfg.jitter_ms) if cfg.jitter_ms else 0.0)
        self.n += 1
        heapq.heappush(self.flight, (t_leave + max(0.0, delay) / 1000.0, self.n, t_arrival, data, meta))

    def _bandwidth(self, now: float) -> None:
        bw = self.cfg.bandwidth_Bps
        if bw is None:                      # limit lifted: release what waits
            for t_a, d, m in self.waiting:
                self._launch(now, t_a, d, m)
            self.waiting = []
            return
        self.tokens = min(float(P.MAX_PACKET), self.tokens + (now - self.t_tokens) * bw)
        self.t_tokens = now
        while self.waiting and self.tokens >= len(self.waiting[0][1]):
            t_a, d, m = self.waiting.pop(0)
            self.tokens -= len(d)
            self._launch(now, t_a, d, m)

    def _deliver(self, now: float) -> None:
        while self.flight and self.flight[0][0] <= now:
            _, _, t_a, data, meta = heapq.heappop(self.flight)
            rev = meta.get("dir") == "rev"
            try:
                self.tx.sendto(data, self.rover_ack if rev else self.ground)   # bytes unchanged
            except OSError:
                self.st["send_errors"] += 1
            if rev:
                self.st["rev_forwarded"] += 1
                self._log("rev_forwarded", meta, len(data), now, delay_ms=round((now - t_a) * 1000.0, 3))
                continue
            d_ms = (now - t_a) * 1000.0
            self.delays_ms.append(d_ms)
            self.st["forwarded"] += 1
            self.st["bytes_forwarded"] += len(data)
            self.st["by_preset"][meta["preset"]]["forwarded"] += 1
            self._log("forwarded", meta, len(data), now, delay_ms=round(d_ms, 3))

    def _drop(self, reason: str, data: bytes, meta: dict, now: float) -> None:
        if meta.get("dir") == "rev":
            self.st[f"rev_dropped_{reason}"] += 1
            self._log(f"rev_dropped_{reason}", meta, len(data), now)
            return
        self.st[f"dropped_{reason}"] += 1
        self.st["by_preset"].setdefault(meta.get("preset", self.preset), {"received": 0, "forwarded": 0, "dropped": 0})["dropped"] += 1
        self._log(f"dropped_{reason}", meta, len(data), now)

    def _log(self, action: str, meta: dict, n: int, now: float, **kw) -> None:
        if self.log:
            self.log.write(json.dumps({"t_s": round(now - self.t0, 6), "wall_s": round(time.time(), 6), "action": action, "len": n,
                                       **{k: meta.get(k) for k in ("type", "session", "seq", "t_obs_s", "preset")}, **kw},
                                      separators=(",", ":")) + "\n")

    def _event(self, kind: str, **kw) -> None:
        if self.log:
            self.log.write(json.dumps({"t_s": round(time.perf_counter() - self.t0, 6), "wall_s": round(time.time(), 6),
                                       "event": kind, **kw}, separators=(",", ":")) + "\n")
            self.log.flush()

    def step(self, max_wait_s: float = 0.05) -> None:
        """One loop iteration: wait for a datagram or the next delivery, then ingest / pace / deliver."""
        now = time.perf_counter()
        wait = max_wait_s
        if self.flight:
            wait = min(wait, max(0.0, self.flight[0][0] - now))
        if self.waiting and self.cfg.bandwidth_Bps:
            wait = min(wait, max(0.0, (len(self.waiting[0][1]) - self.tokens) / self.cfg.bandwidth_Bps))
        socks = [self.rx] + ([self.ctl] if self.ctl else []) + ([self.rev] if self.rev else [])
        ready, _, _ = select.select(socks, [], [], wait)
        with self.lock:
            if self.rx in ready:
                for _ in range(256):                       # drain the socket without blocking
                    try:
                        data, _ = self.rx.recvfrom(65535)
                    except (BlockingIOError, ConnectionResetError):
                        break
                    except OSError:
                        break
                    self._ingest(data, time.perf_counter())
            if self.rev is not None and self.rev in ready:
                for _ in range(256):
                    try:
                        data, _ = self.rev.recvfrom(65535)
                    except (BlockingIOError, ConnectionResetError):
                        break
                    except OSError:
                        break
                    self._ingest_rev(data, time.perf_counter())
            now = time.perf_counter()
            self._bandwidth(now)
            self._deliver(now)
        if self.ctl in ready:
            self._control()

    def run(self, stop: threading.Event | None = None, duration_s: float | None = None) -> dict:
        end = None if duration_s is None else time.perf_counter() + duration_s
        while not (stop is not None and stop.is_set()) and (end is None or time.perf_counter() < end):
            self.step()
        return self.close()

    def stats(self) -> dict:
        now = time.perf_counter()
        st = dict(self.st)
        el = now - self.t0
        bt = self.blackout_total + ((now - self.blackout_since) if self.blackout_since is not None else 0.0)
        st.update({"preset": self.preset, "config": asdict(self.cfg), "link_state": "BLACKOUT" if self.cfg.blackout else "UP",
                   "configured_loss": self.cfg.loss, "elapsed_s": round(el, 3), "blackout_total_s": round(bt, 3),
                   "delivery_ratio": round(st["forwarded"] / st["received"], 4) if st["received"] else None,
                   "queued": len(self.waiting), "in_flight": len(self.flight),
                   "delay_ms": {"n": len(self.delays_ms), "p50": _q(self.delays_ms, 0.5), "p95": _q(self.delays_ms, 0.95),
                                "max": round(max(self.delays_ms), 3) if self.delays_ms else None},
                   "throughput_Bps": round(st["bytes_forwarded"] / el, 1) if el > 0 else None})
        return st

    def close(self) -> dict:
        """Clean shutdown: packets still queued or in flight are counted as dropped_shutdown, sockets closed."""
        with self.lock:
            now = time.perf_counter()
            for item in self.waiting + [(f[2], f[3], f[4]) for f in self.flight]:
                self._drop("shutdown", item[1], item[2], now)
            self.waiting, self.flight = [], []
            s = self.stats()
            self._event("stop", stats=s)
            for sk in (self.rx, self.tx, self.ctl, self.rev):
                if sk is not None:
                    try:
                        sk.close()
                    except OSError:
                        pass
            if self.log:
                self.log.close()
                self.log = None
        return s


def send_control(command: str, control=("127.0.0.1", CONTROL_PORT), timeout_s: float = 2.0) -> dict:
    """Send one text command to a running emulator; returns its JSON reply (stats after the change)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout_s)
    try:
        s.sendto(command.encode(), control)
        data, _ = s.recvfrom(65535)
        return json.loads(data)
    finally:
        s.close()
