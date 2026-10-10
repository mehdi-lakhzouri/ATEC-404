"""Artemis-X binary lunar telemetry, protocol v1 (R5.1). Standard library only.

Every packet: big-endian (network order) header, a typed payload, CRC-16/CCITT-FALSE over header + payload.

    header (18 bytes)  magic "AX" | version u8 | type u8 | session u32 | seq u32 | t_obs_ms u32 | payload_len u16
    payload            POSE (13 B) | OBSTACLES (6 + 26 x n B, n <= MAX_RECORDS) | EVENT (8 B)
    trailer (2 bytes)  crc16 u16

Times: ``t_obs`` is the OBSERVATION time (simulation clock, ms) set by the rover. The reception time is never in the
packet: a receiver records it separately (``Packet.rx_time_s`` is filled by ``decode(..., rx_time_s=...)``), so latency
is ``rx_time_s - t_obs_s``. Frame: every position is in the rover's ``odom`` frame (code 0); other frames are rejected.
Quantisation and limits are in the tables below and in docs/TELEMETRY_R5_1_PROTOCOL.md. ``encode`` never clips silently:
out-of-range values raise ``ProtocolError``; only the two uncertainty fields saturate, explicitly, at their maximum
code (meaning ">= the maximum").
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

MAGIC = b"AX"
VERSION = 1
HEADER = struct.Struct(">2sBBIIIH")            # 18 bytes
CRC = struct.Struct(">H")
T_POSE, T_OBSTACLES, T_EVENT = 1, 2, 3
FRAME_ODOM = 0

POSE = struct.Struct(">BiihH")                 # frame, x_mm, y_mm, yaw_1e-4rad, sigma_xy_cm            = 13 B
OBS_HEAD = struct.Struct(">BBBBH")             # frame, part, parts, count, map_resets                  =  6 B
RECORD = struct.Struct(">BIiiHBHII")           # flags, id, x_mm, y_mm, radius_mm, conf/255, unc_mm, t_obs_ms, link = 26 B
EVENT = struct.Struct(">BBIH")                 # code, health state, reason bitmask, value              =  8 B

MAX_RECORDS = 40                               # obstacles per packet
MAX_PACKET = HEADER.size + OBS_HEAD.size + MAX_RECORDS * RECORD.size + CRC.size   # 1,066 bytes (largest packet)
MM = 1000.0
SAT_U16 = 0xFFFF                               # saturated uncertainty code: ">= 65.535 m" (mm) / ">= 655.35 m" (cm)

TYPES = ["rock", "obstacle", "negative_hazard"]
SOURCES = ["yolo+depth", "depth"]
STATUSES = ["confirmed", "lost", "observed"]
HEALTH = ["OK", "DEGRADED", "STOP"]
REASONS = ["depth_stale", "depth_frozen", "depth_invalid", "depth_degraded", "no_pose", "obstacle_in_footprint",
           "obstacle_in_corridor", "unobserved_corridor", "stale_corridor", "inference_disabled", "inference_unavailable",
           "inference_stale", "cycle_overrun"]    # bit i = REASONS[i]; append only (never reorder) within v1
EVENTS = ["health", "map_reset", "inference"]  # event codes 1..3 (index + 1)


class ProtocolError(ValueError):
    """Invalid field on encode, or a malformed / corrupted / unsupported packet on decode."""


# ---- CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xor-out): check("123456789") == 0x29B1 ----------
def _crc_table() -> list:
    t = []
    for b in range(256):
        c = b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) if c & 0x8000 else (c << 1)
        t.append(c & 0xFFFF)
    return t


_TABLE = _crc_table()


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


# ---- quantisation helpers (raise on overflow; never clip silently) -------------------------------------------------------
def _num(name: str, v) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise ProtocolError(f"{name}: not a finite number ({v!r})")
    return float(v)


def _q(name: str, v, scale: float, lo: int, hi: int) -> int:
    q = round(_num(name, v) * scale)
    if not lo <= q <= hi:
        raise ProtocolError(f"{name}={v} out of range [{lo / scale}, {hi / scale}]")
    return q


def _q_sat(name: str, v, scale: float) -> int:
    """Non-negative quantity that saturates explicitly at SAT_U16 (meaning '>= maximum')."""
    x = _num(name, v)
    if x < 0:
        raise ProtocolError(f"{name}={v} must be >= 0")
    return min(SAT_U16, round(x * scale))


def _t_ms(name: str, t_s) -> int:
    return _q(name, t_s, MM, 0, 0xFFFFFFFF)


def _u32(name: str, v) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 0xFFFFFFFF:
        raise ProtocolError(f"{name}={v!r} must be an integer in [0, 2^32-1]")
    return v


# ---- obstacle ids: two disjoint namespaces, so a track and a depth cluster can never collide ------------------------------
def encode_id(oid: str) -> tuple[int, int]:
    """'trk-<n>' -> (0, n);  'dep-<i>_<j>' -> (1, (i & 0xFFFF) << 16 | (j & 0xFFFF)) with i, j signed 16-bit cell indices."""
    if not isinstance(oid, str):
        raise ProtocolError(f"id {oid!r} is not a string")
    if oid.startswith("trk-"):
        try:
            n = int(oid[4:])
        except ValueError:
            raise ProtocolError(f"bad track id {oid!r}") from None
        return 0, _u32("track id", n)
    if oid.startswith("dep-"):
        try:
            i, j = (int(p) for p in oid[4:].split("_"))
        except ValueError:
            raise ProtocolError(f"bad depth-cluster id {oid!r}") from None
        for name, v in (("cluster i", i), ("cluster j", j)):
            if not -32768 <= v <= 32767:
                raise ProtocolError(f"{name}={v} outside int16 (cell index)")
        return 1, ((i & 0xFFFF) << 16) | (j & 0xFFFF)
    raise ProtocolError(f"unknown id namespace {oid!r}")


def decode_id(kind: int, v: int) -> str:
    if kind == 0:
        return f"trk-{v}"
    i, j = v >> 16, v & 0xFFFF
    i, j = (i - 0x10000 if i & 0x8000 else i), (j - 0x10000 if j & 0x8000 else j)
    return f"dep-{i}_{j}"


# ---- packets ---------------------------------------------------------------------------------------------------------------
@dataclass
class Packet:
    type: int
    session: int
    seq: int
    t_obs_s: float                                 # observation time (simulation clock), from the rover
    body: dict = field(default_factory=dict)
    rx_time_s: float | None = None                 # reception time, set by the receiver - never transmitted


def _frame(ptype: int, session: int, seq: int, t_obs_s: float, payload: bytes) -> bytes:
    if len(payload) > MAX_PACKET - HEADER.size - CRC.size:
        raise ProtocolError(f"payload {len(payload)} B exceeds the packet bound")
    head = HEADER.pack(MAGIC, VERSION, ptype, _u32("session", session), _u32("seq", seq), _t_ms("t_obs", t_obs_s), len(payload))
    return head + payload + CRC.pack(crc16(head + payload))


def encode_pose(session: int, seq: int, pose: dict) -> bytes:
    """pose: the contract's pose dict (frame 'odom', x, y, yaw, sigma_xy, stamp_s)."""
    if pose.get("frame", "odom") != "odom":
        raise ProtocolError(f"frame {pose.get('frame')!r}: only 'odom' is defined in v1")
    yaw = math.atan2(math.sin(_num("yaw", pose["yaw"])), math.cos(pose["yaw"]))        # wrap to [-pi, pi]
    payload = POSE.pack(FRAME_ODOM, _q("x", pose["x"], MM, -2 ** 31, 2 ** 31 - 1), _q("y", pose["y"], MM, -2 ** 31, 2 ** 31 - 1),
                        _q("yaw", yaw, 1e4, -32768, 32767), _q_sat("sigma_xy", pose.get("sigma_xy", 0.0), 100.0))
    return _frame(T_POSE, session, seq, pose["stamp_s"], payload)


def _record(o: dict, link: int) -> bytes:
    kind, ident = encode_id(o["id"])
    try:
        t, src, st = TYPES.index(o["type"]), SOURCES.index(o["source"]), STATUSES.index(o["status"])
    except ValueError as e:
        raise ProtocolError(f"{o.get('id')}: unknown enum value ({e})") from None
    if (kind == 0) != (src == 0):
        raise ProtocolError(f"{o['id']}: id namespace does not match source {o['source']!r}")
    conf = _num("confidence", o["confidence"])
    if not 0.0 <= conf <= 1.0:
        raise ProtocolError(f"{o['id']}: confidence {conf} outside [0, 1]")
    flags = kind | (t << 1) | (src << 3) | (st << 4) | (int(bool(o.get("protected"))) << 6)
    return RECORD.pack(flags, ident, _q("x_m", o["x_m"], MM, -2 ** 31, 2 ** 31 - 1), _q("y_m", o["y_m"], MM, -2 ** 31, 2 ** 31 - 1),
                       _q("radius_m", o["radius_m"], MM, 0, 0xFFFF), round(conf * 255), _q_sat("uncertainty_m", o["uncertainty_m"], MM),
                       _t_ms("timestamp_s", o["timestamp_s"]), link)


def encode_obstacles(session: int, seq0: int, doc: dict, obstacles: list | None = None) -> list:
    """An artemis.obstacles.v1 document (or a filtered ``obstacles`` list from it) -> packets of <= MAX_RECORDS records,
    with consecutive sequence numbers from ``seq0``. A depth cluster linked to tracks carries the first linked track id
    (``link``), so receivers can tell it is the same physical object; 0 = no link."""
    if doc.get("schema") != "artemis.obstacles.v1" or doc.get("frame") != "odom":
        raise ProtocolError(f"expected artemis.obstacles.v1 in the odom frame, got {doc.get('schema')!r} / {doc.get('frame')!r}")
    obs = doc["obstacles"] if obstacles is None else obstacles
    recs = []
    for o in obs:
        links = o.get("track_ids") or []
        link = encode_id(links[0])[1] if links else 0
        recs.append(_record(o, link))
    resets = doc.get("map_resets", 0)
    if not 0 <= resets <= 0xFFFF:
        raise ProtocolError(f"map_resets={resets} out of range")
    chunks = [recs[i:i + MAX_RECORDS] for i in range(0, len(recs), MAX_RECORDS)] or [[]]
    if len(chunks) > 255:
        raise ProtocolError(f"{len(recs)} obstacles exceed 255 packets per snapshot")
    return [_frame(T_OBSTACLES, session, (seq0 + k) & 0xFFFFFFFF, doc["stamp_s"],
                   OBS_HEAD.pack(FRAME_ODOM, k, len(chunks), len(c), resets) + b"".join(c)) for k, c in enumerate(chunks)]


def encode_event(session: int, seq: int, t_obs_s: float, code: str, state: str = "OK", reasons=(), value: int = 0) -> bytes:
    try:
        c, s = EVENTS.index(code) + 1, HEALTH.index(state)
        mask = 0
        for r in reasons:
            mask |= 1 << REASONS.index(r)
    except ValueError as e:
        raise ProtocolError(f"unknown event code / state / reason ({e})") from None
    if not 0 <= value <= 0xFFFF:
        raise ProtocolError(f"event value {value} out of range")
    return _frame(T_EVENT, session, seq, t_obs_s, EVENT.pack(c, s, mask, value))


def decode(data: bytes, rx_time_s: float | None = None) -> Packet:
    """Strict: rejects bad magic, unknown version / type / frame / enums, length mismatch, truncation and CRC errors."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError("packet must be bytes")
    data = bytes(data)
    if len(data) < HEADER.size + CRC.size:
        raise ProtocolError(f"truncated: {len(data)} B < minimum {HEADER.size + CRC.size} B")
    if len(data) > MAX_PACKET:
        raise ProtocolError(f"oversized: {len(data)} B > {MAX_PACKET} B")
    magic, ver, ptype, session, seq, t_ms, plen = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ProtocolError(f"unsupported version {ver}")
    if len(data) != HEADER.size + plen + CRC.size:
        raise ProtocolError(f"length mismatch: payload_len {plen}, packet {len(data)} B")
    if CRC.unpack_from(data, len(data) - 2)[0] != crc16(data[:-2]):
        raise ProtocolError("CRC mismatch")
    p = data[HEADER.size:-CRC.size]
    pk = Packet(ptype, session, seq, t_ms / MM, rx_time_s=rx_time_s)
    if ptype == T_POSE:
        if plen != POSE.size:
            raise ProtocolError(f"POSE payload {plen} B != {POSE.size} B")
        frame, x, y, yaw, sig = POSE.unpack(p)
        _check_frame(frame)
        pk.body = {"frame": "odom", "x": x / MM, "y": y / MM, "yaw": yaw / 1e4,
                   "sigma_xy": sig / 100.0, "sigma_xy_saturated": sig == SAT_U16}
    elif ptype == T_OBSTACLES:
        if plen < OBS_HEAD.size:
            raise ProtocolError("OBSTACLES payload shorter than its header")
        frame, part, parts, n, resets = OBS_HEAD.unpack_from(p)
        _check_frame(frame)
        if n > MAX_RECORDS or plen != OBS_HEAD.size + n * RECORD.size or not (parts >= 1 and part < parts):
            raise ProtocolError(f"OBSTACLES: count {n}, part {part}/{parts}, payload {plen} B inconsistent")
        obs = []
        for k in range(n):
            flags, ident, x, y, r, conf, unc, t, link = RECORD.unpack_from(p, OBS_HEAD.size + k * RECORD.size)
            kind, t_i, src, st = flags & 1, (flags >> 1) & 3, (flags >> 3) & 1, (flags >> 4) & 3
            if flags >> 7 or t_i >= len(TYPES) or st >= len(STATUSES) or kind != src:
                raise ProtocolError(f"record {k}: invalid flags 0x{flags:02x}")
            o = {"id": decode_id(kind, ident), "type": TYPES[t_i], "x_m": x / MM, "y_m": y / MM, "radius_m": r / MM,
                 "confidence": conf / 255, "uncertainty_m": unc / MM, "uncertainty_saturated": unc == SAT_U16,
                 "timestamp_s": t / MM, "source": SOURCES[src], "status": STATUSES[st], "protected": bool(flags >> 6 & 1)}
            if link:
                o["linked_track"] = decode_id(0, link)
            obs.append(o)
        pk.body = {"frame": "odom", "part": part, "parts": parts, "map_resets": resets, "obstacles": obs}
    elif ptype == T_EVENT:
        if plen != EVENT.size:
            raise ProtocolError(f"EVENT payload {plen} B != {EVENT.size} B")
        code, state, mask, value = EVENT.unpack(p)
        if not 1 <= code <= len(EVENTS) or state >= len(HEALTH) or mask >> len(REASONS):
            raise ProtocolError(f"EVENT: invalid code {code} / state {state} / reasons 0x{mask:x}")
        pk.body = {"event": EVENTS[code - 1], "state": HEALTH[state],
                   "reasons": [r for i, r in enumerate(REASONS) if mask >> i & 1], "value": value}
    else:
        raise ProtocolError(f"unknown packet type {ptype}")
    return pk


def _check_frame(frame: int) -> None:
    if frame != FRAME_ODOM:
        raise ProtocolError(f"unknown frame code {frame} (v1 defines only odom = 0)")
