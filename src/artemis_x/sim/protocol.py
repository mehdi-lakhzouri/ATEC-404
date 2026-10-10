"""Artemis-X R3 wire protocol between the Webots controller (stdlib only) and the inference server (ML environment).

One message = fixed 14-byte preamble + JSON header + binary payload:

    magic  b"AXR3" (4) | version u16 (2) | header_len u32 (4) | payload_len u32 (4) | header (UTF-8 JSON) | payload

Message types (header["type"]):
    hello       server -> client on connect: protocol version, model SHA-256, frozen inference settings
    frame       client -> server: frame_id, sim_time_s, width, height, encoding "rgb8"; payload = H*W*3 RGB bytes
    detections  server -> client: frame_id echo, detections [{xyxy, conf}], inference_ms, device, model_sha256
    error       server -> client: frame_id (if known), error text
    bye         client -> server: close the connection cleanly
Transport: TCP on 127.0.0.1 only. Limits are enforced before any payload is read.
"""

from __future__ import annotations

import json
import socket
import struct

MAGIC = b"AXR3"
VERSION = 1
DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 5761
PREAMBLE = struct.Struct("!4sHII")
MAX_HEADER = 64 * 1024
MAX_PAYLOAD = 4096 * 4096 * 3          # 48 MiB hard cap
MAX_SIDE = 4096


class ProtocolError(Exception):
    """Malformed, oversized or unexpected message."""


class ConnectionClosed(Exception):
    """Peer closed the connection."""


def encode(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps({**header, "v": VERSION}, separators=(",", ":")).encode("utf-8")
    if len(h) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("message too large")
    return PREAMBLE.pack(MAGIC, VERSION, len(h), len(payload)) + h + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionClosed("peer closed the connection")
        buf += chunk
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[dict, bytes]:
    """Read one message (honours the socket timeout; raises socket.timeout, ConnectionClosed or ProtocolError)."""
    magic, version, hlen, plen = PREAMBLE.unpack(_recv_exact(sock, PREAMBLE.size))
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise ProtocolError(f"oversized message (header {hlen}, payload {plen})")
    try:
        header = json.loads(_recv_exact(sock, hlen).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad header: {e}") from e
    if not isinstance(header, dict) or "type" not in header:
        raise ProtocolError("header must be an object with a 'type'")
    return header, _recv_exact(sock, plen) if plen else b""


def validate_frame(header: dict, payload: bytes) -> None:
    """Check a 'frame' message before it reaches the model."""
    try:
        fid, w, h = int(header["frame_id"]), int(header["width"]), int(header["height"])
        float(header["sim_time_s"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProtocolError(f"frame header incomplete: {e}") from e
    if fid < 0 or not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
        raise ProtocolError(f"invalid frame geometry or id ({fid}, {w}x{h})")
    if header.get("encoding") != "rgb8":
        raise ProtocolError(f"unsupported encoding {header.get('encoding')!r}")
    if len(payload) != w * h * 3:
        raise ProtocolError(f"payload {len(payload)} bytes != {w}x{h}x3")


def frame_message(frame_id: int, sim_time_s: float, width: int, height: int, rgb: bytes) -> bytes:
    return encode({"type": "frame", "frame_id": frame_id, "sim_time_s": sim_time_s, "width": width, "height": height,
                   "encoding": "rgb8"}, rgb)
