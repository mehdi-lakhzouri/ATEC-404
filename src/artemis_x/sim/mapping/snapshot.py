"""MapSnapshot v1 serialisation (R4 plan section 7): a JSON header + one binary file of little-endian layers.

    <name>.json   header: schema, stamp, frame "odom", resolution, width/height, origin (corner of cell 0,0), axis-aligned,
                  estimated pose, layer table (name, dtype, byte offset, encoding), counters, config hash
    <name>.bin    raw layers back to back, row-major (k = j * width + i), little endian:
                  log_odds    float32  clamped log-odds
                  last_obs_s  float32  simulation time of the cell's last update; -1 = never observed
                  negative    uint8    1 if the cell's latest hit was below the ground plane

Readers derive the cell state with the thresholds stored in the header (``state_rule``):
    OCCUPIED (2) if log_odds > l_occupied (NEGATIVE (3) if negative == 1); FREE (1) if log_odds < l_free and observed;
    otherwise UNKNOWN (0). Raw layers are written because converting 14,400 cells per layer in pure Python cost ~44 ms
    per snapshot in the controller; ``tobytes()`` costs microseconds.

Additive layers under the same schema version (any incompatible change bumps it):
                  cost        uint8    R4.3 inflated cost (costmap.py semantics)
                  provenance  uint8    evidence-source bit flags (grid.py SRC_*): 1 depth hit, 2 YOLO-selected small-rock
                                       depth points, 4 YOLO small-rock footprint, 8 currently protected from FREE evidence
Tracks (R4.4) are in the header. Readers here are stdlib-only.
"""

from __future__ import annotations

import array
import json
import sys
from pathlib import Path

from .grid import NEVER, LogOddsGrid

SCHEMA = "artemis.map.v1"


def _le(a: array.array) -> bytes:
    if sys.byteorder != "little":
        a = array.array(a.typecode, a); a.byteswap()
    return a.tobytes()


def snapshot_payload(grid: LogOddsGrid, stamp_s: float, pose: dict, base: Path, extra: dict | None = None,
                     extra_layers: list | None = None) -> tuple:
    """Everything a snapshot needs, copied from the live grid (bytes are immutable copies): cheap, safe to hand to a thread."""
    n = grid.n
    layers = [("log_odds", "float32", _le(grid.logodds)), ("last_obs_s", "float32", _le(grid.last_obs)), ("negative", "uint8", grid.neg.tobytes())]
    layers += list(extra_layers or [])                     # additive layers, e.g. R4.3 "cost" (uint8, costmap.py semantics)
    table, blob, off = [], bytearray(), 0
    for name, dt, b in layers:
        table.append({"name": name, "dtype": dt, "offset": off, "bytes": len(b)})
        blob += b; off += len(b)
    hdr = {"schema": SCHEMA, "stamp_s": round(stamp_s, 4), "frame": "odom", "resolution_m": grid.res, "width": n, "height": n,
           "origin_xy": [round(grid.origin_x, 6), round(grid.origin_y, 6)], "axis_aligned": True, "pose": pose,
           "layers": table, "binary": base.name + ".bin", "byte_order": "little",
           "state_rule": {"l_occupied": grid.occ_th, "l_free": grid.free_th, "never_observed": NEVER,
                          "states": {"0": "unknown", "1": "free", "2": "occupied", "3": "negative hazard"}},
           "recenters": grid.recenters, "config_hash": grid.cfg.hash(), **(extra or {})}
    return base, hdr, bytes(blob)


def write_payload(payload: tuple) -> dict:
    base, hdr, blob = payload
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.parent / (base.name + ".bin")).write_bytes(blob)                 # not with_suffix(): names may contain dots
    (base.parent / (base.name + ".json")).write_text(json.dumps(hdr), encoding="utf-8")
    return hdr


def write_snapshot(grid: LogOddsGrid, stamp_s: float, pose: dict, base: Path, extra: dict | None = None) -> dict:
    return write_payload(snapshot_payload(grid, stamp_s, pose, base, extra))


def read_snapshot(header_path: Path) -> tuple[dict, dict]:
    """-> (header, {layer name: array}) using only the standard library; adds the derived ``state`` layer (bytes)."""
    hdr = json.loads(Path(header_path).read_text(encoding="utf-8"))
    if hdr.get("schema") != SCHEMA:
        raise ValueError(f"unsupported map schema {hdr.get('schema')!r}")
    blob = (Path(header_path).parent / hdr["binary"]).read_bytes()
    codes = {"uint8": "B", "int8": "b", "uint16": "H", "float32": "f"}
    out = {}
    for L in hdr["layers"]:
        a = array.array(codes[L["dtype"]])
        a.frombytes(blob[L["offset"]:L["offset"] + L["bytes"]])
        if a.itemsize > 1 and sys.byteorder != "little":
            a.byteswap()
        out[L["name"]] = a
    sr = hdr["state_rule"]
    lo, last, neg = out["log_odds"], out["last_obs_s"], out["negative"]
    out["state"] = bytes((3 if neg[k] else 2) if lo[k] > sr["l_occupied"] else (1 if lo[k] < sr["l_free"] and last[k] != sr["never_observed"] else 0)
                         for k in range(len(lo)))
    return hdr, out
