"""Offline evaluation of R4 mapping outputs against the Webots course ground truth (ML environment, numpy).

Ground truth is used HERE only: the course manifest (exact rock placements -> the rendered meshes, engineered objects, crater
terrain) and the supervisor pose logged under ``gt_pose_eval_only``. The live pipeline never reads either.

odom -> world alignment uses a single rigid transform fixed by the FIRST logged cycle (start pose), so any later odometry
drift shows up as error instead of being corrected away.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "artemis-webots" / "tools"))
import make_world  # noqa: E402  (placement -> exact rendered mesh, terrain height)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def alignment(first: dict) -> tuple[float, float, float]:
    """(tx, ty, theta) such that world = R(theta) odom + t, from the first cycle's estimated and ground-truth poses."""
    e, g = first["pose_est"], first["gt_pose_eval_only"]
    th = g["yaw"] - e["yaw"]
    c, s = math.cos(th), math.sin(th)
    return g["x"] - (c * e["x"] - s * e["y"]), g["y"] - (s * e["x"] + c * e["y"]), th


def to_world(xy: np.ndarray, al: tuple) -> np.ndarray:
    tx, ty, th = al
    c, s = math.cos(th), math.sin(th)
    return np.stack([c * xy[:, 0] - s * xy[:, 1] + tx, s * xy[:, 0] + c * xy[:, 1] + ty], 1)


def _rot(axis_angle) -> np.ndarray:
    x, y, z, a = axis_angle
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def distractor_distance(P: np.ndarray, d: dict) -> np.ndarray:
    """Exact distance from points to a box / cylinder (axis = local z) / sphere surface (Webots geometry conventions)."""
    R = _rot(d.get("rotation", [0, 0, 1, 0]))
    q = (P - np.array([d["x"], d["y"], d["z"]])) @ R                 # world -> local (R^T applied to row vectors)
    if d["shape"] == "box":
        e = np.abs(q) - np.array(d["size"]) / 2
    elif d["shape"] == "cylinder":
        e = np.stack([np.hypot(q[:, 0], q[:, 1]) - d["radius"], np.abs(q[:, 2]) - d["height"] / 2], 1)
    else:
        return np.abs(np.linalg.norm(q, axis=1) - d["radius"])
    return np.abs(np.linalg.norm(np.maximum(e, 0), axis=1) + np.minimum(e.max(1), 0))


def rock_vertex_distance(P: np.ndarray, rocks: list, margin: float = 0.4) -> np.ndarray:
    """Distance to the nearest rendered rock vertex (an upper bound on the distance to the surface; vertex spacing is
    millimetres for the small scans and a few centimetres on the simplified boulders)."""
    best = np.full(len(P), np.inf)
    for rk in rocks:
        V = np.array(make_world.ph_place(**rk["placement"])["vertices"])
        lo, hi = V.min(0) - margin, V.max(0) + margin
        idx = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1))[0]
        for k in range(0, len(idx), 2048):
            sel = idx[k:k + 2048]
            d2 = ((P[sel, None, :] - V[None, :, :]) ** 2).sum(-1).min(1)
            best[sel] = np.minimum(best[sel], np.sqrt(d2))
    return best


def surface_distance(P: np.ndarray, man: dict) -> tuple[np.ndarray, np.ndarray]:
    """Distance from world points to the nearest true surface, and which kind it is: 0 rock, 1 engineered object, 2 terrain."""
    d_rock = rock_vertex_distance(P, man["rocks"])
    d_obj = np.min([distractor_distance(P, d) for d in man["distractors"]], axis=0) if man["distractors"] else np.full(len(P), np.inf)
    th = np.array([make_world.terrain_height(x, y) for x, y in P[:, :2]]) + 0.002   # rendered terrain is offset +2 mm
    d_ter = np.abs(P[:, 2] - th)                                                     # vertical distance (terrain slopes are gentle)
    D = np.stack([d_rock, d_obj, d_ter], 1)
    return D.min(1), D.argmin(1)


def q(a, p: float):
    a = np.asarray(a, float)
    return round(float(np.quantile(a, p)), 4) if len(a) else None


# ---- R4.2: occupancy-grid evaluation ------------------------------------------------------------------------------------
UNKNOWN, FREE, OCCUPIED, NEGATIVE = 0, 1, 2, 3
H_OBS, H_NEG = 0.05, 0.08                    # same thresholds as the live mapper (MappingConfig, decision D3)


class WorldGrid:
    """Axis-aligned world raster used for ground truth and for the accumulated map (same 0.10 m cells as the mapper)."""

    def __init__(self, x0=-3.0, x1=41.0, y0=-5.0, y1=5.0, res=0.10):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx, self.ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))

    def ij(self, x, y):
        i = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        j = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, ok

    def centres(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)                       # arrays [ny, nx]


def ground_truth(man: dict, wg: WorldGrid) -> dict:
    """Ground-truth rasters: positive hazard (>= H_OBS above the local terrain), negative hazard (terrain <= -H_NEG),
    low rocks (lower than H_OBS everywhere, traversable by policy D3), and the object owning each hazard cell."""
    pos = np.zeros((wg.ny, wg.nx), bool)
    low = np.zeros_like(pos)
    owner = np.full((wg.ny, wg.nx), -1, int)
    objects = []
    for k, rk in enumerate(man["rocks"]):
        V = np.array(make_world.ph_place(**rk["placement"])["vertices"])
        th = np.array([make_world.terrain_height(x, y) for x, y in V[:, :2]]) + 0.002
        h = V[:, 2] - th
        i, j, ok = wg.ij(V[:, 0], V[:, 1])
        tall = ok & (h >= H_OBS)
        if tall.any():
            pos[j[tall], i[tall]] = True
            owner[j[tall], i[tall]] = k
        else:
            above = ok & (h > 0.005)
            low[j[above], i[above]] = True
        objects.append({"id": k, "name": rk["name"], "zone": rk["zone"], "kind": "rock", "x": rk["x"], "y": rk["y"],
                        "width_m": rk["width_m"], "height_m": round(float(h.max()), 3),
                        "cells": len(set(zip(i[tall].tolist(), j[tall].tolist())))})
    for m, d in enumerate(man["distractors"]):
        k = len(man["rocks"]) + m
        g = np.mgrid[d["x"] - 0.7:d["x"] + 0.7:0.02, d["y"] - 0.7:d["y"] + 0.7:0.02, H_OBS:1.3:0.02].reshape(3, -1).T
        R = _rot(d.get("rotation", [0, 0, 1, 0]))
        q = (g - np.array([d["x"], d["y"], d["z"]])) @ R
        if d["shape"] == "box":
            inside = np.all(np.abs(q) <= np.array(d["size"]) / 2 + 0.011, axis=1)
        elif d["shape"] == "cylinder":
            inside = (np.hypot(q[:, 0], q[:, 1]) <= d["radius"] + 0.011) & (np.abs(q[:, 2]) <= d["height"] / 2 + 0.011)
        else:
            inside = np.linalg.norm(q, axis=1) <= d["radius"] + 0.011
        P = g[inside]
        i, j, ok = wg.ij(P[:, 0], P[:, 1])
        pos[j[ok], i[ok]] = True
        owner[j[ok], i[ok]] = k
        objects.append({"id": k, "name": d["name"], "zone": d["zone"], "kind": "engineered object", "x": d["x"], "y": d["y"],
                        "width_m": 0.6, "height_m": round(float(P[:, 2].max()) if len(P) else 0.0, 3),
                        "cells": len(set(zip(i[ok].tolist(), j[ok].tolist())))})
    X, Y = wg.centres()
    T = np.vectorize(make_world.terrain_height)(X, Y) + 0.002
    rim = T >= H_OBS
    neg = T <= -H_NEG
    pos |= rim
    for c, cr in enumerate(man["craters"]):
        k = len(man["rocks"]) + len(man["distractors"]) + c
        near = np.hypot(X - cr["x"], Y - cr["y"]) <= cr["radius"] * 1.6
        owner[(rim | neg) & near & (owner < 0)] = k
        objects.append({"id": k, "name": cr["name"], "zone": "Z7_craters", "kind": "crater", "x": cr["x"], "y": cr["y"],
                        "width_m": 2 * cr["radius"], "height_m": round(float(T[near].min()), 3),
                        "bowl_cells": int((neg & near).sum()), "rim_cells": int((rim & near).sum())})
    return {"pos": pos, "neg": neg, "low": low & ~pos, "owner": owner, "objects": objects, "terrain": T}


def load_snapshots(snap_dir: Path) -> list[dict]:
    out = []
    for h in sorted(snap_dir.glob("map_*.json")):
        hdr = json.loads(h.read_text(encoding="utf-8"))
        blob = (snap_dir / hdr["binary"]).read_bytes()
        L = {}
        for lay in hdr["layers"]:
            dt = np.dtype({"uint8": "<u1", "int8": "<i1", "uint16": "<u2", "float32": "<f4"}[lay["dtype"]])
            L[lay["name"]] = np.frombuffer(blob, dt, count=lay["bytes"] // dt.itemsize, offset=lay["offset"]).reshape(hdr["height"], hdr["width"])
        sr = hdr["state_rule"]
        lo, last, neg = L["log_odds"], L["last_obs_s"], L["negative"]
        observed = last != sr["never_observed"]
        state = np.where(lo > sr["l_occupied"], np.where(neg == 1, 3, 2), np.where((lo < sr["l_free"]) & observed, 1, 0)).astype(np.uint8)
        age = np.where(observed, np.clip(np.round((hdr["stamp_s"] - last) * 10), 0, 65534), 65535).astype(np.uint16)
        out.append({"hdr": hdr, **L, "state": state, "obs_age_ds": age, "final": h.stem == "map_final"})
    out.sort(key=lambda s: (s["hdr"]["stamp_s"], s["final"]))
    return out


def snapshot_to_world(s: dict, al: tuple, wg: WorldGrid):
    """World-grid indices of every snapshot cell centre (odom -> world with the start alignment)."""
    h = s["hdr"]
    r, n = h["resolution_m"], h["width"]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n))
    xo = h["origin_xy"][0] + (ii + 0.5) * r
    yo = h["origin_xy"][1] + (jj + 0.5) * r
    w = to_world(np.column_stack([xo.ravel(), yo.ravel()]), al)
    return wg.ij(w[:, 0], w[:, 1])


def accumulate(snaps: list[dict], al: tuple, wg: WorldGrid, upto: int | None = None) -> np.ndarray:
    """Whole-course map: each world cell takes its state from the latest snapshot in which it had been observed
    (the live map is a 12 m rolling window; cells that left it keep their last observed state here)."""
    acc = np.zeros((wg.ny, wg.nx), np.uint8)
    for s in snaps[: (upto if upto is not None else len(snaps))]:
        i, j, ok = snapshot_to_world(s, al, wg)
        seen = (s["obs_age_ds"].ravel() != 65535) & ok
        acc[j[seen], i[seen]] = s["state"].ravel()[seen]
    return acc


def recenter_consistency(snaps: list[dict]) -> dict:
    """Across consecutive snapshots whose window moved: every cell inside both windows that was NOT updated in between
    (its observation age grew by exactly the elapsed time) must carry the same log-odds at the same odom position."""
    checked = mismatched = pairs = 0
    for a, b in zip(snaps, snaps[1:]):
        ha, hb = a["hdr"], b["hdr"]
        if ha["origin_xy"] == hb["origin_xy"] or b["final"]:
            continue
        pairs += 1
        r, n = ha["resolution_m"], ha["width"]
        di = int(round((hb["origin_xy"][0] - ha["origin_xy"][0]) / r))
        dj = int(round((hb["origin_xy"][1] - ha["origin_xy"][1]) / r))
        dt_ds = int(round((hb["stamp_s"] - ha["stamp_s"]) * 10))
        jb, ib = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ia, ja = ib + di, jb + dj
        ok = (ia >= 0) & (ia < n) & (ja >= 0) & (ja < n)
        a_age = a["obs_age_ds"][ja[ok], ia[ok]].astype(int)
        b_age = b["obs_age_ds"][jb[ok], ib[ok]].astype(int)
        untouched = (a_age != 65535) & (np.abs(b_age - a_age - dt_ds) <= 1)
        checked += int(untouched.sum())
        mismatched += int((a["log_odds"][ja[ok], ia[ok]][untouched] != b["log_odds"][jb[ok], ib[ok]][untouched]).sum())
    return {"window_shifts_between_snapshots": pairs, "untouched_cells_compared": checked, "mismatched": mismatched}


def best_shift(acc: np.ndarray, gt_haz: np.ndarray, r: int = 3) -> dict:
    """Registration check: the shift (in cells) that maximises the overlap of mapped hazards with ground-truth hazards."""
    m = (acc == OCCUPIED) | (acc == NEGATIVE)
    best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ov = int((np.roll(np.roll(m, dy, 0), dx, 1) & gt_haz).sum())
            if best is None or ov > best[0]:
                best = (ov, dx, dy)
    return {"best_shift_cells_xy": [best[1], best[2]], "overlap_at_best": best[0], "overlap_at_zero": int((m & gt_haz).sum())}
