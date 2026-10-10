"""Stable mapping output contract (R4.6): the obstacle list the telemetry module will consume.

Schema ``artemis.obstacles.v1`` - one JSON object, stdlib only, read-only on the mapping state:

    {"schema": "artemis.obstacles.v1", "frame": "odom", "stamp_s": <sim time>, "pose": {x, y, yaw, sigma_xy, stamp_s},
     "grid": {"resolution_m", "origin_xy", "size_m"}, "config_hash": ..., "map_resets": <int>,
     "obstacles": [ {
        "id":            "trk-<n>" (persistent tracker id, never reused within a mission)
                         | "dep-<i>_<j>" (depth cluster; i, j = global cell index of its lowest-left cell - NOT persistent:
                           a cluster that grows or splits gets a new id),
        "type":          "rock" (YOLO class) | "obstacle" (depth, positive) | "negative_hazard" (depth, crater bowl),
        "x_m", "y_m":    centre in the odom frame (estimated pose; ground truth is never used),
        "radius_m":      tracks: estimated radius; clusters: max centre-to-cell distance + half a cell,
        "confidence":    tracks: YOLO confidence (EMA, 0-1); clusters: occupancy probability from the mean log-odds (0-1),
        "uncertainty_m": 1-sigma position uncertainty relative to the rover: tracks: tracker sigma; clusters: largest
                         odometry sigma_rel of its cells (R4.3 model) + half a cell,
        "timestamp_s":   tracks: last associated detection; clusters: latest cell update (simulation time),
        "source":        "yolo+depth" (track: YOLO box localised with the same frame's depth) | "depth",
        "status":        tracks: "confirmed" | "lost" (out of view or missed - the object is kept); clusters: "observed",
        "protected":     true for a confirmed small rock whose cells are withheld from FREE evidence (plan D3),
        "track_ids":     clusters only: ids of tracks whose centre lies inside the cluster (same physical object)
     } ... ]}

Tracks and depth clusters may describe the same object (a tracked boulder is also an occupied blob); consumers that need
one entry per object use ``track_ids``. Tentative tracks are never exported. UNKNOWN space is not an obstacle and is not
listed: absence from this list does NOT mean free (use the occupancy / cost layers for traversability).
Cost: about 5-15 ms (connected components over the 120 x 120 window) - call it off the mapping cycle.
"""

from __future__ import annotations

import math

from .grid import NEGATIVE, OCCUPIED

SCHEMA = "artemis.obstacles.v1"


def _clusters(grid) -> list:
    """8-connected components of OCCUPIED / NEGATIVE cells -> list of cell-index lists."""
    n, seen, out = grid.n, set(), []
    occ = [k for k in range(n * n) if grid.state_at(k) in (OCCUPIED, NEGATIVE)]
    occ_set = set(occ)
    for k0 in occ:
        if k0 in seen:
            continue
        comp, stack = [], [k0]
        seen.add(k0)
        while stack:
            k = stack.pop()
            comp.append(k)
            j, i = divmod(k, n)
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    jj, ii = j + dj, i + di
                    if 0 <= jj < n and 0 <= ii < n:
                        kk = jj * n + ii
                        if kk in occ_set and kk not in seen:
                            seen.add(kk)
                            stack.append(kk)
        out.append(comp)
    return out


def export_obstacles(mapper) -> dict:
    g, cm, est = mapper.grid, mapper.costmap, mapper.estimator
    now = est.pose()
    now_DA = (est.odo.dist, est.odo.turned)
    rover = (now.x, now.y)
    res = g.res
    out = []
    tracks = [t for t in mapper.tracker.tracks if t.status in ("confirmed", "lost")]
    r_max = mapper.cfg.fusion_protect_max_radius_m
    for t in tracks:
        out.append({"id": f"trk-{t.id}", "type": t.cls, "x_m": round(t.x, 3), "y_m": round(t.y, 3),
                    "radius_m": round(t.radius_m, 3), "confidence": round(t.confidence, 3), "uncertainty_m": round(t.sigma, 3),
                    "timestamp_s": round(t.last_seen_s, 3), "source": "yolo+depth", "status": t.status,
                    "protected": bool(mapper.cfg.fusion_protect_small_rocks and t.radius_m <= r_max)})
    for comp in _clusters(g):
        pts = [g.cell_centre(k) for k in comp]
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        rad = max(math.hypot(px - cx, py - cy) for px, py in pts) + res / 2
        neg = sum(1 for k in comp if g.neg[k]) * 2 > len(comp)
        lo = sum(g.logodds[k] for k in comp) / len(comp)
        sig = max(cm.sigma_rel(k, now_DA, rover) for k in comp) + res / 2
        j, i = divmod(min(comp, key=lambda k: (k // g.n, k % g.n)), g.n)
        gi, gj = int(round(g.origin_x / res)) + i, int(round(g.origin_y / res)) + j
        inside = [f"trk-{t.id}" for t in tracks if math.hypot(t.x - cx, t.y - cy) <= rad]
        out.append({"id": f"dep-{gi}_{gj}", "type": "negative_hazard" if neg else "obstacle", "x_m": round(cx, 3),
                    "y_m": round(cy, 3), "radius_m": round(rad, 3), "confidence": round(1 - 1 / (1 + math.exp(lo)), 3),
                    "uncertainty_m": round(sig, 3), "timestamp_s": round(max(g.last_obs[k] for k in comp), 3),
                    "source": "depth", "status": "observed", "protected": any(k in g.protected for k in comp),
                    "track_ids": inside, "cells": len(comp)})
    return {"schema": SCHEMA, "frame": "odom", "stamp_s": round(now.stamp_s, 3), "pose": now.to_dict(),
            "grid": {"resolution_m": res, "origin_xy": [round(g.origin_x, 3), round(g.origin_y, 3)], "size_m": g.n * res},
            "config_hash": mapper.cfg.hash(), "map_resets": g.resets, "obstacles": out}
