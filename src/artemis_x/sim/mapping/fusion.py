"""YOLO + depth fusion (R4.4, plan section 6.4). Stdlib only.

Principle: DEPTH DECIDES OCCUPANCY, YOLO ADDS MEANING AND SENSITIVITY. YOLO never clears a cell and never invents one:
  * every YOLO detection becomes a ``RockMeasurement`` (an odom-frame position, size and uncertainty) built from the
    frame's OWN depth image and the ESTIMATED pose at that frame's capture time - for the obstacle tracker;
  * the small-rock rule: inside a confirmed-quality YOLO box the obstacle height threshold drops from 0.05 m to 0.02 m,
    so low rocks that depth alone classifies as ground/uncertain are mapped from their measured geometry. These are real
    depth points; YOLO only says where to look more closely.

Occlusion (R3 finding): when a nearer object covers part of the box, the box depth is bimodal. The measurement then uses
the LARGER depth cluster (the dominant surface in the box), is flagged ``depth_bimodal`` and gets a larger sigma.

Accepted detections: result OK, not stale (capture age <= max_age_s), frame still in the ring buffer, box with enough
valid depth. Everything else is counted and dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..projection import CameraModel, camera_to_rover, pixel_to_camera
from .config import MappingConfig
from .frames import Pose2D, base_to_odom


@dataclass(frozen=True)
class RockMeasurement:
    stamp_s: float                     # capture (sensor sample) time
    frame_index: int
    conf: float
    box_xyxy: tuple
    surface_odom: tuple                # visible-surface point (x, y)
    centre_odom: tuple                 # estimated footprint centre (surface + radius along the viewing ray)
    radius_m: float
    height_m: float                    # top of the box above the ground plane, from depth
    range_m: float
    sigma_m: float
    depth_bimodal: bool
    low_points: list                   # odom (x, y, z) depth points 0.02-0.05 m high inside the box (small-rock rule)
    support: int = 0                   # depth samples in the box at least fusion_low_obstacle_m above the ground


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def measure(det: dict, depth, pose: Pose2D, cam: CameraModel, cfg: MappingConfig, stamp_s: float, frame_index: int):
    """One detection -> RockMeasurement, or (None, reason)."""
    x0, y0, x1, y1 = det["xyxy"]
    mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25                # central 50 % of the box (as projection.localize)
    u0, u1 = max(0, int(x0 + mx)), min(cam.width - 1, int(math.ceil(x1 - mx)))
    v0, v1 = max(0, int(y0 + my)), min(cam.height - 1, int(math.ceil(y1 - my)))
    n_px = max(0, u1 - u0 + 1) * max(0, v1 - v0 + 1)
    if n_px == 0:
        return None, "empty box"
    stride = max(1, int(math.sqrt(n_px / cfg.fusion_max_samples)))
    lo, hi = cam.min_range, cam.max_range
    cp, sp = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
    fx, fy, cx, cy, tz = cam.fx, cam.fy, cam.cx, cam.cy, cam.mount_t[2]
    samples = []                                              # (d, u, v, z_base): z_base = tz + d * (-sp + cp * zc)
    for v in range(v0, v1 + 1, stride):
        row = v * cam.width
        az = -sp + cp * (-(v - cy) / fy)
        for u in range(u0, u1 + 1, stride):
            d = depth[row + u]
            if lo <= d < hi:
                samples.append((d, u, v, tz + d * az))
    total = len(range(v0, v1 + 1, stride)) * len(range(u0, u1 + 1, stride))
    if not samples or len(samples) < cfg.fusion_min_valid_fraction * total:
        return None, "insufficient valid depth in box"
    samples.sort()
    ds = [s[0] for s in samples]
    # bimodality: the largest gap in sorted depth; two clusters if the gap exceeds fusion_bimodal_gap_m
    gap_i, gap = 0, 0.0
    for i in range(len(ds) - 1):
        g = ds[i + 1] - ds[i]
        if g > gap:
            gap_i, gap = i, g
    bimodal = gap > cfg.fusion_bimodal_gap_m and min(gap_i + 1, len(ds) - gap_i - 1) >= 0.15 * len(ds)
    if bimodal:
        near, far = samples[:gap_i + 1], samples[gap_i + 1:]
        cluster = far if len(far) >= len(near) else near
    else:
        cluster = samples[:max(1, len(samples) // 2)]           # near half, as in R3 localize (visible surface)
    p_cam = (_median([d for d, _u, _v, _z in cluster]), _median([-(u - cx) * d / fx for d, u, _v, _z in cluster]),
             _median([-(v - cy) * d / fy for d, _u, v, _z in cluster]))
    p_base = camera_to_rover(p_cam, cam)
    rng = math.sqrt(sum(c * c for c in p_cam))
    surf = base_to_odom(p_base[:2], pose)
    radius = max(cfg.fusion_min_radius_m, 0.5 * (x1 - x0) * p_cam[0] / cam.fx)
    # centre: push the surface point back along the horizontal viewing ray by the radius
    sx, sy = cam.mount_t[0], cam.mount_t[1]
    bx, by = p_base[0] - sx, p_base[1] - sy
    nrm = math.hypot(bx, by) or 1.0
    centre = base_to_odom((p_base[0] + radius * bx / nrm, p_base[1] + radius * by / nrm), pose)
    # height: the box top row at the cluster's depth
    top = camera_to_rover(pixel_to_camera((x0 + x1) / 2, y0, p_cam[0], cam), cam)[2]
    # small-rock rule: depth points 0.02-0.05 m high inside the box (would be ground / uncertain without YOLO)
    low = []
    support = 0
    lo_h, h_obs = cfg.fusion_low_obstacle_m, cfg.obstacle_height_m
    for d, u, v, zb in samples:                               # z_base precomputed above; full point only when needed
        if zb >= lo_h:
            support += 1
            if zb < h_obs:
                pb = camera_to_rover(pixel_to_camera(u, v, d, cam), cam)
                low.append(base_to_odom(pb[:2], pose) + (pb[2],))
    sigma = (cfg.fusion_sigma0_m + cfg.fusion_sigma_per_m * rng) * (2.0 if bimodal else 1.0)
    return RockMeasurement(stamp_s, frame_index, det["conf"], tuple(det["xyxy"]), surf, centre, radius, max(0.0, top), rng,
                           sigma, bimodal, low, support), None
