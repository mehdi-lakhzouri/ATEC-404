"""Detection box + aligned optical-axis depth -> visible rock-surface point in camera and rover coordinates (pure Python).

Conventions (verified in R2): camera and range finder share pose, FOV and resolution; depth is distance along the
optical axis (camera x) in metres; ``inf`` / out-of-range pixels are invalid. Camera frame FLU (x forward, y left,
z up); image u to the right, v downwards: ``y = -(u - cx) * x / fx``, ``z = -(v - cy) * x / fy``.

What is estimated: a point on the **visible surface** of the detected object (median of the back-projected foreground
pixels inside the central part of the box). It is NOT the object's centre; no centre estimate is made in R3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Sensor mounting on the Koala (robot frame), shared with artemis-webots/tools/make_world.py
MOUNT_TRANSLATION = (0.16, 0.0, 0.24)
MOUNT_PITCH_RAD = 0.12                       # rotation about +y: optical axis tilted down
SENSOR_FOV_RAD, SENSOR_WIDTH, SENSOR_HEIGHT = 1.0, 720, 480


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    min_range: float
    max_range: float
    mount_t: tuple = MOUNT_TRANSLATION
    mount_pitch: float = MOUNT_PITCH_RAD

    @classmethod
    def from_sensor(cls, d: dict) -> "CameraModel":
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], d["min_range_m"], d["max_range_m"])


def pixel_to_camera(u: float, v: float, depth: float, cam: CameraModel) -> tuple[float, float, float]:
    return depth, -(u - cam.cx) * depth / cam.fx, -(v - cam.cy) * depth / cam.fy


def camera_to_rover(p: tuple, cam: CameraModel) -> tuple[float, float, float]:
    """p_rover = R_y(pitch) p_cam + t_mount."""
    c, s = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
    x, y, z = p
    tx, ty, tz = cam.mount_t
    return c * x + s * z + tx, y + ty, -s * x + c * z + tz


def rover_to_world(p: tuple, pose: dict) -> tuple[float, float, float]:
    """Evaluation only (uses the supervisor's ground-truth pose; never fed back to the detector)."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    x, y, z = p
    return pose["x"] + c * x - s * y, pose["y"] + s * x + c * y, pose["z"] + z


def _median(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def foreground_pixels(box: list, depth: list, cam: CameraModel, inner: float = 0.5, max_samples: int = 400) -> dict:
    """Valid depth pixels in the central ``inner`` fraction of the box; the nearer half (depth <= median) is kept as foreground."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * (1 - inner) / 2, (y1 - y0) * (1 - inner) / 2
    u0, u1 = max(0, int(math.floor(x0 + mx))), min(cam.width - 1, int(math.ceil(x1 - mx)))
    v0, v1 = max(0, int(math.floor(y0 + my))), min(cam.height - 1, int(math.ceil(y1 - my)))
    n_px = max(0, u1 - u0 + 1) * max(0, v1 - v0 + 1)
    stride = max(1, int(math.sqrt(n_px / max_samples))) if n_px else 1
    samples, total, invalid = [], 0, 0
    for v in range(v0, v1 + 1, stride):
        row = v * cam.width
        for u in range(u0, u1 + 1, stride):
            total += 1
            d = depth[row + u]
            if math.isfinite(d) and cam.min_range <= d < cam.max_range:
                samples.append((u, v, d))
            else:
                invalid += 1
    if not samples:
        return {"region": [u0, v0, u1, v1], "sampled": total, "valid": 0, "invalid": invalid, "foreground": []}
    med = _median([d for _, _, d in samples])
    fg = [s for s in samples if s[2] <= med]
    return {"region": [u0, v0, u1, v1], "sampled": total, "valid": len(samples), "invalid": invalid, "foreground": fg,
            "depth_median_m": med, "depth_min_m": min(d for _, _, d in samples)}


def localize(det: dict, depth: list | None, cam: CameraModel, min_valid_fraction: float = 0.2) -> dict:
    """Surface-point estimate for one detection with validity flags (never invents a distance)."""
    out = {"xyxy": det["xyxy"], "conf": det["conf"], "valid": False}
    if depth is None:
        out["reason"] = "no depth frame"
        return out
    fp = foreground_pixels(det["xyxy"], depth, cam)
    out.update({k: fp[k] for k in ("region", "sampled", "valid", "invalid") if k in fp})
    out["n_valid"] = out.pop("valid")
    out["valid"] = False
    frac = fp["valid"] / fp["sampled"] if fp["sampled"] else 0.0
    out["valid_fraction"] = round(frac, 3)
    if not fp["foreground"] or frac < min_valid_fraction:
        out["reason"] = "insufficient valid depth in box"
        return out
    pts = [pixel_to_camera(u, v, d, cam) for u, v, d in fp["foreground"]]
    p_cam = tuple(_median([p[i] for p in pts]) for i in range(3))
    out.update({"valid": True, "reason": None, "n_foreground": len(pts), "surface_point_camera_m": [round(c, 4) for c in p_cam],
                "surface_point_rover_m": [round(c, 4) for c in camera_to_rover(p_cam, cam)],
                "range_m": round(math.sqrt(sum(c * c for c in p_cam)), 4), "foreground_pixels": [[u, v] for u, v, _ in fp["foreground"]]})
    return out
