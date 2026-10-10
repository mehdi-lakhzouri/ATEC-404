"""Forward RGB camera + aligned RangeFinder: initialisation, timestamped capture, conversion and diagnostic export.

Both devices share pose, field of view and resolution (see ``artemis-webots/tools/make_world.py``), so pixel (u, v)
of the RGB image and of the depth image observe the same ray. Webots conventions used here:
* ``Camera.getImage()`` -> bytes, BGRA, row-major, width*height*4  -> converted to RGB
* ``RangeFinder.getRangeImage()`` -> list of floats, metres, row-major; values at/above maxRange (or inf) mean
  "no return" (e.g. sky) and are reported as invalid, never as a distance
* sensor frame: x forward, y left, z up (Webots FLU); image u to the right, v downwards
"""

from __future__ import annotations

import array
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def bgra_to_rgb(buf: bytes, width: int, height: int) -> bytes:
    if len(buf) != width * height * 4:
        raise ValueError(f"expected {width * height * 4} BGRA bytes, got {len(buf)}")
    out = bytearray(width * height * 3)
    out[0::3] = buf[2::4]
    out[1::3] = buf[1::4]
    out[2::3] = buf[0::4]
    return bytes(out)


def depth_stats(values, min_range: float, max_range: float) -> dict:
    valid = [v for v in values if math.isfinite(v) and min_range <= v < max_range]
    n = len(values)
    s = sorted(valid)
    pct = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1)))] if s else None  # noqa: E731
    return {"pixels": n, "valid": len(valid), "valid_fraction": len(valid) / n if n else 0.0,
            "non_finite": sum(1 for v in values if not math.isfinite(v)), "no_return": sum(1 for v in values if math.isfinite(v) and v >= max_range),
            "min_m": s[0] if s else None, "p05_m": pct(0.05), "median_m": pct(0.5), "p95_m": pct(0.95), "max_m": s[-1] if s else None,
            "units": "metres along the optical axis (Webots RangeFinder)"}


def intrinsics(width: int, height: int, hfov_rad: float) -> dict:
    """Pinhole intrinsics shared by both sensors (square pixels, principal point at the image centre)."""
    fx = (width / 2) / math.tan(hfov_rad / 2)
    return {"width": width, "height": height, "fx": fx, "fy": fx, "cx": (width - 1) / 2, "cy": (height - 1) / 2,
            "hfov_rad": hfov_rad, "vfov_rad": 2 * math.atan((height / 2) / fx)}


@dataclass
class Frame:
    index: int
    sim_time_s: float
    width: int
    height: int
    rgb: bytes | None
    depth: list | None
    meta: dict = field(default_factory=dict)


class SensorSuite:
    """Enables the camera and range finder (if present) at a configurable sampling period and captures frames."""

    def __init__(self, robot, period_ms: int, camera_name: str = "front_camera", depth_name: str = "front_depth"):
        self.robot, self.period_ms = robot, period_ms
        self.errors: list[str] = []
        self.camera = self._device(camera_name, "Camera")
        self.depth = self._device(depth_name, "RangeFinder")
        for dev in (self.camera, self.depth):
            if dev is not None:
                dev.enable(period_ms)
        self.count = 0

    def _device(self, name: str, kind: str):
        try:
            dev = self.robot.getDevice(name)
        except Exception as e:  # pragma: no cover - API raises on some versions
            dev = None
            self.errors.append(f"{kind} '{name}': {e}")
        if dev is None:
            self.errors.append(f"{kind} '{name}' not found - continuing without it")
        return dev

    def describe(self) -> dict:
        d = {"period_ms": self.period_ms, "errors": self.errors}
        if self.camera is not None:
            d["camera"] = intrinsics(self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov())
        if self.depth is not None:
            d["depth"] = {**intrinsics(self.depth.getWidth(), self.depth.getHeight(), self.depth.getFov()),
                          "min_range_m": self.depth.getMinRange(), "max_range_m": self.depth.getMaxRange()}
        return d

    def capture(self) -> Frame:
        t = self.robot.getTime()
        rgb = depth = None
        w = h = 0
        if self.camera is not None:
            raw = self.camera.getImage()
            w, h = self.camera.getWidth(), self.camera.getHeight()
            rgb = bgra_to_rgb(bytes(raw), w, h) if raw else None
        if self.depth is not None:
            depth = list(self.depth.getRangeImage())
        f = Frame(self.count, t, w, h, rgb, depth)
        self.count += 1
        return f

    def export(self, frame: Frame, out_dir: Path, tag: str) -> dict:
        """PNG previews via Webots' own encoder, plus raw RGB (uint8) and depth (float32) for exact downstream use."""
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"frame{frame.index:02d}_{tag}"
        meta = {"frame": frame.index, "tag": tag, "sim_time_s": round(frame.sim_time_s, 4), "width": frame.width, "height": frame.height}
        if self.camera is not None and frame.rgb is not None:
            self.camera.saveImage(str(out_dir / f"{base}_rgb.png"), 100)
            (out_dir / f"{base}_rgb.raw").write_bytes(frame.rgb)
            meta["rgb_png"] = f"{base}_rgb.png"; meta["rgb_raw"] = f"{base}_rgb.raw (uint8 RGB, row-major)"
        if self.depth is not None and frame.depth is not None:
            self.depth.saveImage(str(out_dir / f"{base}_depth.png"), 100)
            arr = array.array("f", (v if math.isfinite(v) else float("inf") for v in frame.depth))
            (out_dir / f"{base}_depth.f32").write_bytes(arr.tobytes())
            meta["depth_png"] = f"{base}_depth.png (Webots visualisation)"; meta["depth_raw"] = f"{base}_depth.f32 (float32 metres, row-major)"
            meta["depth_stats"] = depth_stats(frame.depth, self.depth.getMinRange(), self.depth.getMaxRange())
        (out_dir / f"{base}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta
