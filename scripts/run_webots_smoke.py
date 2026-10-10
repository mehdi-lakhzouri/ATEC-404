"""Automated Webots R2 smoke test: launch artemis_moon.wbt headless, let the rover controller run its scripted demo,
then verify its diagnostics independently.

Checks (in addition to the controller's own self-checks in summary.json):
  * RGB conversion: raw RGB bytes == Webots' own PNG export, pixel for pixel
  * spatial consistency: each test rock visible in a frame is projected into the image from the ground-truth rover
    pose and the camera mounting; the measured depth at that pixel must match the analytic ray-sphere distance
    (this also determines whether the RangeFinder reports optical-axis depth or Euclidean range)

Runs Webots with the SYSTEM Python on PATH (as a normal GUI launch would), not the project venv.
Usage (repo root):  uv run --project backend --no-sync python scripts/run_webots_smoke.py [--rendering]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "artemis-webots" / "worlds" / "artemis_moon.wbt"
WEBOTS = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")) / "msys64" / "mingw64" / "bin" / "webots.exe"

spec = importlib.util.spec_from_file_location("make_world", ROOT / "artemis-webots" / "tools" / "make_world.py")
mw = importlib.util.module_from_spec(spec); spec.loader.exec_module(mw)


def run_webots(out: Path, rendering: bool, timeout: int) -> tuple[int, str]:
    env = dict(os.environ, ARTEMIS_WEBOTS_QUIT="1", ARTEMIS_WEBOTS_OUT=str(out))
    venv_scripts = str(Path(sys.executable).parent).lower()
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if p.lower().rstrip("\\/") != venv_scripts.rstrip("\\/"))
    env.pop("VIRTUAL_ENV", None)
    cmd = [str(WEBOTS), "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr", "--port=1240", str(WORLD)]
    if not rendering:
        cmd.insert(3, "--no-rendering")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def spatial_check(frames_dir: Path, frames: list, cam: dict) -> list[dict]:
    t_mount = np.array([float(v) for v in mw.SENSOR["translation"].split()])
    ax = [float(v) for v in mw.SENSOR["rotation"].split()]
    assert ax[:3] == [0, 1, 0]
    r_mount = _ry(ax[3])
    fx, fy, cx, cy, w, h = cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["width"], cam["height"]
    out = []
    for f in frames:
        pose = f["rover_pose"]
        depth = np.fromfile(frames_dir / f["depth_raw"].split()[0], np.float32).reshape(h, w)
        r_robot, p_robot = _rz(pose["yaw"]), np.array([pose["x"], pose["y"], pose["z"]])
        for name, x, y, r in mw.ROCKS:
            centre_w = np.array([x, y, r * 0.6])
            c_cam = r_mount.T @ (r_robot.T @ (centre_w - p_robot) - t_mount)          # camera frame: x fwd, y left, z up
            if c_cam[0] <= 0.1:
                continue
            u, v = cx - fx * c_cam[1] / c_cam[0], cy - fy * c_cam[2] / c_cam[0]
            if not (10 <= u < w - 10 and 10 <= v < h - 10):
                continue
            d = np.array([1.0, (cx - round(u)) / fx, (cy - round(v)) / fy])      # ray through the sampled pixel
            a, b, c = d @ d, -2 * d @ c_cam, c_cam @ c_cam - r * r
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            s = (-b - math.sqrt(disc)) / (2 * a)
            axial, euclid = s, s * math.sqrt(a)
            meas = float(depth[int(round(v)), int(round(u))])
            out.append({"frame": f["tag"], "rock": name, "pixel_uv": [round(u, 1), round(v, 1)], "measured_m": round(meas, 4),
                        "expected_axial_m": round(axial, 4), "expected_euclidean_m": round(euclid, 4),
                        "abs_err_axial_m": round(abs(meas - axial), 4), "abs_err_euclidean_m": round(abs(meas - euclid), 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rendering", action="store_true", help="run with rendering enabled (default: --no-rendering)")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    if not WEBOTS.exists():
        print(f"Webots not found at {WEBOTS}"); return 2
    out = ROOT / "artemis-webots" / "output" / ("smoke-" + time.strftime("%Y%m%d-%H%M%S"))
    t0 = time.time()
    code, log = run_webots(out, a.rendering, a.timeout)
    (out.mkdir(parents=True, exist_ok=True), (out / "webots_console.log").write_text(log, encoding="utf-8"))
    res = {"webots_exit_code": code, "wall_s": round(time.time() - t0, 1), "world": WORLD.relative_to(ROOT).as_posix(),
           "rendering": a.rendering, "python_used_by_webots": shutil.which("python", path=os.pathsep.join(
               p for p in os.environ["PATH"].split(os.pathsep) if Path(sys.executable).parent.as_posix().lower() not in Path(p).as_posix().lower())),
           "controller_started": "Starting controller: python" in log}
    summ = out / "summary.json"
    if not summ.exists():
        res["error"] = "controller produced no summary.json"
        print(json.dumps(res, indent=1)); print(log[-3000:]); return 1
    s = json.loads(summ.read_text(encoding="utf-8"))
    res["controller_checks"] = {k: v["ok"] for k, v in s["checks"].items()}
    fd = out / "frames"
    rgb_ok = []
    for f in s["frames"]:
        png = np.asarray(Image.open(fd / f["rgb_png"]).convert("RGB"))
        raw = np.frombuffer((fd / f["rgb_raw"].split()[0]).read_bytes(), np.uint8).reshape(f["height"], f["width"], 3)
        rgb_ok.append(bool(np.array_equal(png, raw)))
    res["rgb_raw_equals_webots_png"] = rgb_ok
    sc = spatial_check(fd, s["frames"], s["sensor"]["depth"])
    res["spatial_consistency"] = sc
    axial_ok = bool(sc) and all(r["abs_err_axial_m"] < 0.03 for r in sc)
    euclid_ok = bool(sc) and all(r["abs_err_euclidean_m"] < 0.03 for r in sc)
    res["depth_semantics"] = "optical-axis depth" if axial_ok and not euclid_ok else ("euclidean range" if euclid_ok and not axial_ok else "ambiguous")
    res["all_pass"] = bool(code == 0 and all(res["controller_checks"].values()) and all(rgb_ok) and (axial_ok or euclid_ok))
    (out / "smoke_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
