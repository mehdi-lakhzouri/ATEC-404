"""Webots R3 runner: inference server + Webots (headless) for the demo, the responsiveness checks and the paired
40-capture domain-gap evaluation, followed by the analysis.

    uv run --project backend --no-sync python scripts/run_webots_r3.py demo            # R1/R2 motion + async detection
    uv run --project backend --no-sync python scripts/run_webots_r3.py robustness      # no server / slow server
    uv run --project backend --no-sync python scripts/run_webots_r3.py eval            # 5 seeds x 4 viewpoints x 2 variants
    uv run --project backend --no-sync python scripts/run_webots_r3.py course          # test course v2, low sun + high sun
    uv run --project backend --no-sync python scripts/run_webots_r3.py tour            # scripted course tour (closed-loop odometry)
    uv run --project backend --no-sync python scripts/run_webots_r3.py analyse <eval_dir>

Webots runs the controller with the SYSTEM Python (stdlib only); the server runs here, in the ML environment.
This is a limited exploratory Webots domain-gap evaluation - not a real-world benchmark. No training, no threshold changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.evaluation.ignore_aware import ImageEval, ap_101, iou, match_image  # noqa: E402
from artemis_x.sim import protocol  # noqa: E402
from artemis_x.sim.projection import CameraModel  # noqa: E402

WB = ROOT / "artemis-webots"
sys.path.insert(0, str(WB / "tools"))
import make_world  # noqa: E402  (placement -> exact rendered mesh)
WEBOTS = Path(os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")) / "msys64" / "mingw64" / "bin" / "webots.exe"
SEEDS = [0, 1, 2, 3, 4]
MIN_REF_PIXELS = 50             # reference box reliable only if >= 50 visible pixels (same floor as the label policy)


def system_env(extra: dict) -> dict:
    env = dict(os.environ, **{k: str(v) for k, v in extra.items()})
    venv = str(Path(sys.executable).parent).lower().rstrip("\\/")
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if p.lower().rstrip("\\/") != venv)
    env.pop("VIRTUAL_ENV", None)
    return env


def start_server(delay_ms: float = 0, port: int = protocol.DEFAULT_PORT, log: Path | None = None) -> subprocess.Popen:
    cmd = [sys.executable, str(ROOT / "scripts" / "inference_server.py"), "--port", str(port)]
    if delay_ms:
        cmd += ["--delay-ms", str(delay_ms)]
    fh = open(log, "w", encoding="utf-8") if log else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    end = time.time() + 120
    while time.time() < end:
        try:
            import socket
            with socket.create_connection(("127.0.0.1", port), timeout=0.5) as s:
                s.settimeout(30)
                protocol.read_message(s)
                s.sendall(protocol.encode({"type": "bye"}))
            return p
        except OSError:
            if p.poll() is not None:
                raise RuntimeError("inference server exited during start-up")
            time.sleep(0.5)
    p.kill(); raise RuntimeError("inference server did not start")


def stop_server(p: subprocess.Popen, port: int = protocol.DEFAULT_PORT) -> None:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1) as s:
            s.settimeout(5); protocol.read_message(s); s.sendall(protocol.encode({"type": "shutdown"}))
        p.wait(10)
    except (OSError, subprocess.TimeoutExpired):
        p.kill()


def run_webots(world: Path, env: dict, timeout: int = 240, mode: str = "fast") -> tuple[int, str]:
    """Run one headless Webots instance. On timeout (e.g. a crashed controller leaves Webots running in batch mode) the
    whole Webots process tree is killed and the run is reported as failed instead of hanging the evaluation."""
    cmd = [str(WEBOTS), "--batch", f"--mode={mode}", "--minimize", "--stdout", "--stderr", "--port=1242", str(world)]
    p = subprocess.Popen(cmd, env=system_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        out, _ = p.communicate()
        return -9, (out or "") + f"\nTIMEOUT after {timeout}s: Webots process tree killed"


# ---- demo / robustness ---------------------------------------------------------------------------------------------------
def demo(out: Path, server: bool = True, delay_ms: float = 0) -> dict:
    srv = start_server(delay_ms, log=out.parent / f"{out.name}_server.log") if server else None
    try:
        code, log = run_webots(WB / "worlds" / "artemis_moon.wbt", {"ARTEMIS_WEBOTS_QUIT": "1", "ARTEMIS_WEBOTS_OUT": out,
                                                                    "ARTEMIS_INFER_TIMEOUT_S": "3"})
    finally:
        if srv:
            stop_server(srv)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    inf = s.get("inference", {})
    return {"run": out.name, "server": server, "delay_ms": delay_ms, "webots_exit": code,
            "r1r2_checks_pass": all(c["ok"] for c in s["checks"].values()), "sim_time_s": s["sim_time_s"], "wall_time_s": s["wall_time_s"],
            "controller_work_ms_per_step": s["controller_work_ms_per_step"],
            "inference": {k: inf.get(k) for k in ("submitted", "results", "ok", "failed", "stale", "evicted", "detections", "localized",
                                                   "latency_ms_p50", "latency_ms_p95")},
            "client": {k: (inf.get("client") or {}).get(k) for k in ("connected", "sent", "received", "dropped", "timeouts", "mismatched",
                                                                    "connect_failures", "last_error", "model_sha256")}}


# ---- evaluation ------------------------------------------------------------------------------------------------------------
def run_eval(out: Path) -> Path:
    subprocess.run([sys.executable, str(WB / "tools" / "make_world.py"), "--eval", *map(str, SEEDS)], check=True, capture_output=True)
    srv = start_server(log=out / "server.log") if out.mkdir(parents=True, exist_ok=True) is None else None
    runs = []
    try:
        for seed in SEEDS:
            for variant in ("baseline", "enhanced"):
                name = f"eval_s{seed}_{variant}"
                t0 = time.time()
                code, log = run_webots(WB / "worlds" / f"{name}.wbt", {"ARTEMIS_MODE": "eval", "ARTEMIS_WEBOTS_OUT": out / name,
                                                                         "ARTEMIS_SCENE_MANIFEST": WB / "worlds" / f"{name}.json"})
                (out / f"{name}_webots.log").write_text(log, encoding="utf-8")
                runs.append({"world": name, "exit": code, "wall_s": round(time.time() - t0, 1)})
                print(f"{name}: exit {code} ({runs[-1]['wall_s']} s)", flush=True)
    finally:
        stop_server(srv)
    (out / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return out


def run_course(out: Path) -> Path:
    subprocess.run([sys.executable, str(WB / "tools" / "make_world.py")], check=True, capture_output=True)
    out.mkdir(parents=True, exist_ok=True)
    srv = start_server(log=out / "server.log")
    runs = []
    try:
        for name in COURSE_WORLDS:
            t0 = time.time()
            code, log = run_webots(WB / "worlds" / f"{name}.wbt", {"ARTEMIS_MODE": "eval", "ARTEMIS_WEBOTS_OUT": out / name.replace("artemis_", ""),
                                                                     "ARTEMIS_SCENE_MANIFEST": WB / "worlds" / f"{name}.json"})
            (out / f"{name}_webots.log").write_text(log, encoding="utf-8")
            runs.append({"world": name, "exit": code, "wall_s": round(time.time() - t0, 1)})
            print(f"{name}: exit {code} ({runs[-1]['wall_s']} s)", flush=True)
    finally:
        stop_server(srv)
    (out / "runs.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return out


def run_tour(out: Path, port: int = protocol.DEFAULT_PORT, extra_env: dict | None = None) -> dict:
    """Default controller mode in the course world: the customData tour plan drives the rover along the lane.
    ``port``: use another port when an interactive Webots session already holds the default one."""
    subprocess.run([sys.executable, str(WB / "tools" / "make_world.py")], check=True, capture_output=True)
    out.mkdir(parents=True, exist_ok=True)
    srv = start_server(port=port, log=out / "server.log")
    try:
        code, log = run_webots(WB / "worlds" / "artemis_test_course.wbt", {"ARTEMIS_WEBOTS_QUIT": "1", "ARTEMIS_WEBOTS_OUT": out / "tour",
                                                                           "ARTEMIS_INFER_TIMEOUT_S": "3", "ARTEMIS_INFER_PORT": port,
                                                                           **(extra_env or {})}, timeout=420)
    finally:
        stop_server(srv, port)
    (out / "webots.log").write_text(log, encoding="utf-8")
    s = json.loads((out / "tour" / "summary.json").read_text(encoding="utf-8"))
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    plan = {f"observe_{x['zone']}": x for x in man["tour"]["stops"]}
    stops = []
    for m in s["moves"]:                                     # where the rover really stood when it observed each zone (GT)
        if m["move"] in plan:
            st, g = plan[m["move"]], m["ground_truth_end"]
            ox, oy = st.get("observe_x", st["x"]), st.get("observe_y", man["tour"]["lane_y"])
            stops.append({"zone": st["zone"], "x_err_mm": round((g["x"] - ox) * 1000, 1), "y_err_mm": round((g["y"] - oy) * 1000, 1),
                          "yaw_err_deg": round(math.degrees(g["yaw"] - st["look_rad"]), 2),
                          "distance_to_zone_centre_m": st.get("observe_to_zone_centre_m"), "frame": m.get("frame")})
    # clearance along the TRUE path (supervisor ground truth, every step) to every object footprint and crater rim
    obs = make_world.course_obstacles(man["rocks"], man["distractors"])
    import csv
    with (out / "tour" / "telemetry.csv").open(encoding="utf-8") as fh:
        path = [(float(r["gt_x"]), float(r["gt_y"])) for r in csv.DictReader(fh) if r["gt_x"]]
    clear = min(math.hypot(px - x, py - y) - rad for px, py in path for x, y, rad in obs)
    inf = s.get("inference", {})
    return {"webots_exit": code, "completed_moves": s["final"]["completed_moves"], "planned_moves": s["final"]["planned_moves"],
            "sim_time_s": round(sum(m["duration_s"] for m in s["moves"]), 1), "stops": stops,
            "min_clearance_along_true_path_m": round(clear, 3), "required_clearance_m": make_world.ROVER_CLEARANCE_M,
            "max_abs_x_err_mm": max(abs(x["x_err_mm"]) for x in stops), "max_abs_yaw_err_deg": max(abs(x["yaw_err_deg"]) for x in stops),
            "final_odometry_vs_ground_truth": s["final"], "controller_work_ms_per_step": s["controller_work_ms_per_step"],
            "inference": {k: inf.get(k) for k in ("submitted", "ok", "failed", "stale", "evicted", "detections", "localized",
                                                   "latency_ms_p50", "latency_ms_p95")},
            "dir": out.relative_to(ROOT).as_posix()}


COURSE_WORLDS = ("artemis_test_course", "artemis_test_course_highsun")
_MESH_CACHE: dict = {}


def rock_mesh(rk: dict) -> dict:
    """World-frame mesh of a rock: stored in older manifests, otherwise rebuilt from its placement with the generator's own
    transform (identical to the rendered Webots node)."""
    if "mesh" in rk:
        return rk["mesh"]
    if rk["name"] not in _MESH_CACHE:
        m = make_world.ph_place(**rk["placement"])
        _MESH_CACHE[rk["name"]] = {"vertices": m["vertices"], "triangles": m["triangles"]}
    return _MESH_CACHE[rk["name"]]


def manifest_path(c: dict) -> Path:
    var = c["variant"]
    return WB / "worlds" / (f"artemis_{var}.json" if var.startswith("test_course") else f"eval_s{c['seed']}_{var}.json")


def crater_box(cr: dict, cam_pose: dict, cam: CameraModel) -> list | None:
    """Image box of a crater rim (sampled on the terrain surface), for the crater false-positive category."""
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    pts = [[cr["x"] + f * cr["radius"] * math.cos(t), cr["y"] + f * cr["radius"] * math.sin(t), 0.0] for f in (0.3, 0.7, 1.15) for t in a]
    pts = [[x, y, make_world.terrain_height(x, y)] for x, y, _ in pts]
    return projected_box({"shape": "points", "points": pts}, cam_pose, cam, floor=-1.0)


def reference_boxes(seg_path: Path, w: int, h: int, rocks: list) -> list[dict]:
    """Visible-extent boxes per rock from Webots' recognition segmentation (exact occlusion handling)."""
    seg = np.frombuffer(seg_path.read_bytes(), np.uint8).reshape(h, w, 4)[..., :3][..., ::-1].astype(int)    # BGRA -> RGB
    out = []
    for rk in rocks:
        col = np.round(np.array(rk["color"]) * 255).astype(int)
        m = np.all(np.abs(seg - col) <= 2, axis=-1)
        n = int(m.sum())
        if n == 0:
            continue
        ys, xs = np.nonzero(m)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        touches = box[0] <= 0 or box[1] <= 0 or box[2] >= w or box[3] >= h
        out.append({"rock": rk["name"], "box": box, "visible_px": n, "touches_border": touches,
                    "reliable": n >= MIN_REF_PIXELS, "mask": m})
    return out


def projected_box(rk: dict, cam_pose: dict, cam: CameraModel, floor: float | None = None) -> list | None:
    """Independent reference check: bounding box of the rock's above-ground geometry projected with the GT camera pose."""
    if rk["shape"] == "sphere":
        g = np.random.default_rng(0).normal(size=(400, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
        pts = np.array(rk["center"]) + rk["r"] * g
    elif rk["shape"] == "points":
        pts = np.array(rk["points"])
    else:
        pts = np.array(rock_mesh(rk)["vertices"])
    pts = pts[pts[:, 2] >= (rk.get("z0", 0.0) if floor is None else floor)]          # above the local terrain surface
    cp, sp = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cy_, sy_ = math.cos(cam_pose["yaw"]), math.sin(cam_pose["yaw"])
    rz = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    pc = (ry.T @ rz.T @ (pts - np.array([cam_pose["x"], cam_pose["y"], cam_pose["z"]])).T).T
    pc = pc[pc[:, 0] > 0.05]
    if not len(pc):
        return None
    u = cam.cx - cam.fx * pc[:, 1] / pc[:, 0]; v = cam.cy - cam.fy * pc[:, 2] / pc[:, 0]
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def ray_hits(rk: dict, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Distance along each unit ray to the rock surface (nan if missed). Analytic sphere or Moller-Trumbore mesh."""
    if rk["shape"] == "sphere":
        c, r = np.array(rk["center"]), rk["r"]
        oc = origins - c
        b = np.einsum("ij,ij->i", oc, dirs)
        disc = b * b - (np.einsum("ij,ij->i", oc, oc) - r * r)
        t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        return np.where(t > 0, t, np.nan)
    m = rock_mesh(rk)
    V = np.array(m["vertices"]); F = np.array(m["triangles"])
    v0, e1, e2 = V[F[:, 0]], V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    best = np.full(len(dirs), np.nan)
    for i in range(len(dirs)):
        p = np.cross(dirs[i], e2); det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        s = origins[i] - v0; u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1); v = (q @ dirs[i]) * inv; t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        if hit.any():
            best[i] = t[hit].min()
    return best


def camera_rays(pixels: list, cam_pose: dict, cam: CameraModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame unit rays through pixels, plus the camera-frame directions (x = 1 normalisation for axial depth)."""
    px = np.array(pixels, float)
    d_cam = np.stack([np.ones(len(px)), -(px[:, 0] - cam.cx) / cam.fx, -(px[:, 1] - cam.cy) / cam.fy], 1)
    cp, sp = math.cos(cam.mount_pitch), math.sin(cam.mount_pitch)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cy_, sy_ = math.cos(cam_pose["yaw"]), math.sin(cam_pose["yaw"])
    rz = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
    d_world = (rz @ ry @ d_cam.T).T
    norm = np.linalg.norm(d_world, axis=1)
    origin = np.array([cam_pose["x"], cam_pose["y"], cam_pose["z"]])
    return np.repeat(origin[None], len(px), 0), d_world / norm[:, None], norm


def analyse(ev: Path) -> dict:
    caps = sorted(list(ev.glob("eval_s*/capture_*.json")) + list(ev.glob("test_course*/capture_*.json")) + list(ev.glob("course/capture_*.json")))
    from collections import defaultdict
    per_cap, rows = [], defaultdict(list)
    lat, inf_ms, loc_err = defaultdict(list), defaultdict(list), defaultdict(list)
    ref_check = []
    ref_outcomes, fp_cats = defaultdict(list), defaultdict(list)
    ov_dir = ev / "overlays"; ov_dir.mkdir(exist_ok=True)
    for cp in caps:
        c = json.loads(cp.read_text(encoding="utf-8"))
        var, d = c["variant"], cp.parent
        man = json.loads(manifest_path(c).read_text(encoding="utf-8"))
        f = c["frame"]; w, h = f["width"], f["height"]
        refs = reference_boxes(d / f["segmentation_raw"].split()[0], w, h, man["rocks"])
        drefs = reference_boxes(d / f["segmentation_raw"].split()[0], w, h, man.get("distractors", []))   # non-rock objects
        fx0 = (w / 2) / math.tan(man["camera"]["fov_rad"] / 2)
        cam0 = CameraModel(fx=fx0, fy=fx0, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h, min_range=0.05, max_range=15.0)
        for r in refs:                                       # verify segmentation boxes against projected geometry
            pb = projected_box(next(x for x in man["rocks"] if x["name"] == r["rock"]), c["camera_pose_world"], cam0)
            r["projected_iou"] = None if pb is None else round(float(iou(np.array([r["box"]]), np.array([pb]))[0, 0]), 3)
            ref_check.append({"tag": c["tag"], "rock": r["rock"], "visible_px": r["visible_px"], "touches_border": r["touches_border"],
                              "reliable": r["reliable"], "iou_seg_vs_projected": r["projected_iou"]})
        rel = [r for r in refs if r["reliable"]]
        dets = c.get("detections", [])
        pred = np.array([[*x["xyxy"], x["conf"]] for x in dets]).reshape(-1, 5)
        gt = np.array([r["box"] for r in rel]).reshape(-1, 4)
        ign = np.array([r["box"] for r in refs if not r["reliable"]]).reshape(-1, 4)       # tiny visible slivers: ignore regions
        img = ImageEval(gt, pred, ign)
        s, l, n = match_image(img, 0.5, 0.5, 50.0)
        rows[var].append((s, l, n))
        tp, fp = int((l == 1).sum()), int((l == 0).sum())
        # error analysis: which reliable references were found, what the false positives overlap with
        all_boxes = np.array([r["box"] for r in refs]).reshape(-1, 4)
        if len(pred):
            ov_all = iou(pred[:, :4], all_boxes) if len(all_boxes) else np.zeros((len(pred), 0))
            ov_rel = iou(pred[:, :4], gt) if len(gt) else np.zeros((len(pred), 0))
        # localisation: matched detections vs the simulated visible surface along the same foreground rays
        fx = (w / 2) / math.tan(man["camera"]["fov_rad"] / 2)                    # same intrinsics as the controller
        cam = CameraModel(fx=fx, fy=fx, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h, min_range=0.05, max_range=15.0)
        cboxes = [b for b in (crater_box(cr, c["camera_pose_world"], cam) for cr in man.get("craters", [])) if b is not None]
        matched = []
        if len(gt) and len(pred):
            ov = iou(pred[:, :4], gt)
            used = set()
            for k in np.argsort(-pred[:, 4]):
                j = int(np.argmax(np.where([jj not in used for jj in range(len(gt))], ov[k], -1)))
                if ov[k, j] >= 0.5 and j not in used:
                    used.add(j); matched.append((int(k), j))
        mj = {j for _, j in matched}
        for j, r in enumerate(rel):
            rk = next(x for x in man["rocks"] if x["name"] == r["rock"])
            dist = math.hypot(rk["x"] - c["camera_pose_world"]["x"], rk["y"] - c["camera_pose_world"]["y"])
            ref_outcomes[var].append({"tag": c["tag"], "zone": c["viewpoint"].get("zone"), "rock": r["rock"], "found": j in mj, "visible_px": r["visible_px"],
                                      "touches_border": r["touches_border"], "radius_m": rk["r"], "distance_m": round(dist, 2)})
        mk = {k for k, _ in matched}
        for k in range(len(pred)):
            if k in mk:
                continue
            best_all = float(ov_all[k].max()) if ov_all.shape[1] else 0.0
            best_rel = float(ov_rel[k].max()) if ov_rel.shape[1] else 0.0
            dov = iou(pred[k:k + 1, :4], np.array([r["box"] for r in drefs]).reshape(-1, 4))
            if dov.size and float(dov.max()) >= 0.3:
                cat = "on a non-rock object (distractor)"
            elif best_rel >= 0.5:
                cat = "duplicate on an already-matched rock"
            elif best_all >= 0.5:
                cat = "on a <50 px rock (ignored)"
            elif best_all >= 0.1:
                cat = "partial / merged rock box (IoU 0.1-0.5)"
            elif cboxes and float(iou(pred[k:k + 1, :4], np.array(cboxes)).max()) >= 0.05:
                cat = "on a crater (no rock)"
            else:
                cat = "background (no rock overlap: terrain texture or shadow)"
            fp_cats[var].append({"tag": c["tag"], "zone": c["viewpoint"].get("zone"), "conf": round(float(pred[k, 4]), 3), "category": cat,
                                 "area_px2": round(float((pred[k, 2] - pred[k, 0]) * (pred[k, 3] - pred[k, 1])), 1)})
        for k, j in matched:
            dl = dets[k]
            if not dl.get("valid"):
                loc_err[var].append({"tag": c["tag"], "rock": rel[j]["rock"], "valid_depth": False}); continue
            rk = next(r for r in man["rocks"] if r["name"] == rel[j]["rock"])
            pix = dl["foreground_pixels"]
            o, dw, norm = camera_rays(pix, c["camera_pose_world"], cam)
            # first surface along each ray among every rock whose visible box overlaps the detection (occluders included)
            cand = [rk] + [x for x in man["rocks"] if x["name"] != rk["name"] and any(
                r["rock"] == x["name"] and float(iou(np.array([r["box"]]), pred[k:k + 1, :4])[0, 0]) > 0 for r in refs)]
            T = np.stack([ray_hits(x, o, dw) for x in cand], 1)
            hit = ~np.all(np.isnan(T), 1)
            t = np.where(hit, np.nanmin(np.where(hit[:, None], T, np.inf), 1), np.nan)
            owner = np.argmin(np.where(np.isnan(T), np.inf, T), 1)
            ref_axial = t[hit] / norm[hit]                                   # analytic surface depth along the optical axis
            pts_ref = np.stack([ref_axial, -(np.array(pix)[hit, 0] - cam.cx) * ref_axial / cam.fx,
                                -(np.array(pix)[hit, 1] - cam.cy) * ref_axial / cam.fy], 1) if hit.any() else np.zeros((0, 3))
            ref = np.median(pts_ref, 0) if len(pts_ref) else None
            est = np.array(dl["surface_point_camera_m"])
            occ = [cand[i]["name"] for i in sorted(set(owner[hit].tolist())) if i != 0]
            loc_err[var].append({"tag": c["tag"], "rock": rk["name"], "valid_depth": True,
                                 "on_target_fraction": round(float((hit & (owner == 0)).mean()), 3), "occluded_by": occ,
                                 "surface_error_m": None if ref is None else round(float(np.linalg.norm(est - ref)), 4),
                                 "range_m": dl["range_m"]})
        lat[var].append(c["inference"].get("latency_ms")); inf_ms[var].append(c["inference"].get("inference_ms"))
        per_cap.append({"tag": c["tag"], "seed": c["seed"], "variant": var, "zone": c["viewpoint"].get("zone"), "viewpoint": c["viewpoint"],
                        "distractors_visible": [r["rock"] for r in drefs if r["reliable"]], "camera_pose_world": c["camera_pose_world"],
                        "inference_ok": c["inference"].get("ok"), "n_pred": len(dets), "n_ref_reliable": len(rel), "n_ref_unreliable": len(refs) - len(rel),
                        "tp": tp, "fp": fp, "fn": len(rel) - tp, "no_detections": len(dets) == 0,
                        "recognition_object_count": c.get("recognition_object_count_ground_truth")})
        # overlay
        im = Image.open(d / f["rgb_png"]).convert("RGB"); dr = ImageDraw.Draw(im)
        for r in refs:
            dr.rectangle(r["box"], outline=(0, 220, 0) if r["reliable"] else (255, 200, 0), width=1)
        for r in drefs:
            dr.rectangle(r["box"], outline=(0, 160, 255), width=1)
        for x in dets:
            dr.rectangle(x["xyxy"], outline=(255, 0, 255), width=2)
            lbl = f"{x['conf']:.2f}" + (f" {x['range_m']:.2f}m" if x.get("valid") else " no-depth")
            dr.text((x["xyxy"][0] + 2, max(0, x["xyxy"][1] - 11)), lbl, fill=(255, 0, 255))
        dr.rectangle([0, 0, w - 1, 14], fill=(0, 0, 0))
        dr.text((3, 2), f"{c['tag']} | YOLO magenta {len(dets)} | reference green {len(rel)} (yellow <{MIN_REF_PIXELS}px) | TP {tp} FP {fp} FN {len(rel) - tp}", fill=(255, 255, 255))
        im.save(ov_dir / f"{c['tag']}.jpg", quality=90)

    def summarise(var):
        S = np.concatenate([r[0] for r in rows[var]]) if rows[var] else np.zeros(0)
        L = np.concatenate([r[1] for r in rows[var]]) if rows[var] else np.zeros(0, np.int8)
        N = sum(r[2] for r in rows[var])
        tp, fp = int((L == 1).sum()), int((L == 0).sum())
        le = [e["surface_error_m"] * 1000 for e in loc_err[var] if e.get("surface_error_m") is not None]
        ot = sorted(e["on_target_fraction"] for e in loc_err[var] if e.get("valid_depth"))
        ro = ref_outcomes[var]
        size_bins = {"<400 px": (0, 400), "400-2500 px": (400, 2500), ">=2500 px": (2500, 10 ** 9)}
        recall_by_visible = {b: {"n": sum(1 for r in ro if a <= r["visible_px"] < z), "found": sum(1 for r in ro if a <= r["visible_px"] < z and r["found"])}
                             for b, (a, z) in size_bins.items()}
        fpc = {}
        for f_ in fp_cats[var]:
            fpc[f_["category"]] = fpc.get(f_["category"], 0) + 1
        ll = sorted(x for x in lat[var] if x is not None); ii = sorted(x for x in inf_ms[var] if x is not None)
        q = lambda a, p: round(a[min(len(a) - 1, int(p * (len(a) - 1)))], 2) if a else None  # noqa: E731
        return {"captures": len(rows[var]), "reference_boxes": N, "predictions": int(len(S)), "tp": tp, "fp": fp, "fn": N - tp,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / N, 4) if N else None,
                "AP50_from_frozen_outputs": round(ap_101(S, L, N)["ap"], 4) if N else None,
                "captures_without_detections": sum(1 for p in per_cap if p["variant"] == var and p["no_detections"]),
                "matched_with_valid_depth": sum(1 for e in loc_err[var] if e.get("valid_depth")),
                "surface_error_mm": {"n": len(le), "median": q(sorted(le), 0.5), "p95": q(sorted(le), 0.95), "max": round(max(le), 2) if le else None,
                                     "note": "sensor-derived visible-surface point vs ray-cast of the true rock geometry along the same rays (hit rays only)"},
                "foreground_on_target_fraction": {"median": q(ot, 0.5), "p05": q(ot, 0.05), "min": ot[0] if ot else None},
                "recall_by_visible_area": recall_by_visible, "false_positive_categories": fpc,
                "latency_ms_end_to_end": {"p50": q(ll, 0.5), "p95": q(ll, 0.95)}, "inference_ms_server": {"p50": q(ii, 0.5), "p95": q(ii, 0.95)}}
    res = {"label": "LIMITED EXPLORATORY WEBOTS DOMAIN-GAP EVALUATION - synthetic simulator scenes, not a real-world benchmark",
           "model": "frozen YOLO11n (sha256 413d0481...), imgsz 736, conf 0.20, NMS 0.60, max_det 300, min area 50 px^2 - unchanged",
           "reference": f"visible-extent boxes from Webots recognition segmentation (occlusion-aware); rocks with < {MIN_REF_PIXELS} visible px are ignore regions",
           "matching": "IoU >= 0.5, greedy by confidence (artemis_x.evaluation.ignore_aware.match_image)",
           "ap_note": "AP computed from the frozen operating outputs (conf >= 0.20) only - a truncated AP, not comparable to the dataset AP50",
           **{v: summarise(v) for v in sorted(rows)}, "per_capture": per_cap, "localisation": dict(loc_err),
           "reference_box_verification": ref_check, "reference_outcomes": dict(ref_outcomes), "false_positives": dict(fp_cats),
           "zones": zone_summary(per_cap, ref_outcomes, fp_cats)}
    (ev / "r3_eval_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def zone_summary(per_cap: list, ref_outcomes: dict, fp_cats: dict) -> dict:
    z = {}
    for var, ro in ref_outcomes.items():
        for r in ro:
            k = f"{var}:{r.get('zone')}"
            z.setdefault(k, {"rocks_reliable": 0, "found": 0, "fp": 0, "fp_on_distractors": 0})
            z[k]["rocks_reliable"] += 1; z[k]["found"] += r["found"]
    for var, fl in fp_cats.items():
        for f in fl:
            k = f"{var}:{f.get('zone')}"
            z.setdefault(k, {"rocks_reliable": 0, "found": 0, "fp": 0, "fp_on_distractors": 0})
            z[k]["fp"] += 1; z[k]["fp_on_distractors"] += f["category"].startswith("on a non-rock")
    return z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("demo", "robustness", "eval", "course", "tour", "analyse"))
    ap.add_argument("path", nargs="?")
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if a.cmd == "demo":
        r = demo(WB / "output" / f"r3-demo-{stamp}")
    elif a.cmd == "robustness":
        r = {"no_server": demo(WB / "output" / f"r3-noserver-{stamp}", server=False),
             "slow_server_400ms": demo(WB / "output" / f"r3-slow-{stamp}", delay_ms=400)}
    elif a.cmd == "eval":
        out = run_eval(WB / "output" / f"r3-eval-{stamp}")
        r = analyse(out); r = {k: r[k] for k in ("label", "baseline", "enhanced")}; r["dir"] = out.relative_to(ROOT).as_posix()
    elif a.cmd == "course":
        out = run_course(WB / "output" / f"course-{stamp}")
        res = analyse(out)
        r = {"test_course": res.get("test_course"), "test_course_highsun": res.get("test_course_highsun"), "zones": res["zones"],
             "dir": out.relative_to(ROOT).as_posix()}
    elif a.cmd == "tour":
        r = run_tour(WB / "output" / f"tour-{stamp}")
    else:
        res = analyse(Path(a.path)); r = {k: v for k, v in res.items() if k not in ("per_capture", "localisation", "reference_box_verification",
                                                                                    "reference_outcomes", "false_positives")}
    print(json.dumps(r, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
