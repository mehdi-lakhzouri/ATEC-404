"""R3 perception in the Webots controller (stdlib only): asynchronous YOLO requests, depth association, localisation,
and the evaluation-capture mode.

Everything here runs on the Webots simulation thread except the network I/O, which lives in ``DetectionClient``'s worker.
Supervisor ground truth (rover pose, recognition segmentation) is logged for evaluation only; it is never sent to the
detector and never used to produce a detection.
"""

from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

from .client import DetectionClient
from .projection import CameraModel, localize


def draw_detections(display, rgb: bytes, w: int, h: int, dets: list, frame_id: int, latency_ms: float, encoding: str = "rgb8") -> None:
    """Paint the analysed frame (not the newest one) with its YOLO boxes: magenta = depth-localised, orange = no valid depth."""
    img = display.imageNew(rgb, display.BGRA if encoding == "bgra8" else display.RGB, w, h)
    display.imagePaste(img, 0, 0, False)
    display.imageDelete(img)
    for d in dets:
        x0, y0, x1, y1 = (int(round(v)) for v in d["xyxy"])
        display.setColor(0xFF00FF if d.get("valid") else 0xFFA000)
        for k in (0, 1):                                            # 2 px outline
            display.drawRectangle(x0 - k, y0 - k, max(1, x1 - x0 + 2 * k), max(1, y1 - y0 + 2 * k))
        lbl = f"rock {d['conf']:.2f}" + (f" {d['range_m']:.1f}m" if d.get("valid") else " no depth")
        display.drawText(lbl, x0, max(0, y0 - 11))
    display.setColor(0x000000)
    display.fillRectangle(0, 0, w, 14)
    display.setColor(0xFFFFFF)
    display.drawText(f"YOLO11n (frozen) | frame {frame_id} | {len(dets)} rock(s) | {latency_ms:.0f} ms", 3, 3)


class InferenceLink:
    """Bounded async pipeline: capture every ``every_s`` -> submit -> poll -> associate with the frame's own depth + pose."""

    def __init__(self, sensors, client: DetectionClient, out_dir: Path, every_s: float = 0.256, max_age_s: float = 1.0,
                 ring: int = 8, save_frames: int = 20, display=None, on_detections=None):
        self.sensors, self.client, self.out = sensors, client, out_dir
        self.display = display                     # optional Webots Display "yolo_view": live frame + YOLO boxes (GUI only)
        self.on_detections = on_detections         # R4.4: callback(detections, depth, pose, stamp, frame_id, t_now) -> track ids
        self.every_s, self.max_age_s = every_s, max_age_s
        self.cam = CameraModel.from_sensor(sensors.describe()["depth"])
        self.ring: collections.OrderedDict = collections.OrderedDict()
        self.ring_size, self.save_frames = ring, save_frames
        self.next_t = 0.0
        self.stats = {"submitted": 0, "results": 0, "ok": 0, "failed": 0, "stale": 0, "evicted": 0, "detections": 0,
                      "localized": 0, "latency_ms": [], "saved_frames": 0}
        self.last_ok_sim_s = None                       # R4.3 health inputs (simulation time): last successful answer,
        self.last_stale = False                         # fresh or not, and whether the latest answer was stale
        self.recent = collections.deque(maxlen=64)      # (received sim time, stale) of successful answers
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (out_dir / "detections.jsonl").open("w", encoding="utf-8")

    def tick(self, t: float, pose: dict | None, gt: dict | None = None, frame=None, stamp_s: float | None = None) -> None:
        """``pose``: the ESTIMATED pose at the frame's sample time (odom, R4). ``gt``: supervisor pose, logged under
        ``gt_pose_eval_only`` and never used. ``frame``: an already captured frame (the R4 mapper's) to submit instead of
        capturing a second one; it is submitted whenever given, otherwise this link captures on its own timer."""
        f = None
        if frame is not None:
            f = frame
        elif t >= self.next_t and self.sensors.camera is not None:
            f = self.sensors.capture()
            self.next_t = t + self.every_s
        if f is not None:
            if f.rgb is not None:
                stamp = f.sim_time_s if stamp_s is None else stamp_s
                enc = (f.meta or {}).get("encoding", "rgb8")
                self.ring[f.index] = (stamp, f.depth, pose, gt, f.rgb, f.width, f.height, enc)
                while len(self.ring) > self.ring_size:
                    self.ring.popitem(last=False)
                self.client.submit(f.index, f.sim_time_s, f.width, f.height, f.rgb, enc)
                self.stats["submitted"] += 1
        for res in self.client.poll():
            self.handle(res, t)

    def handle(self, res: dict, t_now: float) -> dict:
        self.stats["results"] += 1
        fid = res["frame_id"]
        age = t_now - res["sim_time_s"]
        rec = {"frame_id": fid, "sim_time_s": res["sim_time_s"], "received_sim_time_s": round(t_now, 4), "age_s": round(age, 4),
               "stale": age > self.max_age_s, "ok": res.get("ok", False), "latency_ms": round(res.get("latency_ms") or 0, 2),
               "inference_ms": res.get("inference_ms"), "device": res.get("device"), "error": res.get("error")}
        self.stats["stale"] += rec["stale"]
        if t_now == t_now and rec["ok"]:                    # finish() drains with t_now = NaN; failures are not answers
            self.recent.append((t_now, rec["stale"]))
            self.last_ok_sim_s = t_now                      # the server answers (possibly late): it is available
            self.last_stale = rec["stale"]
        if not rec["ok"]:
            self.stats["failed"] += 1
        else:
            self.stats["ok"] += 1
            self.stats["latency_ms"].append(rec["latency_ms"])
            entry = self.ring.get(fid)
            if entry is None:
                self.stats["evicted"] += 1
                rec["association"] = "frame evicted from ring buffer (reply too late)"
                rec["detections"] = res["detections"]
            else:
                stamp, depth, pose, gt, rgb, w, h, *rest = entry
                enc = rest[0] if rest else "rgb8"
                rec["association"] = "same capture (RGB and depth from one sensor sample)"
                rec["sample_time_s"] = round(stamp, 4)
                rec["rover_pose_estimated"] = pose
                if gt is not None:
                    rec["gt_pose_eval_only"] = gt
                locs = [localize(d, depth, self.cam) for d in res["detections"]]
                for loc in locs:
                    loc.pop("foreground_pixels", None)
                if self.on_detections is not None and not rec["stale"] and pose is not None:   # empty results count misses
                    ids = self.on_detections(res["detections"], depth, pose, stamp, fid, t_now)
                    for loc, tid in zip(locs, ids or []):
                        loc["track_id"] = tid
                rec["detections"] = locs
                self.stats["detections"] += len(locs)
                self.stats["localized"] += sum(1 for l in locs if l["valid"])
                if self.display is not None:
                    draw_detections(self.display, rgb, w, h, locs, fid, rec["latency_ms"], enc)
                if self.stats["saved_frames"] < self.save_frames:
                    fr = self.out / "frames"
                    fr.mkdir(exist_ok=True)
                    kind = "bgra" if enc == "bgra8" else "rgb"
                    (fr / f"f{fid:05d}_{kind}.raw").write_bytes(rgb)
                    rec["saved_rgb"] = f"frames/f{fid:05d}_{kind}.raw ({w}x{h} {kind.upper()})"
                    self.stats["saved_frames"] += 1
        self.log.write(json.dumps(rec) + "\n")
        return rec

    def health_status(self, t: float, window_s: float = 5.0) -> dict:
        """Inputs for the R4.3 health monitor: connection, last fresh result, recent stale fraction."""
        rec = [s for (tr, s) in self.recent if t - tr <= window_s]
        return {"enabled": True, "connected": bool(self.client.status.get("connected")), "last_ok_sim_s": self.last_ok_sim_s,
                "latest_stale": self.last_stale, "stale_fraction": (sum(rec) / len(rec)) if rec else 0.0, "results_in_window": len(rec)}

    def finish(self, drain_s: float = 0.0) -> dict:
        end = time.perf_counter() + drain_s
        while time.perf_counter() < end:
            for res in self.client.poll():
                self.handle(res, float("nan"))
            time.sleep(0.02)
        self.log.close()
        lat = sorted(self.stats["latency_ms"])
        s = {k: v for k, v in self.stats.items() if k != "latency_ms"}
        s["latency_ms_p50"] = lat[len(lat) // 2] if lat else None
        s["latency_ms_p95"] = lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))] if lat else None
        s["client"] = self.client.snapshot()
        return s


def camera_world_pose(rover_pose: dict, cam: CameraModel) -> dict:
    c, s = math.cos(rover_pose["yaw"]), math.sin(rover_pose["yaw"])
    tx, ty, tz = cam.mount_t
    return {"x": rover_pose["x"] + c * tx - s * ty, "y": rover_pose["y"] + s * tx + c * ty, "z": rover_pose["z"] + tz,
            "yaw": rover_pose["yaw"], "pitch_down_rad": cam.mount_pitch}


def run_eval(robot, rover, sensors, client: DetectionClient | None, manifest: dict, out_dir: Path,
             settle_steps: int = 6, reply_timeout_s: float = 20.0) -> dict:
    """Visit the manifest's viewpoints (supervisor teleport - evaluation harness, not navigation), capture RGB + depth,
    request detections WITHOUT blocking the simulation (it keeps stepping while waiting), and log ground truth."""
    ts = rover.timestep
    cam = CameraModel.from_sensor(sensors.describe()["depth"])
    camera = sensors.camera
    if camera.hasRecognition():
        camera.recognitionEnable(sensors.period_ms)
        camera.enableRecognitionSegmentation()
    node = rover.self_node
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = []
    for vp in manifest["viewpoints"]:
        node.getField("translation").setSFVec3f([vp["x"], vp["y"], 0.0])
        node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, vp["yaw"]])
        node.resetPhysics()
        rover.stop()
        for _ in range(settle_steps):
            robot.step(ts)
        pose = rover.pose()
        f = sensors.capture()
        tag = f"s{manifest['seed']}_{manifest['variant']}_v{vp['id']}"
        meta = sensors.export(f, out_dir, tag)
        seg = None
        if camera.hasRecognition() and camera.isRecognitionSegmentationEnabled():
            seg = camera.getRecognitionSegmentationImage()
            (out_dir / f"frame{f.index:02d}_{tag}_seg.bgra").write_bytes(bytes(seg))
            meta["segmentation_raw"] = f"frame{f.index:02d}_{tag}_seg.bgra (uint8 BGRA, ground truth - evaluation only)"
        # Only the object COUNT is read: dereferencing the per-object colour pointers (getRecognitionObjects()[i].getColors())
        # crashed the controller natively in Webots R2025a. Reference boxes come from the segmentation image instead.
        n_obj = camera.getRecognitionNumberOfObjects() if camera.hasRecognition() else 0
        rec = {"tag": tag, "seed": manifest["seed"], "variant": manifest["variant"], "viewpoint": vp, "rover_pose_ground_truth": pose,
               "camera_pose_world": camera_world_pose(pose, cam), "frame": meta, "recognition_object_count_ground_truth": n_obj}
        if client is None:
            rec["inference"] = {"ok": False, "error": "inference disabled"}
        else:
            t_submit = time.perf_counter()
            client.submit(f.index, f.sim_time_s, f.width, f.height, f.rgb)
            got, steps_waited = None, 0
            while got is None and time.perf_counter() - t_submit < reply_timeout_s:
                if robot.step(ts) == -1:                      # simulation keeps running while we wait
                    break
                steps_waited += 1
                for r in client.poll():
                    if r["frame_id"] == f.index:
                        got = r
            if got is None:
                rec["inference"] = {"ok": False, "error": "no reply within timeout", "steps_waited": steps_waited}
            else:
                rec["inference"] = {k: got.get(k) for k in ("ok", "error", "latency_ms", "inference_ms", "device", "model_sha256")}
                rec["inference"]["steps_waited_while_simulation_ran"] = steps_waited
                rec["detections"] = [localize(d, f.depth, cam) for d in got.get("detections", [])] if got.get("ok") else []
        (out_dir / f"capture_{tag}.json").write_text(json.dumps(rec), encoding="utf-8")
        caps.append({"tag": tag, "ok": rec["inference"].get("ok"), "n_detections": len(rec.get("detections", []))})
    return {"captures": caps, "client": client.snapshot() if client else None}
