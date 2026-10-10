"""Mapping orchestrator - the only mapping object the controller talks to.

Per simulation step: the estimated pose. Per cycle (every ``cfg.cycle_s`` on a sensor sampling instant, so the frame's
sample time is exact): depth measurement (R4.1) -> rolling log-odds grid (R4.2) -> inflated cost layer and advisory
health (R4.3) -> snapshot every ``snapshot_every_s``. YOLO results (R4.4) arrive asynchronously through ``fuse()``:
obstacle measurements -> tracker; depth points 0.02-0.05 m high inside a confirmed rock's box -> grid hits.

Faults (R4.3) are injected only when a FaultInjector is passed in (ARTEMIS_FAULTS); a frozen depth frame is detected
(byte-identical while the rover moved) and never integrated.

Ground truth: the controller may pass the supervisor pose as ``gt``; it is only written to telemetry under
``gt_pose_eval_only`` for offline evaluation and is never read by any computation in this package.
"""

from __future__ import annotations

import gc
import json
import math
import queue
import threading
import time
import zlib
from pathlib import Path

from ..projection import CameraModel
from .config import MappingConfig
from .costmap import CostLayer
from .depth_scan import DepthScanner
from .frames import Pose2D, wrap
from .fusion import measure
from .tracker import ObstacleTracker
from .grid import SRC_YOLO_FOOTPRINT, LogOddsGrid
from .health import HealthMonitor
from .pose import PoseBuffer, PoseEstimator, sample_time
from .safety import SafetyInterface
from .snapshot import snapshot_payload, write_payload
from .uncertainty import OdometerHistory, UncertaintyModel


def process_memory_mb() -> float | None:
    """Resident memory of this (controller) process, stdlib only: Windows psapi via ctypes, else resource."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
        gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        gpmi.restype = wintypes.BOOL
        if gpmi(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return round(c.WorkingSetSize / 2 ** 20, 1)
    except (AttributeError, OSError, ctypes.ArgumentError):
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:
        return None


def _q(xs: list, p: float):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 3) if s else None


class Mapper:
    def __init__(self, sensors, out_dir: Path, cfg: MappingConfig | None = None, estimator: PoseEstimator | None = None,
                 want_rgb: bool = True, faults=None):
        self.cfg = cfg or MappingConfig()
        self.want_rgb = want_rgb                       # False when no inference link consumes the RGB image
        self.sensors = sensors
        self.period_s = sensors.period_ms / 1000.0
        self.cam = CameraModel.from_sensor(sensors.describe()["depth"])
        self.estimator = estimator or PoseEstimator(self.cfg)
        self.buffer = PoseBuffer(self.cfg.pose_buffer_s)
        self.scanner = DepthScanner(self.cam, self.cfg)
        self.grid = LogOddsGrid(self.cfg)
        self.model = UncertaintyModel(self.cfg)
        self.history = OdometerHistory()
        self.costmap = CostLayer(self.cfg, self.grid, self.model, self.history)
        self.health_monitor = HealthMonitor(self.cfg)
        self.safety = SafetyInterface(self.cfg, self.grid, self.costmap, self.health_monitor, self.estimator)
        self.faults = faults
        self.tracker = ObstacleTracker(self.cfg)
        self.fusion_stats = {"results": 0, "detections": 0, "measurements": 0, "rejected": {}, "bimodal": 0,
                             "small_rock_hits": 0, "latency_s": [], "fuse_ms": []}
        self.inference = {"enabled": False}             # set every step by the controller (InferenceLink.health_status)
        self.last_depth_stamp = None                   # sample time of the last ACCEPTED (integrated) depth frame
        self._prev = None                              # (crc, pose) of the previous depth frame, for freeze detection
        self.frozen_flag = False
        self.no_pose_flag = False
        self.last_cycle_ms = 0.0
        self.next_t = 0.0
        self.next_snap = 0.0
        self.snapshots = 0
        self.pending = None                            # (obs, frame index, timings) scanned, waiting for the grid update
        self._q: queue.Queue = queue.Queue(maxsize=8)  # snapshot payloads -> writer thread (disk I/O off the sim thread)
        self.snap_dropped = 0
        self._writer = threading.Thread(target=self._write_loop, name="artemis-map-snapshots", daemon=True)
        self._writer.start()
        self.mem_start_mb = process_memory_mb()
        self.gc_stats = {"young": 0, "full": 0, "full_ms": []}
        self._gc_next_full = 0.0
        self._gc_was_enabled = gc.isenabled()
        if self.cfg.gc_managed:
            gc.collect()
            gc.freeze()                                    # start-up objects never need scanning again
            gc.disable()
        self.mem_trace = []                            # (cycle, resident MB) every 200 cycles: growth vs leak
        self.out = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tele = (out_dir / "mapping.jsonl").open("w", encoding="utf-8")
        self.pts_log = (out_dir / "obstacle_points.jsonl").open("w", encoding="utf-8")
        self.timing = {"read_ms": [], "scan_ms": [], "grid_ms": [], "cost_ms": [], "health_ms": [], "snapshot_ms": [],
                       "cycle_ms": [], "step_max_ms": []}
        self.stats = {"cycles": 0, "no_pose": 0, "no_depth": 0, "points_total": 0, "frozen_frames_rejected": 0}
        (out_dir / "mapping_config.json").write_text(json.dumps({"config": self.cfg.to_dict(), "config_hash": self.cfg.hash(),
                                                                 "sensor_period_s": self.period_s, "camera": self.cam.__dict__,
                                                                 "faults": faults.spec if faults is not None else None},
                                                                default=str, indent=1), encoding="utf-8")

    # ---- every simulation step -----------------------------------------------------------------------------------
    def update_pose(self, enc_left, enc_right, t: float) -> Pose2D:
        p = self.estimator.update(enc_left, enc_right, t)
        self.buffer.add(p)
        return p

    def fuse(self, detections: list, depth, pose: dict, stamp: float, frame_index: int, t_now: float) -> list:
        """R4.4: one fresh YOLO result (its own frame's depth, the ESTIMATED pose at that frame's capture time) -> tracker;
        the small-rock rule adds depth points 0.02-0.05 m high from boxes whose track is confirmed. Returns track ids."""
        cfg = self.cfg
        if not cfg.fusion_enabled or depth is None:
            return [None] * len(detections)
        t0 = time.perf_counter()
        fs = self.fusion_stats
        fs["results"] += 1
        fs["detections"] += len(detections)
        if t_now - stamp > cfg.fusion_max_age_s:
            fs["rejected"]["stale"] = fs["rejected"].get("stale", 0) + len(detections)
            return [None] * len(detections)
        p = Pose2D(pose["x"], pose["y"], pose["yaw"], stamp)
        meas, idx = [], []
        for i, d in enumerate(detections):
            m, why = measure(d, depth, p, self.cam, cfg, stamp, frame_index)
            if m is None:
                fs["rejected"][why] = fs["rejected"].get(why, 0) + 1
                continue
            meas.append(m); idx.append(i)
            fs["bimodal"] += int(m.depth_bimodal)
        fs["measurements"] += len(meas)
        ids = self.tracker.update(meas, p, stamp, depth, self.cam)
        conf = {tr.id for tr in self.tracker.confirmed() if tr.status == "confirmed"}
        low = [pt for m, tid in zip(meas, ids) if tid in conf for pt in m.low_points]
        if low:
            fs["small_rock_hits"] += self.grid.add_hits(low, stamp)
        if cfg.fusion_protect_small_rocks:              # plan 6.4 + D3: a confirmed small rock's footprint gets evidence
            byid = {tr.id: tr for tr in self.tracker.tracks}
            foot = []
            for tid in set(i for i in ids if i in conf):
                tr = byid[tid]
                if tr.radius_m <= cfg.fusion_protect_max_radius_m:
                    rr = max(tr.radius_m, self.grid.res / 2)
                    n_ = int(math.ceil(rr / self.grid.res))
                    foot += [(tr.x + di * self.grid.res, tr.y + dj * self.grid.res, 0.0) for dj in range(-n_, n_ + 1)
                             for di in range(-n_, n_ + 1) if math.hypot(di, dj) * self.grid.res <= rr]
            if foot:
                fs["footprint_hits"] = fs.get("footprint_hits", 0) + self.grid.add_hits(foot, stamp, cfg.fusion_footprint_l,
                                                                                         SRC_YOLO_FOOTPRINT)
        out = [None] * len(detections)
        for i, tid in zip(idx, ids):
            out[i] = tid
        fs["latency_s"].append(t_now - stamp)
        fs["fuse_ms"].append((time.perf_counter() - t0) * 1000)
        return out

    def idle_step(self, t: float) -> None:
        """Called on simulation steps with no mapping work: young-generation GC every idle step, a full collection every
        gc_full_every_s. Reference counting frees almost everything; only cyclic garbage waits for these collections."""
        if getattr(self, "_refine_due", False):         # track refinement against the depth blobs (~1 Hz)
            self._refine_due = False
            t0 = time.perf_counter()
            self.fusion_stats.setdefault("refine", {"runs": 0, "refined": 0, "merged": 0, "ms": []})
            r = self.tracker.refine(self.grid)
            fr = self.fusion_stats["refine"]
            fr["runs"] += 1; fr["refined"] += r["refined"]; fr["merged"] += r["merged"]
            fr["ms"].append((time.perf_counter() - t0) * 1000)
            return                                     # one job per idle step
        if not self.cfg.gc_managed:
            return
        if t >= self._gc_next_full:
            t0 = time.perf_counter()
            gc.collect()
            self.gc_stats["full"] += 1
            self.gc_stats["full_ms"].append((time.perf_counter() - t0) * 1000)
            self._gc_next_full = t + self.cfg.gc_full_every_s
        else:
            gc.collect(0)
            self.gc_stats["young"] += 1

    def _protected_cells(self) -> set:
        """Cells of confirmed small-rock tracks - also once LOST (out of view or missed by YOLO; the rock is still there):
        no FREE evidence there (plan D3: YOLO-confirmed small rocks are obstacles). YOLO still never clears space; this
        only withholds clearing where a confirmed rock is. Tentative tracks never protect anything."""
        g, r_max = self.grid, self.cfg.fusion_protect_max_radius_m
        out = set()
        for tr in self.tracker.tracks:
            if tr.status not in ("confirmed", "lost") or tr.radius_m > r_max:
                continue
            rr = tr.radius_m + 0.05
            n_ = int(math.ceil(rr / g.res))
            for dj in range(-n_, n_ + 1):
                for di in range(-n_, n_ + 1):
                    if math.hypot(di, dj) * g.res <= rr:
                        k = g.index(tr.x + di * g.res, tr.y + dj * g.res)
                        if k >= 0:
                            out.add(k)
        return out

    def reset_map(self, reason: str = "explicit") -> dict:
        """Explicit map / mission reset - the ONLY thing that clears persistent small-rock hazards. Within a mission a
        confirmed small rock stays an obstacle (also once its track is lost); a new mission, or an operator who knows the
        scene changed, starts from an empty map: grid UNKNOWN (so the safety layer sees nothing traversable), no tracks,
        no protected cells, cost layer rebuilt, and a scan captured before the reset is discarded."""
        now = self.estimator.pose()
        dropped = self.tracker.reset()
        self.grid.reset((now.x, now.y))
        self.pending = None
        self._refine_due = False
        self.costmap._last_pass = None
        self.costmap.rebuild(now.stamp_s, (self.estimator.odo.dist, self.estimator.odo.turned), (now.x, now.y))
        ev = {"event": "map_reset", "stamp_s": round(now.stamp_s, 4), "reason": reason, "tracks_dropped": dropped}
        self.stats.setdefault("map_resets", []).append(ev)
        self.tele.write(json.dumps(ev) + "\n")
        return ev

    def set_inference_status(self, status: dict) -> None:
        self.inference = status

    def due(self, t: float) -> bool:
        """A cycle runs at the first sensor sampling instant at or after ``next_t`` (so the frame is fresh: sample time == t)."""
        return t + 1e-9 >= self.next_t and abs(t - sample_time(t, self.period_s)) < 1e-6

    # ---- one mapping cycle, split over two simulation steps -------------------------------------------------------
    def cycle(self, t: float, gt: dict | None = None):
        """Step 1 of a cycle: capture one RGB + depth frame and measure it with the pose at its sample time. The grid update
        runs on the NEXT simulation step (``integrate_pending``), so no single 32 ms step carries the whole cycle.
        Returns (frame, pose) so the inference link can submit the SAME frame with the same pose (one depth read per cycle)."""
        self.integrate_pending()                       # never let two scans queue up
        t0 = time.perf_counter()
        self.next_t = t + self.cfg.cycle_s
        frame = self.sensors.capture(depth_buffer=True, with_rgb=self.want_rgb, bgra=True)
        stamp = sample_time(t, self.period_s)
        faults = self.faults.apply_depth(frame, stamp, self._sample_idx(), self.cam.width, self.cam.height) if self.faults else []
        if self.faults is not None and self.faults.overrun(stamp):
            faults.append("cycle_overrun")
        t1 = time.perf_counter()
        self.history.add(stamp, self.estimator.odo.dist, self.estimator.odo.turned)
        pose = self.buffer.at(stamp)
        self.no_pose_flag = pose is None
        if pose is None:
            self.stats["no_pose"] += 1
            return frame, None
        if frame.depth is None:
            self.stats["no_depth"] += 1
            return frame, pose
        crc = zlib.crc32(frame.depth[::97].tobytes() if hasattr(frame.depth, "tobytes") else repr(frame.depth[::97]).encode())
        prev, self._prev = self._prev, (crc, pose)
        self.frozen_flag = (prev is not None and crc == prev[0] and
                            (math.hypot(pose.x - prev[1].x, pose.y - prev[1].y) > self.cfg.frozen_motion_m
                             or abs(wrap(pose.yaw - prev[1].yaw)) > self.cfg.frozen_motion_rad))
        log_only = {"gt_pose_eval_only": gt} if gt is not None else {}    # carried to the telemetry line, never read
        if self.frozen_flag:                           # a stuck sensor's image does not match the current pose: never map it
            self.stats["frozen_frames_rejected"] += 1
            self.pending = (None, log_only, (t1 - t0) * 1000, 0.0, stamp, pose, faults)
            self.timing["step_max_ms"].append((t1 - t0) * 1000)
            return frame, pose
        obs = self.scanner.scan(frame.depth, pose, stamp, frame.index)
        t2 = time.perf_counter()
        self.pending = (obs, log_only, (t1 - t0) * 1000, (t2 - t1) * 1000, stamp, pose, faults)
        self.timing["step_max_ms"].append((t2 - t0) * 1000)
        return frame, pose

    def _sample_idx(self) -> list:
        idx = getattr(self, "_sidx", None)
        if idx is None:
            idx = self._sidx = [sm[0] for sm in self.scanner.samples]
        return idx

    def integrate_pending(self) -> None:
        """Step 2 of a cycle: grid update, telemetry and (every ``snapshot_every_s``) a snapshot handed to the writer thread."""
        if self.pending is None:
            return
        obs, log_only, read_ms, scan_ms, stamp, pose, faults = self.pending
        self.pending = None
        t2 = time.perf_counter()
        recentered, gstats = False, {}
        if obs is not None:                            # a rejected (frozen) frame changes nothing in the map
            recentered = self.grid.recenter(obs.pose.x, obs.pose.y)
            self.grid.protected = self._protected_cells() if self.cfg.fusion_protect_small_rocks else set()
            gstats = self.grid.integrate(obs)
            self.last_depth_stamp = stamp
        t3 = time.perf_counter()
        now = self.estimator.pose()
        odo = self.estimator.odo
        cstats = self.costmap.update(now.stamp_s, (odo.dist, odo.turned), (now.x, now.y))
        if self.tracker.tracks and self.stats["cycles"] % self.cfg.refine_every_cycles == 0:
            self._refine_due = True                    # done on the next idle step, off the mapping steps
        t4 = time.perf_counter()
        health = self._evaluate_health(stamp, now, obs, read_ms + scan_ms + (t4 - t2) * 1000)
        t5 = time.perf_counter()
        self.stats["cycles"] += 1
        if obs is not None:
            self.stats["points_total"] += len(obs.obstacle_points)
        if self.stats["cycles"] % 200 == 0:
            self.mem_trace.append((self.stats["cycles"], process_memory_mb()))
        trs = self.tracker.tracks
        rec = {"stamp_s": round(stamp, 4), "wall_s": round(time.time(), 3), "frame": obs.frame_index if obs else None,
               "tracks": {"tentative": sum(1 for x in trs if x.status == "tentative"), "confirmed": sum(1 for x in trs if x.status == "confirmed"),
                          "lost": sum(1 for x in trs if x.status == "lost")},
               "pose_est": (obs.pose if obs else pose).to_dict(), "pose_now": now.to_dict(), "frozen_rejected": obs is None,
               "faults_active": faults,
               "health": {"state": health.state, "reasons": list(health.reasons), "measurements": health.measurements},
               "cost": cstats, "timing_ms": {"read": round(read_ms, 3), "scan": round(scan_ms, 3), "grid": round((t3 - t2) * 1000, 3),
                                             "cost": round((t4 - t3) * 1000, 3), "health": round((t5 - t4) * 1000, 3)},
               **log_only}
        if obs is not None:
            rec.update({"valid_fraction": round(obs.valid_fraction, 4), "lower_valid_fraction": round(obs.lower_valid_fraction, 4),
                        "counts": obs.counts, "columns_hit": sum(1 for c in obs.columns if c.hit_range_m is not None),
                        "columns_with_free_space": sum(1 for c in obs.columns if c.free_chains or c.free_segments),
                        "obstacle_points": len(obs.obstacle_points),
                        "grid": {**gstats, "recentered": recentered, "origin_xy": [self.grid.origin_x, self.grid.origin_y]}})
        self.tele.write(json.dumps(rec) + "\n")
        if obs is not None:
            pts = obs.obstacle_points
            k = max(1, math.ceil(len(pts) / self.cfg.log_points_per_cycle)) if pts else 1
            self.pts_log.write(json.dumps({"stamp_s": round(obs.stamp_s, 4), "pose_est": obs.pose.to_dict(),
                                           "points": [[round(x, 4), round(y, 4), round(z, 4), kd[0]] for x, y, z, kd in pts[::k]]}) + "\n")
        if stamp + 1e-9 >= self.next_snap:
            ts0 = time.perf_counter()
            self._snapshot(stamp, now)
            self.timing["snapshot_ms"].append((time.perf_counter() - ts0) * 1000)
            self.next_snap = stamp + self.cfg.snapshot_every_s
        t6 = time.perf_counter()
        self.last_cycle_ms = read_ms + scan_ms + (t6 - t2) * 1000
        self.timing["read_ms"].append(read_ms)
        self.timing["scan_ms"].append(scan_ms)
        self.timing["grid_ms"].append((t3 - t2) * 1000)
        self.timing["cost_ms"].append((t4 - t3) * 1000)
        self.timing["health_ms"].append((t5 - t4) * 1000)
        self.timing["cycle_ms"].append(self.last_cycle_ms)
        self.timing["step_max_ms"][-1] = max(self.timing["step_max_ms"][-1], (t6 - t2) * 1000) if self.timing["step_max_ms"] else (t6 - t2) * 1000

    def _evaluate_health(self, stamp: float, now: Pose2D, obs, work_ms: float):
        """Turn this cycle's inputs into reason conditions (health.py decides states and hysteresis)."""
        cfg = self.cfg
        depth_age = (stamp - self.last_depth_stamp) if self.last_depth_stamp is not None else math.inf
        low = obs.lower_valid_fraction if obs is not None else 0.0
        inf = self.inference
        inf_age = None
        if inf.get("enabled"):
            last_ok = inf.get("last_ok_sim_s")
            inf_age = (stamp - last_ok) if last_ok is not None else math.inf
        ok, corr_reasons, corr = self.safety.corridor_clear(self.safety.corridor_ahead(now), now_t=now.stamp_s)
        fp_cost = self.safety.footprint_cost(now)
        conditions = {
            "depth_stale": depth_age > cfg.depth_stale_s,
            "depth_frozen": self.frozen_flag,
            "depth_invalid": obs is not None and low < cfg.depth_low_valid_stop,
            "depth_degraded": obs is not None and cfg.depth_low_valid_stop <= low < cfg.depth_low_valid_degraded,
            "no_pose": self.no_pose_flag,
            "obstacle_in_footprint": 253 <= fp_cost < 255,
            "obstacle_in_corridor": "obstacle_in_corridor" in corr_reasons,
            "unobserved_corridor": "unobserved_corridor" in corr_reasons,
            "stale_corridor": "stale_corridor" in corr_reasons,
            "inference_disabled": not inf.get("enabled", False),
            "inference_unavailable": bool(inf.get("enabled")) and (not inf.get("connected", False) or inf_age > cfg.inference_timeout_s),
            "inference_stale": bool(inf.get("enabled")) and (bool(inf.get("latest_stale")) or inf.get("stale_fraction", 0.0) > cfg.inference_stale_fraction),
            "cycle_overrun": max(work_ms, self.last_cycle_ms) > cfg.cycle_overrun_ms,
        }
        info = {"depth_age_s": min(depth_age, 1e6), "inference_age_s": None if inf_age is None else min(inf_age, 1e6),
                "inference_connected": bool(inf.get("connected", False)), "depth_valid_fraction": low, "pose_sigma_xy": now.sigma_xy,
                "measurements": {"depth_age_s": round(min(depth_age, 1e6), 3), "lower_valid_fraction": round(low, 3),
                                 "frozen": self.frozen_flag, "work_ms": round(work_ms, 2),
                                 "inference_age_s": None if inf_age is None else round(min(inf_age, 1e6), 3),
                                 "inference_stale_fraction": inf.get("stale_fraction"), "footprint_cost": fp_cost,
                                 "corridor": corr, "sigma_abs_m": round(now.sigma_xy, 4),
                                 "global_pose_uncertain": now.sigma_xy > cfg.global_sigma_degraded_m}}
        return self.health_monitor.evaluate(stamp, conditions, info)

    def _snapshot(self, stamp: float, pose: Pose2D, name: str | None = None, block: bool = False) -> None:
        """Copy the layers (microseconds) on the sim thread; serialise and write them on the writer thread."""
        h = self.health_monitor.latest
        payload = snapshot_payload(self.grid, stamp, pose.to_dict(), self.out / "snapshots" / (name or f"map_{round(stamp * 1000):08d}ms"),
                                   {"cycle": self.stats["cycles"], "health": h.to_dict() if h is not None else None,
                                    "tracks": [tr.to_dict() for tr in self.tracker.tracks if tr.status != "tentative"],
                                    "tentative_tracks": sum(1 for tr in self.tracker.tracks if tr.status == "tentative"),
                                    "cost_rule": {"free": 0, "graded": "1-252", "inscribed": 253, "lethal": 254, "unknown": 255,
                                                  "collision_radius_m": self.cfg.collision_radius_m, "safety_margin_m": self.cfg.safety_margin_m}},
                                   extra_layers=[("cost", "uint8", self.costmap.cost.tobytes()),
                                                 ("provenance", "uint8", self.grid.provenance_layer())])
        try:
            self._q.put(payload, block=block, timeout=10 if block else None)
            self.snapshots += 1
        except queue.Full:
            self.snap_dropped += 1

    def _write_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            try:
                write_payload(item)
            finally:
                self._q.task_done()

    def finish(self) -> dict:
        self.integrate_pending()
        from .contract import export_obstacles          # R4.6: the stable obstacle contract, once at the end of a run
        t0 = time.perf_counter()
        obstacles = export_obstacles(self)
        self.stats["obstacles_export"] = {"ms": round((time.perf_counter() - t0) * 1000, 2), "count": len(obstacles["obstacles"])}
        (self.out / "obstacles_final.json").write_text(json.dumps(obstacles, indent=1), encoding="utf-8")
        self.tele.close()
        self.pts_log.close()
        last = self.buffer.latest()
        if last is not None and self.stats["cycles"]:
            self._snapshot(last.stamp_s, last, "map_final", block=True)
        self._q.put(None)
        self._q.join()                                 # every snapshot is on disk before the summary is written
        if self.cfg.gc_managed:
            gc.unfreeze()
            if self._gc_was_enabled:
                gc.enable()
        s = dict(self.stats)
        s["snapshots"] = self.snapshots
        s["snapshots_dropped"] = self.snap_dropped
        s["grid"] = {**self.grid.counts(), "occupied_by_source": self.grid.occupied_by_source(), "recenters": self.grid.recenters,
                     "resets": self.grid.resets, "grid_bytes": self.grid.nbytes()}
        s["process_memory_mb"] = {"at_start": self.mem_start_mb, "at_end": process_memory_mb(), "trace_cycle_mb": self.mem_trace}
        s["timing_ms"] = {k: {"p50": _q(v, 0.5), "p95": _q(v, 0.95), "max": round(max(v), 3) if v else None} for k, v in self.timing.items()}
        s["health"] = {**self.health_monitor.summary(), "transitions_log": self.health_monitor.transitions,
                       "reason_events": self.health_monitor.reason_events}
        s["costmap"] = self.costmap.stats
        fm = self.gc_stats["full_ms"]
        s["gc"] = {"managed": self.cfg.gc_managed, "young": self.gc_stats["young"], "full": self.gc_stats["full"],
                   "full_ms": {"p50": _q(fm, 0.5), "max": round(max(fm), 2) if fm else None}}
        fs = self.fusion_stats
        if self.tracker.tracks:
            self.tracker.refine(self.grid)                     # final snap before the summary
        if "refine" in fs:
            rm = fs["refine"].pop("ms", [])
            fs["refine"]["ms"] = {"p50": _q(rm, 0.5), "p95": _q(rm, 0.95), "max": round(max(rm), 2) if rm else None}
        s["fusion"] = {**{k: v for k, v in fs.items() if k not in ("latency_s", "fuse_ms")},
                       "latency_s": {"p50": _q(fs["latency_s"], 0.5), "p95": _q(fs["latency_s"], 0.95), "max": round(max(fs["latency_s"]), 3) if fs["latency_s"] else None},
                       "fuse_ms": {"p50": _q(fs["fuse_ms"], 0.5), "p95": _q(fs["fuse_ms"], 0.95), "max": round(max(fs["fuse_ms"]), 3) if fs["fuse_ms"] else None},
                       "tracker": self.tracker.stats, "tracks_final": [tr.to_dict() for tr in self.tracker.tracks]}
        s["faults"] = {"spec": self.faults.spec, "applied_cycles": self.faults.counts} if self.faults is not None else None
        s["pose_final"] = self.estimator.pose().to_dict()
        s["config_hash"] = self.cfg.hash()
        return s
