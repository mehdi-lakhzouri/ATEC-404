"""Artemis-X rover runtime for Webots (Koala): motors, scripted demo, sensors, ground-truth pose and diagnostics.

Run from the Webots controller ``artemis-webots/controllers/artemis_rover/artemis_rover.py``. Environment options:
    ARTEMIS_WEBOTS_OUT   output folder for diagnostics (default: artemis-webots/output/<timestamp>)
    ARTEMIS_WEBOTS_QUIT  "1" -> quit the simulation when the demo ends (automated runs); otherwise idle with motors stopped
    ARTEMIS_SENSOR_MS    camera/depth sampling period in ms (default 64, rounded up to a multiple of basicTimeStep)
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

from . import drive
from .sensors import SensorSuite


class KoalaRover:
    def __init__(self, robot, max_wheel_speed: float | None = None):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.left = robot.getDevice(drive.LEFT_MOTOR)
        self.right = robot.getDevice(drive.RIGHT_MOTOR)
        if self.left is None or self.right is None:
            raise RuntimeError(f"Koala motors not found ({drive.LEFT_MOTOR!r}, {drive.RIGHT_MOTOR!r})")
        hw = min(self.left.getMaxVelocity(), self.right.getMaxVelocity())
        self.max_wheel_speed = min(hw, max_wheel_speed) if max_wheel_speed else hw
        for m in (self.left, self.right):
            m.setPosition(float("inf"))         # velocity-control mode
            m.setVelocity(0.0)
        self.encoders = [robot.getDevice(n) for n in (drive.LEFT_ENCODER, drive.RIGHT_ENCODER)]
        for e in self.encoders:
            if e is not None:
                e.enable(self.timestep)
        self.self_node = robot.getSelf() if hasattr(robot, "getSelf") else None   # Supervisor only (ground truth)
        self.last_command = drive.STOP

    def apply(self, cmd: drive.WheelCommand) -> drive.WheelCommand:
        c = cmd.clamped(self.max_wheel_speed)
        self.left.setVelocity(c.left)
        self.right.setVelocity(c.right)
        self.last_command = c
        return c

    def stop(self) -> None:
        self.apply(drive.STOP)

    def pose(self) -> dict | None:
        """Ground-truth pose from the supervisor (simulation only): x, y, z in metres, yaw in rad."""
        if self.self_node is None:
            return None
        x, y, z = self.self_node.getPosition()
        r = self.self_node.getOrientation()          # 3x3 row-major
        return {"x": x, "y": y, "z": z, "yaw": math.atan2(r[3], r[0])}

    def encoder_rad(self) -> list:
        return [e.getValue() if e is not None else None for e in self.encoders]


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def run_controller(robot, out_dir: Path, quit_at_end: bool, sensor_ms: int = 64) -> dict:
    rover = KoalaRover(robot)
    ts = rover.timestep
    period = max(ts, int(math.ceil(sensor_ms / ts)) * ts)
    sensors = SensorSuite(robot, period)
    steps = drive.demo_sequence()
    player = drive.SequencePlayer(steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    tele = (out_dir / "telemetry.csv").open("w", newline="", encoding="utf-8")
    w = csv.writer(tele)
    w.writerow(["sim_time_s", "step", "cmd_left_rad_s", "cmd_right_rad_s", "enc_left_rad", "enc_right_rad", "x", "y", "z", "yaw"])
    summary = {"controller": "artemis_rover", "webots_time_step_ms": ts, "sensor": sensors.describe(), "max_wheel_speed_rad_s": rover.max_wheel_speed,
               "sequence": [{"name": s.name, "left": s.command.left, "right": s.command.right, "duration_s": round(s.duration_s, 3)} for s in steps],
               "frames": [], "phases": {}}
    phase_start, current, frames_done = {}, None, set()
    wall0 = time.time()
    try:
        # first step: let enabled devices produce their first sample
        while robot.step(ts) != -1:
            t = robot.getTime()
            i = player.index_at(t)
            if i is None:
                break
            step = steps[i]
            if i != current:
                if current is not None:
                    summary["phases"][steps[current].name]["end_pose"] = rover.pose()
                summary["phases"][step.name] = {"start_time_s": round(t, 3), "start_pose": rover.pose()}
                current = i
            applied = rover.apply(step.command)
            p = rover.pose() or {}
            ph = summary["phases"][step.name]
            if "mid_pose" not in ph and t >= (player.bounds[i][0] + player.bounds[i][1]) / 2:
                ph["mid_pose"] = rover.pose()
            enc = rover.encoder_rad()
            w.writerow([round(t, 4), step.name, round(applied.left, 4), round(applied.right, 4), *[None if v is None else round(v, 4) for v in enc],
                        *[round(p[k], 5) if k in p else None for k in ("x", "y", "z", "yaw")]])
            end_t = player.bounds[i][1]
            if step.capture and i not in frames_done and t + 2 * ts / 1000 >= end_t and t >= player.bounds[i][0] + period / 1000:
                f = sensors.capture()
                meta = sensors.export(f, out_dir / "frames", step.name)
                meta["rover_pose"] = p
                summary["frames"].append(meta)
                frames_done.add(i)
        if current is not None:
            summary["phases"][steps[current].name]["end_pose"] = rover.pose()
    finally:
        rover.stop()
        tele.close()
    summary["wall_time_s"] = round(time.time() - wall0, 2)
    summary["sim_time_s"] = round(robot.getTime(), 3)
    summary["checks"] = evaluate(summary, sensors)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"[artemis_rover] demo finished: {sum(c['ok'] for c in summary['checks'].values())}/{len(summary['checks'])} checks passed -> {out_dir}")
    for k, c in summary["checks"].items():
        print(f"[artemis_rover]   {'PASS' if c['ok'] else 'FAIL'} {k}: {c['detail']}")
    if quit_at_end and hasattr(robot, "simulationQuit"):
        robot.simulationQuit(0 if all(c["ok"] for c in summary["checks"].values()) else 1)
    else:
        while robot.step(ts) != -1:              # idle, motors stopped, until the user stops the simulation
            pass
    return summary


def evaluate(s: dict, sensors: SensorSuite) -> dict:
    ph = s["phases"]
    chk = {}

    def dist(a, b):
        return math.hypot(b["x"] - a["x"], b["y"] - a["y"]) if a and b else None

    d1 = dist(ph.get("forward_1", {}).get("start_pose"), ph.get("forward_1", {}).get("end_pose"))
    d2 = dist(ph.get("forward_2", {}).get("start_pose"), ph.get("forward_2", {}).get("end_pose"))
    exp = 3.0 * 4.0 * drive.WHEEL_RADIUS_M          # ideal 0.51 m per forward phase
    chk["rover_moved_forward"] = {"ok": d1 is not None and d1 > 0.5 * exp and d2 is not None and d2 > 0.5 * exp,
                                  "detail": f"forward_1 {d1 if d1 is None else round(d1, 3)} m, forward_2 {d2 if d2 is None else round(d2, 3)} m (ideal {exp:.3f} m each)"}
    # motors ramp down (~0.2 s), so the rover coasts briefly after STOP; it must be stationary in the second half of each stop
    det, ok = [], True
    for name in ("stop_1", "stop_2"):
        st = ph.get(name, {})
        coast, still = dist(st.get("start_pose"), st.get("mid_pose")), dist(st.get("mid_pose"), st.get("end_pose"))
        ok &= still is not None and still < 0.002
        det.append(f"{name}: coast {None if coast is None else round(coast, 3)} m then {None if still is None else round(still, 4)} m")
    chk["rover_stops"] = {"ok": ok, "detail": "; ".join(det)}
    tl = ph.get("turn_left", {})
    dyaw = _wrap(tl["end_pose"]["yaw"] - tl["start_pose"]["yaw"]) if tl.get("start_pose") and tl.get("end_pose") else None
    chk["rover_turned_left"] = {"ok": dyaw is not None and dyaw > math.radians(30),
                                "detail": f"yaw change {None if dyaw is None else round(math.degrees(dyaw), 1)} deg (ideal kinematic target 90 deg)"}
    cam = s["sensor"].get("camera")
    fr = s["frames"]
    chk["camera_dimensions"] = {"ok": bool(cam) and len(fr) >= 3 and all(f["width"] == cam["width"] and f["height"] == cam["height"] and "rgb_raw" in f for f in fr),
                                "detail": f"{len(fr)} frames, expected {cam['width']}x{cam['height']}" if cam else "no camera"}
    dep = [f.get("depth_stats") for f in fr]
    chk["depth_finite_where_valid"] = {"ok": bool(dep) and all(d and d["valid"] > 0 and d["min_m"] >= sensors.depth.getMinRange() for d in dep),
                                       "detail": [None if not d else f"valid {d['valid_fraction']:.2f}, median {d['median_m']}" for d in dep]}
    chk["devices_available"] = {"ok": not s["sensor"]["errors"], "detail": s["sensor"]["errors"] or "camera and range finder found"}
    return chk


def main() -> None:   # pragma: no cover - requires Webots
    from controller import Supervisor          # Webots Python API (PYTHONPATH set by Webots)
    robot = Supervisor()
    repo = Path(__file__).resolve().parents[3]
    out = Path(os.environ.get("ARTEMIS_WEBOTS_OUT") or repo / "artemis-webots" / "output" / time.strftime("%Y%m%d-%H%M%S"))
    run_controller(robot, out, quit_at_end=os.environ.get("ARTEMIS_WEBOTS_QUIT") == "1",
                   sensor_ms=int(os.environ.get("ARTEMIS_SENSOR_MS", "64")))
