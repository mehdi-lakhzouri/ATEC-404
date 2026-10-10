"""Wheel commands, limits and the scripted R2 motion sequence for the Koala rover.

Koala geometry (Webots R2025a ``Koala.proto``): six wheels, only the two middle ones are motorised by
``RotationalMotor`` devices "left wheel motor" / "right wheel motor" (hinge axis +y, wheel radius 0.0425 m,
wheel centres at y = +/-0.1425 m -> track 0.285 m). Forward is +x of the robot frame. Commands are wheel angular
velocities in rad/s, applied in velocity-control mode (``setPosition(inf)`` + ``setVelocity``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WHEEL_RADIUS_M = 0.0425
TRACK_M = 0.285
LEFT_MOTOR, RIGHT_MOTOR = "left wheel motor", "right wheel motor"
LEFT_ENCODER, RIGHT_ENCODER = "left wheel sensor", "right wheel sensor"


@dataclass(frozen=True)
class WheelCommand:
    left: float      # rad/s
    right: float     # rad/s

    def clamped(self, limit: float) -> "WheelCommand":
        """Scale both wheels by the same factor so neither exceeds ``limit`` (keeps the curvature)."""
        m = max(abs(self.left), abs(self.right))
        if m <= limit or m == 0:
            return self
        k = limit / m
        return WheelCommand(self.left * k, self.right * k)


STOP = WheelCommand(0.0, 0.0)


def forward(speed: float) -> WheelCommand:
    return WheelCommand(abs(speed), abs(speed))


def backward(speed: float) -> WheelCommand:
    return WheelCommand(-abs(speed), -abs(speed))


def turn_left(speed: float) -> WheelCommand:       # counter-clockwise seen from above (+yaw)
    return WheelCommand(-abs(speed), abs(speed))


def turn_right(speed: float) -> WheelCommand:
    return WheelCommand(abs(speed), -abs(speed))


def twist_to_wheels(v: float, w: float) -> WheelCommand:
    """Linear v (m/s) and yaw rate w (rad/s) -> ideal differential-drive wheel speeds (rad/s)."""
    return WheelCommand((v - w * TRACK_M / 2) / WHEEL_RADIUS_M, (v + w * TRACK_M / 2) / WHEEL_RADIUS_M)


def wheels_to_twist(cmd: WheelCommand) -> tuple[float, float]:
    v = WHEEL_RADIUS_M * (cmd.left + cmd.right) / 2
    w = WHEEL_RADIUS_M * (cmd.right - cmd.left) / TRACK_M
    return v, w


@dataclass(frozen=True)
class Step:
    name: str
    command: WheelCommand
    duration_s: float
    capture: bool = False          # capture a camera/depth frame at the end of this step


def demo_sequence(drive_speed: float = 4.0, turn_speed: float = 2.0, forward_s: float = 3.0,
                  turn_angle_rad: float = math.pi / 2, pause_s: float = 1.0) -> list[Step]:
    """R2 demo: stationary -> forward -> stop -> turn left -> forward -> stop.

    The turn duration is computed from ideal differential-drive kinematics; the six-wheel Koala is not an ideal
    differential drive, so the achieved yaw is measured (ground truth) rather than assumed."""
    _, w = wheels_to_twist(turn_left(turn_speed))
    return [Step("start_stationary", STOP, pause_s, capture=True),
            Step("forward_1", forward(drive_speed), forward_s),
            Step("stop_1", STOP, pause_s, capture=True),
            Step("turn_left", turn_left(turn_speed), turn_angle_rad / w),
            Step("forward_2", forward(drive_speed), forward_s),
            Step("stop_2", STOP, pause_s, capture=True)]


class SequencePlayer:
    """Maps simulation time to the active step (pure, deterministic)."""

    def __init__(self, steps: list[Step]):
        self.steps = steps
        self.bounds = []
        t = 0.0
        for s in steps:
            self.bounds.append((t, t + s.duration_s))
            t += s.duration_s
        self.total_s = t

    def index_at(self, t: float) -> int | None:
        for i, (a, b) in enumerate(self.bounds):
            if a <= t < b:
                return i
        return None


# ---- encoder odometry and closed-loop scripted moves (R3 course tour) -----------------------------------------------------
# Calibration from the R2 runs (supervisor ground truth vs wheel encoders, deterministic): straight-line distance from the
# encoders is exact; the six-wheel Koala turns 4.2 % less than ideal differential-drive kinematics predict.
YAW_SCALE = 0.958


class Odometry:
    """Dead reckoning from the two wheel encoders only (no ground truth)."""

    def __init__(self):
        self.x = self.y = self.yaw = 0.0
        self.dist = 0.0
        self._last = None

    def update(self, enc_left: float, enc_right: float) -> None:
        if self._last is None:
            self._last = (enc_left, enc_right)
            return
        dl = (enc_left - self._last[0]) * WHEEL_RADIUS_M
        dr = (enc_right - self._last[1]) * WHEEL_RADIUS_M
        self._last = (enc_left, enc_right)
        d, dth = (dl + dr) / 2, YAW_SCALE * (dr - dl) / TRACK_M
        self.x += d * math.cos(self.yaw + dth / 2)
        self.y += d * math.sin(self.yaw + dth / 2)
        self.yaw += dth
        self.dist += abs(d)


@dataclass(frozen=True)
class Move:
    kind: str                 # "drive" (metres, signed), "drive_to" (absolute odometry x, m), "turn" (rad, + = left), "pause" (s)
    value: float
    label: str = ""
    capture: bool = False


def move_command(move: Move, progress: float, drive_speed: float = 4.0, turn_speed: float = 2.0) -> WheelCommand:
    """Wheel command for a move given the remaining amount (same units as ``value``); slows down near the target."""
    if move.kind in ("drive", "drive_to"):
        s = drive_speed * min(1.0, max(0.15, abs(progress) / 0.15))
        return forward(s) if progress > 0 else backward(s)
    if move.kind == "turn":
        s = turn_speed * min(1.0, max(0.15, abs(progress) / 0.25))
        return turn_left(s) if progress > 0 else turn_right(s)
    return STOP


def tour_moves(stops: list[dict]) -> list[Move]:
    """Course tour along a clear lane: drive to each stop, turn to face the zone, approach it along a checked-clear straight
    line (``approach_m``), pause (capture + detections), reverse the same distance, turn back onto the lane.
    Stops are absolute odometry x positions (the lane start is x = 0), so stopping errors do not accumulate."""
    moves = []
    for s in stops:
        a = s.get("approach_m", 0.0)
        moves.append(Move("drive_to", s["x"], f"to_{s['zone']}"))
        moves.append(Move("turn", s["look_rad"], f"look_{s['zone']}"))
        if a > 0:
            moves.append(Move("drive", a, f"approach_{s['zone']}"))
        moves.append(Move("pause", 2.0, f"observe_{s['zone']}", capture=True))
        if a > 0:
            moves.append(Move("drive", -a, f"retreat_{s['zone']}"))
        moves.append(Move("turn", -s["look_rad"], f"back_{s['zone']}"))
    moves.append(Move("pause", 1.0, "end"))
    return moves
