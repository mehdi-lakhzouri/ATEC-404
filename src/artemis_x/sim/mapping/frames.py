"""Coordinate frames (R4 plan, section 6.1) and SE(2) helpers.

    camera  FLU at the sensor, x along the optical axis           (Webots convention)
    base    rover centre on the ground, x forward, z up            camera -> base: projection.camera_to_rover
    odom    the rover's start pose is (0, 0, 0); drifts with odometry
    map     identical to odom in R4 (no correction exists yet)
    world   Webots global frame - evaluation tools only, never here

Function names say which transform they apply (``base_to_odom``); there is no global frame state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    """Rover pose estimate in ``odom`` at simulation time ``stamp_s`` with 1-sigma uncertainty."""
    x: float
    y: float
    yaw: float
    stamp_s: float
    sigma_xy: float = 0.0
    sigma_yaw: float = 0.0
    source: str = "wheel_odometry"

    def to_dict(self) -> dict:
        return {"frame": "odom", "x": round(self.x, 5), "y": round(self.y, 5), "yaw": round(self.yaw, 6), "stamp_s": round(self.stamp_s, 4),
                "sigma_xy": round(self.sigma_xy, 5), "sigma_yaw": round(self.sigma_yaw, 6), "source": self.source}


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """a (+) b: pose b expressed in a's frame, mapped to a's parent frame."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap(a.yaw + b.yaw), b.stamp_s, a.sigma_xy, a.sigma_yaw, a.source)


def inverse(a: Pose2D) -> Pose2D:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(-c * a.x - s * a.y, s * a.x - c * a.y, wrap(-a.yaw), a.stamp_s, a.sigma_xy, a.sigma_yaw, a.source)


def base_to_odom(p: tuple, pose: Pose2D) -> tuple:
    """Point (x, y[, z]) in the rover base frame -> odom frame (z unchanged: the rover never tilts in this simulator)."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    out = (pose.x + c * p[0] - s * p[1], pose.y + s * p[0] + c * p[1])
    return out + tuple(p[2:])


def odom_to_base(p: tuple, pose: Pose2D) -> tuple:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx, dy = p[0] - pose.x, p[1] - pose.y
    return (c * dx + s * dy, -s * dx + c * dy) + tuple(p[2:])
