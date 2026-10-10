"""Every mapping threshold and size in one place; the active configuration is logged with each run (and hashed)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MappingConfig:
    # cadence
    cycle_s: float = 0.256                 # one mapping cycle every 0.256 s of simulation time (same cadence as InferenceLink)
    pose_buffer_s: float = 2.0             # history kept for "pose at capture time" lookups
    # depth sampling (pixels of the 720 x 480 depth image)
    col_stride: int = 8                    # 90 columns
    row_stride: int = 6
    row_start_frac: float = 0.30           # sample the lower 70 % of the image (the upper part is mostly sky / far terrain)
    max_range_m: float = 6.0               # depth beyond this is not used for mapping
    # ground / obstacle separation: height z_b above the rover's ground plane (exact: kinematic rover, never tilts)
    ground_band_m: float = 0.03            # |z_b| <= band  -> ground
    obstacle_height_m: float = 0.05        # z_b >= h_obs   -> positive obstacle (plan decision D3)
    negative_depth_m: float = 0.08         # z_b <= -depth  -> negative hazard (crater bowl)
    # free-space evidence (R4.2): join consecutive ground samples only across flat-ground-sized gaps
    free_gap_factor: float = 1.5
    free_gap_min_m: float = 0.25
    free_gap_max_m: float = 0.60
    overhang_max_height_m: float = 0.40    # obstacles up to this height above a ground sample make it "under an obstacle"
    overhang_margin_m: float = 0.02
    # occupancy grid (R4.2, plan section 6.2, decision D2)
    resolution_m: float = 0.10
    grid_cells: int = 120                  # 12 x 12 m rolling window
    recenter_m: float = 1.0                # shift the window (whole cells) once the rover is this far from its centre
    l_hit: float = 0.85
    l_miss: float = -0.40
    l_min: float = -2.0
    l_max: float = 3.5
    l_occupied: float = 0.85               # l > this -> OCCUPIED
    l_free: float = -0.85                  # l < this -> FREE ; otherwise UNKNOWN
    snapshot_every_s: float = 2.0
    # odometry uncertainty (R4.3, uncertainty.py). 1-sigma DESIGN ENVELOPE, not a measurement: 2 sigma covers the +-3 %
    # distance-scale and +-3 % turn-scale errors used in robustness tests. The deterministic kinematic simulator itself
    # shows ~0 odometry error (measured, R4.1/R4.2), so these values are deliberately conservative for real regolith.
    odo_k_scale: float = 0.02              # distance error per metre driven
    odo_k_yaw_turn: float = 0.02           # heading error per radian turned
    odo_k_yaw_dist: float = 0.002          # heading error (rad) per metre driven
    # rover footprint (R4.3, measured from Koala.proto R2025a: boundingObject + six wheels)
    footprint_radius_measured_m: float = 0.210   # circumscribed radius about the rover origin (wheel corners)
    collision_radius_m: float = 0.25       # measured + 0.04 m allowance (plan decision D3): INSCRIBED cost inside this
    safety_margin_m: float = 0.15          # graded cost band beyond the collision radius
    uncertainty_margin_cap_m: float = 0.20 # 2 sigma_rel added to obstacles, capped (plan section 6.6)
    uncertainty_margin_step_m: float = 0.05  # kernels are precomputed per 5 cm of uncertainty margin
    cost_decay_per_m: float = 25.0         # graded cost = 252 * exp(-k (d - inflated radius))
    cost_tile_cells: int = 10              # dirty-region granularity of the incremental cost update
    cost_level_headroom_m: float = 0.05    # inflation levels: re-evaluated only after the rover moved this far (or turned
    cost_level_headroom_rad: float = 0.05  # this much), and always computed WITH this headroom -> never under-inflated
    # health monitor (R4.3, health.py); all times in simulation seconds unless noted
    corridor_length_m: float = 1.0         # prospective straight travel corridor ahead of the estimated pose
    corridor_step_m: float = 0.05
    corridor_max_age_s: float = 10.0       # FREE evidence older than this is "stale"
    corridor_max_sigma_m: float = 0.10     # FREE evidence whose relative position uncertainty exceeds this is "stale"
    depth_stale_s: float = 0.5             # no accepted depth frame for this long -> STOP
    depth_low_valid_degraded: float = 0.80 # valid fraction of the LOWER half of the image (ground region)
    depth_low_valid_stop: float = 0.40
    frozen_motion_m: float = 0.01          # a byte-identical depth frame while the rover moved this much -> frozen
    frozen_motion_rad: float = 0.009
    inference_timeout_s: float = 1.0       # no successful YOLO result for this long -> inference unavailable
    inference_stale_window_s: float = 5.0
    inference_stale_fraction: float = 0.5
    cycle_overrun_ms: float = 50.0         # wall-clock mapping work per cycle
    global_sigma_degraded_m: float = 1.0   # absolute position uncertainty flag (measurement only, not a health reason)
    clear_cycles: int = 2                  # hysteresis: a reason clears only after this many consecutive clean cycles
    # YOLO + depth fusion (R4.4, fusion.py)
    fusion_enabled: bool = True
    fusion_max_age_s: float = 1.0          # results older than this (capture -> arrival, sim time) are not fused
    fusion_max_samples: int = 400          # depth samples per box
    fusion_min_valid_fraction: float = 0.2
    fusion_bimodal_gap_m: float = 0.30     # depth gap inside a box that indicates an occluder (two surfaces)
    fusion_min_radius_m: float = 0.03
    fusion_sigma0_m: float = 0.03          # measurement sigma = sigma0 + per_m * range (x2 when bimodal)
    fusion_sigma_per_m: float = 0.02
    fusion_low_obstacle_m: float = 0.02    # small-rock rule: obstacle threshold inside a YOLO box (vs obstacle_height_m)
    fusion_protect_small_rocks: bool = True   # plan D3: a CONFIRMED small rock is an obstacle - its cells receive no FREE
    fusion_protect_max_radius_m: float = 0.15 # evidence (depth's 3 cm ground band cannot resolve it); only hits accumulate
    fusion_footprint_l: float = 0.4        # plan 6.4: + this log-odds to a confirmed small rock's footprint per association
    # obstacle tracker (R4.4, tracker.py)
    track_gate0_m: float = 0.25
    track_gate_per_m: float = 0.05
    track_ema: float = 0.3
    confirm_hits: int = 2
    confirm_window_s: float = 2.0
    tentative_timeout_s: float = 3.0
    lost_misses: int = 4
    track_miss_range_m: float = 5.0
    track_fov_half_rad: float = 0.45
    merge_overlap: float = 0.5
    track_min_support: int = 3             # a hit counts toward confirmation only with >= this many depth points >= 2 cm high
    refine_every_cycles: int = 4           # snap confirmed tracks to their depth blob (~1 s)
    refine_search_m: float = 0.30          # nearest occupied cell to a track centre (+ the track radius: a boulder's centre
                                           # lies behind its visible face)
    refine_fit_min_extent_m: float = 0.50  # circle-fit the blob (centre, radius) only for blobs at least this wide
    refine_merge_min_extent_m: float = 0.80   # tracks on one blob this wide (a boulder) are fragments of one object
    refine_max_extent_m: float = 3.0       # wider blobs are terrain (ridges), never fitted or merged; craters: NEGATIVE guard
    # garbage collection (R4 performance review): automatic GC off; young generations on idle simulation steps, a full
    # collection every gc_full_every_s on an idle step - so collection pauses never land inside a mapping step
    gc_managed: bool = True
    gc_full_every_s: float = 2.0
    # telemetry
    log_points_per_cycle: int = 150        # obstacle points written per cycle for offline evaluation (subsampled)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]
