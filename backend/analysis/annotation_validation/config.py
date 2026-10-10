"""Configuration for S2 candidate extraction, sampling quotas and drawing styles."""

from __future__ import annotations

from dataclasses import dataclass

SEED = 20261009
IMG_W, IMG_H = 720, 480
MATCH_IOU = 0.5
CLASS_TOL = 127  # channel > 127 counts as "on"; anti-aliased edge pixels (21..127) are background


@dataclass(frozen=True)
class ExtractVariant:
    """How mask pixels are turned into connected components."""

    name: str
    cyan_is_blue: bool  # cyan = blue/green blend at large-rock edges -> large rock
    close_px: int       # square closing kernel size; 0 = none (bridges 1-px anti-aliasing gaps)
    description: str


VARIANTS = (
    ExtractVariant("V0_strict", False, 0, "pure blue / pure green pixels, no morphology (Task A rule)"),
    ExtractVariant("V1_cyan", True, 0, "cyan edge pixels assigned to large rock, no morphology"),
    ExtractVariant("V2_cyan_close3", True, 3, "V1 + 3x3 closing per class before labelling"),
)
PRIMARY_VARIANT = "V1_cyan"  # chosen on full-dataset evidence: closing merges rocks (see report)
AREA_THRESHOLDS = (25, 50, 100, 200)  # native 720x480 pixels
STORE_MIN_AREA = 10                  # components below this are counted but not stored
PROPOSED_MIN_AREA = 50               # S2 proposal used in overlays; justified in the report from review data
LOW_VIS_CONTRAST = 6.0               # |mean(rock) - mean(surround)| in grey levels
DARK_LEVEL = 15.0                    # rock and surround both below this -> inside black shadow

# Drawing styles: RGB colour + short text tag (colour is never the only cue).
STYLE = {
    "csv": ((0, 255, 255), "C"),              # S1 CSV large-rock box
    "csv_unmatched": ((255, 140, 0), "C?"),   # S1 box without an S2 large-rock match (suspected)
    "s2_large": ((60, 110, 255), "L"),        # S2 large rock matching a CSV box
    "s2_large_new": ((190, 90, 255), "L+"),   # S2 large rock without CSV box
    "s2_small": ((0, 230, 0), "S"),           # S2 additional small/medium rock
    "s2_rejected": ((255, 40, 40), "X"),      # below threshold / dark / low visibility
}

# Sampling quotas per stratum (distinct frames; a frame keeps every reason it qualifies for).
QUOTAS = {
    "random_non_anomalous": 14,
    "zero_csv_boxes": 8,
    "one_large_rock": 8,
    "multiple_large_rocks": 8,
    "high_density_small_rocks": 8,
    "small_near_large": 6,
    "touching_or_overlapping": 6,
    "very_dark": 6,
    "heavy_shadow_non_listed": 6,
    "border_objects": 6,
    "suspected_fragmentation": 6,
    "uncertain_csv_match": 8,
    "list_cam_anomaly": 5,
    "list_mismatch": 6,
    "list_shadow": 6,
    "list_ground_facing": 4,
    "exact_duplicate_pair": 6,
}
ANOMALY_FILES = {
    "cam_anomaly": "cam_anomaly_IDs.txt",
    "mismatch": "mismatch_IDs.txt",
    "shadow": "shadow_IDs.txt",
    "ground_facing": "ground_facing_IDs.txt",
}

# NASA review
FIXED_LINEAR_SHIFT = 4          # 12-bit -> 8-bit: value >> 4 (i.e. /16), clipped to 255
ENH_PERCENTILES = (0.5, 99.5)   # contrast-enhanced view: per-image percentile stretch ...
ENH_GAMMA = 1 / 2.2             # ... followed by gamma
BOX_VISIBLE_RANGE = 12          # p95-p5 grey levels inside a box needed to call it "visible"
NOISE_K = 6                     # raw contrast must exceed NOISE_K * sigma_noise for "enhanced-visible"
REVIEW_STATES = ("ACCEPT", "REJECT", "NEEDS_CORRECTION", "UNCERTAIN", "NOT_REVIEWED")
