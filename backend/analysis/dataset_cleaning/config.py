"""S2-H policy configuration. Values marked (calibrated) are overwritten by calibration.py evidence at run time."""

from __future__ import annotations

POLICY_VERSION = "S2-H/1.1"   # 1.1: rule D05 also applied to watershed split pieces
DATASET_NAME = "curated-dataset-v1.1"
PREVIOUS_DATASET = "curated-dataset-v1"   # archived, read-only
EXP_NAME = "dataset-cleaning-v1.1"         # v1 outputs stay in experiments/dataset-cleaning
REVIEW_SAMPLE_FROM = "dataset-cleaning"     # reuse the v1 review sample + verdicts (matched by ann_id, never by review_id)
SEED = 20261010                 # Task C review sample (Task B used 20261009 -> independent sample)
CLASS_NAMES = {0: "rock"}
IMG_W, IMG_H = 720, 480

# candidate extraction (Task B evidence: V1_cyan, no closing, >= 50 px)
VARIANT = "V1_cyan"
MIN_AREA = 50
MIN_SIDE = 4                    # thinner boxes are degenerate artefacts
BLUE_MAX_AREA_UNSPLIT = 1000    # blue comps >= this were merged groups in 9/10 Task B reviews

# duplicates / overlap with CSV instances
BLUE_COVERED_BY_CSV = 0.5       # blue comp >= 50% inside a CSV box -> duplicate of the CSV instance
BLUE_IOU_WITH_CSV = 0.3
GREEN_SAME_AS_CSV_IOU = 0.5     # green comp with IoU >= 0.5 to a CSV box -> duplicate
GREEN_INSIDE_CSV = 0.9          # green comp >= 90% inside a CSV box and >= 30% of its area -> review
GREEN_INSIDE_CSV_AREA = 0.3

# darkness / visibility (render evidence)
DARK_LEVEL = 15.0               # rock and surround mean below this -> invisible in black shadow
CSV_DARK_FRACTION = 0.85        # CSV box with >= 85% pixels < DARK_LEVEL ...
CSV_SILHOUETTE_RING = 30.0      # ... AND surroundings (non-sky) mean < 30 -> review; otherwise a visible dark silhouette (Task C review: 6/8 flagged boxes were visible silhouettes under the box-only rule)
VIS_EDGE_GRID = tuple(range(40, 181, 10))        # boundary Sobel magnitude; Task B medians: visible 135, ambiguous 70 (calibrated)
VIS_CONTRAST_GRID = tuple(range(10, 61, 5))     # |mean in - mean ring|; Task B medians: visible 27, ambiguous 11 (calibrated)
VIS_MIN_RECALL = 0.85           # calibration keeps >= 85% of visible rocks seen in Task B review
VIS_EDGE_REL_GRID = tuple(round(0.5 * k, 2) for k in range(2, 17))     # edge / scene median ground gradient (calibrated)
VIS_CONTRAST_REL_GRID = tuple(round(0.25 * k, 2) for k in range(1, 13))  # contrast / scene ground std (calibrated)

# instance separation (distance transform + watershed)
PEAK_REL = 0.35                 # marker = local max of distance >= PEAK_REL * max distance
PEAK_ABS = 2.5                  # ... and >= 2.5 px from the boundary
NECK_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)            # accepted split if neck / min(peak) <= value (calibrated)
NECK_REVIEW_MARGIN = 0.15       # neck ratio within (accept, accept + margin] -> REVIEW_REQUIRED
MIN_PIECE_AREA = 25
SOLIDITY_REVIEW = 0.80          # unsplit component with solidity below -> possible merge -> review
MAX_OVERSPLIT_RATE = 0.05       # calibration constraint on known single CSV rocks

STATUS = ("VALID", "REVIEW_REQUIRED", "REJECTED")
