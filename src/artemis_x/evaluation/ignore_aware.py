"""Ignore-aware detection evaluation (Task D.2), re-exported from the proven implementation.

The implementation stays in ``backend/analysis/evaluation/ignore_eval.py`` (covered by 16 regression tests); this module
only makes it importable as ``artemis_x.evaluation.ignore_aware``. Policy used for every published number:
IoU 0.5 matching (greedy by score); a prediction overlapping a REVIEW region (IoU or IoA >= 0.5) is ignored
(neither TP nor FP); 101-point AP; AP50-95 over IoU 0.50:0.95:0.05; predictions < 50 px^2 removed.
"""

from __future__ import annotations

from ..config.paths import ensure_backend_importable

ensure_backend_importable()

from analysis.evaluation.ignore_eval import (  # noqa: E402
    IOU_THRS,
    SIZE_BINS,
    ImageEval,
    ap_101,
    evaluate,
    ioa,
    iou,
    load_split,
    match_image,
    yolo_to_xyxy,
)

PRIMARY_POLICY = {"match_iou": 0.5, "ignore_ioa": 0.5, "min_pred_area_px2": 50.0, "ignore_files": ["eval_ignore_regions.json"],
                  "ap": "101-point", "ap50_95_thresholds": [float(t) for t in IOU_THRS]}

__all__ = ["IOU_THRS", "SIZE_BINS", "ImageEval", "ap_101", "evaluate", "ioa", "iou", "load_split", "match_image", "yolo_to_xyxy",
           "PRIMARY_POLICY"]
