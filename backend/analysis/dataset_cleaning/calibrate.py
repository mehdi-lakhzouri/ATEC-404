"""Evidence-based threshold selection.

1. Splitter: blue (large-rock) mask components have instance ground truth in the CSV. Components whose bbox
   contains >= 2 CSV box centres are known merges; components matching exactly one CSV box (IoU >= 0.7) are
   known single rocks. For each neck threshold we measure instance recovery on merges and over-splitting on singles.
2. Visibility: Task B's assistant-reviewed S2 candidates (VISIBLE_ROCK vs AMBIGUOUS/NOT_VISIBLE) calibrate the
   render-evidence rule. This is in-sample; the independent Task C review sample evaluates it out of sample.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset_exploration.common import greedy_match, iou_xyxy
from .config import (MAX_OVERSPLIT_RATE, NECK_GRID, VIS_CONTRAST_GRID, VIS_CONTRAST_REL_GRID, VIS_EDGE_GRID,
                     VIS_EDGE_REL_GRID, VIS_MIN_RECALL)
from .worker import C


def _centres_in(box: np.ndarray, csv: np.ndarray) -> np.ndarray:
    cx = (csv[:, 0] + csv[:, 2]) / 2; cy = (csv[:, 1] + csv[:, 3]) / 2
    return (cx >= box[0]) & (cx <= box[2]) & (cy >= box[1]) & (cy <= box[3])


def splitter(frames: dict[int, Any], csv_by_frame: dict[int, np.ndarray], frame_ids: list[int]) -> dict[str, Any]:
    singles, merges = [], []
    for fid in frame_ids:
        f = frames[fid]
        cb = csv_by_frame.get(fid, np.zeros((0, 4)))
        if "large" not in f or len(cb) == 0:
            continue
        L = f["large"]
        parts = {s[0]: s[3] for s in f["large_splits"]}
        for j, row in enumerate(L):
            if row[C["area"]] < 200:
                continue
            box = row[:4]
            inside = _centres_in(box, cb)
            ious = iou_xyxy(box[None], cb)[0]
            if inside.sum() >= 2:
                merges.append((cb[inside], box, parts.get(j, {})))
            elif inside.sum() == 1 and ious.max() >= 0.7:
                singles.append(parts.get(j, {}))
    res = {"known_single_components": len(singles), "known_merged_components": len(merges),
           "csv_instances_in_merges": int(sum(len(m[0]) for m in merges)), "by_threshold": {}}
    base = sum(len(greedy_match(iou_xyxy(gt, box[None]), 0.5)) for gt, box, _ in merges)
    res["instances_recovered_without_split"] = base
    for thr in NECK_GRID:
        over = sum(1 for p in singles if thr in p) / max(len(singles), 1)
        split = sum(1 for _, _, p in merges if thr in p)
        rec = 0
        for gt, box, p in merges:
            pieces = p[thr][:, :4] if thr in p else box[None]
            rec += len(greedy_match(iou_xyxy(gt, pieces), 0.5))
        res["by_threshold"][str(thr)] = {
            "oversplit_rate_on_single_rocks": round(over, 4),
            "merged_components_split": split, "merged_components_split_rate": round(split / max(len(merges), 1), 4),
            "csv_instances_recovered_iou_0.5": rec,
            "instance_recovery_rate": round(rec / max(res["csv_instances_in_merges"], 1), 4),
        }
    ok = [t for t in NECK_GRID if res["by_threshold"][str(t)]["oversplit_rate_on_single_rocks"] <= MAX_OVERSPLIT_RATE]
    best = max(ok, key=lambda t: (res["by_threshold"][str(t)]["csv_instances_recovered_iou_0.5"], -t)) if ok else None
    gain = (res["by_threshold"][str(best)]["csv_instances_recovered_iou_0.5"] - base) if best is not None else 0
    res["selected_neck_threshold"] = best
    res["net_instance_gain_vs_unsplit"] = gain
    res["splitting_enabled"] = bool(best is not None and gain > 0)
    res["rule"] = f"largest instance recovery with over-split rate <= {MAX_OVERSPLIT_RATE} on known single rocks"
    return res


def visibility(frames: dict[int, Any], task_b_dir: Path) -> dict[str, Any]:
    cand = {r["candidate_id"]: r for r in csv.DictReader(open(task_b_dir / "s2_mask_overlays" / "candidate_review" / "candidates.csv", encoding="utf-8"))}
    rev = {r["candidate_id"]: r for r in csv.DictReader(open(task_b_dir / "assistant_review" / "candidate_review.csv", encoding="utf-8"))}
    X, y, found, missing = [], [], 0, 0
    for cid, c in cand.items():
        v = rev.get(cid, {}).get("verdict")
        if v not in ("VISIBLE_ROCK", "AMBIGUOUS", "NOT_VISIBLE"):
            continue
        f = frames.get(int(c["frame_id"]), {})
        arr = f.get(c["mask_class"])
        if arr is None or not len(arr):
            missing += 1; continue
        hit = np.nonzero((arr[:, 0] == int(c["x0"])) & (arr[:, 1] == int(c["y0"])) & (arr[:, 2] == int(c["x1"])) & (arr[:, 3] == int(c["y1"])))[0]
        if not len(hit):
            missing += 1; continue  # e.g. 25-49 px candidates are below the Task C minimum area
        found += 1
        r = arr[hit[0]]
        gm = f["scene_grad_med"] if f["scene_grad_med"] == f["scene_grad_med"] and f["scene_grad_med"] > 0 else np.nan
        sd = f["scene_ground_std"] if f["scene_ground_std"] == f["scene_ground_std"] and f["scene_ground_std"] > 0 else np.nan
        c = np.nan_to_num(r[C["contrast"]], nan=0.0)
        X.append((r[C["edge_mean"]], c, r[C["in_mean"]], r[C["edge_mean"]] / gm, c / sd))
        y.append(v == "VISIBLE_ROCK")
    X, y = np.nan_to_num(np.array(X), nan=0.0), np.array(y, bool)
    grid = []
    for mode, ie, ic, ge, gc in (("absolute", 0, 1, VIS_EDGE_GRID, VIS_CONTRAST_GRID),
                                 ("scene_relative", 3, 4, VIS_EDGE_REL_GRID, VIS_CONTRAST_REL_GRID)):
        for te in ge:
            for tc in gc:
                keep = (X[:, ie] >= te) | (X[:, ic] >= tc)
                tpr = (keep & y).sum() / max(y.sum(), 1)
                fpr = (keep & ~y).sum() / max((~y).sum(), 1)
                grid.append({"mode": mode, "edge": round(float(te), 3), "contrast": round(float(tc), 3), "visible_kept": round(float(tpr), 4),
                             "ambiguous_kept": round(float(fpr), 4), "youden_j": round(float(tpr - fpr), 4)})
    ok = [g for g in grid if g["visible_kept"] >= VIS_MIN_RECALL]
    # ties prefer scene-relative (robust to lighting, see frame 8894) then stricter thresholds
    best = max(ok, key=lambda g: (g["youden_j"], g["mode"] == "scene_relative", g["edge"], g["contrast"])) if ok else max(grid, key=lambda g: g["youden_j"])
    best_by_mode = {m: max((g for g in ok if g["mode"] == m), key=lambda g: (g["youden_j"], g["edge"], g["contrast"]), default=None)
                    for m in ("absolute", "scene_relative")}
    cols = ["edge_mean", "contrast", "in_mean", "edge_rel", "contrast_rel"]
    return {"labelled_candidates_used": found, "not_matched_or_below_min_area": missing,
            "positives_visible": int(y.sum()), "negatives_ambiguous_or_not_visible": int((~y).sum()),
            "feature_medians": {"visible": np.median(X[y], axis=0).round(3).tolist() if y.any() else None,
                                "not_visible": np.median(X[~y], axis=0).round(3).tolist() if (~y).any() else None, "columns": cols},
            "grid": grid, "best_by_mode": best_by_mode, "selected": best,
            "rule": "keep if edge >= t_e OR contrast >= t_c; absolute (grey levels) or scene-relative (edge / median ground gradient, "
                    f"contrast / ground grey-level std); maximise Youden J subject to keeping >= {VIS_MIN_RECALL} of visible rocks",
            "caveat": "in-sample on Task B assistant review (small n); evaluated out of sample by the Task C review"}
