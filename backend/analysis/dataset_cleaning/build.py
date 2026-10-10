"""S2-H decision engine: one record per candidate with status, reasons, flags and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..annotation_validation.al_analysis import _contained
from ..dataset_exploration.common import iou_xyxy
from .config import (BLUE_COVERED_BY_CSV, BLUE_IOU_WITH_CSV, BLUE_MAX_AREA_UNSPLIT, CSV_DARK_FRACTION, CSV_SILHOUETTE_RING, DARK_LEVEL,
                     GREEN_INSIDE_CSV, GREEN_INSIDE_CSV_AREA, GREEN_SAME_AS_CSV_IOU, MIN_SIDE, NECK_GRID,
                     NECK_REVIEW_MARGIN, SOLIDITY_REVIEW)
from .worker import C

FEAT_OUT = ("area", "fill", "solidity", "in_mean", "ring_mean", "contrast", "edge_mean", "dark_frac")


def _vis_ok(r: np.ndarray, vis: dict[str, float]) -> bool:
    c = 0.0 if np.isnan(r[C["contrast"]]) else r[C["contrast"]]
    return r[C["edge_mean"]] >= vis["edge"] or c >= vis["contrast"]


def _record(fid: int, k: int, source: str, r: np.ndarray, status: str, reasons: list[str], flags: list[str],
            parent: str = "", split_thr: float | None = None, csv_line: int | None = None) -> dict[str, Any]:
    rec = {"ann_id": f"{fid:04d}-{source}-{k:03d}", "frame_id": fid, "class_id": 0, "class_name": "rock",
           "x0": round(float(r[0]), 2), "y0": round(float(r[1]), 2), "x1": round(float(r[2]), 2), "y1": round(float(r[3]), 2),
           "w": round(float(r[2] - r[0]), 2), "h": round(float(r[3] - r[1]), 2), "source": source, "status": status,
           "reasons": ";".join(reasons), "flags": ";".join(flags), "parent_id": parent,
           "split_neck_threshold": "" if split_thr is None else split_thr, "csv_line": "" if csv_line is None else csv_line,
           "border": bool(r[C["border"]]) if len(r) > 4 else False}
    for k2 in FEAT_OUT:
        v = r[C[k2]] if len(r) > 4 else np.nan
        rec[k2] = "" if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, np.floating) and np.isnan(v)) else round(float(v), 3)
    return rec


def duplicates_csv(box: np.ndarray, cls: str, cb: np.ndarray) -> bool:
    """Rule D05: a mask-derived box duplicates a CSV instance (blue: >=50% inside or IoU>=0.3; green: IoU>=0.5)."""
    if not len(cb):
        return False
    b = np.asarray(box, dtype=np.float64)[None, :4]
    iou = iou_xyxy(b, cb)[0]
    if cls == "large":
        return bool(_contained(b, cb)[0].max() >= BLUE_COVERED_BY_CSV or iou.max() >= BLUE_IOU_WITH_CSV)
    return bool(iou.max() >= GREEN_SAME_AS_CSV_IOU)


def frame_records(fid: int, f: dict[str, Any], csv_rows: list[dict[str, Any]], vis: dict[str, float],
                  neck: float | None, split_on: bool) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if vis.get("mode") == "scene_relative":
        gm, sd = f["scene_grad_med"], f["scene_ground_std"]
        if gm == gm and sd == sd and gm > 0 and sd > 0:
            vis = {"edge": vis["edge"] * gm, "contrast": vis["contrast"] * sd}
        else:  # no plain ground in frame: fall back to the absolute rule
            vis = {"edge": vis["fallback_abs"]["edge"], "contrast": vis["fallback_abs"]["contrast"]}
    cb = np.array([[b["x"] - 0.5, b["y"] - 0.5, b["x"] - 0.5 + b["w"], b["y"] - 0.5 + b["h"]] for b in csv_rows]).reshape(-1, 4)
    for k, (b, box) in enumerate(zip(csv_rows, cb)):
        dark = f["csv_dark_frac"][k]
        ring = f["csv_ring_mean"][k]
        if dark >= CSV_DARK_FRACTION and (np.isnan(ring) or ring < CSV_SILHOUETTE_RING):
            st, rs, fl = "REVIEW_REQUIRED", ["csv_box_in_black_shadow"], []
        elif dark >= CSV_DARK_FRACTION:
            st, rs, fl = "VALID", [], ["dark_silhouette"]   # black rock against lit surroundings: visible obstacle
        else:
            st, rs, fl = "VALID", [], []
        if box[0] <= 0 or box[1] <= 0 or box[2] >= 720 or box[3] >= 480:
            fl.append("border_truncated")
        rec = _record(fid, k, "csv_large", np.r_[box, np.zeros(len(C) - 4)], st, rs, fl, csv_line=b["line"])
        rec.update({"area": round(float(b["w"] * b["h"]), 1), "dark_frac": round(dark, 3), "border": bool(fl)})
        for k2 in ("fill", "solidity", "in_mean", "ring_mean", "contrast", "edge_mean"):
            rec[k2] = ""
        recs.append(rec)
    review_band = None if neck is None else round(neck + NECK_REVIEW_MARGIN, 4)
    for cls, src in (("large", "mask_blue"), ("small", "mask_green")):
        A = f[cls]
        splits = {s[0]: s for s in f[f"{cls}_splits"]}
        n = 0
        for j, r in enumerate(A):
            dup = "duplicate_of_csv_instance" if duplicates_csv(r[:4], cls, cb) else None
            if dup:
                recs.append(_record(fid, n, src, r, "REJECTED", [dup], [])); n += 1
                continue
            parent_id = f"{fid:04d}-{src}-{n:03d}"
            s = splits.get(j)
            parts = s[3] if s else {}
            if split_on and neck is not None and neck in parts:
                # accepted split: the parent is recorded as REJECTED(replaced_by_split); pieces are judged individually
                recs.append(_record(fid, n, src, r, "REJECTED", ["replaced_by_split_pieces"], ["auto_split_parent"])); n += 1
                for p in parts[neck]:
                    if duplicates_csv(p[:4], cls, cb):
                        # v1.1 fix: rule D05 now also applies to split pieces (v1 applied it to parents only)
                        recs.append(_record(fid, n, src + "_split", p, "REJECTED", ["duplicate_of_csv_instance"],
                                            ["auto_split", "split_piece_dedup"], parent_id, neck))
                    else:
                        recs.append(_decide(fid, n, src + "_split", p, cb, cls, vis, ["auto_split"], parent_id, neck))
                    n += 1
                continue
            extra_review = []
            if s and review_band is not None and any(t for t in parts if neck is not None and neck < t <= review_band):
                extra_review.append("possible_merge_borderline_split")
            elif r[C["solidity"]] < SOLIDITY_REVIEW:
                extra_review.append("low_solidity_possible_merge")
            if cls == "large" and r[C["area"]] >= BLUE_MAX_AREA_UNSPLIT:
                extra_review.append("large_blue_without_csv_likely_merged")
            recs.append(_decide(fid, n, src, r, cb, cls, vis, [], "", None, extra_review)); n += 1
    return recs


def _decide(fid: int, n: int, src: str, r: np.ndarray, cb: np.ndarray, cls: str, vis: dict[str, float],
            flags: list[str], parent: str, thr: float | None, extra_review: list[str] | None = None) -> dict[str, Any]:
    rej, rev = [], list(extra_review or [])
    w, h = r[2] - r[0], r[3] - r[1]
    ring = r[C["ring_mean"]]
    if w < MIN_SIDE or h < MIN_SIDE:
        rej.append("thin_artifact")
    if r[C["in_mean"]] < DARK_LEVEL and (np.isnan(ring) or ring < DARK_LEVEL):
        rej.append("invisible_black_shadow")
    if not rej:
        if r[C["dark_frac"]] >= 0.85:
            rev.append("mostly_in_shadow")
        if not _vis_ok(r, vis):
            rev.append("ambiguous_low_visibility")
        if cls == "small" and len(cb):
            cont = _contained(r[None, :4], cb)[0]
            big = ((cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])) * GREEN_INSIDE_CSV_AREA <= w * h
            if ((cont >= GREEN_INSIDE_CSV) & big).any():
                rev.append("overlaps_csv_large_rock")
    st = "REJECTED" if rej else ("REVIEW_REQUIRED" if rev else "VALID")
    return _record(fid, n, src, r, st, rej or rev, flags, parent, thr)
