"""Ignore-aware single-class detection evaluation (COCO-style, 101-point AP).

Ultralytics 8.4.174 `val` has no ignore-region support (verified: `iscrowd` is only written as 0 in COCO export).
This evaluator treats ignore regions like COCO crowd regions:

* a prediction that does not match a ground-truth box but matches an ignore region
  (IoU >= iou_thr, or >= ioa_thr of the prediction's own area inside the region) is IGNORED - neither TP nor FP;
* ignore regions never create false negatives;
* predictions smaller than the evaluation scope (min_pred_area) are dropped before matching;
* for size-binned metrics, ground truth outside the bin is ignored, and unmatched predictions outside the bin are ignored.

Boxes are pixel xyxy (x1/y1 exclusive). Coordinates of YOLO .txt files are converted with `yolo_to_xyxy`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SIZE_BINS = {"small": (0.0, 400.0), "medium": (400.0, 2500.0), "large": (2500.0, float("inf"))}   # bbox area, native 720x480 px
IOU_THRS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


@dataclass
class ImageEval:
    gt: np.ndarray                                   # (N,4) xyxy
    pred: np.ndarray                                 # (M,5) xyxy + score
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def yolo_to_xyxy(lines: Iterable[str], w: int = 720, h: int = 480, with_score: bool = False) -> np.ndarray:
    out = []
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        cx, cy, bw, bh = (float(x) for x in t[1:5])
        row = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        if with_score:
            row.append(float(t[5]) if len(t) > 5 else 1.0)
        out.append(row)
    return np.array(out, np.float64).reshape(-1, 5 if with_score else 4)


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return ix * iy


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    i = _inter(a, b)
    return i / np.maximum(_area(a)[:, None] + _area(b)[None, :] - i, 1e-12)


def ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of each box in a."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return _inter(a, b) / np.maximum(_area(a)[:, None], 1e-12)


def match_image(img: ImageEval, iou_thr: float, ioa_thr: float = 0.5, min_pred_area: float = 0.0,
                area_rng: tuple[float, float] = (0.0, float("inf"))) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (scores, labels) for kept predictions (label 1 = TP, 0 = FP) and the number of counted GT."""
    pred = img.pred[_area(img.pred[:, :4]) >= min_pred_area] if len(img.pred) else img.pred.reshape(0, 5)
    pred = pred[np.argsort(-pred[:, 4], kind="stable")]
    gt = img.gt.reshape(-1, 4)
    ga = _area(gt)
    gt_in = (ga >= area_rng[0]) & (ga < area_rng[1])
    ign = img.ignore.reshape(-1, 4)
    gt_out = gt[~gt_in]                                  # GT outside the size bin: ignored if matched by IoU (COCO rule)
    gt = gt[gt_in]
    ov = iou(pred[:, :4], gt)
    ov_ign_iou = iou(pred[:, :4], ign)
    ov_ign_ioa = ioa(pred[:, :4], ign)
    ov_out = iou(pred[:, :4], gt_out)
    used = np.zeros(len(gt), bool)
    scores, labels = [], []
    pa = _area(pred[:, :4]) if len(pred) else np.zeros(0)
    for k in range(len(pred)):
        j = -1
        if len(gt):
            cand = np.where(~used, ov[k], -1.0)
            j = int(np.argmax(cand)) if cand.max() >= iou_thr else -1
        if j >= 0:
            used[j] = True; scores.append(pred[k, 4]); labels.append(1); continue
        if len(ign) and (ov_ign_iou[k].max() >= iou_thr or ov_ign_ioa[k].max() >= ioa_thr):
            continue                                     # ignored: neither TP nor FP
        if len(gt_out) and ov_out[k].max() >= iou_thr:
            continue                                     # matches a GT of another size bin
        if not (area_rng[0] <= pa[k] < area_rng[1]):
            continue                                     # unmatched prediction outside the size bin
        scores.append(pred[k, 4]); labels.append(0)
    return np.array(scores, np.float64), np.array(labels, np.int8), int(len(gt))


def ap_101(scores: np.ndarray, labels: np.ndarray, n_gt: int) -> dict[str, Any]:
    if n_gt == 0:
        return {"ap": float("nan"), "precision": None, "recall": None}
    o = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[o] == 1); fp = np.cumsum(labels[o] == 0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    env = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    p = np.array([env[i] if i < len(env) else 0.0 for i in idx])
    return {"ap": float(p.mean()), "precision": prec, "recall": rec}


def evaluate(images: dict[str, ImageEval], conf: float = 0.25, ioa_thr: float = 0.5, min_pred_area: float = 0.0) -> dict[str, Any]:
    res: dict[str, Any] = {"images": len(images), "conf": conf, "ioa_thr": ioa_thr, "min_pred_area": min_pred_area}

    def collect(thr: float, rng: tuple[float, float]):
        S, L, n = [], [], 0
        for im in images.values():
            s, l, g = match_image(im, thr, ioa_thr, min_pred_area, rng)
            S.append(s); L.append(l); n += g
        return (np.concatenate(S) if S else np.zeros(0)), (np.concatenate(L) if L else np.zeros(0, np.int8)), n

    full = (0.0, float("inf"))
    aps = []
    for thr in IOU_THRS:
        s, l, n = collect(float(thr), full)
        aps.append(ap_101(s, l, n)["ap"])
        if thr == 0.5:
            keep = s >= conf
            tp, fp = int((l[keep] == 1).sum()), int((l[keep] == 0).sum())
            res.update({"gt": n, "AP50": aps[-1], "tp@conf": tp, "fp@conf": fp, "fn@conf": n - tp,
                        "precision@conf": tp / max(tp + fp, 1), "recall@conf": tp / max(n, 1)})
    res["AP50_95"] = float(np.nanmean(aps))
    res["by_size"] = {}
    for name, rng in SIZE_BINS.items():
        s, l, n = collect(0.5, rng)
        a = ap_101(s, l, n)["ap"]
        keep = s >= conf
        res["by_size"][name] = {"gt": n, "AP50": a, "recall@conf": int((l[keep] == 1).sum()) / max(n, 1)}
    return res


def load_split(dataset: Path, split: str, pred_dir: Path | None, ignore_files: list[Path]) -> dict[str, ImageEval]:
    """GT from dataset/labels/<split>, predictions from Ultralytics `save_txt=True, save_conf=True` output."""
    ign: dict[str, list] = {}
    for f in ignore_files:
        regions = json.loads(f.read_text(encoding="utf-8"))["regions"].get(split, {})
        for name, boxes in regions.items():
            ign.setdefault(name, []).extend(b[:4] for b in boxes)
    out = {}
    for lp in sorted((dataset / "labels" / split).glob("*.txt")):
        name = lp.stem + ".png"
        gt = yolo_to_xyxy(lp.read_text(encoding="utf-8").splitlines())
        pp = (pred_dir / lp.name) if pred_dir else None
        pred = yolo_to_xyxy(pp.read_text(encoding="utf-8").splitlines(), with_score=True) if pp and pp.exists() else np.zeros((0, 5))
        out[name] = ImageEval(gt, pred, np.array(ign.get(name, []), np.float64).reshape(-1, 4))
    return out
