"""S2 - mask-derived rock candidates from the Artificial Lunar `ground` masks (read-only).

Candidates are provisional analysis objects, not ground truth. Nothing here writes to the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..dataset_exploration.artificial_lunar import _dhash
from .config import CLASS_TOL, DARK_LEVEL, IMG_H, IMG_W, LOW_VIS_CONTRAST, PRIMARY_VARIANT, STORE_MIN_AREA, VARIANTS

# feature columns of primary-variant candidates
FEAT = ("x", "y", "w", "h", "area", "fill", "border", "in_mean", "in_std", "ring_mean", "contrast")
F = {k: i for i, k in enumerate(FEAT)}


def load_rgb(p: Path) -> tuple[np.ndarray, str]:
    with Image.open(p) as im:
        mode = im.mode
        return np.asarray(im.convert("RGB")), mode  # handles palette ('P') masks


def class_masks(ground: np.ndarray, cyan_is_blue: bool) -> dict[str, np.ndarray]:
    on = ground > CLASS_TOL
    r, g, b = on[..., 0], on[..., 1], on[..., 2]
    blue = ~r & b & (~g if not cyan_is_blue else np.ones_like(g))
    return {"sky": r & ~g & ~b, "large": blue, "small": ~r & g & ~b}


def _label(mask: np.ndarray, close_px: int) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(np.uint8)
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return labels, stats[1:].astype(np.int64)  # x, y, w, h, area


def _features(gray: np.ndarray, labels: np.ndarray, stats: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Visibility evidence in the *render*: rock vs surrounding-ground contrast."""
    n = len(stats)
    if n == 0:
        return np.zeros((0, len(FEAT)))
    flat = labels.ravel()
    g = gray.ravel().astype(np.float64)
    cnt = np.bincount(flat, minlength=n + 1)[1:]
    s1 = np.bincount(flat, weights=g, minlength=n + 1)[1:]
    s2 = np.bincount(flat, weights=g * g, minlength=n + 1)[1:]
    mean = s1 / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0))
    out = np.zeros((n, len(FEAT)))
    for i, (x, y, w, h, a) in enumerate(stats):
        out[i, :5] = (x, y, w, h, a)
        out[i, F["fill"]] = a / max(w * h, 1)
        out[i, F["border"]] = float(x <= 0 or y <= 0 or x + w >= IMG_W or y + h >= IMG_H)
        out[i, F["in_mean"]] = mean[i]
        out[i, F["in_std"]] = std[i]
        if a < STORE_MIN_AREA:
            out[i, F["ring_mean"]] = np.nan
            continue
        x0, y0, x1, y1 = max(x - 4, 0), max(y - 4, 0), min(x + w + 4, IMG_W), min(y + h + 4, IMG_H)
        bg = background[y0:y1, x0:x1]
        out[i, F["ring_mean"]] = float(gray[y0:y1, x0:x1][bg].mean()) if bg.any() else np.nan
    out[:, F["contrast"]] = np.abs(out[:, F["in_mean"]] - out[:, F["ring_mean"]])
    return out


def visibility_flag(row: np.ndarray) -> str:
    """Automated proxy only - never a substitute for visual review."""
    if row[F["in_mean"]] < DARK_LEVEL and (np.isnan(row[F["ring_mean"]]) or row[F["ring_mean"]] < DARK_LEVEL):
        return "dark"
    if np.isnan(row[F["contrast"]]):
        return "no_surround"
    if row[F["contrast"]] < LOW_VIS_CONTRAST and row[F["in_std"]] < LOW_VIS_CONTRAST:
        return "low_contrast"
    return "ok"


def analyse_frame(args: tuple[str, int, list[list[float]]]) -> dict[str, Any]:
    """One frame: all variants' components, primary-variant features, per-CSV-box mask evidence."""
    root, fid, csv_boxes = args
    root = Path(root)
    out: dict[str, Any] = {"id": fid}
    try:
        render, _ = load_rgb(root / "images" / "render" / f"render{fid:04d}.png")
        ground, mode = load_rgb(root / "images" / "ground" / f"ground{fid:04d}.png")
    except Exception as e:  # corrupt file: report, never skip silently
        out["error"] = repr(e)
        return out
    gray = render[..., 0]
    out["ground_mode"] = mode
    out["render_mean"] = float(gray.mean())
    out["dhash"] = _dhash(gray)
    out["variants"] = {}
    for v in VARIANTS:
        cm = class_masks(ground, v.cyan_is_blue)
        res = {}
        for cls in ("large", "small"):
            labels, stats = _label(cm[cls], v.close_px)
            keep = stats[:, 4] >= STORE_MIN_AREA
            res[cls] = stats[keep]
            res[f"n_{cls}_below_store"] = int((~keep).sum())
            if v.name == PRIMARY_VARIANT:
                rock_any = cm["large"] | cm["small"]
                background = ~rock_any & ~cm["sky"]
                feats = _features(gray, labels, stats, background)
                res[f"{cls}_feat"] = feats[keep]
        if v.name == PRIMARY_VARIANT:
            sky = cm["sky"]
            out["sky_frac"] = float(sky.mean())
            ns = ~sky
            out["dark_frac_non_sky"] = float((gray[ns] < 20).mean()) if ns.any() else 0.0
            # evidence for each CSV box (0-based edges)
            ev = []
            for x0, y0, x1, y1 in csv_boxes:
                xa, ya, xb, yb = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
                sub_s = class_masks(ground[ya:yb, xa:xb], False)["large"]
                sub_c = cm["large"][ya:yb, xa:xb]
                frac_s = float(sub_s.mean()) if sub_s.size else 0.0
                frac_c = float(sub_c.mean()) if sub_c.size else 0.0
                union_iou = 0.0
                if sub_c.any():
                    ys, xs = np.nonzero(sub_c)
                    ub = (xa + xs.min(), ya + ys.min(), xa + xs.max() + 1, ya + ys.max() + 1)
                    inter = (ub[2] - ub[0]) * (ub[3] - ub[1])
                    union_iou = inter / ((x1 - x0) * (y1 - y0))  # union bbox lies inside the CSV box
                ev.append({"blue_frac_strict": frac_s, "blue_frac_cyan": frac_c, "union_bbox_iou": float(union_iou),
                           "render_mean_in_box": float(gray[ya:yb, xa:xb].mean()) if sub_c.size else 0.0})
            out["csv_evidence"] = ev
        out["variants"][v.name] = res
    return out
