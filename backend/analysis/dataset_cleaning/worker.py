"""Per-frame extraction: source hashes, mask components, render-visibility features, watershed split analysis."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..dataset_exploration.artificial_lunar import _dhash
from ..annotation_validation.s2_extract import class_masks
from .config import DARK_LEVEL, IMG_H, IMG_W, MIN_AREA, MIN_PIECE_AREA, NECK_GRID, PEAK_ABS, PEAK_REL

# feature columns for components and split pieces
COLS = ("x0", "y0", "x1", "y1", "area", "fill", "solidity", "in_mean", "in_std", "ring_mean", "contrast", "edge_mean",
        "dark_frac", "border")
C = {k: i for i, k in enumerate(COLS)}


def _read(p: Path) -> tuple[np.ndarray, str, str]:
    b = p.read_bytes()
    with Image.open(io.BytesIO(b)) as im:
        mode = im.mode
        a = np.asarray(im.convert("RGB"))
    return a, mode, hashlib.sha256(b).hexdigest()


def region_features(mask: np.ndarray, ox: int, oy: int, gray: np.ndarray, grad: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Features of a binary region given in a crop whose top-left is (ox, oy) in image coordinates."""
    ys, xs = np.nonzero(mask)
    f = np.zeros(len(COLS))
    if len(xs) == 0:
        return f
    x0, y0, x1, y1 = xs.min() + ox, ys.min() + oy, xs.max() + 1 + ox, ys.max() + 1 + oy
    area = len(xs)
    f[:5] = (x0, y0, x1, y1, area)
    f[C["fill"]] = area / max((x1 - x0) * (y1 - y0), 1)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts)) if cnts else None
    ha = cv2.contourArea(hull) if hull is not None else 0
    f[C["solidity"]] = min(1.0, area / ha) if ha > 0 else 1.0
    H, W = mask.shape
    g = gray[oy:oy + H, ox:ox + W].astype(np.float64)
    vals = g[mask > 0]
    f[C["in_mean"]] = vals.mean(); f[C["in_std"]] = vals.std()
    f[C["dark_frac"]] = float((vals < DARK_LEVEL).mean())
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~mask.astype(bool) & bg[oy:oy + H, ox:ox + W]
    f[C["ring_mean"]] = g[ring].mean() if ring.any() else np.nan
    f[C["contrast"]] = abs(f[C["in_mean"]] - f[C["ring_mean"]]) if ring.any() else np.nan
    edge = mask.astype(bool) & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    f[C["edge_mean"]] = float(grad[oy:oy + H, ox:ox + W][edge].mean()) if edge.any() else 0.0
    f[C["border"]] = float(x0 <= 0 or y0 <= 0 or x1 >= IMG_W or y1 >= IMG_H)
    return f


def split_analysis(mask: np.ndarray) -> dict[str, Any] | None:
    """Distance-transform markers + watershed; returns per-threshold partitions (label maps) or None."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    maxd = float(dist.max())
    if maxd < PEAK_ABS * 1.2:
        return None
    r = max(2, int(round(0.3 * maxd)))
    dil = cv2.dilate(dist, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    peaks = (dist >= dil - 1e-6) & (dist >= PEAK_REL * maxd) & (dist >= PEAK_ABS) & (mask > 0)
    n, pl = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    k = n - 1
    if k < 2:
        return None
    markers = pl.astype(np.int32)
    markers[mask == 0] = n  # background label
    img = cv2.cvtColor((255 - dist / maxd * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    peak_val = np.array([dist[pl == i].max() for i in range(1, n)])
    # neck between adjacent basins = max distance value on their shared watershed line
    neck = np.zeros((k, k))
    by, bx = np.nonzero((markers == -1) & (mask > 0))
    H, W = mask.shape
    for y, x in zip(by, bx):
        nb = markers[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)]
        labs = np.unique(nb[(nb >= 1) & (nb < n)])
        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                i, j = labs[a] - 1, labs[b] - 1
                neck[i, j] = neck[j, i] = max(neck[i, j], dist[y, x])
    ratio = np.full((k, k), np.inf)
    adj = neck > 0
    ratio[adj] = (neck / np.minimum(peak_val[:, None], peak_val[None, :]))[adj]
    # fill watershed lines inside the mask with an adjacent basin label
    lab = markers.copy()
    lab[(lab == n) | (mask == 0)] = 0
    for y, x in zip(by, bx):
        nb = markers[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)]
        c = nb[(nb >= 1) & (nb < n)]
        lab[y, x] = c[0] if len(c) else 0
    partitions = {}
    for thr in NECK_GRID:
        parent = list(range(k))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in range(k):
            for j in range(i + 1, k):
                if adj[i, j] and ratio[i, j] > thr:
                    parent[find(i)] = find(j)
        groups = {find(i) for i in range(k)}
        if len(groups) < 2:
            continue
        part = np.zeros_like(lab)
        for gi, g in enumerate(sorted(groups), start=1):
            for i in range(k):
                if find(i) == g:
                    part[lab == i + 1] = gi
        # absorb tiny pieces into their largest neighbour
        changed = True
        while changed:
            changed = False
            ids = [i for i in np.unique(part) if i > 0]
            if len(ids) < 2:
                break
            for i in ids:
                m = part == i
                if m.sum() < MIN_PIECE_AREA:
                    ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
                    nbrs = part[ring & (part > 0)]
                    if len(nbrs):
                        part[m] = np.bincount(nbrs).argmax()
                        changed = True
                        break
        if len([i for i in np.unique(part) if i > 0]) >= 2:
            partitions[thr] = part
    finite = ratio[np.isfinite(ratio)]
    return {"n_markers": k, "min_ratio": float(finite.min()) if finite.size else None, "partitions": partitions}


def process_frame(args: tuple[str, int, list[list[float]]]) -> dict[str, Any]:
    root, fid, csv_boxes = args
    root = Path(root)
    out: dict[str, Any] = {"id": fid}
    try:
        render, rmode, rsha = _read(root / "images" / "render" / f"render{fid:04d}.png")
        ground, gmode, gsha = _read(root / "images" / "ground" / f"ground{fid:04d}.png")
    except Exception as e:
        out["error"] = repr(e)
        return out
    gray = render[..., 0]
    grad = cv2.magnitude(*(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32), cv2.CV_32F, dx, dy, ksize=3)
                           for dx, dy in ((1, 0), (0, 1))))
    cm = class_masks(ground, cyan_is_blue=True)  # V1_cyan
    sky = cm["sky"]
    bg = ~(cm["large"] | cm["small"]) & ~sky
    # scene texture scale (plain ground only): used to normalise visibility features across lighting conditions
    gb = bg & (gray >= DARK_LEVEL)
    out["scene_grad_med"] = float(np.median(grad[gb])) if gb.sum() > 100 else float("nan")
    out["scene_ground_std"] = float(gray[gb].std()) if gb.sum() > 100 else float("nan")
    out.update(render_sha256=rsha, ground_sha256=gsha, render_mode=rmode, ground_mode=gmode,
               render_mean=float(gray.mean()), sky_frac=float(sky.mean()),
               dark_frac_non_sky=float((gray[~sky] < 20).mean()) if (~sky).any() else 0.0,
               dhash=_dhash(gray), thumb=cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA))
    out["csv_dark_frac"], out["csv_ring_mean"] = [], []
    for x0, y0, x1, y1 in csv_boxes:
        xa, ya, xb, yb = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        sub = gray[ya:yb, xa:xb]
        out["csv_dark_frac"].append(float((sub < DARK_LEVEL).mean()) if sub.size else 1.0)
        # surroundings (6-px frame outside the box, sky excluded): a black rock against lit ground is a visible silhouette
        ring = np.zeros_like(sky)
        ring[max(0, ya - 6):min(IMG_H, yb + 6), max(0, xa - 6):min(IMG_W, xb + 6)] = True
        ring[ya:yb, xa:xb] = False
        ring &= ~sky
        out["csv_ring_mean"].append(float(gray[ring].mean()) if ring.any() else float("nan"))
    for cls in ("large", "small"):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cm[cls].astype(np.uint8), connectivity=8)
        stats = stats[1:]
        keep = np.nonzero(stats[:, 4] >= MIN_AREA)[0]
        out[f"{cls}_below_min"] = int((stats[:, 4] < MIN_AREA).sum())
        out[f"{cls}_below_min_ge25"] = int(((stats[:, 4] >= 25) & (stats[:, 4] < MIN_AREA)).sum())
        feats, splits = [], []
        for ci in keep:
            x, y, w, h, a = stats[ci]
            pad = 2
            ox, oy = max(0, x - pad), max(0, y - pad)
            ex, ey = min(IMG_W, x + w + pad), min(IMG_H, y + h + pad)
            m = (labels[oy:ey, ox:ex] == ci + 1).astype(np.uint8)
            feats.append(region_features(m, ox, oy, gray, grad, bg))
            sa = split_analysis(m)
            if sa and sa["partitions"]:
                parts = {}
                for thr, part in sa["partitions"].items():
                    parts[thr] = np.array([region_features((part == p).astype(np.uint8), ox, oy, gray, grad, bg)
                                           for p in sorted(np.unique(part)) if p > 0])
                splits.append((len(feats) - 1, sa["n_markers"], sa["min_ratio"], parts))
            elif sa:
                splits.append((len(feats) - 1, sa["n_markers"], sa["min_ratio"], {}))
        out[cls] = np.array(feats).reshape(-1, len(COLS))
        out[f"{cls}_splits"] = splits
    return out
