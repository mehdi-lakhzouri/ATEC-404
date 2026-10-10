"""Scene-similarity evidence for leakage control: thumbnails, consecutive-ID correlation and near-duplicate pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

TW, TH = 48, 32


def thumb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"))
    return cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)


def _norm(t: np.ndarray) -> np.ndarray:
    v = t.reshape(len(t), -1)
    v = v - v.mean(axis=1, keepdims=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def nearest_neighbours(thumbs: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Best Pearson correlation (and index) of each thumbnail against all others."""
    v = _norm(thumbs)
    best = np.full(len(v), -2.0, np.float32); arg = np.zeros(len(v), np.int64)
    for s in range(0, len(v), chunk):
        c = v[s:s + chunk] @ v.T
        for k in range(len(c)):
            c[k, s + k] = -2.0
        arg[s:s + chunk] = c.argmax(axis=1); best[s:s + chunk] = c.max(axis=1)
    return best, arg


def pairs_above(thumbs: np.ndarray, thr: float, chunk: int = 1024) -> list[tuple[int, int, float]]:
    v = _norm(thumbs)
    out = []
    for s in range(0, len(v), chunk):
        c = v[s:s + chunk] @ v.T
        ii, jj = np.nonzero(c >= thr)
        for i, j in zip(ii, jj):
            if s + i < j:
                out.append((int(s + i), int(j), float(c[i, j])))
    return out


ORB_N = 600


def orb_features(path: str) -> dict[str, Any]:
    """ORB keypoints/descriptors on the render (sky is black and featureless) + a gradient-detail retrieval vector."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"))
    orb = cv2.ORB_create(nfeatures=ORB_N, fastThreshold=12)
    kp, des = orb.detectAndCompute(g, None)
    pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    det = cv2.resize(grad, (96, 64), interpolation=cv2.INTER_AREA)
    return {"pts": pts, "des": des if des is not None else np.zeros((0, 32), np.uint8), "detail": det.astype(np.float32)}


def verify(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int]:
    """(cross-checked matches, RANSAC homography inliers) between two ORB feature sets."""
    if len(a["des"]) < 8 or len(b["des"]) < 8:
        return 0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(a["des"], b["des"])
    m = [x for x in m if x.distance <= 40]
    if len(m) < 8:
        return len(m), 0
    p = a["pts"][[x.queryIdx for x in m]]; q = b["pts"][[x.trainIdx for x in m]]
    cv2.setRNGSeed(0)  # RANSAC is randomised; fixed seed keeps the result reproducible
    _, inl = cv2.findHomography(p, q, cv2.RANSAC, 4.0)
    return len(m), int(inl.sum()) if inl is not None else 0


def consecutive_test(thumbs: np.ndarray, ids: list[int], seed: int) -> dict[str, Any]:
    """Are frames with consecutive IDs more similar than random pairs? (sequence leakage evidence)"""
    v = _norm(thumbs)
    pos = {fid: k for k, fid in enumerate(ids)}
    cons = [float(v[pos[i]] @ v[pos[i + 1]]) for i in ids if i + 1 in pos]
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(v), 20000); b = rng.integers(0, len(v), 20000)
    keep = a != b
    rnd = (v[a[keep]] * v[b[keep]]).sum(axis=1)
    lag = {}
    for d in (1, 2, 5, 10, 50):
        x = [float(v[pos[i]] @ v[pos[i + d]]) for i in ids if i + d in pos]
        lag[str(d)] = {"mean": round(float(np.mean(x)), 4), "p95": round(float(np.percentile(x, 95)), 4)}
    return {"consecutive_mean": round(float(np.mean(cons)), 4), "consecutive_p95": round(float(np.percentile(cons, 95)), 4),
            "random_mean": round(float(rnd.mean()), 4), "random_p95": round(float(np.percentile(rnd, 95)), 4),
            "random_p99": round(float(np.percentile(rnd, 99)), 4), "by_id_lag": lag,
            "note": "Pearson correlation of 48x32 grey thumbnails; similar consecutive/random means => IDs are not sequence-ordered"}
