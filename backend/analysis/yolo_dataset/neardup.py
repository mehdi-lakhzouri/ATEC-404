"""Near-duplicate / same-terrain detection: retrieval by thumbnail + gradient detail, verification by ORB + RANSAC.

Calibration (recorded in the report): identical frames 1/6658 -> 600 inliers; same frame shifted + zoomed -> 217;
visually different high-correlation pairs -> 0..8; 40 random pairs -> max 5.  DUP_INLIERS = 40.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .similarity import _norm, orb_features, verify

DUP_INLIERS = 40
INSPECT_INLIERS = 15
K_RETRIEVE = 15
_F: dict[str, Any] = {}


def _init(cache: str) -> None:
    # cache holds plain numeric arrays (no pickled objects): concatenated features + per-image offsets
    with np.load(cache) as d:  # read each array once (indexing an NpzFile re-reads it from disk)
        off, pts, des = d["offsets"], d["pts"], d["des"]
    _F["pts"] = [pts[off[k]:off[k + 1]] for k in range(len(off) - 1)]
    _F["des"] = [des[off[k]:off[k + 1]] for k in range(len(off) - 1)]


def _verify_rows(rows: list[tuple[int, list[int]]]) -> list[tuple[int, int, int, int]]:
    out = []
    for i, js in rows:
        a = {"pts": _F["pts"][i], "des": _F["des"][i]}
        for j in js:
            m, inl = verify(a, {"pts": _F["pts"][j], "des": _F["des"][j]})
            out.append((i, j, m, inl))
    return out


def run(paths: list[str], thumbs: np.ndarray, cache: Path, workers: int, seed: int) -> dict[str, Any]:
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(orb_features, paths, chunksize=32))
    detail = np.stack([f["detail"] for f in feats])
    off = np.cumsum([0] + [len(f["des"]) for f in feats])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pts=np.concatenate([f["pts"] for f in feats]), des=np.concatenate([f["des"] for f in feats]),
             detail=detail, offsets=off)
    # candidate retrieval: top-K by detail correlation + top-K by global thumbnail correlation
    cand: dict[int, set[int]] = {i: set() for i in range(len(paths))}
    for vecs in (_norm(detail), _norm(thumbs)):
        for s in range(0, len(vecs), 1024):
            c = vecs[s:s + 1024] @ vecs.T
            for k in range(len(c)):
                c[k, s + k] = -2
            top = np.argpartition(-c, K_RETRIEVE, axis=1)[:, :K_RETRIEVE]
            for k, row in enumerate(top):
                cand[s + k].update(int(j) for j in row)
    rows = [(i, sorted(j for j in js if j > i or i not in cand[j])) for i, js in cand.items()]
    chunks = [rows[s:s + 200] for s in range(0, len(rows), 200)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(cache),)) as ex:
        res = [r for part in ex.map(_verify_rows, chunks) for r in part]
    seen, pairs = set(), []
    for i, j, m, inl in res:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((key[0], key[1], m, inl))
    inl = np.array([p[3] for p in pairs])
    # exhaustive audit of retrieval recall on a random query subset
    rng = np.random.default_rng(seed)
    q = sorted(rng.choice(len(paths), size=min(60, len(paths)), replace=False).tolist())
    audit_rows = [(int(i), [j for j in range(len(paths)) if j != i]) for i in q]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(cache),)) as ex:
        audit = [r for part in ex.map(_verify_rows, [[r] for r in audit_rows]) for r in part]
    audit_hits = [(i, j, n) for i, j, _, n in audit if n >= INSPECT_INLIERS]
    missed = [(i, j, n) for i, j, n in audit_hits if j not in cand[i] and i not in cand[j]]
    a_inl = np.array([r[3] for r in audit])
    return {
        "pairs_verified": len(pairs),
        "inlier_histogram": {"0-4": int((inl < 5).sum()), "5-14": int(((inl >= 5) & (inl < 15)).sum()),
                             "15-39": int(((inl >= 15) & (inl < 40)).sum()), ">=40": int((inl >= 40).sum())},
        "duplicates": [(i, j, n) for i, j, _, n in pairs if n >= DUP_INLIERS],
        "inspect": [(i, j, n) for i, j, _, n in pairs if INSPECT_INLIERS <= n < DUP_INLIERS],
        "audit": {"queries": len(q), "pairs_checked": len(audit), "max_inliers": int(a_inl.max()) if len(a_inl) else 0,
                  "pairs_ge_15": len(audit_hits), "pairs_ge_15_missed_by_retrieval": len(missed),
                  "missed_examples": missed[:10],
                  "inlier_p999": float(np.percentile(a_inl, 99.9)) if len(a_inl) else 0.0},
        "method": f"candidates = top-{K_RETRIEVE} by 96x64 gradient-detail correlation + top-{K_RETRIEVE} by 48x32 thumbnail correlation; "
                  f"verified with ORB ({600} kp, Hamming<=40, cross-check) + RANSAC homography (4 px); duplicate if inliers >= {DUP_INLIERS}",
    }
