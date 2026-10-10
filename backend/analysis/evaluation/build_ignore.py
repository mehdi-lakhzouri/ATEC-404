"""Build PROPOSED sub-threshold evaluation ignore regions for val/test (not part of the label policy).

Regions = V1_cyan mask components of 10-49 px at class threshold 127, plus components >= 50 px that only appear at
class threshold 80 (dim far-field rocks). Detections on these out-of-scope rocks are then neither TP nor FP.

Usage (from backend/):  uv run --no-sync python -m analysis.evaluation.build_ignore --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..remediation.threshold_study import COLS, frame_components


def _overlaps(b: np.ndarray, others: list[np.ndarray], frac: float = 0.3) -> bool:
    for c in others:
        inter = max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
        if inter > frac * (b[2] - b[0]) * (b[3] - b[1]):
            return True
    return False


def run(root: Path, workers: int, dataset: str = "yolo-rock-v1.1") -> dict:
    ds = root / "dataset/processed" / dataset
    with (ds / "manifests/split_manifest.csv").open(newline="", encoding="utf-8") as fh:
        man = {int(r["frame_id"]): r["split"] for r in csv.DictReader(fh)}
    frames = sorted(f for f, s in man.items() if s in ("val", "test"))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(frame_components, [(str(root / "dataset/artificial-lunar"), f) for f in frames], chunksize=8))
    c = {k: i for i, k in enumerate(COLS)}
    regions = {"val": {}, "test": {}}
    counts = {"val": [0, 0], "test": [0, 0]}
    for f, (A, _) in zip(frames, res):
        t127 = A[A[:, c["tol"]] == 127]
        ge50 = [r[3:7] for r in t127 if r[c["area"]] >= 50]
        sub = [r for r in t127 if 10 <= r[c["area"]] < 50]
        dim = [r for r in A[A[:, c["tol"]] == 80] if r[c["area"]] >= 50 and not _overlaps(r[3:7], ge50)]
        boxes = [[float(r[3]), float(r[4]), float(r[5]), float(r[6]), "subthreshold_10_49px"] for r in sub] + \
                [[float(r[3]), float(r[4]), float(r[5]), float(r[6]), "dim_mask_ge50px_tol80"] for r in dim]
        if boxes:
            regions[man[f]][f"render{f:04d}.png"] = boxes
        counts[man[f]][0] += len(sub); counts[man[f]][1] += len(dim)
    out = {"status": "PROPOSED - evaluation-only ignore regions; NOT labels; NOT approved",
           "definition": "V1_cyan mask components 10-49 px (class threshold 127) + components >= 50 px only present at class threshold 80",
           "counts": {s: {"subthreshold_10_49px": v[0], "dim_mask_ge50px_tol80": v[1]} for s, v in counts.items()},
           "frames_with_regions": {s: len(r) for s, r in regions.items()}, "regions": regions}
    p = ds / "manifests/eval_ignore_regions_subthreshold_PROPOSED.json"
    p.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True), encoding="utf-8", newline="\n")
    return {k: v for k, v in out.items() if k != "regions"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    print(json.dumps(run(Path(__file__).resolve().parents[3], ap.parse_args().workers), indent=1))
