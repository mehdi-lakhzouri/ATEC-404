"""Item 4 - minimum-component-area and class-intensity threshold study (read-only; official policy unchanged).

For every eligible frame, rock components of the V1_cyan classes are extracted at two class-intensity thresholds
(127 = current policy, 80 = candidate) and summarised for candidate minimum areas. Visibility uses the calibrated
Task C rule (edge >= 100 or contrast >= 25) as an automated proxy only.

Usage (from backend/):  uv run --no-sync python -m analysis.remediation.threshold_study --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ..annotation_validation.drawing import crop_tile, grid, save  # noqa: E402
from ..annotation_validation.s2_extract import load_rgb  # noqa: E402
from ..dataset_cleaning.worker import C, region_features  # noqa: E402
from ..dataset_exploration.common import write_json  # noqa: E402

TOLS = (127, 80)
AREAS = (10, 15, 20, 25, 30, 40, 50)
VIS_EDGE, VIS_CONTRAST = 100.0, 25.0     # calibrated Task C rule (curated-dataset-v1.1 policy.json)
SEED = 20261012
COLS = ("frame", "tol", "cls", "x0", "y0", "x1", "y1", "area", "edge", "contrast", "solidity", "in_mean", "dark")


def class_masks_tol(ground: np.ndarray, tol: int) -> dict[str, np.ndarray]:
    r, g, b = (ground[..., i] > tol for i in range(3))
    return {"sky": r & ~g & ~b, "large": ~r & b, "small": ~r & g & ~b}


def frame_components(args: tuple[str, int]) -> tuple[np.ndarray, float]:
    root, fid = args
    t0 = time.perf_counter()
    render, _ = load_rgb(Path(root) / "images/render" / f"render{fid:04d}.png")
    ground, _ = load_rgb(Path(root) / "images/ground" / f"ground{fid:04d}.png")
    gray = render[..., 0]
    grad = cv2.magnitude(*(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32), cv2.CV_32F, dx, dy, ksize=3) for dx, dy in ((1, 0), (0, 1))))
    rows = []
    for tol in TOLS:
        cm = class_masks_tol(ground, tol)
        bg = ~(cm["large"] | cm["small"]) & ~cm["sky"]
        for ci, cls in enumerate(("large", "small")):
            n, lab, st, _ = cv2.connectedComponentsWithStats(cm[cls].astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if a < AREAS[0]:
                    continue
                ox, oy = max(0, x - 2), max(0, y - 2)
                m = (lab[oy:min(480, y + h + 2), ox:min(720, x + w + 2)] == i).astype(np.uint8)
                f = region_features(m, ox, oy, gray, grad, bg)
                rows.append((fid, tol, ci, x, y, x + w, y + h, a, f[C["edge_mean"]], np.nan_to_num(f[C["contrast"]], nan=0.0),
                             f[C["solidity"]], f[C["in_mean"]], f[C["dark_frac"]]))
    return np.array(rows, np.float64).reshape(-1, len(COLS)), time.perf_counter() - t0


def run(root: Path, workers: int) -> dict[str, Any]:
    out = root / "experiments/remediation/threshold_study"
    out.mkdir(parents=True, exist_ok=True)
    with (root / "dataset/processed/yolo-rock-v1.1/manifests/split_manifest.csv").open(newline="", encoding="utf-8") as fh:
        man = {int(r["frame_id"]): r for r in csv.DictReader(fh)}
    frames = sorted(man)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(frame_components, [(str(root / "dataset/artificial-lunar"), f) for f in frames], chunksize=16))
    wall = time.perf_counter() - t0
    A = np.concatenate([r[0] for r in res]); per_frame_s = np.array([r[1] for r in res])
    col = {k: i for i, k in enumerate(COLS)}
    vis = (A[:, col["edge"]] >= VIS_EDGE) | (A[:, col["contrast"]] >= VIS_CONTRAST)
    dark = (A[:, col["in_mean"]] < 15)
    lowsol = A[:, col["solidity"]] < 0.8
    side = np.minimum(A[:, col["x1"]] - A[:, col["x0"]], A[:, col["y1"]] - A[:, col["y0"]])
    nf = len(frames)
    table = {}
    for tol in TOLS:
        for amin in AREAS:
            m = (A[:, col["tol"]] == tol) & (A[:, col["area"]] >= amin)
            fr_with = len(np.unique(A[m, col["frame"]]))
            table[f"tol{tol}_min{amin}"] = {
                "components": int(m.sum()), "per_image": round(m.sum() / nf, 2),
                "frames_with_any": fr_with, "frames_with_none": nf - fr_with,
                "visible_by_proxy_rate": round(float(vis[m].mean()), 4) if m.any() else None,
                "dark_invisible_rate": round(float(dark[m].mean()), 4) if m.any() else None,
                "low_solidity_merge_risk_rate": round(float(lowsol[m].mean()), 4) if m.any() else None,
                "min_side_lt_6px_rate": round(float((side[m] < 6).mean()), 4) if m.any() else None,
                "median_box_side_px": float(np.median(side[m])) if m.any() else None,
            }
    # frames that have NO component >= 50 at tol127 but gain one at tol80
    def frames_with(tol, amin):
        m = (A[:, col["tol"]] == tol) & (A[:, col["area"]] >= amin)
        return set(np.unique(A[m, col["frame"]]).astype(int))
    gain = sorted(frames_with(80, 50) - frames_with(127, 50))
    no_rec = {f for f, r in man.items() if r["annotation_state"] == "no_records"}
    # deterministic visual-review samples
    rng = np.random.default_rng(SEED)
    samples = {}
    def pick(name: str, m: np.ndarray, n: int):
        idx = np.nonzero(m)[0]
        sel = sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist()) if len(idx) else []
        samples[name] = [{"sample_id": f"{name}_{k:02d}", **{c: (int(A[i, col[c]]) if c not in ("edge", "contrast", "solidity", "in_mean", "dark") else round(float(A[i, col[c]]), 2)) for c in COLS}} for k, i in enumerate(sel)]
    t127 = A[:, col["tol"]] == 127
    pick("t127_area10_24", t127 & (A[:, col["area"]] >= 10) & (A[:, col["area"]] < 25), 20)
    pick("t127_area25_49", t127 & (A[:, col["area"]] >= 25) & (A[:, col["area"]] < 50), 20)
    # components that only reach >= 50 px with the lower intensity threshold (approximate: tol80 comps >= 50 in frames' regions not covered at 127)
    t80_50 = (A[:, col["tol"]] == 80) & (A[:, col["area"]] >= 50)
    b127 = A[t127 & (A[:, col["area"]] >= 50)]
    keys127 = {}
    for r in b127:
        keys127.setdefault(int(r[0]), []).append(r[3:7])
    new80 = np.zeros(len(A), bool)
    for i in np.nonzero(t80_50)[0]:
        bx = A[i, 3:7]; cands = keys127.get(int(A[i, 0]), [])
        hit = any(max(0, min(bx[2], c[2]) - max(bx[0], c[0])) * max(0, min(bx[3], c[3]) - max(bx[1], c[1])) > 0.3 * (bx[2] - bx[0]) * (bx[3] - bx[1]) for c in cands)
        new80[i] = not hit
    pick("t80_only_ge50", new80, 20)
    al = root / "dataset/artificial-lunar"
    sheets = []
    for name, items in samples.items():
        tiles = []
        for s in items:
            render, _ = load_rgb(al / "images/render" / f"render{s['frame']:04d}.png")
            ground, _ = load_rgb(al / "images/ground" / f"ground{s['frame']:04d}.png")
            tiles.append(crop_tile(render, ground, [s["x0"], s["y0"], s["x1"], s["y1"]], "s2_small",
                                   f"{s['sample_id']} f{s['frame']} a={s['area']} e={s['edge']:.0f} c={s['contrast']:.0f}"))
        sheets.append(save(grid(tiles, 4, f"threshold study sample {name}: render crop | mask crop"), out / f"sample_{name}.jpg"))
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for tol in TOLS:
        ax[0].plot(AREAS, [table[f"tol{tol}_min{a}"]["per_image"] for a in AREAS], "o-", label=f"class tol {tol}")
        ax[1].plot(AREAS, [table[f"tol{tol}_min{a}"]["visible_by_proxy_rate"] for a in AREAS], "o-", label=f"class tol {tol}")
        ax[2].plot(AREAS, [table[f"tol{tol}_min{a}"]["frames_with_none"] for a in AREAS], "o-", label=f"class tol {tol}")
    for a_, t in zip(ax, ("mask components per image", "visible by proxy (edge>=100 or contrast>=25)", "frames with no component")):
        a_.set(title=t, xlabel="minimum component area (px)"); a_.grid(alpha=.3); a_.legend()
    fig.tight_layout(); fig.savefig(out / "threshold_curves.png", dpi=110); plt.close(fig)
    result = {
        "frames": nf, "components_rows": int(len(A)), "wall_s": round(wall, 1), "per_frame_extraction_s_median": round(float(np.median(per_frame_s)), 4),
        "table": table, "frames_gaining_ge50_component_with_tol80": len(gain), "of_which_task_d_no_record_frames": len(set(gain) & no_rec),
        "no_record_frames": len(no_rec), "no_record_frames_with_ge50_at_tol80": sorted(set(gain) & no_rec)[:50],
        "visibility_rule": f"edge >= {VIS_EDGE} or contrast >= {VIS_CONTRAST} (Task C calibrated proxy; misses soft-lit rocks, see scene investigation)",
        "samples": samples, "sheets": sheets,
    }
    write_json(out / "threshold_study.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    r = run(Path(__file__).resolve().parents[3], a.workers)
    print(json.dumps(json.loads((Path(__file__).resolve().parents[3] / "experiments/remediation/threshold_study/threshold_study.json").read_text()) | {"samples": "omitted"}, indent=1))
