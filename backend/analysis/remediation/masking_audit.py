"""Item 5 - Config B masking audit over the full v1.1 training export, plus alternatives (nothing is replaced).

Usage (from backend/):  uv run --no-sync python -m analysis.remediation.masking_audit --workers 8
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..annotation_validation.drawing import font, header, save, stack
from ..dataset_exploration.common import describe, write_json
from ..yolo_dataset.export import INPAINT_RADIUS, PROTECT_PX, _box_slices, build_mask

SEED = 20261013
CAPS = (0.05, 0.10, 0.15, 0.20, 0.25, 1.0)
LARGE_BOX = 1000


def _lap(gray: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))


def audit_frame(args: tuple[str, str, int, list[dict], list[dict]]) -> dict[str, Any]:
    src_p, b_p, fid, valid, review = args
    # read in colour: OpenCV's grayscale path applies the source PNG gAMA chunk (absent in config B) and fakes differences
    src3 = cv2.imread(src_p, cv2.IMREAD_COLOR); b3 = cv2.imread(b_p, cv2.IMREAD_COLOR)
    src, b = src3[..., 2], b3[..., 2]
    m, info = build_mask(valid, review)
    mb = m.astype(bool)
    changed = (src3 != b3).any(axis=2)
    prot = np.zeros_like(mb)
    for v in valid:
        prot[_box_slices(v)] = True                      # VALID box pixels (without the 2 px margin)
    ground = (~mb) & (src >= 15)
    ls, lb = _lap(src), _lap(b)
    tex_nat = float(ls[ground].mean()) if ground.any() else np.nan
    tex_fill = float(lb[mb].mean()) if mb.any() else np.nan
    tex_orig_under = float(ls[mb].mean()) if mb.any() else np.nan
    big = [r for r in review if (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) >= LARGE_BOX and m[_box_slices(r)].any()]
    merge = [r for r in review if "merge" in r["reasons"] and m[_box_slices(r)].any()]
    H = m.shape[0]
    return {"frame_id": fid, "masked_fraction": float(mb.mean()), "foreground_masked_fraction": float(mb[2 * H // 3:].mean()),
            "changed_inside_valid_boxes": int((changed & prot).sum()), "changed_outside_mask": int((changed & ~mb).sum()),
            "texture_natural": tex_nat, "texture_fill": tex_fill, "texture_original_under_mask": tex_orig_under,
            "texture_ratio_fill_vs_original": (tex_fill / tex_orig_under) if mb.any() and tex_orig_under >= 2.0 else np.nan,
            "erased_review_boxes": info["review_boxes"] - info["review_boxes_fully_protected"],
            "erased_large_review_boxes": len(big), "erased_merge_flagged_review_boxes": len(merge),
            "n_valid": len(valid)}


def noise_matched_fill(src_bgr: np.ndarray, m: np.ndarray, seed: int) -> np.ndarray:
    """Prototype: Telea inpaint + re-injected high-frequency ground texture sampled from unmasked ground (seeded)."""
    base = cv2.inpaint(src_bgr, m, INPAINT_RADIUS, cv2.INPAINT_TELEA).astype(np.float32)
    g = src_bgr[..., 0].astype(np.float32)
    resid = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    donor = (m == 0) & (g >= 15)
    ys, xs = np.nonzero(donor)
    out = base.copy()
    if len(ys) == 0:
        return base.astype(np.uint8)
    rng = np.random.default_rng(seed)
    T = 16
    my, mx = np.nonzero(m)
    for ty in range(my.min() // T * T, my.max() + 1, T):
        for tx in range(mx.min() // T * T, mx.max() + 1, T):
            blk = m[ty:ty + T, tx:tx + T]
            if not blk.any():
                continue
            for _ in range(20):                      # find a donor tile fully inside unmasked ground
                k = rng.integers(len(ys)); sy, sx = ys[k] - T // 2, xs[k] - T // 2
                if 0 <= sy and 0 <= sx and sy + T <= g.shape[0] and sx + T <= g.shape[1] and donor[sy:sy + T, sx:sx + T].all():
                    break
            else:
                continue
            h, w = blk.shape
            patch = resid[sy:sy + h, sx:sx + w]
            for c in range(3):
                out[ty:ty + h, tx:tx + w, c] += patch * blk
    out = np.clip(out, 0, 255)
    res = src_bgr.copy(); res[m > 0] = out[m > 0].astype(np.uint8)
    return res


def run(root: Path, workers: int) -> dict[str, Any]:
    out = root / "experiments/remediation/masking_audit"
    out.mkdir(parents=True, exist_ok=True)
    exp = root / "dataset/processed/yolo-rock-v1.1"
    with (exp / "manifests/split_manifest.csv").open(newline="", encoding="utf-8") as fh:
        man = {int(r["frame_id"]): r for r in csv.DictReader(fh)}
    train = sorted(f for f, r in man.items() if r["split"] == "train")
    by = defaultdict(lambda: {"VALID": [], "REVIEW_REQUIRED": []})
    with (root / "dataset/processed/curated-dataset-v1.1/annotations/annotations_all.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] in ("VALID", "REVIEW_REQUIRED") and int(r["frame_id"]) in man:
                by[int(r["frame_id"])][r["status"]].append({"x0": float(r["x0"]), "y0": float(r["y0"]), "x1": float(r["x1"]), "y1": float(r["y1"]),
                                                            "reasons": r["reasons"]})
    jobs = [(str(root / "dataset/artificial-lunar/images/render" / f"render{f:04d}.png"), str(exp / "config_B/images/train" / f"render{f:04d}.png"),
             f, by[f]["VALID"], by[f]["REVIEW_REQUIRED"]) for f in train]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(audit_frame, jobs, chunksize=16))
    with (out / "masking_audit_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader()
        w.writerows([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    mf = np.array([r["masked_fraction"] for r in rows]); nv = np.array([r["n_valid"] for r in rows])
    tr = np.array([r["texture_ratio_fill_vs_original"] for r in rows]); tr = tr[~np.isnan(tr)]
    caps = {}
    for c in CAPS:
        k = mf <= c
        caps[str(c)] = {"scenes_kept": int(k.sum()), "scenes_dropped": int((~k).sum()), "valid_boxes_kept": int(nv[k].sum()),
                        "valid_boxes_dropped": int(nv[~k].sum())}
    summary = {
        "train_images": len(rows), "masked_fraction": describe(mf),
        "images_masked_fraction_ge": {str(t): int((mf >= t).sum()) for t in (0.05, 0.10, 0.15, 0.20, 0.25)},
        "foreground_masked_fraction": describe([r["foreground_masked_fraction"] for r in rows]),
        "changed_pixels_inside_valid_boxes_total": int(sum(r["changed_inside_valid_boxes"] for r in rows)),
        "changed_pixels_outside_mask_total": int(sum(r["changed_outside_mask"] for r in rows)),
        "texture_ratio_fill_vs_original": describe(tr),
        "images_texture_ratio_lt_0.5": int((tr < 0.5).sum()),
        "erased_review_boxes_total": int(sum(r["erased_review_boxes"] for r in rows)),
        "erased_large_review_boxes_total": int(sum(r["erased_large_review_boxes"] for r in rows)),
        "images_with_erased_large_review_box": int(sum(r["erased_large_review_boxes"] > 0 for r in rows)),
        "erased_merge_flagged_review_boxes_total": int(sum(r["erased_merge_flagged_review_boxes"] for r in rows)),
        "cap_tradeoff": caps,
        "notes": "changed-inside-VALID counts pixels inside VALID boxes (no margin); the 2-px protection margin means this must be 0. "
                 "texture ratio = mean |Laplacian| of the filled pixels / of the original pixels at the same place (1 = same texture, low = smear); images whose original masked area is nearly flat (mean |Laplacian| < 2) are excluded from the ratio.",
    }
    # prototype comparison on the worst / audited frames
    worst = [int(r["frame_id"]) for r in sorted(rows, key=lambda r: -r["masked_fraction"])[:3]]
    pick = [f for f in (977, 8894, 4205, 7449, 7488) if f in by] + [f for f in worst if f not in (977, 8894, 4205, 7449, 7488)]
    tiles, proto = [], []
    for f in pick:
        src = cv2.imread(jobs[train.index(f)][0], cv2.IMREAD_COLOR)
        cur = cv2.imread(jobs[train.index(f)][1], cv2.IMREAD_COLOR)
        m, _ = build_mask(by[f]["VALID"], by[f]["REVIEW_REQUIRED"])
        nm = noise_matched_fill(src, m, SEED + f)
        mb = m.astype(bool)
        ref = max(float(_lap(src[..., 2])[mb].mean()), 2.0)
        ratio_cur = float(_lap(cur[..., 2])[mb].mean()) / ref
        ratio_new = float(_lap(nm[..., 2])[mb].mean()) / ref
        proto.append({"frame_id": f, "masked_fraction": round(float(mb.mean()), 4), "texture_ratio_telea": round(ratio_cur, 3),
                      "texture_ratio_noise_matched": round(ratio_new, 3)})
        row = stack([Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).resize((480, 320), Image.BILINEAR) for x in (src, cur, nm)], True)
        tiles.append(stack([header(row.width, [f"frame {f:04d}: original | config B v1.1 (Telea) | PROTOTYPE noise-matched fill; masked {mb.mean():.1%}; "
                                                f"texture ratio {ratio_cur:.2f} -> {ratio_new:.2f}"]), row], False))
    save(stack(tiles, False), out / "fill_comparison.jpg")
    summary["prototype_noise_matched_fill"] = {"frames": proto, "status": "PROTOTYPE ONLY - not applied to any dataset"}
    write_json(out / "masking_audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    s = run(Path(__file__).resolve().parents[3], a.workers)
    import json
    print(json.dumps(s, indent=1, default=str)[:6000])
