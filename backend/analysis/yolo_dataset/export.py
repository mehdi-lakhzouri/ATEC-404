"""Provisional YOLO export (labels = VALID boxes, class 0 'rock') and post-export validation.

Layout (Ultralytics maps .../images/<split>/x.png -> .../labels/<split>/x.txt):
    images/{val,test}, labels/{val,test}                    shared evaluation split, UNMASKED source copies
    config_A/{images,labels}/train                          conservative low-uncertainty train scenes, unmasked
    config_B/{images,labels}/train                          all train scenes, REVIEW regions inpainted (train only)
"""

from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

W, H = 720, 480
PROTECT_PX = 2         # VALID boxes dilated by this before masking -> no VALID pixel is ever altered
INPAINT_RADIUS = 5
HIGH_MASK_FRACTION = 0.25


def yolo_lines(boxes: list[dict[str, Any]]) -> str:
    out = []
    for b in boxes:
        cx = (b["x0"] + b["x1"]) / 2 / W; cy = (b["y0"] + b["y1"]) / 2 / H
        out.append(f"0 {cx:.6f} {cy:.6f} {(b['x1'] - b['x0']) / W:.6f} {(b['y1'] - b['y0']) / H:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def _box_slices(b: dict[str, Any], pad: int = 0) -> tuple[slice, slice]:
    x0, y0 = max(0, int(np.floor(b["x0"])) - pad), max(0, int(np.floor(b["y0"])) - pad)
    x1, y1 = min(W, int(np.ceil(b["x1"])) + pad), min(H, int(np.ceil(b["y1"])) + pad)
    return slice(y0, y1), slice(x0, x1)


def build_mask(valid: list[dict[str, Any]], review: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    m = np.zeros((H, W), np.uint8); protect = np.zeros((H, W), bool)
    for b in review:
        m[_box_slices(b)] = 1
    for b in valid:
        protect[_box_slices(b, PROTECT_PX)] = True
    raw = m.astype(bool)
    m[protect] = 0
    fully = sum(1 for b in review if not m[_box_slices(b)].any())
    return m, {"review_boxes": len(review), "review_boxes_fully_protected": fully,
               "review_pixels_left_unmasked_due_to_valid_overlap": int((raw & protect).sum())}


def _place(data: bytes, dst: str, prev: str | None) -> str:
    """Hard-link to an identical file of the previous export (no duplicate storage), else write the bytes."""
    if prev and os.path.exists(prev) and os.path.getsize(prev) == len(data) and Path(prev).read_bytes() == data:
        os.link(prev, dst)
        return "linked"
    Path(dst).write_bytes(data)
    return "written"


def _mask_one(args: tuple[str, str, list[dict[str, Any]], list[dict[str, Any]], str | None]) -> dict[str, Any]:
    src, dst, valid, review, prev = args
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    m, info = build_mask(valid, review)
    out = cv2.inpaint(img, m, INPAINT_RADIUS, cv2.INPAINT_TELEA) if m.any() else img
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    info["storage"] = _place(buf.tobytes(), dst, prev)
    # verification: pixels outside the mask must be untouched
    changed_outside = int((np.any(out != img, axis=2) & (m == 0)).sum())
    return {**info, "masked_fraction": round(float(m.mean()), 6), "changed_pixels_outside_mask": changed_outside}


def _copy(args: tuple[str, str, str | None]) -> str:
    src, dst, prev = args
    if prev and os.path.exists(prev) and _sha(prev) == _sha(src):
        os.link(prev, dst)
        return "linked"
    shutil.copyfile(src, dst)
    return "written"


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(root: Path, out: Path, split_of: dict[int, str], in_a: set[int], by_frame: dict[int, dict[str, list]],
          workers: int, link_from: Path | None = None) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    if out.exists():
        for sub in ("images", "labels", "config_A", "config_B"):
            shutil.rmtree(out / sub, ignore_errors=True)  # derived export only; regenerated deterministically
    targets = []
    for f, s in sorted(split_of.items()):
        name = f"render{f:04d}"
        src = str(root / "dataset" / "artificial-lunar" / "images" / "render" / f"{name}.png")
        if s in ("val", "test"):
            targets.append(("copy", f, src, out / "images" / s / f"{name}.png", out / "labels" / s / f"{name}.txt"))
        else:
            if f in in_a:
                targets.append(("copy", f, src, out / "config_A" / "images" / "train" / f"{name}.png", out / "config_A" / "labels" / "train" / f"{name}.txt"))
            targets.append(("mask", f, src, out / "config_B" / "images" / "train" / f"{name}.png", out / "config_B" / "labels" / "train" / f"{name}.txt"))
    for _, _, _, ip, lp in targets:
        ip.parent.mkdir(parents=True, exist_ok=True); lp.parent.mkdir(parents=True, exist_ok=True)
    for _, f, _, _, lp in targets:
        lp.write_text(yolo_lines(by_frame[f]["VALID"]), encoding="utf-8", newline="\n")
    prev = (lambda ip: str(link_from / ip.relative_to(out))) if link_from else (lambda ip: None)
    with ProcessPoolExecutor(workers) as ex:
        copied = list(ex.map(_copy, [(src, str(ip), prev(ip)) for k, _, src, ip, _ in targets if k == "copy"], chunksize=32))
        mjobs = [(src, str(ip), by_frame[f]["VALID"], by_frame[f]["REVIEW_REQUIRED"], prev(ip)) for k, f, src, ip, _ in targets if k == "mask"]
        minfo = list(ex.map(_mask_one, mjobs, chunksize=8))
    mframes = [f for k, f, *_ in targets if k == "mask"]
    storage = {"copies_linked": copied.count("linked"), "copies_written": copied.count("written"),
               "masked_linked": sum(m["storage"] == "linked" for m in minfo), "masked_written": sum(m["storage"] == "written" for m in minfo)}
    return dict(zip(mframes, minfo)), storage


def validate(out: Path, split_of: dict[int, str], in_a: set[int], by_frame: dict[int, dict[str, list]],
             source_hash: dict[int, str], workers: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    sets = {"val": out / "images" / "val", "test": out / "images" / "test",
            "train_A": out / "config_A" / "images" / "train", "train_B": out / "config_B" / "images" / "train"}
    expected = {"val": {f for f, s in split_of.items() if s == "val"}, "test": {f for f, s in split_of.items() if s == "test"},
                "train_A": {f for f, s in split_of.items() if s == "train" and f in in_a},
                "train_B": {f for f, s in split_of.items() if s == "train"}}
    worst_err, bad_lines, count_mismatch = 0.0, [], []
    for name, d in sets.items():
        imgs = {int(p.stem[6:]) for p in d.glob("*.png")}
        labs = {int(p.stem[6:]) for p in Path(str(d).replace("images", "labels")).glob("*.txt")}
        sizes = {Image.open(p).size for p in d.glob("*.png")}
        res[name] = {"images": len(imgs), "labels": len(labs), "image_sizes": sorted(sizes), "match_expected": imgs == expected[name],
                     "every_image_has_label": imgs == labs}
        for f in imgs:
            lines = (Path(str(d).replace("images", "labels")) / f"render{f:04d}.txt").read_text(encoding="utf-8").splitlines()
            gt = by_frame[f]["VALID"]
            if len(lines) != len(gt):
                count_mismatch.append((name, f))
            for ln, b in zip(lines, gt):
                t = ln.split()
                if len(t) != 5 or t[0] != "0":
                    bad_lines.append((name, f, ln)); continue
                cx, cy, w, h = map(float, t[1:])
                if not (0 < w <= 1 and 0 < h <= 1 and -1e-6 <= cx - w / 2 and cx + w / 2 <= 1 + 1e-6 and -1e-6 <= cy - h / 2 and cy + h / 2 <= 1 + 1e-6):
                    bad_lines.append((name, f, ln))
                err = max(abs((cx - w / 2) * W - b["x0"]), abs((cy - h / 2) * H - b["y0"]), abs((cx + w / 2) * W - b["x1"]), abs((cy + h / 2) * H - b["y1"]))
                worst_err = max(worst_err, err)
    # no two label boxes in one image may be (near-)identical
    dup = 0
    for f in set().union(*expected.values()):
        a = np.array([[b["x0"], b["y0"], b["x1"], b["y1"]] for b in by_frame[f]["VALID"]]).reshape(-1, 4)
        if len(a) > 1:
            ix = np.clip(np.minimum(a[:, None, 2], a[None, :, 2]) - np.maximum(a[:, None, 0], a[None, :, 0]), 0, None) *                  np.clip(np.minimum(a[:, None, 3], a[None, :, 3]) - np.maximum(a[:, None, 1], a[None, :, 1]), 0, None)
            ar = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
            dup += int((np.triu(ix / (ar[:, None] + ar[None, :] - ix), 1) > 0.9).sum())
    res["duplicate_label_pairs_iou_gt_0.9"] = dup
    res["label_lines_invalid"] = len(bad_lines); res["label_line_examples"] = bad_lines[:5]
    res["label_count_mismatch"] = len(count_mismatch)
    res["max_roundtrip_error_px"] = round(worst_err, 6)
    # byte-identity: eval + config A images must equal the source renders (no masking on evaluation)
    ident = [(n, f) for n in ("val", "test", "train_A") for f in expected[n]]
    with ProcessPoolExecutor(workers) as ex:
        hs = list(ex.map(_sha, [str(sets[n] / f"render{f:04d}.png") for n, f in ident], chunksize=64))
    res["unmasked_copies_byte_identical_to_source"] = all(h == source_hash[f] for (n, f), h in zip(ident, hs))
    res["unmasked_copies_checked"] = len(ident)
    return res
