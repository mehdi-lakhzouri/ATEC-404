"""Item 3 - priority-scene investigation sheets and evidence tables (read-only, curated-dataset-v1.1).

Usage (from backend/):  uv run --no-sync python -m analysis.remediation.scene_investigation
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..annotation_validation.drawing import font, grid, header, save, stack
from ..annotation_validation.s2_extract import class_masks, load_rgb
from ..dataset_cleaning.worker import C, region_features

FRAMES = (977, 8894, 4357, 4465, 5311, 6352, 6864, 4138)
CITED = {977: ["0977-mask_green-103", "0977-mask_green-106", "0977-mask_green-107"],
         8894: ["8894-mask_green-038", "8894-mask_green-058", "8894-mask_green-064"],
         6352: ["6352-csv_large-000", "6352-csv_large-001", "6352-mask_blue-000"]}
COL = {"VALID_csv": (0, 229, 255), "VALID": (34, 216, 34), "REVIEW_REQUIRED": (255, 154, 31), "REJECTED": (255, 48, 48)}


def _style(r: dict[str, Any]) -> tuple[tuple[int, int, int], int]:
    if r["status"] == "VALID":
        return (COL["VALID_csv"] if r["source"] == "csv_large" else COL["VALID"]), 2
    return COL[r["status"]], 1


def _draw(img: np.ndarray, recs: list[dict[str, Any]], hide_dups: bool = True, tags: set[str] | None = None) -> Image.Image:
    im = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(im)
    for r in sorted(recs, key=lambda r: {"REJECTED": 0, "REVIEW_REQUIRED": 1, "VALID": 2}[r["status"]]):
        if hide_dups and r["status"] == "REJECTED" and r["reasons"] in ("duplicate_of_csv_instance", "replaced_by_split_pieces"):
            continue
        c, w = _style(r)
        d.rectangle([float(r["x0"]), float(r["y0"]), float(r["x1"]) - 1, float(r["y1"]) - 1], outline=c, width=w)
        if tags and r["ann_id"] in tags:
            d.rectangle([float(r["x0"]), float(r["y0"]) - 12, float(r["x0"]) + 26, float(r["y0"])], fill=(0, 0, 0))
            d.text((float(r["x0"]) + 1, float(r["y0"]) - 12), r["ann_id"].split("-")[-1], fill=c, font=font(10))
    return im


def _crop(img: np.ndarray, box: list[float], size: int = 200) -> Image.Image:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    m = max(10, 0.35 * max(x1 - x0, y1 - y0))
    cx0, cy0, cx1, cy1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
    c = Image.fromarray(img[cy0:cy1, cx0:cx1]).convert("RGB")
    s = size / max(c.width, c.height)
    c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.NEAREST)
    can = Image.new("RGB", (size, size), (60, 60, 60)); can.paste(c, (0, 0))
    ImageDraw.Draw(can).rectangle([(x0 - cx0) * s, (y0 - cy0) * s, (x1 - cx0) * s - 1, (y1 - cy0) * s - 1], outline=(255, 0, 255), width=2)
    return can


def small_components(root: Path, fid: int, min_area: int = 10) -> list[dict[str, Any]]:
    """All V1_cyan components >= min_area with render-visibility features (for frames with no records)."""
    render, _ = load_rgb(root / "dataset/artificial-lunar/images/render" / f"render{fid:04d}.png")
    ground, _ = load_rgb(root / "dataset/artificial-lunar/images/ground" / f"ground{fid:04d}.png")
    gray = render[..., 0]
    grad = cv2.magnitude(*(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32), cv2.CV_32F, dx, dy, ksize=3) for dx, dy in ((1, 0), (0, 1))))
    cm = class_masks(ground, cyan_is_blue=True)
    bg = ~(cm["large"] | cm["small"]) & ~cm["sky"]
    out = []
    for cls in ("large", "small"):
        n, lab, st, _ = cv2.connectedComponentsWithStats(cm[cls].astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < min_area:
                continue
            ox, oy = max(0, x - 2), max(0, y - 2)
            m = (lab[oy:min(480, y + h + 2), ox:min(720, x + w + 2)] == i).astype(np.uint8)
            f = region_features(m, ox, oy, gray, grad, bg)
            out.append({"class": cls, "x0": int(f[0]), "y0": int(f[1]), "x1": int(f[2]), "y1": int(f[3]), "area": int(a),
                        "edge_mean": round(float(f[C["edge_mean"]]), 1),
                        "contrast": None if np.isnan(f[C["contrast"]]) else round(float(f[C["contrast"]]), 1)})
    return out


def run(root: Path) -> dict[str, Any]:
    cur = root / "dataset/processed/curated-dataset-v1.1"
    out = root / "experiments/remediation/scene_investigation"
    out.mkdir(parents=True, exist_ok=True)
    with (cur / "annotations/annotations_all.csv").open(newline="", encoding="utf-8") as fh:
        recs = [r for r in csv.DictReader(fh) if int(r["frame_id"]) in FRAMES]
    with (root / "dataset/processed/yolo-rock-v1.1/manifests/split_manifest.csv").open(newline="", encoding="utf-8") as fh:
        man = {int(r["frame_id"]): r for r in csv.DictReader(fh)}
    by = defaultdict(list)
    for r in recs:
        by[int(r["frame_id"])].append(r)
    summary: dict[str, Any] = {}
    evidence = []
    for fid in FRAMES:
        rs = by.get(fid, [])
        render, _ = load_rgb(root / "dataset/artificial-lunar/images/render" / f"render{fid:04d}.png")
        ground, _ = load_rgb(root / "dataset/artificial-lunar/images/ground" / f"ground{fid:04d}.png")
        split = man[fid]["split"]
        bpath = root / "dataset/processed/yolo-rock-v1.1/config_B/images/train" / f"render{fid:04d}.png"
        cited = set(CITED.get(fid, []))
        review = [r for r in rs if r["status"] == "REVIEW_REQUIRED"]
        # suspicious = audit-cited + 9 largest REVIEW boxes (deterministic)
        sus = [r for r in rs if r["ann_id"] in cited]
        sus += [r for r in sorted(review, key=lambda r: (-float(r["area"] or 0), r["ann_id"])) if r["ann_id"] not in cited][:max(0, 9 - len(sus))]
        top = [_draw(render, rs, tags={r["ann_id"] for r in sus}), Image.fromarray(ground)]
        if bpath.exists():
            top.append(Image.fromarray(np.asarray(Image.open(bpath).convert("RGB"))))
        row = stack(top, True)
        tiles = []
        for r in sus:
            b = [float(r[k]) for k in ("x0", "y0", "x1", "y1")]
            t = stack([_crop(render, b), _crop(ground, b)], True)
            cap = Image.new("RGB", (t.width, t.height + 30), (255, 255, 255)); cap.paste(t, (0, 30))
            dd = ImageDraw.Draw(cap)
            dd.text((3, 1), f"{r['ann_id']} {r['status']} a={r['area']}", fill=(0, 0, 0), font=font(11))
            dd.text((3, 15), f"edge {r['edge_mean'] or '-'} contrast {r['contrast'] or '-'} solid {r['solidity'] or '-'} | {r['reasons'][:40]}", fill=(80, 80, 80), font=font(10))
            tiles.append(cap)
        comps = []
        if not rs:
            comps = small_components(root, fid)
            im = Image.fromarray(render).convert("RGB"); d = ImageDraw.Draw(im)
            for c in comps:
                col = (255, 60, 60) if c["area"] < 25 else (255, 200, 0) if c["area"] < 50 else (0, 255, 120)
                d.rectangle([c["x0"] - 1, c["y0"] - 1, c["x1"], c["y1"]], outline=col, width=1)
            row = stack([im, Image.fromarray(ground)], True)
            for c in sorted(comps, key=lambda c: -c["area"])[:9]:
                b = [c["x0"], c["y0"], c["x1"], c["y1"]]
                t = stack([_crop(render, b), _crop(ground, b)], True)
                cap = Image.new("RGB", (t.width, t.height + 16), (255, 255, 255)); cap.paste(t, (0, 16))
                ImageDraw.Draw(cap).text((3, 1), f"{c['class']} a={c['area']} edge {c['edge_mean']} contrast {c['contrast']}", fill=(0, 0, 0), font=font(10))
                tiles.append(cap)
        legend = ("cyan VALID csv | green VALID mask | orange REVIEW | red REJECTED (duplicates/split parents hidden)" if rs else
                  "NO RECORDS: all mask components >= 10 px drawn: red 10-24 px, yellow 25-49 px, green >= 50 px (none expected)")
        sheet = stack([header(row.width, [f"frame {fid:04d} | split {split} | {dict(Counter(r['status'] for r in rs)) or 'no records'} | panels: render+boxes | ground mask" + (" | config B" if bpath.exists() else ""), legend]),
                       row] + ([grid(tiles, 3, "zoomed: render crop | mask crop (magenta = box)")] if tiles else []), False)
        save(sheet, out / f"scene_{fid:04d}.jpg")
        summary[str(fid)] = {"split": split, "status_counts": dict(Counter(r["status"] for r in rs)),
                             "review_reasons": dict(Counter(x for r in review for x in r["reasons"].split(";") if x)),
                             "config_B_masked_fraction": man[fid].get("config_B_masked_fraction", ""),
                             "small_components": ({"n": len(comps), "lt25": sum(c["area"] < 25 for c in comps),
                                                   "25_49": sum(25 <= c["area"] < 50 for c in comps), "ge50": sum(c["area"] >= 50 for c in comps),
                                                   "max_area": max((c["area"] for c in comps), default=0)} if not rs else None)}
        for r in sus:
            evidence.append({"frame_id": fid, "ann_id": r["ann_id"], "status": r["status"], "reasons": r["reasons"], "source": r["source"],
                             "area": r["area"], "solidity": r["solidity"], "edge_mean": r["edge_mean"], "contrast": r["contrast"],
                             "in_mean": r["in_mean"], "dark_frac": r["dark_frac"], "cited_by_ai_audit": r["ann_id"] in cited})
        for c in sorted(comps, key=lambda c: -c["area"])[:9]:
            evidence.append({"frame_id": fid, "ann_id": f"component_{c['class']}_{c['x0']}_{c['y0']}", "status": "NO_RECORD (below 50 px)",
                             "reasons": "", "source": f"mask_{c['class']}", "area": c["area"], "solidity": "", "edge_mean": c["edge_mean"],
                             "contrast": c["contrast"], "in_mean": "", "dark_frac": "", "cited_by_ai_audit": False})
    with (out / "evidence_boxes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(evidence[0]), lineterminator="\n"); w.writeheader(); w.writerows(evidence)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[3]), indent=1))
