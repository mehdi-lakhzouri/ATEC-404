"""Before/after visuals for S2 (raw) vs S2-H (cleaned)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..annotation_validation.drawing import _dashed, crop_tile, font, grid, header, save, stack, titled
from ..annotation_validation.s2_extract import load_rgb

# style: colour, width, dash (0 solid, -1 dotted), tag
ST = {
    "raw": ((255, 220, 0), 1, 0, "R"),
    "csv": ((0, 255, 255), 2, 0, "C"),
    "valid_green": ((0, 230, 0), 1, 0, "S"),
    "valid_blue": ((60, 110, 255), 2, 0, "L"),
    "valid_split": ((255, 0, 200), 2, 0, "P"),
    "review": ((255, 140, 0), 2, 6, "?"),
    "rejected": ((255, 40, 40), 1, 2, "X"),
}
LEGEND = [("raw", "R  raw S2: CSV + every V1 mask component >= 50 px (Task B)"),
          ("csv", "C  CSV large-rock instance (kept)"), ("valid_green", "S  VALID green mask rock"),
          ("valid_blue", "L  VALID blue mask rock (no CSV box)"), ("valid_split", "P  VALID piece from watershed split"),
          ("review", "?  REVIEW_REQUIRED (not a training target)"), ("rejected", "X  REJECTED (shadow/duplicate/artefact/replaced)")]


def _box(d: ImageDraw.ImageDraw, b, style: str, tag: bool = True) -> None:
    col, w, dash, t = ST[style]
    x0, y0, x1, y1 = b
    pts = [(x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1), (x0, y0)]
    if dash:
        _dashed(d, pts, col, w, abs(dash) if dash > 0 else 2)
    else:
        d.line(pts, fill=col, width=w)
    if tag and (x1 - x0) * (y1 - y0) >= 400:
        d.rectangle([x0, max(0, y0 - 12), x0 + 10, max(0, y0 - 12) + 11], fill=(0, 0, 0))
        d.text((x0 + 1, max(0, y0 - 12)), t, fill=col, font=font(10))


def style_of(r: dict[str, Any]) -> str:
    if r["status"] == "REJECTED":
        return "rejected"
    if r["status"] == "REVIEW_REQUIRED":
        return "review"
    if r["source"] == "csv_large":
        return "csv"
    if r["source"].endswith("_split"):
        return "valid_split"
    return "valid_blue" if r["source"].startswith("mask_blue") else "valid_green"


def _legend(width: int) -> Image.Image:
    im = Image.new("RGB", (width, 10 + 20 * ((len(LEGEND) + 1) // 2)), (245, 245, 245))
    d = ImageDraw.Draw(im)
    for k, (s, t) in enumerate(LEGEND):
        x = 10 + (k % 2) * (width // 2); y = 6 + (k // 2) * 20
        _box(d, [x, y + 2, x + 28, y + 15], s, tag=False)
        d.text((x + 36, y), t, fill=(20, 20, 20), font=font(13))
    return im


def scene_sheet(root: Path, fid: int, raw_boxes: list[list[float]], recs: list[dict[str, Any]], out: Path, note: str) -> str:
    render, _ = load_rgb(root / "images" / "render" / f"render{fid:04d}.png")
    panels = []
    for name, boxes in (("A  raw S2 (before)", [(b, "raw") for b in raw_boxes]),
                        ("B  cleaned S2-H: VALID training targets (after)", [([r["x0"], r["y0"], r["x1"], r["y1"]], style_of(r)) for r in recs if r["status"] == "VALID"]),
                        ("C  decisions: VALID + REVIEW_REQUIRED + REJECTED", [([r["x0"], r["y0"], r["x1"], r["y1"]], style_of(r)) for r in recs
                                                                              if not (r["status"] == "REJECTED" and r["reasons"] == "duplicate_of_csv_instance")])):
        im = Image.fromarray(render).convert("RGB")
        d = ImageDraw.Draw(im)
        order = {"rejected": 0, "raw": 1, "valid_green": 2, "valid_blue": 3, "review": 4, "valid_split": 5, "csv": 6}
        for b, s in sorted(boxes, key=lambda x: order[x[1]]):
            _box(d, b, s)
        panels.append(titled(im, f"{name} [{len(boxes)}]"))
    row = stack(panels, True)
    n = {s: sum(1 for r in recs if r["status"] == s) for s in ("VALID", "REVIEW_REQUIRED", "REJECTED")}
    sheet = stack([header(row.width, [f"frame {fid:04d} | {note} | raw S2 boxes {len(raw_boxes)} -> VALID {n['VALID']}, REVIEW {n['REVIEW_REQUIRED']}, REJECTED {n['REJECTED']} | native 720x480"]),
                   row, _legend(row.width)], False)
    return save(sheet, out / f"scene_{fid:04d}.jpg")


def split_grid(root: Path, items: list[tuple[int, dict[str, Any], list[dict[str, Any]]]], out: Path, title: str, name: str) -> str:
    tiles = []
    for fid, parent, pieces in items:
        render, _ = load_rgb(root / "images" / "render" / f"render{fid:04d}.png")
        ground, _ = load_rgb(root / "images" / "ground" / f"ground{fid:04d}.png")
        x0, y0, x1, y1 = parent["x0"], parent["y0"], parent["x1"], parent["y1"]
        m = max(8, 0.4 * max(x1 - x0, y1 - y0))
        cx0, cy0, cx1, cy1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(720, x1 + m)), int(min(480, y1 + m))
        s = 170 / max(cx1 - cx0, cy1 - cy0)
        ims = []
        for src, boxes in ((render, [(parent, "raw")]), (render, [(p, "valid_split" if p["status"] == "VALID" else style_of(p)) for p in pieces]), (ground, [])):
            c = Image.fromarray(src[cy0:cy1, cx0:cx1]).convert("RGB")
            c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.NEAREST)
            can = Image.new("RGB", (170, 170), (60, 60, 60)); can.paste(c, (0, 0))
            d = ImageDraw.Draw(can)
            for b, st in boxes:
                _box(d, [(b["x0"] - cx0) * s, (b["y0"] - cy0) * s, (b["x1"] - cx0) * s, (b["y1"] - cy0) * s], st, tag=False)
            ims.append(can)
        t = stack(ims, True)
        cap = Image.new("RGB", (t.width, t.height + 16), (255, 255, 255)); cap.paste(t, (0, 16))
        ImageDraw.Draw(cap).text((3, 1), f"f{fid:04d} {parent['source']} a={parent['area']} -> {len(pieces)} pieces "
                                         f"({sum(p['status'] == 'VALID' for p in pieces)} VALID)", fill=(0, 0, 0), font=font(11))
        tiles.append(cap)
    return save(grid(tiles, 3, title + "  [before | after (pieces) | ground mask]"), out / name)


STYLE_MAP = {"csv": "csv", "valid_green": "s2_small", "valid_blue": "s2_large_new", "valid_split": "s2_large_new",
             "review": "csv_unmatched", "rejected": "s2_rejected", "raw": "s2_small"}


def review_grids(root: Path, sample: list[dict[str, Any]], out: Path) -> list[str]:
    tiles = []
    for r in sample:
        render, _ = load_rgb(root / "images" / "render" / f"render{r['frame_id']:04d}.png")
        ground, _ = load_rgb(root / "images" / "ground" / f"ground{r['frame_id']:04d}.png")
        tiles.append(crop_tile(render, ground, [r["x0"], r["y0"], r["x1"], r["y1"]], STYLE_MAP[style_of(r)],
                               f"{r['review_id']} f{r['frame_id']:04d} {r['stratum']}"))
    files = []
    for s in range(0, len(tiles), 24):
        files.append(save(grid(tiles[s:s + 24], 4, f"Task C independent review sample {s // 24 + 1}: left = render crop + box, right = ground mask"),
                          out / f"review_sample_{s // 24 + 1:02d}.jpg"))
    return files
