"""Overlay drawing: S1 / S2 / diagnostic panels, comparison sheets and crop grids.

Colour is never the only cue: every style also has a line pattern and a text tag (see LEGEND).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import STYLE

LINE = {  # style -> (width, dash length or 0 for solid)
    "csv": (2, 0), "csv_unmatched": (2, 6), "s2_large": (1, 0), "s2_large_new": (2, 4),
    "s2_small": (1, 0), "s2_rejected": (1, 2),
}
LEGEND = [
    ("csv", "C  solid 2px   S1 CSV large-rock box"),
    ("csv_unmatched", "C? dashed 2px  S1 box with no S2 large-rock match (suspected)"),
    ("s2_large", "L  solid 1px   S2 large rock (blue mask) matching a CSV box"),
    ("s2_large_new", "L+ dashed 2px  S2 large rock without CSV box"),
    ("s2_small", "S  solid 1px   S2 additional small/medium rock (green mask)"),
    ("s2_rejected", "X  dotted 1px  S2 candidate rejected (below area threshold or dark/low-contrast)"),
]
JPEG_Q = 92


def font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without FreeType
        return ImageFont.load_default()


def _dashed(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], fill, width: int, dash: int) -> None:
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        length = max(abs(x1 - x0), abs(y1 - y0))
        if length == 0:
            continue
        n = max(int(length // dash), 1)
        for k in range(0, n, 2):
            a, b = k / n, min((k + 1) / n, 1)
            d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=fill, width=width)


def draw_box(d: ImageDraw.ImageDraw, xyxy: Iterable[float], style: str, tag: bool = True, scale: float = 1.0) -> None:
    x0, y0, x1, y1 = (v * scale for v in xyxy)
    colour, text = STYLE[style]
    width, dash = LINE[style]
    # box edges are drawn on the outer pixel boundary: x1/y1 are exclusive edges
    rect = [(x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1), (x0, y0)]
    if dash:
        _dashed(d, rect, colour, width, dash)
    else:
        d.line(rect, fill=colour, width=width)
    if tag and (x1 - x0) * (y1 - y0) >= 300 * scale * scale:
        f = font(11)
        tw = d.textlength(text, font=f)
        ty = y0 - 13 if y0 >= 13 else y0 + 1
        d.rectangle([x0, ty, x0 + tw + 3, ty + 12], fill=(0, 0, 0))
        d.text((x0 + 1, ty), text, fill=colour, font=f)


def overlay(img: np.ndarray, boxes: list[tuple[list[float], str]], tag: bool = True) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    order = {"s2_rejected": 0, "s2_small": 1, "s2_large": 2, "s2_large_new": 3, "csv": 4, "csv_unmatched": 5}
    for xyxy, style in sorted(boxes, key=lambda b: order[b[1]]):
        draw_box(d, xyxy, style, tag)
    return im


def titled(im: Image.Image, title: str) -> Image.Image:
    f = font(14)
    out = Image.new("RGB", (im.width, im.height + 22), (25, 25, 25))
    out.paste(im, (0, 22))
    ImageDraw.Draw(out).text((6, 3), title, fill=(255, 255, 255), font=f)
    return out


def legend_strip(width: int, styles: list[str] | None = None) -> Image.Image:
    items = [l for l in LEGEND if styles is None or l[0] in styles]
    f = font(13)
    cols = 2
    rows = (len(items) + cols - 1) // cols
    im = Image.new("RGB", (width, 10 + rows * 20), (245, 245, 245))
    d = ImageDraw.Draw(im)
    for k, (style, text) in enumerate(items):
        x = 10 + (k % cols) * (width // cols)
        y = 6 + (k // cols) * 20
        draw_box(d, [x, y + 2, x + 28, y + 15], style, tag=False)
        d.text((x + 36, y), text, fill=(20, 20, 20), font=f)
    return im


def stack(images: list[Image.Image], horizontal: bool, bg=(25, 25, 25)) -> Image.Image:
    if horizontal:
        w, h = sum(i.width for i in images), max(i.height for i in images)
        out = Image.new("RGB", (w, h), bg); x = 0
        for i in images:
            out.paste(i, (x, 0)); x += i.width
    else:
        w, h = max(i.width for i in images), sum(i.height for i in images)
        out = Image.new("RGB", (w, h), bg); y = 0
        for i in images:
            out.paste(i, (0, y)); y += i.height
    return out


def header(width: int, lines: list[str]) -> Image.Image:
    f = font(15)
    im = Image.new("RGB", (width, 8 + 20 * len(lines)), (10, 10, 40))
    d = ImageDraw.Draw(im)
    for k, t in enumerate(lines):
        d.text((8, 4 + 20 * k), t, fill=(255, 255, 255), font=f)
    return im


def save(im: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(path, quality=JPEG_Q, subsampling=0, optimize=True)
    else:
        im.save(path, optimize=True)
    return path.name


def crop_tile(img: np.ndarray, mask: np.ndarray, xyxy: list[float], style: str, caption: str, size: int = 190) -> Image.Image:
    """Render crop (with box) next to the ground-mask crop, upscaled with nearest neighbour."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = xyxy
    m = max(8, 0.6 * max(x1 - x0, y1 - y0))
    cx0, cy0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    cx1, cy1 = int(min(W, x1 + m)), int(min(H, y1 + m))
    s = size / max(cx1 - cx0, cy1 - cy0)
    tiles = []
    for src in (img, mask):
        c = Image.fromarray(src[cy0:cy1, cx0:cx1]).convert("RGB")
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.NEAREST)
        canvas = Image.new("RGB", (size, size), (60, 60, 60)); canvas.paste(c, (0, 0))
        draw_box(ImageDraw.Draw(canvas), [(x0 - cx0) * s, (y0 - cy0) * s, (x1 - cx0) * s, (y1 - cy0) * s], style, tag=False)
        tiles.append(canvas)
    t = stack(tiles, True)
    f = font(12)
    out = Image.new("RGB", (t.width, t.height + 18), (255, 255, 255))
    out.paste(t, (0, 18))
    ImageDraw.Draw(out).text((4, 2), caption, fill=(0, 0, 0), font=f)
    return out


def grid(tiles: list[Image.Image], cols: int, title: str) -> Image.Image:
    if not tiles:
        return header(800, [title, "(no items)"])
    tw, th = tiles[0].width, tiles[0].height
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 6) + 6), (200, 200, 200))
    for k, t in enumerate(tiles):
        out.paste(t, (6 + (k % cols) * (tw + 6), 6 + (k // cols) * (th + 6)))
    return stack([header(out.width, [title]), out], False)


def boxes_for_frame(frame_csv: list[list[float]], large: np.ndarray, small: np.ndarray, pairs: list[tuple[int, int, float]],
                    keep_large: np.ndarray, keep_small: np.ndarray, rej_large: np.ndarray, rej_small: np.ndarray) -> dict[str, Any]:
    """Assemble S1 / S2 / diagnostic box lists for one frame."""
    def xy(r):
        return [r[0], r[1], r[0] + r[2], r[1] + r[3]]
    mi = {i for i, _, _ in pairs}; mj = {j for _, j, _ in pairs}
    s1 = [(b, "csv") for b in frame_csv]
    s2 = [(xy(large[j]), "s2_large" if j in mj else "s2_large_new") for j in range(len(large)) if keep_large[j]]
    s2 += [(xy(small[j]), "s2_small") for j in range(len(small)) if keep_small[j]]
    rej = [(xy(large[j]), "s2_rejected") for j in range(len(large)) if rej_large[j] and j not in mj]
    rej += [(xy(small[j]), "s2_rejected") for j in range(len(small)) if rej_small[j]]
    diag = [(b, "csv" if i in mi else "csv_unmatched") for i, b in enumerate(frame_csv)]
    diag += [(xy(large[j]), "s2_large" if j in mj else "s2_large_new") for j in range(len(large)) if keep_large[j]]
    diag += [(xy(small[j]), "s2_small") for j in range(len(small)) if keep_small[j]] + rej
    return {"s1": s1, "s2": s2, "s2_with_rejected": s2 + rej, "diag": diag,
            "counts": {"csv": len(frame_csv), "csv_unmatched": len(frame_csv) - len(mi),
                       "s2_large_matched": sum(1 for j in range(len(large)) if keep_large[j] and j in mj),
                       "s2_large_new": sum(1 for j in range(len(large)) if keep_large[j] and j not in mj),
                       "s2_small": int(keep_small.sum()), "s2_rejected": len(rej)}}
