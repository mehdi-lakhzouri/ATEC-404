"""NASA POLAR review: Terrain 06 negative-set check (B6) and dark-exposure label reliability (B7).

Reuses Task A outputs (strict six-field matching, image index) instead of re-deriving them.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..dataset_exploration.common import LOG, Paths, describe
from ..dataset_exploration.polar_sim import IMG_H, IMG_W, parse_det_file
from .config import BOX_VISIBLE_RANGE, ENH_GAMMA, ENH_PERCENTILES, FIXED_LINEAR_SHIFT, NOISE_K
from .drawing import font, grid, header, save, stack

ROCK, SHADOW = 0, 1
SEG_COL = {0: (128, 128, 128), 1: (40, 200, 40), 2: (255, 43, 214), 3: (43, 160, 255)}


def fixed8(raw: np.ndarray) -> np.ndarray:
    """Fixed linear 12-bit -> 8-bit: raw >> 4, clipped (documented, image-independent)."""
    return np.clip(raw.astype(np.int32) >> FIXED_LINEAR_SHIFT, 0, 255).astype(np.uint8)


def enhanced8(raw: np.ndarray) -> np.ndarray:
    """Display-only contrast enhancement: per-image percentile stretch then gamma."""
    a = raw.astype(np.float32)
    lo, hi = np.percentile(a, ENH_PERCENTILES)
    x = np.clip((a - lo) / max(hi - lo, 1.0), 0, 1) ** ENH_GAMMA
    return (x * 255).astype(np.uint8)


def noise_sigma(raw: np.ndarray) -> float:
    d = np.abs(np.diff(raw.astype(np.float32), axis=1))
    return float(1.4826 * np.median(d) / np.sqrt(2))


def box_px(b: dict[str, Any]) -> list[int]:
    x0 = int(max(0, (b["cx"] - b["w"] / 2) * IMG_W)); y0 = int(max(0, (b["cy"] - b["h"] / 2) * IMG_H))
    x1 = int(min(IMG_W, np.ceil((b["cx"] + b["w"] / 2) * IMG_W))); y1 = int(min(IMG_H, np.ceil((b["cy"] + b["h"] / 2) * IMG_H)))
    return [x0, y0, x1, y1]


def _rng(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    p5, p95 = np.percentile(a, [5, 95])
    return float(p95 - p5)


def box_visibility(args: tuple[str, str, str]) -> list[dict[str, Any]]:
    img_path, label_path, key = args
    raw = np.asarray(Image.open(img_path))
    f8, e8 = fixed8(raw), enhanced8(raw)
    sig = noise_sigma(raw)
    out = []
    for k, b in enumerate(parse_det_file(Path(label_path), 2)["boxes"]):
        if b["cls"] != ROCK or b["flags"]:
            continue
        x0, y0, x1, y1 = box_px(b)
        rr, fr, er = _rng(raw[y0:y1, x0:x1].astype(np.float32)), _rng(f8[y0:y1, x0:x1]), _rng(e8[y0:y1, x0:x1])
        if fr >= BOX_VISIBLE_RANGE:
            v = "visible_fixed_linear"
        elif er >= BOX_VISIBLE_RANGE and rr >= NOISE_K * sig:
            v = "visible_only_enhanced"
        else:
            v = "not_identifiable"
        out.append({"key": key, "box_index": k, "raw_range": rr, "fixed_range": fr, "enh_range": er, "noise_sigma": sig, "auto_visibility": v})
    return out


def _load_task_a(paths: Paths) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    a = paths.project_root / "experiments" / "dataset-exploration"
    match = json.loads((a / "annotation_matching_report.json").read_text(encoding="utf-8"))
    index = {m["rel"]: m for m in json.loads((a / "nasa_polar_analysis.json").read_text(encoding="utf-8"))["image_index"]}
    return match, index


def exposure_review(paths: Paths, out: Path, workers: int) -> dict[str, Any]:
    match, index = _load_task_a(paths)
    det = match["det"]["unambiguous_matches"]
    root, psroot = paths.nasa_polar, paths.polar_sim
    args = [(str(root / r["image"]), str(psroot / r["label"]), r["key"]) for r in det]
    LOG.info("B7: per-box visibility over %d matched NASA images", len(args))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        boxes = [b for res in ex.map(box_visibility, args, chunksize=4) for b in res]
    meta = {r["key"]: index[r["image"]] | {"image": r["image"], "label": r["label"]} for r in det}
    by_exp = defaultdict(Counter)
    by_terr_exp = defaultdict(Counter)
    for b in boxes:
        m = meta[b["key"]]
        by_exp[m["exposure"]][b["auto_visibility"]] += 1
        by_terr_exp[f"{m['terrain']}_{m['exposure']}"][b["auto_visibility"]] += 1
    frac = lambda c: {k: round(v / max(sum(c.values()), 1), 4) for k, v in sorted(c.items())} | {"rock_boxes": sum(c.values())}  # noqa: E731

    # exposure-selection policy: per bracket, the 512-2048 ms image with the largest well-exposed fraction
    brackets = defaultdict(list)
    for r in det:
        m = meta[r["key"]]
        brackets[(m["terrain"], m["pos"], m["light"], m["sun"], m["cam"])].append(m)
    chosen = {}
    for k, ms in sorted(brackets.items()):
        c = [m for m in ms if m["exposure"] in ("0512", "1024", "2048")]
        if c:
            best = max(c, key=lambda m: (m["well_exposed_frac"], -int(m["exposure"])))
            chosen[k] = best
    chosen_keys = {f"{m['terrain']}_{m['pos']}_{m['light']}_{m['sun']}_{m['cam']}_{m['exposure']}" for m in chosen.values()}
    pol = Counter(b["auto_visibility"] for b in boxes if b["key"] in chosen_keys)
    policy = {
        "rule": "for each (terrain, position, light, sun, camera) bracket with a detection label, take the 512/1024/2048 ms image "
                "with the largest fraction of pixels in 600..3800 (Task A well_exposed_frac); ties -> shorter exposure",
        "selected_images": len(chosen), "selected_by_terrain": dict(Counter(k[0] for k in chosen)),
        "selected_exposures": dict(Counter(m["exposure"] for m in chosen.values())),
        "rock_box_visibility_in_selection": frac(pol),
        "selected_image_list": sorted(m["image"] for m in chosen.values()),
        "excluded": "brackets without a labelled 512-2048 ms image (NoSun light-off sets are all dark)",
    }
    sheets = _exposure_sheets(paths, out, det, meta)
    return {
        "visualisation": {"fixed_linear": f"raw >> {FIXED_LINEAR_SHIFT} (= /16), clip 0..255",
                          "enhanced": f"per-image percentile stretch {ENH_PERCENTILES} then gamma {ENH_GAMMA:.4f}",
                          "auto_visibility_rule": f"visible_fixed_linear: p95-p5 inside box >= {BOX_VISIBLE_RANGE} grey levels in fixed view; "
                                                  f"visible_only_enhanced: same in enhanced view AND raw p95-p5 >= {NOISE_K} x sigma_noise "
                                                  "(sigma from MAD of horizontal pixel differences); else not_identifiable. "
                                                  "This is a contrast proxy; a box with texture but no rock would also pass."},
        "rock_boxes_evaluated": len(boxes),
        "by_exposure": {e: frac(c) for e, c in sorted(by_exp.items())},
        "by_terrain_exposure": {k: frac(c) for k, c in sorted(by_terr_exp.items())},
        "policy": policy, "sheets": sheets,
    }


def _thumb(a8: np.ndarray, boxes: list[dict[str, Any]], scale: float, title: str) -> Image.Image:
    im = Image.fromarray(a8).convert("RGB").resize((int(IMG_W * scale), int(IMG_H * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for b in boxes:
        x0, y0, x1, y1 = (v * scale for v in box_px(b))
        col = (255, 43, 214) if b["cls"] == ROCK else (43, 240, 255)
        d.rectangle([x0, y0, x1, y1], outline=col, width=1 if b["cls"] == SHADOW else 2)
    out = Image.new("RGB", (im.width, im.height + 18), (20, 20, 20))
    out.paste(im, (0, 18))
    ImageDraw.Draw(out).text((4, 2), title, fill=(255, 255, 255), font=font(12))
    return out


PICKS = [("01", "A", "off", "30", "L"), ("01", "B", "on", "180", "L"), ("01", "C", "off", "270", "R"),
         ("11", "A", "off", "350", "L"), ("11", "B", "on", "30", "R"), ("11", "C", "off", "180", "L")]


def _exposure_sheets(paths: Paths, out: Path, det: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    root, psroot = paths.nasa_polar, paths.polar_sim
    by_br = defaultdict(list)
    for r in det:
        m = meta[r["key"]]
        by_br[(m["terrain"], m["pos"], m["light"], m["sun"], m["cam"])].append(m)
    files = []
    for pk in PICKS:
        ms = sorted(by_br.get(pk, []), key=lambda m: int(m["exposure"]))
        if not ms:
            LOG.warning("exposure pick %s has no labelled images", pk)
            continue
        rows = {"fixed": [], "enh": []}
        raws = {}
        for m in ms:
            raw = np.asarray(Image.open(root / m["image"]))
            raws[m["exposure"]] = raw
            boxes = parse_det_file(psroot / m["label"], 2)["boxes"]
            rows["fixed"].append(_thumb(fixed8(raw), boxes, 0.25, f"{m['exposure']} ms fixed >>4"))
            rows["enh"].append(_thumb(enhanced8(raw), boxes, 0.25, f"{m['exposure']} ms enhanced"))
        name = "T{}_{}_L{}_sun{}_Cam{}".format(*pk)
        sheet = stack([header(sum(t.width for t in rows["fixed"]), [f"{name}: same scene across exposures; top = fixed linear, bottom = enhanced; magenta = rock, cyan = shadow (POLAR-Sim)"]),
                       stack(rows["fixed"], True), stack(rows["enh"], True)], False)
        files.append(save(sheet, out / "nasa_exposure_review" / f"bracket_{name}.jpg"))
        # zoomed crops of the 4 largest rock boxes across exposures (both modes)
        boxes = [b for b in parse_det_file(psroot / ms[-1]["label"], 2)["boxes"] if b["cls"] == ROCK]
        boxes = sorted(boxes, key=lambda b: -b["w"] * b["h"])[:4]
        tiles = []
        for bi, b in enumerate(boxes):
            x0, y0, x1, y1 = box_px(b)
            mg = int(0.5 * max(x1 - x0, y1 - y0)) + 8
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(IMG_W, x1 + mg), min(IMG_H, y1 + mg)
            for e, raw in raws.items():
                for mode, fn in (("fix", fixed8), ("enh", enhanced8)):
                    c = Image.fromarray(fn(raw)[cy0:cy1, cx0:cx1]).convert("RGB")
                    s = 110 / max(c.width, c.height)
                    c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.NEAREST)
                    t = Image.new("RGB", (110, 126), (60, 60, 60)); t.paste(c, (0, 16))
                    d = ImageDraw.Draw(t)
                    d.rectangle([(x0 - cx0) * s, 16 + (y0 - cy0) * s, (x1 - cx0) * s, 16 + (y1 - cy0) * s], outline=(255, 43, 214))
                    d.rectangle([0, 0, 110, 15], fill=(255, 255, 255))
                    d.text((2, 1), f"R{bi} {e} {mode}", fill=(0, 0, 0), font=font(11))
                    tiles.append(t)
        files.append(save(grid(tiles, 2 * len(raws), f"{name}: 4 largest rock boxes (rows: R0..R3 x fix/enh pairs per exposure)"),
                          out / "nasa_exposure_review" / f"crops_{name}.jpg"))
    return files


def terrain06_review(paths: Paths, out: Path) -> dict[str, Any]:
    match, index = _load_task_a(paths)
    root = paths.nasa_polar
    t06 = [m for m in index.values() if m["terrain"] == "06"]
    sel = []
    for pos in "ABC":
        for cam in "LR":
            for cond, light, sun in (("sunlit_30", "off", "30"), ("rover_light_only", "on", "no"), ("low_light_no_sun_no_light", "off", "no")):
                c = [m for m in t06 if (m["pos"], m["cam"], m["light"], m["sun"]) == (pos, cam, light, sun)]
                if c:
                    best = max(c, key=lambda m: (m["well_exposed_frac"], -int(m["exposure"])))
                    sel.append({"condition": cond, **best})
    images = []
    for pos in "ABC":
        tiles = []
        for m in [s for s in sel if s["pos"] == pos]:
            raw = np.asarray(Image.open(root / m["rel"]))
            full = Image.fromarray(enhanced8(raw)).convert("RGB")
            name = Path(m["rel"]).stem
            fp = out / "nasa_terrain06" / "full" / f"T06_Pos{pos}_{m['cam']}_{m['condition']}_{name}.jpg"
            save(full, fp)
            images.append({"image": m["rel"], "position": pos, "camera": m["cam"], "condition": m["condition"],
                           "exposure": m["exposure"], "well_exposed_frac": m["well_exposed_frac"], "enhanced_full_res": fp.relative_to(out).as_posix()})
            t = full.resize((IMG_W // 4, IMG_H // 4), Image.BILINEAR)
            cap = Image.new("RGB", (t.width, t.height + 18), (20, 20, 20)); cap.paste(t, (0, 18))
            ImageDraw.Draw(cap).text((4, 2), f"Pos{pos} Cam{m['cam']} {m['condition']} {m['exposure']}ms we={m['well_exposed_frac']:.2f}",
                                     fill=(255, 255, 255), font=font(12))
            tiles.append(cap)
        save(grid(tiles, 3, f"Terrain06_Smooth Pos{pos}: enhanced view (percentile {ENH_PERCENTILES} + gamma)"),
             out / "nasa_terrain06" / f"T06_Pos{pos}_overview.jpg")
    # the six segmentation files of terrain 06
    seg_rows = [r for r in match["seg"]["unambiguous_matches"] if r["key"].startswith("06_")]
    seg_info, tiles = [], []
    for r in seg_rows:
        lp = paths.polar_sim / r["label"]
        classes = Counter()
        raw = np.asarray(Image.open(root / r["image"]))
        im = Image.fromarray(enhanced8(raw)).convert("RGB")
        d = ImageDraw.Draw(im, "RGBA")
        for line in lp.read_text(encoding="utf-8").splitlines():
            tok = line.split()
            if len(tok) < 7:
                continue
            c = int(tok[0]); classes[c] += 1
            pts = (np.array([float(t) for t in tok[1:]]).reshape(-1, 2) * [IMG_W, IMG_H]).tolist()
            d.polygon([tuple(p) for p in pts], outline=SEG_COL.get(c, (255, 0, 0)) + (255,), width=4)
        seg_info.append({"label": r["label"], "image": r["image"], "polygons_per_class": {str(k): v for k, v in sorted(classes.items())}})
        t = im.resize((IMG_W // 4, IMG_H // 4), Image.BILINEAR)
        cap = Image.new("RGB", (t.width, t.height + 18), (20, 20, 20)); cap.paste(t, (0, 18))
        ImageDraw.Draw(cap).text((4, 2), Path(r["label"]).name + " " + str(dict(classes)), fill=(255, 255, 255), font=font(12))
        tiles.append(cap)
    save(grid(tiles, 3, "Terrain06 segmentation files (grey=background, green=ground; classes 2 rocks / 3 shadows would be magenta/blue)"),
         out / "nasa_terrain06" / "T06_segmentation_files.jpg")
    det_dir = paths.polar_sim / "Labels" / "Terrain06"
    return {"selected_images": images, "segmentation_files": seg_info,
            "detection_label_folder_exists": det_dir.exists(),
            "terrain06_nasa_images_total": len(t06)}
