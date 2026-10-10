"""A2 - Artificial Lunar Landscape analysis (renders, masks, CSV boxes, anomaly lists)."""

from __future__ import annotations

import csv
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .common import LOG, SEED, ErrorLog, Paths, describe, greedy_match, iou_xyxy, save_fig

KINDS = ("render", "ground", "clean")
ANOMALY_FILES = (
    "cam_anomaly_IDs.txt",
    "ground_facing_IDs.txt",
    "mismatch_IDs.txt",
    "shadow_IDs.txt",
    "top200_largerocks_IDs.txt",
)
MATCH_IOU = 0.5


# --------------------------------------------------------------------------- per-frame worker
def _colour_masks(a: np.ndarray, tol: int = 127) -> dict[str, np.ndarray]:
    r, g, b = (a[..., i].astype(np.int16) for i in range(3))
    return {
        "red": (r > tol) & (g <= tol) & (b <= tol),
        "green": (g > tol) & (r <= tol) & (b <= tol),
        "blue": (b > tol) & (r <= tol) & (g <= tol),
    }


def _components(mask: np.ndarray) -> np.ndarray:
    """(N,5) int array x0,y0,w,h,area of 8-connected components."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return stats[1:n].astype(np.int64)


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in bits), 2)


def _iou_masks(a: np.ndarray, b: np.ndarray) -> float | None:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else None


def analyse_frame(args: tuple[str, int]) -> dict[str, Any]:
    root, fid = args
    out: dict[str, Any] = {"id": fid, "files": {}, "errors": []}
    arrays: dict[str, np.ndarray] = {}
    for kind in KINDS:
        p = Path(root) / "images" / kind / f"{kind}{fid:04d}.png"
        info: dict[str, Any] = {"exists": p.exists()}
        if p.exists():
            try:
                with Image.open(p) as im:
                    info.update(mode=im.mode, size=list(im.size), bytes=p.stat().st_size)
                    a = np.asarray(im.convert("RGB"))
                arrays[kind] = a
                info["decoded"] = True
            except Exception as e:  # corrupt/truncated PNG
                info["decoded"] = False
                out["errors"].append(f"{kind}: {e!r}")
        out["files"][kind] = info
    if len(arrays) < 3:
        return out

    r = arrays["render"]
    gray = r[..., 0]
    out["render_channel_max_diff"] = int(
        max(np.abs(r[..., 0].astype(int) - r[..., 1]).max(), np.abs(r[..., 0].astype(int) - r[..., 2]).max())
    )
    gm = _colour_masks(arrays["ground"])
    cm = _colour_masks(arrays["clean"])
    clean_vals = np.unique(arrays["clean"])
    out["clean_is_binary"] = bool(np.isin(clean_vals, [0, 255]).all())
    npx = gray.size
    non_sky = ~gm["red"]
    out["render"] = {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "dark_frac": float((gray < 20).mean()),
        "dark_frac_non_sky": float((gray[non_sky] < 20).mean()) if non_sky.any() else None,
        "sat_frac": float((gray >= 250).mean()),
        "dhash": _dhash(gray),
    }
    out["ground"] = {k: float(v.sum() / npx) for k, v in gm.items()}
    out["clean"] = {k: float(v.sum() / npx) for k, v in cm.items()}
    out["ground_unclassified_frac"] = float(1 - sum(out["ground"].values()) - (np.all(arrays["ground"] < 128, axis=2)).mean())
    # Scene-consistency evidence between the three representations.
    out["consistency"] = {
        "sky_iou_ground_vs_clean": _iou_masks(gm["red"], cm["red"]),
        "blue_iou_ground_vs_clean": _iou_masks(gm["blue"], cm["blue"]),
        "render_mean_in_ground_sky": float(gray[gm["red"]].mean()) if gm["red"].any() else None,
        "render_mean_outside_sky": float(gray[non_sky].mean()) if non_sky.any() else None,
        "ground_sky_px_render_dark_frac": float((gray[gm["red"]] <= 5).mean()) if gm["red"].any() else None,
        "render_black_px_in_ground_sky_frac": float(gm["red"][gray <= 5].mean()) if (gray <= 5).any() else None,
    }
    cb = _components(cm["blue"])
    gb = _components(gm["blue"])
    cg = _components(cm["green"])
    gg = _components(gm["green"])
    out["components"] = {
        "clean_blue": cb[:, :5].tolist(),
        "ground_blue": gb[:, :5].tolist(),
        "clean_green_n": int(len(cg)),
        "clean_green_n_ge50px": int((cg[:, 4] >= 50).sum()) if len(cg) else 0,
        "clean_green_areas_ge50": sorted(cg[cg[:, 4] >= 50, 4].tolist()) if len(cg) else [],
        "ground_green_n": int(len(gg)),
        "ground_green_n_ge50px": int((gg[:, 4] >= 50).sum()) if len(gg) else 0,
    }
    return out


# --------------------------------------------------------------------------- CSV / lists
def load_csv(path: Path, errors: ErrorLog) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for ln, rec in enumerate(reader, start=2):
            try:
                if len(rec) != len(header):
                    raise ValueError(f"{len(rec)} fields")
                fr = int(rec[0])
                x, y, w, h = (float(v) for v in rec[1:5])
                if any(np.isnan([x, y, w, h])):
                    raise ValueError("NaN")
                rows.append({"line": ln, "frame": fr, "x": x, "y": y, "w": w, "h": h})
            except ValueError as e:
                bad.append({"line": ln, "raw": rec, "error": str(e)})
                errors.add("artificial_lunar.csv", f"{path.name}:{ln}", str(e))
    schema = {
        "header": header,
        "expected_header": ["Frame", "TopLeftCornerX", "TopLeftCornerY", "Length", "Height"],
        "header_ok": header == ["Frame", "TopLeftCornerX", "TopLeftCornerY", "Length", "Height"],
        "data_rows": len(rows) + len(bad),
        "parse_failures": bad,
    }
    return rows, schema


def load_id_list(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").split()
    ok = [int(t) for t in raw if re.fullmatch(r"\d{1,5}", t)]
    return {
        "tokens": len(raw),
        "ids": ok,
        "non_numeric_tokens": [t for t in raw if not re.fullmatch(r"\d{1,5}", t)],
        "duplicates": sorted(k for k, v in Counter(ok).items() if v > 1),
        "zero_padded_4": all(len(t) == 4 for t in raw),
    }


# --------------------------------------------------------------------------- drawing
def _panel(root: Path, fid: int, boxes: list[dict[str, Any]], extra: list[list[int]] | None = None) -> np.ndarray:
    """render+boxes | ground | clean side by side (uint8 RGB)."""
    ims = []
    for kind in KINDS:
        im = Image.open(root / "images" / kind / f"{kind}{fid:04d}.png").convert("RGB")
        if kind != "ground":
            d = ImageDraw.Draw(im)
            for b in boxes:
                x0, y0 = b["x"] - 0.5, b["y"] - 0.5
                d.rectangle([x0, y0, x0 + b["w"], y0 + b["h"]], outline=(255, 0, 255), width=2)
            for c in extra or []:
                d.rectangle([c[0], c[1], c[0] + c[2], c[1] + c[3]], outline=(255, 255, 0), width=2)
        ims.append(np.asarray(im))
    return np.concatenate(ims, axis=1)


def _grid(root: Path, ids: list[int], by_frame: dict[int, list[dict[str, Any]]], title: str, path: Path,
          extra: dict[int, list[list[int]]] | None = None, notes: dict[int, str] | None = None) -> str:
    n = len(ids)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.4 * n))
    axes = np.atleast_1d(axes)
    for ax, fid in zip(axes, ids):
        ax.imshow(_panel(root, fid, by_frame.get(fid, []), (extra or {}).get(fid)))
        t = f"frame {fid:04d}  |  render (+CSV boxes magenta) | ground | clean  |  CSV boxes: {len(by_frame.get(fid, []))}"
        if notes and fid in notes:
            t += f"  |  {notes[fid]}"
        ax.set_title(t, fontsize=9, loc="left")
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    return save_fig(fig, path)


# --------------------------------------------------------------------------- main
def run(paths: Paths, errors: ErrorLog, workers: int) -> dict[str, Any]:
    root = paths.artificial_lunar
    LOG.info("A2 artificial lunar: %s", root)
    res: dict[str, Any] = {"root": paths.rel(root)}

    # ---- file presence and ID space
    ids_per_kind: dict[str, set[int]] = {}
    naming_violations: dict[str, list[str]] = {}
    for kind in KINDS:
        names = sorted(p.name for p in (root / "images" / kind).iterdir())
        pat = re.compile(rf"^{kind}(\d{{4}})\.png$")
        ids_per_kind[kind] = {int(m.group(1)) for n in names if (m := pat.match(n))}
        naming_violations[kind] = [n for n in names if not pat.match(n)]
    all_ids = sorted(set.union(*ids_per_kind.values()))
    common_ids = sorted(set.intersection(*ids_per_kind.values()))
    res["files"] = {
        "per_kind_count": {k: len(v) for k, v in ids_per_kind.items()},
        "naming_violations": naming_violations,
        "id_min": min(all_ids), "id_max": max(all_ids),
        "ids_present_in_all_three": len(common_ids),
        "ids_missing_in_some_kind": sorted(set(all_ids) - set(common_ids)),
        "gaps_in_id_range": sorted(set(range(min(all_ids), max(all_ids) + 1)) - set(all_ids)),
    }

    # ---- full decode pass (integrity + mask statistics), parallel but ordered
    LOG.info("decoding %d frames x 3 images with %d workers", len(common_ids), workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        frames = list(ex.map(analyse_frame, [(str(root), i) for i in common_ids], chunksize=32))
    fmap = {f["id"]: f for f in frames}
    for f in frames:
        for e in f["errors"]:
            errors.add("artificial_lunar.decode", f"frame {f['id']:04d}", e)
    ok = [f for f in frames if "render" in f]
    modes = Counter((k, f["files"][k].get("mode"), tuple(f["files"][k].get("size", []))) for f in frames for k in KINDS)
    res["image_properties"] = {
        "decoded_ok_frames": len(ok),
        "decode_failures": [f["id"] for f in frames if "render" not in f],
        "mode_size_counts": [{"kind": k, "mode": m, "size_wh": list(s), "count": c} for (k, m, s), c in sorted(modes.items())],
        "bit_depth": "8 bits per channel (PIL mode RGB, uint8)",
        "render_channels_identical_frames": sum(f["render_channel_max_diff"] == 0 for f in ok),
        "render_channel_max_diff": describe(f["render_channel_max_diff"] for f in ok),
        "clean_binary_frames": sum(f["clean_is_binary"] for f in ok),
        "file_bytes": {k: describe(f["files"][k]["bytes"] for f in ok) for k in KINDS},
        "render_mean_intensity": describe(f["render"]["mean"] for f in ok),
        "render_dark_frac_lt20": describe(f["render"]["dark_frac"] for f in ok),
        "render_dark_frac_non_sky": describe(f["render"]["dark_frac_non_sky"] for f in ok if f["render"]["dark_frac_non_sky"] is not None),
        "render_saturated_frac": describe(f["render"]["sat_frac"] for f in ok),
    }
    res["mask_composition"] = {
        f"{m}_{c}_frac": describe(f[m][c] for f in ok) for m in ("ground", "clean") for c in ("red", "green", "blue")
    }
    res["mask_composition"]["frames_without_sky_in_ground"] = sum(f["ground"]["red"] == 0 for f in ok)
    res["scene_consistency"] = {
        k: describe(v for f in ok if (v := f["consistency"][k]) is not None)
        for k in ok[0]["consistency"]
    }

    # ---- CSV
    rows, schema = load_csv(root / "bounding_boxes.csv", errors)
    res["csv_schema"] = schema
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    W, H = 720, 480
    frac = lambda v: v - np.floor(v)  # noqa: E731
    xs = np.array([r["x"] for r in rows]); ys = np.array([r["y"] for r in rows])
    ws = np.array([r["w"] for r in rows]); hs = np.array([r["h"] for r in rows])
    x0 = xs - 0.5; y0 = ys - 0.5  # hypothesis: MATLAB regionprops 1-based half-pixel convention
    invalid = {
        "nonpositive_w_or_h": [r["line"] for r in rows if r["w"] <= 0 or r["h"] <= 0],
        "negative_xy": [r["line"] for r in rows if r["x"] < 0 or r["y"] < 0],
        "x_out_of_bounds": [r["line"] for r, a, w in zip(rows, x0, ws) if a < 0 or a + w > W],
        "y_out_of_bounds": [r["line"] for r, a, h in zip(rows, y0, hs) if a < 0 or a + h > H],
        "frame_not_in_image_set": [r["line"] for r in rows if r["frame"] not in fmap],
        "non_integer_w_h": [r["line"] for r in rows if frac(r["w"]) or frac(r["h"])],
        "xy_not_half_pixel": [r["line"] for r in rows if frac(r["x"]) != 0.5 or frac(r["y"]) != 0.5],
    }
    dup_exact = Counter((r["frame"], r["x"], r["y"], r["w"], r["h"]) for r in rows)
    dup_lines = [k for k, v in dup_exact.items() if v > 1]
    near_dups = []
    for fid, bs in by_frame.items():
        if len(bs) < 2:
            continue
        a = np.array([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for b in bs])
        iou = iou_xyxy(a, a)
        np.fill_diagonal(iou, 0)
        for i, j in zip(*np.where(np.triu(iou) > 0.8)):
            near_dups.append({"frame": fid, "lines": [bs[i]["line"], bs[j]["line"]], "iou": float(iou[i, j])})
    tiny = [r["line"] for r in rows if r["w"] * r["h"] < 16 or min(r["w"], r["h"]) < 3]
    edge_touching = sum(1 for a, b, w, h in zip(x0, y0, ws, hs) if a <= 0 or b <= 0 or a + w >= W or b + h >= H)
    per_frame_counts = np.array([len(by_frame.get(i, [])) for i in common_ids])
    res["bounding_boxes"] = {
        "total_boxes": len(rows),
        "frames_with_boxes": len(by_frame),
        "frames_without_boxes": int((per_frame_counts == 0).sum()),
        "frames_without_boxes_ids_head": [i for i in common_ids if i not in by_frame][:50],
        "boxes_per_frame_all_frames": describe(per_frame_counts),
        "boxes_per_frame_histogram": dict(sorted(Counter(per_frame_counts.tolist()).items())),
        "frame_id_range_in_csv": [min(by_frame), max(by_frame)],
        "frame_column_sorted": all(rows[i]["frame"] <= rows[i + 1]["frame"] for i in range(len(rows) - 1)),
        "width_px": describe(ws), "height_px": describe(hs),
        "aspect_w_over_h": describe(ws / np.maximum(hs, 1e-9)),
        "area_px": describe(ws * hs),
        "area_frac_of_image": describe(ws * hs / (W * H)),
        "coordinate_convention": {
            "x_fraction_values": dict(Counter(np.round(frac(xs), 3).tolist())),
            "y_fraction_values": dict(Counter(np.round(frac(ys), 3).tolist())),
            "hypothesis": "TopLeftCorner is a MATLAB regionprops-style 1-based half-pixel edge; 0-based x0 = X - 0.5",
        },
        "invalid_counts": {k: len(v) for k, v in invalid.items()},
        "invalid_line_examples": {k: v[:25] for k, v in invalid.items()},
        "exact_duplicate_boxes": len(dup_lines),
        "exact_duplicate_examples": [list(k) for k in dup_lines[:20]],
        "near_duplicate_pairs_iou_gt_0.8": len(near_dups),
        "near_duplicate_examples": near_dups[:20],
        "tiny_boxes_area_lt16_or_side_lt3": len(tiny),
        "tiny_box_lines": tiny[:25],
        "boxes_touching_image_border": edge_touching,
    }

    # ---- CSV vs mask components (which mask generated the boxes? are all large rocks boxed?)
    agreement: dict[str, Any] = {}
    match_cache: dict[str, dict[int, Any]] = {}
    for src in ("clean_blue", "ground_blue"):
        offsets, matched_csv, unmatched_csv, unmatched_comp_areas = [], 0, [], []
        exact = 0
        per: dict[int, Any] = {}
        for fid in common_ids:
            f = fmap[fid]
            if "components" not in f:
                continue
            comps = np.array(f["components"][src], dtype=np.float64).reshape(-1, 5)
            bs = by_frame.get(fid, [])
            a = np.array([[b["x"] - 0.5, b["y"] - 0.5, b["x"] - 0.5 + b["w"], b["y"] - 0.5 + b["h"]] for b in bs]).reshape(-1, 4)
            c = np.c_[comps[:, 0], comps[:, 1], comps[:, 0] + comps[:, 2], comps[:, 1] + comps[:, 3]]
            pairs = greedy_match(iou_xyxy(a, c), MATCH_IOU)
            mi = {i for i, _, _ in pairs}; mj = {j for _, j, _ in pairs}
            matched_csv += len(pairs)
            for i, j, _ in pairs:
                d = (a[i] - c[j]).tolist()
                offsets.append(d)
                exact += int(np.allclose(a[i], c[j]))
            um = [bs[i]["line"] for i in range(len(bs)) if i not in mi]
            unmatched_csv += [(fid, ln) for ln in um]
            uc = [comps[j].astype(int).tolist() for j in range(len(comps)) if j not in mj]
            unmatched_comp_areas += [u[4] for u in uc]
            per[fid] = {"unmatched_csv_lines": um, "unmatched_components": uc}
        off = np.array(offsets).reshape(-1, 4)
        ua = np.array(unmatched_comp_areas)
        agreement[src] = {
            "csv_boxes_matched_iou_ge_0.5": matched_csv,
            "csv_boxes_unmatched": len(unmatched_csv),
            "csv_match_rate": matched_csv / max(len(rows), 1),
            "matched_with_exact_coordinates": exact,
            "offset_x0_y0_x1_y1_median": np.median(off, axis=0).tolist() if len(off) else None,
            "offset_abs_mean": np.abs(off).mean(axis=0).tolist() if len(off) else None,
            "components_total": int(sum(len(fmap[i]["components"][src]) for i in common_ids if "components" in fmap[i])),
            "components_without_csv_box": len(ua),
            "components_without_csv_box_by_area": {
                "lt_50px": int((ua < 50).sum()), "50_to_199px": int(((ua >= 50) & (ua < 200)).sum()),
                "200_to_999px": int(((ua >= 200) & (ua < 1000)).sum()), "ge_1000px": int((ua >= 1000).sum()),
            },
            "frames_with_ge1000px_unboxed_component": sum(
                any(u[4] >= 1000 for u in p["unmatched_components"]) for p in per.values()
            ),
        }
        match_cache[src] = per
    best = max(agreement, key=lambda k: agreement[k]["csv_match_rate"])
    agreement["best_matching_mask"] = best
    res["csv_vs_masks"] = agreement
    unmatched_lines = [(fid, ln) for fid, p in match_cache[best].items() for ln in p["unmatched_csv_lines"]]
    res["uncertain_annotations"] = {
        "definition": f"CSV boxes with no {best} component at IoU>={MATCH_IOU}; not fixed, flagged only",
        "count": len(unmatched_lines),
        "frames": len({f for f, _ in unmatched_lines}),
        "examples": [{"frame": f, "csv_line": ln} for f, ln in unmatched_lines[:40]],
    }
    comp_counts = {
        "clean_blue_per_frame": describe(len(fmap[i]["components"]["clean_blue"]) for i in common_ids if "components" in fmap[i]),
        "clean_green_per_frame": describe(fmap[i]["components"]["clean_green_n"] for i in common_ids if "components" in fmap[i]),
        "clean_green_ge50px_per_frame": describe(fmap[i]["components"]["clean_green_n_ge50px"] for i in common_ids if "components" in fmap[i]),
        "ground_green_per_frame": describe(fmap[i]["components"]["ground_green_n"] for i in common_ids if "components" in fmap[i]),
        "ground_green_ge50px_per_frame": describe(fmap[i]["components"]["ground_green_n_ge50px"] for i in common_ids if "components" in fmap[i]),
        "frames_without_csv_box_but_clean_blue": sum(1 for i in common_ids if i not in by_frame and fmap[i].get("components", {}).get("clean_blue")),
        "frames_without_csv_box_but_clean_green_ge50px": sum(1 for i in common_ids if i not in by_frame and fmap[i].get("components", {}).get("clean_green_n_ge50px", 0) > 0),
        "frames_without_any_rock_pixels_in_ground": sum(1 for f in ok if f["ground"]["green"] == 0 and f["ground"]["blue"] == 0),
    }
    res["label_coverage"] = comp_counts

    # ---- anomaly lists
    lists = {}
    metric = lambda ids, fn: describe(fn(fmap[i]) for i in ids if i in fmap and "render" in fmap[i])  # noqa: E731
    max_area = {fid: max(b["w"] * b["h"] for b in bs) for fid, bs in by_frame.items()}
    top200_by_csv = set(sorted(max_area, key=lambda k: (-max_area[k], k))[:200])
    for name in ANOMALY_FILES:
        p = root / name
        if not p.exists():
            errors.add("artificial_lunar.lists", name, "missing")
            continue
        info = load_id_list(p)
        ids = info.pop("ids")
        info.update({
            "count": len(ids),
            "ids_not_in_image_set": [i for i in ids if i not in fmap],
            "ids_with_csv_boxes": sum(i in by_frame for i in ids),
            "order_is_sorted": ids == sorted(ids),
            "metrics": {
                "render_mean": metric(ids, lambda f: f["render"]["mean"]),
                "render_dark_frac_non_sky": metric(ids, lambda f: f["render"]["dark_frac_non_sky"] or 0),
                "ground_sky_frac": metric(ids, lambda f: f["ground"]["red"]),
                "ground_sky_px_render_dark_frac": metric(ids, lambda f: f["consistency"]["ground_sky_px_render_dark_frac"] or 0),
                "render_black_px_in_ground_sky_frac": metric(ids, lambda f: f["consistency"]["render_black_px_in_ground_sky_frac"] or 0),
                "csv_boxes": describe(len(by_frame.get(i, [])) for i in ids),
                "max_csv_box_area": describe(max_area.get(i, 0) for i in ids),
            },
            "_ids": ids,
        })
        if name.startswith("top200"):
            info["overlap_with_top200_by_max_csv_box_area"] = len(set(ids) & top200_by_csv)
        lists[name] = info
    rest = [i for i in common_ids if not any(i in l["_ids"] for l in lists.values())]
    baseline = {
        "count": len(rest),
        "metrics": {
            "render_mean": metric(rest, lambda f: f["render"]["mean"]),
            "render_dark_frac_non_sky": metric(rest, lambda f: f["render"]["dark_frac_non_sky"] or 0),
            "ground_sky_frac": metric(rest, lambda f: f["ground"]["red"]),
            "ground_sky_px_render_dark_frac": metric(rest, lambda f: f["consistency"]["ground_sky_px_render_dark_frac"] or 0),
            "render_black_px_in_ground_sky_frac": metric(rest, lambda f: f["consistency"]["render_black_px_in_ground_sky_frac"] or 0),
            "csv_boxes": describe(len(by_frame.get(i, [])) for i in rest),
            "max_csv_box_area": describe(max_area.get(i, 0) for i in rest),
        },
    }
    names = list(lists)
    overlaps = {f"{a}&{b}": len(set(lists[a]["_ids"]) & set(lists[b]["_ids"])) for k, a in enumerate(names) for b in names[k + 1:]}
    union_flagged = sorted(set().union(*(set(l["_ids"]) for n, l in lists.items() if not n.startswith("top200"))))
    res["anomaly_lists"] = {
        "lists": {n: {k: v for k, v in l.items() if k != "_ids"} for n, l in lists.items()},
        "baseline_unlisted_frames": baseline,
        "pairwise_overlaps": overlaps,
        "union_of_quality_flag_lists_excluding_top200": len(union_flagged),
    }

    # ---- near-duplicate renders (64-bit dHash, Hamming <= 3)
    hashes = np.array([fmap[i]["render"]["dhash"] for i in common_ids if "render" in fmap[i]], dtype=np.uint64)
    hid = [i for i in common_ids if "render" in fmap[i]]
    pairs = []
    for s in range(0, len(hashes), 512):
        d = np.bitwise_count(hashes[s:s + 512, None] ^ hashes[None, :])
        for i, j in zip(*np.where(d <= 3)):
            if s + i < j:
                pairs.append((hid[s + i], hid[j], int(d[i, j])))
    res["near_duplicates"] = {
        "method": "64-bit difference hash of render, Hamming distance <= 3",
        "pairs": len(pairs),
        "examples": [list(p) for p in pairs[:30]],
        "exact_hash_collisions": sum(1 for p in pairs if p[2] == 0),
    }

    # ---- real moon images shipped with the Kaggle set
    res["real_moon_images"] = _real_moon(paths, root / "real_moon_images", errors)

    # ---- figures and samples
    res["figures"], res["samples"] = _figures(paths, root, rows, by_frame, fmap, common_ids, lists, match_cache[best], res)
    for f in frames:  # keep JSON compact: drop per-frame component lists
        f.pop("components", None)
    return res


def _real_moon(paths: Paths, d: Path, errors: ErrorLog) -> dict[str, Any]:
    if not d.exists():
        return {"present": False}
    out: dict[str, Any] = {"present": True, "images": {}, "masks": {}}
    for p in sorted(d.glob("*.png")):
        try:
            with Image.open(p) as im:
                entry = {"mode": im.mode, "size_wh": list(im.size)}
        except Exception as e:
            errors.add("artificial_lunar.real_moon", p.name, repr(e))
            continue
        (out["masks"] if p.name.startswith("g_") else out["images"])[p.name] = entry
    size_mismatch = []
    for name, e in out["images"].items():
        m = out["masks"].get("g_" + name)
        if m is None:
            size_mismatch.append({"image": name, "issue": "no mask"})
        elif m["size_wh"] != e["size_wh"]:
            size_mismatch.append({"image": name, "image_wh": e["size_wh"], "mask_wh": m["size_wh"]})
    out["image_mask_size_mismatches"] = size_mismatch
    out["counts"] = {"images": len(out["images"]), "masks": len(out["masks"]),
                     "PCAM": sum(n.startswith("PCAM") for n in out["images"]),
                     "TCAM": sum(n.startswith("TCAM") for n in out["images"])}
    csvp = d / "PCAM_bounding_boxes.csv"
    if csvp.exists():
        with csvp.open(newline="", encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        oob = []
        for r in recs:
            img = out["images"].get(f"PCAM{r['Frame']}.png")
            if img is None:
                oob.append({"frame": r["Frame"], "issue": "image missing"})
                continue
            w, h = img["size_wh"]
            x0, y0 = float(r["TopLeftCornerX"]) - 0.5, float(r["TopLeftCornerY"]) - 0.5
            if x0 < 0 or y0 < 0 or x0 + float(r["Length"]) > w or y0 + float(r["Height"]) > h:
                oob.append({"frame": r["Frame"], "box": [r["TopLeftCornerX"], r["TopLeftCornerY"], r["Length"], r["Height"]]})
        out["pcam_csv"] = {
            "columns": list(recs[0].keys()) if recs else [],
            "rows": len(recs),
            "frames": sorted({int(r["Frame"]) for r in recs}),
            "color_counts": dict(Counter(r.get("Color", "") for r in recs)),
            "out_of_bounds": oob,
        }
    out["tcam_boxes_available"] = (d / "TCAM_bounding_boxes.csv").exists()
    return out


def _figures(paths: Paths, root: Path, rows, by_frame, fmap, ids, lists, match, res) -> tuple[list[str], list[str]]:
    figs: list[str] = []
    samples: list[str] = []
    F, S = paths.figures, paths.samples
    ws = np.array([r["w"] for r in rows]); hs = np.array([r["h"] for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    ax[0, 0].hist(ws, bins=80, color="#4C72B0"); ax[0, 0].set(title="Box width (px)", xlabel="px", ylabel="boxes")
    ax[0, 1].hist(hs, bins=80, color="#4C72B0"); ax[0, 1].set(title="Box height (px)", xlabel="px")
    ax[1, 0].hist(np.log10(ws * hs), bins=80, color="#55A868"); ax[1, 0].set(title="Box area (log10 px²)", xlabel="log10(area)")
    ax[1, 1].hist(np.clip(ws / hs, 0, 8), bins=80, color="#C44E52"); ax[1, 1].set(title="Aspect ratio w/h (clipped at 8)", xlabel="w/h")
    fig.suptitle(f"Artificial Lunar - bounding_boxes.csv ({len(rows)} boxes, image 720x480)")
    figs.append(save_fig(fig, F / "al_bbox_distributions.png"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    hb = ax[0].hexbin(ws, hs, gridsize=60, bins="log", cmap="viridis"); fig.colorbar(hb, ax=ax[0], label="log count")
    ax[0].set(title="Width vs height (px)", xlabel="width", ylabel="height")
    cx = np.array([r["x"] - 0.5 + r["w"] / 2 for r in rows]); cy = np.array([r["y"] - 0.5 + r["h"] / 2 for r in rows])
    h2 = ax[1].hist2d(cx, cy, bins=[72, 48], range=[[0, 720], [0, 480]], cmap="magma"); fig.colorbar(h2[3], ax=ax[1], label="boxes")
    ax[1].invert_yaxis(); ax[1].set(title="Box centre heat-map", xlabel="x", ylabel="y"); ax[1].set_aspect("equal")
    figs.append(save_fig(fig, F / "al_bbox_wh_and_centres.png"))

    have = [i for i in ids if "components" in fmap[i]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    c = [len(by_frame.get(i, [])) for i in ids]
    ax[0].bar(*np.unique(c, return_counts=True), color="#4C72B0"); ax[0].set(title="CSV boxes per frame (all frames)", xlabel="boxes", ylabel="frames")
    ax[1].hist([len(fmap[i]["components"]["clean_blue"]) for i in have], bins=range(0, 15), color="#8172B2")
    ax[1].set(title="Blue components per frame (clean mask)", xlabel="components")
    ax[2].hist([fmap[i]["components"]["ground_green_n_ge50px"] for i in have], bins=60, color="#55A868")
    ax[2].set(title="Green (small-rock) components >=50px per frame\n(ground mask; NOT in CSV)", xlabel="components")
    figs.append(save_fig(fig, F / "al_objects_per_frame.png"))

    # anomaly list comparison
    keys = [("render_mean", "render mean intensity"), ("render_dark_frac_non_sky", "dark (<20) fraction, non-sky"),
            ("ground_sky_frac", "sky fraction (ground mask)"), ("render_black_px_in_ground_sky_frac", "render black px labelled sky")]
    fn = {
        "render_mean": lambda f: f["render"]["mean"],
        "render_dark_frac_non_sky": lambda f: f["render"]["dark_frac_non_sky"] or 0,
        "ground_sky_frac": lambda f: f["ground"]["red"],
        "render_black_px_in_ground_sky_frac": lambda f: f["consistency"]["render_black_px_in_ground_sky_frac"] or 0,
    }
    groups = {"unlisted": [i for i in ids if not any(i in l["_ids"] for l in lists.values())]}
    groups.update({n.replace("_IDs.txt", ""): l["_ids"] for n, l in lists.items()})
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
    for a, (k, label) in zip(ax, keys):
        data = [[fn[k](fmap[i]) for i in g if "render" in fmap.get(i, {})] for g in groups.values()]
        a.boxplot(data, showfliers=False); a.set_xticks(range(1, len(groups) + 1), list(groups), rotation=40, ha="right")
        a.set_title(label, fontsize=10)
    fig.suptitle("Artificial Lunar - anomaly lists vs unlisted frames")
    figs.append(save_fig(fig, F / "al_anomaly_list_metrics.png"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist([fmap[i]["render"]["mean"] for i in ids if "render" in fmap[i]], bins=80, color="#4C72B0")
    ax[0].set(title="Render mean intensity per frame", xlabel="mean (0-255)", ylabel="frames")
    ax[1].hist([fmap[i]["ground"]["red"] for i in ids if "render" in fmap[i]], bins=80, color="#C44E52")
    ax[1].set(title="Sky fraction per frame (ground mask red)", xlabel="fraction of pixels")
    figs.append(save_fig(fig, F / "al_render_intensity_and_sky.png"))

    # ---- samples (deterministic)
    rng = np.random.default_rng(SEED)
    rand = sorted(rng.choice([i for i in ids if "render" in fmap[i]], size=6, replace=False).tolist())
    samples.append(_grid(root, rand, by_frame, "Random frames (seeded)", S / "al_random_frames.png"))
    for n, l in lists.items():
        pick = sorted(rng.choice(l["_ids"], size=min(4, len(l["_ids"])), replace=False).tolist())
        samples.append(_grid(root, pick, by_frame, f"Anomaly list: {n}", S / f"al_list_{n.replace('_IDs.txt', '')}.png"))
    darkest = sorted(ids, key=lambda i: -(fmap[i]["render"]["dark_frac_non_sky"] or 0) if "render" in fmap[i] else 0)[:4]
    samples.append(_grid(root, darkest, by_frame, "Most shadowed frames (highest dark fraction outside sky)", S / "al_heavy_shadow.png"))
    flat = sorted(((r["w"] * r["h"], r["frame"]) for r in rows))
    small = []
    for _, fr in flat:
        if fr not in small:
            small.append(fr)
        if len(small) == 3:
            break
    large = []
    for _, fr in reversed(flat):
        if fr not in large:
            large.append(fr)
        if len(large) == 3:
            break
    samples.append(_grid(root, small + large, by_frame, "Smallest-box frames (top 3) and largest-box frames (bottom 3)", S / "al_small_and_large_boxes.png"))
    bad = [f for f, p in sorted(match.items()) if p["unmatched_csv_lines"]][:3]
    miss = sorted((f for f, p in match.items() if any(u[4] >= 1000 for u in p["unmatched_components"])),
                  key=lambda f: -max(u[4] for u in match[f]["unmatched_components"]))[:3]
    extra = {f: [u for u in match[f]["unmatched_components"] if u[4] >= 200] for f in bad + miss}
    notes = {f: "CSV box w/o matching blue component" for f in bad}
    notes.update({f: "yellow = blue component (>=1000px) without CSV box" for f in miss})
    samples.append(_grid(root, bad + miss, by_frame, "Potentially problematic annotations", S / "al_problematic_annotations.png", extra, notes))

    rm = root / "real_moon_images"
    if rm.exists():
        pcsv = {}
        p = rm / "PCAM_bounding_boxes.csv"
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    pcsv.setdefault(int(r["Frame"]), []).append(r)
        fig, ax = plt.subplots(3, 2, figsize=(12, 13))
        for k, (name, frame) in enumerate([("PCAM1", 1), ("PCAM7", 7), ("TCAM10", None)]):
            im = Image.open(rm / f"{name}.png").convert("RGB")
            d = ImageDraw.Draw(im)
            for r in pcsv.get(frame, []) if frame else []:
                x0, y0 = float(r["TopLeftCornerX"]) - 0.5, float(r["TopLeftCornerY"]) - 0.5
                col = (0, 255, 0) if r["Color"] == "Green" else (0, 128, 255)
                d.rectangle([x0, y0, x0 + float(r["Length"]), y0 + float(r["Height"])], outline=col, width=2)
            ax[k, 0].imshow(im); ax[k, 0].set_title(f"{name} (+PCAM CSV boxes: green/blue)" if frame else name, fontsize=9)
            ax[k, 1].imshow(Image.open(rm / f"g_{name}.png").convert("RGB")); ax[k, 1].set_title(f"g_{name} mask", fontsize=9)
            ax[k, 0].axis("off"); ax[k, 1].axis("off")
        samples.append(save_fig(fig, S / "al_real_moon_images.png"))
    return figs, samples
