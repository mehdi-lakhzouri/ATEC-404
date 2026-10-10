"""Artificial Lunar review artefacts: per-scene overlays, A/B/C/D comparisons, contact sheets, crop grids."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..dataset_exploration.common import LOG, Paths
from .al_analysis import csv_xyxy
from .config import PRIMARY_VARIANT, PROPOSED_MIN_AREA, SEED
from .drawing import boxes_for_frame, crop_tile, grid, header, legend_strip, overlay, save, stack, titled
from .s2_extract import F, load_rgb, visibility_flag

CANDIDATE_STRATA = [  # (class, area_lo, area_hi, n)
    ("small", 25, 50, 16), ("small", 50, 100, 16), ("small", 100, 200, 16), ("small", 200, 1000, 16), ("small", 1000, 10 ** 9, 12),
    ("large_new", 25, 200, 10), ("large_new", 200, 1000, 10), ("large_new", 1000, 10 ** 9, 10), ("large_matched", 25, 10 ** 9, 8),
]


def scene_boxes(al: dict[str, Any], fid: int, min_area: int = PROPOSED_MIN_AREA) -> dict[str, Any]:
    f = al["_frames"][fid]["variants"][PRIMARY_VARIANT]
    L, S = f["large_feat"], f["small_feat"]
    pairs = al["_match"][PRIMARY_VARIANT][fid]["pairs"]
    mj = {j for _, j, _ in pairs}
    okL = np.array([visibility_flag(r) == "ok" for r in L], bool) if len(L) else np.zeros(0, bool)
    okS = np.array([visibility_flag(r) == "ok" for r in S], bool) if len(S) else np.zeros(0, bool)
    aL = L[:, F["area"]] if len(L) else np.zeros(0); aS = S[:, F["area"]] if len(S) else np.zeros(0)
    matched = np.array([j in mj for j in range(len(L))], bool)
    keep_L = matched | ((aL >= min_area) & okL)
    keep_S = (aS >= min_area) & okS
    rej_L = ~keep_L & (aL >= 25)
    rej_S = ~keep_S & (aS >= 25)
    csv = [csv_xyxy(b) for b in al["_by_frame"].get(fid, [])]
    return boxes_for_frame(csv, L, S, pairs, keep_L, keep_S, rej_L, rej_S)


def render_scenes(paths: Paths, out: Path, al: dict[str, Any], manifest: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    root = paths.artificial_lunar
    files: dict[int, dict[str, Any]] = {}
    rows_by_cat = defaultdict(list)
    for m in manifest:
        fid = m["frame_id"]
        render, _ = load_rgb(root / "images" / "render" / f"render{fid:04d}.png")
        ground, mode = load_rgb(root / "images" / "ground" / f"ground{fid:04d}.png")
        b = scene_boxes(al, fid)
        c = b["counts"]
        lists = m["anomaly_lists"] or "none"
        head = [f"frame {fid:04d} | {m['primary_category']} | anomaly lists: {lists} | ground mode {mode}",
                f"S1 CSV boxes {c['csv']} (unmatched {c['csv_unmatched']}) | S2 large {c['s2_large_matched']}+{c['s2_large_new']} new | "
                f"S2 small {c['s2_small']} | rejected {c['s2_rejected']} | S2 min area {PROPOSED_MIN_AREA}px | native 720x480, no resizing"]
        A = titled(Image.fromarray(render), "A  raw render (detector input)")
        B = titled(overlay(render, b["s1"]), f"B  S1 CSV large-rock boxes ({c['csv']})")
        C = titled(overlay(render, b["s2_with_rejected"]), f"C  S2 mask-derived candidates ({c['s2_large_matched'] + c['s2_large_new'] + c['s2_small']} kept)")
        D = titled(overlay(render, b["diag"]), "D  diagnostic: S1 vs S2 differences")
        G = titled(Image.fromarray(ground), "ground mask (red sky / green small / blue large)")
        grid2 = stack([stack([A, B], True), stack([C, D], True)], False)
        comp = stack([header(grid2.width, head), grid2, legend_strip(grid2.width)], False)
        p = {
            "s1_overlay": out / "s1_csv_overlays" / f"f{fid:04d}_s1.jpg",
            "s1_side_by_side": out / "s1_csv_overlays" / f"f{fid:04d}_raw_s1_ground.jpg",
            "s2_overlay": out / "s2_mask_overlays" / f"f{fid:04d}_s2.jpg",
            "comparison": out / "comparison_sheets" / "scenes" / f"f{fid:04d}_compare.jpg",
        }
        save(stack([header(B.width, head[:1]), B, legend_strip(B.width, ["csv"])], False), p["s1_overlay"])
        side = stack([A, B, G], True)
        save(stack([header(side.width, head[:1]), side, legend_strip(side.width, ["csv"])], False), p["s1_side_by_side"])
        save(stack([header(C.width, head), C, legend_strip(C.width, ["s2_large", "s2_large_new", "s2_small", "s2_rejected"])], False), p["s2_overlay"])
        save(comp, p["comparison"])
        files[fid] = {k: v for k, v in p.items()} | {"counts": c}
        small = [im.resize((im.width // 2, im.height // 2), Image.BILINEAR) for im in (A, B, C, D)]
        row = stack([header(small[0].width * 4, [head[0], head[1]]), stack(small, True)], False)
        rows_by_cat[m["primary_category"]].append(row)
    # contact sheets: 5 scenes per sheet, grouped by sampling category
    k = 0
    for cat in sorted(rows_by_cat):
        rows = rows_by_cat[cat]
        for s in range(0, len(rows), 5):
            k += 1
            sheet = stack([header(rows[0].width, [f"Contact sheet {k:02d} - category: {cat} (A raw | B S1 | C S2 | D diagnostic), 50% display scale"])]
                          + rows[s:s + 5] + [legend_strip(rows[0].width)], False)
            save(sheet, out / "comparison_sheets" / f"contact_{k:02d}_{cat}.jpg")
    LOG.info("rendered %d scenes, %d contact sheets", len(files), k)
    return files


def candidate_review(paths: Paths, out: Path, al: dict[str, Any]) -> list[dict[str, Any]]:
    """Stratified per-candidate crops (clean subset) for visual pass/fail review."""
    rng = np.random.default_rng(SEED + 1)
    pool = defaultdict(list)
    for fid in al["_clean_ids"]:
        f = al["_frames"][fid]
        if "error" in f:
            continue
        v = f["variants"][PRIMARY_VARIANT]
        mj = {j for _, j, _ in al["_match"][PRIMARY_VARIANT][fid]["pairs"]}
        for cls, arr in (("small", v["small_feat"]), ("large", v["large_feat"])):
            for j, r in enumerate(arr):
                c = cls if cls == "small" else ("large_matched" if j in mj else "large_new")
                for cc, lo, hi, _ in CANDIDATE_STRATA:
                    if cc == c and lo <= r[F["area"]] < hi:
                        pool[(cc, lo)].append((fid, cls, j))
    picks = []
    for cc, lo, hi, n in CANDIDATE_STRATA:
        items = pool[(cc, lo)]
        for k in sorted(rng.choice(len(items), size=min(n, len(items)), replace=False)):
            picks.append((cc, lo, hi, *items[k]))
    records, tiles = [], []
    for k, (cc, lo, hi, fid, cls, j) in enumerate(picks, start=1):
        r = al["_frames"][fid]["variants"][PRIMARY_VARIANT][f"{cls}_feat"][j]
        xyxy = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
        flag = visibility_flag(r)
        cid = f"K{k:03d}"
        records.append({"candidate_id": cid, "frame_id": fid, "stratum": f"{cc}_{lo}-{hi if hi < 10**9 else 'inf'}", "mask_class": cls,
                        "x0": int(xyxy[0]), "y0": int(xyxy[1]), "x1": int(xyxy[2]), "y1": int(xyxy[3]), "area_px": int(r[F["area"]]),
                        "fill": round(float(r[F["fill"]]), 3), "contrast": None if np.isnan(r[F["contrast"]]) else round(float(r[F["contrast"]]), 2),
                        "visibility_proxy": flag, "border": bool(r[F["border"]])})
        render, _ = load_rgb(paths.artificial_lunar / "images" / "render" / f"render{fid:04d}.png")
        ground, _ = load_rgb(paths.artificial_lunar / "images" / "ground" / f"ground{fid:04d}.png")
        style = "s2_small" if cls == "small" else ("s2_large" if cc == "large_matched" else "s2_large_new")
        tiles.append(crop_tile(render, ground, xyxy, style, f"{cid} f{fid:04d} {cc} a={int(r[F['area']])} {flag}"))
    for s in range(0, len(tiles), 24):
        save(grid(tiles[s:s + 24], 4, f"S2 candidate review {s // 24 + 1}: left = render crop (nearest-neighbour zoom), right = ground mask crop"),
             out / "s2_mask_overlays" / "candidate_review" / f"candidates_{s // 24 + 1:02d}.jpg")
    return records


UNCERTAIN_STRATA = [  # (name, predicate, n)
    ("clean_fragmented", lambda u: not u["anomaly_lists"] and u["auto_class"] == "fragmented_mask_components" and not u["border"], 8),
    ("clean_extends_beyond_box", lambda u: not u["anomaly_lists"] and u["auto_class"] == "mask_component_extends_beyond_box" and not u["border"], 8),
    ("clean_other_mechanism", lambda u: not u["anomaly_lists"] and u["auto_class"] in
     ("matched_after_cyan_assignment", "box_covers_mask_pixels_other", "partial_mask_support") and not u["border"], 6),
    ("clean_no_rock_pixels", lambda u: not u["anomaly_lists"] and u["auto_class"] == "no_large_rock_pixels_in_box", 6),
    ("clean_border", lambda u: not u["anomaly_lists"] and u["border"], 6),
    ("list_mismatch", lambda u: "mismatch" in u["anomaly_lists"], 6),
    ("list_cam_anomaly", lambda u: "cam_anomaly" in u["anomaly_lists"], 6),
    ("list_shadow", lambda u: "shadow" in u["anomaly_lists"], 6),
]


def uncertain_review(paths: Paths, out: Path, al: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 2)
    unc = al["_uncertain"]
    used: set[tuple[int, int]] = set()
    records, tiles = [], []
    for name, pred, n in UNCERTAIN_STRATA:
        pool = [u for u in unc if pred(u) and (u["frame"], u["csv_line"]) not in used]
        for k in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False)):
            u = pool[k]
            used.add((u["frame"], u["csv_line"]))
            uid = f"U{len(records) + 1:03d}"
            records.append({"uncertain_id": uid, "stratum": name, "frame_id": u["frame"], "csv_line": u["csv_line"],
                            "x0": round(u["box_xyxy"][0], 1), "y0": round(u["box_xyxy"][1], 1), "x1": round(u["box_xyxy"][2], 1), "y1": round(u["box_xyxy"][3], 1),
                            "auto_class": u["auto_class"], "anomaly_lists": ";".join(u["anomaly_lists"]), "border": u["border"],
                            "blue_frac_strict": round(u["blue_frac_strict"], 3), "blue_frac_cyan": round(u["blue_frac_cyan"], 3),
                            "union_bbox_iou": round(u["union_bbox_iou"], 3)})
            render, _ = load_rgb(paths.artificial_lunar / "images" / "render" / f"render{u['frame']:04d}.png")
            ground, _ = load_rgb(paths.artificial_lunar / "images" / "ground" / f"ground{u['frame']:04d}.png")
            tiles.append(crop_tile(render, ground, u["box_xyxy"], "csv_unmatched",
                                   f"{uid} f{u['frame']:04d} L{u['csv_line']} {name} {u['auto_class'][:12]}"))
    for s in range(0, len(tiles), 24):
        save(grid(tiles[s:s + 24], 4, f"Uncertain CSV boxes {s // 24 + 1}: left = render crop + CSV box, right = ground mask crop"),
             out / "uncertain_boxes" / f"uncertain_{s // 24 + 1:02d}.jpg")
    return records
