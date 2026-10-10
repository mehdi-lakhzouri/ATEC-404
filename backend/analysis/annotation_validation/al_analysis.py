"""Artificial Lunar: full-dataset S1/S2 statistics, uncertain-box triage and stratified sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ..dataset_exploration.artificial_lunar import load_csv, load_id_list
from ..dataset_exploration.common import LOG, ErrorLog, Paths, describe, greedy_match, iou_xyxy
from .config import (AREA_THRESHOLDS, ANOMALY_FILES, IMG_H, IMG_W, MATCH_IOU, PRIMARY_VARIANT, PROPOSED_MIN_AREA,
                     QUOTAS, SEED, VARIANTS)
from .s2_extract import F, analyse_frame, visibility_flag

AREA_BINS = ((10, 25), (25, 50), (50, 100), (100, 200), (200, 1000), (1000, 10 ** 9))


def csv_xyxy(b: dict[str, Any]) -> list[float]:
    """MATLAB half-pixel TopLeftCorner -> 0-based edges."""
    x0, y0 = b["x"] - 0.5, b["y"] - 0.5
    return [x0, y0, x0 + b["w"], y0 + b["h"]]


def to_xyxy(stats: np.ndarray) -> np.ndarray:
    """x, y, w, h[, ...] rows -> x0, y0, x1, y1."""
    s = np.asarray(stats, dtype=np.float64)
    if s.size == 0:
        return np.zeros((0, 4))
    return np.c_[s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3]]


def _contained(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """(N,M) fraction of each inner box's area lying inside each outer box."""
    if len(inner) == 0 or len(outer) == 0:
        return np.zeros((len(inner), len(outer)))
    ix0 = np.maximum(inner[:, None, 0], outer[None, :, 0]); iy0 = np.maximum(inner[:, None, 1], outer[None, :, 1])
    ix1 = np.minimum(inner[:, None, 2], outer[None, :, 2]); iy1 = np.minimum(inner[:, None, 3], outer[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = (inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def _gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,M) Chebyshev gap in px between boxes (0 = touching/overlapping)."""
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a), len(b)), np.inf)
    dx = np.maximum(0, np.maximum(a[:, None, 0] - b[None, :, 2], b[None, :, 0] - a[:, None, 2]))
    dy = np.maximum(0, np.maximum(a[:, None, 1] - b[None, :, 3], b[None, :, 1] - a[:, None, 3]))
    return np.maximum(dx, dy)


def match_frame(csv_boxes: np.ndarray, comps: np.ndarray) -> dict[str, Any]:
    c = to_xyxy(comps)
    pairs = greedy_match(iou_xyxy(csv_boxes, c), MATCH_IOU)
    cont = _contained(c, csv_boxes)  # comp inside csv box
    big = comps[:, 4] >= 25 if len(comps) else np.zeros(0, bool)
    split = int(sum(((cont[:, j] >= 0.8) & big).sum() >= 2 for j in range(len(csv_boxes))))
    merge = 0
    if len(csv_boxes) and len(c):
        cx = (csv_boxes[:, 0] + csv_boxes[:, 2]) / 2; cy = (csv_boxes[:, 1] + csv_boxes[:, 3]) / 2
        inside = (cx[None, :] >= c[:, None, 0]) & (cx[None, :] <= c[:, None, 2]) & (cy[None, :] >= c[:, None, 1]) & (cy[None, :] <= c[:, None, 3])
        merge = int((inside.sum(axis=1) >= 2).sum())
    return {"pairs": pairs, "split_csv_boxes": split, "merged_components": merge}


def run(paths: Paths, errors: ErrorLog, workers: int) -> dict[str, Any]:
    root = paths.artificial_lunar
    rows, schema = load_csv(root / "bounding_boxes.csv", errors)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    ids = list(range(1, 9767))
    lists = {k: set(load_id_list(root / f)["ids"]) for k, f in ANOMALY_FILES.items()}
    anomalous = set().union(*lists.values())
    clean_ids = [i for i in ids if i not in anomalous]
    LOG.info("B: S2 extraction over %d frames (%d variants)", len(ids), len(VARIANTS))
    args = [(str(root), i, [csv_xyxy(b) for b in by_frame.get(i, [])]) for i in ids]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        frames = {f["id"]: f for f in ex.map(analyse_frame, args, chunksize=32)}
    for f in frames.values():
        if "error" in f:
            errors.add("annotation_validation.extract", f"frame {f['id']:04d}", f["error"])
    ok = [i for i in ids if "error" not in frames[i]]

    # ---------------- variant comparison against S1 (full dataset + clean subset)
    variant_metrics = {}
    match_cache: dict[str, dict[int, dict[str, Any]]] = {}
    for v in VARIANTS:
        per = {}
        for i in ok:
            cb = np.array([csv_xyxy(b) for b in by_frame.get(i, [])]).reshape(-1, 4)
            per[i] = match_frame(cb, frames[i]["variants"][v.name]["large"])
        match_cache[v.name] = per

        def agg(sel: list[int]) -> dict[str, Any]:
            n_csv = sum(len(by_frame.get(i, [])) for i in sel)
            m = sum(len(per[i]["pairs"]) for i in sel)
            return {"csv_boxes": n_csv, "matched_iou_ge_0.5": m, "match_rate": m / max(n_csv, 1),
                    "csv_boxes_split_into_ge2_components": sum(per[i]["split_csv_boxes"] for i in sel),
                    "components_covering_ge2_csv_centres": sum(per[i]["merged_components"] for i in sel),
                    "large_components_ge25px": int(sum((frames[i]["variants"][v.name]["large"][:, 4] >= 25).sum() for i in sel))}
        variant_metrics[v.name] = {"description": v.description, "all_frames": agg(ok),
                                   "candidate_subset": agg([i for i in ok if i not in anomalous])}

    # ---------------- what rule generated the CSV? (V0 strict components vs CSV)
    matched_dims, unmatched_dims = [], []
    for i in ok:
        comps = frames[i]["variants"]["V0_strict"]["large"]
        mj = {j for _, j, _ in match_cache["V0_strict"][i]["pairs"]}
        for j, c in enumerate(comps):
            (matched_dims if j in mj else unmatched_dims).append((c[2], c[3], c[4]))
    md, ud = np.array(matched_dims), np.array(unmatched_dims)
    csv_rule = {
        "matched_component_min_w_h_area": [int(md[:, 0].min()), int(md[:, 1].min()), int(md[:, 2].min())] if len(md) else None,
        "unmatched_components": len(ud),
        "unmatched_with_w_and_h_ge_21": int(((ud[:, 0] >= 21) & (ud[:, 1] >= 21)).sum()) if len(ud) else 0,
        "unmatched_with_w_or_h_lt_21": int(((ud[:, 0] < 21) | (ud[:, 1] < 21)).sum()) if len(ud) else 0,
        "hypothesis": "CSV = strict-blue components of ground mask with width and height >= 21 px (plus manual/other edits)",
    }

    # ---------------- S2 (primary) full-dataset statistics
    P = PRIMARY_VARIANT
    s2 = {}
    for scope, sel in (("all_frames", ok), ("candidate_subset", [i for i in ok if i not in anomalous])):
        res = {"frames": len(sel), "csv_boxes": sum(len(by_frame.get(i, [])) for i in sel)}
        for T in AREA_THRESHOLDS:
            large_new, small, total, flags = [], [], [], Counter()
            for i in sel:
                v = frames[i]["variants"][P]
                mj = {j for _, j, _ in match_cache[P][i]["pairs"]}
                L = v["large_feat"]; S = v["small_feat"]
                ln = [j for j in range(len(L)) if j not in mj and L[j, F["area"]] >= T]
                sm = [j for j in range(len(S)) if S[j, F["area"]] >= T]
                large_new.append(len(ln)); small.append(len(sm))
                total.append(len(by_frame.get(i, [])) + len(ln) + len(sm))
                for j in sm:
                    flags[visibility_flag(S[j])] += 1
                for j in ln:
                    flags[visibility_flag(L[j])] += 1
            res[f"min_area_{T}"] = {
                "additional_large_without_csv_per_image": describe(large_new),
                "additional_small_medium_per_image": describe(small),
                "s2_total_boxes_per_image(csv+additional)": describe(total),
                "additional_total": int(sum(large_new) + sum(small)),
                "additional_relative_to_s1_boxes": (sum(large_new) + sum(small)) / max(res["csv_boxes"], 1),
                "additional_visibility_proxy_counts": dict(flags),
                "frames_with_zero_csv_but_ge1_additional": sum(1 for i, a, b in zip(sel, large_new, small) if not by_frame.get(i) and a + b > 0),
            }
        sa = np.concatenate([frames[i]["variants"][P]["small_feat"][:, F["area"]] for i in sel]) if sel else np.zeros(0)
        la = np.concatenate([frames[i]["variants"][P]["large_feat"][:, F["area"]] for i in sel]) if sel else np.zeros(0)
        res["small_component_area_px"] = describe(sa)
        res["large_component_area_px"] = describe(la)
        res["small_area_bins"] = {f"{a}-{b if b < 10**9 else 'inf'}": int(((sa >= a) & (sa < b)).sum()) for a, b in AREA_BINS}
        res["large_area_bins"] = {f"{a}-{b if b < 10**9 else 'inf'}": int(((la >= a) & (la < b)).sum()) for a, b in AREA_BINS}
        sw = np.concatenate([frames[i]["variants"][P]["small_feat"][:, [F["w"], F["h"]]] for i in sel])
        res["small_box_w_px"] = describe(sw[:, 0]); res["small_box_h_px"] = describe(sw[:, 1])
        csv_areas = np.array([b["w"] * b["h"] for i in sel for b in by_frame.get(i, [])])
        med = float(np.median(csv_areas)) if len(csv_areas) else 0.0
        res["apparent_size_overlap"] = {
            "note": "green = physically small/medium rocks; many appear as large in the image as CSV 'large' rocks",
            "small_components_with_w_and_h_ge_21px": int(((sw[:, 0] >= 21) & (sw[:, 1] >= 21)).sum()),
            "small_components_bbox_area_ge_csv_median_box_area": int((sw[:, 0] * sw[:, 1] >= med).sum()),
            "csv_median_box_area_px": med,
            "csv_boxes_with_bbox_area_lt_1000px": int((csv_areas < 1000).sum()),
            "frames": len(sel),
        }
        s2[scope] = res
    # candidate hybrid label policy H: CSV instance boxes for large rocks (mask merges touching boulders)
    # + blue components 50..999 px not covered by a CSV box (distant large rocks below the 21-px CSV cutoff)
    # + green components >= 50 px whose visibility proxy is "ok"
    def hybrid_counts(sel: list[int]) -> dict[str, Any]:
        csv_n, blue_n, green_n = [], [], []
        for i in sel:
            v = frames[i]["variants"][P]
            cb = np.array([csv_xyxy(b) for b in by_frame.get(i, [])]).reshape(-1, 4)
            L = v["large_feat"]; S = v["small_feat"]
            lb = to_xyxy(L[:, :4]) if len(L) else np.zeros((0, 4))
            covered = (_contained(lb, cb).max(axis=1) >= 0.5) if len(cb) and len(lb) else np.zeros(len(lb), bool)
            okL = np.array([visibility_flag(r) == "ok" for r in L], bool) if len(L) else np.zeros(0, bool)
            okS = np.array([visibility_flag(r) == "ok" for r in S], bool) if len(S) else np.zeros(0, bool)
            aL = L[:, F["area"]] if len(L) else np.zeros(0)
            aS = S[:, F["area"]] if len(S) else np.zeros(0)
            csv_n.append(len(cb)); blue_n.append(int(((aL >= 50) & (aL < 1000) & ~covered & okL).sum()))
            green_n.append(int(((aS >= PROPOSED_MIN_AREA) & okS).sum()))
        tot = np.array(csv_n) + np.array(blue_n) + np.array(green_n)
        return {"definition": "CSV boxes + blue comps 50..999px not >=50% inside a CSV box (proxy ok) + green comps >= "
                              f"{PROPOSED_MIN_AREA}px (proxy ok); variant {P}",
                "csv_per_image": describe(csv_n), "blue_small_large_rocks_per_image": describe(blue_n),
                "green_per_image": describe(green_n), "total_per_image": describe(tot),
                "frames_with_zero_boxes": int((tot == 0).sum()), "total_boxes": int(tot.sum())}
    s2["hybrid_policy_H_candidate_subset"] = hybrid_counts([i for i in ok if i not in anomalous])
    # by image condition (candidate subset): dark vs bright, shadowy vs not
    cond = {}
    for name, pred in (("render_mean_lt_60", lambda f: f["render_mean"] < 60), ("render_mean_ge_60", lambda f: f["render_mean"] >= 60),
                       ("dark_frac_non_sky_ge_0.2", lambda f: f["dark_frac_non_sky"] >= 0.2), ("dark_frac_non_sky_lt_0.2", lambda f: f["dark_frac_non_sky"] < 0.2)):
        sel = [i for i in clean_ids if i in frames and "error" not in frames[i] and pred(frames[i])]
        vis = Counter()
        n_small = 0
        for i in sel:
            S = frames[i]["variants"][P]["small_feat"]
            S = S[S[:, F["area"]] >= PROPOSED_MIN_AREA]
            n_small += len(S)
            vis.update(visibility_flag(r) for r in S)
        cond[name] = {"frames": len(sel), "csv_boxes_per_frame": sum(len(by_frame.get(i, [])) for i in sel) / max(len(sel), 1),
                      "small_ge_min_area_per_frame": n_small / max(len(sel), 1), "small_visibility_proxy": dict(vis)}
    s2["by_condition_candidate_subset"] = cond

    # ---------------- uncertain CSV boxes (Task A definition: no V0 component at IoU >= 0.5)
    uncertain = []
    for i in ok:
        bs = by_frame.get(i, [])
        mi_v0 = {a for a, _, _ in match_cache["V0_strict"][i]["pairs"]}
        mi_p = {a for a, _, _ in match_cache[P][i]["pairs"]}
        for k, b in enumerate(bs):
            if k in mi_v0:
                continue
            ev = frames[i]["csv_evidence"][k]
            box = csv_xyxy(b)
            border = box[0] <= 0 or box[1] <= 0 or box[2] >= IMG_W or box[3] >= IMG_H
            comps = frames[i]["variants"]["V0_strict"]["large"]
            cx = to_xyxy(comps)
            bx = np.array([box])
            inside = _contained(cx, bx)[:, 0] if len(cx) else np.zeros(0)
            n_inside = int(((inside >= 0.8) & (comps[:, 4] >= 25)).sum()) if len(cx) else 0
            box_in_comp = _contained(bx, cx)[0] if len(cx) else np.zeros(0)
            comp_area = (cx[:, 2] - cx[:, 0]) * (cx[:, 3] - cx[:, 1]) if len(cx) else np.zeros(0)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            extends = bool(((box_in_comp >= 0.9) & (comp_area >= 1.5 * box_area)).any()) if len(cx) else False
            if ev["blue_frac_cyan"] < 0.02:
                auto = "no_large_rock_pixels_in_box"
            elif n_inside >= 2 and ev["union_bbox_iou"] >= 0.5:
                auto = "fragmented_mask_components"
            elif extends:
                auto = "mask_component_extends_beyond_box"
            elif k in mi_p:
                auto = "matched_after_cyan_assignment"
            elif ev["union_bbox_iou"] >= 0.5 and ev["blue_frac_cyan"] >= 0.2:
                auto = "box_covers_mask_pixels_other"
            else:
                auto = "partial_mask_support"
            ev = {**ev, "n_strict_components_inside": n_inside, "mask_component_extends_beyond_box": extends}
            frame_lists = sorted(n for n, s in lists.items() if i in s)
            uncertain.append({"frame": i, "csv_line": b["line"], "box_xyxy": box, "auto_class": auto,
                              "anomaly_lists": frame_lists, "border": border, **ev})
    unc_summary = {
        "total": len(uncertain),
        "by_auto_class": dict(Counter(u["auto_class"] for u in uncertain)),
        "by_auto_class_candidate_subset": dict(Counter(u["auto_class"] for u in uncertain if not u["anomaly_lists"])),
        "by_anomaly_list": dict(Counter((u["anomaly_lists"][0] if u["anomaly_lists"] else "none") for u in uncertain)),
        "border_touching": sum(u["border"] for u in uncertain),
        "auto_class_definitions": {  # evaluated in this order
            "no_large_rock_pixels_in_box": "<2% of box pixels are blue/cyan in the ground mask",
            "fragmented_mask_components": ">=2 strict-blue components (>=25px) lie >=80% inside the box and their joint bbox covers >=50% of it",
            "mask_component_extends_beyond_box": "a strict-blue component bbox contains >=90% of the box and is >=1.5x larger (box covers part of a larger/merged mask region)",
            "matched_after_cyan_assignment": f"matched at IoU>=0.5 once cyan edge pixels count as large rock ({P})",
            "box_covers_mask_pixels_other": "joint bbox of blue/cyan pixels in the box covers >=50% of it with >=20% fill, none of the above",
            "partial_mask_support": "everything else",
            "note": "automated mechanism triage only; whether the box is a correct label is decided by visual review",
        },
    }

    # ---------------- per-frame descriptors for sampling
    desc = {}
    for i in ok:
        f = frames[i]; v = f["variants"][P]
        cb = np.array([csv_xyxy(b) for b in by_frame.get(i, [])]).reshape(-1, 4)
        S = v["small_feat"]; L = v["large_feat"]
        s50 = to_xyxy(S[S[:, F["area"]] >= 50][:, :5]) if len(S) else np.zeros((0, 4))
        l200 = to_xyxy(L[L[:, F["area"]] >= 200][:, :5]) if len(L) else np.zeros((0, 4))
        g = _gap(s50, l200)
        iou = iou_xyxy(cb, cb)
        np.fill_diagonal(iou, 0) if len(cb) else None
        desc[i] = {
            "n_csv": len(cb), "n_small50": len(s50),
            "small_near_large": int((g.min(axis=1) <= 3).sum()) if g.size else 0,
            "csv_overlap_pairs": int((np.triu(iou) > 0.05).sum()) if len(cb) > 1 else 0,
            "csv_border": int(((cb[:, 0] <= 0) | (cb[:, 1] <= 0) | (cb[:, 2] >= IMG_W) | (cb[:, 3] >= IMG_H)).sum()) if len(cb) else 0,
            "split_v0": match_cache["V0_strict"][i]["split_csv_boxes"],
            "n_uncertain": 0, "render_mean": f["render_mean"], "dark": f["dark_frac_non_sky"], "dhash": f["dhash"],
        }
    for u in uncertain:
        desc[u["frame"]]["n_uncertain"] += 1
    sampling = _sample(desc, lists, anomalous)

    return {
        "csv_schema_ok": schema["header_ok"] and not schema["parse_failures"],
        "csv_rows": len(rows), "frames_ok": len(ok), "anomalous_frames": len(anomalous),
        "candidate_subset": {"frames": len(clean_ids), "csv_boxes": sum(len(by_frame.get(i, [])) for i in clean_ids)},
        "ground_palette_mode_frames": sum(1 for i in ok if frames[i]["ground_mode"] == "P"),
        "variants": variant_metrics, "csv_generation_rule": csv_rule, "s2": s2,
        "uncertain_summary": unc_summary,
        "_frames": frames, "_by_frame": by_frame, "_lists": lists, "_anomalous": anomalous,
        "_match": match_cache, "_uncertain": uncertain, "_desc": desc, "_sampling": sampling, "_clean_ids": clean_ids,
    }


def _sample(desc: dict[int, dict[str, Any]], lists: dict[str, set[int]], anomalous: set[int]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    clean = sorted(i for i in desc if i not in anomalous)
    p90_small = np.percentile([desc[i]["n_small50"] for i in clean], 90)
    dark_cut = np.percentile([desc[i]["render_mean"] for i in clean], 2)
    hashes = defaultdict(list)
    for i, d in desc.items():
        hashes[d["dhash"]].append(i)
    dup_frames = sorted(i for g in hashes.values() if len(g) > 1 for i in g)
    pools: dict[str, tuple[list[int], Any]] = {
        "random_non_anomalous": (clean, lambda i: "uniform random draw from candidate subset"),
        "zero_csv_boxes": ([i for i in clean if desc[i]["n_csv"] == 0], lambda i: f"0 CSV boxes; {desc[i]['n_small50']} small comps>=50px"),
        "one_large_rock": ([i for i in clean if desc[i]["n_csv"] == 1], lambda i: "exactly 1 CSV box"),
        "multiple_large_rocks": ([i for i in clean if desc[i]["n_csv"] >= 4], lambda i: f"{desc[i]['n_csv']} CSV boxes"),
        "high_density_small_rocks": ([i for i in clean if desc[i]["n_small50"] >= p90_small], lambda i: f"{desc[i]['n_small50']} small comps>=50px (p90={p90_small:.0f})"),
        "small_near_large": ([i for i in clean if desc[i]["small_near_large"] >= 3], lambda i: f"{desc[i]['small_near_large']} small rocks within 3px of a large rock"),
        "touching_or_overlapping": ([i for i in clean if desc[i]["csv_overlap_pairs"] >= 1], lambda i: f"{desc[i]['csv_overlap_pairs']} overlapping CSV box pairs"),
        "very_dark": ([i for i in clean if desc[i]["render_mean"] <= dark_cut], lambda i: f"render mean {desc[i]['render_mean']:.1f} (<= p2 {dark_cut:.1f})"),
        "heavy_shadow_non_listed": ([i for i in clean if desc[i]["dark"] >= 0.4], lambda i: f"dark fraction outside sky {desc[i]['dark']:.2f}"),
        "border_objects": ([i for i in clean if desc[i]["csv_border"] >= 2], lambda i: f"{desc[i]['csv_border']} CSV boxes touch border"),
        "suspected_fragmentation": ([i for i in clean if desc[i]["split_v0"] >= 1], lambda i: f"{desc[i]['split_v0']} CSV boxes split into >=2 strict components"),
        "uncertain_csv_match": ([i for i in sorted(desc) if desc[i]["n_uncertain"] >= 1], lambda i: f"{desc[i]['n_uncertain']} uncertain CSV boxes"),
        "list_cam_anomaly": (sorted(lists["cam_anomaly"]), lambda i: "in cam_anomaly_IDs.txt"),
        "list_mismatch": (sorted(lists["mismatch"]), lambda i: "in mismatch_IDs.txt"),
        "list_shadow": (sorted(lists["shadow"]), lambda i: "in shadow_IDs.txt"),
        "list_ground_facing": (sorted(lists["ground_facing"]), lambda i: "in ground_facing_IDs.txt"),
        "exact_duplicate_pair": (dup_frames, lambda i: f"identical 64-bit dHash with {[j for j in hashes[desc[i]['dhash']] if j != i]}"),
    }
    chosen: dict[int, dict[str, Any]] = {}
    for cat, quota in QUOTAS.items():
        pool, why = pools[cat]
        if cat == "uncertain_csv_match":  # half clean, half anomalous
            parts = [[i for i in pool if i not in anomalous], [i for i in pool if i in anomalous]]
            quotas = [quota // 2, quota - quota // 2]
        elif cat == "exact_duplicate_pair":
            groups = sorted({tuple(sorted(hashes[desc[i]["dhash"]])) for i in pool})
            pick = [groups[k] for k in sorted(rng.choice(len(groups), size=min(quota // 2, len(groups)), replace=False))] if groups else []
            parts, quotas = [[i for g in pick for i in g]], [quota]
        else:
            parts, quotas = [pool], [quota]
        for part, q in zip(parts, quotas):
            avail = [i for i in part if i not in chosen]
            if cat == "exact_duplicate_pair":
                take = part
            else:
                take = sorted(rng.choice(avail, size=min(q, len(avail)), replace=False).tolist()) if avail else []
            for i in take:
                chosen.setdefault(i, {"frame_id": i, "primary_category": cat, "rationale": why(i)})
    out = []
    for i in sorted(chosen):
        cats = [c for c, (pool, _) in pools.items() if c != "random_non_anomalous" and i in set(pool)]
        lst = sorted(n for n, s in lists.items() if i in s)
        out.append({**chosen[i], "all_categories": ";".join(cats),
                    "anomaly_lists": ";".join(lst),
                    "inclusion_status": "excluded_from_candidate_subset" if lst else "candidate_subset",
                    "exclusion_reason": ("in " + ",".join(lst)) if lst else ""})
    return out
