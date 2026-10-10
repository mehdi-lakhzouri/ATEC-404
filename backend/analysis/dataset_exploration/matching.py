"""A5 - deterministic NASA POLAR image <-> POLAR-Sim label matching and its validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import LOG, SEED, Paths, describe, save_fig, tone_map_12bit
from .polar_sim import IMG_H, IMG_W

ALGORITHM = [
    "1. Parse each NASA path Terrain<TT>_<name>/Pos<P>_<n>_L<on|off>/<NoSun|Sun_<az>>/Cam<L|R>_<timestamp>_<exp>.png "
    "with a strict regex (no fuzzy matching).",
    "2. Build key <TT>_<P>_<on|off>_<no|az>_<L|R>_<exp>, mapping NoSun->'no' and keeping the 4-digit exposure string. "
    "The NASA timestamp and the Pos distance suffix are not part of the key; they are only used to check key uniqueness.",
    "3. Index POLAR-Sim files whose names match the README rule exactly ('canonical'). Files whose names merely start "
    "with a valid key (e.g. '... copy.txt', '...good.txt') are 'variant copies' and are never used as a primary match.",
    "4. Image status: 'unambiguous' = exactly one NASA image for the key AND one canonical label AND (no variant copies "
    "OR all variant copies byte-identical); 'ambiguous' = >1 NASA image per key, or variant copies whose content differs; "
    "'no_label_file' = no canonical file (README states these photos have no labels, usually nothing visible; this is "
    "NOT verified and must not be used as negative ground truth).",
    "5. Exposure time alone is never used: all six key fields must agree.",
    "6. Validation: (a) stereo-disparity sign test on rock boxes of L/R label pairs (cx_L - cx_R must be > 0 for a "
    "left/right rig with T_x < 0); (b) depth implied by box disparity compared with LiDAR depth range; (c) visual overlays.",
]


def nasa_key(m: dict[str, Any]) -> str:
    return f"{m['terrain']}_{m['pos']}_{m['light']}_{m['sun']}_{m['cam']}_{m['exposure']}"


def _disparity_test(parsed_det: dict[str, Any], keys: set[str], fx: float, baseline: float) -> dict[str, Any]:
    """Pair rock boxes between L and R label files of the same shot; sign of cx_L - cx_R checks L/R semantics."""
    dx_all, depth = [], []
    pairs_files = 0
    by_shot = defaultdict(dict)
    for k in keys:
        t, p, l, s, cam, e = k.split("_")
        by_shot[(t, p, l, s, e)][cam] = k
    for shot, cams in sorted(by_shot.items()):
        if set(cams) != {"L", "R"}:
            continue
        L = [b for b in parsed_det[cams["L"]]["boxes"] if b["cls"] == 0]
        R = [b for b in parsed_det[cams["R"]]["boxes"] if b["cls"] == 0]
        if not L or not R:
            continue
        pairs_files += 1
        used = set()
        for b in sorted(L, key=lambda b: (b["cy"], b["cx"])):
            best, cost = None, 1e9
            for j, r in enumerate(R):
                if j in used:
                    continue
                dy = abs(b["cy"] - r["cy"]) * IMG_H
                sr = (b["w"] * b["h"]) / max(r["w"] * r["h"], 1e-12)
                if dy > 20 or not 0.6 < sr < 1.67:
                    continue
                c = dy + abs(np.log(sr)) * 20
                if c < cost:
                    best, cost = j, c
            if best is not None:
                used.add(best)
                d = (b["cx"] - R[best]["cx"]) * IMG_W
                dx_all.append(d)
                if d > 1:
                    depth.append(fx * baseline / d)
    dx = np.array(dx_all)
    return {
        "stereo_label_file_pairs_with_rocks": pairs_files,
        "matched_box_pairs": int(len(dx)),
        "disparity_px": describe(dx),
        "positive_disparity_frac": float((dx > 0).mean()) if len(dx) else None,
        "implied_depth_m": describe(depth),
        "note": "Greedy pairing on |dy|<=20px and area ratio; wrong pairings add noise but cannot bias the sign systematically.",
    }


def run(paths: Paths, nasa: dict[str, Any], psim: dict[str, Any], nasa_terrains: dict[str, Any]) -> dict[str, Any]:
    LOG.info("A5 matching")
    idx = psim["_index"]
    images = nasa["image_index"]
    key_to_imgs: dict[str, list[str]] = defaultdict(list)
    for m in images:
        key_to_imgs[nasa_key(m)].append(m["rel"])
    in_scope_terrains = sorted({m["terrain"] for m in images})
    det_variant_identical = {}
    for v in psim["detection"]["variant_copy_report"]:
        det_variant_identical[v["key"]] = all(v.get("identical_to_canonical", [False]))

    out: dict[str, Any] = {"algorithm": ALGORITHM, "in_scope_terrains": in_scope_terrains}
    for kind in ("det", "seg"):
        labels = idx[kind]
        variants = idx["det_variants"] if kind == "det" else {}
        conflicting = set() if kind == "det" else {
            d["key"] for d in psim["segmentation"]["duplicate_keys"] if not all(d["identical_to_kept"])}
        rows = {"unambiguous": [], "ambiguous": [], "no_label_file": []}
        for key, rels in sorted(key_to_imgs.items()):
            if len(rels) > 1:
                rows["ambiguous"].append({"key": key, "images": rels, "reason": "multiple NASA images share the key"})
                continue
            img = rels[0]
            if key in conflicting:
                rows["ambiguous"].append({"key": key, "image": img, "label": labels[key]["rel"],
                                          "reason": "several segmentation files share this key with different content"})
            elif key in labels:
                if key in variants and not det_variant_identical.get(key, False):
                    rows["ambiguous"].append({"key": key, "image": img, "label": labels[key]["rel"],
                                              "variants": variants[key], "reason": "variant copies differ from canonical"})
                else:
                    e = {"key": key, "image": img, "label": labels[key]["rel"]}
                    if key in variants:
                        e["identical_variant_copies"] = variants[key]
                    rows["unambiguous"].append(e)
            elif key in variants:
                rows["ambiguous"].append({"key": key, "image": img, "variants": variants[key], "reason": "only variant copies exist"})
            else:
                rows["no_label_file"].append({"key": key, "image": img})
        label_keys_in_scope = {k for k in labels if k[:2] in in_scope_terrains}
        unmatched_labels = sorted(k for k in label_keys_in_scope if k not in key_to_imgs)
        out_of_scope = Counter(k[:2] for k in labels if k[:2] not in in_scope_terrains)
        per_terrain = {}
        for t in in_scope_terrains:
            n_img = sum(1 for k in key_to_imgs if k.startswith(t + "_"))
            per_terrain[t] = {
                "nasa_images": n_img,
                "unambiguous": sum(r["key"].startswith(t + "_") for r in rows["unambiguous"]),
                "ambiguous": sum(r["key"].startswith(t + "_") for r in rows["ambiguous"]),
                "no_label_file": sum(r["key"].startswith(t + "_") for r in rows["no_label_file"]),
                "labels_in_repo_for_terrain": sum(1 for k in labels if k.startswith(t + "_")),
                "labels_without_nasa_image": sum(1 for k in unmatched_labels if k.startswith(t + "_")),
            }
        no_lab_exp = Counter(r["key"].split("_")[-1] for r in rows["no_label_file"])
        unmatched_reason = Counter()
        img_folders = {tuple(k.split("_")[:4]) for k in key_to_imgs}
        for k in unmatched_labels:
            unmatched_reason["acquisition folder not downloaded / absent" if tuple(k.split("_")[:4]) not in img_folders
                             else "image for this camera/exposure absent"] += 1
        out[kind] = {
            "summary": {
                "nasa_images": len(images),
                "unambiguous": len(rows["unambiguous"]),
                "ambiguous": len(rows["ambiguous"]),
                "no_label_file": len(rows["no_label_file"]),
                "coverage_unambiguous_frac": len(rows["unambiguous"]) / max(len(images), 1),
                "labels_in_scope_without_image": len(unmatched_labels),
                "labels_out_of_scope_by_terrain": dict(sorted(out_of_scope.items())),
            },
            "per_terrain": per_terrain,
            "no_label_file_by_exposure": dict(sorted(no_lab_exp.items())),
            "unmatched_label_reasons": dict(unmatched_reason),
            "unambiguous_matches": rows["unambiguous"],
            "ambiguous_matches": rows["ambiguous"],
            "unmatched_images": rows["no_label_file"],
            "unmatched_labels": unmatched_labels,
        }
    out["nasa_key_uniqueness"] = {"keys": len(key_to_imgs), "images": len(images),
                                  "keys_with_multiple_images": sum(len(v) > 1 for v in key_to_imgs.values())}
    out["examples"] = [
        {"nasa_image": r["image"], "key": r["key"], "detection_label": r["label"]}
        for r in out["det"]["unambiguous_matches"][:: max(1, len(out["det"]["unambiguous_matches"]) // 6)][:6]
    ]

    # validation (a)+(b) on all canonical detection labels of in-scope terrains (and all terrains, for context)
    calib = next(iter(nasa_terrains.values()))["calibration"]
    fx, base = calib["K_left"][0, 0], calib["derived"]["baseline_m_assuming_metres"]
    det_keys = set(psim["_parsed_det"])
    out["validation"] = {
        "disparity_in_scope": _disparity_test(psim["_parsed_det"], {k for k in det_keys if k[:2] in in_scope_terrains}, fx, base),
        "disparity_all_terrains": _disparity_test(psim["_parsed_det"], det_keys, fx, base),
        "lidar_depth_reference": {
            t: {p: v["projection_test"][v["projection_selected"]]["depth_m_in_image"]
                for p, v in info["ground_truth_xyz"].items() if v.get("readable")}
            for t, info in nasa_terrains.items()
        },
    }
    out["figures"] = _overlays(paths, out, psim)
    return out


def _overlays(paths: Paths, out: dict[str, Any], psim: dict[str, Any]) -> list[str]:
    S = paths.samples
    root = paths.nasa_polar
    rng = np.random.default_rng(SEED)
    files = []
    det = psim["_parsed_det"]
    seg = psim["_parsed_seg"]
    um = out["det"]["unambiguous_matches"]
    with_rocks = [r for r in um if any(b["cls"] == 0 for b in det[r["key"]]["boxes"])]
    pick = []
    for t in sorted({r["key"][:2] for r in with_rocks}):
        cand = [r for r in with_rocks if r["key"].startswith(t)]
        pick += [cand[i] for i in sorted(rng.choice(len(cand), size=min(2, len(cand)), replace=False))]
    # one stereo pair, to visualise disparity consistency
    lr = None
    for r in with_rocks:
        if r["key"].startswith("11_") and "_L_" in r["key"]:
            rk = r["key"].replace("_L_", "_R_")
            if rk in det:
                lr = (r, next((x for x in um if x["key"] == rk), None))
                if lr[1]:
                    break
    rows = pick + ([lr[0], lr[1]] if lr and lr[1] else [])
    if rows:
        fig, ax = plt.subplots(len(rows), 1, figsize=(12, 6.3 * len(rows)))
        for a, r in zip(np.atleast_1d(ax), rows):
            img = tone_map_12bit(np.asarray(Image.open(root / r["image"])))
            a.imshow(img, cmap="gray")
            for b in det[r["key"]]["boxes"]:
                x0, y0 = (b["cx"] - b["w"] / 2) * IMG_W, (b["cy"] - b["h"] / 2) * IMG_H
                a.add_patch(plt.Rectangle((x0, y0), b["w"] * IMG_W, b["h"] * IMG_H, fill=False,
                                          ec="#ff2bd6" if b["cls"] == 0 else "#2bf0ff", lw=1.2))
            a.set_title(f"{r['image']}  <->  {r['label']}  (magenta=rock, cyan=shadow)", fontsize=8); a.axis("off")
        fig.tight_layout()
        files.append(save_fig(fig, S / "match_detection_overlays.png"))
    seg_um = [r for r in out["seg"]["unambiguous_matches"] if r["key"] in seg]
    if seg_um:
        sel = [seg_um[i] for i in sorted(rng.choice(len(seg_um), size=min(3, len(seg_um)), replace=False))]
        colours = {0: (0.5, 0.5, 0.5), 1: (0.2, 0.8, 0.2), 2: (1.0, 0.2, 0.8), 3: (0.2, 0.6, 1.0)}
        fig, ax = plt.subplots(len(sel), 1, figsize=(12, 6.3 * len(sel)))
        for a, r in zip(np.atleast_1d(ax), sel):
            a.imshow(tone_map_12bit(np.asarray(Image.open(root / r["image"]))), cmap="gray")
            txt = (paths.polar_sim / r["label"]).read_text(encoding="utf-8").splitlines()
            for line in txt:
                tok = line.split()
                if len(tok) < 7:
                    continue
                c = int(tok[0]); pts = np.array([float(t) for t in tok[1:]]).reshape(-1, 2) * [IMG_W, IMG_H]
                a.add_patch(plt.Polygon(pts, closed=True, fill=c in (2, 3), alpha=0.35 if c in (2, 3) else 1,
                                        ec=colours.get(c), fc=colours.get(c), lw=1))
            a.set_title(f"{r['image']}  <->  {r['label']}\n(grey=background, green=ground outline, magenta=rocks, blue=shadows)", fontsize=8)
            a.axis("off")
        fig.tight_layout()
        files.append(save_fig(fig, S / "match_segmentation_overlays.png"))
    return files
