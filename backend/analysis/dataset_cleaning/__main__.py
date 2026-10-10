"""Run Task AI-01.C: build curated-dataset-v1 (S2-H annotations) and the cleaning reports.

Usage (from backend/):
    uv run --no-sync python -m analysis.dataset_cleaning [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..annotation_validation.al_analysis import _contained
from ..annotation_validation.config import ANOMALY_FILES
from ..annotation_validation.review_io import read_csv, write_csv
from ..dataset_exploration.artificial_lunar import load_csv, load_id_list
from ..dataset_exploration.common import LOG, ErrorLog, Paths, describe, setup_logging, write_json
from . import build, calibrate, config, visuals
from .worker import C, process_frame

AREA_BINS = ((50, 100), (100, 200), (200, 1000), (1000, 10 ** 9))
BEFORE_AFTER_SCENES = (742, 977, 2875, 676, 8894, 851, 3933, 6294, 2042, 4191, 5409, 7220, 1928, 3804, 2155, 6133)
REVIEW_STRATA = [  # name, predicate, n
    ("V_green_50_100", lambda r: r["status"] == "VALID" and r["source"] == "mask_green" and 50 <= r["area"] < 100, 16),
    ("V_green_100_200", lambda r: r["status"] == "VALID" and r["source"] == "mask_green" and 100 <= r["area"] < 200, 16),
    ("V_green_200_1000", lambda r: r["status"] == "VALID" and r["source"] == "mask_green" and 200 <= r["area"] < 1000, 16),
    ("V_green_ge1000", lambda r: r["status"] == "VALID" and r["source"] == "mask_green" and r["area"] >= 1000, 12),
    ("V_split_piece", lambda r: r["status"] == "VALID" and r["source"].endswith("_split"), 20),
    ("V_blue", lambda r: r["status"] == "VALID" and r["source"] == "mask_blue", 12),
    ("R_ambiguous", lambda r: r["status"] == "REVIEW_REQUIRED" and "ambiguous_low_visibility" in r["reasons"], 12),
    ("R_possible_merge", lambda r: r["status"] == "REVIEW_REQUIRED" and "merge" in r["reasons"], 10),
    ("X_invisible_shadow", lambda r: r["status"] == "REJECTED" and r["reasons"] == "invisible_black_shadow", 8),
    ("R_csv_shadow", lambda r: r["status"] == "REVIEW_REQUIRED" and r["reasons"] == "csv_box_in_black_shadow", 8),
]
ANN_FIELDS = ["ann_id", "frame_id", "image_path", "class_id", "class_name", "x0", "y0", "x1", "y1", "w", "h", "source", "status",
              "reasons", "flags", "parent_id", "split_neck_threshold", "csv_line", "border", *build.FEAT_OUT]


def source_fingerprint(dataset_root: Path) -> dict[str, Any]:
    """sha256 over (path, size, mtime) of source data: excludes .git internals and dataset/processed (our output)."""
    h = hashlib.sha256(); n = b = 0
    for dp, dn, fn in os.walk(dataset_root):
        rel = Path(dp).relative_to(dataset_root).as_posix()
        dn[:] = sorted(d for d in dn if d != ".git" and not (rel == "." and d == "processed"))
        for f in sorted(fn):
            p = Path(dp) / f
            st = p.stat()
            r = p.relative_to(dataset_root).as_posix()
            h.update(f"{r}\t{st.st_size}\t{st.st_mtime_ns}\n".encode()); n += 1; b += st.st_size
    return {"files": n, "bytes": b, "sha256": h.hexdigest(), "scope": "dataset/ excluding .git internals and dataset/processed"}


def code_hash() -> str:
    h = hashlib.sha256()
    base = Path(__file__).resolve().parents[1]
    for sub in ("dataset_cleaning", "annotation_validation", "dataset_exploration"):
        for p in sorted((base / sub).glob("*.py")):
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    root = args.project_root.resolve()
    exp = root / "experiments" / config.EXP_NAME
    cur = root / "dataset" / "processed" / config.DATASET_NAME
    paths = Paths(root, exp)
    for d in (exp / "before_after", exp / "logs", exp / "assistant_review", cur / "annotations", cur / "manifests", cur / "metadata"):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(exp / "logs" / "run.log")
    errors = ErrorLog()
    t0 = time.perf_counter()
    fp_before = source_fingerprint(paths.dataset_root)
    LOG.info("source fingerprint %s (%d files)", fp_before["sha256"][:16], fp_before["files"])
    al = paths.artificial_lunar

    rows, schema = load_csv(al / "bounding_boxes.csv", errors)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)
    lists = {k: set(load_id_list(al / f)["ids"]) for k, f in ANOMALY_FILES.items()}
    ids = list(range(1, 9767))
    eligible = [i for i in ids if not any(i in s for s in lists.values())]
    LOG.info("C: processing %d frames (%d eligible)", len(ids), len(eligible))
    cbx = lambda i: [[b["x"] - 0.5, b["y"] - 0.5, b["x"] - 0.5 + b["w"], b["y"] - 0.5 + b["h"]] for b in by_frame.get(i, [])]  # noqa: E731
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        frames = {f["id"]: f for f in ex.map(process_frame, [(str(al), i, cbx(i)) for i in ids], chunksize=16)}
    for f in frames.values():
        if "error" in f:
            errors.add("dataset_cleaning.process", f"frame {f['id']:04d}", f["error"])
    ok_eligible = [i for i in eligible if "error" not in frames[i]]

    # ---------------- calibration
    csv_np = {i: np.array(cbx(i)).reshape(-1, 4) for i in ids}
    cal_split = calibrate.splitter(frames, csv_np, ok_eligible)
    cal_vis = calibrate.visibility(frames, root / "experiments" / "annotation-validation")
    vis = dict(cal_vis["selected"])
    vis["fallback_abs"] = cal_vis["best_by_mode"]["absolute"] or {"edge": 100, "contrast": 25}
    neck = cal_split["selected_neck_threshold"]
    split_on = cal_split["splitting_enabled"]
    LOG.info("calibration: neck=%s split_on=%s vis=%s", neck, split_on, vis)

    # ---------------- decisions
    recs: list[dict[str, Any]] = []
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for i in ok_eligible:
        rr = build.frame_records(i, frames[i], by_frame.get(i, []), vis, neck, split_on)
        for r in rr:
            r["image_path"] = f"dataset/artificial-lunar/images/render/render{i:04d}.png"
        by_scene[i] = rr
        recs.extend(rr)
    LOG.info("records: %s", dict(Counter(r["status"] for r in recs)))

    # ---------------- duplicate / leakage groups (exact dHash confirmed by thumbnail MAD)
    groups = defaultdict(list)
    for i in ids:
        if "error" not in frames[i]:
            groups[frames[i]["dhash"]].append(i)
    dup_rows, dup_of = [], {}
    gid = 0
    for h, members in sorted(groups.items(), key=lambda kv: min(kv[1])):
        if len(members) < 2:
            continue
        base = frames[members[0]]["thumb"].astype(np.float32)
        confirmed = [members[0]] + [m for m in members[1:] if np.abs(frames[m]["thumb"].astype(np.float32) - base).mean() <= 2.0]
        if len(confirmed) >= 2:
            gid += 1
            for m in confirmed:
                dup_of[m] = f"D{gid:03d}"
            dup_rows.append({"duplicate_group": f"D{gid:03d}", "frames": ";".join(map(str, confirmed)),
                             "eligible_frames": ";".join(str(m) for m in confirmed if m in set(eligible)),
                             "method": "identical 64-bit dHash + mean |thumbnail diff| <= 2 grey levels (48x32)"})

    # ---------------- scenes manifest
    scenes = []
    for i in ids:
        f = frames[i]
        lst = sorted(k for k, s in lists.items() if i in s)
        rr = by_scene.get(i, [])
        c = Counter(r["status"] for r in rr)
        flags = []
        if "error" not in f:
            if f["render_mean"] < 40: flags.append("very_dark")
            if f["dark_frac_non_sky"] >= 0.4: flags.append("heavy_shadow")
            if f["sky_frac"] == 0: flags.append("no_sky_ground_facing_like")
            if f["ground_mode"] == "P": flags.append("palette_ground_mask")
            if rr and c["REVIEW_REQUIRED"] > c["VALID"]: flags.append("review_heavy")
            if rr and c["VALID"] == 0: flags.append("no_valid_annotation")
        scenes.append({"frame_id": i, "render_path": f"dataset/artificial-lunar/images/render/render{i:04d}.png",
                       "ground_path": f"dataset/artificial-lunar/images/ground/ground{i:04d}.png",
                       "render_sha256": f.get("render_sha256", ""), "ground_sha256": f.get("ground_sha256", ""),
                       "eligibility": "EXCLUDED" if lst or "error" in f else "ELIGIBLE",
                       "exclusion_reason": ("anomaly_list:" + ",".join(lst)) if lst else ("read_error" if "error" in f else ""),
                       "quality_flags": ";".join(flags), "csv_boxes": len(by_frame.get(i, [])),
                       "valid": c["VALID"], "review_required": c["REVIEW_REQUIRED"], "rejected": c["REJECTED"],
                       "duplicate_group": dup_of.get(i, ""), "leakage_group": dup_of.get(i, f"S{i:04d}")})

    # ---------------- write curated dataset
    write_csv(cur / "annotations" / "annotations_all.csv", recs, ANN_FIELDS)
    valid = [r for r in recs if r["status"] == "VALID"]
    write_csv(cur / "annotations" / "rock_valid.csv", valid, ANN_FIELDS)
    review = [r for r in recs if r["status"] == "REVIEW_REQUIRED"]
    write_csv(cur / "annotations" / "review_required.csv", review, ANN_FIELDS)
    img_ids = sorted({r["frame_id"] for r in valid} | set(ok_eligible))
    coco = {"info": {"description": f"Artemis-X {config.DATASET_NAME} (S2-H VALID only)", "version": config.POLICY_VERSION},
            "categories": [{"id": 0, "name": "rock"}],
            "images": [{"id": i, "file_name": f"dataset/artificial-lunar/images/render/render{i:04d}.png", "width": 720, "height": 480} for i in img_ids],
            "annotations": [{"id": k, "image_id": r["frame_id"], "category_id": 0, "bbox": [r["x0"], r["y0"], r["w"], r["h"]],
                             "area": r["area"] if r["area"] != "" else r["w"] * r["h"], "iscrowd": 0, "source": r["source"], "ann_id": r["ann_id"]}
                            for k, r in enumerate(valid)]}
    (cur / "annotations" / "rock_valid_coco.json").write_text(json.dumps(coco, separators=(",", ":")), encoding="utf-8")
    scene_fields = list(scenes[0])
    write_csv(cur / "manifests" / "scenes.csv", scenes, scene_fields)
    write_csv(cur / "manifests" / "duplicate_groups.csv", dup_rows, ["duplicate_group", "frames", "eligible_frames", "method"])
    write_csv(cur / "manifests" / "excluded_frames.csv", [s for s in scenes if s["eligibility"] == "EXCLUDED"], scene_fields)
    write_csv(exp / "review_required.csv", review, ANN_FIELDS)

    policy = {
        "policy_version": config.POLICY_VERSION, "class_map": config.CLASS_NAMES, "target_ontology": "rock",
        "coordinate_convention": "0-based pixel edges, x1/y1 exclusive, native 720x480; CSV boxes converted with x0 = X - 0.5, y0 = Y - 0.5, x1 = x0 + Length, y1 = y0 + Height (values preserved exactly)",
        "sources": {"csv_large": "bounding_boxes.csv instance boxes (large rocks)", "mask_blue": "V1_cyan blue mask component not duplicating a CSV box",
                    "mask_green": "V1_cyan green mask component", "*_split": "piece of a mask component separated by distance-transform watershed"},
        "status_semantics": {"VALID": "training target", "REVIEW_REQUIRED": "not a training target until a human decides; Task D should treat as ignore/uncertain, not as background",
                             "REJECTED": "not a target (reason recorded)"},
        "thresholds": {k: getattr(config, k) for k in ("MIN_AREA", "MIN_SIDE", "BLUE_MAX_AREA_UNSPLIT", "BLUE_COVERED_BY_CSV", "BLUE_IOU_WITH_CSV",
                                                         "GREEN_SAME_AS_CSV_IOU", "GREEN_INSIDE_CSV", "GREEN_INSIDE_CSV_AREA", "DARK_LEVEL",
                                                         "CSV_DARK_FRACTION", "CSV_SILHOUETTE_RING", "PEAK_REL", "PEAK_ABS", "NECK_REVIEW_MARGIN", "MIN_PIECE_AREA", "SOLIDITY_REVIEW")},
        "calibrated": {"visibility_rule": vis, "neck_threshold": neck, "splitting_enabled": split_on},
        "eligibility": "Artificial Lunar frames not in cam_anomaly, mismatch, shadow or ground_facing lists",
        "human_validation": "none - all statuses are automated; see experiments/dataset-cleaning/ for the assistant review sample",
    }
    write_json(cur / "metadata" / "policy.json", policy)
    write_json(cur / "metadata" / "calibration.json", {"splitter": cal_split, "visibility": cal_vis})
    write_json(cur / "metadata" / "provenance.json", {
        "dataset": config.DATASET_NAME, "policy_version": config.POLICY_VERSION, "seed": config.SEED,
        "command": "cd backend; uv run --no-sync python -m analysis.dataset_cleaning", "code_sha256": code_hash(),
        "source_fingerprint": fp_before,
        "inputs": {"renders": "dataset/artificial-lunar/images/render", "masks": "dataset/artificial-lunar/images/ground",
                   "csv": "dataset/artificial-lunar/bounding_boxes.csv", "anomaly_lists": list(ANOMALY_FILES.values()),
                   "task_b_calibration_labels": "experiments/annotation-validation/assistant_review/candidate_review.csv"},
        "per_image_hashes": "manifests/scenes.csv (render_sha256, ground_sha256)",
        "leakage_control": "manifests/scenes.csv leakage_group + manifests/duplicate_groups.csv; keep a group in one split"})

    # ---------------- metrics
    raw = {}
    for i in ok_eligible:
        f = frames[i]
        raw[i] = cbx(i) + [list(r[:4]) for cls in ("large", "small") for r in f[cls]]
    raw_n = np.array([len(raw[i]) for i in ok_eligible])
    val_n = np.array([sum(r["status"] == "VALID" for r in by_scene[i]) for i in ok_eligible])
    csv_n = np.array([len(by_frame.get(i, [])) for i in ok_eligible])

    def sizes(boxes):
        a = np.array(boxes).reshape(-1, 4)
        w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        return {"w": describe(w), "h": describe(h), "box_area": describe(w * h),
                "area_bins": {f"{lo}-{hi if hi < 10**9 else 'inf'}": int(((w * h >= lo) & (w * h < hi)).sum()) for lo, hi in ((0, 100), (100, 400), (400, 1600), (1600, 6400), (6400, 10 ** 9))}}
    cov = {"S1_csv_only": [0, 0], "S2H_valid": [0, 0], "S2H_valid_or_review": [0, 0]}
    for i in ok_eligible:
        f = frames[i]
        comps = np.vstack([f["large"][:, :4], f["small"][:, :4]]) if len(f["large"]) + len(f["small"]) else np.zeros((0, 4))
        if not len(comps):
            continue
        sets = {"S1_csv_only": csv_np[i], "S2H_valid": np.array([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in by_scene[i] if r["status"] == "VALID"]).reshape(-1, 4),
                "S2H_valid_or_review": np.array([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in by_scene[i] if r["status"] != "REJECTED"]).reshape(-1, 4)}
        for k, bx in sets.items():
            covered = (_contained(comps, bx).max(axis=1) >= 0.5) if len(bx) else np.zeros(len(comps), bool)
            cov[k][0] += int(covered.sum()); cov[k][1] += len(comps)
    reasons = Counter(x for r in recs for x in r["reasons"].split(";") if x)
    status_src = Counter((r["source"], r["status"]) for r in recs)
    automated = {
        "eligible_frames": len(ok_eligible), "excluded_frames": len(ids) - len(eligible),
        "raw_s2_boxes": {"total": int(raw_n.sum()), "per_image": describe(raw_n),
                         "definition": "CSV boxes + every V1_cyan blue/green component >= 50 px (Task B raw S2, no filtering, no dedup)"},
        "s1_csv_boxes": {"total": int(csv_n.sum()), "per_image": describe(csv_n)},
        "cleaned_valid": {"total": int(val_n.sum()), "per_image": describe(val_n), "frames_with_zero_valid": int((val_n == 0).sum())},
        "status_counts": dict(Counter(r["status"] for r in recs)),
        "status_by_source": {f"{s}|{st}": n for (s, st), n in sorted(status_src.items())},
        "reason_counts": dict(sorted(reasons.items())),
        "below_min_area_components_not_listed": {"large": int(sum(frames[i]["large_below_min"] for i in ok_eligible)),
                                                 "small": int(sum(frames[i]["small_below_min"] for i in ok_eligible))},
        "csv_instances_preserved": {"eligible_csv_boxes": int(csv_n.sum()),
                                    "csv_records": sum(1 for r in recs if r["source"] == "csv_large"),
                                    "csv_valid": status_src[("csv_large", "VALID")], "csv_review": status_src[("csv_large", "REVIEW_REQUIRED")],
                                    "csv_rejected": status_src[("csv_large", "REJECTED")]},
        "rock_component_coverage": {k: {"covered": a, "components": b, "rate": round(a / max(b, 1), 4)} for k, (a, b) in cov.items()},
        "rock_component_coverage_definition": "V1 mask components >= 50 px whose bbox lies >= 50% inside at least one box of the set",
        "box_sizes": {"raw_s2": sizes([b for i in ok_eligible for b in raw[i]]), "s1_csv": sizes([b for i in ok_eligible for b in csv_np[i]]),
                      "cleaned_valid": sizes([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in valid])},
        "split": {"parents_split": sum(1 for r in recs if r["reasons"] == "replaced_by_split_pieces"),
                  "pieces": sum(1 for r in recs if r["source"].endswith("_split")),
                  "pieces_valid": sum(1 for r in recs if r["source"].endswith("_split") and r["status"] == "VALID"),
                  "possible_merge_review": sum(1 for r in recs if "merge" in r["reasons"] and r["status"] == "REVIEW_REQUIRED"),
                  "components_with_ge2_markers": int(sum(sum(1 for s in frames[i][c + "_splits"] if s[1] >= 2) for i in ok_eligible for c in ("large", "small")))},
        "scene_flags": dict(Counter(x for s in scenes if s["eligibility"] == "ELIGIBLE" for x in s["quality_flags"].split(";") if x)),
        "duplicate_groups": len(dup_rows),
    }

    # ---------------- independent review sample + assistant review ingestion
    rng = np.random.default_rng(config.SEED)
    sample, used = [], set()
    prev_dir = root / "experiments" / config.REVIEW_SAMPLE_FROM if config.REVIEW_SAMPLE_FROM else None
    by_ann = {r["ann_id"]: r for r in recs}
    sample_changes = []
    for name, pred, n in REVIEW_STRATA:
        pool = [r for r in recs if pred(r) and r["ann_id"] not in used]
        automated.setdefault("review_strata_population", {})[name] = len(pool)
        if prev_dir is None:
            for k in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False)) if pool else []:
                r = dict(pool[k]); used.add(r["ann_id"])
                r["review_id"] = f"C{len(sample) + 1:03d}"; r["stratum"] = name
                sample.append(r)
    if prev_dir is not None:
        # reuse the previous version's sample so that its verdicts stay attached to the same boxes (matched by ann_id)
        for old in read_csv(prev_dir / "review_sample.csv"):
            r = by_ann.get(old["ann_id"])
            if r is None or r["status"] != old["status"] or r["reasons"] != old["reasons"]:
                sample_changes.append({"review_id": old["review_id"], "ann_id": old["ann_id"], "old_status": old["status"],
                                       "new_status": r["status"] if r else "MISSING", "new_reasons": r["reasons"] if r else ""})
            if r is not None:
                r = dict(r); r["review_id"] = old["review_id"]; r["stratum"] = old["stratum"]
                sample.append(r)
    write_csv(exp / "review_sample.csv", sample, ["review_id", "stratum", *ANN_FIELDS])
    review_files = visuals.review_grids(al, sample, exp / "before_after")
    verdict_src = (prev_dir or exp) / "assistant_review" / "review_verdicts.csv"
    ar = {r["ann_id"]: r for r in read_csv(verdict_src)}
    rev = defaultdict(Counter)
    for r in sample:
        v = ar.get(r["ann_id"], {}).get("verdict") if not any(c["ann_id"] == r["ann_id"] for c in sample_changes) else None
        if v:
            rev[r["stratum"]][v] += 1
    review_metrics = {
        "reviewer": "AI assistant visual assessment (NOT human validation)", "seed": config.SEED, "sample_size": len(sample),
        "sample_source": str(verdict_src.relative_to(root)).replace("\\", "/"), "sample_items_changed_since_previous_version": sample_changes,
        "assessed": sum(sum(c.values()) for c in rev.values()),
        "by_stratum": {k: {"n": sum(c.values()), "counts": dict(c),
                           "visible_rock_rate": round(c["VISIBLE_ROCK"] / sum(c.values()), 4),
                           "merged_rate": round(c["MERGED"] / sum(c.values()), 4),
                           "ambiguous_or_not_visible_rate": round((c["AMBIGUOUS"] + c["NOT_VISIBLE"]) / sum(c.values()), 4),
                           "oversplit_fragment_rate": round(c["OVERSPLIT_FRAGMENT"] / sum(c.values()), 4)} for k, c in sorted(rev.items())},
        "denominators": "per stratum = assessed items in that stratum; strata are not population-weighted",
    }
    # population-weighted estimate over the VALID mask-derived strata only (no CI claimed)
    vs = [s for s in ("V_green_50_100", "V_green_100_200", "V_green_200_1000", "V_green_ge1000", "V_split_piece", "V_blue") if rev.get(s)]
    pop = automated.get("review_strata_population", {})
    if vs:
        wsum = sum(pop[s] for s in vs)
        review_metrics["valid_mask_annotations_weighted_estimate"] = {
            k: round(sum(pop[s] / wsum * rev[s][key] / sum(rev[s].values()) for s in vs), 4)
            for k, key in (("visible_rock_rate", "VISIBLE_ROCK"), ("merged_rate", "MERGED"), ("not_visible_rate", "NOT_VISIBLE"),
                           ("ambiguous_rate", "AMBIGUOUS"), ("oversplit_fragment_rate", "OVERSPLIT_FRAGMENT"))}
        review_metrics["valid_mask_annotations_weighted_estimate"]["note"] = "stratum rates weighted by VALID population counts; small n, no confidence interval"
    task_b = {"source": "experiments/annotation-validation/strategy_metrics.json (raw S2, assistant review)"}
    tb = json.loads((root / "experiments" / "annotation-validation" / "strategy_metrics.json").read_text(encoding="utf-8"))
    task_b["by_stratum"] = {k: v for k, v in tb["sampled_review"]["candidates"]["by_stratum"].items()}
    task_b["additional_ge_50px"] = tb["sampled_review"]["candidates"]["additional_candidates_by_min_area"].get("additional_ge_50px")

    # ---------------- before/after visuals
    ba = []
    for i in BEFORE_AFTER_SCENES:
        if i in by_scene:
            ba.append(visuals.scene_sheet(al, i, raw[i], by_scene[i], exp / "before_after", "Task B reviewed scene"))
    rng2 = np.random.default_rng(config.SEED + 1)
    parents = [r for r in recs if r["reasons"] == "replaced_by_split_pieces"]
    pieces_by_parent = defaultdict(list)
    for r in recs:
        if r["parent_id"]:
            pieces_by_parent[r["parent_id"]].append(r)
    if parents:
        pick = [parents[k] for k in sorted(rng2.choice(len(parents), size=min(24, len(parents)), replace=False))]
        ba.append(visuals.split_grid(al, [(p["frame_id"], p, pieces_by_parent[p["ann_id"]]) for p in pick], exp / "before_after",
                                     "Watershed splits (random 24 accepted)", "splits_accepted.jpg"))
    merges = [r for r in recs if "merge" in r["reasons"] and r["status"] == "REVIEW_REQUIRED"]
    if merges:
        pick = [merges[k] for k in sorted(rng2.choice(len(merges), size=min(12, len(merges)), replace=False))]
        ba.append(visuals.split_grid(al, [(p["frame_id"], p, []) for p in pick], exp / "before_after",
                                     "Possible merges NOT split (REVIEW_REQUIRED, random 12)", "merges_not_split.jpg"))

    # ---------------- decisions table
    rules = [
        ("D01", "scene", "frame in cam_anomaly/mismatch/shadow/ground_facing list", "EXCLUDE scene", len(ids) - len(eligible), "Task A/B: render/mask defects"),
        ("D02", "csv_large", "always", "KEEP as instance (VALID unless D03)", automated["csv_instances_preserved"]["csv_records"], "CSV separates touching boulders"),
        ("D03", "csv_large", f">= {config.CSV_DARK_FRACTION:.0%} of box pixels darker than {config.DARK_LEVEL} AND surroundings mean < {config.CSV_SILHOUETTE_RING}", "REVIEW_REQUIRED (else VALID + flag dark_silhouette)", reasons["csv_box_in_black_shadow"], "rock invisible in render; silhouettes against lit ground stay valid"),
        ("D04", "mask", f"component area < {config.MIN_AREA} px", "NOT LISTED (counted)", sum(automated["below_min_area_components_not_listed"].values()), "Task B: 31% visible below 50 px"),
        ("D05", "mask_blue/green", "duplicates a CSV instance (blue: >=50% inside or IoU>=0.3; green: IoU>=0.5)", "REJECTED", reasons["duplicate_of_csv_instance"], "prevent double boxes"),
        ("D06", "mask", f"width or height < {config.MIN_SIDE} px", "REJECTED", reasons["thin_artifact"], "degenerate artefact"),
        ("D07", "mask", f"mean inside and around < {config.DARK_LEVEL}", "REJECTED", reasons["invisible_black_shadow"], "invisible in render"),
        ("D08", "mask", ">= 85% of pixels dark", "REVIEW_REQUIRED", reasons["mostly_in_shadow"], "partly visible"),
        ("D09", "mask", f"{vis.get('mode')}: edge < {vis['edge']} AND contrast < {vis['contrast']} (calibrated; relative = x scene median ground gradient / ground std)", "REVIEW_REQUIRED", reasons["ambiguous_low_visibility"], "Task B ambiguous textures"),
        ("D10", "mask", f"watershed neck ratio <= {neck} (calibrated; enabled={split_on})", "SPLIT into pieces (parent REJECTED replaced_by_split_pieces)", reasons["replaced_by_split_pieces"], "merged rocks"),
        ("D11", "mask", f"split only at neck in ({neck}, {None if neck is None else round(neck + config.NECK_REVIEW_MARGIN, 2)}] or solidity < {config.SOLIDITY_REVIEW}", "REVIEW_REQUIRED", reasons["possible_merge_borderline_split"] + reasons["low_solidity_possible_merge"], "possible merge, not split"),
        ("D12", "mask_blue", f"area >= {config.BLUE_MAX_AREA_UNSPLIT} px and not split", "REVIEW_REQUIRED", reasons["large_blue_without_csv_likely_merged"], "Task B: 9/10 merged"),
        ("D13", "mask_green", f">= {config.GREEN_INSIDE_CSV:.0%} inside a CSV box and >= {config.GREEN_INSIDE_CSV_AREA:.0%} of its area", "REVIEW_REQUIRED", reasons["overlaps_csv_large_rock"], "overlap with large instance"),
    ]
    write_csv(exp / "cleaning_decisions.csv", [dict(zip(["rule_id", "applies_to", "condition", "action", "count", "rationale"], r)) for r in rules],
              ["rule_id", "applies_to", "condition", "action", "count", "rationale"])

    fp_after = source_fingerprint(paths.dataset_root)
    unchanged = fp_before == fp_after
    write_json(exp / "quality_metrics.json", {
        "policy_version": config.POLICY_VERSION, "automated_full_dataset": automated, "calibration": {"splitter": cal_split, "visibility": cal_vis},
        "review_after_task_c": review_metrics, "review_before_task_b_raw_s2": task_b,
        "integrity": {"source_fingerprint_before": fp_before, "source_fingerprint_after": fp_after, "unchanged": unchanged,
                      "task_a_fingerprint": "fecca0cf45c9a9c063afd284e8267e74d9a767f5923228f23698ee49376ff72d"},
        "outputs": {"before_after": sorted(ba + review_files)}, "logged_errors": errors.items})
    write_json(cur / "metadata" / "counts.json", {"status_counts": automated["status_counts"], "status_by_source": automated["status_by_source"],
                                                   "eligible_frames": len(ok_eligible), "valid_boxes": len(valid)})
    LOG.info("source unchanged: %s | %.1f s | %d errors", unchanged, time.perf_counter() - t0, len(errors.items))
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
