"""Run Task AI-01.D: validate Task C, build leakage-free splits, export provisional YOLO datasets (configs A and B).

Usage (from backend/):
    uv run --no-sync python -m analysis.yolo_dataset [--workers 8]
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

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

from ..annotation_validation.drawing import font, grid, header, save, stack
from ..annotation_validation.review_io import write_csv
from ..dataset_cleaning.__main__ import source_fingerprint
from ..dataset_exploration.common import LOG, ErrorLog, describe, save_fig, setup_logging, write_json
from . import export, gallery, neardup, splits, ultralytics_check, validate
from .similarity import consecutive_test, thumb

SEED = 20261011
DATASET = "yolo-rock-v1"


def tree_fingerprint(*dirs: Path) -> dict[str, Any]:
    h = hashlib.sha256(); n = 0
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.is_file() and "logs" not in p.parts:
                st = p.stat()
                h.update(f"{p.as_posix()}\t{st.st_size}\t{st.st_mtime_ns}\n".encode()); n += 1
    return {"files": n, "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--curated", default="curated-dataset-v1.1", help="Task C dataset under dataset/processed")
    ap.add_argument("--taskc-exp", default="dataset-cleaning-v1.1", help="Task C experiment folder (integrity check only)")
    ap.add_argument("--out-name", default="yolo-rock-v1.1")
    ap.add_argument("--exp-name", default="yolo-dataset-v1.1")
    ap.add_argument("--reuse-splits-from", default="yolo-rock-v1.1", help="previous export whose split/leakage groups are kept ('' = recompute)")
    ap.add_argument("--link-from", default="yolo-rock-v1.1", help="previous export to hard-link identical images from ('' = copy)")
    ap.add_argument("--ai-review", default="experiments/yolo-dataset/ai_review/ai_review_normalized.json")
    a = ap.parse_args()
    if a.out_name in (a.reuse_splits_from, a.link_from):    # yolo-rock-v1 retired 2026-10-10: defaults now point at v1.1
        ap.error(f"--out-name {a.out_name} must differ from --reuse-splits-from/--link-from (would overwrite its own source)")
    root = a.project_root.resolve()
    global DATASET
    DATASET = a.out_name
    exp = root / "experiments" / a.exp_name
    cur = root / "dataset" / "processed" / a.curated
    out = root / "dataset" / "processed" / DATASET
    prev_out = root / "dataset" / "processed" / a.reuse_splits_from if a.reuse_splits_from else None
    link_from = root / "dataset" / "processed" / a.link_from if a.link_from else None
    for d in (exp / "logs", exp / "previews", exp / "figures", exp / "review_gallery", exp / "human_review", exp / "cache", out / "manifests"):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(exp / "logs" / "run.log")
    errors = ErrorLog(); t0 = time.perf_counter()
    src_before = source_fingerprint(root / "dataset")
    taskc_dirs = (cur, root / "experiments" / a.taskc_exp)
    prev_dirs = tuple(d for d in (prev_out, link_from) if d) + ((root / "experiments" / "yolo-dataset",) if prev_out else ())
    prev_before = tree_fingerprint(*prev_dirs) if prev_dirs else None
    taskc_before = tree_fingerprint(*taskc_dirs)

    # ---------------- 1. validate Task C (read-only)
    anns = validate.load_annotations(cur)
    scenes = validate.load_scenes(cur)
    val = validate.run(root, cur, anns, scenes, a.workers, errors)
    LOG.info("Task C validation all_pass=%s", val["all_pass"])
    eligible = sorted(f for f, s in scenes.items() if s["eligibility"] == "ELIGIBLE")
    by_frame: dict[int, dict[str, list]] = {f: {"VALID": [], "REVIEW_REQUIRED": [], "REJECTED": []} for f in eligible}
    anns_by_frame = defaultdict(list)
    for r in anns:
        by_frame[r["frame_id"]][r["status"]].append(r)
        anns_by_frame[r["frame_id"]].append(r)
    # Task C defect: split pieces were never re-checked against CSV instances (rule D05 applied to parents only).
    # Export-time correction, Task C files untouched: drop VALID split pieces that D05 would have rejected.
    exclusions = export_dedup(by_frame)
    write_csv(out / "manifests" / "export_exclusions.csv", exclusions, ["ann_id", "frame_id", "source", "parent_id", "duplicates_ann_id", "iou", "contained", "reason"])
    LOG.info("export-time exclusions (split pieces duplicating CSV instances): %d", len(exclusions))

    # ---------------- 2+3. scene similarity / near-duplicates and splits
    props = {f: splits.scene_props(scenes[f]) for f in eligible}
    reused = None
    if prev_out:
        # keep the canonical split assignment and leakage groups of the previous export (no recomputation)
        with (prev_out / "manifests" / "split_manifest.csv").open(newline="", encoding="utf-8") as fh:
            prev_man = {int(r["frame_id"]): r for r in csv.DictReader(fh)}
        prev_stats = json.loads((root / "experiments" / "yolo-dataset" / "split_statistics.json").read_text(encoding="utf-8"))["split_validation"]
        assert sorted(prev_man) == eligible, "eligible frames differ from the reused split manifest"
        split_of = {f: prev_man[f]["split"] for f in eligible}
        groups = {f: prev_man[f]["leakage_group"] for f in eligible}
        seq, nd = prev_stats["sequence_correlation"], prev_stats["near_duplicates"]
        edges = [(e["a"], e["b"], e["inliers"], e["kind"]) for e in prev_stats["near_duplicate_edges"]]
        reused = {"from": prev_out.relative_to(root).as_posix(),
                  "previous_config_A": {f for f in eligible if prev_man[f]["in_config_A_train"] == "True"}}
    else:
        paths = [str(root / scenes[f]["render_path"]) for f in eligible]
        with ProcessPoolExecutor(a.workers) as ex:
            T = np.stack(list(ex.map(thumb, paths, chunksize=64)))
        seq = consecutive_test(T, eligible, SEED)
        nd = neardup.run(paths, T, exp / "cache" / "orb_features.npz", a.workers, SEED)
        edges = ([(eligible[i], eligible[j], n, "duplicate") for i, j, n in nd["duplicates"]]
                 + [(eligible[i], eligible[j], n, "borderline_grouped_conservatively") for i, j, n in nd["inspect"]])
        groups = splits.leakage_groups(eligible, {f: scenes[f]["leakage_group"] for f in eligible}, edges)
        split_of = splits.assign(eligible, props, groups, SEED)
    LOG.info("near-dup edges: %s", [(x, y, n, k) for x, y, n, k in edges])
    in_a = {f for f in eligible if split_of[f] == "train" and props[f]["uncertainty"] <= splits.UNCERTAINTY_A
            and not any(r["reasons"] == "csv_box_in_black_shadow" for r in by_frame[f]["REVIEW_REQUIRED"])}

    # ---------------- 4. export + validation
    minfo, storage = export.write(root, out, split_of, in_a, by_frame, a.workers, link_from)
    LOG.info("storage: %s", storage)
    src_hash = {f: scenes[f]["render_sha256"] for f in eligible}
    ev = export.validate(out, split_of, in_a, by_frame, src_hash, a.workers)
    ev["config_B_changed_pixels_outside_mask"] = int(sum(m["changed_pixels_outside_mask"] for m in minfo.values()))
    for cfg, train in (("A", "config_A/images/train"), ("B", "config_B/images/train")):
        (out / f"data_{cfg}.yaml").write_text(
            "# PROVISIONAL - labels are automated S2-H VALID boxes, NOT human-verified. Do not train before approval.\n"
            + yaml.safe_dump({"path": out.as_posix(), "train": train, "val": "images/val", "test": "images/test",
                              "names": {0: "rock"}, "nc": 1}, sort_keys=False), encoding="utf-8")

    ev["ultralytics_verifier"] = ultralytics_check.run(out, a.workers)
    ev["export_time_exclusions"] = len(exclusions)
    LOG.info("ultralytics verifier: %s", ev["ultralytics_verifier"])

    # manifests
    man = []
    for f in eligible:
        p = props[f]; m = minfo.get(f, {})
        man.append({"frame_id": f, "split": split_of[f], "leakage_group": groups[f], "task_c_leakage_group": scenes[f]["leakage_group"],
                    "stratum": f"{p['difficulty']}|{p['density']}|{p['uncertainty_bin']}", "difficulty": p["difficulty"], "density": p["density"],
                    "annotation_state": p["annotation_state"], "n_valid": p["n_valid"], "n_review": p["n_review"], "n_rejected": p["n_rejected"],
                    "uncertainty": round(p["uncertainty"], 4), "in_config_A_train": f in in_a, "in_config_B_train": split_of[f] == "train",
                    "config_B_masked_fraction": m.get("masked_fraction", ""), "render_path": scenes[f]["render_path"], "render_sha256": scenes[f]["render_sha256"]})
    fields = list(man[0])
    write_csv(out / "manifests" / "split_manifest.csv", man, fields)
    write_csv(out / "manifests" / "config_B_masking.csv",
              [{"frame_id": f, **m} for f, m in sorted(minfo.items())], ["frame_id", "review_boxes", "review_boxes_fully_protected",
                                                                         "review_pixels_left_unmasked_due_to_valid_overlap", "masked_fraction", "changed_pixels_outside_mask"])
    ignore = {s: {f"render{f:04d}.png": [[r["x0"], r["y0"], r["x1"], r["y1"], r["ann_id"], r["reasons"]] for r in by_frame[f]["REVIEW_REQUIRED"]]
                  for f in eligible if split_of[f] == s} for s in ("val", "test")}
    (out / "manifests" / "eval_ignore_regions.json").write_text(json.dumps(
        {"note": "REVIEW_REQUIRED boxes on evaluation images (pixel xyxy, 720x480). Task E should not count detections matching these as "
                 "false positives nor these regions as missed rocks. Ultralytics val does not support ignore regions natively.",
         "regions": ignore}, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    (out / "PROVISIONAL.md").write_text(
        "# yolo-rock-v1 - PROVISIONAL, NOT FROZEN\n\n"
        "- Labels: automated S2-H VALID boxes (Task C). **No label is human-verified**, including val/test.\n"
        "- `images/val`, `images/test`: unmasked source copies, shared by both configs. Ignore regions: `manifests/eval_ignore_regions.json`.\n"
        "- `config_A`: conservative low-uncertainty train scenes (unmasked). `config_B`: all train scenes with REVIEW regions inpainted (train only).\n"
        "- NASA POLAR is not part of this dataset.\n- Do not train until the review gallery decisions and the REVIEW handling policy are approved.\n"
        "- Regenerate: `cd backend; uv run --no-sync python -m analysis.yolo_dataset`\n", encoding="utf-8")

    # ---------------- 5. split validation
    gsplits = defaultdict(set)
    for f in eligible:
        gsplits[groups[f]].add(split_of[f])
    multi = {g: sorted(s) for g, s in gsplits.items() if len(s) > 1}
    exported = {p.as_posix() for p in out.rglob("*.png")}
    sv = {
        "frames": len(eligible), "splits_disjoint_and_complete": sorted(split_of) == eligible,
        "groups_spanning_multiple_splits": multi, "duplicate_1_6658_same_split": split_of.get(1) == split_of.get(6658),
        "duplicate_1_6658_split": split_of.get(1), "near_duplicate_edges": [{"a": x, "b": y, "inliers": n, "kind": k, "split_a": split_of[x], "split_b": split_of[y]} for x, y, n, k in edges],
        "nasa_files_in_export": sum("nasa" in p.lower() for p in exported), "excluded_anomaly_frames_in_export": sum(1 for f in split_of if scenes[f]["eligibility"] != "ELIGIBLE"),
        "config_A_subset_of_train": all(split_of[f] == "train" for f in in_a),
        "sequence_correlation": seq, "near_duplicates": {k: v for k, v in nd.items() if k not in ("duplicates", "inspect")},
        "stratification_max_abs_deviation": None,
        "splits_reused_from": reused["from"] if reused else None,
        "config_A_membership_change_vs_previous": ({"added": sorted(in_a - reused["previous_config_A"]),
                                                    "removed": sorted(reused["previous_config_A"] - in_a)} if reused else None),
    }
    st = splits.stats(eligible, split_of, props)
    strata = Counter(f"{p['difficulty']}|{p['density']}|{p['uncertainty_bin']}" for p in props.values())
    dev = 0.0
    strat_table = {}
    for s_name, n in sorted(strata.items()):
        row = {k: sum(1 for f in eligible if split_of[f] == k and f"{props[f]['difficulty']}|{props[f]['density']}|{props[f]['uncertainty_bin']}" == s_name) for k in splits.FRACTIONS}
        strat_table[s_name] = {"total": n, **row}
        dev = max(dev, max(abs(row[k] / n - splits.FRACTIONS[k]) for k in row))
    sv["stratification_max_abs_deviation"] = round(dev, 4)
    sv["strata"] = strat_table

    # ---------------- 6. config comparison
    def cfg_stats(fs: list[int], masked: bool) -> dict[str, Any]:
        vb = [b for f in fs for b in by_frame[f]["VALID"]]
        areas = np.array([(b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in vb])
        rv = sum(len(by_frame[f]["REVIEW_REQUIRED"]) for f in fs)
        mf = np.array([minfo[f]["masked_fraction"] for f in fs]) if masked else np.zeros(0)
        return {"scenes": len(fs), "valid_boxes": len(vb), "valid_per_image_mean": round(len(vb) / max(len(fs), 1), 3),
                "review_boxes_in_scenes": rv,
                "review_boxes_effectively_background": 0 if masked else rv,
                "review_boxes_masked": rv - (sum(minfo[f]["review_boxes_fully_protected"] for f in fs) if masked else rv),
                "review_boxes_left_unmasked_inside_valid_protection": sum(minfo[f]["review_boxes_fully_protected"] for f in fs) if masked else None,
                "valid_box_area_bins": {lab: int(((areas >= lo) & (areas < hi)).sum()) for lab, lo, hi in
                                        (("<100", 0, 100), ("100-400", 100, 400), ("400-1600", 400, 1600), ("1600-6400", 1600, 6400), (">=6400", 6400, 1e9))},
                "valid_by_source": dict(Counter(b["source"] for b in vb)),
                "difficulty": dict(Counter(props[f]["difficulty"] for f in fs)), "density": dict(sorted(Counter(props[f]["density"] for f in fs).items())),
                "annotation_state": dict(Counter(props[f]["annotation_state"] for f in fs)),
                "masked_fraction": describe(mf) if masked else None,
                "scenes_masked_fraction_ge_0.25": int((mf >= export.HIGH_MASK_FRACTION).sum()) if masked else None}
    train = [f for f in eligible if split_of[f] == "train"]
    configs = {"A_conservative": cfg_stats(sorted(in_a), False), "B_masked": cfg_stats(train, True),
               "eval_val": cfg_stats([f for f in eligible if split_of[f] == "val"], False),
               "eval_test": cfg_stats([f for f in eligible if split_of[f] == "test"], False)}
    tradeoff = {}
    for tau in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
        fs = [f for f in train if props[f]["uncertainty"] <= tau]
        tradeoff[str(tau)] = {"train_scenes": len(fs), "valid_boxes": sum(props[f]["n_valid"] for f in fs),
                              "review_boxes_left_as_background": sum(props[f]["n_review"] for f in fs)}

    # ---------------- 7. previews + figures
    previews = _previews(root, out, exp / "previews", split_of, in_a, by_frame, minfo)
    figs = _figures(exp / "figures", eligible, split_of, props, by_frame)

    # ---------------- 8. review gallery
    sel = gallery.select_scenes(props, split_of, groups, anns_by_frame, SEED)
    gscenes = []
    for f, reason in sel:
        rec = anns_by_frame.get(f, [])
        gscenes.append({"frame_id": f, "split": split_of[f], "group": groups[f], "reason": reason,
                        "image": f"../../../{scenes[f]['render_path']}",
                        "masked_image": f"../../../dataset/processed/{DATASET}/config_B/images/train/render{f:04d}.png" if split_of[f] == "train" else "",
                        "config_note": ("train: config A + B" if f in in_a else "train: config B only") if split_of[f] == "train" else f"{split_of[f]} (unmasked evaluation)",
                        "counts": {s: sum(1 for r in rec if r["status"] == s) for s in ("VALID", "REVIEW_REQUIRED", "REJECTED")},
                        "boxes": [[r["x0"], r["y0"], r["x1"], r["y1"], r["status"], r["source"], r["reasons"], r["ann_id"]] for r in rec]})
    policy_q = [
        {"id": "review_handling", "question": "Which REVIEW_REQUIRED handling should Task E train with?", "options": ["CONFIG_A", "CONFIG_B", "BOTH_AS_ABLATION", "NEITHER"]},
        {"id": "label_policy_s2h", "question": "Approve S2-H VALID labels (50 px floor, watershed splits) for training?", "options": ["APPROVE", "REJECT", "APPROVE_WITH_CHANGES"]},
        {"id": "eval_labels", "question": "Accept provisional (non-human-verified) val/test labels for development metrics?", "options": ["ACCEPT", "REJECT"]},
        {"id": "freeze_dataset", "question": "Freeze yolo-rock-v1 for Task E?", "options": ["FREEZE", "DO_NOT_FREEZE"]},
    ]
    ai_ref = {}
    aip = root / a.ai_review
    if aip.exists():
        aj = json.loads(aip.read_text(encoding="utf-8"))
        ai_ref = {"source_sha256": aj["source_sha256"], "reviewer": aj["reviewer_label_in_source"], "human_review": False,
                  "scenes": {str(r["frame_id"]): {"decision": r["ai_decision"], "flags": [x for x in r["flags"].split(";") if x],
                                                  "notes": r["notes"]} for r in aj["scenes"]}}
    gallery.build(exp / "review_gallery" / "index.html", gscenes, policy_q, ai_ref)
    write_csv(exp / "review_gallery" / "scene_list.csv", [{"frame_id": s["frame_id"], "split": s["split"], "reason": s["reason"],
                                                           "decision": "NOT_REVIEWED"} for s in gscenes], ["frame_id", "split", "reason", "decision"])
    human = _ingest_human(exp / "human_review", {s["frame_id"] for s in gscenes})

    # ---------------- 9. integrity + outputs
    src_after = source_fingerprint(root / "dataset")
    taskc_after = tree_fingerprint(*taskc_dirs)
    prev_after = tree_fingerprint(*prev_dirs) if prev_dirs else None
    integrity = {"source_unchanged": src_before == src_after, "source_fingerprint": src_after,
                 "task_c_artifacts_unchanged": taskc_before == taskc_after, "task_c_fingerprint": taskc_after,
                 "previous_export_unchanged": prev_before == prev_after, "previous_export_fingerprint": prev_after,
                 "curated_input": cur.relative_to(root).as_posix()}
    write_json(exp / "split_statistics.json", {"seed": SEED, "fractions": splits.FRACTIONS, "splits": st, "split_validation": sv,
                                               "export_validation": ev, "configs": configs, "config_A_threshold_tradeoff_train": tradeoff,
                                               "config_A_rule": f"train scenes with REVIEW/(VALID+REVIEW) <= {splits.UNCERTAINTY_A} and no CSV rock withheld in shadow"})
    write_json(exp / "task_c_validation.json", val)
    write_json(exp / "annotation_distributions.json", _distributions(eligible, split_of, by_frame))
    status = {"dataset": DATASET, "frozen": False, "approved_for_training": False,
              "reason": "awaiting human review decisions and REVIEW-handling policy approval",
              "human_review": human, "integrity": integrity, "previews": previews, "figures": figs, "errors": errors.items,
              "task_c_validation_all_pass": val["all_pass"], "export_checks": ev, "storage": storage,
              "export_time_exclusions": len(exclusions)}
    write_json(exp / "dataset_status.json", status)
    write_json(out / "manifests" / "dataset_status.json", {k: status[k] for k in ("dataset", "frozen", "approved_for_training", "reason")})
    ok = integrity["source_unchanged"] and integrity["task_c_artifacts_unchanged"] and val["all_pass"] and not multi \
        and integrity["previous_export_unchanged"] and (not reused or len(exclusions) == 0) \
        and ev["label_lines_invalid"] == 0 and ev["label_count_mismatch"] == 0 and ev["unmasked_copies_byte_identical_to_source"] \
        and ev["config_B_changed_pixels_outside_mask"] == 0         and all(v["corrupt"] == 0 and v["labels_missing"] == 0 and not v["messages"] for v in ev["ultralytics_verifier"].values() if isinstance(v, dict))         and ev["duplicate_label_pairs_iou_gt_0.9"] == 0
    LOG.info("checks ok=%s | source unchanged=%s | task C unchanged=%s | %.1f s | %d errors", ok, integrity["source_unchanged"],
             integrity["task_c_artifacts_unchanged"], time.perf_counter() - t0, len(errors.items))
    return 0 if ok else 2


def export_dedup(by_frame: dict[int, dict[str, list]]) -> list[dict[str, Any]]:
    """Apply Task C rule D05 to VALID *_split pieces (blue: >=50% inside a CSV box or IoU>=0.3; green: IoU>=0.5)."""
    from ..annotation_validation.al_analysis import _contained
    from ..dataset_cleaning.config import BLUE_COVERED_BY_CSV, BLUE_IOU_WITH_CSV, GREEN_SAME_AS_CSV_IOU
    from ..dataset_exploration.common import iou_xyxy
    out = []
    for f, d in sorted(by_frame.items()):
        csvb = [r for r in d["VALID"] + d["REVIEW_REQUIRED"] if r["source"] == "csv_large"]
        if not csvb:
            continue
        cb = np.array([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in csvb])
        keep = []
        for r in d["VALID"]:
            if r["source"].endswith("_split"):
                b = np.array([[r["x0"], r["y0"], r["x1"], r["y1"]]])
                iou, cont = iou_xyxy(b, cb)[0], _contained(b, cb)[0]
                hit = (cont >= BLUE_COVERED_BY_CSV) | (iou >= BLUE_IOU_WITH_CSV) if r["source"].startswith("mask_blue") else iou >= GREEN_SAME_AS_CSV_IOU
                if hit.any():
                    k = int(np.argmax(np.where(hit, iou, -1)))
                    out.append({"ann_id": r["ann_id"], "frame_id": f, "source": r["source"], "parent_id": r["parent_id"],
                                "duplicates_ann_id": csvb[k]["ann_id"], "iou": round(float(iou[k]), 4), "contained": round(float(cont[k]), 4),
                                "reason": "task_c_defect: split piece duplicates CSV instance (rule D05 not applied to pieces)"})
                    continue
            keep.append(r)
        d["VALID"] = keep
    return out


def is_human_export(j: dict[str, Any]) -> bool:
    """Only exports of the v2 gallery made in its UI count as human review (v1 exports and AI audits do not)."""
    return j.get("gallery") == "artemis-rock-review-v2" and j.get("provenance") == "human_ui" and j.get("human_review") is True


def _ingest_human(d: Path, gallery_ids: set[int]) -> dict[str, Any]:
    files = sorted(d.glob("*.json"))
    rejected = []
    accepted = []
    for p in files:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            rejected.append({"file": p.name, "reason": f"unreadable: {e}"}); continue
        if is_human_export(j):
            accepted.append((p, j))
        else:
            rejected.append({"file": p.name, "reason": "not a human_ui export of gallery v2 (AI audit or legacy v1 export)"})
    if not accepted:
        return {"files": [p.name for p in files], "rejected_files": rejected, "scenes_reviewed": 0,
                "decisions": {"NOT_REVIEWED": len(gallery_ids)}, "policy": "UNDECIDED"}
    p, j = accepted[-1]
    human = [s for s in j.get("scenes", []) if s.get("source") == "human_ui" and int(s["frame_id"]) in gallery_ids]
    dec = Counter(s["decision"] for s in human)
    return {"files": [x.name for x in files], "used": p.name, "rejected_files": rejected, "reviewer": j.get("reviewer", ""),
            "scenes_reviewed": sum(v for k, v in dec.items() if k != "NOT_REVIEWED"),
            "decisions": dict(dec) | {"NOT_REVIEWED": len(gallery_ids) - sum(v for k, v in dec.items() if k != "NOT_REVIEWED")},
            "policy": j.get("policy", {})}


def _previews(root: Path, out: Path, pdir: Path, split_of, in_a, by_frame, minfo) -> list[str]:
    """Labels drawn back from the exported YOLO .txt files (round-trip check) + config B before/after."""
    rng = np.random.default_rng(SEED + 7)
    files = []
    sets = [("val", out / "images" / "val"), ("test", out / "images" / "test"), ("config_A_train", out / "config_A" / "images" / "train"),
            ("config_B_train", out / "config_B" / "images" / "train")]
    for name, d in sets:
        imgs = sorted(d.glob("*.png"))
        tiles = []
        for p in [imgs[k] for k in sorted(rng.choice(len(imgs), size=min(6, len(imgs)), replace=False))]:
            im = Image.open(p).convert("RGB"); dr = ImageDraw.Draw(im)
            for ln in Path(str(p).replace("images", "labels")).with_suffix(".txt").read_text(encoding="utf-8").splitlines():
                _, cx, cy, w, h = map(float, ln.split())
                dr.rectangle([(cx - w / 2) * 720, (cy - h / 2) * 480, (cx + w / 2) * 720 - 1, (cy + h / 2) * 480 - 1], outline=(0, 255, 120), width=1)
            t = im.resize((360, 240), Image.BILINEAR)
            cap = Image.new("RGB", (360, 258), (20, 20, 20)); cap.paste(t, (0, 18))
            ImageDraw.Draw(cap).text((4, 2), f"{p.stem} ({name})", fill=(255, 255, 255), font=font(12))
            tiles.append(cap)
        files.append(save(grid(tiles, 3, f"{name}: YOLO labels drawn back from exported .txt (green), 50% scale"), pdir / f"labels_{name}.jpg"))
    # config B before / after
    cand = sorted(f for f, m in minfo.items() if m["masked_fraction"] > 0.02)
    tiles = []
    for f in [cand[k] for k in sorted(rng.choice(len(cand), size=min(6, len(cand)), replace=False))]:
        src = Image.open(root / "dataset" / "artificial-lunar" / "images" / "render" / f"render{f:04d}.png").convert("RGB")
        dst = Image.open(out / "config_B" / "images" / "train" / f"render{f:04d}.png").convert("RGB")
        d1 = ImageDraw.Draw(src)
        for b in by_frame[f]["REVIEW_REQUIRED"]:
            d1.rectangle([b["x0"], b["y0"], b["x1"] - 1, b["y1"] - 1], outline=(255, 154, 31))
        d2 = ImageDraw.Draw(dst)
        for b in by_frame[f]["VALID"]:
            d2.rectangle([b["x0"], b["y0"], b["x1"] - 1, b["y1"] - 1], outline=(0, 255, 120))
        row = stack([src, dst], True)
        tiles.append(stack([header(row.width, [f"frame {f:04d}: left original + REVIEW boxes (orange) | right config B inpainted + VALID boxes (green); "
                                               f"masked {minfo[f]['masked_fraction']:.1%}"]), row], False))
    files.append(save(stack(tiles, False), pdir / "config_B_before_after.jpg"))
    return files


def _figures(fdir: Path, eligible, split_of, props, by_frame) -> list[str]:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, key in enumerate(("density", "difficulty", "uncertainty_bin")):
        cats = sorted({props[f][key] for f in eligible})
        for i, s in enumerate(splits.FRACTIONS):
            fs = [f for f in eligible if split_of[f] == s]
            ax[k].bar(np.arange(len(cats)) + i * 0.27, [sum(props[f][key] == c for f in fs) / len(fs) for c in cats], width=0.27, label=s)
        ax[k].set_xticks(np.arange(len(cats)) + 0.27, cats, rotation=15); ax[k].set_title(f"{key} share per split"); ax[k].legend()
    out = [save_fig(fig, fdir / "split_strata.png")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for s in splits.FRACTIONS:
        a = [np.log10(max(1.0, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))) for f in eligible if split_of[f] == s for b in by_frame[f]["VALID"]]
        ax[0].hist(a, bins=60, density=True, histtype="step", label=s)
        n = [props[f]["n_valid"] for f in eligible if split_of[f] == s]
        ax[1].hist(n, bins=np.arange(0, 120, 4), density=True, histtype="step", label=s)
    ax[0].set(title="VALID box area (log10 px², 720x480)", xlabel="log10 area"); ax[0].legend()
    ax[1].set(title="VALID boxes per image", xlabel="boxes"); ax[1].legend()
    out.append(save_fig(fig, fdir / "split_box_distributions.png"))
    return out


def _distributions(eligible, split_of, by_frame) -> dict[str, Any]:
    res = {}
    for s in splits.FRACTIONS:
        vb = [b for f in eligible if split_of[f] == s for b in by_frame[f]["VALID"]]
        w = np.array([b["x1"] - b["x0"] for b in vb]); h = np.array([b["y1"] - b["y0"] for b in vb])
        res[s] = {"valid_boxes": len(vb), "width_px": describe(w), "height_px": describe(h), "area_px": describe(w * h),
                  "by_source": dict(Counter(b["source"] for b in vb)),
                  "review_reasons": dict(Counter(x for f in eligible if split_of[f] == s for r in by_frame[f]["REVIEW_REQUIRED"] for x in r["reasons"].split(";") if x))}
    return res


if __name__ == "__main__":
    raise SystemExit(main())
