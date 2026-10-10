"""Run Task AI-01.B (annotation visualisation & label-quality validation).

Usage (from backend/):
    uv run --no-sync python -m analysis.annotation_validation [--workers 8] [--out ../experiments/annotation-validation]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from ..dataset_exploration import inventory
from ..dataset_exploration.common import LOG, ErrorLog, Paths, setup_logging, write_json
from . import al_analysis, al_outputs, nasa_review, review_io, summarize
from .config import PRIMARY_VARIANT, PROPOSED_MIN_AREA, SEED

SAMPLING_FIELDS = ["frame_id", "primary_category", "all_categories", "anomaly_lists", "inclusion_status", "exclusion_reason", "rationale"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = (args.out or root / "experiments" / "annotation-validation").resolve()
    paths = Paths(root, out)
    for d in ("figures", "logs", "s1_csv_overlays", "s2_mask_overlays", "comparison_sheets", "uncertain_boxes",
              "nasa_terrain06", "nasa_exposure_review", "gallery", "assistant_review"):
        (out / d).mkdir(parents=True, exist_ok=True)
    setup_logging(out / "logs" / "run.log")
    errors = ErrorLog()
    t0 = time.perf_counter()
    fp_before = inventory.fingerprint(paths.dataset_root, errors)

    al = al_analysis.run(paths, errors, args.workers)
    manifest = al["_sampling"]
    review_io.write_csv(out / "sampling_manifest.csv", manifest, SAMPLING_FIELDS)
    LOG.info("sampled %d scenes (%d candidate subset)", len(manifest), sum(m["inclusion_status"] == "candidate_subset" for m in manifest))

    scene_files = al_outputs.render_scenes(paths, out, al, manifest)
    cand = al_outputs.candidate_review(paths, out, al)
    unc = al_outputs.uncertain_review(paths, out, al)
    review_io.write_csv(out / "s2_mask_overlays" / "candidate_review" / "candidates.csv", cand, list(cand[0]))
    review_io.write_csv(out / "uncertain_boxes" / "uncertain_sample.csv", unc, list(unc[0]))
    review_io.write_csv(out / "uncertain_boxes" / "uncertain_all_auto_triage.csv",
                        [{**u, "anomaly_lists": ";".join(u["anomaly_lists"]), "box_xyxy": " ".join(f"{v:.1f}" for v in u["box_xyxy"])} for u in al["_uncertain"]],
                        ["frame", "csv_line", "box_xyxy", "auto_class", "anomaly_lists", "border", "blue_frac_strict", "blue_frac_cyan", "union_bbox_iou", "render_mean_in_box"])

    t06 = nasa_review.terrain06_review(paths, out)
    expo = nasa_review.exposure_review(paths, out, args.workers)
    assistant = review_io.load_assistant(out)

    # ---------------- human review manifest (all decisions default NOT_REVIEWED)
    rel = lambda p: Path(p).relative_to(out).as_posix()  # noqa: E731
    rows = []
    for m in manifest:
        fid = m["frame_id"]; f = scene_files[fid]; a = assistant["scenes"].get(str(fid), {})
        for strat, ov, cnt in (("S1", f["s1_side_by_side"], f["counts"]["csv"]),
                               ("S2", f["s2_overlay"], f["counts"]["s2_large_matched"] + f["counts"]["s2_large_new"] + f["counts"]["s2_small"])):
            rows.append({"review_item_id": f"{strat}-{fid:04d}", "frame_id": fid, "dataset": "artificial-lunar",
                         "sample_reason": f"{m['primary_category']} ({m['rationale']})", "strategy": strat,
                         "raw_image_path": f"dataset/artificial-lunar/images/render/render{fid:04d}.png",
                         "overlay_path": rel(ov), "comparison_path": rel(f["comparison"]), "candidate_count": cnt,
                         "review_status": "NOT_REVIEWED", "reviewer_decision": "NOT_REVIEWED", "reviewer_notes": "",
                         "assistant_assessment": a.get(f"assistant_{strat.lower()}_decision", ""), "assistant_notes": a.get("notes", "")})
    for c in cand:
        a = assistant["candidates"].get(c["candidate_id"], {})
        rows.append({"review_item_id": c["candidate_id"], "frame_id": c["frame_id"], "dataset": "artificial-lunar",
                     "sample_reason": f"S2 candidate stratum {c['stratum']}", "strategy": "S2_candidate",
                     "raw_image_path": f"dataset/artificial-lunar/images/render/render{c['frame_id']:04d}.png",
                     "overlay_path": "s2_mask_overlays/candidate_review/", "comparison_path": "", "candidate_count": 1,
                     "review_status": "NOT_REVIEWED", "reviewer_decision": "NOT_REVIEWED", "reviewer_notes": "",
                     "assistant_assessment": a.get("verdict", ""), "assistant_notes": a.get("notes", "")})
    for u in unc:
        a = assistant["uncertain"].get(u["uncertain_id"], {})
        rows.append({"review_item_id": u["uncertain_id"], "frame_id": u["frame_id"], "dataset": "artificial-lunar",
                     "sample_reason": f"uncertain CSV box ({u['stratum']}, line {u['csv_line']})", "strategy": "S1_uncertain_box",
                     "raw_image_path": f"dataset/artificial-lunar/images/render/render{u['frame_id']:04d}.png",
                     "overlay_path": "uncertain_boxes/", "comparison_path": "", "candidate_count": 1,
                     "review_status": "NOT_REVIEWED", "reviewer_decision": "NOT_REVIEWED", "reviewer_notes": "",
                     "assistant_assessment": a.get("verdict", ""), "assistant_notes": a.get("notes", "")})
    for k, im in enumerate(t06["selected_images"], start=1):
        a = assistant["t06"].get(im["image"], {})
        rows.append({"review_item_id": f"T06-{k:02d}", "frame_id": "", "dataset": "nasa-polar",
                     "sample_reason": f"Terrain06 negative check Pos{im['position']} Cam{im['camera']} {im['condition']}", "strategy": "NEGATIVE_SET",
                     "raw_image_path": f"dataset/nasa-polar/{im['image']}", "overlay_path": im["enhanced_full_res"], "comparison_path": "",
                     "candidate_count": 0, "review_status": "NOT_REVIEWED", "reviewer_decision": "NOT_REVIEWED", "reviewer_notes": "",
                     "assistant_assessment": a.get("verdict", ""), "assistant_notes": a.get("notes", "")})
    for s in expo["sheets"]:
        a = {k: v for k, v in assistant["exposure"].items() if v.get("sheet") == s}
        rows.append({"review_item_id": f"EXP-{Path(s).stem}", "frame_id": "", "dataset": "nasa-polar+polar-sim",
                     "sample_reason": "dark-exposure label reliability", "strategy": "POLAR_SIM_LABELS",
                     "raw_image_path": "", "overlay_path": f"nasa_exposure_review/{s}", "comparison_path": "", "candidate_count": "",
                     "review_status": "NOT_REVIEWED", "reviewer_decision": "NOT_REVIEWED", "reviewer_notes": "",
                     "assistant_assessment": ";".join(sorted({f"{v.get('exposure')}:{v.get('verdict')}" for v in a.values()})), "assistant_notes": ""})
    manifest_name = review_io.write_human_manifest(out, rows)
    human = review_io.human_review_status(out)

    # ---------------- gallery
    scenes = []
    for m in manifest:
        fid = m["frame_id"]; f = scene_files[fid]; a = assistant["scenes"].get(str(fid), {})
        scenes.append({"frame_id": fid, "categories": [m["primary_category"]] + [c for c in m["all_categories"].split(";") if c and c != m["primary_category"]],
                       "anomaly_lists": m["anomaly_lists"], "counts": f["counts"], "comparison": rel(f["comparison"]),
                       "s1_side_by_side": rel(f["s1_side_by_side"]), "s2_overlay": rel(f["s2_overlay"]),
                       "assistant": {k: a.get(k, "") for k in ("s1_misses_visible_rocks", "s2_splits", "s2_merges", "s2_artifacts", "notes")}})
    sections = {
        "Contact sheets (5 scenes each)": sorted(rel(p) for p in (out / "comparison_sheets").glob("contact_*.jpg")),
        "S2 candidate crops": sorted(rel(p) for p in (out / "s2_mask_overlays" / "candidate_review").glob("*.jpg")),
        "Uncertain CSV boxes": sorted(rel(p) for p in (out / "uncertain_boxes").glob("*.jpg")),
        "NASA Terrain06": sorted(rel(p) for p in (out / "nasa_terrain06").glob("*.jpg")),
        "NASA exposure review": sorted(rel(p) for p in (out / "nasa_exposure_review").glob("*.jpg")),
    }
    gal = review_io.gallery(out, scenes, sections)

    # ---------------- metrics / findings / decision
    figs = summarize.figures(out, al, expo)
    rev = summarize.review_metrics(assistant, cand, unc, manifest)
    full = {k: al[k] for k in ("csv_rows", "frames_ok", "anomalous_frames", "candidate_subset", "ground_palette_mode_frames",
                               "variants", "csv_generation_rule", "s2", "uncertain_summary")}
    metrics = {
        "config": {"seed": SEED, "primary_variant": PRIMARY_VARIANT, "proposed_min_area_px": PROPOSED_MIN_AREA,
                   "match_iou": 0.5, "csv_convention": "x0 = X - 0.5, y0 = Y - 0.5, x1 = x0 + Length, y1 = y0 + Height"},
        "full_dataset_automated": full,
        "sampled_review": rev,
        "nasa_exposure_automated": {k: v for k, v in expo.items() if k != "sheets"},
        "terrain06_inputs": t06,
        "sampling": {"scenes": len(manifest), "candidate_subset_scenes": sum(m["inclusion_status"] == "candidate_subset" for m in manifest),
                     "by_primary_category": dict(sorted(Counter(m["primary_category"] for m in manifest).items()))},
        "figures": figs,
    }
    write_json(out / "strategy_metrics.json", metrics)
    findings_p = out / "assistant_review" / "quality_findings.json"
    findings = json.loads(findings_p.read_text(encoding="utf-8")) if findings_p.exists() else []
    write_json(out / "quality_findings.json", {"author": "AI assistant (automated evidence + assistant visual review; not human-validated)",
                                               "findings": findings, "logged_errors": errors.items})
    s2c = al["s2"]["candidate_subset"]
    key = {
        "candidate_subset_frames": al["candidate_subset"]["frames"], "candidate_subset_csv_boxes": al["candidate_subset"]["csv_boxes"],
        "csv_match_rate_V0_strict": al["variants"]["V0_strict"]["candidate_subset"]["match_rate"],
        f"csv_match_rate_{PRIMARY_VARIANT}": al["variants"][PRIMARY_VARIANT]["candidate_subset"]["match_rate"],
        f"additional_s2_candidates_ge_{PROPOSED_MIN_AREA}px_per_image_mean": s2c[f"min_area_{PROPOSED_MIN_AREA}"]["additional_small_medium_per_image"]["mean"]
            + s2c[f"min_area_{PROPOSED_MIN_AREA}"]["additional_large_without_csv_per_image"]["mean"],
        "uncertain_auto_triage": al["uncertain_summary"]["by_auto_class"],
        "candidate_review_by_stratum": {k: v["pass_rate"] for k, v in rev["candidates"]["by_stratum"].items()},
        "nasa_policy_rock_box_visibility": expo["policy"]["rock_box_visibility_in_selection"],
    }
    write_json(out / "strategy_decision.json", summarize.decision(out, key, human))

    fp_after = inventory.fingerprint(paths.dataset_root, errors)
    unchanged = fp_before["data"] == fp_after["data"]
    write_json(out / "logs" / "integrity.json", {"before": fp_before["data"], "after": fp_after["data"], "unchanged": unchanged,
                                                "manifest_written": manifest_name, "gallery": gal, "errors": errors.items})
    LOG.info("source data unchanged: %s | done in %.1f s, %d logged errors", unchanged, time.perf_counter() - t0, len(errors.items))
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
