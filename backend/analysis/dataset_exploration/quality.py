"""Aggregate measured findings into a severity-ranked data-quality report.

Every issue carries the evidence numbers it was derived from; severities follow fixed rules:
  CRITICAL - blocks Task B/C as planned;  HIGH - will bias training/evaluation if ignored;
  MEDIUM   - needs an explicit handling rule;  LOW - housekeeping / documentation.
"""

from __future__ import annotations

from typing import Any

ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _issue(id_, sev, dataset, title, evidence, impact, recommendation, status="verified"):
    return {"id": id_, "severity": sev, "dataset": dataset, "title": title, "evidence": evidence,
            "impact": impact, "recommendation": recommendation, "evidence_status": status}


def build(inv: dict[str, Any], al: dict[str, Any], nasa: dict[str, Any], ps: dict[str, Any], mt: dict[str, Any],
          errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    I = []
    # ---------------- Artificial Lunar
    cov = al["csv_vs_masks"][al["csv_vs_masks"]["best_matching_mask"]]
    lc = al["label_coverage"]
    I.append(_issue(
        "AL-01", "HIGH", "artificial-lunar", "bounding_boxes.csv annotates only large (blue-mask) rocks",
        {"csv_boxes": al["bounding_boxes"]["total_boxes"], "csv_match_rate_to_blue_components": cov["csv_match_rate"],
         "blue_components_without_csv_box": cov["components_without_csv_box"],
         "blue_components_without_csv_box_by_area": cov["components_without_csv_box_by_area"],
         "green_small_rock_components_ge50px_per_frame_median": lc["ground_green_ge50px_per_frame"].get("median"),
         "frames_without_csv_box": al["bounding_boxes"]["frames_without_boxes"]},
        "Training YOLO on the CSV alone teaches the model that visible medium/small rocks are background (label noise as false negatives).",
        "Define the target class explicitly as 'large rock' OR derive boxes from mask connected components (clean/ground) in Task C; decide in Task B after visual review."))
    if al["uncertain_annotations"]["count"]:
        I.append(_issue(
            "AL-02", "MEDIUM", "artificial-lunar", "CSV boxes without a matching blue-mask component (IoU<0.5)",
            {"count": al["uncertain_annotations"]["count"], "frames": al["uncertain_annotations"]["frames"],
             "examples": al["uncertain_annotations"]["examples"][:10]},
            "In the sampled cases (frames 11, 13, 15) the boxes are correct: anti-aliased edges fragment the ground-mask "
            "component so IoU drops. Other cases (anomaly frames) are genuinely wrong. The split is not quantified.",
            "Keep flagged as uncertain; Task B reviews a sample stratified by anomaly-list membership; do not auto-fix.",
            "partially verified (visual sample)"))
    inv_c = al["bounding_boxes"]["invalid_counts"]
    hard = {k: v for k, v in inv_c.items() if v and k not in ("xy_not_half_pixel", "non_integer_w_h")}
    I.append(_issue(
        "AL-03", "MEDIUM" if hard else "LOW", "artificial-lunar", "Geometric validity of CSV boxes",
        {"invalid_counts": inv_c, "exact_duplicates": al["bounding_boxes"]["exact_duplicate_boxes"],
         "near_duplicates_iou_gt_0.8": al["bounding_boxes"]["near_duplicate_pairs_iou_gt_0.8"],
         "tiny_boxes": al["bounding_boxes"]["tiny_boxes_area_lt16_or_side_lt3"]},
        "Out-of-bounds or degenerate boxes break YOLO conversion or produce clipped targets.",
        "Clip-and-flag (not silently fix) during conversion; keep a list of affected lines."))
    lists = al["anomaly_lists"]["lists"]
    I.append(_issue(
        "AL-04", "MEDIUM", "artificial-lunar", "Author-provided anomaly lists flag frames with known render/label problems",
        {n: l["count"] for n, l in lists.items()} | {"union_excluding_top200": al["anomaly_lists"]["union_of_quality_flag_lists_excluding_top200"]},
        "Visual review: mismatch = render geometry differs from masks (boxes off-target); cam_anomaly = corrupted masks "
        "(sky labelled as rock, rock as sky); shadow = rocks boxed inside fully black shadow; ground_facing = camera on the ground, no scene.",
        "Exclude the union of cam_anomaly, mismatch, shadow and ground_facing frames from training/validation; keep a tagged copy of "
        "shadow frames for a separate hard-case evaluation. top200_largerocks is a selection list, not a defect list.",
        "verified on 4 sampled frames per list + per-list metrics; not exhaustively"))
    rm = al.get("real_moon_images", {})
    if rm.get("image_mask_size_mismatches"):
        I.append(_issue(
            "AL-05", "LOW", "artificial-lunar", "Real-moon images and their masks differ in size",
            {"mismatches": len(rm["image_mask_size_mismatches"]), "examples": rm["image_mask_size_mismatches"][:5]},
            "Pixel-wise comparison with masks needs resizing/cropping; PCAM CSV boxes refer to image coordinates.",
            "Only use the real-moon set for qualitative / small-scale box evaluation; document the resize policy."))
    nd = al["near_duplicates"]
    if nd["pairs"]:
        I.append(_issue("AL-06", "LOW", "artificial-lunar", "Near-duplicate renders", {"pairs": nd["pairs"], "examples": nd["examples"][:10]},
                        "Possible train/val leakage.", "Assign near-duplicate groups to the same split."))

    # ---------------- NASA POLAR
    ip = nasa["image_properties"]
    I.append(_issue(
        "NP-01", "HIGH", "nasa-polar", "Images are 12-bit single-channel data in 16-bit PNG containers",
        {"mode_size_dtype": ip["mode_size_dtype"], "global_max": ip["global_max_value"], "effective_bit_depth": ip["effective_bit_depth"]},
        "Ultralytics/OpenCV default loaders will mis-scale (or 8-bit truncate) these images; the synthetic training data is 8-bit RGB.",
        "Add an explicit, documented 12-bit->8-bit conversion (fixed scale or per-image percentile) before inference; evaluate it in Task C."))
    red = {t: v["redundancy_thumbnail_correlation"]["adjacent_exposures_same_bracket"].get("median") for t, v in nasa["terrains"].items()}
    I.append(_issue(
        "NP-02", "HIGH", "nasa-polar", "Heavy redundancy: 7-8 exposure brackets x stereo x lighting of only 3 camera positions per terrain",
        {"distinct_camera_viewpoints": {t: v["distinct_camera_viewpoints"] for t, v in nasa["terrains"].items()},
         "images": {t: v["images"] for t, v in nasa["terrains"].items()},
         "median_corr_adjacent_exposures": red},
        "Metrics computed over all 1,224 images are dominated by a handful of viewpoints; random splits leak scenes.",
        "Evaluate per (terrain, position) group; pick one well-exposed exposure per bracket (or report per-exposure); never split a bracket."))
    miss = {t: v["missing_position_light_folders"] for t, v in nasa["terrains"].items() if v["missing_position_light_folders"]}
    if miss:
        I.append(_issue("NP-03", "LOW", "nasa-polar", "Incomplete acquisition folders", miss,
                        "Unbalanced lighting coverage per terrain.", "Report per-condition metrics; do not compare raw totals across terrains."))
    I.append(_issue(
        "NP-04", "MEDIUM", "nasa-polar", "LiDAR ground truth only per position (3 point clouds per terrain); CAM_LIDAR_* must be applied inverted",
        {t: {p: {"rows": g.get("rows"), "selected_transform": g.get("projection_selected"),
                 "in_image_frac": {k: x["in_image_frac"] for k, x in g.get("projection_test", {}).items()}}
             for p, g in v["ground_truth_xyz"].items()} for t, v in nasa["terrains"].items()},
        "Using the extrinsics literally (Xc = R X + t) projects 0% of points into the image; spatial evaluation would silently fail.",
        "Use Xc = R^T (X - t); add lens distortion when producing per-pixel depth; validate visually (samples/nasa_lidar_projection_*.png).",
        "verified by projection test; exact semantic of first two / last xyz columns unresolved"))

    # ---------------- POLAR-Sim
    d, s = ps["detection"], ps["segmentation"]
    I.append(_issue(
        "PS-01", "MEDIUM", "polar-sim", "Stray non-canonical label files (copies / 'good' variants / metadata / .DS_Store)",
        {"file_kind_counts": d["file_kind_counts"], "variant_copy_files": d["variant_copy_files"],
         "variants_differing_from_canonical": sum(1 for v in d["variant_copy_report"] if not all(v.get("identical_to_canonical", [False])))},
        "Naive globbing double-counts or parses metadata as annotations (cause of the earlier audit warnings).",
        "Whitelist by the canonical filename regex; never glob *.txt."))
    if d["files_with_issues"]:
        I.append(_issue(
            "PS-02", "MEDIUM", "polar-sim", "Malformed / invalid detection label lines",
            {"files_with_issues": d["files_with_issues"], "issue_type_counts": d["issue_type_counts"],
             "files": list(d["files_with_issues_detail"])[:20]},
            "Conversion would crash or create invalid targets.", "Skip-and-log these lines in Task C; do not repair in place."))
    if s["files_with_issues"]:
        I.append(_issue(
            "PS-03", "LOW" if s["files_with_issues"] < 50 else "MEDIUM", "polar-sim", "Invalid / degenerate segmentation polygons",
            {"files_with_issues": s["files_with_issues"], "issue_type_counts": s["issue_type_counts"]},
            "Degenerate polygons produce empty masks.", "Filter by area/vertex count when rasterising."))
    br = d.get("label_reuse_across_exposure_bracket", {})
    reuse = br.get("brackets_rock_boxes_identical_all_exposures", 0) / max(br.get("brackets_with_ge2_label_files", 1), 1)
    I.append(_issue(
        "PS-05", "HIGH" if reuse > 0.5 else "MEDIUM", "polar-sim",
        f"Rock boxes re-used across exposures of a bracket ({reuse:.0%} of brackets have identical rock boxes at every exposure)",
        {"label_reuse": br, "rock_boxes_mean_by_exposure": {e: v.get("mean") for e, v in d["rock_boxes_per_file_by_exposure"].items()}},
        "Very dark exposures (16-64 ms; <1% well-exposed pixels) carry the same boxes as bright ones, so rocks may be invisible "
        "inside their boxes: detection recall on those frames measures exposure, not the model.",
        "Evaluate on one well-exposed exposure per bracket (512-2048 ms); report dark-exposure results separately if at all."))
    if s.get("duplicate_keys") or s.get("files_in_folder_of_other_terrain"):
        I.append(_issue(
            "PS-06", "LOW", "polar-sim", "Segmentation files duplicated or stored in another terrain's folder",
            {"duplicate_keys": s.get("duplicate_keys"), "misfiled": s.get("files_in_folder_of_other_terrain")},
            "Folder-based globbing assigns labels to the wrong terrain.", "Key by filename, not folder; treat conflicting duplicates as ambiguous."))
    I.append(_issue(
        "PS-04", "MEDIUM", "polar-sim", "No detection labels for Terrain 06 (smooth)",
        {"detection_terrain_folders": d["terrain_folders"], "seg_files_terrain06": s["files_per_terrain"].get("06", 0)},
        "Terrain06 can only be used as a negative/false-positive test if 'no rocks' is confirmed visually (POLAR-Sim demo code sets rock_num=0 for terrain 6).",
        "Treat Terrain06 as an unverified negative set until Task B visual review.", "inferred"))
    m = mt["det"]["summary"]
    I.append(_issue(
        "MT-01", "MEDIUM", "nasa-polar+polar-sim", "NASA images without a detection label file",
        {"no_label_file": m["no_label_file"], "by_exposure": mt["det"]["no_label_file_by_exposure"]},
        "README says unlabelled photos have nothing visible; that is not verified and dark images may still contain rocks.",
        "Exclude images without label files from detection metrics (or treat as 'unknown'), do not count as negatives."))
    v = mt["validation"]["disparity_in_scope"]
    if v.get("positive_disparity_frac") is not None and v["positive_disparity_frac"] < 0.8:
        I.append(_issue("MT-02", "HIGH", "nasa-polar+polar-sim", "Stereo-disparity test does not support L/R mapping", v,
                        "Labels may be attached to the wrong camera.", "Investigate before using labels."))

    # ---------------- housekeeping
    dup = inv.get("duplicate_polar_sim_checkout")
    if dup:
        I.append(_issue("HK-01", "LOW", "repository", "Second POLAR-Sim checkout at project root",
                        {"path": dup["path"], "identical_by_path_and_size": dup["identical_by_path_and_size"]},
                        "Two sources of truth; scripts may point to different copies.", "Use dataset/polar-sim-source only; remove the root copy after team agreement."))
    empties = [k for k, v in inv["datasets"].items() if v["is_empty"]]
    if empties:
        I.append(_issue("HK-02", "LOW", "repository", "Empty dataset directories", {"dirs": empties},
                        "Possible confusion about where processed data lives.", "Reserve dataset/processed for Task C outputs; keep it out of source fingerprints."))
    if errors:
        I.append(_issue("HK-03", "MEDIUM", "analysis", "Errors logged during analysis", {"count": len(errors), "examples": errors[:20]},
                        "Some files could not be read or parsed.", "See logs/run.log."))
    return sorted(I, key=lambda x: (ORDER[x["severity"]], x["id"]))
