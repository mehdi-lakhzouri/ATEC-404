"""Read-only validation of the Task C curated-dataset-v1 outputs. Never modifies them."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ..dataset_exploration.artificial_lunar import load_csv
from ..dataset_exploration.common import ErrorLog

STATUSES = {"VALID", "REVIEW_REQUIRED", "REJECTED"}


def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _f(x: str) -> float:
    return float(x)


def load_annotations(cur: Path) -> list[dict[str, Any]]:
    with (cur / "annotations" / "annotations_all.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["frame_id"] = int(r["frame_id"])
        for k in ("x0", "y0", "x1", "y1", "w", "h"):
            r[k] = _f(r[k])
    return rows


def load_scenes(cur: Path) -> dict[int, dict[str, Any]]:
    with (cur / "manifests" / "scenes.csv").open(newline="", encoding="utf-8") as fh:
        return {int(r["frame_id"]): r for r in csv.DictReader(fh)}


def run(root: Path, cur: Path, anns: list[dict[str, Any]], scenes: dict[int, dict[str, Any]], workers: int,
        errors: ErrorLog) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            errors.add("validate", name, json.dumps(detail, default=str)[:500])

    st = Counter(r["status"] for r in anns)
    counts = json.loads((cur / "metadata" / "counts.json").read_text(encoding="utf-8"))
    check("status_counts_match_counts_json", dict(st) == counts["status_counts"], {"annotations": dict(st), "counts.json": counts["status_counts"]})
    check("statuses_in_enum", set(st) <= STATUSES, sorted(st))
    check("ann_id_unique", len({r["ann_id"] for r in anns}) == len(anns), len(anns))
    bad_reason = [r["ann_id"] for r in anns if r["status"] != "VALID" and not r["reasons"]]
    check("non_valid_have_reasons", not bad_reason, bad_reason[:10])
    bad_geom = [r["ann_id"] for r in anns if not (0 <= r["x0"] < r["x1"] <= 720 and 0 <= r["y0"] < r["y1"] <= 480)
                or abs((r["x1"] - r["x0"]) - r["w"]) > 0.02 or abs((r["y1"] - r["y0"]) - r["h"]) > 0.02]
    check("coordinates_in_bounds_and_consistent", not bad_geom, {"bad": len(bad_geom), "examples": bad_geom[:10]})
    for name, fn, status in (("rock_valid.csv", "rock_valid.csv", "VALID"), ("review_required.csv", "review_required.csv", "REVIEW_REQUIRED")):
        with (cur / "annotations" / fn).open(newline="", encoding="utf-8") as fh:
            ids = [r["ann_id"] for r in csv.DictReader(fh)]
        exp = [r["ann_id"] for r in anns if r["status"] == status]
        check(f"{name}_equals_{status}_subset", ids == exp, {"file": len(ids), "expected": len(exp)})
    coco = json.loads((cur / "annotations" / "rock_valid_coco.json").read_text(encoding="utf-8"))
    vmap = {r["ann_id"]: r for r in anns if r["status"] == "VALID"}
    coco_bad = [a["ann_id"] for a in coco["annotations"] if a["ann_id"] not in vmap
                or abs(a["bbox"][0] - vmap[a["ann_id"]]["x0"]) > 0.01 or abs(a["bbox"][2] - vmap[a["ann_id"]]["w"]) > 0.01]
    check("coco_matches_valid", len(coco["annotations"]) == len(vmap) and not coco_bad, {"coco": len(coco["annotations"]), "valid": len(vmap), "bad": coco_bad[:5]})

    # CSV instance preservation: every eligible CSV row present, coordinates = X-0.5 etc.
    rows, _ = load_csv(root / "dataset" / "artificial-lunar" / "bounding_boxes.csv", ErrorLog())
    eligible = {f for f, s in scenes.items() if s["eligibility"] == "ELIGIBLE"}
    by_line = {r["line"]: r for r in rows}
    csv_recs = [r for r in anns if r["source"] == "csv_large"]
    mism = []
    for r in csv_recs:
        b = by_line.get(int(r["csv_line"]))
        if b is None or b["frame"] != r["frame_id"] or abs(r["x0"] - (b["x"] - 0.5)) > 1e-6 or abs(r["y0"] - (b["y"] - 0.5)) > 1e-6 \
                or abs(r["w"] - b["w"]) > 1e-6 or abs(r["h"] - b["h"]) > 1e-6:
            mism.append(r["ann_id"])
    exp_lines = {r["line"] for r in rows if r["frame"] in eligible}
    got_lines = {int(r["csv_line"]) for r in csv_recs}
    check("csv_instances_preserved_exactly", not mism and exp_lines == got_lines,
          {"csv_records": len(csv_recs), "eligible_csv_rows": len(exp_lines), "coord_mismatch": len(mism), "missing_lines": len(exp_lines - got_lines)})

    # per-scene counts
    per = defaultdict(Counter)
    for r in anns:
        per[r["frame_id"]][r["status"]] += 1
    scene_bad = [f for f in eligible if (int(scenes[f]["valid"]), int(scenes[f]["review_required"]), int(scenes[f]["rejected"]))
                 != (per[f]["VALID"], per[f]["REVIEW_REQUIRED"], per[f]["REJECTED"])]
    check("scene_counts_match_annotations", not scene_bad, scene_bad[:10])
    check("annotations_only_on_eligible_frames", set(per) <= eligible, sorted(set(per) - eligible)[:10])

    # source hashes (re-computed from the source files)
    fids = sorted(scenes)
    with ProcessPoolExecutor(workers) as ex:
        rh = list(ex.map(_sha, [str(root / scenes[f]["render_path"]) for f in fids], chunksize=64))
        gh = list(ex.map(_sha, [str(root / scenes[f]["ground_path"]) for f in fids], chunksize=64))
    hb = [f for f, a, b in zip(fids, rh, gh) if a != scenes[f]["render_sha256"] or b != scenes[f]["ground_sha256"]]
    check("source_image_hashes_match", not hb, {"frames_checked": len(fids), "mismatch": hb[:10]})

    # zero-VALID discrepancy (281 vs 128)
    zero = [f for f in eligible if per[f]["VALID"] == 0]
    flagged = [f for f in zero if "no_valid_annotation" in scenes[f]["quality_flags"]]
    norec = [f for f in zero if sum(per[f].values()) == 0]
    discrepancy = {
        "eligible_frames_with_zero_valid": len(zero), "flagged_no_valid_annotation": len(flagged), "frames_with_no_records_at_all": len(norec),
        "explained": len(flagged) + len(norec) == len(zero),
        "cause": "Task C sets no_valid_annotation only when a frame has >=1 record (`if rr and c['VALID'] == 0`). "
                 "Frames with no CSV box and no mask component >= 50 px have no records, so they are unflagged. Not a data error.",
        "no_record_frames_with_palette_mask": sum("palette_ground_mask" in scenes[f]["quality_flags"] for f in norec),
        "palette_check": "palette masks decode to pure class colours (verified); they are near-empty masks the PNG encoder stored as palette",
        "visual_check": "16 random no-record frames: rock-free slopes; 3 show only tiny distant rocks below the 50 px policy floor (f249, f2644, f9009)",
        "task_d_handling": "derived scene property `annotation_state` in Task D manifests: has_valid / records_no_valid / no_records; Task C files unchanged",
    }
    check("zero_valid_discrepancy_explained", discrepancy["explained"], discrepancy)
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values()), "zero_valid_discrepancy": discrepancy,
            "no_record_frames": sorted(norec), "flagged_frames": sorted(flagged)}
