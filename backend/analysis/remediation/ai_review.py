"""Import the AI-assisted 36-scene audit as AI evidence (never as human review) and reconcile it with Task D.

Usage (from backend/):
    uv run --no-sync python -m analysis.remediation.ai_review --source <path to taskd_ai_review_36_decisions.json>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

AI_PROVENANCE = "ai_assisted_audit"
FIELDS = ["frame_id", "split_in_audit", "split_canonical", "split_match", "in_task_d_gallery", "gallery_reason", "ai_decision",
          "flags", "notes", "reviewed_at", "duplicate_group", "human_review", "provenance"]


def load_manifest(root: Path) -> dict[int, dict[str, str]]:
    # yolo-rock-v1 was retired on 2026-10-10; its split assignment is identical to v1.1 for all 9,193 frames
    with (root / "dataset" / "processed" / "yolo-rock-v1.1" / "manifests" / "split_manifest.csv").open(newline="", encoding="utf-8") as f:
        return {int(r["frame_id"]): r for r in csv.DictReader(f)}


def run(root: Path, source: Path) -> dict[str, Any]:
    out = root / "experiments" / "yolo-dataset" / "ai_review"
    out.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    j = json.loads(raw)
    dst = out / source.name
    if not dst.exists() or dst.read_bytes() != raw:
        shutil.copyfile(source, dst)          # verbatim copy of the evidence
    man = load_manifest(root)
    with (root / "experiments" / "yolo-dataset" / "review_gallery" / "scene_list.csv").open(newline="", encoding="utf-8") as f:
        gallery = {int(r["frame_id"]): r["reason"] for r in csv.DictReader(f)}
    rows = []
    for s in j["scenes"]:
        f = int(s["frame_id"])
        canon = man.get(f, {}).get("split", "NOT_IN_MANIFEST")
        rows.append({"frame_id": f, "split_in_audit": s.get("split", ""), "split_canonical": canon,
                     "split_match": s.get("split", "") == canon, "in_task_d_gallery": f in gallery, "gallery_reason": gallery.get(f, ""),
                     "ai_decision": s["decision"], "flags": ";".join(s.get("flags", [])), "notes": s.get("notes", ""),
                     "reviewed_at": s.get("reviewed_at", ""), "duplicate_group": man.get(f, {}).get("leakage_group", ""),
                     "human_review": False, "provenance": AI_PROVENANCE})
    rows.sort(key=lambda r: r["frame_id"])
    with (out / "taskd_ai_review_36_decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    groups = Counter(r["duplicate_group"] for r in rows)
    normalized = {
        "provenance": AI_PROVENANCE, "human_review": False, "counts_as_human_approval": False,
        "reviewer_label_in_source": j.get("reviewer", ""), "source_file": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_declares_human_review": j.get("human_review"),
        "scenes": rows, "policy_in_source": j.get("policy", {}),
        "note": "AI-generated evidence. APPROVE here never approves, freezes or authorises anything.",
    }
    (out / "ai_review_normalized.json").write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    summary = {
        "scenes": len(rows), "unique_images": len({r["frame_id"] for r in rows}),
        "unique_independent_scenes": len(groups),
        "decisions": dict(Counter(r["ai_decision"] for r in rows)),
        "flags": dict(Counter(x for r in rows for x in r["flags"].split(";") if x)),
        "split_mismatches": [r["frame_id"] for r in rows if not r["split_match"]],
        "frames_not_in_gallery": [r["frame_id"] for r in rows if not r["in_task_d_gallery"]],
        "gallery_frames_not_in_audit": sorted(set(gallery) - {r["frame_id"] for r in rows}),
        "duplicate_groups_in_audit": {g: n for g, n in groups.items() if n > 1},
        "source_sha256": normalized["source_sha256"],
        "missing_companion_files": [n for n in ("taskd_ai_review_36_decisions.csv", "TASKD_AI_REVIEW_36_REPORT.md",
                                                "taskd_configB_masking_audit.csv", "taskd_ai_review_visual_sheets.zip")
                                    if not (source.parent / n).exists()],
    }
    (out / "reconciliation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.project_root.resolve(), a.source.resolve()), indent=2))


if __name__ == "__main__":
    main()
