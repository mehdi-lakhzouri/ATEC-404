"""Build docs/audits/RETENTION_MANIFEST.json: one record per significant workspace item, with hard-link-aware sizes.

For each item: apparent bytes, physical bytes (unique files), reclaimable bytes if deleted ALONE, last modification,
consumers, canonical flag, classification (PROTECTED / REVIEW / SAFE), recommended action and removal risk.
For deletion SCENARIOS (sets of items) it computes the bytes actually freed when the whole set is removed together,
which differs from the sum of single-item figures whenever items share hard links.

Read-only. Usage:  uv run --project backend --no-sync python scripts/retention_manifest.py
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TE, TF, DP = "experiments/task-e", "experiments/task-f", "dataset/processed"
RUN = f"{TE}/full-training-config-a-200-s42"

# (path or glob relative to ROOT, kind, canonical, consumers, classification, action, risk, rationale)
ITEMS = [
    (".git/objects", "git object store", False, "none (no commits, no index)", "REVIEW", "DELETE via `git prune`/`git gc --prune=now`", "LOW",
     "47,954 dangling loose blobs left by a discarded `git add`; no commit or index references them. They are copies of workspace files "
     "as staged on 2026-10-09 (some may be earlier versions of since-regenerated outputs)."),
    ("dataset/artificial-lunar/archive.zip", "source archive", False, "none (pipeline reads the extracted files)", "REVIEW",
     "DELETE only if an external copy is confirmed (re-download or backup)", "MEDIUM",
     "CRC-32 identical to the extracted dataset (29,378/29,378 members). It is the pristine original download; the extracted copy is canonical for the pipeline."),
    ("dataset/artificial-lunar/images", "raw source images", True, "Tasks A-D, data cleaning, YOLO export, ignore regions, inference demos", "PROTECTED", "KEEP", "HIGH",
     "Canonical raw renders/ground/clean images; hard-link source of nothing (YOLO exports are re-encoded)."),
    ("dataset/artificial-lunar/real_moon_images", "raw source images", True, "Task A/B qualitative checks", "PROTECTED", "KEEP", "HIGH",
     "36 real lunar photos + masks; provenance not documented in the folder."),
    ("dataset/artificial-lunar/bounding_boxes.csv", "raw source labels", True, "Task C cleaning (S2-H policy)", "PROTECTED", "KEEP", "HIGH", "Original CSV boxes."),
    ("dataset/nasa-polar/Terrain01_LargeOnly.zip", "source archive", False, "none", "REVIEW", "DELETE only if an external copy is confirmed", "MEDIUM",
     "CRC-32 identical to the extracted folder (340/340)."),
    ("dataset/nasa-polar/Terrain06_Smooth.zip", "source archive", False, "none", "REVIEW", "DELETE only if an external copy is confirmed", "MEDIUM",
     "CRC-32 identical to the extracted folder (448/448)."),
    ("dataset/nasa-polar/Terrain11_FreshCrater.zip", "source archive", False, "none", "REVIEW", "DELETE only if an external copy is confirmed", "MEDIUM",
     "CRC-32 identical to the extracted folder (448/448)."),
    ("dataset/nasa-polar/Terrain01_LargeOnly", "raw source images", True, "Task A/B exposure review; future real-domain evaluation", "PROTECTED", "KEEP", "HIGH", "12-bit stereo PNGs."),
    ("dataset/nasa-polar/Terrain06_Smooth", "raw source images", True, "Task A/B", "PROTECTED", "KEEP", "HIGH", ""),
    ("dataset/nasa-polar/Terrain11_FreshCrater", "raw source images", True, "Task A/B", "PROTECTED", "KEEP", "HIGH", ""),
    ("dataset/polar-sim-source", "external git clone (CC0)", True, "Task A/B code (dataset_exploration paths), preflight fingerprints", "PROTECTED", "KEEP", "LOW",
     "Canonical POLAR-Sim clone (github.com/uwsbel/POLAR-Sim, commit 593ed65); re-clonable."),
    ("polar-sim-source", "external git clone (CC0)", False, "none (Task A flagged it as a duplicate)", "REVIEW", "DELETE", "LOW",
     "Second clone of the same repository at the same commit 593ed65."),
    ("dataset/polar-sim", "empty directory", False, "none", "SAFE", "DELETE", "NONE", "Empty."),
    (f"{DP}/VERSIONS.json", "dataset registry", True, "preflight, training, reports", "PROTECTED", "KEEP", "HIGH", "Content hashes of every dataset version."),
    (f"{DP}/curated-dataset-v1.1", "curated dataset (current)", True, "YOLO export v1.1, remediation, notebooks", "PROTECTED", "KEEP", "HIGH", "S2-H/1.1 annotations."),
    (f"{DP}/curated-dataset-v1", "curated dataset (archived)", False, "preflight fingerprint; dataset_cleaning config PREVIOUS_DATASET (provenance)", "REVIEW",
     "ARCHIVE metadata (counts/policy/hashes already in VERSIONS.json), then DELETE", "LOW", "Superseded by v1.1; 0.24 GB of CSVs."),
    (f"{DP}/yolo-rock-v1.1/config_A", "YOLO dataset (trained)", True, "E.0/E.1a/E.1c training, data views", "PROTECTED", "KEEP", "HIGH", "Config A train/val/test (1,903 train)."),
    (f"{DP}/yolo-rock-v1.1/images", "YOLO dataset (unmasked val/test)", True, "evaluation, Task F, notebooks", "PROTECTED", "KEEP", "HIGH", ""),
    (f"{DP}/yolo-rock-v1.1/labels", "YOLO labels", True, "training, evaluation", "PROTECTED", "KEEP", "HIGH", ""),
    (f"{DP}/yolo-rock-v1.1/manifests", "split + ignore manifests", True, "everything downstream", "PROTECTED", "KEEP", "HIGH", ""),
    (f"{DP}/yolo-rock-v1.1/config_B", "YOLO dataset (masked, untrained)", True, "planned Config B arm (never trained)", "REVIEW",
     "KEEP for a future A/B experiment, or DELETE and regenerate with the Task D exporter", "MEDIUM",
     "7,353 masked training images; regenerable from curated v1.1 + source renders (deterministic export)."),
    (f"{DP}/yolo-rock-v1", "YOLO dataset (archived)", False, "ai_review.py reads its split manifest; preflight fingerprint; collect_taskd_review_images.ps1",
     "REVIEW", "DELETE after re-pointing 2 references (saves little: images are hard-linked into v1.1)", "LOW",
     "Superseded by v1.1; identical labels; most image files are the same NTFS files as v1.1."),
    (f"{RUN}/data_view", "hard-link training view", False, "E.1c run only (excluded from its CHECKSUMS)", "REVIEW", "DELETE", "LOW",
     "Hard links + Ultralytics label caches; frees only the caches."),
    (f"{TE}/benchmark-e1a-001/data_view", "hard-link training view", False, "E.1a only; preflight fingerprint", "REVIEW", "DELETE", "LOW", ""),
    (f"{TE}/smoke-test-001/data_view", "hard-link training view", False, "E.0 only; preflight fingerprint", "REVIEW", "DELETE", "LOW", ""),
    (f"{RUN}/weights/epoch[0-9]*.pt", "periodic checkpoints (0-based names: epochN = N+1 epochs)", False, "none (aligned copies exist ±1 epoch)", "REVIEW",
     "ARCHIVE locally or DELETE", "LOW", "20 files with optimiser state; byte-different from epoch_completed_* (one epoch apart)."),
    (f"{RUN}/weights/epoch_completed_*.pt", "periodic checkpoints (completed-epoch names)", False, "Task F candidates 120/130/140; resume", "REVIEW",
     "KEEP 120, 130, 140, 200; ARCHIVE/DELETE the rest", "LOW", "Full training state (EMA, optimiser, scaler)."),
    (f"{RUN}/weights/best.pt", "selected checkpoint (source)", True, "Task F frozen model source; ignore-aware results", "PROTECTED", "KEEP", "HIGH",
     "sha256 413d0481…; byte-identical to the frozen model copy."),
    (f"{RUN}/weights/last.pt", "final checkpoint", True, "E.1c report", "PROTECTED", "KEEP", "LOW", "Stripped epoch-200 weights (5.5 MB)."),
    (f"{RUN}/ignore_aware_val", "cached val predictions (22 checkpoints)", False, "Task F selection (best, 120, 130, 140); task_f_test_eval self-test", "REVIEW",
     "KEEP the 4 candidates' predictions; DELETE the other 18", "LOW", "Regenerable on GPU in ~20 s per checkpoint."),
    (f"{TE}/smoke-test-001/weights", "E.0 checkpoints", False, "tests/test_training_guard.py (rejects the 'item'-named checkpoint)", "PROTECTED",
     "KEEP best.pt (regression fixture); last.pt/epoch*.pt optional", "LOW", ""),
    (f"{TE}/benchmark-e1a-001/weights", "E.1a checkpoints", False, "none", "REVIEW", "DELETE (report keeps the evidence)", "LOW", "1-epoch benchmark weights."),
    (TF, "Task F selection + once-only test evidence", True, "frozen model, frozen config, test lock, notebooks 06-08", "PROTECTED", "KEEP", "CRITICAL", ""),
    (f"{TE}/runs_registry.jsonl", "training run registry", True, "provenance", "PROTECTED", "KEEP", "HIGH", ""),
    ("experiments/yolo-dataset/cache", "ORB near-duplicate feature cache", False, "only to rebuild the leakage-free split", "REVIEW", "DELETE (regenerable, ~minutes CPU)", "LOW",
     "425 MB orb_features.npz; the split itself is fixed in split_manifest.csv."),
    ("experiments/annotation-validation", "Task B evidence", True, "Task C cleaning reads assistant_review/ and strategy_metrics.json", "PROTECTED", "KEEP", "MEDIUM",
     "Overlays/comparison sheets are evidence for the annotation strategy decision."),
    ("experiments/dataset-cleaning", "Task C v1 evidence", True, "v1.1 build reuses its review sample (REVIEW_SAMPLE_FROM)", "PROTECTED", "KEEP", "MEDIUM", ""),
    ("experiments/dataset-cleaning-v1.1", "Task C v1.1 evidence", True, "preflight fingerprint, notebooks 03", "PROTECTED", "KEEP", "HIGH", ""),
    ("experiments/dataset-exploration", "Task A evidence", True, "notebook 01", "PROTECTED", "KEEP", "MEDIUM", ""),
    ("experiments/dataset-audit", "Task A environment audit", True, "notebook 01", "PROTECTED", "KEEP", "LOW", ""),
    ("experiments/remediation", "Task D.2 evidence", True, "notebooks 02/04", "PROTECTED", "KEEP", "MEDIUM", ""),
    ("experiments/yolo-dataset", "Task D v1 evidence (minus cache)", True, "AI-review JSON read by yolo_dataset and tests", "PROTECTED", "KEEP", "MEDIUM", ""),
    ("experiments/yolo-dataset-v1.1", "Task D v1.1 evidence", True, "tests/test_review_provenance.py, notebook 04", "PROTECTED", "KEEP", "HIGH", ""),
    ("artemis-comparisons.zip", "share package", False, "none", "SAFE", "DELETE", "NONE",
     "All 117 members CRC-identical to experiments/annotation-validation/comparison_sheets/scenes/."),
    ("taskd-review-images.zip", "share package", False, "none", "SAFE", "DELETE", "NONE",
     "Copies of 36 source renders + v1 Config B images; regenerable with collect_taskd_review_images.ps1."),
    ("weights/yolo26n.pt", "auto-downloaded model", False, "none (Ultralytics AMP self-check side effect)", "SAFE", "DELETE", "NONE", "Re-downloaded automatically if ever needed."),
    ("backend/yolo11n.pt", "pretrained weights", True, "training provenance (sha 0ebbc80d…)", "PROTECTED", "KEEP", "MEDIUM", "Official Ultralytics release asset."),
    ("backend/.venv", "Python environment", True, "everything Python", "PROTECTED", "KEEP (regenerable with uv sync)", "MEDIUM", "Files hard-linked from the uv cache."),
    ("frontend/node_modules", "Node dependencies", False, "frontend starter (pnpm install)", "SAFE", "KEEP or DELETE (regenerable)", "NONE", "Hard-linked from the pnpm store: deleting frees ~0."),
    ("frontend", "Turborepo/Next.js starter (unmodified)", False, "future Mission Control", "REVIEW", "KEEP, exclude from the ML submission", "LOW", "No project code yet."),
    ("backend/training/notebooks", "Task E prep notebooks (never executed)", False, "none", "REVIEW", "ARCHIVE (superseded by notebooks/05-07)", "LOW", ""),
    ("audit-artemis.ps1", "Task A environment auditor", True, "produced experiments/dataset-audit", "REVIEW", "MOVE to scripts/legacy", "LOW", ""),
    ("collect_taskd_review_images.ps1", "share-package script", False, "produced taskd-review-images.zip", "REVIEW", "MOVE to scripts/legacy", "LOW", ""),
]

SCENARIOS = {
    "S1_safe": ["dataset/polar-sim", "artemis-comparisons.zip", "taskd-review-images.zip", "weights/yolo26n.pt"],
    "S2_git_garbage": [".git/objects"],
    "S3_duplicate_archives": ["dataset/artificial-lunar/archive.zip", "dataset/nasa-polar/Terrain01_LargeOnly.zip",
                              "dataset/nasa-polar/Terrain06_Smooth.zip", "dataset/nasa-polar/Terrain11_FreshCrater.zip"],
    "S4_duplicate_polar_clone": ["polar-sim-source"],
    "S5_obsolete_v1_and_views": [f"{DP}/curated-dataset-v1", f"{DP}/yolo-rock-v1", f"{RUN}/data_view", f"{TE}/benchmark-e1a-001/data_view",
                                 f"{TE}/smoke-test-001/data_view"],
    "S6_checkpoints_and_caches": [f"{RUN}/weights/epoch[0-9]*.pt", "experiments/yolo-dataset/cache", f"{TE}/benchmark-e1a-001/weights"],
    "S7_config_B_too (only if Config B is abandoned)": [f"{DP}/yolo-rock-v1.1/config_B", f"{DP}/yolo-rock-v1"],
}


def _files(pattern: str) -> list[Path]:
    paths = list(ROOT.glob(pattern)) if any(c in pattern for c in "*?[") else [ROOT / pattern]
    out = []
    for p in paths:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for dp, dn, fn in os.walk(p, followlinks=False):
                dn[:] = [d for d in dn if not (Path(dp, d).is_symlink() or os.path.isjunction(Path(dp, d)))]
                out += [Path(dp, f) for f in fn]
    return out


def measure(patterns: list[str], links_in_ws: dict) -> dict:
    seen, app, phys, newest = {}, 0, 0, 0.0
    for pat in patterns:
        for f in _files(pat):
            st = f.stat(follow_symlinks=False)
            k = (st.st_dev, st.st_ino)
            app += st.st_size; newest = max(newest, st.st_mtime)
            seen.setdefault(k, [st.st_size, st.st_nlink, 0])[2] += 1
    phys = sum(v[0] for v in seen.values())
    recl = sum(v[0] for v in seen.values() if v[2] >= v[1])        # every hard link of the file is inside the set
    return {"files": sum(v[2] for v in seen.values()), "apparent_bytes": app, "physical_bytes": phys, "reclaimable_bytes": recl,
            "last_modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else None}


if __name__ == "__main__":
    t0 = time.time()
    items = []
    for path, kind, canon, cons, cls, action, risk, why in ITEMS:
        m = measure([path], {})
        items.append({"path": path, "type": kind, "canonical": canon, "consumers": cons, "classification": cls,
                      "recommended_action": action, "removal_risk": risk, "notes": why, **m,
                      "expected_reclaim_if_action_applied_bytes": m["reclaimable_bytes"] if ("DELETE" in action.upper() and cls != "PROTECTED") else 0})
    scen = {k: {"paths": v, **measure(v, {})} for k, v in SCENARIOS.items()}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "generator": "scripts/retention_manifest.py",
           "method": "Hard-link aware: reclaimable = bytes of files whose every NTFS hard link lies inside the measured set. "
                     "Physical sizes are file sizes, not cluster-rounded allocation.",
           "status": "DRY RUN - nothing deleted", "items": items, "deletion_scenarios": scen, "seconds": round(time.time() - t0, 1)}
    p = ROOT / "docs/audits/RETENTION_MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    G = lambda b: f"{b / 2**30:8.3f}"  # noqa: E731
    for i in items:
        print(f"{i['classification']:9} {G(i['physical_bytes'])} {G(i['reclaimable_bytes'])}  {i['path']}")
    for k, s in scen.items():
        print(f"SCENARIO {k}: physical {G(s['physical_bytes'])} reclaimable {G(s['reclaimable_bytes'])}")
