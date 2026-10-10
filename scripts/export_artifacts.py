"""Publish lightweight, pixel-free scientific artifacts from experiments/ into artifacts/, data/manifests/ and models/.

Every published file is either a byte-identical COPY (sha256 recorded and checked) or a SANITISED derivative
(absolute local paths replaced by "<repo>"); the source sha256 is recorded for both. Nothing in experiments/ or
dataset/ is modified. Dataset images and anything rendered from dataset pixels are deliberately excluded
(redistribution licence not confirmed).

Usage (repo root):  uv run --project backend --no-sync python scripts/export_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = "experiments"
RUN = f"{EX}/task-e/full-training-config-a-200-s42"
SEL = f"{EX}/task-f/model-selection-001"
TST = f"{EX}/task-f/test-evaluation-001"
YDS = "dataset/processed/yolo-rock-v1.1"

COPIES = {   # destination (relative to ROOT) -> source (relative to ROOT)
    # provenance
    "artifacts/provenance/VERSIONS.json": "dataset/processed/VERSIONS.json",
    "artifacts/provenance/FROZEN_MODEL_CONFIG.json": f"{SEL}/FROZEN_MODEL_CONFIG.json",
    "artifacts/provenance/TEST_RUN_LOCK.json": f"{TST}/TEST_RUN_LOCK.json",
    "artifacts/provenance/model_selection_CHECKSUMS.sha256": f"{SEL}/CHECKSUMS.sha256",
    "artifacts/provenance/test_evaluation_CHECKSUMS.sha256": f"{TST}/CHECKSUMS.sha256",
    "artifacts/provenance/task_e_experiments.yaml": "backend/training/configs/task_e_experiments.yaml",
    # data (metadata only - no pixels, no box coordinates)
    "data/manifests/split_manifest.csv": f"{YDS}/manifests/split_manifest.csv",
    "artifacts/data/artificial_lunar_analysis.json": f"{EX}/dataset-exploration/artificial_lunar_analysis.json",
    "artifacts/data/data_quality_report.json": f"{EX}/dataset-exploration/data_quality_report.json",
    "artifacts/data/dataset_inventory.json": f"{EX}/dataset-exploration/dataset_inventory.json",
    "artifacts/data/strategy_decision.json": f"{EX}/annotation-validation/strategy_decision.json",
    "artifacts/data/strategy_metrics.json": f"{EX}/annotation-validation/strategy_metrics.json",
    "artifacts/data/annotation_quality_findings.json": f"{EX}/annotation-validation/quality_findings.json",
    "artifacts/data/cleaning_quality_metrics_v1.1.json": f"{EX}/dataset-cleaning-v1.1/quality_metrics.json",
    "artifacts/data/split_statistics_v1.1.json": f"{EX}/yolo-dataset-v1.1/split_statistics.json",
    "artifacts/data/annotation_distributions_v1.1.json": f"{EX}/yolo-dataset-v1.1/annotation_distributions.json",
    "artifacts/data/dataset_status_v1.1.json": f"{EX}/yolo-dataset-v1.1/dataset_status.json",
    "artifacts/data/task_c_validation_v1.1.json": f"{EX}/yolo-dataset-v1.1/task_c_validation.json",
    "artifacts/data/masking_audit_summary.json": f"{EX}/remediation/masking_audit/masking_audit_summary.json",
    "artifacts/data/threshold_study.json": f"{EX}/remediation/threshold_study/threshold_study.json",
    "artifacts/data/scene_investigation_summary.json": f"{EX}/remediation/scene_investigation/summary.json",
    # training
    "artifacts/training/results.csv": f"{RUN}/results.csv",
    "artifacts/training/per_epoch_performance.csv": f"{RUN}/analysis/per_epoch_performance.csv",
    "artifacts/training/e1c_summary.json": f"{RUN}/analysis/summary.json",
    "artifacts/training/checkpoint_inventory.json": f"{RUN}/checkpoint_inventory.json",
    # evaluation
    "artifacts/evaluation/ignore_aware_val_22_checkpoints.json": f"{RUN}/ignore_aware_val.json",
    "artifacts/evaluation/selection_analysis.json": f"{SEL}/selection_analysis.json",
    "artifacts/evaluation/inference_check.json": f"{SEL}/inference_check.json",
    "artifacts/evaluation/test_results.json": f"{TST}/test_results.json",
    "artifacts/evaluation/large_rock_edge_analysis.json": f"{TST}/large_rock_edge_analysis.json",
    # figures (charts only; no dataset pixels)
    "artifacts/figures/a_bbox_distributions.png": f"{EX}/dataset-exploration/figures/al_bbox_distributions.png",
    "artifacts/figures/a_objects_per_frame.png": f"{EX}/dataset-exploration/figures/al_objects_per_frame.png",
    "artifacts/figures/a_render_intensity_and_sky.png": f"{EX}/dataset-exploration/figures/al_render_intensity_and_sky.png",
    "artifacts/figures/a_anomaly_list_metrics.png": f"{EX}/dataset-exploration/figures/al_anomaly_list_metrics.png",
    "artifacts/figures/b_s1_vs_s2_counts.png": f"{EX}/annotation-validation/figures/s1_vs_s2_counts.png",
    "artifacts/figures/d_split_strata.png": f"{EX}/yolo-dataset-v1.1/figures/split_strata.png",
    "artifacts/figures/d_split_box_distributions.png": f"{EX}/yolo-dataset-v1.1/figures/split_box_distributions.png",
    "artifacts/figures/d2_threshold_curves.png": f"{EX}/remediation/threshold_study/threshold_curves.png",
    "artifacts/figures/e_training_curves.png": f"{RUN}/analysis/training_curves.png",
    "artifacts/figures/e_hardware_telemetry.png": f"{RUN}/analysis/hardware_telemetry.png",
    "artifacts/figures/e_epoch_performance.png": f"{RUN}/analysis/epoch_performance.png",
    "artifacts/figures/f_threshold_analysis.png": f"{SEL}/threshold_analysis.png",
    "artifacts/figures/f_test_evaluation.png": f"{TST}/test_evaluation_figures.png",
    "artifacts/figures/f_test_confusion_analysis.png": f"{TST}/test_confusion_analysis.png",
    # reports (Markdown, no absolute paths)
    "artifacts/reports/A_DATASET_EXPLORATION_REPORT.md": f"{EX}/dataset-exploration/DATASET_EXPLORATION_REPORT.md",
    "artifacts/reports/B_STRATEGY_COMPARISON.md": f"{EX}/annotation-validation/STRATEGY_COMPARISON.md",
    "artifacts/reports/B_ANNOTATION_VALIDATION_REPORT.md": f"{EX}/annotation-validation/ANNOTATION_VALIDATION_REPORT.md",
    "artifacts/reports/C_ANNOTATION_POLICY.md": f"{EX}/dataset-cleaning/ANNOTATION_POLICY.md",
    "artifacts/reports/C_CLEANING_REPORT.md": f"{EX}/dataset-cleaning/CLEANING_REPORT.md",
    "artifacts/reports/C_DATASET_QUALITY_REPORT.md": f"{EX}/dataset-cleaning/DATASET_QUALITY_REPORT.md",
    "artifacts/reports/D_DATASET_ENGINEERING_REPORT.md": f"{EX}/yolo-dataset/DATASET_ENGINEERING_REPORT.md",
    "artifacts/reports/D_SPLIT_VALIDATION_REPORT.md": f"{EX}/yolo-dataset/SPLIT_VALIDATION_REPORT.md",
    "artifacts/reports/D_REVIEW_HANDLING_DECISION.md": f"{EX}/yolo-dataset/REVIEW_HANDLING_DECISION.md",
    "artifacts/reports/D2_ANNOTATION_QUALITY_REMEDIATION_REPORT.md": f"{EX}/remediation/ANNOTATION_QUALITY_REMEDIATION_REPORT.md",
    "artifacts/reports/D2_EVALUATION_PROTOCOL_REPORT.md": f"{EX}/remediation/EVALUATION_PROTOCOL_REPORT.md",
    "artifacts/reports/D2_TASK_C_V1_1_FIX_REPORT.md": f"{EX}/remediation/TASK_C_V1_1_FIX_REPORT.md",
    "artifacts/reports/D2_MASKING_ABLATION_RECOMMENDATION.md": f"{EX}/remediation/MASKING_ABLATION_RECOMMENDATION.md",
    "artifacts/reports/D2_TRAINING_READINESS_REPORT.md": f"{EX}/remediation/TRAINING_READINESS_REPORT.md",
    "artifacts/reports/E0_SMOKE_TEST_REPORT.md": f"{EX}/task-e/smoke-test-001/SMOKE_TEST_REPORT.md",
    "artifacts/reports/E1A_BENCHMARK_REPORT.md": f"{EX}/task-e/benchmark-e1a-001/E1A_BENCHMARK_REPORT.md",
    "artifacts/reports/E1C_TRAINING_REPORT.md": f"{RUN}/E1C_TRAINING_REPORT.md",
    "artifacts/reports/F_MODEL_SELECTION_REPORT.md": f"{SEL}/TASK_F_MODEL_SELECTION_REPORT.md",
    "artifacts/reports/F_TEST_EVALUATION_REPORT.md": f"{TST}/TASK_F_TEST_EVALUATION_REPORT.md",
    # model
    "models/artemis_rock_yolo11n.pt": f"{SEL}/model/artemis_rock_yolo11n_e1c_s42.pt",
}
SANITISED = {   # destination -> source; absolute workspace paths replaced by "<repo>"
    "data/manifests/data_A.yaml": f"{YDS}/data_A.yaml",
    "data/manifests/data_B.yaml": f"{YDS}/data_B.yaml",
    "artifacts/training/e1c_args.yaml": f"{RUN}/args.yaml",
    "artifacts/training/e1c_train_config.json": f"{RUN}/train_config.json",
    "artifacts/provenance/runs_registry.jsonl": f"{EX}/task-e/runs_registry.jsonl",
    "artifacts/provenance/e1c_preflight_pre.json": f"{RUN}/preflight_pre.json",
    "artifacts/provenance/e1c_preflight_post.json": f"{RUN}/preflight_post.json",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitise(text: str) -> str:
    variants = {str(ROOT), str(ROOT).replace("\\", "/"), str(ROOT).replace("\\", "\\\\"), ROOT.as_posix()}
    for v in sorted(variants, key=len, reverse=True):
        text = text.replace(v, "<repo>")
    # any remaining Windows user-profile path (e.g. uv cache, temp dirs)
    return re.sub(r"[A-Za-z]:(\\\\|\\|/)Users(\\\\|\\|/)[^\\/\"'\s]+", "<user-home>", text)


def main() -> dict:
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "generator": "scripts/export_artifacts.py",
                "policy": "No dataset pixels or box coordinates are published (redistribution licence not confirmed). "
                          "COPY = byte-identical; SANITISED = absolute local paths replaced by <repo>/<user-home>.",
                "files": []}
    missing = []
    for dst, src in {**COPIES, **SANITISED}.items():
        s, d = ROOT / src, ROOT / dst
        if not s.exists():
            missing.append(src); continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if dst in COPIES:
            shutil.copyfile(s, d)
            mode = "COPY"
            assert _sha(s) == _sha(d), dst
        else:
            d.write_text(_sanitise(s.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
            mode = "SANITISED"
        manifest["files"].append({"path": dst, "mode": mode, "source": src, "source_sha256": _sha(s), "sha256": _sha(d), "bytes": d.stat().st_size})
    manifest["missing_sources"] = missing
    out = ROOT / "artifacts/provenance/ARTIFACT_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    m = main()
    tot = sum(f["bytes"] for f in m["files"])
    print(f"published {len(m['files'])} files, {tot / 2**20:.1f} MB; missing sources: {m['missing_sources']}")
