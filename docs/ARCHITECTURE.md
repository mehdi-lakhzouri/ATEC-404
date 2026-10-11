# Architecture

Current R6/R6.1 runtime: Webots RGB + depth / odometry → frozen local inference → `src/artemis_x/sim/mapping`
→ binary UDP / lunar emulator / selective ACK → `ground_station` FastAPI + MongoDB → REST/WebSocket →
`frontend/apps/web` Mission Control. See [jury quickstart](JURY_QUICKSTART.md) for launch/ports and
[ground contract](TELEMETRY_R5_5_GROUND_BACKEND.md). The diagrams below describe the **historical research pipeline**;
their original “planned” integration labels are not the current implementation status.

## Component boundaries

| Component | Location | Responsibility | Published |
|---|---|---|---|
| Public package | `src/artemis_x/` | Frozen model registry and checksum, inference with frozen settings, ignore-aware evaluation wrapper, metric helpers, artifact loaders, plots | yes |
| Research pipeline | `backend/analysis/`, `backend/training/` | Tasks A–F: exploration, annotation strategy, cleaning, YOLO export, remediation, guarded training, selection, locked test evaluation | yes (code only) |
| Notebooks | `notebooks/` | Scientific narrative over published artifacts | yes |
| Evidence | `artifacts/`, `data/manifests/`, `models/` | Hash-verified copies of results, figures, reports, split metadata, frozen model | yes |
| Local data | `dataset/`, `experiments/` | Source and processed datasets, full run outputs | **no** (size, licence) |
| Frontend | `frontend/` | Unmodified Turborepo starter (future Mission Control) | not part of the ML submission |

`src/artemis_x` **wraps** the research pipeline instead of moving it. `artemis_x.evaluation.ignore_aware` re-exports
`backend/analysis/evaluation/ignore_eval.py` (the identity is unit-tested). Paths are resolved centrally in
`artemis_x.config.paths`, with `ARTEMIS_ROOT` and `ARTEMIS_DATA_ROOT` overrides, so no module depends on a machine-specific
location. The 52 historical regression tests and every documented `backend` command keep working unchanged.

## Data flow

```mermaid
flowchart TB
  subgraph local[Local only]
    SRC[dataset/artificial-lunar<br/>dataset/nasa-polar<br/>dataset/polar-sim-source]
    CUR[dataset/processed/curated-dataset-v1.1]
    YOLO[dataset/processed/yolo-rock-v1.1]
    RUN[experiments/task-e/full-training-config-a-200-s42]
    SEL[experiments/task-f/model-selection-001<br/>FROZEN_MODEL_CONFIG.json]
    TST[experiments/task-f/test-evaluation-001<br/>TEST_RUN_LOCK.json]
  end
  subgraph pub[Published]
    ART[artifacts/ + data/manifests/]
    MOD[models/artemis_rock_yolo11n.pt]
    PKG[src/artemis_x]
    NB[notebooks/01-08]
  end
  SRC -->|Task C| CUR -->|Task D| YOLO -->|Task E| RUN -->|Task F| SEL -->|once| TST
  RUN & SEL & TST -->|scripts/export_artifacts.py<br/>copy or sanitise + sha256| ART
  SEL --> MOD
  ART --> PKG --> NB
  MOD --> PKG
  PKG -.planned.-> WEB[Webots rover controller]
```

## Inference path

```mermaid
sequenceDiagram
  participant C as Caller (script, notebook, future Webots controller)
  participant R as artemis_x.ml.model_registry
  participant D as artemis_x.ml.inference.RockDetector
  participant U as Ultralytics YOLO11n
  C->>R: load_frozen_spec()
  R->>R: read FROZEN_MODEL_CONFIG.json, check sha256 of weights
  R-->>C: FrozenModelSpec (imgsz 736, conf 0.20, IoU 0.60, max_det 300, min area 50 px²)
  C->>D: RockDetector(spec, device=auto)
  C->>D: predict(frame)
  D->>U: predict(imgsz, conf, iou, max_det)
  U-->>D: boxes + scores
  D->>D: drop boxes < 50 px²
  D-->>C: DetectionResult (detections, latency, device)
```

## Safety mechanisms preserved from the research pipeline

| Mechanism | What it does |
|---|---|
| Training authorisation | `ARTEMIS_TRAINING_AUTHORIZED=1` plus `--authorized`, `--fresh`, and limit callbacks (epochs, batch, OOM retry, NaN replay) |
| `TEST_RUN_LOCK.json` and hash verification | `task_f_test_eval.py` refuses any second test run |
| `scripts/verify_artifacts.py` | Frozen model, both Task F checksum sets, protected-file baseline, every published artifact, run registry |
| `scripts/demo_inference.py` | Refuses images from the held-out test split |
