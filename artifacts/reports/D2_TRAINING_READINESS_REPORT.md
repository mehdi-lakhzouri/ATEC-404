# Training Readiness — Task E (YOLO11n), prepared in D.2

**No training, fine-tuning or GPU benchmark was run.**
The only GPU interaction was a metadata query: device properties, no kernels launched.

## 1. What is prepared (`backend/training/`)

| File | Purpose | State |
|---|---|---|
| `configs/task_e_experiments.yaml` | Full plan: dataset version, identical hyperparameters, seeds, budgets, arms, model selection, metrics | Ready |
| `env_check.py` | CUDA / AMP check. Default: metadata only. `--compute`: tiny fp16/bf16 autocast + GradScaler step (for the start of an authorised session) | Metadata run: **PASS** (`experiments/task-e/env_check_metadata.json`) |
| `monitor.py` | Background GPU / CPU / RAM sampler (pynvml + psutil) → `resources.csv` | Ready |
| `train.py` | Guarded launcher: dry run, run registry (JSONL), dataset / manifest / config / weights hashes, automatic resume from `last.pt`, monitor | **Refuses** unless `ARTEMIS_TRAINING_AUTHORIZED=1` **and** `--authorized` |
| `evaluate.py` | Predict on unmasked val/test → ignore-aware metrics | Ready (needs a model) |
| `notebooks/task_e_training.ipynb` | Environment, dry-run plan, guarded training loop, resource plots | Valid nbformat 4; code cells parse |
| `notebooks/task_e_evaluation_analysis.ipynb` | Evaluator sanity check, checkpoint evaluation, A vs B over seeds | Sanity cells executed: AP50 = 1.0, FP 0 on real val labels |

**Environment** (metadata only): RTX 5060 Laptop, **sm_120 present in the torch 2.14.1+cu130 build**, 7.93 GB VRAM, bf16 supported, cuDNN 9.24, Ultralytics 8.4.174.

## 2. Key choices

**Shared settings**
- `imgsz 736`: no downscaling of the 720×480 renders; the median rock box side is 12 px.
- `batch 16`, fixed (never `auto`); out-of-memory fallback 8. Expected well under 8 GB for YOLO11n with AMP. *Not measured, by design.*
- `workers 4`, `cache false`.

**Reproducibility**
- Seeds 0, 1, 2 per arm.
- `deterministic: true`. Bitwise GPU reproducibility is still not guaranteed (cuDNN), so runs are compared as mean ± spread.

**Fair A vs B**
- Identical hyperparameters (tested).
- Primary budget is **equal optimiser steps**: A 193 epochs ≈ B 50 epochs ≈ 23,000 iterations, verified within 1%. Equal epochs is a secondary arm.
- `patience 0`, so both get the full budget.

**Augmentation**
- No hue or saturation jitter (grayscale renders), `flipud 0` (sky stays up), mosaic with `close_mosaic 10`.

**Checkpoints and selection**
- Checkpoints every 5 epochs, with resume from `last.pt`.
- Selection with the **ignore-aware evaluator on val**, not Ultralytics' `best.pt`.

**Experiment tracking**
- `experiments/task-e/runs_registry.jsonl`, with dataset status and hashes, environment and timings, plus Ultralytics `results.csv`.
- MLflow and TensorBoard are not installed, deliberately.

## 3. Automated test results (`experiments/remediation/test_results.txt`)

**38 tests, all pass.**

| Suite | Tests |
|---|---|
| Task C v1.1 regression | 11 |
| Ignore-aware evaluator (incl. Ultralytics agreement and real val split) | 16 |
| Review provenance | 7 |
| Training guard | 4 |

```
cd backend
uv run --no-sync python -m unittest discover -s tests -t . -v
```

## 4. Dataset versions and storage

| Version | Status | Content hashes |
|---|---|---|
| curated-dataset-v1 | ARCHIVED, unchanged | `annotations_all` `764eff12…` |
| curated-dataset-v1.1 | CURRENT, provisional | `annotations_all` `87213054…`, `rock_valid` `b61e1e60…` |
| yolo-rock-v1 | ARCHIVED, unchanged | labels tree `e2a4b406…` |
| yolo-rock-v1.1 | CURRENT, **not frozen, not approved** | labels tree `e2a4b406…` (identical labels), `split_manifest` `3274dcd2…`, `data_A` `fc5f0fdc…`, `data_B` `cda93d8d…` |

- The full registry is in `dataset/processed/VERSIONS.json`.
- Splits, leakage groups and config A/B membership are identical between v1 and v1.1 (all 9,193 frames).
- **Storage added in D.2: ≈ 355 MB.**

| Item | Size | Note |
|---|---|---|
| curated v1.1 | 242 MB | |
| yolo-rock-v1.1 | 46 MB | Real, new data only: 11,088 of 11,096 images are hard links to v1 (8 new config B PNGs). Both export trees are read-only derived data, and a hard-linked file shares its bytes across them. |
| experiments | ≈ 65 MB | |
| ORB cache | 0 | Not regenerated |

## 5. Proposed vs applied

| Item | Applied | Proposed only (needs approval) |
|---|---|---|
| D05 for split pieces (Task C v1.1) | ✅ new version, v1 archived | — |
| YOLO rebuild from v1.1, same splits | ✅ | — |
| Gallery provenance fix and AI-review import | ✅ | — |
| Ignore-aware evaluator | ✅ tested | Using the sub-threshold ignore file in the official protocol |
| 50 px floor / class threshold 127 | unchanged | Keep; in-scope definition in the remediation report |
| 27 REVIEW→VALID promotion candidates, 5 manual instance-box scenes | — | Human adjudication |
| Visibility-rule recalibration | — | Needs ≥ 300 human labels across lighting conditions |
| Config B cap 15% / noise-matched fill | — | Ablation only; fill is a prototype |
| Ignore-aware loss (arm C) | — | Design only |
| Human-verified test benchmark (150 images) | — | Protocol in the evaluation report |

## 6. Unresolved risks

1. **Labels are automated and unverified**, including val/test. The visibility rule withholds many visible rocks in bright, soft-lit scenes, so recall-type metrics are only meaningful with ignore regions.
2. **Config B trains on smeared images** (texture ratio 0.11) and erases real rocks.
3. **Out-of-scope rocks** in 65 of 153 "negative" frames (dim masks). Do not use those frames as negatives.
4. **The synthetic → NASA lab domain gap is unaddressed.** The NASA evaluation is blocked on the 12→8-bit conversion decision.
5. **`yolo11n.pt` is not on disk.** It needs a one-time download (network) and its sha256 pinned in the registry.
6. Bitwise reproducibility on GPU is not guaranteed. The repository has no git commits yet, so runs cannot record a commit ID until the code is committed.

## 7. Decision for Task E — **CONDITIONAL GO (not yet authorised to start)**

The infrastructure, evaluation and data versioning are ready and tested. Training may start only after all of these:
1. A human reviews the v1.1 gallery (at least 20 scenes, decisions exported with `provenance: human_ui`) and answers the four policy questions.
2. Approval of the evaluation protocol: ignore-aware primary metric; whether the sub-threshold ignore regions are used.
3. A decision on arms: A, B with a 15% cap, and whether to build arm C (ignore-aware loss) first. **Recommended:** A + C, with B as an ablation.
4. Approval to freeze `yolo-rock-v1.1`, or a decision to apply P1/P2 first and re-version.
5. Approval of the `yolo11n.pt` download, then authorisation through `ARTEMIS_TRAINING_AUTHORIZED=1` and `--authorized`.
