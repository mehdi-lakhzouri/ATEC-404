# Smoke Test 001: YOLO11n, Config A, 3 epochs (Task E.0)

**Verdict: PASS**, with 3 issues to fix before E.1 (§11).
**Dataset:** `yolo-rock-v1.1`, still **PROVISIONAL and NOT FROZEN**.
**Validation metrics in this report are PROVISIONAL:** they come from a 3-epoch model scored against automated, non-human-verified labels. The model is **not** production-ready, and these numbers do not measure final accuracy.

**Run history:**
- **First attempt aborted** after epoch 1: a bug in my epoch-logging callback (`tloss` is a dict in Ultralytics 8.4). No checkpoint was written. The wrapper also mis-recorded the status as COMPLETED. Both bugs are fixed, with 4 regression tests.
- **You approved one re-run** with identical settings. That re-run is the run reported here. The aborted attempt's logs are in `aborted_attempt_1/`.

## 1. CUDA and hardware

**Environment check** (`env_check.json`, metadata + AMP compute check): **PASS**.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, sm_120 (in the torch build), 7.93 GB, driver 616.64 |
| Software | torch 2.14.1+cu130, cuDNN 9.24, Ultralytics 8.4.174, Python 3.12.13, Windows 11 |
| AMP | fp16 and bf16 autocast plus a GradScaler step all finite; Ultralytics' own AMP check passed |
| Disk free | 372.7 GB |

**Pre-flight** (`preflight_pre.json`): 12 of 12 checks pass.
- `data_A.yaml`, `split_manifest.csv` and the labels tree match `dataset/processed/VERSIONS.json`.
- Class map is `{0: rock}`, `nc: 1`.
- Train has 1,903 images and val 923, every image with a label file.
- The dataset is not frozen, and the run folder had no earlier checkpoint.

**Post-flight** (`preflight_post.json`): unchanged after training:
- source datasets (Artificial Lunar, NASA POLAR, POLAR-Sim);
- curated v1.1 and its experiment folder;
- yolo-rock-v1.1 and its experiment folder;
- the archived v1 versions.

**Isolation:** training read a **hard-link view** of the train/val split in `data_view/`, so Ultralytics' `labels/*.cache` files were written there, not into the dataset. No image bytes were copied.

**Weights**

| File | Source | Size | SHA-256 | Used for |
|---|---|---|---|---|
| `backend/yolo11n.pt` | `github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt` | 5,613,764 B | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | Training start point |
| `weights/yolo26n.pt` (project root) | Downloaded by Ultralytics' AMP self-check | 5,544,453 B | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | AMP check only, not training |

## 2. Command and hyperparameters

```
cd backend
$env:ARTEMIS_TRAINING_AUTHORIZED=1   # set for this single command only
uv run --no-sync python -m training.train --arm A_steps --seed 42 --authorized --fresh --data-view `
  --run-dir ../experiments/task-e/smoke-test-001 `
  --set epochs=3 imgsz=640 batch=4 workers=2 amp=true cache=false deterministic=true device=0 plots=true
```

**Effective arguments** (full list in `args.yaml` and `train_config.json`):
- Model and data: yolo11n.pt (pretrained); `data_view/data_view.yaml` (Config A train, unmasked val).
- Run settings: epochs 3, imgsz 640, batch 4, workers 2, AMP on, cache off, seed 42, deterministic on, device 0.
- Optimiser and schedule: SGD, lr0 0.01, momentum 0.937, weight decay 5e-4, cosine LR, warm-up 3 epochs.
- Augmentation: hsv_h/s 0, hsv_v 0.3, fliplr 0.5, flipud 0, mosaic 1.0 (close_mosaic 10).
- Other: single_cls true, patience 0.

**Smoke-test caveats:**
- With 3 epochs the whole run sits inside the 3-epoch learning-rate warm-up.
- `close_mosaic 10` never triggers.

## 3. Exactly 3 epochs ran

- `results.csv` has exactly 3 rows (epochs 1–3).
- The log says "3 epochs completed in 0.030 hours".
- The wrapper's status is `COMPLETED` and `resumed: false`.
- No further run was started.

## 4. Durations

| Epoch | Train (s) | Train + val (s) | Train throughput |
|---|---|---|---|
| 1 | 29.0 | 40.3 | 66 img/s |
| 2 | 24.1 | 30.1 | 79 img/s |
| 3 | 30.6 | 45.1 | 62 img/s |

- **Ultralytics total:** 107.3 s for 3 epochs.
- **Wrapper wall time:** 121 s (20:30:43 → 20:32:44), including environment check, data-view build and AMP check.
- **Per epoch:** 476 iterations at batch 4, about 16–20 it/s. Validation took 6–15 s for 923 images.

## 5. GPU and system telemetry

61 `nvidia-smi` samples at 2 s intervals (`nvidia_smi.csv`), plus `resources.csv` (pynvml/psutil).

| Metric | Value |
|---|---|
| Temperature | max **57 °C**, mean 53.4 °C |
| VRAM | `nvidia-smi` max **1,151 MiB** of 8,151 (torch peak allocated 660 MB, reserved 762 MB) |
| Utilisation | mean 33%, max 47%: the run is **input-bound**, not GPU-bound |
| Power | max 47.3 W, mean 36.1 W |
| SM clock | up to 2,850 MHz |
| Thermal slowdown (HW or SW) | 0 of 61 samples |
| Power cap | 0 of 61 samples |
| System RAM | 14.3–19.0 GB used (whole machine) |
| OOM / NaN / Inf | none (per-epoch finite-loss check passed) |

The raw `clocks_event_reasons.active` bitmask read `0x400` in every sample, including the idle first one. It's static, not caused by training, and none of the thermal or power flags were active. My initial summary parsed the old `clocks_throttle_reasons` column names and missed these fields; I fixed it and re-summarised from the logged CSV.

## 6. Loss evolution (`results.csv`, `results.png`)

| Epoch | train box | train cls | train dfl | val box | val cls | val dfl |
|---|---|---|---|---|---|---|
| 1 | 2.348 | 2.169 | 1.162 | 2.069 | 1.611 | 1.098 |
| 2 | 2.220 | 1.651 | 1.070 | 2.115 | 1.513 | 1.068 |
| 3 | 2.024 | 1.464 | 1.034 | 1.859 | 1.387 | 1.028 |

All training losses decreased every epoch; validation box loss rose briefly at epoch 2.

**Provisional validation metrics**

*Ultralytics val (no ignore support, so REVIEW-region detections count as false positives):*

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 0.409 | 0.410 | 0.339 | 0.135 |
| 2 | 0.442 | 0.427 | 0.369 | 0.152 |
| 3 | 0.518 | 0.502 | 0.454 | 0.210 |

*Ignore-aware evaluator on val with `best.pt`* (min prediction area 50 px², conf 0.25):

| Ignore regions | AP50 | AP50-95 | P@0.25 | R@0.25 | AP50 small / medium / large |
|---|---|---|---|---|---|
| none (naive) | 0.465 | 0.215 | 0.660 | 0.345 | 0.401 / 0.597 / 0.606 |
| **REVIEW (policy)** | **0.593** | 0.271 | 0.855 | 0.345 | 0.503 / 0.749 / 0.767 |
| REVIEW + sub-threshold (proposed) | 0.606 | 0.276 | 0.865 | 0.345 | 0.528 / 0.751 / 0.767 |

The ignore regions turn about 3,100 "false positives" on uncertain REVIEW regions into neither TP nor FP. This confirms the evaluation protocol matters even at this stage.

## 7. Checkpoint validation

| | `best.pt` | `last.pt` |
|---|---|---|
| Size | 5,447,514 B | 5,447,514 B |
| SHA-256 | `647a4f8b…71e8` | `caa0ce03…3086` |
| Loads with `YOLO()` | yes | yes |
| Parameters | 2,590,035, all finite | 2,590,035, all finite |
| Class names | **`{0: 'item'}`** (issue 1 below) | same |

- `epoch0.pt` was also saved, because `save_period 5` triggers at epoch index 0. Harmless.
- `best.pt` corresponds to epoch 3, which had the highest fitness.
- The optimiser state was stripped by Ultralytics as usual (`epoch = -1` in the stripped files).

## 8. Example inference (`inference/`, conf ≥ 0.25, imgsz 640; val/test images only, never trained on)

| Frame | Split | Scene | Pred | VALID GT | REVIEW | TP | FP | Ignored |
|---|---|---|---|---|---|---|---|---|
| 0189 | test | sparse | 1 | 1 | 0 | 1 | 0 | 0 |
| 0115 | val | sparse, many REVIEW | 2 | 4 | 15 | 1 | 0 | 1 |
| 9433 | test | dense | 33 | 111 | 29 | 18 | 12 | 3 |
| 5589 | val | dense | 37 | 110 | 34 | 22 | 6 | 9 |
| 0575 | val | dark / shadow | 9 | 41 | 44 | 4 | 2 | 3 |
| 0586 | test | dark / shadow | 11 | 32 | 9 | 8 | 1 | 2 |
| 4357 | test | 155 REVIEW boxes | 30 | 31 | 155 | 9 | 6 | 15 |
| 6352 | val | shadowed boulders | 14 | 60 | 26 | 9 | 2 | 3 |
| 5311 | test | no records (dim-mask rocks) | 1 | 0 | 0 | 0 | 1 | 0 |

Qualitative observations only, not accuracy claims:
- Prominent rocks are found with tight boxes.
- Most small rocks are still missed (recall 0.34).
- In 6352, one box covers both touching boulders.
- The single "FP" in 5311 sits on the real but out-of-scope distant rock cluster identified in D.2.

## 9. Warnings and issues

**Fixed during this task:**

| # | Issue | Fix |
|---|---|---|
| 1 | Attempt 1 aborted: wrapper callback crashed on `tloss` (dict) | `loss_record()` handles dict or tensor; 4 regression tests |
| 2 | Wrapper reported COMPLETED after the crash | Any exception is now recorded as `FAILED: …` and re-raised |
| 3 | `nvidia-smi` summary missed renamed columns | Suffix-based column lookup; fails loudly if a column is missing |

**Open (fix in E.1):**
- **Class named `item`:** `single_cls: true` makes Ultralytics rename the class. The dataset is already single-class, so set `single_cls: false` to keep `rock` for telemetry and integration.
- **Low GPU utilisation (33%):** with batch 4 and 2 workers the run is input-bound. The planned batch 16 and 4 workers should improve it, but that needs measuring.
- **Extra download:** Ultralytics' AMP check downloads `yolo26n.pt` into `weights/` at the project root. Keep it, so later runs don't need the network.
- **`epoch0.pt`:** an extra checkpoint from `save_period`. Ignore it, or use `save_period` with an offset.
- **No Ultralytics warnings were logged.** `deterministic=True` was accepted. Bitwise GPU reproducibility is still not guaranteed.

## 10. Duration estimates for full Task E (from measured throughput)

**Directly extrapolated at the smoke settings** (640, batch 4, workers 2): about 68 img/s training, about 10 s validation per epoch.

| Arm | Images seen | Estimate per seed |
|---|---|---|
| A, equal steps (193 epochs × 1,903) | 367k | ≈ 2.0 h |
| A, equal epochs (50 × 1,903) | 95k | ≈ 0.5 h |
| B, equal steps (50 epochs × 7,353) | 368k | ≈ 1.6 h |
| **3 arms × 3 seeds** | | **≈ 12.5 h** of GPU time |

**At the planned settings** (imgsz 736, batch 16, workers 4):
- **Not measured.** The images are 32% larger, but larger batches and more workers should raise the 33% GPU utilisation.
- Expected range: 0.8–1.5× the figures above, about **10–19 h** for the full matrix.
- **Measure a single epoch at the planned settings at the start of E.1** before scheduling.

**Storage per run:** about 45 MB real (weights 21 MB, predictions 15 MB, label caches 11 MB). The 1.4 GB that `du` reports for the run folder is the hard-link view, sharing bytes with the dataset.

## 11. Verdict and recommendations for E.1

**PASS:**
- The pipeline trains on the RTX 5060 end-to-end: data view, AMP, deterministic seed, monitoring, checkpoints, inference, ignore-aware evaluation.
- Thermals (57 °C) and VRAM (≤ 1.2 GB) leave large headroom.
- Losses were finite and decreasing.
- No dataset or Task C/D artefact changed.
- 42 automated tests pass (`test_results.txt`).

**Before E.1:**
1. Set `single_cls: false` in `task_e_experiments.yaml`.
2. Start E.1 with a one-epoch throughput check at the planned settings (736, batch 16, workers 4), then confirm the time budget.
3. Keep `--data-view` and `--fresh` for every run, and select checkpoints with the ignore-aware evaluator, not `best.pt`.
4. The open D.2 conditions still apply: human gallery review, the REVIEW-handling choice (A / B-capped / ignore-aware-loss arm C), the evaluation protocol, and the dataset freeze.

**No further training was started. Waiting for approval before Task E.1.**
