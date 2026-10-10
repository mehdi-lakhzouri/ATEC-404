# E.1c: YOLO11n Full Fine-Tuning, Config A, 200 Epochs, Seed 42

**Verdict: PASS.** The run completed and is technically valid.
**Dataset:** `yolo-rock-v1.1`, PROVISIONAL, NOT FROZEN; labels are automated and not human-verified.
**Accuracy figures are research-stage and provisional. Nothing here supports rover-deployment readiness.**

## A. Executive summary

| Item | Value |
|---|---|
| Status | `COMPLETED`: 200 of 200 epochs, exit 0, `results.csv` has 200 rows |
| Wall clock | 2026-10-09 21:02:23 → 22:11:04: **68.7 min**. The 200 epochs took 1.14 h; final validation of `best.pt` took 7.2 s |
| Epoch duration | mean 20.5 s, median 19.8 s: 15.6 s training plus 4.9 s validation and save |
| Training throughput | median **125 img/s**, mean 123 img/s |
| GPU | training-window utilisation median **87%**; VRAM max 5.5 GB of 8; temperature max **71 °C**; 0 thermal-throttle samples; 1 one-second power-cap sample |
| Stability | no OOM, no NaN, no repeated epoch, no limit violation, no resource warning |
| RAM | available never below 5.3 GB of 31.2 GB; no sample below 4 GB |
| Checkpoints | 42 files verified: all load, all parameters finite, all `{0: 'rock'}` |
| Integrity | post-flight passes; datasets, Task C/D, E.0 and E.1a unchanged |
| Provisional validation (Ultralytics, no ignore regions) | best fitness at **epoch 111**: P 0.638, R 0.667, mAP50 0.659, mAP50-95 0.372 |
| Ignore-aware validation (D.2 primary policy) | `best.pt`: AP50 **0.776**, AP50-95 0.432. The plateau runs from about epoch 70; there is a slight decline after epoch 150 |

The run took about a quarter of the E.1a pessimistic estimate (68.7 min against 9.4–15.8 h for the whole matrix). E.1a's 33 s "validation + save" also included the final `best.pt` validation and one-time start-up; routine per-epoch validation takes about 4.9 s.

## B. Provenance

**Model:**
- `backend/yolo11n.pt`, pretrained, official v8.4.0.
- SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Training did not start from smoke-test or benchmark weights.

**Dataset:** `yolo-rock-v1.1`, Config A.
- 1,903 train images; 923 validation images. The 917 test images were **not used**.
- Split manifest `3274dcd2…6576`. `data_A.yaml`, the manifest and the label tree all match `VERSIONS.json` before and after the run.
- Training read from a hard-link view (`data_view/`), so label caches never touched the dataset.

**Software:** Python 3.12.13, torch 2.14.1+cu130, Ultralytics 8.4.174, Windows 11.

**Hardware:** RTX 5060 Laptop GPU (sm_120, 7.93 GB), on AC power, with sleep disabled on AC.

**Configuration:** checked against the trainer's effective arguments before the first batch.

| Setting | Value |
|---|---|
| epochs | 200 |
| imgsz | 736 |
| batch | 16 |
| workers | 4 |
| device | 0 |
| amp | true |
| cache | false |
| seed | 42 |
| optimizer | SGD |
| lr0 | 0.01 |
| momentum | 0.937 |
| weight decay | 5e-4 |
| cos_lr | true |
| warmup_epochs | 3 |
| single_cls | false |
| deterministic | true |
| close_mosaic | 10 |
| save_period | 10 |
| patience | 0 (early stopping disabled; Ultralytics 8.4 maps 0 to infinity) |
| resume | false |
| time | None |
| augmentation | project policy unchanged: hsv_h/s 0, hsv_v 0.3, fliplr 0.5, flipud 0, mosaic 1.0 |

The full arguments are in `args.yaml`, `train_config.json → effective_args_all` and `dry_run.json`.

**Authorisation:** `ARTEMIS_TRAINING_AUTHORIZED=1` plus `--authorized` plus `--fresh`.
- `runs_registry.jsonl` gained exactly **one** entry: Config A, seed 42, 200 epochs.
- No Config B training, no other seed, no retries.

**Wrapper hardening before launch** (covered by 5 new regression tests; suite 52/52 OK before and after the run):
- A Ctrl-C or kill is now recorded as `INTERRUPTED`; previously the `finally` block could report `COMPLETED`.
- Status is `RUNNING` while training. `train_config.json` is rewritten atomically after every epoch, and `batch_timing.csv` is appended per epoch.
- **Repeated-epoch guard:** Ultralytics 8.4 silently redoes an epoch after a NaN loss or fitness by reloading `last.pt`. The guard turns that into a stop.
- The final status is `FAILED_EPOCH_COUNT` unless exactly the authorised number of epochs completed.
- The Ultralytics `final_eval` callback no longer overwrites the last epoch's timing. Its metrics now go to `final_eval_best_pt`.
- Copies aligned to completed epochs (`epoch_completed_010…200.pt`), explained in §E.
- Pre-flight additions:
  - E.1a is added to the protected fingerprints;
  - available RAM must be ≥ 16 GB;
  - the machine must be on AC power;
  - `resources.csv` gains `ram_available_gb`.

## C. Training performance and convergence

Plots: `analysis/training_curves.png` (losses, P/R, mAP, LR), `results.png` (Ultralytics) and `analysis/epoch_performance.png`.

| Milestone | Train box / cls / dfl | Val box / cls / dfl | P / R (Ultralytics) | mAP50 / mAP50-95 (Ultralytics) | Primary ignore-aware AP50 / AP50-95 | lr |
|---|---|---|---|---|---|---|
| 50 | 1.589 / 0.943 / 0.941 | 1.502 / 1.044 / 0.982 | 0.623 / 0.635 | 0.638 / 0.348 | 0.750 / 0.405 | 0.0086 |
| 100 | 1.479 / 0.855 / 0.914 | 1.499 / 1.119 / 0.986 | 0.595 / 0.628 | 0.608 / 0.336 | 0.725 / 0.396 | 0.0051 |
| 150 | 1.386 / 0.775 / 0.889 | 1.446 / 0.990 / 0.973 | 0.641 / 0.667 | 0.646 / 0.366 | 0.772 / 0.431 | 0.0016 |
| 200 | 1.290 / 0.723 / 0.889 | 1.463 / 1.007 / 0.982 | 0.633 / 0.666 | 0.639 / 0.358 | 0.764 / 0.424 | 0.0001 |

**Observed in the data:**
- **Training losses fell steadily the whole time.** Box loss went from 1.589 to 1.290 between epochs 50 and 200, and classification loss from 0.943 to 0.723.
- **Validation losses stopped improving at epochs 104–138.** Best values: box 1.439 at epoch 121, classification 0.970 at epoch 138, DFL 0.964 at epoch 104. Over the last 20 epochs they rose slightly (box +0.004 and classification +0.004 per 10 epochs).
- **Ultralytics mAP plateaued from about epoch 40.** mAP50 stayed around 0.61–0.66, peaking at epoch 97. mAP50-95 peaked at epoch 111 (0.372).
- **Ignore-aware AP followed the same shape:** a rise to about epoch 40, a plateau from 0.75 to 0.775 from epoch 70, and a decline from 0.772 to 0.764 over epochs 150–200.
- **The learning-rate schedule ran as configured:** a 3-epoch warm-up, then a cosine decay from 0.01 to 1e-4 that was uninterrupted.
- **Closing mosaic at epoch 191 lowered training loss** (box 1.338 → 1.290) without improving validation.
- Single checkpoints vary noticeably; epoch 100 shows a dip to AP50 0.725.

**Interpretation (hypotheses, not proven):**
- The model **converged** by about epoch 100–120.
- Training past about epoch 150 shows **mild overfitting**: training loss falls while validation loss and AP drift down. The overfitting may be to label noise.
- **Longer training is not justified** by this data.
- An annotation-noise ceiling is plausible: the ignore-aware gain of about 0.12 AP50 and the visible label artefacts (Task F) point that way. This can't be separated from model capacity without human-verified labels.
- Results come from synthetic validation renders only. They say **nothing about real lunar imagery**.

## D. GPU and system performance

Data: `nvidia_smi.csv` (4,098 samples at 1 s), `resources.csv` (1 s), `batch_timing.csv` (23,800 batches) and `analysis/per_epoch_performance.csv`. Plot: `analysis/hardware_telemetry.png`.

| Metric | Training windows | Validation + save windows |
|---|---|---|
| GPU util mean / median / p95 | 83.6 / **87** / 94 % | 47.9 / 49 / 83 % |
| Memory-controller util median | 70% | — |
| Power mean / max | 76.4 / 103.4 W | — |
| SM clock median | 2,782 MHz | — |

- **VRAM** (`nvidia-smi`, all processes): median 4.1 GB, max 5.5 GB.
- **Torch per epoch:** peak allocated 3.49 GB, peak reserved 5.30 GB.
- **Temperature:** mean 65.7 °C, max 71 °C.
- **Throttling:** 0 thermal-slowdown samples. 1 power-cap sample at 21:33:05, lasting 1 s, about epoch 90.
- **Throughput:**
  - epoch-level median 125.4 img/s (min 78 in epoch 1 with start-up and caching; 86.7 in epoch 191 with the mosaic-close loader rebuild);
  - batches only: median 125.6 img/s.
- **CPU:** mean 17.4% of 32 logical cores; workers 4 during training, up to 12 during validation.
- **RAM:**
  - used max 25.9 GB, in the first 6 minutes only;
  - steady state about 17.6 GB;
  - available minimum 5.3 GB;
  - no growth trend over 69 min, so no leak.

**Slow windows (observed):**

| Epochs | Throughput | Change | GPU util (median) |
|---|---|---|---|
| 124–132 | about 101 img/s | −19% | about 70% |
| 161–169 | about 111 img/s | −11% | — |
| 90–91 | about 101–111 img/s | — | — |

- In these windows the GPU was **cooler and drew less power**, so they aren't thermal.
- The training process tree used **more** CPU (≈ 3.6 cores against ≈ 2.7) for the same work.
- **Hypothesis:** host-CPU slowdown, for example lower CPU clocks under the laptop's shared power budget, or background OS activity. CPU frequency wasn't logged, so this is unverified.
- Impact: about +1 min of wall time; none on correctness.

## E. Checkpoint inventory (`checkpoint_inventory.json`, `CHECKSUMS.sha256`)

**Naming in Ultralytics 8.4.174.** `epoch{N}.pt` uses the **0-based** epoch index, so `epoch10.pt` holds 11 completed epochs. The wrapper therefore also copies `last.pt` to `epoch_completed_{010…200}.pt` after epochs 10, 20, …, 200. Both series are kept: 40 periodic files.

| Files | Completed epochs | Size | Optimiser, scaler and EMA state (resumable) | Names | Finite |
|---|---|---|---|---|---|
| `epoch_completed_010…200.pt` (20) | 10, 20, …, 200 | 10.75–10.77 MB | yes | `{0: rock}` | yes |
| `epoch0…epoch190.pt` (20) | 1, 11, …, 191 | 10.75–10.77 MB | yes | `{0: rock}` | yes |
| `best.pt` | 111 (Ultralytics fitness) | 5.48 MB | no (stripped at the end) | `{0: rock}` | yes |
| `last.pt` | 200 | 5.48 MB | no (stripped) | `{0: rock}` | yes |

**SHA-256 of key files:**

| File | SHA-256 |
|---|---|
| `best.pt` | `413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb` |
| `last.pt` | `f24a35ef7c97…` |
| `epoch_completed_200.pt` | `52167b452797…` |

Every hash is in `checkpoint_inventory.json` and `CHECKSUMS.sha256`.

**Resume safety:**
- The periodic checkpoints hold the EMA model, optimiser (fp16), AMP scaler, epoch index, `best_fitness` and train arguments.
- Ultralytics rebuilds the learning-rate scheduler from the epoch index (`scheduler.last_epoch = start_epoch − 1`), so the schedule continues exactly.
- **RNG and data-loader state are not saved**, so a resumed run would not be bitwise identical.
- `best.pt` and `last.pt` are stripped inference checkpoints and **cannot** resume training.
- No resume was performed.

**Storage:** about 700 MB of real data: 422 MB of weights, 261 MB of cached ignore-aware predictions, plus telemetry. `data_view/` is hard links.

**Recommended retention (not applied; you said no deletions):**
- keep `best.pt`, `last.pt`, `epoch_completed_*` and the cached predictions until Task F closes;
- after that, the `epoch{N}.pt` series (about 215 MB) duplicates the aligned series ±1 epoch and could be pruned with approval.

## F. Provisional evaluation

All checkpoints were scored on **val only** with the unchanged D.2 evaluator:
- imgsz 736, conf 0.001, NMS IoU 0.6, max_det 300;
- match IoU 0.5; ignore rule IoA ≥ 0.5 on REVIEW regions; min_pred_area 50 px²; 101-point AP.

**Standard and ignore-aware metrics are reported separately.** The about +0.12 AP50 of the ignore-aware score over the naive one comes from the evaluation policy, **not from the model**.

| Checkpoint | Primary AP50 | Primary AP50-95 | AP50 + sub-threshold (PROPOSED) | Naive AP50 (no ignores) | Ultralytics mAP50 (provisional) |
|---|---|---|---|---|---|
| epoch 10 | 0.678 | 0.329 | 0.710 | 0.555 | 0.542 |
| epoch 50 | 0.750 | 0.405 | 0.786 | 0.644 | 0.638 |
| epoch 70 | 0.774 | 0.426 | 0.794 | 0.650 | 0.645 |
| epoch 100 | 0.725 | 0.396 | 0.773 | 0.613 | 0.608 |
| epoch 120 | 0.775 | 0.433 | 0.798 | 0.652 | — |
| epoch 140 | 0.775 | 0.433 | 0.798 | 0.645 | — |
| epoch 200 / `last.pt` | 0.764 | 0.424 | 0.795 | 0.642 | 0.639 |
| **`best.pt` (epoch 111)** | **0.776** | 0.432 | 0.802 | 0.663 | 0.659 |

All 22 rows are in `ignore_aware_val.json`. `last.pt` and `epoch_completed_200.pt` give identical results, which serves as a consistency check.

**Label limitations:**
- Labels are automated S2-H VALID boxes, and REVIEW regions are ignored.
- Task F's visual inspection found oversized or merged boxes, shadow-only boxes and edge slivers among the "missed" large rocks.
- The metrics are therefore biased in an unknown direction.

## G. Error and risk analysis

**Fitting:**
- **Overfitting:** mild after about epoch 150, per §C.
- **Underfitting:** none at the scale of this model and data.

**Hardware:**
- RAM was the tightest resource: 5.3 GB minimum available. It stayed within safe limits.
- One 1 s power-cap event and two CPU-side slow windows had no effect on correctness.
- Two Brave browser GPU contexts were present throughout. They weren't training jobs.

**Reproducibility:**
- One seed only, so checkpoint-to-checkpoint differences of about ±0.01 AP50 can't be told apart from noise.
- `deterministic=true` doesn't guarantee bitwise GPU reproducibility.

**Labels:**
- Labels aren't human-validated.
- REVIEW regions are numerous: up to 110 in one image.
- Small rocks dominate the ground truth (17,001 of 25,822).

**Domain:** synthetic renders only. There has been no NASA POLAR or real-image evaluation, as the brief required.

## H. Recommendations for Task F (since carried out; see `experiments/task-f/model-selection-001/`)

- **Candidates:** `best.pt` (epoch 111), then epochs 120, 130 and 140. They are tied within 0.006 primary AP50.
- **Evaluation:** use the primary ignore-aware policy on val only. Analyse thresholds, recall by size and false-negative categories.
- **Further training:** another experiment of the **same** setup isn't justified, because the run plateaued.
- **What might help, each needing your approval:**
  - human-verified labels (E.1b);
  - multi-seed runs, to measure variance;
  - Config B;
  - a larger model.

## I. Final verdict

**PASS.** Exactly one authorised 200-epoch Config A, seed 42 run completed. Checkpoints are verified, the class is `rock`, data integrity is confirmed, and telemetry and convergence are documented.

The model and metrics remain **research-stage and provisional**. The test split was not used. Rover-deployment readiness isn't claimed.

## Files

| File | Content |
|---|---|
| `train_console.log`, `args.yaml`, `dry_run.json`, `train_config.json` | Log, effective arguments, per-epoch records, status |
| `results.csv`, `results.png`, Ultralytics plots | Training results |
| `analysis/` | `summary.json`, curves, telemetry and per-epoch performance |
| `nvidia_smi.csv`, `resources.csv`, `batch_timing.csv` | Raw telemetry |
| `preflight_pre.json`, `preflight_post.json` | Integrity checks |
| `checkpoint_inventory.json` | Checkpoint verification |
| `ignore_aware_val.json`, `ignore_aware_val/` | Ignore-aware results and cached predictions |
| `pre_run_test_results.txt`, `test_results.txt` | 52 tests OK |
| `CHECKSUMS.sha256` | Checksums |
