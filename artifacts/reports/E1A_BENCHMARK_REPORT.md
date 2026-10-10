# E.1a Benchmark: YOLO11n, Config A, 1 epoch, imgsz 736, batch 16, workers 4

**Verdict: GO for E.1b planning.** No further training has started; the next run needs your explicit approval.
**Dataset:** `yolo-rock-v1.1`, PROVISIONAL, NOT FROZEN.
**Validation numbers here are PROVISIONAL and say nothing about final accuracy.**

## 1. Executive summary

- **Exactly one epoch ran.** It took 57.3 s: 24.2 s training plus 33.1 s validation and saving.
- **The checkpoints now carry `{0: 'rock'}`.** Fixed via `single_cls: false`, with 5 new regression tests.
- **GPU use roughly doubled versus E.0.** Training-window utilisation median is **77.5%** (p95 85%), against 42% / 44% in E.0.
  - Epoch-level throughput: **78.7 img/s** at 736 px, against 65.5 img/s in E.0's first epoch at 640 px. That is 1.6× more pixels per second.
  - Steady-state throughput: **about 120 img/s**.
- **Data loading is not the bottleneck.** The measured wait between batches is 0.2 ms (0.1% of batch time). CPU use is 15% of 32 logical cores.
  - The remaining GPU idle time comes from a single busy host thread (one core at about 91%).
  - It also comes from about 6 s of epoch start-up (spawning workers on Windows).
- **Resources are comfortable and safe.**
  - VRAM: up to 3.6 GB of 8 (`nvidia-smi`); CUDA peak 3.0 GB allocated, 3.2 GB reserved.
  - Temperature 65 °C, power up to 78.5 W.
  - No thermal or power throttling, no OOM, no NaN.
- **Full E.1 matrix estimate** (3 arms × 3 seeds): about **9.4–15.8 h**, extrapolated (§11). Disabling per-epoch Ultralytics validation would save 1.5–4 h.

## 2. Pre-flight (`preflight_pre.json`): 15 of 15 checks pass

**Dataset and versions**
- `data_A.yaml` (`fc5f0fdc…`), `split_manifest.csv` (`3274dcd2…`) and the labels tree all match `dataset/processed/VERSIONS.json`.
- 1,903 train and 923 val images, all with labels; class map `{0: rock}`, `nc: 1`.
- All 311,299 label lines in the 11,096 label files use class 0.
- The dataset is not frozen.

**Weights and environment**
- `yolo11n.pt` sha256 `0ebbc80d…44ee1` (official v8.4.0 release).
- RTX 5060 Laptop, sm_120 in the torch 2.14.1+cu130 build, 7.93 GB.
- AMP checks: fp16 and bf16 both pass.

**Run environment**
- Disk free: 372.6 GB.
- No earlier checkpoint in the run folder.
- **No concurrent GPU training job**: GPU at 0%, 217 MiB.

**Warning recorded:** two Brave browser processes held a hardware-accelerated GPU context (type C+G) throughout. They aren't training jobs, but they may perturb the measurement slightly.

**Change to the check:** my first check was over-broad and flagged any GPU client. It now fails on other Python/CUDA processes or a busy GPU (≥ 10% or ≥ 1 GB), and lists other clients as warnings.

**Post-flight** (`preflight_post.json`): `all_pass: true`. Unchanged, by path, size and mtime fingerprints:
- source datasets;
- Task C v1.1;
- Task D v1.1;
- archived v1;
- **smoke-test-001**.

## 3. Effective configuration (verified inside the trainer before the first batch)

The wrapper's `on_train_start` check compared Ultralytics' **effective** arguments with the authorised ones and would have aborted on any difference. Verified values:

| Setting | Value |
|---|---|
| epochs | 1 |
| batch | 16 |
| imgsz | 736 |
| workers | 4 |
| seed | 42 |
| optimizer | SGD |
| amp | True |
| deterministic | True |
| cache | False |
| resume | False |
| time | None |
| single_cls | False |
| class names | `{0: rock}` |
| model | `yolo11n.pt` (pretrained) |

**Unchanged project hyperparameters:**
- lr0 0.01, momentum 0.937, weight decay 5e-4, cosine LR;
- warm-up 3 epochs, so the single epoch is entirely warm-up;
- mosaic 1.0, fliplr 0.5, hsv_v 0.3, hsv_h/s 0, flipud 0;
- patience 0, save_period 5.

The full effective list is in `args.yaml` and `train_config.json → effective_args_all`.

**Risks found in Phase 1 and how they were neutralised:**

| Risk | Effect | Neutralised by |
|---|---|---|
| Ultralytics' **automatic OOM retry** | In epoch 1 it halves the batch and retries up to 3× | `on_train_epoch_start` and `on_train_batch_end` stop on any batch change or OOM retry |
| `time` argument | Recomputes the epoch count from a time budget | Asserted `None` |
| Extra epochs | — | `epochs` asserted and an epoch counter guard |
| Resume | — | `--fresh` refuses an existing `last.pt` |
| `save_period 5` | Writes an extra `epoch0.pt` | Harmless; 11 MB |

## 4. Command and authorisation scope

```
cd backend
$env:ARTEMIS_TRAINING_AUTHORIZED=1        # this single command only
uv run --no-sync python -m training.train --arm A_steps --seed 42 --authorized --fresh --data-view `
  --run-dir ../experiments/task-e/benchmark-e1a-001 `
  --set epochs=1 imgsz=736 batch=16 workers=4 device=0 amp=true cache=false deterministic=true `
        optimizer=SGD pretrained=true single_cls=false plots=true
```

- **Scope:** one authorised 1-epoch Config A benchmark.
- The data came from a hard-link view in `data_view/`, so no cache files were written into the dataset.
- The authorisation guard is unchanged.
- No further run was started.

## 5. Exactly one epoch

- `results.csv` has 1 row.
- The log says "1 epochs completed in 0.013 hours".
- `train_config.json`: `status COMPLETED`, `resumed false`, one epoch record.
- Exit code 0.
- 119 iterations of 16 images.

## 6. GPU telemetry (`nvidia_smi.csv` at 1 s; `resources.csv` at 1 s with in-process CUDA memory)

| Window | Samples | GPU util mean / median / p95 / max | Memory-controller util | VRAM max | Temp max | Power mean / max | SM clock | Throttling |
|---|---|---|---|---|---|---|---|---|
| Training (24.2 s) | 24 | 51.2 / **77.5** / 84.8 / 89 % | 39.2% | 3,630 MiB | 65 °C | 47.2 / 78.5 W | 225–2,790 MHz | 0 thermal, 0 power cap |
| Validation + save (33.1 s) | 33 | 29.1 / 12 / 78.8 / 84 % | 9.2% | 3,823 MiB | 61 °C | 24.3 / 63.5 W | up to 2,842 MHz | 0 / 0 |
| Whole run | 65 | 34.4 / 12 / 84 / 89 % | 19.2% | 3,823 MiB | 65 °C | 30.9 / 78.5 W | — | 0 / 0 |

- **CUDA memory** (torch, epoch peak): 3,006 MB allocated, 3,238 MB reserved.
- **Raw throttle bitmask:** `clocks_event_reasons.active` read `0x400` in every sample, including idle ones, as in E.0. It's static, and the thermal and power-cap flags were all "Not Active".
- **Why the training mean is low:** the mean (51%) sits below the median (77.5%) because of the start-up ramp before the first batch.

## 7. CPU and data-loader analysis

**Per-batch timing** (`batch_timing.csv`: timestamps only, no added GPU synchronisation; Ultralytics' progress bar reads the loss each iteration, which already synchronises):

| Measure | Value |
|---|---|
| Training step (GPU preprocess + forward/backward + optimiser) | mean 0.153 s, median 0.125 s, p95 0.142 s; steady state (batch ≥ 10) 0.133 s → **≈ 120 img/s** |
| Wait for the next batch from the data loader | mean **0.2 ms**, max 0.4 ms (0.1% of batch time) |
| First batch | 2.36 s (cuDNN autotune and warm-up) |
| Epoch start-up before the first batch, plus teardown | **≈ 6.0 s** (24.19 s epoch − 18.21 s of batches); includes spawning the 4 worker processes on Windows |

**CPU and memory** (training window, aligned to `nvidia-smi` timestamps within about ±1 s):

| Measure | Value |
|---|---|
| Overall CPU | mean 14.9%, p95 20.4% of 32 logical cores |
| Busiest single core | mean **91%** |
| Process tree (main + workers) | mean 205% (≈ 2.1 cores), max 421% |
| Data-loader workers seen | 1–4 during training. During validation 5–12, because Ultralytics adds 2 × workers validation workers while the training workers stay alive |
| System RAM | 15.8–19.8 GB in training; **20.1–25.6 GB of 32 GB during validation** |
| Process-tree RSS | max 7.3 GB, rising to 14.4 GB in validation (Windows RSS double-counts shared pages) |
| Disk read | mean 0.5 MB/s: images came from the OS file cache |

**Conclusion (evidence-based):**
- The data pipeline keeps the GPU fed. Loader waits are effectively zero, and CPU capacity is mostly idle.
- The remaining gap to full GPU use is host-side per-step overhead on one thread, plus epoch start-up. It is **not** image decoding.
- The E.0 hypothesis of a CPU or data-loading bottleneck was only partly right. At batch 4, small per-step GPU work made host overhead dominant; batch 16 amortises it.

**Config B decode check** (CPU only, no training): 300 matched images.
- Config B PNGs decode 16% slower (5.76 vs 4.96 ms per image, single thread).
- With 4 workers that is still about 690 img/s of decode capacity against about 120 img/s consumed, so **no throughput change is expected** for Config B. This is inferred, not measured on GPU.

## 8. Comparison with E.0

E.0 training-window figures were re-derived from its logged telemetry for a like-for-like comparison.

| Metric | E.0 (smoke-test-001) | E.1a (this run) |
|---|---|---|
| Resolution / batch / workers | 640 / 4 / 2 | 736 / 16 / 4 |
| GPU util, training window (mean / median / p95) | 38 / 42 / 44 % | 51 / **77.5** / 85 % |
| GPU util, whole run (mean) | 33% | 34% (1 epoch, so validation and start-up weigh heavily) |
| Peak VRAM (`nvidia-smi`) | 1,151 MiB | 3,823 MiB |
| Peak CUDA allocated / reserved | 660 / 762 MB | 3,006 / 3,238 MB |
| Throughput, epoch level (first epoch) | 65.5 img/s (62–79 over 3 epochs) | **78.7 img/s** at 1.32× the pixels per image |
| Throughput, steady state | ≈ 64–80 img/s (16–20 it/s × 4) | **≈ 120 img/s** |
| Training time per epoch | 24.1–30.6 s | 24.2 s |
| Validation time per epoch | 6–15 s at 640 | 33.1 s at 736, including first-time validation-worker spawn and final plots |
| Max temperature | 57 °C | 65 °C |
| Max power | 47 W | 78.5 W |

**Throughput and utilisation:**
- **Better.** The GPU now does about 1.6× more pixel work per second and is busy about 78% of the time during training.
- **Validation is the new relative cost.** At 736 px, with 8 extra validation workers spawned per validation, it took longer than training in this 1-epoch run.

## 9. Class mapping fix

- **Cause:** `single_cls: true` in `task_e_experiments.yaml` made Ultralytics rename the class to `item`.
- **Fix:** it is now `false`. The dataset already has exactly one class, declared as `names: {0: rock}`, `nc: 1` in `data_A.yaml` and `data_B.yaml`. No label file was changed.
- **Result:**
  - `best.pt` / `last.pt` names: `{0: 'rock'}`.
  - Inference metadata: `{0: 'rock'}`.
  - The wrapper now fails a run as `FAILED_CLASS_NAMES` if a checkpoint has any other mapping.

**Regression tests** (`tests/test_training_guard.py::TestClassNaming`, 5 tests):
- the validator accepts only `{0: 'rock'}`;
- all arms resolve to `single_cls: False`;
- both data YAMLs declare `{0: rock}`;
- **the preserved smoke-test checkpoint (`item`) is rejected**;
- an effective-argument mismatch (epochs, batch, `time`, `single_cls`) is a limit violation.

**Full suite:** 47 tests, all pass (`test_results.txt`).

The smoke-test-001 checkpoints and results are untouched; their fingerprint is unchanged.

## 10. Checkpoint and inference verification (`benchmark_checks.json`, `inference/`)

**Checkpoints:**

| | `best.pt` | `last.pt` |
|---|---|---|
| Size | 5,458,586 B | 5,458,586 B |
| SHA-256 (see `CHECKSUMS.sha256`) | `2b8b5f5e…` | `7ad388fb…` |
| Parameters | 2,590,035, all finite | 2,590,035, all finite |
| Class names | `{0: rock}` | `{0: rock}` |

`epoch0.pt` is 11 MB because it includes optimiser state.

**Provisional Ultralytics validation after 1 epoch** (no ignore support): P 0.282, R 0.292, mAP50 0.201, mAP50-95 0.072.
- This is lower than E.0's epoch 1 (0.339) **as expected, not as a regression**.
- Batch 16 gives 4× fewer optimiser steps per epoch (119 vs 476), and the whole epoch was learning-rate warm-up.
- Not an accuracy measure.

**Inference** (imgsz 736, conf 0.25; val/test images never trained on):

| Frame | Scene | Predictions | VALID GT | REVIEW | TP | FP | Ignored |
|---|---|---|---|---|---|---|---|
| 0189 (test) | sparse | 2 | 1 | 0 | 1 | 1 | 0 |
| 5589 (val) | dense | 281 | 110 | 34 | 54 | 152 | 75 |
| 0586 (test) | dark / shadow | 102 | 32 | 9 | 16 | 69 | 17 |

The 1-epoch model over-predicts heavily, as expected so early. These checks confirm only **compatibility**: the checkpoint loads, the class names are right and the overlays render.

## 11. E.1 duration estimates

**Measured** (this run):
- training at 8.33 ms per image steady state (≈ 120 img/s);
- per-epoch start-up ≈ 6.0 s;
- validation plus save 33.1 s, including one-time and final-epoch costs.

**Extrapolated:**
- **Optimistic:** steady-state rate only; validation ≈ 15 s per epoch (non-final epochs skip the plots, and workers may persist).
- **Pessimistic:** measured epoch costs repeated every epoch (24.2 s for A, 61.2 + 6 s for B, plus 33 s validation).
- Checkpoint saving is under 1 s per epoch and ignored.
- Config B is assumed to have the same per-image rate (§7).

| Arm (per seed) | Epochs × train images | Optimistic | Pessimistic |
|---|---|---|---|
| A, equal epochs | 50 × 1,903 | 0.43 h | 0.79 h |
| A, equal steps | 193 × 1,903 | 1.66 h | 3.07 h |
| B, equal steps / epochs | 50 × 7,353 | 1.06 h | 1.39 h |
| **One seed of all 3 arms** | | **3.15 h** | **5.25 h** |
| **3 seeds** | | **≈ 9.4 h** | **≈ 15.8 h** |

- **Validation dominates A-equal-steps:** 193 validations take 0.8–1.8 h per seed.
- Since checkpoints are selected with the ignore-aware evaluator, `val: false` (Ultralytics validates only at the end) would save about 1.5–4 h over the matrix. That would need approval.
- The ignore-aware evaluation of saved checkpoints adds about 15–30 s per checkpoint.

**Disk projection:**

| Item | Size |
|---|---|
| `best.pt` / `last.pt` | 5.3 MB each |
| `epochN.pt` every 5 epochs | 11 MB each: A equal steps 39 files ≈ 430 MB, B and A equal epochs 10 files ≈ 110 MB each |
| Predictions for evaluating each checkpoint | about 15 MB each |
| Whole matrix (3 seeds) | about **2 GB checkpoints + 2.7 GB predictions ≈ 5 GB peak** |

**Recommended retention:** after checkpoint selection, keep `last.pt`, the ignore-aware-selected checkpoint, `results.csv`, the evaluation JSON and the telemetry, and delete the intermediate `epochN.pt` files and prediction text files. That leaves about 100 MB per run, about 1 GB for the matrix.

## 12. Recommended configuration for E.1b / E.1c

**Keep:**
- imgsz 736, batch 16, workers 4, AMP, seed list, deterministic, SGD, the project augmentations;
- `single_cls: false`;
- `--data-view` and `--fresh`;
- the limit callbacks: epochs, batch and OOM-retry guard.

**Proposed changes (need approval):**
1. `val: false` plus ignore-aware selection over the `save_period` checkpoints, to remove most of the validation overhead.
2. Close browsers and other GPU clients during benchmarks and runs.
3. Keep batch 16. VRAM headroom is about 4 GB, but larger batches change optimisation dynamics, which would be a hyperparameter change, so it is **not** recommended without an experiment.

**Do not change:** the REVIEW-handling arm design or the evaluation policy from D.2. They are still pending your decisions.

## 13. Outstanding risks

1. All labels are automated and provisional. The human gallery review and the policy decisions from D.2 are still open.
2. Duration estimates are extrapolated from one epoch. Later epochs (mosaic close, longer-run thermal behaviour, worker persistence) aren't measured.
3. **Validation RAM peak of 25.6 GB of 32 GB:** watch it over long runs, and close other applications.
4. Bitwise reproducibility on GPU isn't guaranteed; compare across seeds.
5. Config B throughput is inferred, not measured on the GPU.

## 14. Verdict

**GO** for preparing E.1b / E.1c. The configuration is correct and safe: class `rock`, limits enforced, data and earlier experiments untouched, and measured performance is good.

Any further training (full Config A, Config B, multiple seeds, or a `val: false` variant) **requires your explicit authorisation**. Nothing has been started.

## Files

| File | Content |
|---|---|
| `train_console.log`, `args.yaml`, `train_config.json` | Log, effective arguments, epoch records, status |
| `results.csv`, `results.png`, Ultralytics plots | Training results |
| `weights/` | Checkpoints |
| `nvidia_smi.csv`, `resources.csv`, `batch_timing.csv` | Raw telemetry |
| `preflight_pre.json`, `preflight_post.json` | Integrity checks |
| `benchmark_checks.json`, `inference/` | Checkpoint and inference validation |
| `pre_run_test_results.txt`, `test_results.txt` | Test runs |
| `CHECKSUMS.sha256` | Checksums |

`data_view/` holds hard links plus label caches. Real new storage is **31 MB**.
