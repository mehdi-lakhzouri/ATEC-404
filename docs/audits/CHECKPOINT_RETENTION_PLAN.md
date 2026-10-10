# Checkpoint Retention Plan

**Status: PLAN, nothing deleted.** Run: `experiments/task-e/full-training-config-a-200-s42` (E.1c).

## Semantics (verified in Ultralytics 8.4.174)

| File | Completed epochs | Content | Bytes |
|---|---|---|---|
| `epochN.pt` (N = 0, 10, …, 190) | **N + 1** (0-based name) | EMA weights + fp16 optimiser + AMP scaler + epoch index + train args | 10.75–10.77 MB |
| `epoch_completed_NNN.pt` (NNN = 010…200) | **NNN** (copy of `last.pt` taken by the wrapper) | same full training state | 10.75–10.77 MB |
| `best.pt` | 111 (Ultralytics fitness) | stripped (no optimiser) | 5.48 MB |
| `last.pt` | 200 | stripped | 5.48 MB |

`epoch10.pt` and `epoch_completed_010.pt` are **not duplicates**: they are one epoch apart and byte-different. Every file has a
unique SHA-256 (`checkpoint_inventory.json`).

## Minimal public set (published in Git)

| Artifact | Location |
|---|---|
| Frozen model (byte-identical to `best.pt`) | `models/artemis_rock_yolo11n.pt`, 5.48 MB, SHA-256 `413d0481…` |
| Frozen configuration, test lock, checksum lists | `artifacts/provenance/` |
| Training history, per-epoch performance, effective args (sanitised), checkpoint inventory | `artifacts/training/` |
| 22-checkpoint ignore-aware results, selection analysis, test results | `artifacts/evaluation/` |

This set is sufficient for every claim in the README, docs and notebooks.

## Local archive set (KEEP on disk)

| File(s) | Why |
|---|---|
| `best.pt`, `last.pt` | Source of the frozen model; final state |
| `epoch_completed_120.pt`, `_130.pt`, `_140.pt` | Task F comparison candidates (120 is the documented alternative with better large-rock recall) |
| `epoch_completed_200.pt` | Only full training state at the end of the run (optimiser + scaler): needed to *extend* training if ever approved |
| `ignore_aware_val/{best,epoch_completed_120,130,140}/` | Cached predictions used by `task_f_selection.py` and the test-evaluator self-test; enable re-scoring without a GPU |
| `experiments/task-e/smoke-test-001/weights/best.pt` | Regression fixture: `tests/test_training_guard.py` asserts that its `item` class name is rejected |
| All reports, `results.csv`, telemetry, `train_config.json`, `CHECKSUMS.sha256` | Provenance |

## Review set (candidates for deletion; approval required)

| Set | Files | Frees | Consequence |
|---|---|---|---|
| `weights/epoch{0..190}.pt` (0-based series) | 20 | 205 MB | None for published claims; aligned copies exist within ±1 epoch |
| `weights/epoch_completed_{010..110,150..190}.pt` | 16 | 164 MB | Loses the ability to re-run inference on those checkpoints. Their metrics stay in `ignore_aware_val_22_checkpoints.json` |
| `ignore_aware_val/` predictions of the other 18 checkpoints | 16,605 files | 184 MB | Re-generable on GPU in ~20 s per checkpoint (if the weights are kept) |
| `experiments/task-e/benchmark-e1a-001/weights/` | 3 | 21 MB | 1-epoch benchmark weights; the report keeps the evidence |
| `experiments/task-e/smoke-test-001/weights/{last,epoch0}.pt` | 2 | 15 MB | Keep `best.pt` (fixture) |

**Recommended:** delete the 0-based series, the 18 non-candidate prediction folders, the E.1a weights and the E.0 non-fixture
weights, which frees about **426 MB**. Keep all 20 `epoch_completed_*` files (215 MB), since they back the 22-checkpoint figure.
The optional extra deletion of 16 of them frees 164 MB.

**Provenance after pruning:**
- E.1c `CHECKSUMS.sha256` will list files that no longer exist. Keep it unchanged as the historical record.
- Add `PRUNED_FILES.json` (path, SHA-256, deletion date, approval reference) next to it.
- `scripts/verify_artifacts.py` checks the Task F checksum sets, which are unaffected.
