# Cleanup Execution Report

| | |
|---|---|
| Executed | 2026-10-10 |
| Approval | User approval via AskUserQuestion: "SAFE group", "Git garbage", "Duplicates & caches", "Obsolete v1 + data views". Source ZIPs: **keep** |
| Line-by-line log | `docs/audits/raw/cleanup_execution.log` (git-ignored) |

## Free disk space (C:)

| Moment | Free bytes |
|---|---|
| Before execution | 390,357,131,264 |
| After execution | 422,373,924,864 |
| **Recovered** | **≈ 32.0 GB** |

The biggest item was `git gc`, which freed 31,534,551,040 bytes. The predictions in `CLEANUP_DRY_RUN.md` were met: ≈ 31.4 GB for Git and ≈ 0.85 GB for the duplicates. Unrelated system activity adds a little noise to the totals.

## What was done

| Group | Action | Result |
|---|---|---|
| SAFE | `dataset/polar-sim/` (empty), `weights/` (already empty), 18 `__pycache__/` folders | Removed. `artemis-comparisons.zip`, `taskd-review-images.zip`, `weights/yolo26n.pt`, `audit-artemis.ps1` and `collect_taskd_review_images.ps1` had **already been removed outside this session** before execution |
| Git garbage | Exported 111 small unreachable blobs (a superset of the 21 with unique content) and the full object listing to `experiments/_git_orphans_backup/` (4.9 MB; every blob re-hashes to its ID), then `git gc --prune=now` | `.git` went from 31.4 GB to **9.2 MB**. `fsck` clean, `count-objects` 0 loose. Remote `origin` unchanged. `refs/codex/turn-diffs/...` was created by another tool (a Codex snapshot of the deliverable: 218 files, 13.8 MB, no data); it was **kept**, with its 249 reachable objects |
| Duplicates & caches | Root `polar-sim-source/`, `experiments/yolo-dataset/cache/`, 20 E.1c `epochN.pt` (0-based), 17 non-candidate `ignore_aware_val/` folders, E.1a weights, E.0 `last.pt` and `epoch0.pt`, `backend/main.py` | Removed. SHA-256 (files) or tree hashes (folders) recorded in each run's `PRUNED_FILES.json`. Historical `CHECKSUMS.sha256` files left unchanged |
| Obsolete v1 + views | `dataset/processed/yolo-rock-v1/`, three `experiments/task-e/*/data_view/` | Removed (≈ 53 MB actually freed: hard links). Recorded in `dataset/processed/RETIRED_VERSIONS.json`. `VERSIONS.json` left unchanged |

## Deviations from the dry run (all conservative)

- **`curated-dataset-v1` KEPT (0.24 GB).** It is the comparison input of 7 regression tests (`backend/tests/test_task_c_v11.py`) that
  prove the v1 → v1.1 fix. This dependency was not listed in the dry run.
- **`ignore_aware_val/last` KEPT** (about 10 MB, identical-scoring to `epoch_completed_200`), so 17 folders were removed instead of 18.
- **Extra test fixture:** `backend/tests/fixtures/yolo_rock_v1_export_exclusions.csv`, a 26 KB byte-identical copy of v1's 146
  export exclusions. `test_task_c_v11.py` now reads it, so its 11 tests still run and pass without `yolo-rock-v1`.

## Code changes needed by the retirement

| File | Change |
|---|---|
| `backend/analysis/remediation/ai_review.py` | Reads the v1.1 split manifest; split assignment is identical for all 9,193 frames |
| `backend/analysis/yolo_dataset/__main__.py` | `--reuse-splits-from` and `--link-from` default to `yolo-rock-v1.1`; refuses an `--out-name` equal to its own source |
| `backend/training/preflight.py` | `archived_v1` fingerprint covers only `curated-dataset-v1` |

## Post-cleanup verification

| Check | Result |
|---|---|
| Root tests | 29/29 OK |
| Backend tests | 52/52 OK (Task C regression tests executed, not skipped) |
| `scripts/verify_artifacts.py` | 8/8 PASS: frozen model, both Task F checksum sets (30 / 932), 17 protected files, 71 published artifacts, 3 historical runs only, test lock |
| Prepared dataset | All 7,490 release files hashed, 0 missing, 0 mismatched |
| Registry (`training/preflight.checks`) | `data_A.yaml`, split manifest and label tree match `VERSIONS.json`; 1,903 / 923 images |

## Still on disk by decision

| Item | Size | Reason |
|---|---|---|
| Source ZIPs | 7.46 GiB | Your choice: the only pristine originals |
| `curated-dataset-v1` | 0.24 GB | Test fixture |
| E.1c `epoch_completed_*` (20) | 215 MB | Back the 22-checkpoint evidence |
| `yolo-rock-v1.1/config_B` | 3.03 GiB | Planned A/B comparison |
| `frontend/` | ~0 real | Kept locally, git-ignored |
