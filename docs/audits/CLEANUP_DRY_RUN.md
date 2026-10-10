# Cleanup Dry Run (Phase 8)

**EXECUTED on 2026-10-10 for the approved groups. See `CLEANUP_EXECUTION_REPORT.md`.**

| Approval decision | Items |
|---|---|
| Approved and done | SAFE, R1, R4–R12 (except `curated-dataset-v1`, kept as a test fixture) |
| Declined (kept) | R2/R3 (source ZIPs), R13, R14 |
| Not acted on | R15–R17 (R17 scripts were already removed externally) |

The table below is the original dry run.

Measured on 2026-10-09 with hard links resolved: **Frees** = bytes actually released when that row is deleted. Disk C: had
**371 GB free** at audit time. Data: `RETENTION_MANIFEST.json`, `STORAGE_ANALYSIS.md`.

## SAFE: disposable, no dependencies, verified redundant or regenerable

| # | Candidate | Reason | Dependencies checked | Canonical alternative | Apparent | Frees | Risk | Action | Approval |
|---|---|---|---|---|---|---|---|---|---|
| S1 | `artemis-comparisons.zip` | Share package | No code references | All 117 members CRC-identical in `experiments/annotation-validation/comparison_sheets/scenes/` | 52 MB | 52 MB | none | DELETE | yes |
| S2 | `taskd-review-images.zip` | Share package | No code references | Source renders + `yolo-rock-v1` Config B; regenerable with the root `.ps1` | 21 MB | 21 MB | none | DELETE | yes |
| S3 | `weights/yolo26n.pt` | Side effect of the Ultralytics AMP self-check | Not referenced; re-downloaded automatically if needed | — | 5 MB | 5 MB | none | DELETE | yes |
| S4 | `dataset/polar-sim/` | Empty directory | Task A flagged it (HK-02) | — | 0 | 0 | none | DELETE | yes |
| S5 | `__pycache__/` (backend, src, tests, scripts; not `.venv`) | Byte-code cache | Regenerated on import | — | 1 MB | 1 MB | none | DELETE | yes |

**SAFE subtotal: about 0.08 GB.**

## REVIEW: obsolete, duplicate or regenerable, needs a decision

| # | Candidate | Reason | Dependencies checked | Backup / canonical alternative | Apparent | Frees | Risk | Action | Approval |
|---|---|---|---|---|---|---|---|---|---|
| R1 | `.git/objects` (47,954 unreachable objects, loose **and** packed since GitKraken maintenance at 23:40) | Left by a discarded `git add` of the whole workspace; no commit, index, ref or reflog | 47,625 byte-identical to current files; 96 identical after CRLF→LF; 1 truncated (corrupt) copy of `archive.zip`; **only 21 small blobs (126 KB) of superseded logs/status files are unique** (`GIT_INTEGRITY_REPORT.md`) | Working-tree files; the 126 KB of unique blobs are exported first | 29.27 GiB | **29.27 GiB** | low | Back up the 21 blobs, then `git gc --prune=now` (close GitKraken first) | yes |
| R2 | `dataset/artificial-lunar/archive.zip` | CRC-identical to the extraction | Pipeline reads the extraction only. The Git copy of this ZIP is truncated and unusable, so this file is the **only intact original archive** | **Requires** a confirmed external copy (Kaggle or your backup) | 5.02 GiB | **5.02 GiB** | medium | DELETE only if backed up, else KEEP | yes |
| R3 | `dataset/nasa-polar/Terrain*.zip` (3) | CRC-identical to the extractions | Not read by code | Same condition | 2.44 GiB | **2.44 GiB** | medium | DELETE if backed up, else KEEP | yes |
| R4 | `polar-sim-source/` (root) | Duplicate content of `dataset/polar-sim-source/` | Only *reported* by Task A inventory code | Content in `dataset/polar-sim-source/`; history on github.com/uwsbel/POLAR-Sim. Since 23:29 this is the only *local* clone with `.git` | 0.71 GiB | 0.71 GiB | low | DELETE (re-clonable) | yes |
| R5 | `experiments/yolo-dataset/cache/orb_features.npz` | Near-duplicate feature cache | Used only to rebuild the split; the split is fixed in `split_manifest.csv` | Regenerable (CPU minutes) | 425 MB | 425 MB | low | DELETE | yes |
| R6 | E.1c `weights/epoch{0..190}.pt` (20) | 0-based periodic series | No code reads them | `epoch_completed_*` (±1 epoch); metrics in artifacts | 205 MB | 205 MB | low | DELETE (log in `PRUNED_FILES.json`) | yes |
| R7 | E.1c `ignore_aware_val/` for 18 non-candidate checkpoints | Cached predictions | Task F code reads only best/120/130/140 | Metrics in `ignore_aware_val_22_checkpoints.json` | 184 MB | 184 MB | low | DELETE | yes |
| R8 | `experiments/task-e/benchmark-e1a-001/weights/` | 1-epoch benchmark weights | None | E.1a report | 21 MB | 21 MB | low | DELETE | yes |
| R9 | `experiments/task-e/smoke-test-001/weights/{last,epoch0}.pt` | E.0 weights | Test fixture is `best.pt` only | E.0 report | 15 MB | 15 MB | low | DELETE (keep `best.pt`) | yes |
| R10 | `dataset/processed/curated-dataset-v1/` | Superseded by v1.1 | `preflight.py` fingerprint, `dataset_cleaning/config.py` constant (need edits) | Hashes and counts in `VERSIONS.json` | 236 MB | 236 MB | low | DELETE after code update | yes |
| R11 | `dataset/processed/yolo-rock-v1/` | Superseded by v1.1 | `ai_review.py`, `yolo_dataset --link-from` default, `preflight.py`, `.ps1` (need edits) | v1.1 shares the same image files | 4.72 GiB | **0.02 GiB** | low | DELETE after code update (tidiness, not space) | yes |
| R12 | Three `experiments/task-e/*/data_view/` | Hard-link training views | Excluded from E.1c checksums; preflight fingerprints of E.0/E.1a change (expected) | `yolo-rock-v1.1` | 3.84 GiB | **0.006 GiB** | low | DELETE (tidiness) | yes |
| R13 | E.1c `epoch_completed_{010..110,150..190}.pt` (16) | Non-candidate checkpoints | Back the 22-checkpoint figure (results in artifacts) | Metrics in artifacts | 164 MB | 164 MB | low | **KEEP (recommended)** | yes |
| R14 | `dataset/processed/yolo-rock-v1.1/config_B/` (+ R11) | Untrained arm | Needed for the planned A/B comparison | Regenerable by Task D exporter | 3.03 GiB | 3.03 GiB (only with R11) | medium | **KEEP (recommended)** | yes |
| R15 | `backend/training/notebooks/*.ipynb` (2) | Never-executed Task E prep notebooks | None | `notebooks/05-07` | <1 MB | ~0 | low | ARCHIVE or DELETE | yes |
| R16 | `backend/main.py` | uv template stub | None | — | <1 KB | ~0 | none | DELETE | yes |
| R17 | `audit-artemis.ps1`, `collect_taskd_review_images.ps1` | One-off helpers (git-ignored) | Produced Task A audit and S2 zip | — | 18 KB | ~0 | low | MOVE to `scripts/legacy/` or KEEP | yes |
| R18 | `frontend/node_modules/` | pnpm install output | Hard-linked from the pnpm store | `pnpm install` | 0.43 GiB | **~0** | none | KEEP (no gain). `frontend/` is now git-ignored and kept locally | — |

## PROTECTED: never delete

| Path | Reason |
|---|---|
| `experiments/task-f/` | Frozen model, frozen config, selection evidence, **once-only test results and `TEST_RUN_LOCK.json`** |
| `models/`, `artifacts/`, `data/manifests/` | Published, hash-verified evidence |
| E.1c `best.pt`, `last.pt`, `epoch_completed_{120,130,140,200}.pt`, reports, `results.csv`, telemetry, `CHECKSUMS.sha256` | Model provenance and alternatives |
| `dataset/artificial-lunar/images`, `real_moon_images`, CSVs; `dataset/nasa-polar/Terrain*/`; `dataset/polar-sim-source/` | Source data (no verified external backup) |
| `dataset/processed/VERSIONS.json`, `curated-dataset-v1.1/`, `yolo-rock-v1.1/` | Canonical datasets and registry |
| `experiments/{dataset-exploration,dataset-audit,annotation-validation,dataset-cleaning,dataset-cleaning-v1.1,yolo-dataset (minus cache),yolo-dataset-v1.1,remediation}` | Task A–D.2 evidence; partly read by code and tests |
| `experiments/task-e/runs_registry.jsonl`, E.0 `best.pt` | Provenance; test fixture |
| `backend/` code, `uv.lock`, `.venv`, `yolo11n.pt` | Working pipeline and environment |

## Totals

| Approval set | Frees |
|---|---|
| SAFE only | ≈ 0.08 GB |
| SAFE + R1 (git garbage) | ≈ 31.5 GB |
| SAFE + R1 + R4–R12 | ≈ 33.4 GB |
| + R2/R3 (archives, only with an external backup) | ≈ **41.4 GB** (≈ 38.6 GiB) |
| + R13 and R14 (not recommended) | ≈ 44.7 GB |

*R1 doubled from 14.69 to 29.27 GiB during this session: GitKraken's maintenance packed the dangling blobs at 23:40 without
removing the loose copies. See `STORAGE_ANALYSIS.md` §3.*

**Hard-link note:** R11 and R12 list 8.6 GiB of apparent size but free only about 0.03 GiB. Their files are the same NTFS records
as `yolo-rock-v1.1`. The post-cleanup report will compare actual free space with these predictions.
