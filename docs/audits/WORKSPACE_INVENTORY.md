# Workspace Inventory (Phase 1 audit, read-only)

| Item | Value |
|---|---|
| Audited | 2026-10-09, workspace `hackathon/` |
| Tools | `scripts/audit_workspace.py` (raw scan: `raw/storage_scan.json`), `scripts/retention_manifest.py` (→ `RETENTION_MANIFEST.json`) |
| Changes | **Nothing was moved, modified or deleted.** |

## 1. Top level

| Path | What it is | Physical size | Notes |
|---|---|---|---|
| `.git/` | Git repo: branch `master`, **no commits, no index**; remote `origin` = github.com/mehdi-lakhzouri/ATEC (added externally at 23:28) | 14.69 GiB at audit, **29.27 GiB** after GitKraken maintenance at 23:40 | 47,954 dangling blobs from a discarded `git add`, now both loose and packed. See STORAGE_ANALYSIS §3 |
| `backend/` | Python project (uv, `pyproject.toml` + `uv.lock`, Python 3.12) | 3.34 GiB (3.33 is `.venv`) | All ML code; package names `analysis`, `training`, `tests` |
| `dataset/` | Raw sources, archives, processed versions | 25.9 GiB | See §3 |
| `experiments/` | Reports and evidence for Tasks A–F | 5.8 GiB apparent | Mostly hard-link data views; see §4 |
| `frontend/` | **Unmodified** Turborepo / Next.js starter (`apps/web`, `apps/docs`, `packages/ui`) | 0.42 GiB | Placeholder for the future Mission Control. Its `node_modules` is hard-linked from the pnpm store |
| `polar-sim-source/` | Second clone of github.com/uwsbel/POLAR-Sim | 0.71 GiB | Same commit (`593ed65`) as `dataset/polar-sim-source`; not used by code. Since 23:29 it is the only local clone with `.git`: `dataset/polar-sim-source/.git` was removed externally and its label files are intact |
| `weights/yolo26n.pt` | Auto-download by the Ultralytics AMP self-check | 5 MB | Not a project model |
| `artemis-comparisons.zip` | Share package | 52 MB | 117/117 members CRC-identical to `experiments/annotation-validation/comparison_sheets/scenes/` |
| `taskd-review-images.zip` | Share package | 21 MB | 36 source renders + v1 Config B images (from `collect_taskd_review_images.ps1`) |
| `audit-artemis.ps1` | Task A environment / dataset auditor (PowerShell) | 16 KB | Produced `experiments/dataset-audit/` |
| `collect_taskd_review_images.ps1` | Builds `taskd-review-images.zip` | 1.5 KB | Hard-codes `yolo-rock-v1` |
| `scripts/` | **New, Phase 1:** audit tools | — | Read-only |
| `docs/audits/` | **New, Phase 1:** these audit documents | — | |

## 2. Python code (`backend/`)

| Package / module | Task | Purpose |
|---|---|---|
| `analysis/dataset_exploration/` | A | Inventory, statistics, quality issues of the three sources |
| `analysis/annotation_validation/` | B | S1 (CSV) vs S2 (mask) strategy comparison, overlays, galleries |
| `analysis/dataset_cleaning/` | C | S2-H policy, VALID / REVIEW_REQUIRED / REJECTED, curated v1 → v1.1 (`config.py` holds the version names) |
| `analysis/yolo_dataset/` | D | Leakage-free grouped split (ORB near-duplicates), YOLO export with hard links, Config A/B, review gallery |
| `analysis/remediation/` | D.2 | AI-review integration, scene investigation, threshold study, Config B masking audit |
| `analysis/evaluation/` | D.2 | `ignore_eval.py` (ignore-aware AP, COCO-crowd semantics), `build_ignore.py` |
| `training/train.py`, `monitor.py`, `preflight.py`, `env_check.py` | E | Guarded trainer (authorisation, limits, telemetry), integrity checks |
| `training/evaluate.py`, `smoke_report_data.py`, `e1c_analysis.py` | E | Ignore-aware checkpoint evaluation, run analysis |
| `training/task_f_selection.py`, `task_f_test_eval.py` | F | Validation-only selection and freezing; once-only locked test evaluation |
| `training/configs/task_e_experiments.yaml` | E | Hyperparameters, arms, seeds |
| `training/notebooks/*.ipynb` | E prep | 2 notebooks, **never executed** (no outputs); superseded |
| `tests/` | — | 52 unittest regression tests (`test_task_c_v11`, `test_ignore_eval`, `test_review_provenance`, `test_training_guard`); all pass |
| `main.py`, `README.md` | — | uv template stubs (empty README) |

`pyproject.toml` pins torch 2.14.1 / torchvision 0.29.1 (CUDA 13.0 index), ultralytics ≥ 8.4.174, numpy, pandas, opencv-python, pillow, matplotlib, pyyaml. There is no Jupyter / nbformat dependency yet.

## 3. Datasets (`dataset/`)

| Path | Version / identity | Physical size | Canonical? |
|---|---|---|---|
| `artificial-lunar/images/{render,ground,clean}` | Artificial Lunar Landscape renders (9,766 scenes, 720×480) | 5.02 GiB | Yes (source) |
| `artificial-lunar/archive.zip` | Original download | 5.02 GiB | No: **CRC-identical** to the extraction (29,378/29,378 members) |
| `artificial-lunar/real_moon_images/` | 36 real lunar photos (PCAM/TCAM) + masks + CSVs | 44 MB | Yes (source); origin not documented |
| `artificial-lunar/*.csv, *_IDs.txt` | Original boxes and ID lists | < 1 MB | Yes |
| `nasa-polar/Terrain{01,06,11}_*/` | NASA POLAR stereo subset (12-bit PNG) | 3.47 GiB | Yes (source) |
| `nasa-polar/*.zip` | Original downloads | 2.44 GiB | No: **CRC-identical** to the extractions (340/448/448) |
| `polar-sim-source/` | POLAR-Sim labels repo, CC0 (README still says BSD-3) | 0.60 GiB | Yes (clone; re-clonable) |
| `polar-sim/` | Empty directory | 0 | — |
| `processed/VERSIONS.json` | Registry with content hashes for every version | — | Yes |
| `processed/curated-dataset-v1` | S2-H/1.0: **ARCHIVED** | 0.24 GiB | No |
| `processed/curated-dataset-v1.1` | S2-H/1.1: **CURRENT** (VALID 261,239 / REVIEW 192,784 / REJECTED 50,005) | 0.24 GiB | Yes |
| `processed/yolo-rock-v1` | **ARCHIVED** export | 4.72 GiB apparent, **0.02 GiB reclaimable** | No (its images are the same NTFS files as v1.1) |
| `processed/yolo-rock-v1.1` | **CURRENT**: `config_A` (1,903 train), `config_B` (7,353 masked train), `images` (val/test), `labels`, `manifests` (split, ignore regions) | 4.72 GiB apparent | Yes |

Dataset status: **PROVISIONAL, NOT FROZEN**. Labels are automated and not human-verified.

## 4. Experiments (`experiments/`)

| Path | Task | Contents | Notes |
|---|---|---|---|
| `dataset-audit/` | A | Environment and dataset audit (PowerShell) | |
| `dataset-exploration/` | A | `DATASET_EXPLORATION_REPORT.md`, inventories, statistics | |
| `annotation-validation/` | B | Strategy comparison, overlays, galleries, review manifests | 158 MB of figures |
| `dataset-cleaning/`, `dataset-cleaning-v1.1/` | C, C v1.1 | Policy, cleaning decisions, quality reports | v1.1 reuses the v1 review sample |
| `yolo-dataset/`, `yolo-dataset-v1.1/` | D | Engineering, split and review reports; AI-review JSON; galleries | `yolo-dataset/cache/orb_features.npz` = 425 MB |
| `remediation/` | D.2 | AI review, scene investigation, threshold study, masking audit | |
| `task-e/smoke-test-001/` | E.0 | 3-epoch smoke test (class name `item`; regression fixture) | |
| `task-e/benchmark-e1a-001/` | E.1a | 1-epoch benchmark | |
| `task-e/full-training-config-a-200-s42/` | E.1c | 200-epoch run: 42 checkpoints, telemetry, cached val predictions, report | |
| `task-e/runs_registry.jsonl` | E | 3 registered runs (E.0, E.1a, E.1c) | |
| `task-f/model-selection-001/` | F | **Frozen model** `model/artemis_rock_yolo11n_e1c_s42.pt` (sha256 `413d0481…`), `FROZEN_MODEL_CONFIG.json`, selection analysis | |
| `task-f/test-evaluation-001/` | F | **Once-only test evaluation**, `TEST_RUN_LOCK.json`, results, figures | Must never be re-run |

All three `task-e/*/data_view/` folders are hard-link views: 1.28 GiB apparent each, **≈ 0 physical**.

## 5. Checkpoints

| File(s) | Count | Size each | State | Role |
|---|---|---|---|---|
| E.1c `weights/best.pt` | 1 | 5.5 MB | stripped | Epoch 111; source of the frozen model (byte-identical) |
| E.1c `weights/last.pt` | 1 | 5.5 MB | stripped | Epoch 200 |
| E.1c `weights/epoch_completed_{010…200}.pt` | 20 | 10.8 MB | EMA + optimiser + scaler | Completed-epoch copies |
| E.1c `weights/epoch{0…190}.pt` | 20 | 10.8 MB | EMA + optimiser + scaler | Ultralytics 0-based names: `epochN.pt` = **N+1** completed epochs. Not duplicates of the aligned series |
| `task-f/model-selection-001/model/artemis_rock_yolo11n_e1c_s42.pt` | 1 | 5.5 MB | stripped | **Frozen deliverable** |
| E.0 / E.1a weights | 3 + 3 | 5–11 MB | mixed | E.0 `best.pt` is a test fixture |
| `backend/yolo11n.pt` | 1 | 5.6 MB | pretrained | Official v8.4.0 asset, sha256 `0ebbc80d…` |
| `weights/yolo26n.pt` | 1 | 5.4 MB | pretrained | Accidental AMP-check download |

## 6. Other

- **Git:**
  - No commits.
  - `backend/.gitignore` covers only Python caches and `.venv`.
  - `frontend/.gitignore` is the starter's.
  - There is no root `.gitignore`.
- **Licences found on disk:** POLAR-Sim `LICENSE` = CC0 1.0. **No licence file or statement for the Artificial Lunar Landscape dataset or the NASA POLAR subset was found in the workspace**, and no project licence exists.
- **Junctions / symlinks:** only inside `frontend/node_modules` (pnpm), 200 entries recorded and not followed.
- **Temporary files:** `__pycache__/` folders; scratch outputs live outside the workspace.
