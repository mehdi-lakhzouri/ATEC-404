# Migration Report: Artemis-X research workspace → GitHub-ready repository

| | |
|---|---|
| Date | 2026-10-09 |
| Scope | Audit, modular package, notebooks, documentation, verification, cleanup dry run |
| Constraints | **No training, no dataset regeneration, no test re-evaluation, no deletion, no Git commit or push** |

## 1. Old vs new architecture

| Before | After (in place, non-destructive) |
|---|---|
| `backend/`: pipeline code, 52 tests, uv env; empty README | **unchanged** code and tests; README written |
| No public package | `src/artemis_x/`: `config.paths`, `ml.model_registry` (sha-verified frozen spec), `ml.inference.RockDetector` (CPU/GPU), `evaluation.ignore_aware` (re-export of the proven evaluator), `evaluation.metrics`, `reporting.artifacts`/`plots`, `datasets.registry` |
| 2 never-executed prep notebooks | `notebooks/01–08`: executed, pixel-free outputs, 21–653 KB each |
| Evidence only inside `experiments/` (local, 5.8 GB apparent) | `artifacts/` (71 files, 9.7 MB, hash-verified copies or sanitised derivatives), `data/manifests/`, `models/` |
| No docs | `README.md`, `docs/{ARCHITECTURE,DATASET,METHODOLOGY,RESULTS,REPRODUCIBILITY,LIMITATIONS,ROADMAP}.md`, `docs/audits/*`, `future/WEBOTS_INTEGRATION_PLAN.md`, `models/model_card.md`, `CITATION.cff` |
| No root `.gitignore` | `.gitignore` and `.gitattributes`: Git would now see **267 files, about 13.3 MB** (largest: model 5.5 MB, split manifest 2.0 MB) |
| No root tests | `tests/`: 20 tests (artifact integrity, model loading + tamper refusal + parity, metrics, dataset contracts, notebook safety) |
| — | `scripts/`: `verify_artifacts`, `demo_inference`, `inspect_environment`, `export_artifacts`, `build_notebooks`, `audit_workspace`, `retention_manifest` |

## 2. Verification (final pass)

| Gate | Result |
|---|---|
| Root tests | **20/20 OK** (with `--with nbformat`) |
| Backend regression tests | **52/52 OK** |
| Notebooks | 8/8 valid (nbformat), fully executed, 0 errors, no local paths in outputs, no forbidden calls (trainer, test evaluator, test images, dataset rebuild) |
| Frozen model | `models/artemis_rock_yolo11n.pt` and historical copy = `413d0481…0752eb` ✔ |
| Test lock and evidence | `TEST_RUN_LOCK.json` unchanged; 932/932 test-evaluation checksums ✔; 917 prediction files ✔ |
| Model-selection evidence | 30/30 checksums ✔ |
| Protected baseline | 17/17 files unchanged (`docs/audits/raw/baseline_protected.json`) |
| Published artifacts | 71/71 match the manifest; sources unchanged since export |
| No new training | Run registry = the 3 historical runs |
| Dependency lock | `backend/uv.lock` sha256 `e5fa9a11…` unchanged; notebook tooling layered with `uv run --with` (small download, no lock change) |
| Inference smoke test | GPU and CPU; parity with Task F cached predictions on 4 val scenes; the demo script refuses test-split paths |
| Personal paths | None in published text or notebooks. **Exception:** the frozen `.pt` embeds Ultralytics `train_args` with local absolute paths, which can't be removed without changing the frozen SHA-256 (see the model card) |

The new `RockDetector` entry point was proven equivalent to the historical protocol: live predictions equal the cached Task F
predictions filtered at conf 0.20 and 50 px². This is `tests/test_model_loading.py::test_parity_with_task_f_cached_predictions`.

## 3. Observed external changes during the session (not made by this migration)

| Time | Change |
|---|---|
| 23:28 | `.git/config` gained `remote origin = https://github.com/mehdi-lakhzouri/ATEC.git` |
| 23:28 | GitKraken maintenance ran (`.git/gk/config`) |
| 23:40 | A 15.65 GB `loose-*.pack` appeared, duplicating all 47,954 dangling blobs. `.git` is now **29.3 GiB** and free disk fell from 371 GB to 356 GB |
| 23:29 | `dataset/polar-sim-source/.git` was removed; label files intact |

These are recorded in `docs/audits/STORAGE_ANALYSIS.md` §3. **Before staging or committing**, make sure `.gitignore` (added in
this migration) is in place, or Git would try to add `dataset/` (about 17 GB).

## 4. Storage

| Measure | Value |
|---|---|
| Workspace physical now | **56.6 GiB** (hard links counted once; was 42.5 GiB at audit before the external Git maintenance) |
| Added by this migration | ≈ 25 MB |
| Reclaimable with recommended approvals (SAFE + R1 + R4–R12) | ≈ **33.4 GB** |
| Plus duplicate archives (only with an external backup) | ≈ **41.4 GB** |

Details: `docs/audits/CLEANUP_DRY_RUN.md`.

## 5. Waiting for your approval

1. **Cleanup** (nothing deleted yet), by category in `CLEANUP_DRY_RUN.md`:
   - **SAFE:** S1–S5.
   - **REVIEW:** R1 (git garbage, close GitKraken first), R2/R3 (archives, needs a backup), R4–R12.
   - **Kept unless you decide otherwise:** R13 and R14.
2. **Code edits before retiring the v1 datasets:** `preflight.py`, `dataset_cleaning/config.py`, `remediation/ai_review.py`,
   `yolo_dataset` `--link-from` default.
3. **Licences:** the project licence, publication of the AGPL-trained weights, and the dataset licences (Artificial Lunar
   Landscape, NASA POLAR).
4. **Model file privacy:** accept the embedded local paths, or approve publishing a re-saved model with a new SHA-256 (this would
   require a documented, verified re-freeze).
5. **`frontend/`:** keep the unmodified starter in the repository, or exclude it until Mission Control work begins.

## 6. Known scientific limitations (unchanged)

- Labels are automated and not human-verified.
- Results are synthetic-only.
- There is one seed.
- About 21% of large rocks are missed on test.
- Metrics depend on the evaluation policy.

See `docs/LIMITATIONS.md`.
