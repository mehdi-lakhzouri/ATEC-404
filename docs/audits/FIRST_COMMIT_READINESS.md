# First-Commit Readiness

**Nothing has been staged, committed or pushed.** This listing is produced read-only with
`git ls-files --others --exclude-standard`. The full list is in `FIRST_COMMIT_FILES.txt`.

## Summary

| Item | Value |
|---|---|
| Files | **219** (after cleanup, 2026-10-10) |
| Total size | **13.8 MB** |
| Largest files | `models/artemis_rock_yolo11n.pt` 5.48 MB · `data/manifests/split_manifest.csv` 2.02 MB · `data/release/…files.sha256` 0.73 MB · `notebooks/05_…ipynb` 0.67 MB |
| Files over GitHub's 50 MB warning or 100 MB limit | none |
| Dataset images, experiment outputs, ZIPs, `.venv`, `__pycache__`, `frontend/`, `polar-sim-source/`, `weights/` | **0 files** |
| Absolute local paths in text files | none (`tests/test_artifact_integrity.py`). Exception: inside the binary `.pt`; see `docs/MODEL_PRIVACY_AND_LICENSING.md` |

## By folder

| Folder | Files | Content |
|---|---|---|
| root | 5 | README, MIGRATION_REPORT, CITATION.cff, .gitignore, .gitattributes |
| `src/artemis_x/` | 16 | Package |
| `backend/` | 72 | Research pipeline code, 52 tests, `pyproject.toml`, `uv.lock`, configs, 2 legacy prep notebooks |
| `notebooks/` | 9 | 01–09 |
| `artifacts/` | 68 | Metrics, figures, reports, provenance |
| `data/` | 6 | `README`, split manifest, 2 data YAMLs, release record + file manifest |
| `models/` | 2 | Frozen model + model card |
| `docs/` | 20 | Documentation and audits (the raw storage scans are excluded) |
| `scripts/` | 10 | Tools |
| `tests/` | 8 | 29 tests |
| `future/` | 1 | Webots plan |

## Open decisions before the first commit

1. **Remote:** `origin` → `mehdi-lakhzouri/ATEC.git` (private or not yet created; see `GIT_INTEGRITY_REPORT.md` §4).
2. **Model file:** publish the historical checkpoint (contains local paths), the sanitised copy, or no weights.
3. **Licence:** without a `LICENSE` file the repository is all-rights-reserved.
4. **Legacy notebooks:** `backend/training/notebooks/*.ipynb` (2, never executed) are superseded by notebooks 05–09. Keep them, or
   remove them (cleanup item R15).
5. **Clean `.git` first** (cleanup item R1), so that the first commit starts from a 29 GiB-free repository.
