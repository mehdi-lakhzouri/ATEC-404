# backend/: Artemis-X research pipeline (Tasks A–F)

This folder is the uv project (`pyproject.toml`, `uv.lock`, Python 3.12) and the historical, tested pipeline that produced the
dataset and the frozen model. The public package in `../src/artemis_x` wraps these modules; it does not replace them.

| Package | Task | Entry point (run from `backend/`) |
|---|---|---|
| `analysis/dataset_exploration` | A: data inventory and quality | `uv run --no-sync python -m analysis.dataset_exploration` |
| `analysis/annotation_validation` | B: S1 vs S2 strategy | `uv run --no-sync python -m analysis.annotation_validation` |
| `analysis/dataset_cleaning` | C: S2-H policy, curated dataset | `uv run --no-sync python -m analysis.dataset_cleaning` |
| `analysis/yolo_dataset` | D: grouped split, YOLO export | `uv run --no-sync python -m analysis.yolo_dataset --help` |
| `analysis/remediation`, `analysis/evaluation` | D.2: audits, ignore-aware evaluator | see module docstrings |
| `training/train.py` | E: guarded training | **requires** `ARTEMIS_TRAINING_AUTHORIZED=1` and `--authorized`; `--dry-run` is safe |
| `training/preflight.py`, `e1c_analysis.py` | E: integrity checks, run analysis | `--help` |
| `training/task_f_selection.py` | F: validation-only selection and freeze | `--stage analyse/freeze/inference` |
| `training/task_f_test_eval.py` | F: **once-only** test evaluation | locked; refuses a second run |

**Regression tests** (52):

```powershell
uv run --no-sync python -m unittest discover -s tests -t .
```

Commands that rebuild datasets, train, or evaluate the test split must not be run without explicit approval. See
`../docs/REPRODUCIBILITY.md`.
