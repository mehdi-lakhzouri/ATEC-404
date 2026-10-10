# Reproducibility

## What can be reproduced, and how

| Level | What | Needs | Command |
|---|---|---|---|
| 1. Read | Every reported number, figure and notebook | This repository only | Open `notebooks/` on GitHub, or `uv run --project backend --with jupyterlab jupyter lab` |
| 2. Verify | Frozen model and published-artifact hashes | This repository | `uv run --project backend python scripts/verify_artifacts.py` |
| 3. Infer | Run the frozen detector on your images | This repository; CPU or NVIDIA GPU | `uv run --project backend python scripts/demo_inference.py frame.png` |
| 4. Re-score | Ignore-aware validation metrics from saved predictions | Local `dataset/` and `experiments/` (not published) | `backend/training/task_f_selection.py --stage analyse` |
| 5. Rebuild | Curated / YOLO datasets from the sources | Source datasets; CPU only (duration not benchmarked) | Task C/D commands in `backend/analysis/*/__main__.py` |
| 6. Get the prepared dataset | `yolo-rock-v1.1-configA` (7,490 files, 1.82 GB), verified per file | An approved release (**not yet published**) | `scripts/download_dataset.py --source …`; `--check-only` verifies a local copy |
| 7. Retrain | YOLO11n E.1c (200 epochs, ≈ 70 min on an 8 GB laptop GPU) | Prepared dataset, official `yolo11n.pt`, NVIDIA GPU ≥ 8 GB, **explicit opt-in** | `notebooks/09_reproduce_training.ipynb`: preflight + dry run by default; it launches the guarded `backend/training/train.py` only with `RUN_TRAINING = True` and the authorisation phrase. Outputs go to `experiments/reproductions/`, with a separate run registry |

The held-out **test evaluation is not reproducible by design**: it is locked after its single run (`TEST_RUN_LOCK.json`).

## Environment (as used)

| Component | Version |
|---|---|
| OS | Windows 11 |
| Python | 3.12.13 (uv-managed; `backend/.python-version`) |
| PyTorch / torchvision | 2.14.1 + CUDA 13.0 / 0.29.1 (index `pytorch-cu130`) |
| Ultralytics | 8.4.174 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, sm_120, driver 616.64 |
| Lock file | `backend/uv.lock` (unchanged by the repository migration) |

```powershell
uv sync --project backend                                  # exact locked environment, CUDA 13.0 wheels (multi-GB)
uv run --project backend python scripts/inspect_environment.py
uv run --project backend python -m unittest discover -s tests -t .                      # 20 repository tests
cd backend; uv run --no-sync python -m unittest discover -s tests -t . ; cd ..          # 52 pipeline regression tests
```

**Notebook tooling** is layered on the locked environment without changing `uv.lock`:

```powershell
uv run --project backend --with jupyterlab jupyter lab
uv run --project backend --with nbformat --with nbclient --with ipykernel python scripts/build_notebooks.py   # regenerate notebooks
```

## CPU-only path (unverified)

The locked environment pins CUDA wheels. For a machine without an NVIDIA GPU, a separate CPU environment *should* work, but it
**has not been tested on a clean machine**:

```powershell
uv venv .venv-cpu --python 3.12
uv pip install --python .venv-cpu torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu ultralytics==8.4.174 pandas matplotlib pyyaml pillow jupyterlab
.venv-cpu\Scripts\python scripts\demo_inference.py frame.png --device cpu
```

## Determinism caveats

- Training used `deterministic=true` and seed 42, but GPU kernels are not guaranteed to be bitwise reproducible.
- Only one seed was trained.
- Inference is deterministic for a given device. Live inference reproduced the cached validation predictions exactly (1,904 of
  1,904 boxes; also covered by `tests/test_model_loading.py`).
