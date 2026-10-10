"""Reproduce the historical E.1c training run with the existing guarded trainer (``backend/training/train.py``).

Nothing here trains unless ``launch()`` receives BOTH ``run=True`` and the exact authorisation phrase. Even then the
trainer itself still requires ``ARTEMIS_TRAINING_AUTHORIZED=1`` + ``--authorized`` (set only for that one subprocess).

Safety rules enforced by ``plan_run``:
* outputs go to ``experiments/reproductions/<name>/`` - never into ``experiments/task-e``, ``experiments/task-f`` or ``models/``
* an existing run folder with checkpoints is never reused (``--fresh``)
* reproductions append to their own registry, so the historical ``runs_registry.jsonl`` is never modified
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import paths
from ..datasets import release
from .model_registry import sha256_file

AUTH_PHRASE = "I authorise one 200-epoch YOLO11n training run"
PRETRAINED = paths.BACKEND / "yolo11n.pt"
PRETRAINED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"   # official Ultralytics v8.4.0 asset
REPRO_ROOT = paths.EXPERIMENTS / "reproductions"
REPRO_REGISTRY = REPRO_ROOT / "reproductions_registry.jsonl"
PROTECTED_DIRS = [paths.EXPERIMENTS / "task-e", paths.EXPERIMENTS / "task-f", paths.MODELS, paths.ARTIFACTS]

# The exact overrides used for E.1c (artifacts/training/e1c_train_config.json -> overrides)
E1C_ARM, E1C_SEED = "A_steps", 42
E1C_OVERRIDES = {"epochs": 200, "save_period": 10, "imgsz": 736, "batch": 16, "workers": 4, "device": 0, "amp": True, "cache": False,
                 "deterministic": True, "optimizer": "SGD", "pretrained": True, "single_cls": False, "plots": True}


@dataclass(frozen=True)
class RunPlan:
    name: str
    run_dir: Path
    data_yaml: Path
    overrides: dict
    registry: Path

    def cli(self, authorised: bool) -> list[str]:
        sets = [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in self.overrides.items()]
        sets.append(f"data={self.data_yaml.as_posix()}")
        cmd = [sys.executable, "-m", "training.train", "--arm", E1C_ARM, "--seed", str(E1C_SEED), "--run-dir", str(self.run_dir),
               "--fresh", "--data-view", "--registry", str(self.registry), "--set", *sets]
        return cmd + (["--authorized"] if authorised else ["--dry-run"])


def historical_config() -> dict:
    """E.1c effective configuration from the published (sanitised) artifacts."""
    cfg = json.loads((paths.ARTIFACTS / "training" / "e1c_train_config.json").read_text(encoding="utf-8"))
    args = yaml.safe_load((paths.ARTIFACTS / "training" / "e1c_args.yaml").read_text(encoding="utf-8"))
    return {"arm": cfg["arm"], "seed": cfg["seed"], "overrides": cfg["overrides"], "effective_args": args,
            "status": cfg["status"], "epochs_completed": cfg.get("epochs_completed"), "weights_sha256": cfg["weights_sha256"]}


def check_pretrained() -> dict:
    ok = PRETRAINED.exists() and sha256_file(PRETRAINED) == PRETRAINED_SHA256
    return {"path": paths.rel(PRETRAINED), "present": PRETRAINED.exists(), "sha256_ok": ok,
            "hint": None if ok else "Download the official yolo11n.pt (Ultralytics assets v8.4.0) into backend/ and re-check its SHA-256."}


def check_environment() -> dict:
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), enough_vram=p.total_memory >= 7.5 * 2**30)
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
        info["ultralytics_matches_e1c"] = ultralytics.__version__ == "8.4.174"
    except ImportError:
        info["ultralytics"] = None
    return info


def plan_run(dataset_root: Path | None = None, name: str | None = None) -> RunPlan:
    root = Path(dataset_root) if dataset_root else paths.YOLO_DATASET
    name = name or time.strftime("repro-config-a-200-s42-%Y%m%d-%H%M%S")
    run_dir = (REPRO_ROOT / name).resolve()
    for d in PROTECTED_DIRS:
        if d.resolve() in run_dir.parents or run_dir == d.resolve():
            raise PermissionError(f"refusing to write a reproduction into protected {paths.rel(d)}")
    if (run_dir / "weights").exists():
        raise FileExistsError(f"{paths.rel(run_dir)} already contains checkpoints; choose a new name")
    data_yaml = release.write_local_data_yaml(root, out=run_dir.parent / f"{name}.data_A.yaml")
    return RunPlan(name, run_dir, data_yaml, dict(E1C_OVERRIDES), REPRO_REGISTRY)


def preflight(dataset_root: Path | None = None, full_hash: bool = False) -> dict:
    root = Path(dataset_root) if dataset_root else paths.YOLO_DATASET
    ds = release.verify_dataset(root, full=full_hash) if root.exists() else {"ok": False, "missing": ["dataset folder"], "n_missing": 1}
    pre = check_pretrained()
    env = check_environment()
    return {"dataset": ds, "pretrained": pre, "environment": env,
            "ready": bool(ds.get("ok") and pre["sha256_ok"] and env.get("cuda") and env.get("ultralytics"))}


def dry_run(plan: RunPlan) -> dict:
    """Resolve the exact Ultralytics arguments through the real trainer without touching the GPU."""
    out = subprocess.run(plan.cli(authorised=False), cwd=paths.BACKEND, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    meta = json.loads(out.stdout)
    return {"run_dir": paths.rel(Path(meta["run_dir"])), "dataset_status": meta["dataset_status"], "ultralytics_args":
            {k: (paths.rel(Path(v)) if k in ("data", "project") else v) for k, v in meta["ultralytics_args"].items()}}


def launch(plan: RunPlan, run: bool = False, authorisation: str = "") -> int | None:
    """Start ONE training run in a subprocess. Returns None (and does nothing) unless both safeguards are satisfied."""
    if not run or authorisation.strip() != AUTH_PHRASE:
        print("Training NOT started. Set RUN_TRAINING = True and AUTHORISATION = the exact phrase to launch one run.")
        return None
    plan.run_dir.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ARTEMIS_TRAINING_AUTHORIZED="1", PYTHONUNBUFFERED="1")
    log = plan.run_dir.parent / f"{plan.name}.console.log"
    print(f"Launching {plan.name}; console log: {paths.rel(log)}")
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run(plan.cli(authorised=True), cwd=paths.BACKEND, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
