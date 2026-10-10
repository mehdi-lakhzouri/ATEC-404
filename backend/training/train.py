"""Guarded YOLO11n training launcher for Task E (prepared in D.2, NOT executed).

Refuses to train unless BOTH the environment variable ARTEMIS_TRAINING_AUTHORIZED=1 and the --authorized flag are set.
`--dry-run` resolves and prints the exact Ultralytics arguments and run metadata without touching the GPU.

Usage (from backend/):
    uv run --no-sync python -m training.train --arm A_steps --seed 0 --dry-run
    # after explicit human authorisation only:
    $env:ARTEMIS_TRAINING_AUTHORIZED=1; uv run --no-sync python -m training.train --arm A_steps --seed 0 --authorized
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = Path(__file__).resolve().parent / "configs/task_e_experiments.yaml"
RUNS = ROOT / "experiments/task-e/runs"
REGISTRY = ROOT / "experiments/task-e/runs_registry.jsonl"


class NotAuthorized(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def _parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """`key=value` pairs; values parsed as YAML scalars (3 -> int, true -> bool, 0.5 -> float)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not _:
            raise ValueError(f"bad override {it!r}, expected key=value")
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve(arm_id: str, seed: int, overrides: dict[str, Any] | None = None, run_dir: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    arm = next(a for a in cfg["arms"] if a["id"] == arm_id)
    ds = ROOT / cfg["dataset"]["root"]
    epochs = cfg["budgets"][arm["budget"]]["epochs"][arm["config"]]
    run_name = run_dir.name if run_dir else f"{arm_id}_s{seed}"
    project = run_dir.parent if run_dir else RUNS
    args = dict(cfg["common_hyperparameters"], data=str(ds / cfg["dataset"]["data_yaml"][arm["config"]]), epochs=epochs,
                seed=seed, project=str(project), name=run_name, exist_ok=True)
    args.update(overrides or {})
    status = json.loads((ds / "manifests/dataset_status.json").read_text(encoding="utf-8"))
    meta = {"run": run_name, "arm": arm, "seed": seed, "dataset": cfg["dataset"]["name"], "dataset_status": status,
            "manifest_sha256": _sha(ds / cfg["dataset"]["manifest"]), "data_yaml_sha256": _sha(Path(args["data"])),
            "weights": cfg["model"]["weights"], "weights_sha256": _sha(ROOT / "backend" / cfg["model"]["weights"]),
            "experiment_config_sha256": _sha(CFG), "ultralytics_args": args,
            "overrides": overrides or {}, "run_dir": str(Path(project) / run_name)}
    return meta


def build_data_view(data_yaml: Path, view: Path) -> Path:
    """Hard-link view of the train/val split inside the run folder so that Ultralytics' label caches
    (labels/*.cache) are written there and never into the dataset. No image bytes are duplicated."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(cfg["path"])
    for split in ("train", "val"):
        img_src = base / cfg[split]
        lab_src = Path(str(img_src).replace("images", "labels"))
        for src_dir, kind in ((img_src, "images"), (lab_src, "labels")):
            dst = view / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.iterdir()):
                if f.suffix in (".png", ".txt") and not (dst / f.name).exists():
                    os.link(f, dst / f.name)
    out = view / "data_view.yaml"
    out.write_text(yaml.safe_dump({"path": view.as_posix(), "train": "images/train", "val": "images/val",
                                   "names": cfg["names"], "nc": cfg["nc"]}, sort_keys=False), encoding="utf-8")
    return out


def authorize(flag: bool) -> None:
    if not (flag and os.environ.get("ARTEMIS_TRAINING_AUTHORIZED") == "1"):
        raise NotAuthorized("Training is not authorised. Requires explicit human approval: set ARTEMIS_TRAINING_AUTHORIZED=1 "
                            "AND pass --authorized. Dataset yolo-rock-v1.1 is PROVISIONAL and not frozen.")


def _nvidia_smi_logger(out_csv: Path):
    """Start `nvidia-smi -l 1` logging (temperature, VRAM, util, power, clocks, throttle reasons) or return None."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    q = ("timestamp,temperature.gpu,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,clocks.sm,"
         "clocks_throttle_reasons.active,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,"
         "clocks_throttle_reasons.sw_power_cap")
    fh = out_csv.open("w", encoding="utf-8")
    return subprocess.Popen([exe, f"--query-gpu={q}", "--format=csv", "-l", "1"], stdout=fh, stderr=subprocess.STDOUT), fh


def loss_record(tloss: Any) -> tuple[dict[str, float] | None, bool]:
    """Normalise Ultralytics' running loss (dict name->tensor in 8.4, tensor/sequence in older versions).
    Returns ({name: value}, all_finite)."""
    import math
    if tloss is None:
        return None, True
    if isinstance(tloss, dict):
        vals = {str(k): float(v) for k, v in tloss.items()}
    else:
        seq = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
        seq = seq if isinstance(seq, list) else [seq]
        vals = {f"loss_{i}": float(v) for i, v in enumerate(seq)}
    return {k: round(v, 5) for k, v in vals.items()}, all(math.isfinite(v) for v in vals.values())


EXPECTED_NAMES = {0: "rock"}


class LimitViolation(RuntimeError):
    """Raised by the limit callbacks when Ultralytics would exceed the authorised run."""


def check_class_names(names: Any) -> None:
    """Reject any class mapping other than {0: 'rock'} (e.g. Ultralytics single_cls renames it to 'item')."""
    norm = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    if norm != EXPECTED_NAMES:
        raise ValueError(f"unexpected class names {norm}; expected {EXPECTED_NAMES}")


def check_effective_args(args: Any, data_names: Any, expected: dict[str, Any]) -> dict[str, Any]:
    """Compare the trainer's effective arguments with the authorised ones before the first batch."""
    eff = {k: getattr(args, k, None) for k in expected}
    bad = {k: (eff[k], v) for k, v in expected.items() if eff[k] != v}
    if bad:
        raise LimitViolation(f"effective Ultralytics arguments differ from the authorised ones: {bad}")
    check_class_names(data_names)
    return eff


def check_epoch_start(epoch_idx: int, n_started: int, authorised_epochs: int, batch_size: int, authorised_batch: int,
                      oom_retries: int = 0) -> None:
    """Limits at the start of each epoch: epochs run strictly in order, never beyond the authorised count, never
    repeated (Ultralytics silently redoes an epoch after a NaN by reloading last.pt), and never with an OOM-reduced batch."""
    if epoch_idx + 1 > authorised_epochs or n_started >= authorised_epochs:
        raise LimitViolation(f"epoch {epoch_idx + 1} would exceed the authorised {authorised_epochs} epoch(s)")
    if epoch_idx != n_started:
        raise LimitViolation(f"epoch {epoch_idx + 1} started after {n_started} epoch start(s): repeated epoch "
                             "(Ultralytics NaN recovery?) - stopping, no automatic retry")
    if batch_size != authorised_batch or oom_retries:
        raise LimitViolation(f"batch changed to {batch_size} / OOM retry {oom_retries}: stopping, no automatic retry")


def aligned_checkpoint(epoch_idx: int, period: int) -> str | None:
    """Ultralytics names periodic checkpoints by the 0-based epoch index (epoch10.pt = 11 completed epochs).
    Return the name of an extra copy aligned to completed epochs (10, 20, ...), or None."""
    done = epoch_idx + 1
    return f"epoch_completed_{done:03d}.pt" if period > 0 and done % period == 0 else None


def _write_json_atomic(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


def train(meta: dict[str, Any], fresh: bool = False, data_view: bool = False,
          registry: Path = REGISTRY) -> dict[str, Any]:  # pragma: no cover - GPU
    import shutil
    import psutil
    import torch
    from ultralytics import YOLO
    from .env_check import check
    from .monitor import ResourceMonitor
    run_dir = Path(meta["run_dir"])
    last = run_dir / "weights/last.pt"
    if fresh and last.exists():
        raise RuntimeError(f"--fresh requested but {last} exists; refusing to resume or overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta["env"] = check(compute=True)
    if meta["env"]["verdict"] != "PASS":
        raise RuntimeError(f"environment check failed: {meta['env']}")
    if data_view:
        meta["data_view"] = str(build_data_view(Path(meta["ultralytics_args"]["data"]), run_dir / "data_view"))
        meta["ultralytics_args"]["data"] = meta["data_view"]
    meta["resumed"] = (not fresh) and last.exists()
    meta["weights_sha256"] = _sha(ROOT / "backend" / meta["weights"])
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["status"] = "RUNNING"
    cfg_json = run_dir / "train_config.json"
    _write_json_atomic(cfg_json, meta)
    epochs: list[dict[str, Any]] = []
    meta["epochs"] = epochs
    meta["resource_warnings"] = []
    period = int(meta["ultralytics_args"].get("save_period", 0) or 0)
    timing_csv = run_dir / "batch_timing.csv"
    with timing_csv.open("w", newline="", encoding="utf-8") as fh:
        fh.write("epoch,batch,start_s,end_s,step_s,wait_before_s\n")
    ua = meta["ultralytics_args"]
    authorised = {"epochs": ua["epochs"], "batch": ua["batch"], "imgsz": ua["imgsz"], "workers": ua["workers"], "seed": ua["seed"],
                  "amp": ua["amp"], "single_cls": ua.get("single_cls", False), "deterministic": ua["deterministic"],
                  "cache": ua["cache"], "resume": False, "time": None, "optimizer": ua["optimizer"]}
    batch_t: list[tuple[float, float]] = []          # (start, end) wall-clock per training batch, current epoch
    t_ref = [None]                                   # perf_counter origin of the first batch of the run

    def flush_timing(epoch_no: int) -> None:
        if not batch_t:
            return
        t0 = t_ref[0]
        with timing_csv.open("a", newline="", encoding="utf-8") as fh:
            for i, (a, b) in enumerate(batch_t):
                fh.write(f"{epoch_no},{i},{a - t0:.5f},{b - t0:.5f},{b - a:.5f},{'' if i == 0 else f'{a - batch_t[i - 1][1]:.5f}'}\n")
        batch_t.clear()

    def on_train_start(tr):
        meta["effective_args"] = check_effective_args(tr.args, tr.data["names"], authorised)
        meta["effective_args_all"] = {k: str(v) for k, v in vars(tr.args).items()}
        _write_json_atomic(cfg_json, meta)

    def on_epoch_start(tr):
        # limits: in order, never repeated, never more epochs than authorised, never an OOM-reduced batch
        check_epoch_start(tr.epoch, len(epochs), authorised["epochs"], tr.batch_size, authorised["batch"], getattr(tr, "_oom_retries", 0))
        torch.cuda.reset_peak_memory_stats()
        epochs.append({"epoch": tr.epoch + 1, "t_start": time.time()})

    def on_batch_start(tr):
        now = time.perf_counter()
        if t_ref[0] is None:
            t_ref[0] = now
        batch_t.append((now, float("nan")))

    def on_batch_end(tr):
        s0, _ = batch_t[-1]
        batch_t[-1] = (s0, time.perf_counter())
        if tr.batch_size != authorised["batch"]:
            raise LimitViolation(f"batch size changed to {tr.batch_size} during training")

    def on_epoch_end(tr):
        e = epochs[-1]
        e["train_s"] = round(time.time() - e["t_start"], 2)
        e["tloss"], finite = loss_record(tr.tloss)
        e["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        e["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        e["n_batches"] = len(batch_t)
        e["train_batches_s"] = round(sum(b - a for a, b in batch_t), 2)
        e["lr"] = {k: float(v) for k, v in (tr.lr or {}).items()}
        flush_timing(e["epoch"])
        if not finite:
            raise FloatingPointError(f"non-finite training loss at epoch {e['epoch']}: {e['tloss']}")

    def on_fit_epoch_end(tr):
        # Ultralytics' final_eval also fires this callback (with epoch bumped by one) after validating best.pt
        if tr.epoch + 1 > len(epochs):
            meta["final_eval_best_pt"] = {k: float(v) for k, v in (tr.metrics or {}).items()}
            meta["final_eval_s"] = round(time.time() - epochs[-1]["t_end"], 2) if epochs and "t_end" in epochs[-1] else None
            _write_json_atomic(cfg_json, meta)
            return
        e = epochs[-1]
        e["t_end"] = time.time()
        e["epoch_total_s"] = round(e["t_end"] - e["t_start"], 2)   # train + val + save
        e["val_metrics"] = {k: float(v) for k, v in (tr.metrics or {}).items()}
        vm = psutil.virtual_memory()
        e["ram_available_gb"] = round(vm.available / 2**30, 2)
        if vm.available < 2 * 2**30:
            meta["resource_warnings"].append({"epoch": e["epoch"], "warning": f"low available RAM {e['ram_available_gb']} GB"})
        _write_json_atomic(cfg_json, meta)                          # crash-safe progress record

    def on_model_save(tr):
        name = aligned_checkpoint(tr.epoch, period)
        if name and tr.last.exists():                               # copy: last.pt is rewritten in place every epoch
            shutil.copyfile(tr.last, tr.wdir / name)
            epochs[-1]["aligned_checkpoint"] = name

    smi = _nvidia_smi_logger(run_dir / "nvidia_smi.csv")
    status = "COMPLETED"
    try:
        with ResourceMonitor(run_dir / "resources.csv", period_s=1.0):
            model = YOLO(str(last) if meta["resumed"] else meta["weights"])
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_start", on_epoch_start)
            model.add_callback("on_train_batch_start", on_batch_start)
            model.add_callback("on_train_batch_end", on_batch_end)
            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.add_callback("on_model_save", on_model_save)
            model.train(resume=True) if meta["resumed"] else model.train(**meta["ultralytics_args"])
    except torch.cuda.OutOfMemoryError as e:
        status = f"CUDA_OOM: {e}"            # stop safely; never retry automatically
    except FloatingPointError as e:
        status = f"NONFINITE_LOSS: {e}"
    except LimitViolation as e:
        status = f"STOPPED_LIMIT: {e}"
    except Exception as e:                 # anything else: record the failure honestly, then re-raise
        status = f"FAILED: {type(e).__name__}: {e}"
        raise
    except BaseException as e:             # Ctrl-C / SystemExit must never be recorded as COMPLETED
        status = f"INTERRUPTED: {type(e).__name__}"
        raise
    finally:
        if smi:
            smi[0].terminate(); smi[0].wait(timeout=10); smi[1].close()
        if batch_t and epochs:
            flush_timing(epochs[-1]["epoch"])                       # partial epoch of an interrupted run
        n_done = sum(1 for e in epochs if "epoch_total_s" in e)
        if status == "COMPLETED" and n_done != authorised["epochs"]:
            status = f"FAILED_EPOCH_COUNT: {n_done} of {authorised['epochs']} epochs completed"
        if status == "COMPLETED":
            for ck in ("best.pt", "last.pt"):
                f = run_dir / "weights" / ck
                if f.exists():
                    try:
                        check_class_names(YOLO(str(f)).names)
                    except ValueError as e:
                        status = f"FAILED_CLASS_NAMES: {ck}: {e}"
        meta["status"] = status
        meta["epochs_completed"] = n_done
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json_atomic(cfg_json, meta)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, default=str) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override Ultralytics arguments (e.g. epochs=3 batch=4)")
    ap.add_argument("--run-dir", type=Path, help="isolated run folder (default experiments/task-e/runs/<arm>_s<seed>)")
    ap.add_argument("--fresh", action="store_true", help="refuse to resume if the run folder already has last.pt")
    ap.add_argument("--data-view", action="store_true", help="train from a hard-link view inside the run folder (keeps caches out of the dataset)")
    ap.add_argument("--registry", type=Path, default=REGISTRY,
                    help="run registry to append to (reproductions use their own, so the historical registry stays unchanged)")
    a = ap.parse_args(argv)
    meta = resolve(a.arm, a.seed, _parse_overrides(a.set), a.run_dir.resolve() if a.run_dir else None)
    if a.dry_run:
        print(json.dumps(meta, indent=2, default=str))
        return 0
    try:
        authorize(a.authorized)
    except NotAuthorized as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    meta = train(meta, fresh=a.fresh, data_view=a.data_view, registry=a.registry.resolve())
    return 0 if meta.get("status") == "COMPLETED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
