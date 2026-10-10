"""Pre-flight / post-flight integrity checks for a Task E run (read-only).

Usage (from backend/):
    uv run --no-sync python -m training.preflight --run-dir ../experiments/task-e/smoke-test-001 --stage pre
    uv run --no-sync python -m training.preflight --run-dir ../experiments/task-e/smoke-test-001 --stage post
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"
PROTECTED = {  # path+size+mtime fingerprints: anything written here during a run is detected
    "source_datasets": [ROOT / "dataset/artificial-lunar", ROOT / "dataset/nasa-polar", ROOT / "dataset/polar-sim-source"],
    "task_c_v1_1": [ROOT / "dataset/processed/curated-dataset-v1.1", ROOT / "experiments/dataset-cleaning-v1.1"],
    "task_d_v1_1": [DS, ROOT / "experiments/yolo-dataset-v1.1"],
    "archived_v1": [ROOT / "dataset/processed/curated-dataset-v1"],     # yolo-rock-v1 retired 2026-10-10 (hard links into v1.1)
    "previous_experiments": [ROOT / "experiments/task-e/smoke-test-001", ROOT / "experiments/task-e/benchmark-e1a-001"],
}


def gpu_compute_processes() -> list[str] | None:
    """Running CUDA compute processes per nvidia-smi (None if nvidia-smi is unavailable)."""
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = subprocess.run([exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree_content(d: Path, pattern: str) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob(pattern)):
        h.update(p.relative_to(d).as_posix().encode()); h.update(_sha(p).encode())
    return h.hexdigest()


def _tree_meta(dirs: list[Path]) -> dict[str, Any]:
    h = hashlib.sha256(); n = 0
    for d in dirs:
        for dp, dn, fn in os.walk(d):
            dn[:] = sorted(x for x in dn if x != ".git")
            for f in sorted(fn):
                p = Path(dp) / f; st = p.stat()
                h.update(f"{p.as_posix()}\t{st.st_size}\t{st.st_mtime_ns}\n".encode()); n += 1
    return {"files": n, "sha256": h.hexdigest()}


def gpu_load() -> tuple[float | None, float | None]:
    """(utilisation %, memory used MiB) from nvidia-smi, sampled 3 times 1 s apart (max taken)."""
    import subprocess
    import time
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    u = m = 0.0
    for _ in range(3):
        out = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().split(",")
        u, m = max(u, float(out[0])), max(m, float(out[1]))
        time.sleep(1)
    return u, m


def checks(run_dir: Path) -> dict[str, Any]:
    reg = json.loads((ROOT / "dataset/processed/VERSIONS.json").read_text(encoding="utf-8"))["versions"]["yolo-rock-v1.1"]
    data = yaml.safe_load((DS / "data_A.yaml").read_text(encoding="utf-8"))
    status = json.loads((DS / "manifests/dataset_status.json").read_text(encoding="utf-8"))
    r: dict[str, Any] = {"dataset": "yolo-rock-v1.1", "config": "A"}
    c = {}
    c["data_A_yaml_hash_matches_registry"] = _sha(DS / "data_A.yaml") == reg["files"]["data_A.yaml"]
    c["split_manifest_hash_matches_registry"] = _sha(DS / "manifests/split_manifest.csv") == reg["files"]["manifests/split_manifest.csv"]
    c["labels_tree_matches_registry"] = _tree_content(DS, "*.txt") == reg["labels_tree"]["content_sha256"]
    c["class_mapping_single_rock"] = data.get("names") == {0: "rock"} and data.get("nc") == 1
    for split, rel in (("train", data["train"]), ("val", data["val"])):
        imgs = {p.stem for p in (DS / rel).glob("*.png")}
        labs = {p.stem for p in Path(str(DS / rel).replace("images", "labels")).glob("*.txt")}
        r[f"{split}_images"] = len(imgs)
        c[f"{split}_images_have_labels"] = imgs == labs and len(imgs) > 0
    c["train_count_is_config_A_1903"] = r["train_images"] == 1903
    c["val_count_923"] = r["val_images"] == 923
    c["dataset_provisional_not_frozen"] = status["frozen"] is False and status["approved_for_training"] is False
    c["no_previous_checkpoint_in_run_dir"] = not (run_dir / "weights/last.pt").exists()
    w = ROOT / "backend/yolo11n.pt"
    r["weights"] = {"path": str(w), "sha256": _sha(w) if w.exists() else "MISSING", "bytes": w.stat().st_size if w.exists() else 0}
    c["weights_present"] = w.exists()
    c["weights_sha256_is_official_yolo11n"] = r["weights"]["sha256"] == "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    c["class_names_rock"] = data.get("names") == {0: "rock"}
    procs = gpu_compute_processes()
    r["gpu_compute_processes"] = procs
    # a concurrent *training/compute* job = another Python/CUDA process on the GPU, or a busy GPU. Desktop apps with
    # hardware acceleration (e.g. browsers, type C+G) are reported as a measurement-contamination warning instead.
    trainers = [p for p in (procs or []) if any(k in p.lower() for k in ("python", "torch", "yolo", "ultralytics", "cuda"))]
    util, used = gpu_load()
    r["gpu_idle_snapshot"] = {"util_pct": util, "mem_used_mib": used}
    r["warnings"] = [f"other GPU client (not a training job): {p}" for p in (procs or []) if p not in trainers]
    c["no_concurrent_gpu_job"] = procs is not None and not trainers and util is not None and util < 10 and used < 1024
    free = shutil.disk_usage(ROOT).free / 2**30
    r["disk_free_gb"] = round(free, 1)
    c["disk_free_ge_20gb"] = free >= 20
    import psutil
    vm = psutil.virtual_memory()
    r["ram"] = {"total_gb": round(vm.total / 2**30, 1), "available_gb": round(vm.available / 2**30, 1)}
    c["ram_available_ge_16gb"] = vm.available >= 16 * 2**30     # E.1a validation peaked at 25.6 of 32 GB used
    bat = psutil.sensors_battery()
    r["power"] = None if bat is None else {"plugged_in": bat.power_plugged, "battery_pct": bat.percent}
    c["on_ac_power"] = bat is None or bool(bat.power_plugged)
    r["checks"] = c
    r["fingerprints"] = {k: _tree_meta(v) for k, v in PROTECTED.items()}
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=("pre", "post"), required=True)
    a = ap.parse_args()
    run = a.run_dir.resolve(); run.mkdir(parents=True, exist_ok=True)
    res = checks(run)
    if a.stage == "post":
        pre = json.loads((run / "preflight_pre.json").read_text(encoding="utf-8"))
        res["unchanged_since_pre"] = {k: pre["fingerprints"][k] == res["fingerprints"][k] for k in PROTECTED}
        res["checks"].pop("no_previous_checkpoint_in_run_dir")   # expected to exist after training
        for gate in ("no_concurrent_gpu_job", "ram_available_ge_16gb", "on_ac_power"):
            res["checks"].pop(gate)                                 # pre-launch resource gates, not integrity checks
    (run / f"preflight_{a.stage}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ok = all(res["checks"].values()) and all(res.get("unchanged_since_pre", {}).values())
    print(json.dumps({"stage": a.stage, "all_pass": ok, **{k: v for k, v in res.items() if k != "fingerprints"}}, indent=1))
    raise SystemExit(0 if ok else 2)
