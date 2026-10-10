"""Verify that the frozen model, the once-only test evidence and all published artifacts are intact.

    uv run --project backend --no-sync python scripts/verify_artifacts.py              # verify (exit 1 on any failure)
    uv run --project backend --no-sync python scripts/verify_artifacts.py --baseline   # record hashes of protected files

Checks performed (each is skipped, and reported as such, when the local-only experiments/ folder is absent):
  1. models/artemis_rock_yolo11n.pt and the historical frozen copy match the frozen SHA-256
  2. every file listed in experiments/task-f/*/CHECKSUMS.sha256 still matches
  3. the test-run lock and the protected files match the recorded baseline
  4. every published artifact matches artifacts/provenance/ARTIFACT_MANIFEST.json (and its source, when present)
  5. the run registry still lists exactly the three historical runs (no new training)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/audits/raw/baseline_protected.json"
PROTECTED = [
    "experiments/task-f/model-selection-001/model/artemis_rock_yolo11n_e1c_s42.pt",
    "experiments/task-f/model-selection-001/FROZEN_MODEL_CONFIG.json",
    "experiments/task-f/model-selection-001/CHECKSUMS.sha256",
    "experiments/task-f/test-evaluation-001/TEST_RUN_LOCK.json",
    "experiments/task-f/test-evaluation-001/test_results.json",
    "experiments/task-f/test-evaluation-001/CHECKSUMS.sha256",
    "experiments/task-e/full-training-config-a-200-s42/weights/best.pt",
    "experiments/task-e/full-training-config-a-200-s42/results.csv",
    "experiments/task-e/full-training-config-a-200-s42/CHECKSUMS.sha256",
    "experiments/task-e/runs_registry.jsonl",
    "dataset/processed/VERSIONS.json",
    "dataset/processed/yolo-rock-v1.1/manifests/split_manifest.csv",
    "dataset/processed/yolo-rock-v1.1/manifests/eval_ignore_regions.json",
    "dataset/processed/yolo-rock-v1.1/data_A.yaml",
    "backend/uv.lock",
    "backend/pyproject.toml",
    "backend/yolo11n.pt",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checksums_file(cs: Path) -> tuple[int, list[str]]:
    bad, n = [], 0
    for ln in cs.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        digest, name = ln.split(maxsplit=1)
        f = cs.parent / name.lstrip("*").lstrip("./")
        n += 1
        if not f.exists() or sha(f) != digest:
            bad.append(f.relative_to(ROOT).as_posix() if f.exists() else f"missing: {name}")
    return n, bad


def verify() -> dict:
    res: dict[str, dict] = {}
    cfg_p = ROOT / "artifacts/provenance/FROZEN_MODEL_CONFIG.json"
    frozen = json.loads(cfg_p.read_text(encoding="utf-8"))["model"]["sha256"]
    for p in (ROOT / "models/artemis_rock_yolo11n.pt", ROOT / "experiments/task-f/model-selection-001/model/artemis_rock_yolo11n_e1c_s42.pt"):
        res[f"frozen_model:{p.relative_to(ROOT).as_posix()}"] = {"ok": sha(p) == frozen} if p.exists() else {"ok": None, "note": "absent (local-only)"}
    for cs in sorted((ROOT / "experiments/task-f").glob("*/CHECKSUMS.sha256")):
        n, bad = check_checksums_file(cs)
        res[f"checksums:{cs.relative_to(ROOT).as_posix()}"] = {"ok": not bad, "files": n, "mismatches": bad[:10]}
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
        changed = [k for k, v in base.items() if (ROOT / k).exists() and sha(ROOT / k) != v]
        absent = [k for k in base if not (ROOT / k).exists()]
        res["baseline_protected_files"] = {"ok": not changed, "checked": len(base) - len(absent), "changed": changed, "absent_local_only": absent}
    man = json.loads((ROOT / "artifacts/provenance/ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    bad_pub, bad_src = [], []
    for f in man["files"]:
        p = ROOT / f["path"]
        if not p.exists() or sha(p) != f["sha256"]:
            bad_pub.append(f["path"])
        s = ROOT / f["source"]
        if s.exists() and sha(s) != f["source_sha256"]:
            bad_src.append(f["source"])
    res["published_artifacts"] = {"ok": not bad_pub and not bad_src, "files": len(man["files"]), "published_mismatch": bad_pub,
                                  "source_changed_since_export": bad_src}
    reg = ROOT / "experiments/task-e/runs_registry.jsonl"
    if reg.exists():
        runs = [json.loads(l)["run"] for l in reg.read_text(encoding="utf-8").splitlines() if l.strip()]
        res["no_new_training_runs"] = {"ok": runs == ["smoke-test-001", "benchmark-e1a-001", "full-training-config-a-200-s42"], "runs": runs}
    lock = ROOT / "experiments/task-f/test-evaluation-001/TEST_RUN_LOCK.json"
    if lock.exists():
        res["test_lock_matches_published_copy"] = {"ok": sha(lock) == sha(ROOT / "artifacts/provenance/TEST_RUN_LOCK.json")}
    return res


def baseline() -> dict:
    files = {k: sha(ROOT / k) for k in PROTECTED if (ROOT / k).exists()}
    out = {"recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files}
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    if a.baseline:
        b = baseline(); print(f"baseline recorded for {len(b['files'])} protected files"); sys.exit(0)
    r = verify()
    for k, v in r.items():
        print(f"{'PASS' if v['ok'] else ('SKIP' if v['ok'] is None else 'FAIL'):4}  {k}  {json.dumps({x: y for x, y in v.items() if x != 'ok'})[:160]}")
    sys.exit(0 if all(v["ok"] is not False for v in r.values()) else 1)
