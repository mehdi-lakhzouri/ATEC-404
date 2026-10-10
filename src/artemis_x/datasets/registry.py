"""Dataset version registry and split contracts (read-only).

The datasets themselves were built by ``backend/analysis/{dataset_cleaning,yolo_dataset}`` (Tasks C and D) and are
never regenerated here. This module reads ``VERSIONS.json`` and the published split manifest, and can verify a local
dataset copy against the registered content hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..config import paths

CANONICAL = {"curated": "curated-dataset-v1.1", "yolo": "yolo-rock-v1.1"}
EXPECTED_SPLIT = {"train": 7353, "val": 923, "test": 917}
EXPECTED_CONFIG_A_TRAIN = 1903


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def versions(path: Path | None = None) -> dict:
    local = paths.PROCESSED / "VERSIONS.json"
    p = path or (local if local.exists() else paths.ARTIFACTS / "provenance" / "VERSIONS.json")
    return json.loads(p.read_text(encoding="utf-8"))


def split_manifest(path: Path | None = None) -> list[dict]:
    p = path or (paths.ROOT / "data" / "manifests" / "split_manifest.csv")
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_counts(rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else split_manifest()
    c = Counter(r["split"] for r in rows)
    a = sum(1 for r in rows if r["in_config_A_train"] == "True")
    groups: dict[str, set] = {}
    for r in rows:
        groups.setdefault(r["leakage_group"], set()).add(r["split"])
    return {"splits": dict(c), "config_A_train": a, "groups": len(groups),
            "groups_spanning_splits": sum(1 for s in groups.values() if len(s) > 1)}


def verify_local_yolo_dataset() -> dict:
    """Compare the local yolo-rock-v1.1 files that the registry pins (manifests + data YAMLs) with their hashes."""
    reg = versions()["versions"][CANONICAL["yolo"]]["files"]
    out = {}
    for rel, sha in reg.items():
        p = paths.YOLO_DATASET / rel
        out[rel] = (p.exists() and _sha(p) == sha) if p.exists() else None
    return out
