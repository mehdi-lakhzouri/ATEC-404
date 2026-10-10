"""Build the prepared-dataset release manifest from the canonical local yolo-rock-v1.1 (read-only on the dataset).

Writes data/release/yolo-rock-v1.1-configA.files.sha256 and data/release/dataset_release.json. Before writing it proves
that the local dataset is the registered one: the full label and image tree hashes must equal VERSIONS.json.
No archive is created and nothing is uploaded; the release stays NOT PUBLISHED until the owner approves licensing and a host.

Usage (repo root):  uv run --project backend --no-sync python scripts/build_release_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.config import paths  # noqa: E402
from artemis_x.datasets import release  # noqa: E402


def tree_hash(d: Path, pattern: str) -> tuple[int, str]:
    """Same definition as VERSIONS.json: sha256 over sorted (relative path + file sha256)."""
    h = hashlib.sha256(); n = 0
    for p in sorted(d.rglob(pattern)):
        h.update(p.relative_to(d).as_posix().encode()); h.update(release.sha256_file(p).encode()); n += 1
    return n, h.hexdigest()


def main() -> dict:
    ds = paths.YOLO_DATASET
    reg = json.loads((paths.PROCESSED / "VERSIONS.json").read_text(encoding="utf-8"))["versions"]["yolo-rock-v1.1"]
    nl, lh = tree_hash(ds, "*.txt")
    ni, ih = tree_hash(ds, "*.png")
    proof = {"labels_tree": {"files": nl, "sha256": lh, "matches_registry": lh == reg["labels_tree"]["content_sha256"]},
             "images_tree": {"files": ni, "sha256": ih, "matches_registry": ih == reg["images_tree"]["content_sha256"]},
             "split_manifest_matches_registry": release.sha256_file(ds / "manifests/split_manifest.csv") == reg["files"]["manifests/split_manifest.csv"]}
    if not all([proof["labels_tree"]["matches_registry"], proof["images_tree"]["matches_registry"], proof["split_manifest_matches_registry"]]):
        raise SystemExit(f"local dataset is not the registered yolo-rock-v1.1: {proof}")
    rows = release.build_file_manifest(ds)
    release.RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{d}  {r}\n" for r, d, _ in rows)
    release.FILE_MANIFEST.write_text(f"# {release.RELEASE_NAME}: SHA-256 of every release file, relative to the dataset root.\n" + body,
                                     encoding="utf-8", newline="\n")
    by = lambda pre: sum(b for r, _, b in rows if r.startswith(pre))  # noqa: E731
    record = {
        "release": release.RELEASE_NAME, "dataset_version": "yolo-rock-v1.1", "configuration": "A (conservative, 1,903 training scenes)",
        "status": "NOT PUBLISHED - no archive built, no host configured",
        "blocked_by": ["Redistribution licence of the Artificial Lunar Landscape renders (the images are re-encoded renders) not confirmed",
                       "Owner approval of the hosting platform and of the dataset licence/terms text"],
        "contents": {"train (Config A) images+labels": 1903, "val images+labels": 923, "test images+labels": 917,
                     "manifests": [f for f in release.INCLUDE_FILES]},
        "excluded": ["original source datasets", "Config B (masked) training images", "curated CSV annotations", "experiment outputs"],
        "files": len(rows), "uncompressed_bytes": sum(b for _, _, b in rows),
        "uncompressed_bytes_by_part": {"train_config_A": by("config_A/"), "val": by("images/val") + by("labels/val"),
                                       "test": by("images/test") + by("labels/test"), "manifests": by("manifests/")},
        "expected_archive_bytes_note": "PNG data is already compressed: a ZIP (store or deflate) is about the uncompressed size",
        "file_manifest": "data/release/" + release.FILE_MANIFEST.name,
        "file_manifest_sha256": release.sha256_file(release.FILE_MANIFEST),
        "archives": [], "registry_proof": proof,
        "annotation_policy": "S2-H/1.1 (VALID boxes only; REVIEW regions provided as evaluation ignore regions)",
        "label_status": "PROVISIONAL - automated, not human-verified",
        "provenance": "Derived from the Artificial Lunar Landscape dataset (Kaggle) by backend/analysis/dataset_cleaning and yolo_dataset",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    release.RELEASE_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return record


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("files", "uncompressed_bytes", "uncompressed_bytes_by_part", "registry_proof", "file_manifest_sha256")}, indent=1))
