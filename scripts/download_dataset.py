"""Download (or locate) and verify the prepared Artemis-X dataset release `yolo-rock-v1.1-configA`.

STATUS: no public release exists yet (see data/release/dataset_release.json). This script therefore FAILS unless you
provide an authorised source. It never invents a URL.

Sources, in order of precedence:
    --source URL_OR_PATH                     https:// URL of the release ZIP, a local ZIP, or an already-extracted folder
    environment ARTEMIS_DATASET_SOURCE       same forms
    data/release/dataset_release.json        "archives": [{"url": ..., "sha256": ..., "bytes": ...}] once a release is approved

Verification (always): every file is checked against data/release/yolo-rock-v1.1-configA.files.sha256, which is
committed to Git and is the trust anchor. When the release record (or --archive-sha256) provides an archive SHA-256,
the ZIP is checked before extraction. A local data YAML is written to data/local/data_A.local.yaml (git-ignored).

Examples:
    uv run --project backend --no-sync python scripts/download_dataset.py --check-only          # verify the default location
    uv run --project backend --no-sync python scripts/download_dataset.py --source D:/yolo-rock-v1.1-configA.zip
    uv run --project backend --no-sync python scripts/download_dataset.py --source D:/datasets/yolo-rock-v1.1 --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.config import paths  # noqa: E402
from artemis_x.datasets import release  # noqa: E402

EXIT_NO_SOURCE, EXIT_VERIFY, EXIT_ARCHIVE = 2, 3, 4


def _configured_source(cli: str | None) -> tuple[str | None, str | None, str]:
    if cli:
        return cli, None, "--source"
    env = os.environ.get("ARTEMIS_DATASET_SOURCE")
    if env:
        return env, None, "ARTEMIS_DATASET_SOURCE"
    rec = release.release_record() if release.RELEASE_RECORD.exists() else {}
    for a in rec.get("archives", []):
        if a.get("url"):
            return a["url"], a.get("sha256"), "dataset_release.json"
    return None, None, ""


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> Path:
    if not url.lower().startswith("https://"):
        raise SystemExit(f"refusing non-HTTPS URL: {url}")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, dst.open("wb") as fh:      # noqa: S310 - https only, user-authorised source
        total = int(r.headers.get("Content-Length") or 0); done = 0
        for chunk in iter(lambda: r.read(1 << 22), b""):
            fh.write(chunk); done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="", flush=True)
    print()
    return dst


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if root not in target.parents and target != root:
            raise SystemExit(f"unsafe path in archive: {m.filename}")
    zf.extractall(dest)


def _locate_root(extracted: Path) -> Path:
    """Accept archives that contain the files directly or inside one top-level folder."""
    if (extracted / "config_A").is_dir():
        return extracted
    subs = [d for d in extracted.iterdir() if d.is_dir()]
    if len(subs) == 1 and (subs[0] / "config_A").is_dir():
        return subs[0]
    raise SystemExit("archive layout not recognised (expected config_A/, images/, labels/, manifests/)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="https URL of the release ZIP, local ZIP, or extracted folder")
    ap.add_argument("--dest", type=Path, default=paths.YOLO_DATASET, help="where to place an archive's contents")
    ap.add_argument("--archive-sha256", help="expected SHA-256 of the ZIP (overrides the release record)")
    ap.add_argument("--check-only", action="store_true", help="only verify the dataset at --dest")
    ap.add_argument("--quick", action="store_true", help="hash labels/manifests and every 50th image instead of all files")
    a = ap.parse_args(argv)
    manifest = release.read_file_manifest()

    if a.check_only:
        res = release.verify_dataset(a.dest, manifest, full=not a.quick)
        print(json.dumps(res, indent=1))
        if res["ok"]:
            print("local data YAML:", paths.rel(release.write_local_data_yaml(a.dest)))
        return 0 if res["ok"] else EXIT_VERIFY

    src, rec_sha, origin = _configured_source(a.source)
    if not src:
        print("ERROR: no authorised dataset source is configured.\n"
              "The prepared dataset release has NOT been published yet (data/release/dataset_release.json: status NOT PUBLISHED).\n"
              "Provide one with --source <https URL | local ZIP | extracted folder> or ARTEMIS_DATASET_SOURCE once the owner has\n"
              "approved a release. See docs/DATASET_DISTRIBUTION_PLAN.md.", file=sys.stderr)
        return EXIT_NO_SOURCE
    expected = a.archive_sha256 or rec_sha
    print(f"source ({origin}): {src if src.startswith('https://') else Path(src).name}")

    p = Path(src)
    if not src.startswith("https://") and p.is_dir():                 # already-extracted folder: verify in place, no copy
        res = release.verify_dataset(p, manifest, full=not a.quick)
        print(json.dumps(res, indent=1))
        if not res["ok"]:
            return EXIT_VERIFY
        print("local data YAML:", paths.rel(release.write_local_data_yaml(p)))
        return 0

    with tempfile.TemporaryDirectory(prefix="artemis_ds_") as tmp:
        archive = _download(src, Path(tmp) / "release.zip") if src.startswith("https://") else p
        if not archive.is_file():
            print(f"ERROR: source not found: {archive.name}", file=sys.stderr); return EXIT_NO_SOURCE
        if expected:
            got = _sha(archive)
            if got != expected:
                print(f"ERROR: archive SHA-256 {got} != expected {expected}", file=sys.stderr); return EXIT_ARCHIVE
            print("archive SHA-256 verified")
        else:
            print("note: no archive SHA-256 provided; relying on per-file verification against the committed manifest")
        staging = Path(tmp) / "x"
        with zipfile.ZipFile(archive) as zf:
            _safe_extract(zf, staging)
        root = _locate_root(staging)
        res = release.verify_dataset(root, manifest, full=True)
        if not res["ok"]:
            print(json.dumps(res, indent=1)); return EXIT_VERIFY
        if a.dest.exists() and any(a.dest.iterdir()):
            existing = release.verify_dataset(a.dest, manifest, full=False)
            if existing["ok"]:
                print(f"{paths.rel(a.dest)} already holds a verified copy; nothing replaced")
                print("local data YAML:", paths.rel(release.write_local_data_yaml(a.dest)))
                return 0
            print(f"ERROR: {paths.rel(a.dest)} exists and is not a verified copy; choose another --dest", file=sys.stderr)
            return EXIT_VERIFY
        a.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(a.dest))
    print(f"dataset verified ({res['files_in_manifest']} files) at {paths.rel(a.dest)}")
    print("local data YAML:", paths.rel(release.write_local_data_yaml(a.dest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
