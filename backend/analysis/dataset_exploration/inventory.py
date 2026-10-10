"""A1 - read-only dataset inventory and source-integrity fingerprint."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import LOG, ErrorLog, Paths

EXPECTED = {
    "artificial-lunar": "Artificial Lunar Landscape (Kaggle, synthetic renders + masks)",
    "nasa-polar": "NASA POLAR stereo (real HDR-bracketed images, LiDAR ground truth)",
    "polar-sim-source": "POLAR-Sim annotation repository (YOLO boxes + polygons)",
}


def _walk(root: Path, errors: ErrorLog) -> list[tuple[str, int, int]]:
    """(relpath, size, mtime_ns) for every file below root, sorted."""
    out: list[tuple[str, int, int]] = []

    def onerror(e: OSError) -> None:
        errors.add("inventory.walk", str(getattr(e, "filename", root)), repr(e))

    for dp, dn, fn in os.walk(root, onerror=onerror):
        dn.sort()
        for f in sorted(fn):
            p = Path(dp) / f
            try:
                st = p.stat()
                out.append((p.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns))
            except OSError as e:
                errors.add("inventory.stat", p.as_posix(), repr(e))
    return out


def fingerprint(root: Path, errors: ErrorLog) -> dict[str, Any]:
    """Hash of (path, size, mtime) for all data files; detects any modification.

    `.git/` internals are hashed separately: IDE git tooling (e.g. GitLens writing .git/gk/config)
    touches them independently of this analysis, which would make the data check flaky.
    """
    files = _walk(root, errors)
    data = [f for f in files if "/.git/" not in f"/{f[0]}"]
    out = {}
    for name, sel in (("data", data), ("including_git_metadata", files)):
        h = hashlib.sha256()
        for rel, size, mt in sel:
            h.update(f"{rel}\t{size}\t{mt}\n".encode())
        out[name] = {"files": len(sel), "bytes": sum(s for _, s, _ in sel), "sha256": h.hexdigest()}
    return out


def _summarise(files: list[tuple[str, int, int]], depth: int = 3) -> dict[str, Any]:
    ext_n: Counter[str] = Counter()
    ext_b: Counter[str] = Counter()
    dirs: dict[str, Counter[str]] = defaultdict(Counter)
    zero = []
    git_n = git_b = 0
    for rel, size, _ in files:
        parts = rel.split("/")
        if ".git" in parts:
            git_n += 1
            git_b += size
            continue
        ext = Path(rel).suffix.lower() or "<none>"
        ext_n[ext] += 1
        ext_b[ext] += size
        d = "/".join(parts[: min(depth, len(parts) - 1)]) or "."
        dirs[d][ext] += 1
        if size == 0:
            zero.append(rel)
    return {
        "files_excluding_git": sum(ext_n.values()),
        "bytes_excluding_git": sum(ext_b.values()),
        "gib_excluding_git": round(sum(ext_b.values()) / 2**30, 3),
        "git_metadata": {"files": git_n, "bytes": git_b},
        "extensions": {e: {"files": ext_n[e], "bytes": ext_b[e]} for e in sorted(ext_n)},
        "directories": {d: dict(sorted(c.items())) for d, c in sorted(dirs.items())},
        "zero_byte_files": zero,
    }


def _check_zip(zpath: Path, extracted_root: Path, errors: ErrorLog) -> dict[str, Any]:
    """Compare archive member basenames+sizes with extracted files (reads central directory only)."""
    try:
        with zipfile.ZipFile(zpath) as z:
            members = [i for i in z.infolist() if not i.is_dir()]
    except (zipfile.BadZipFile, OSError) as e:
        errors.add("inventory.zip", zpath.as_posix(), repr(e))
        return {"readable": False}
    on_disk: dict[str, int] = {}
    for dp, _, fn in os.walk(extracted_root):
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() == ".zip" or ".git" in p.parts:
                continue
            on_disk.setdefault(f, p.stat().st_size)
    missing, size_mismatch = [], []
    for m in members:
        name = Path(m.filename).name
        if name not in on_disk:
            missing.append(m.filename)
        elif on_disk[name] != m.file_size:
            size_mismatch.append(m.filename)
    return {
        "readable": True,
        "members": len(members),
        "members_missing_on_disk": len(missing),
        "members_missing_examples": sorted(missing)[:20],
        "members_size_mismatch": len(size_mismatch),
        "size_mismatch_examples": sorted(size_mismatch)[:20],
    }


def run(paths: Paths, errors: ErrorLog) -> dict[str, Any]:
    LOG.info("A1 inventory")
    ds_root = paths.dataset_root
    result: dict[str, Any] = {"dataset_root": paths.rel(ds_root), "datasets": {}, "other_entries": {}}

    for child in sorted(ds_root.iterdir()):
        if not child.is_dir():
            result["other_entries"][child.name] = {"type": "file", "bytes": child.stat().st_size}
            continue
        files = _walk(child, errors)
        info = _summarise(files)
        info["expected_role"] = EXPECTED.get(child.name, "not referenced by the task brief")
        info["is_empty"] = len(files) == 0
        result["datasets"][child.name] = info

    for name in EXPECTED:
        if name not in result["datasets"]:
            errors.add("inventory", name, "expected dataset directory missing")
    result["missing_expected_datasets"] = [n for n in EXPECTED if n not in result["datasets"]]

    # Archive completeness: extracted copies vs. the zips sitting next to them.
    zips = {}
    al = paths.artificial_lunar
    if (al / "archive.zip").exists():
        zips["artificial-lunar/archive.zip"] = _check_zip(al / "archive.zip", al, errors)
    for z in sorted(paths.nasa_polar.glob("*.zip")):
        target = paths.nasa_polar / z.stem
        zips[f"nasa-polar/{z.name}"] = _check_zip(z, target, errors)
    result["archive_vs_extracted"] = zips

    # Second POLAR-Sim checkout at the project root (outside dataset/).
    dup = paths.project_root / "polar-sim-source"
    if dup.exists():
        a = {r: s for r, s, _ in _walk(paths.polar_sim, errors) if ".git/" not in r}
        b = {r: s for r, s, _ in _walk(dup, errors) if ".git/" not in r}
        result["duplicate_polar_sim_checkout"] = {
            "path": paths.rel(dup),
            "files": len(b),
            "only_in_dataset_copy": sorted(set(a) - set(b))[:50],
            "only_in_root_copy": sorted(set(b) - set(a))[:50],
            "size_differences": sorted(k for k in set(a) & set(b) if a[k] != b[k])[:50],
            "identical_by_path_and_size": set(a) == set(b) and all(a[k] == b[k] for k in a),
        }
    return result
