"""Read-only, hard-link-aware storage audit of the Artemis-X workspace.

For every audit unit (a directory at a chosen depth) it reports:
  apparent_bytes     sum of file sizes as listed (hard links counted once per path)
  physical_bytes     unique on-disk files inside the unit (each NTFS file record counted once)
  reclaimable_bytes  bytes freed if ONLY this unit were deleted: files whose every hard link lies inside the unit
Files whose link count exceeds the links found inside the workspace are shared with something outside it
(for example the uv or pnpm stores) and are never counted as reclaimable.

Junctions and symbolic links are recorded and not followed. Nothing is modified.

Usage:  uv run --project backend --no-sync python scripts/audit_workspace.py [--out docs/audits/raw/storage_scan.json]
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# directories audited at a finer depth (relative POSIX prefix -> depth of the unit key below ROOT)
DEPTH = {"dataset/processed": 3, "dataset": 2, "experiments/task-e": 3, "experiments/task-f": 3, "experiments": 2,
         "backend": 2, "frontend": 2, "polar-sim-source": 2}
FINER = ("experiments/task-e/full-training-config-a-200-s42", "experiments/task-e/smoke-test-001", "experiments/task-e/benchmark-e1a-001",
         "dataset/processed/yolo-rock-v1.1", "dataset/processed/yolo-rock-v1", "dataset/processed/curated-dataset-v1.1",
         "dataset/processed/curated-dataset-v1", "dataset/artificial-lunar", "dataset/nasa-polar")


def unit_of(rel: str) -> str:
    parts = rel.split("/")
    for pre in FINER:
        if rel.startswith(pre + "/"):
            return "/".join(parts[:len(pre.split("/")) + 1]) if len(parts) > len(pre.split("/")) + 1 else pre
    for pre, d in sorted(DEPTH.items(), key=lambda kv: -len(kv[0])):
        if rel == pre or rel.startswith(pre + "/"):
            return "/".join(parts[:d]) if len(parts) > d else "/".join(parts[:-1]) or parts[0]
    return parts[0] if len(parts) > 1 else "(root files)"


def scan() -> dict:
    files: dict[str, list] = defaultdict(list)       # unit -> [(inode_key, size, nlink, mtime, ext)]
    inode_units: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    inode_info: dict[tuple, tuple[int, int]] = {}
    links_skipped, errors = [], 0
    big: list[tuple[int, str, int]] = []
    t0 = time.time()
    for dp, dn, fn in os.walk(ROOT, followlinks=False):
        keep = []
        for d in dn:
            p = Path(dp) / d
            try:
                if p.is_symlink() or os.path.isjunction(p):
                    links_skipped.append(p.relative_to(ROOT).as_posix()); continue
            except OSError:
                continue
            keep.append(d)
        dn[:] = keep
        for f in fn:
            p = Path(dp) / f
            try:
                st = p.stat(follow_symlinks=False)
            except OSError:
                errors += 1; continue
            if stat.S_ISLNK(st.st_mode):
                links_skipped.append(p.relative_to(ROOT).as_posix()); continue
            rel = p.relative_to(ROOT).as_posix()
            u = unit_of(rel)
            key = (st.st_dev, st.st_ino)
            files[u].append((key, st.st_size, st.st_nlink, st.st_mtime, p.suffix.lower()))
            inode_units[key][u] += 1
            inode_info[key] = (st.st_size, st.st_nlink)
            if st.st_size >= 50 * 2**20:
                big.append((st.st_size, rel, st.st_nlink))
    units = {}
    for u, rows in files.items():
        seen = set(); phys = recl = 0; ext = defaultdict(int); newest = 0.0; multi = 0
        for key, size, nlink, mt, e in rows:
            ext[e or "(none)"] += size; newest = max(newest, mt)
            if nlink > 1:
                multi += 1
            if key in seen:
                continue
            seen.add(key); phys += size
            found_total = sum(inode_units[key].values())
            if found_total >= nlink and set(inode_units[key]) == {u}:
                recl += size
        units[u] = {"files": len(rows), "apparent_bytes": sum(r[1] for r in rows), "physical_bytes": phys, "reclaimable_bytes": recl,
                    "files_with_multiple_links": multi, "last_modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)),
                    "top_types": dict(sorted(ext.items(), key=lambda kv: -kv[1])[:5])}
    total_phys = sum(s for s, _ in inode_info.values())
    external = sum(s for k, (s, n) in inode_info.items() if sum(inode_units[k].values()) < n)
    shared_units = defaultdict(int)
    for k, us in inode_units.items():
        if len(us) > 1:
            shared_units[" <-> ".join(sorted(us))] += inode_info[k][0]
    return {"root": ROOT.name, "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "scan_seconds": round(time.time() - t0, 1),
            "total_files": sum(len(v) for v in files.values()), "total_apparent_bytes": sum(u["apparent_bytes"] for u in units.values()),
            "total_physical_bytes": total_phys, "bytes_hardlinked_outside_workspace": external,
            "hardlink_sharing_between_units": dict(sorted(shared_units.items(), key=lambda kv: -kv[1])[:40]),
            "links_and_junctions_not_followed": links_skipped[:200], "stat_errors": errors,
            "large_files": [{"path": r, "bytes": s, "nlink": n} for s, r, n in sorted(big, reverse=True)],
            "units": dict(sorted(units.items(), key=lambda kv: -kv[1]["physical_bytes"]))}


def archives() -> list[dict]:
    """Inventory every ZIP and whether its members exist extracted (name + size + CRC-32 of the extracted file)."""
    import zlib
    out = []
    for z in sorted(p for p in ROOT.rglob("*") if p.suffix.lower() in (".zip", ".rar", ".7z", ".tar", ".gz") and ".venv" not in p.parts
                    and "node_modules" not in p.parts and ".git" not in p.parts):
        rec = {"path": z.relative_to(ROOT).as_posix(), "bytes": z.stat().st_size,
               "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(z.stat().st_mtime))}
        if z.suffix.lower() != ".zip":
            out.append(rec); continue
        with zipfile.ZipFile(z) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            rec.update(members=len(infos), uncompressed_bytes=sum(i.file_size for i in infos))
            roots = {i.filename.split("/")[0] for i in infos}
            rec["top_level_entries"] = sorted(roots)[:10]
            # candidate extraction bases: next to the zip, or a folder named like the zip
            bases = [z.parent, z.parent / z.stem]
            best = None
            for base in bases:
                found = same = 0
                for i in infos:
                    f = base / i.filename
                    if f.is_file():
                        found += 1
                        if f.stat().st_size == i.file_size:
                            same += 1
                if best is None or found > best[1]:
                    best = (base, found, same)
            base, found, same = best
            rec["extracted_base"] = base.relative_to(ROOT).as_posix() if found else None
            rec["members_found_extracted"] = found
            rec["members_same_size"] = same
            if found and found == len(infos):
                crc_ok = 0
                for i in infos:
                    h = 0
                    with open(base / i.filename, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h = zlib.crc32(chunk, h)
                    crc_ok += int(h == i.CRC)
                rec["members_crc32_identical"] = crc_ok
                rec["fully_extracted_identical"] = crc_ok == len(infos)
            else:
                rec["fully_extracted_identical"] = False
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/audits/raw/storage_scan.json")
    ap.add_argument("--skip-archives", action="store_true")
    a = ap.parse_args()
    res = scan()
    if not a.skip_archives:
        res["archives"] = archives()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: res[k] for k in ("total_files", "total_apparent_bytes", "total_physical_bytes", "bytes_hardlinked_outside_workspace",
                                          "scan_seconds", "stat_errors")}, indent=1))
