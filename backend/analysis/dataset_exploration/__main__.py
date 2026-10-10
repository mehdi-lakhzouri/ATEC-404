"""Run the full read-only dataset exploration (Task AI-01.A).

Usage (from backend/):
    uv run python -m analysis.dataset_exploration [--project-root ..] [--out ../experiments/dataset-exploration] [--workers 8]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from . import artificial_lunar, inventory, matching, nasa_polar, polar_sim, quality
from .common import LOG, ErrorLog, Paths, setup_logging, write_json


def main() -> int:
    here = Path(__file__).resolve()
    default_root = here.parents[3]  # .../hackathon
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", type=Path, default=default_root)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args()
    root = args.project_root.resolve()
    paths = Paths(root, (args.out or root / "experiments" / "dataset-exploration").resolve())
    paths.ensure()
    setup_logging(paths.logs / "run.log")
    errors = ErrorLog()
    t0 = time.perf_counter()

    fp_before = inventory.fingerprint(paths.dataset_root, errors)
    LOG.info("source data fingerprint before: %s (%d files)", fp_before["data"]["sha256"][:16], fp_before["data"]["files"])

    inv = inventory.run(paths, errors)
    inv["source_fingerprint"] = fp_before
    write_json(paths.out_dir / "dataset_inventory.json", inv)

    al = artificial_lunar.run(paths, errors, args.workers)
    write_json(paths.out_dir / "artificial_lunar_analysis.json", al)

    nasa = nasa_polar.run(paths, errors, args.workers)
    write_json(paths.out_dir / "nasa_polar_analysis.json", nasa)

    ps = polar_sim.run(paths, errors)
    write_json(paths.out_dir / "polar_sim_analysis.json", {k: v for k, v in ps.items() if not k.startswith("_")})

    mt = matching.run(paths, nasa, ps, nasa["terrains"])
    write_json(paths.out_dir / "annotation_matching_report.json", mt)

    fp_after = inventory.fingerprint(paths.dataset_root, errors)
    unchanged = fp_before["data"] == fp_after["data"]
    issues = quality.build(inv, al, nasa, ps, mt, errors.items)
    samples = sorted(p.name for p in paths.samples.glob("*.png"))
    figures = sorted(p.name for p in paths.figures.glob("*.png"))
    write_json(paths.out_dir / "data_quality_report.json", {
        "issues": issues,
        "severity_counts": {s: sum(i["severity"] == s for i in issues) for s in quality.ORDER},
        "source_integrity": {
            "fingerprint_before": fp_before, "fingerprint_after": fp_after,
            "unchanged": unchanged,
            "git_metadata_unchanged": fp_before["including_git_metadata"] == fp_after["including_git_metadata"],
            "method": "sha256 over (relative path, size, mtime_ns) of every file under dataset/; "
                      "'data' excludes .git/ internals, which IDE git tooling may touch",
        },
        "logged_errors": errors.items,
        "outputs": {"figures": figures, "samples": samples},
    })
    LOG.info("source data unchanged: %s", unchanged)
    LOG.info("done in %.1f s, %d logged errors", time.perf_counter() - t0, len(errors.items))
    return 0 if unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
