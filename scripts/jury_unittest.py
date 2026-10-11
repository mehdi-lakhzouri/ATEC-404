"""Run unittest with explicit machine-readable counts, including every skip.

Used by jury.ps1; this runner never trains or evaluates the held-out test set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("runtime", "research", "ground"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-integrations", action="store_true")
    args = ap.parse_args()
    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "backend"), str(ROOT / "ground_station")]
    directory = {"runtime": ROOT / "tests", "research": ROOT / "backend/tests", "ground": ROOT / "ground_station/tests"}[args.suite]
    suite = unittest.defaultTestLoader.discover(str(directory), top_level_dir=str(directory))
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"suite": args.suite, "tests": result.testsRun,
              "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
              "failures": [t.id() for t, _ in result.failures], "errors": [t.id() for t, _ in result.errors],
              "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
              "elapsed_s": round(time.monotonic() - start, 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not result.wasSuccessful():
        return 1
    # Historical dataset/checkpoint parity may be unavailable. MongoDB tests may
    # not count as a successful integration run when their prerequisite is down.
    if args.require_integrations and args.suite == "ground" and result.skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
