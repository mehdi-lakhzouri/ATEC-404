"""Opt-in controller profiling (stdlib only). Enabled only by ARTEMIS_PROFILE=1; otherwise every call is a no-op.

Writes, next to the run's summary:
    profile.pstats         cProfile statistics (open with ``python -m pstats`` or ``pstats.Stats``)
    profile_top.txt        top functions by own time and by cumulative time
    gc_pauses.json         every garbage-collection pause (generation, duration) - pauses land inside whatever stage
                           happens to be running, so stage timers alone can blame the wrong stage
"""

from __future__ import annotations

import gc
import io
import json
import os
import time
from pathlib import Path


class _Prof:
    def __init__(self):
        import cProfile
        self.pr = cProfile.Profile()
        self.gc_events: list = []
        self._t0 = None
        gc.callbacks.append(self._gc)
        self.pr.enable()

    def _gc(self, phase, info):
        if phase == "start":
            self._t0 = time.perf_counter()
        elif self._t0 is not None:
            self.gc_events.append((info.get("generation"), (time.perf_counter() - self._t0) * 1000, info.get("collected", 0)))
            self._t0 = None

    def dump(self, out_dir: Path) -> dict:
        import pstats
        self.pr.disable()
        gc.callbacks.remove(self._gc)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.pr.dump_stats(str(out_dir / "profile.pstats"))
        buf = io.StringIO()
        st = pstats.Stats(self.pr, stream=buf)
        st.sort_stats("tottime").print_stats(30)
        st.sort_stats("cumulative").print_stats(30)
        (out_dir / "profile_top.txt").write_text(buf.getvalue(), encoding="utf-8")
        ev = self.gc_events
        by_gen = {}
        for g, ms, _ in ev:
            d = by_gen.setdefault(str(g), {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            d["count"] += 1; d["total_ms"] += ms; d["max_ms"] = max(d["max_ms"], ms)
        s = {"gc_by_generation": {k: {**v, "total_ms": round(v["total_ms"], 1), "max_ms": round(v["max_ms"], 2)} for k, v in by_gen.items()},
             "gc_pauses_over_2ms": sum(1 for _, ms, _ in ev if ms > 2.0)}
        (out_dir / "gc_pauses.json").write_text(json.dumps({**s, "events": ev}, indent=0), encoding="utf-8")
        return s


def start():
    return _Prof() if os.environ.get("ARTEMIS_PROFILE") == "1" else None


def dump(prof, out_dir: Path):
    return prof.dump(out_dir) if prof is not None else None
