"""R4.5 display part: evaluation of the 2 x 2 monitoring world runs (scripts/run_webots_r4.py monitor).

Per scenario: tour completion, mapping timing (the mapping cycle must not be slowed by the panels), panel rendering cost
on idle steps, on-screen map age, memory, health, and - for the panels-on vs panels-off pair - the offline map and track
accuracy (ground truth, offline only), which must be identical in kind. Also the YOLO-absent run (depth mapping goes on)
and the explicit map reset (the map panel is cleared). Figures: the GUI screenshots and the map panel at mission stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import r4_fusion_analysis as F
import r4_safety_analysis as sa
from run_webots_r4 import FIG, ROOT, WB, me


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "tour" / "summary.json").read_text(encoding="utf-8"))


def analyse_monitor(root: Path) -> dict:
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    wg = me.WorldGrid()
    gt = me.ground_truth(man, wg)
    zone = sa.gt_collision_zone(gt, wg, sa.CFG.collision_radius_m)
    foot = F.rock_footprints(man, wg)
    out = {"label": "R4.5 display part - 2 x 2 monitoring world; ground truth used offline here only",
           "root": root.relative_to(ROOT).as_posix(), "scenarios": {}}
    for d in sorted(p for p in root.iterdir() if (p / "tour" / "summary.json").exists()):
        s = _summary(d)
        mp = s.get("mapping", {})
        meta = json.loads((d / "scenario.json").read_text(encoding="utf-8")) if (d / "scenario.json").exists() else {}
        res = {"tour": {"completed_moves": s["final"]["completed_moves"], "planned_moves": s["final"]["planned_moves"]},
               "mapping_timing_ms": {k: mp.get("timing_ms", {}).get(k) for k in ("cycle_ms", "step_max_ms", "scan_ms", "grid_ms", "cost_ms")},
               "controller_work_ms_per_step": s.get("controller_work_ms_per_step"),
               "cycles": mp.get("cycles"), "memory_mb_end": (mp.get("process_memory_mb") or {}).get("at_end"),
               "health_time_in_state_s": (mp.get("health") or {}).get("time_in_state_s"),
               "grid_final": {k: (mp.get("grid") or {}).get(k) for k in ("free", "occupied", "negative", "unknown", "occupied_by_source", "resets")},
               "map_resets": mp.get("map_resets", []), "monitor": s.get("monitor"),
               "inference": {k: (s.get("inference") or {}).get(k) for k in ("submitted", "ok", "failed")},
               "screenshots": meta.get("screenshots", []), "wall_s": meta.get("wall_s")}
        if d.name in ("monitor", "monitor_off"):
            run = sa.load_run(d)
            mr = F.map_report(run, man, wg, gt, zone, foot)
            tr = F.track_report(run, man, mr["_al"])
            res["map"] = {"recall": {k: (v["mapped"], v["in_view"]) for k, v in mr["recall"].items()},
                          "occupied_precision_strict": mr["occupied_precision_strict"],
                          "false_free_hazard_cells": mr["false_free_hazard_cells"],
                          "unsafe_cells": mr["cost_vs_truth"]["unsafe_cells"],
                          "unsafe_inside_hazard": mr["cost_vs_truth"]["unsafe_cells_inside_hazard"]}
            res["tracks"] = {"centre_error_m": tr["centre_error_m"], "duplicate_rate": tr["duplicates"]["rate"],
                             "rocks_tracked": tr["rocks_tracked"]}
        out["scenarios"][d.name] = res
    (root / "r4_5_monitor_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figures(root)
    return out


def figures(root: Path) -> None:
    """GUI screenshots (full 2 x 2 layout) and map-panel stills at mission stages / around the reset -> docs/figures."""
    from PIL import Image, ImageDraw
    FIG.mkdir(parents=True, exist_ok=True)
    gui = root / "monitor_gui"
    shots = sorted(gui.glob("screen_*.png")) if gui.exists() else []
    for i, p in enumerate(shots):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, h - int(h * 0.05)))                       # drop the OS task bar
        im.save(FIG / f"r4_5_layout_{i + 1}.jpg", quality=90)

    def strip(files: list, labels: list, name: str, cols: int) -> None:
        ims = [Image.open(f).convert("RGB") for f in files]
        if not ims:
            return
        w, h = ims[0].size
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * (h + 26)), (11, 17, 24))
        dr = ImageDraw.Draw(sheet)
        for k, (im, lab) in enumerate(zip(ims, labels)):
            x, y = (k % cols) * w, (k // cols) * (h + 26)
            sheet.paste(im, (x, y + 26))
            dr.text((x + 8, y + 7), lab, fill=(230, 237, 243))
        sheet.save(FIG / name, quality=90) if name.endswith(".jpg") else sheet.save(FIG / name)

    stages = sorted((gui / "panels").glob("t*_map.png")) if (gui / "panels").exists() else []
    strip(stages, [f"LOCAL MAP at t = {int(p.name[1:5])} s (simulation)" for p in stages], "r4_5_map_stages.png", 3)
    rs = root / "monitor_reset" / "panels"
    if rs.exists():
        files = sorted(rs.glob("t*_map.png"))
        strip(files, [f"t = {int(p.name[1:5])} s" + (" (reset at 150 s)" if int(p.name[1:5]) >= 150 else " (before reset)") for p in files],
              "r4_5_map_reset.png", 4)


if __name__ == "__main__":
    import sys
    print(json.dumps(analyse_monitor(Path(sys.argv[1]).resolve()), indent=1, default=float))
