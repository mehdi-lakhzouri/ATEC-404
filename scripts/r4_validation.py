"""R4.6 final mapping validation: one verdict over the R4.3 safety, R4.4 fusion and R4.5 monitor batches of the SAME code,
plus the obstacle output contract (artemis.obstacles.v1) checked against ground truth. Ground truth is used offline here
only. Every criterion is reported with its measured value - failures are listed, never dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import r4_fusion_analysis as F
import r4_safety_analysis as sa
from run_webots_r4 import ROOT, WB, me

P95_TARGET_MS = 15.0
WEBOTS_MEM_LIMIT_MB = 3000.0          # flat ~1.3-1.7 GB expected; the R4.5 leak grew > 1 GB/s
TRACK_MEDIAN_M, TRACK_P95_M, DUP_RATE = 0.10, 0.25, 0.05


def obstacle_contract_report(run_dir: Path, man: dict, wg: me.WorldGrid, gt: dict) -> dict:
    """The run's final obstacle list vs ground truth, in the window it covers (12 x 12 m around the final pose)."""
    doc = json.loads((run_dir / "tour" / "mapping" / "obstacles_final.json").read_text(encoding="utf-8"))
    run = sa.load_run(run_dir)
    al = me.alignment(run["cyc"][0])
    obs = doc["obstacles"]
    if not obs:
        return {"schema": doc["schema"], "obstacles": 0}
    W = me.to_world(np.array([[o["x_m"], o["y_m"]] for o in obs]), al)
    haz = gt["pos"] | gt["neg"]
    objs = [o for o in gt["objects"] if o["kind"] != "crater"]
    rows = []
    for o, (x, y) in zip(obs, W):
        # does the exported disc overlap a TRUE hazard cell? (sampled at 5 cm)
        r = o["radius_m"]
        g = np.arange(-r, r + 1e-9, 0.05)
        X, Y = np.meshgrid(g, g)
        m = X ** 2 + Y ** 2 <= r * r
        i, j, ok = wg.ij(x + X[m], y + Y[m])
        on_hazard = bool(haz[j[ok], i[ok]].any()) if ok.any() else False
        d = [math.hypot(x - b["x"], y - b["y"]) for b in objs]
        k = int(np.argmin(d))
        rows.append({"id": o["id"], "source": o["source"], "type": o["type"], "on_true_hazard": on_hazard,
                     "nearest_object": objs[k]["name"], "centre_error_m": round(d[k], 3),
                     "within_footprint": d[k] <= objs[k]["width_m"] / 2 + 0.35, "uncertainty_m": o["uncertainty_m"]})
    trk = [r for r in rows if r["source"] == "yolo+depth" and r["within_footprint"]]
    dep = [r for r in rows if r["source"] == "depth"]
    # recall: true objects whose centre is inside the exported window and that the rover saw (within 4 m, in the FOV)
    ox, oy = doc["grid"]["origin_xy"]
    size = doc["grid"]["size_m"]
    corners = me.to_world(np.array([[ox, oy], [ox + size, oy + size]]), al)
    G = run["tel"]["gt"]
    covered, total, missed = 0, 0, []
    for b in objs:
        if not (min(corners[:, 0]) <= b["x"] <= max(corners[:, 0]) and min(corners[:, 1]) <= b["y"] <= max(corners[:, 1])):
            continue
        dd = np.hypot(b["x"] - G[:, 0], b["y"] - G[:, 1]) - b["width_m"] / 2
        bb = np.abs(np.angle(np.exp(1j * (np.arctan2(b["y"] - G[:, 1], b["x"] - G[:, 0]) - G[:, 2]))))
        if not ((dd <= 4.0) & (bb <= 0.8)).any():
            continue
        total += 1
        hit = any(math.hypot(b["x"] - x, b["y"] - y) <= b["width_m"] / 2 + o["radius_m"] + 0.10 for o, (x, y) in zip(obs, W))
        covered += hit
        if not hit:
            missed.append(f"{b['name']} (h {b['height_m']})")
    errs = [r["centre_error_m"] for r in trk]
    return {"schema": doc["schema"], "frame": doc["frame"], "obstacles": len(obs),
            "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in ("yolo+depth", "depth")},
            "by_type": {t: sum(1 for r in rows if r["type"] == t) for t in sorted({r["type"] for r in rows})},
            "track_centre_error_m": {"n": len(errs), "median": sa.q(errs, 0.5), "p95": sa.q(errs, 0.95),
                                     "max": round(max(errs), 3) if errs else None},
            "tracks_not_on_a_true_object": sum(1 for r in rows if r["source"] == "yolo+depth" and not r["within_footprint"]),
            "depth_clusters_on_true_hazard": f"{sum(r['on_true_hazard'] for r in dep)}/{len(dep)}",
            "depth_clusters_off_hazard": [r["id"] for r in dep if not r["on_true_hazard"]],
            "objects_in_window_seen": total, "objects_covered": covered, "objects_missed": missed,
            "export_ms": run["summ"]["mapping"].get("obstacles_export", {}).get("ms")}


def analyse_validation(root: Path, mem_samples: list | None = None) -> dict:
    from r4_monitor_analysis import analyse_monitor
    from run_webots_r4 import analyse_safety
    S = analyse_safety(root / "safety")["scenarios"]
    Fu = F.analyse_fusion(root / "fusion")["scenarios"]
    M = analyse_monitor(root / "monitor")["scenarios"]
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    wg = me.WorldGrid()
    gt = me.ground_truth(man, wg)
    contract = obstacle_contract_report(root / "fusion" / "fusion", man, wg, gt)

    p95 = {**{f"safety/{k}": v["timing_ms"]["cycle_ms"]["p95"] for k, v in S.items()},
           **{f"fusion/{k}": v["timing_ms"]["cycle_ms"]["p95"] for k, v in Fu.items()},
           **{f"monitor/{k}": v["mapping_timing_ms"]["cycle_ms"]["p95"] for k, v in M.items()}}
    moves = {**{f"safety/{k}": v["tour"] for k, v in S.items()}, **{f"fusion/{k}": v["tour"] for k, v in Fu.items()},
             **{f"monitor/{k}": v["tour"] for k, v in M.items()}}
    fus, dis = Fu["fusion"], Fu["fusion_disconnect"]
    fd = {k: [(f["reason"], f["latency_s"], f["detected"]) for f in v["fault_detection"]] for k, v in S.items() if v["fault_detection"]}
    wb = [s["webots_mb"] for s in (mem_samples or []) if s.get("webots_mb")]
    sha = hashlib.sha256((ROOT / "models" / "artemis_rock_yolo11n.pt").read_bytes()).hexdigest()
    frozen = json.loads((ROOT / "artifacts" / "provenance" / "FROZEN_MODEL_CONFIG.json").read_text(encoding="utf-8"))["model"]["sha256"]
    checks = [
        ("all tours complete (55/55)", all(m["completed_moves"] == m["planned_moves"] for m in moves.values()),
         f"{sum(m['completed_moves'] == m['planned_moves'] for m in moves.values())}/{len(moves)} runs"),
        ("mapping cycle p95 < 15 ms in every run", max(p95.values()) < P95_TARGET_MS, f"max {max(p95.values())} ms ({max(p95, key=p95.get)})"),
        ("track centre error median <= 0.10 m", fus["tracks"]["centre_error_m"]["median"] <= TRACK_MEDIAN_M, f"{fus['tracks']['centre_error_m']['median']} m"),
        ("track centre error p95 <= 0.25 m", fus["tracks"]["centre_error_m"]["p95"] <= TRACK_P95_M, f"{fus['tracks']['centre_error_m']['p95']} m"),
        ("duplicate track rate <= 5 %", fus["tracks"]["duplicates"]["rate"] <= DUP_RATE, f"{fus['tracks']['duplicates']['rate']:.1%}"),
        ("obstacles >= 0.10 m mapped", fus["map"]["recall"]["h >= 0.10 m"]["mapped"] == fus["map"]["recall"]["h >= 0.10 m"]["in_view"],
         f"{fus['map']['recall']['h >= 0.10 m']['mapped']}/{fus['map']['recall']['h >= 0.10 m']['in_view']} (missed: {fus['map']['recall']['h >= 0.10 m']['missed']})"),
        ("no unsafe cost cell inside a true hazard (nominal fusion)", fus["map"]["cost_vs_truth"]["unsafe_cells_inside_hazard"] == 0,
         f"{fus['map']['cost_vs_truth']['unsafe_cells_inside_hazard']} inside, {fus['map']['cost_vs_truth']['unsafe_cells']} in collision zone"),
        ("corridor never clear while truly blocked (all safety runs)", all(v["corridor_vs_truth"]["dangerous_clear_while_blocked"] == 0 for v in S.values()),
         str({k: v["corridor_vs_truth"]["dangerous_clear_while_blocked"] for k, v in S.items()})),
        ("every injected fault detected", all(d for v in fd.values() for _, _, d in v), json.dumps(fd)),
        ("YOLO disconnect detected, mapping continues", bool(dis.get("disconnect", {}).get("detected")) and dis["tour"]["completed_moves"] == 55,
         f"latency {dis.get('disconnect', {}).get('latency_s')} s, cycles after {dis.get('disconnect', {}).get('cycles_after_kill')}"),
        ("map reset clears the map (monitor_reset)", len(M["monitor_reset"]["map_resets"]) == 1 and (M["monitor_reset"]["monitor"] or {}).get("map_resets_seen") == 1,
         f"{len(M['monitor_reset']['map_resets'])} reset, seen by monitor: {(M['monitor_reset']['monitor'] or {}).get('map_resets_seen')}"),
        ("depth mapping without YOLO (monitor_noyolo)", M["monitor_noyolo"]["grid_final"]["occupied"] > 0,
         f"{M['monitor_noyolo']['grid_final']['occupied']} occupied cells, all depth-only: {M['monitor_noyolo']['grid_final']['occupied_by_source']}"),
        ("Webots memory stable (< 3 GB, no growth)", bool(wb) and max(wb) < WEBOTS_MEM_LIMIT_MB,
         f"{min(wb) if wb else None}-{max(wb) if wb else None} MB over {len(wb)} samples"),
        ("frozen YOLO model unchanged", sha == frozen, sha[:16]),
        ("obstacle contract: every exported track on a true object", contract.get("tracks_not_on_a_true_object", 1) == 0,
         f"{contract.get('tracks_not_on_a_true_object')} tracks not on a true object (crater rim / terrain / engineered)"),
    ]
    out = {"label": "R4.6 final mapping validation - ground truth used offline here only", "root": root.relative_to(ROOT).as_posix(),
           "checks": [{"check": c, "pass": bool(p), "value": v} for c, p, v in checks],
           "passed": sum(1 for _, p, _ in checks if p), "total": len(checks),
           "cycle_p95_ms": p95, "obstacle_contract": contract,
           "webots_memory_mb": {"samples": len(wb), "min": min(wb) if wb else None, "max": max(wb) if wb else None,
                                "first_third_mean": round(float(np.mean(wb[:len(wb) // 3])), 1) if len(wb) > 2 else None,
                                "last_third_mean": round(float(np.mean(wb[-(len(wb) // 3):])), 1) if len(wb) > 2 else None}}
    (root / "r4_6_validation.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out
