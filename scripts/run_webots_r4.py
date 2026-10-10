"""Webots R4 runner: headless mapping scenarios and their offline evaluation against ground truth.

    uv run --project backend --no-sync python scripts/run_webots_r4.py measure               # R4.1: tour + measurement checks
    uv run --project backend --no-sync python scripts/run_webots_r4.py measure --no-inference
    uv run --project backend --no-sync python scripts/run_webots_r4.py analyse <run_dir>
    uv run --project backend --no-sync python scripts/run_webots_r4.py map                   # R4.2: grid with YOLO disabled
    uv run --project backend --no-sync python scripts/run_webots_r4.py analyse-map <run_dir>
    uv run --project backend --no-sync python scripts/run_webots_r4.py safety                # R4.3: 8 safety / fault scenarios
    uv run --project backend --no-sync python scripts/run_webots_r4.py fusion                # R4.4: fusion vs depth-only, 400 ms, disconnect
    uv run --project backend --no-sync python scripts/run_webots_r4.py monitor               # R4.5 display: 2 x 2 panels, timing, screenshots
    uv run --project backend --no-sync python scripts/run_webots_r4.py validate              # R4.6: all of the above, one verdict (~25 min)

The rover drives the existing scripted course tour (no autonomy). The controller maps with its ESTIMATED pose only; this
script is the only place where ground truth (course manifest, logged supervisor pose) is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_webots_r3 as r3  # noqa: E402  (server lifecycle, headless Webots, tour stop accuracy)
from artemis_x.evaluation import map_eval as me  # noqa: E402

WB = ROOT / "artemis-webots"
FIG = ROOT / "docs" / "figures" / "webots"
PORT = 5771                     # not 5761: an interactive Webots session may be holding the default port


def measure(out: Path, inference: bool = True) -> dict:
    """Run the course tour with mapping on (default for the tour world) and return R3's tour summary for this run."""
    if inference:
        res = r3.run_tour(out, port=PORT)
    else:
        r3.subprocess.run([sys.executable, str(WB / "tools" / "make_world.py")], check=True, capture_output=True)
        out.mkdir(parents=True, exist_ok=True)
        code, log = r3.run_webots(WB / "worlds" / "artemis_test_course.wbt", {"ARTEMIS_WEBOTS_QUIT": "1", "ARTEMIS_WEBOTS_OUT": out / "tour",
                                                                              "ARTEMIS_INFERENCE": "0"}, timeout=420)
        (out / "webots.log").write_text(log, encoding="utf-8")
        res = {"webots_exit": code}
    return res


def _telemetry_est(tour: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-step estimated pose (odom) from the controller telemetry, for the capture-time diagnostic."""
    rows = list(csv.DictReader((tour / "telemetry.csv").open(encoding="utf-8")))
    t = np.array([float(r["sim_time_s"]) for r in rows])
    E = np.array([[float(r["odo_x"]), float(r["odo_y"]), float(r["odo_yaw"])] for r in rows])
    return t, E


def analyse(run: Path) -> dict:
    tour = run / "tour"
    mdir = tour / "mapping"
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    cyc = me.load_jsonl(mdir / "mapping.jsonl")
    al = me.alignment(cyc[0])

    # 1. estimated pose vs ground truth at every cycle (alignment fixed at the start only)
    pe = np.array([[c["pose_est"]["x"], c["pose_est"]["y"], c["pose_est"]["yaw"]] for c in cyc])
    pg = np.array([[c["gt_pose_eval_only"]["x"], c["gt_pose_eval_only"]["y"], c["gt_pose_eval_only"]["yaw"]] for c in cyc])
    pw = me.to_world(pe[:, :2], al)
    pos_err = np.hypot(*(pw - pg[:, :2]).T)
    yaw_err = np.abs(np.arctan2(np.sin(pe[:, 2] + al[2] - pg[:, 2]), np.cos(pe[:, 2] + al[2] - pg[:, 2])))

    # 2. obstacle points (estimated pose) vs the true surfaces
    pts_log = me.load_jsonl(mdir / "obstacle_points.jsonl")
    rows = [(r["stamp_s"], r["pose_est"]["x"], r["pose_est"]["y"], r["pose_est"]["yaw"], *p) for r in pts_log for p in r["points"]]
    stamp = np.array([r[0] for r in rows]); pose = np.array([r[1:4] for r in rows], float)
    P_odom = np.array([r[4:7] for r in rows], float); kind = np.array([r[7] for r in rows])
    Pw = np.column_stack([me.to_world(P_odom[:, :2], al), P_odom[:, 2]])
    dist, nearest = me.surface_distance(Pw, man)
    pos, neg = kind == "p", kind == "n"
    names = ["rock", "engineered object", "terrain"]

    # 3. capture-time diagnostic: re-project the same points with the pose one sensor period EARLIER
    tt, E = _telemetry_est(tour)
    per = summ["sensor"]["period_ms"] / 1000
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    dx, dy = P_odom[:, 0] - pose[:, 0], P_odom[:, 1] - pose[:, 1]
    bx, by = c * dx + s * dy, -s * dx + c * dy                         # back to the base frame
    alt = []
    for k, sh in enumerate((0.0, per)):
        j = np.clip(np.searchsorted(tt, stamp - sh - 1e-9), 0, len(tt) - 1)
        ex, ey, eyaw = E[j, 0], E[j, 1], E[j, 2]
        ox, oy = ex + np.cos(eyaw) * bx - np.sin(eyaw) * by, ey + np.sin(eyaw) * bx + np.cos(eyaw) * by
        sel = np.nonzero(pos)[0][::7]
        d, _ = me.surface_distance(np.column_stack([me.to_world(np.column_stack([ox, oy])[sel], al), P_odom[sel, 2]]), man)
        alt.append({"pose_time": "sample instant (used)" if k == 0 else f"one period earlier (-{per * 1000:.0f} ms)",
                    "positive_points": int(len(sel)), "p50_m": me.q(d, 0.5), "p95_m": me.q(d, 0.95)})

    det = me.load_jsonl(tour / "inference" / "detections.jsonl") if (tour / "inference" / "detections.jsonl").exists() else []
    det_ok = [d for d in det if d.get("association", "").startswith("same capture")]
    mp = summ.get("mapping", {})
    res = {
        "label": "R4.1 measurement pipeline - Webots course tour, estimated pose only; ground truth used offline here",
        "run": run.relative_to(ROOT).as_posix(),
        "tour": {"completed_moves": summ["final"]["completed_moves"], "planned_moves": summ["final"]["planned_moves"],
                 "controller_work_ms_per_step": summ["controller_work_ms_per_step"]},
        "mapping_cycles": mp.get("cycles"), "cycles_without_pose": mp.get("no_pose"), "cycles_without_depth": mp.get("no_depth"),
        "cycle_period_s": round(float(np.median(np.diff([c["stamp_s"] for c in cyc]))), 4),
        "timing_ms": mp.get("timing_ms"),
        "valid_depth_fraction": {"p05": me.q([c["valid_fraction"] for c in cyc], 0.05), "median": me.q([c["valid_fraction"] for c in cyc], 0.5)},
        "pose_estimate_vs_ground_truth": {"position_err_m": {"median": me.q(pos_err, 0.5), "max": round(float(pos_err.max()), 4)},
                                          "yaw_err_deg": {"median": me.q(np.degrees(yaw_err), 0.5), "max": round(float(np.degrees(yaw_err.max())), 4)},
                                          "final_sigma_xy_m": cyc[-1]["pose_est"]["sigma_xy"]},
        "obstacle_points": {
            "logged": int(len(P_odom)), "positive": int(pos.sum()), "negative": int(neg.sum()),
            "positive_dist_to_true_surface_m": {"p50": me.q(dist[pos], 0.5), "p95": me.q(dist[pos], 0.95), "max": me.q(dist[pos], 1.0),
                                                "within_0.05_m": round(float((dist[pos] <= 0.05).mean()), 4)},
            "positive_nearest_surface": {names[i]: int(((nearest == i) & pos).sum()) for i in range(3)},
            "negative_dist_to_true_surface_m": {"p50": me.q(dist[neg], 0.5), "p95": me.q(dist[neg], 0.95)},
            "negative_nearest_surface": {names[i]: int(((nearest == i) & neg).sum()) for i in range(3)},
            "note": "rock distance = nearest rendered vertex (upper bound); terrain = vertical distance"},
        "capture_time_diagnostic": alt,
        "detections_log": {"records": len(det), "with_estimated_pose": sum(1 for d in det_ok if d.get("rover_pose_estimated")),
                           "with_gt_eval_only_key": sum(1 for d in det_ok if "gt_pose_eval_only" in d),
                           "legacy_ground_truth_key": sum(1 for d in det if "rover_pose_ground_truth" in d)},
    }
    (run / "r4_measure_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    _figure(run, Pw, kind, dist, man, pw, pg)
    return res


def _figure(run: Path, Pw, kind, dist, man, pw, pg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 4.6), dpi=110)
    for rk in man["rocks"]:
        ax.add_patch(plt.Circle((rk["x"], rk["y"]), rk["width_m"] / 2, fill=False, ec="#555", lw=0.8))
    for d in man["distractors"]:
        ax.add_patch(plt.Circle((d["x"], d["y"]), 0.3, fill=False, ec="#2a6fdb", lw=0.8, ls="--"))
    for cr in man["craters"]:
        ax.add_patch(plt.Circle((cr["x"], cr["y"]), cr["radius"], fill=False, ec="#b07a00", lw=0.8, ls=":"))
    p = kind == "p"
    sc = ax.scatter(Pw[p, 0], Pw[p, 1], s=1.2, c=np.minimum(dist[p], 0.15), cmap="viridis_r", vmin=0, vmax=0.15, lw=0)
    ax.scatter(Pw[kind == "n", 0], Pw[kind == "n", 1], s=1.2, c="#d62728", lw=0, label="negative hazard points")
    ax.plot(pg[:, 0], pg[:, 1], c="#999", lw=1, label="true path")
    ax.plot(pw[:, 0], pw[:, 1], c="k", lw=0.6, ls="--", label="estimated path")
    ax.set_aspect("equal"); ax.set_xlim(-1, 39); ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
    ax.set_title("R4.1 - obstacle points from depth + estimated pose (colour: distance to the true surface, m); grey circles = true rocks")
    fig.colorbar(sc, ax=ax, fraction=0.015, pad=0.01)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "r4_1_obstacle_points.png")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "r4_1_obstacle_points.png")
    plt.close(fig)


# ---- R4.2: occupancy grid ---------------------------------------------------------------------------------------------
COL = {0: (0.50, 0.50, 0.50), 1: (0.95, 0.94, 0.90), 2: (0.10, 0.10, 0.12), 3: (0.80, 0.15, 0.15)}


def _dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def _erode4(m: np.ndarray) -> np.ndarray:
    return m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)


def _in_view(objs: list, cyc: list, max_range: float = 4.0, half_fov: float = 0.5) -> dict:
    """Objects the camera had in view within ``max_range`` (edge distance) at some cycle (ground-truth poses; offline only)."""
    G = np.array([[c["gt_pose_eval_only"]["x"], c["gt_pose_eval_only"]["y"], c["gt_pose_eval_only"]["yaw"]] for c in cyc])
    cx, cy = G[:, 0] + 0.16 * np.cos(G[:, 2]), G[:, 1] + 0.16 * np.sin(G[:, 2])
    out = {}
    for o in objs:
        d = np.hypot(o["x"] - cx, o["y"] - cy) - o["width_m"] / 2
        b = np.arctan2(o["y"] - cy, o["x"] - cx) - G[:, 2]
        b = np.abs(np.arctan2(np.sin(b), np.cos(b)))
        ang = np.arcsin(np.clip((o["width_m"] / 2) / np.maximum(d + o["width_m"] / 2, 1e-6), 0, 1))   # angular half-size
        vis = (d <= max_range) & (b - ang <= half_fov)
        out[o["id"]] = {"in_view": bool(vis.any()), "closest_view_m": round(float(d[vis].min()), 2) if vis.any() else None}
    return out


def analyse_map(run: Path) -> dict:
    tour = run / "tour"
    mdir = tour / "mapping"
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    cyc = me.load_jsonl(mdir / "mapping.jsonl")
    al = me.alignment(cyc[0])
    snaps = me.load_snapshots(mdir / "snapshots")
    wg = me.WorldGrid()
    gt = me.ground_truth(man, wg)
    acc = me.accumulate(snaps, al, wg)
    haz = gt["pos"] | gt["neg"]
    m_haz = (acc == me.OCCUPIED) | (acc == me.NEGATIVE)
    m_free = acc == me.FREE
    ff = m_free & haz
    ff_core = m_free & _erode4(haz)
    owner = gt["owner"]
    view = _in_view(gt["objects"], cyc)

    per_obj = []
    for o in gt["objects"]:
        mine = owner == o["id"]
        if o["kind"] == "crater":
            near = np.hypot(*(np.array(wg.centres()) - np.array([o["x"], o["y"]])[:, None, None])) <= o["width_m"] / 2 * 1.6
            bowl, rim = gt["neg"] & near, gt["pos"] & near & ~(owner >= 0) | (gt["pos"] & mine)
            rec = {**o, **view[o["id"]],
                   "bowl_detected_negative": bool(((acc == me.NEGATIVE) & _dilate(bowl)).any()),
                   "bowl_cells_negative": int(((acc == me.NEGATIVE) & bowl).sum()),
                   "bowl_cells_unknown": int(((acc == me.UNKNOWN) & bowl).sum()),
                   "bowl_cells_free": int((m_free & bowl).sum()),
                   "rim_cells_occupied": int(((acc == me.OCCUPIED) & gt["pos"] & near).sum()),
                   "rim_cells_free": int((m_free & gt["pos"] & near).sum())}
        else:
            rec = {**o, **view[o["id"]], "detected": bool((m_haz & _dilate(mine)).any()) if mine.any() else None,
                   "false_free_cells": int((m_free & mine).sum()), "mapped_hazard_cells": int((m_haz & mine).sum())}
        per_obj.append(rec)

    def recall(sel):
        xs = [r for r in per_obj if r["kind"] != "crater" and r["in_view"] and sel(r)]
        return {"in_view_within_4m": len(xs), "detected": sum(1 for r in xs if r["detected"]),
                "recall": round(sum(1 for r in xs if r["detected"]) / len(xs), 4) if xs else None,
                "missed": [f"{r['name']} ({r['zone']}, h {r['height_m']} m)" for r in xs if not r["detected"]]}

    mp = summ.get("mapping", {})
    sim_t = cyc[-1]["stamp_s"] - cyc[0]["stamp_s"]
    res = {
        "label": "R4.2 occupancy grid - Webots course tour, YOLO disabled, estimated pose only; ground truth used offline here",
        "run": run.relative_to(ROOT).as_posix(),
        "inference_enabled": "inference" in summ,
        "tour": {"completed_moves": summ["final"]["completed_moves"], "planned_moves": summ["final"]["planned_moves"],
                 "controller_work_ms_per_step": summ["controller_work_ms_per_step"]},
        "mapping": {"cycles": mp.get("cycles"), "update_rate_hz_sim": round((len(cyc) - 1) / sim_t, 3) if sim_t > 0 else None,
                    "cycles_without_pose": mp.get("no_pose"), "cycles_without_depth": mp.get("no_depth"),
                    "timing_ms": mp.get("timing_ms"), "snapshots": mp.get("snapshots"), "window_recenters": mp.get("grid", {}).get("recenters"),
                    "grid_arrays_kb": round(mp.get("grid", {}).get("grid_bytes", 0) / 1024, 1),
                    "controller_process_memory_mb": mp.get("process_memory_mb")},
        "map_cells_accumulated": {"free": int(m_free.sum()), "occupied": int((acc == me.OCCUPIED).sum()),
                                  "negative": int((acc == me.NEGATIVE).sum()), "unknown": int((acc == me.UNKNOWN).sum())},
        "occupied_precision": {"strict": round(float((m_haz & haz).sum() / max(1, m_haz.sum())), 4),
                               "within_1_cell": round(float((m_haz & _dilate(haz)).sum() / max(1, m_haz.sum())), 4)},
        "free_precision": {"strict": round(float((m_free & ~haz).sum() / max(1, m_free.sum())), 4),
                           "excluding_cells_touching_a_hazard_edge": round(float((m_free & ~_erode4(haz)).sum() / max(1, m_free.sum())), 4)},
        "false_free_hazard_cells": {"total": int(ff.sum()), "positive": int((ff & gt["pos"]).sum()), "negative_crater_bowl": int((ff & gt["neg"]).sum()),
                                    "interior": int(ff_core.sum()),
                                    "by_object": {o["name"]: int((ff & (owner == o["id"])).sum()) for o in gt["objects"] if (ff & (owner == o["id"])).any()},
                                    "note": "strict: a FREE cell containing any part of a hazard >= 5 cm (often a boundary cell); 'interior' = all 4 neighbours are hazard too"},
        "free_over_low_rocks_cells": int((m_free & gt["low"]).sum()),
        "obstacle_recall": {"height_ge_0.10_m": recall(lambda r: r["height_m"] >= 0.10),
                            "height_0.05_to_0.10_m": recall(lambda r: 0.05 <= r["height_m"] < 0.10),
                            "height_lt_0.05_m_traversable_by_policy": recall(lambda r: r["height_m"] < 0.05)},
        "craters": [r for r in per_obj if r["kind"] == "crater"],
        "objects": per_obj,
        "alignment": me.best_shift(acc, haz),
        "recentering": me.recenter_consistency(snaps),
    }
    (run / "r4_2_map_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    _map_figures(run, snaps, al, wg, gt, man, acc, ff, per_obj, cyc)
    return res


def _rgb(state: np.ndarray) -> np.ndarray:
    img = np.zeros(state.shape + (3,))
    for k, c in COL.items():
        img[state == k] = c
    return img


def _outlines(ax, man, lw=0.6):
    import matplotlib.pyplot as plt
    for rk in man["rocks"]:
        ax.add_patch(plt.Circle((rk["x"], rk["y"]), rk["width_m"] / 2, fill=False, ec="#19a7c9", lw=lw))
    for d in man["distractors"]:
        ax.add_patch(plt.Circle((d["x"], d["y"]), 0.3, fill=False, ec="#2a6fdb", lw=lw, ls="--"))
    for cr in man["craters"]:
        ax.add_patch(plt.Circle((cr["x"], cr["y"]), cr["radius"], fill=False, ec="#e08a00", lw=lw, ls=":"))


def _map_figures(run, snaps, al, wg, gt, man, acc, ff, per_obj, cyc) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ext = [wg.x0, wg.x0 + wg.nx * wg.res, wg.y0, wg.y0 + wg.ny * wg.res]
    G = np.array([[c["gt_pose_eval_only"]["x"], c["gt_pose_eval_only"]["y"]] for c in cyc])
    E = me.to_world(np.array([[c["pose_est"]["x"], c["pose_est"]["y"]] for c in cyc]), al)
    legend = [Patch(color=COL[0], label="unknown"), Patch(color=COL[1], label="free"), Patch(color=COL[2], label="occupied"),
              Patch(color=COL[3], label="negative hazard"), Patch(fill=False, ec="#19a7c9", label="true rock"),
              Patch(fill=False, ec="#e08a00", ls=":", label="true crater")]
    # 1. rolling window sequence (what the rover holds in memory at that moment)
    live = [s for s in snaps if not s["final"]]
    pick = [live[int(round(k * (len(live) - 1) / 7))] for k in range(8)]
    fig, axs = plt.subplots(2, 4, figsize=(18, 9.4), dpi=100)
    for ax, s in zip(axs.ravel(), pick):
        h = s["hdr"]
        n, r = h["width"], h["resolution_m"]
        tx, ty, th = al
        x0, y0 = h["origin_xy"][0] + tx, h["origin_xy"][1] + ty          # start yaw is 0 in this course: pure translation
        ax.imshow(_rgb(s["state"]), origin="lower", extent=[x0, x0 + n * r, y0, y0 + n * r], interpolation="nearest")
        _outlines(ax, man)
        p = h["pose"]
        pw = me.to_world(np.array([[p["x"], p["y"]]]), al)[0]
        ax.arrow(pw[0], pw[1], 0.6 * np.cos(p["yaw"] + th), 0.6 * np.sin(p["yaw"] + th), width=0.08, color="#1db954")
        ax.set_xlim(x0, x0 + n * r); ax.set_ylim(y0, y0 + n * r); ax.set_aspect("equal")
        ax.set_title(f"t = {h['stamp_s']:.0f} s   window origin ({x0:.1f}, {y0:.1f})", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("R4.2 - the rover's live 12 x 12 m map at 8 moments of the tour (YOLO off; world frame via the start pose only)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for d in (run, FIG):
        fig.savefig(d / "r4_2_map_sequence.png")
    plt.close(fig)
    # 2. accumulated map growing over the tour
    fig, axs = plt.subplots(4, 1, figsize=(17, 11.5), dpi=100)
    for ax, frac in zip(axs, (0.25, 0.5, 0.75, 1.0)):
        k = max(1, int(round(frac * len(snaps))))
        a = me.accumulate(snaps, al, wg, upto=k)
        ax.imshow(_rgb(a), origin="lower", extent=ext, interpolation="nearest")
        _outlines(ax, man, 0.5)
        tt = snaps[k - 1]["hdr"]["stamp_s"]
        sel = np.array([c["stamp_s"] <= tt for c in cyc])
        ax.plot(E[sel, 0], E[sel, 1], c="#1db954", lw=0.8)
        ax.set_xlim(-1.5, 39.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        ax.set_title(f"after {tt:.0f} s of {snaps[-1]['hdr']['stamp_s']:.0f} s (green: estimated path)", fontsize=9)
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for d in (run, FIG):
        fig.savefig(d / "r4_2_map_progress.png")
    plt.close(fig)
    # 3. final map with errors
    fig, axs = plt.subplots(2, 1, figsize=(17, 7.6), dpi=110)
    axs[0].imshow(_rgb(acc), origin="lower", extent=ext, interpolation="nearest")
    _outlines(axs[0], man, 0.6)
    axs[0].plot(G[:, 0], G[:, 1], c="#1db954", lw=0.7)
    axs[0].set_title("final accumulated map (whole course) with the true objects")
    err = np.zeros(acc.shape + (3,)) + 0.97
    haz = gt["pos"] | gt["neg"]
    err[haz] = (0.75, 0.75, 0.75)
    err[((acc == me.OCCUPIED) | (acc == me.NEGATIVE)) & _dilate(haz)] = (0.1, 0.55, 0.2)
    err[((acc == me.OCCUPIED) | (acc == me.NEGATIVE)) & ~_dilate(haz)] = (1.0, 0.55, 0.0)
    err[ff] = (0.85, 0.0, 0.85)
    axs[1].imshow(err, origin="lower", extent=ext, interpolation="nearest")
    for r in per_obj:
        if r["kind"] != "crater" and r["in_view"] and not r["detected"]:
            if r["height_m"] >= me.H_OBS:
                axs[1].plot(r["x"], r["y"], "x", c="#d00000", ms=7, mew=1.6)              # a real miss
            else:
                axs[1].plot(r["x"], r["y"], "o", mfc="none", mec="#777", ms=6, mew=1.0)   # below 5 cm: traversable by policy
    axs[1].set_title("errors: green = mapped hazard on a true hazard (+-1 cell), orange = mapped hazard where none exists, "
                     "magenta = FREE cell on a true hazard, grey = true hazard never mapped,\n"
                     "red x = in-view object >= 5 cm missed, grey o = rock < 5 cm high (traversable by policy D3, not mapped by design)", fontsize=9)
    for ax in axs:
        ax.set_xlim(-1.5, 39.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for d in (run, FIG):
        fig.savefig(d / "r4_2_map_final.png")
    plt.close(fig)


# ---- R4.3: safety layers, faults, uncertainty --------------------------------------------------------------------------
DEPTH_FAULTS = {"depth_dropout": {"t0": 45.0, "t1": 55.0, "fraction": 0.3, "seed": 7},
                "depth_freeze": {"t0": 104.0, "t1": 106.0},
                "depth_invalid": {"t0": 160.0, "t1": 163.0, "value": "inf"},
                "cycle_overrun": {"t0": 210.0, "t1": 214.0, "ms": 60}}
SCENARIOS = {    # name -> (inference mode, ARTEMIS_FAULTS spec or None, extra)
    "baseline":          ("on", None, {}),
    "yolo_off_start":    ("absent", None, {}),                    # client enabled, no server ever
    "yolo_disconnect":   ("kill", None, {"kill_after_wall_s": 30.0}),
    "yolo_stale":        ("delay", None, {"delay_ms": 1500}),
    "depth_faults":      ("disabled", DEPTH_FAULTS, {}),          # YOLO off: depth hazards must still be reported
    "odo_scale_p3":      ("disabled", {"odometry": {"scale": 1.03}}, {}),
    "odo_scale_m3":      ("disabled", {"odometry": {"scale": 0.97}}, {}),
    "odo_heading":       ("disabled", {"odometry": {"yaw_scale": 1.03, "yaw_drift_per_m": 0.0005}}, {}),
    # R4.4 fusion scenarios
    "fusion":            ("on", None, {}),
    "depth_only":        ("disabled", None, {}),
    "fusion_delay400":   ("delay", None, {"delay_ms": 400, "realtime": True}),   # real time: 400 ms wall = 400 ms simulated
    "fusion_disconnect": ("kill", None, {"kill_after_wall_s": 30.0}),
    "fusion_profile":    ("on", None, {"env": {"ARTEMIS_PROFILE": "1"}}),   # hotspots only: profiling inflates timings
    # R4.5 display part: the 2 x 2 monitoring world (same scene, rover and tour as the course)
    "monitor":           ("on", None, {"world": "artemis_r4_monitor"}),                   # headless: timing with panels on
    "monitor_off":       ("on", None, {"world": "artemis_r4_monitor", "env": {"ARTEMIS_MONITOR": "0"}}),   # same, panels off
    "monitor_noyolo":    ("absent", None, {"world": "artemis_r4_monitor"}),               # YOLO missing: depth map goes on
    "monitor_reset":     ("on", None, {"world": "artemis_r4_monitor",                       # explicit reset mid-tour
                                       "env": {"ARTEMIS_MAP_RESET_AT": "150", "ARTEMIS_MONITOR_SAVE_AT": "149,152,170,230"}}),
    "monitor_gui":       ("on", None, {"world": "artemis_r4_monitor", "gui": True, "realtime": True,
                                       "screenshots_wall_s": (30, 120, 240, 360, 490),
                                       "env": {"ARTEMIS_MONITOR_SAVE_AT": "28,118,238,358,488"}}),
}
MONITOR_SCENARIOS = ["monitor", "monitor_off", "monitor_noyolo", "monitor_reset", "monitor_gui"]
# R5.2 live UDP telemetry (course world, YOLO on). The ground receiver runs in THIS process, independent of the controller.
TELEMETRY_PORT = 5791
SCENARIOS.update({
    "telemetry":            ("on", None, {"telemetry": "on"}),
    "telemetry_restart":    ("on", None, {"telemetry": "restart", "ground_down_wall_s": (15.0, 25.0)}),
    "telemetry_noreceiver": ("on", None, {"telemetry": "absent"}),
})
TELEMETRY_SCENARIOS = ["fusion", "telemetry", "telemetry_restart", "telemetry_noreceiver"]


def start_ground(out: Path, mode: str, env: dict) -> dict:
    """Enable controller telemetry and run a GroundReceiver thread ('on'), none ('absent'), or one that is stopped and
    restarted on the same port ('restart', wall-clock window from SCENARIOS). Output: ground.jsonl + ground_stats.json."""
    import threading
    from artemis_x.telemetry.transport import GroundReceiver
    env.update({"ARTEMIS_TELEMETRY": "1", "ARTEMIS_TELEMETRY_PORT": TELEMETRY_PORT})
    stats, stop, threads = [], threading.Event(), []
    if mode == "absent":
        return {"stop": lambda: (out / "ground_stats.json").write_text(json.dumps({"mode": mode, "receivers": []}), encoding="utf-8")}

    def serve(down: threading.Event):
        rx = GroundReceiver("127.0.0.1", TELEMETRY_PORT, out / "ground.jsonl", timeout_s=0.2)
        while not stop.is_set() and not down.is_set():
            rx.receive_once()
        stats.append(rx.close())

    first_down = threading.Event()
    threads.append(threading.Thread(target=serve, args=(first_down,), daemon=True))
    threads[0].start()
    timers = []
    if mode == "restart":
        a, b = SCENARIOS["telemetry_restart"][2]["ground_down_wall_s"]
        def restart():
            threads[0].join()
            th = threading.Thread(target=serve, args=(threading.Event(),), daemon=True)
            threads.append(th)
            th.start()
        timers = [threading.Timer(a, first_down.set), threading.Timer(b, restart)]
        for tm in timers:
            tm.start()

    def _stop():
        for tm in timers:
            tm.cancel()
        stop.set()
        for th in list(threads):
            th.join()
        (out / "ground_stats.json").write_text(json.dumps({"mode": mode, "receivers": stats}, indent=1), encoding="utf-8")
    return {"stop": _stop}


VALIDATE_MONITOR = ["monitor", "monitor_off", "monitor_noyolo", "monitor_reset"]   # headless (GUI evidence is from R4.5)


def validate(root: Path) -> dict:
    """R4.6: run the 8 safety, 4 fusion and 4 headless monitor scenarios back to back while sampling the Webots
    simulator's private memory (psutil, every 10 s), then the combined verdict (scripts/r4_validation.py)."""
    import threading
    import psutil
    samples, stop = [], threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.wait(10.0):
            mb = [p.memory_info().private / 2 ** 20 for p in psutil.process_iter(["name"])
                  if (p.info["name"] or "").lower().startswith("webots-bin")]
            samples.append({"wall_s": round(time.time() - t0, 1), "webots_mb": round(max(mb), 1) if mb else None})
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        for group, names in (("safety", SAFETY_SCENARIOS), ("fusion", FUSION_SCENARIOS), ("monitor", VALIDATE_MONITOR)):
            for name in names:
                run_scenario(root / group, name)
    finally:
        stop.set()
        th.join()
    (root / "webots_memory.json").write_text(json.dumps(samples), encoding="utf-8")
    from r4_validation import analyse_validation
    return analyse_validation(root, samples)


def screen_capture(path: Path, attempts: int = 5) -> bool:
    """Maximise and raise the Webots window, then save the primary screen (DPI-aware) as PNG - ONLY if the Webots window
    really is in the foreground at capture time (Windows may refuse to raise it while the user works in another window).
    Otherwise nothing is written: a capture must never contain another application. GUI evidence only."""
    ps = r"""
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public class U { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
  [DllImport("user32.dll")] public static extern int GetSystemMetrics(int i); }
'@
[U]::SetProcessDPIAware() | Out-Null
$w = Get-Process | Where-Object { $_.ProcessName -like 'webots*' -and $_.MainWindowHandle -ne 0 } | Sort-Object StartTime -Descending | Select-Object -First 1   # the run just started, not another open Webots
if (-not $w) { exit 2 }
$h = $w.MainWindowHandle
for ($i = 0; $i -lt __N__; $i++) {
  [U]::ShowWindow($h, 3) | Out-Null
  [U]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero); [U]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)   # Alt: allows the raise
  [U]::SetForegroundWindow($h) | Out-Null
  Start-Sleep -Milliseconds 1200
  if ([U]::GetForegroundWindow() -eq $h) {
    $sw = [U]::GetSystemMetrics(0); $sh = [U]::GetSystemMetrics(1)
    $b = New-Object System.Drawing.Bitmap $sw, $sh
    $g = [System.Drawing.Graphics]::FromImage($b); $g.CopyFromScreen(0, 0, 0, 0, $b.Size)
    if ([U]::GetForegroundWindow() -eq $h) { $b.Save('__OUT__', [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $b.Dispose(); exit 0 }
    $g.Dispose(); $b.Dispose()
  }
}
exit 3
""".replace("__OUT__", str(path)).replace("__N__", str(attempts))
    import subprocess
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    return r.returncode == 0 and path.exists()


FUSION_SCENARIOS = ["fusion", "depth_only", "fusion_delay400", "fusion_disconnect"]
SAFETY_SCENARIOS = ["baseline", "yolo_off_start", "yolo_disconnect", "yolo_stale", "depth_faults", "odo_scale_p3", "odo_scale_m3", "odo_heading"]


def run_scenario(root: Path, name: str) -> dict:
    """One headless course tour under a scenario. Inference faults are produced here (server absent / killed / slow);
    sensor and odometry faults by the controller's FaultInjector, enabled explicitly through ARTEMIS_FAULTS."""
    import threading
    mode, faults, extra = SCENARIOS[name]
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    env = {"ARTEMIS_WEBOTS_QUIT": "1", "ARTEMIS_WEBOTS_OUT": out / "tour", "ARTEMIS_INFER_TIMEOUT_S": "3", "ARTEMIS_INFER_PORT": PORT}
    if faults:
        env["ARTEMIS_FAULTS"] = json.dumps(faults)
    env.update(extra.get("env", {}))
    if mode == "disabled":
        env["ARTEMIS_INFERENCE"] = "0"
    srv, killer, killed = None, None, {}
    if mode in ("on", "kill", "delay"):
        srv = r3.start_server(delay_ms=extra.get("delay_ms", 0), port=PORT, log=out / "server.log")
    if mode == "kill":
        def _kill():
            killed["wall_s"] = time.time()
            srv.kill()
        killer = threading.Timer(extra["kill_after_wall_s"], _kill)
    shots = []
    ground = start_ground(out, extra["telemetry"], env) if extra.get("telemetry") else None
    if "ARTEMIS_MONITOR_SAVE_AT" in env:
        env["ARTEMIS_MONITOR_SAVE"] = out / "panels"
    if extra.get("gui"):
        def _shoot(ws):
            p = out / f"screen_{int(ws):04d}s.png"
            if screen_capture(p):
                shots.append(p.name)
        shooters = [threading.Timer(ws, _shoot, (ws,)) for ws in extra.get("screenshots_wall_s", ())]
    else:
        shooters = []
    t0 = time.time()
    try:
        if killer:
            killer.start()
        for s_ in shooters:
            s_.start()
        code, log = r3.run_webots(WB / "worlds" / f"{extra.get('world', 'artemis_test_course')}.wbt", env,
                                  timeout=900 if extra.get("realtime") else 600, mode="realtime" if extra.get("realtime") else "fast",
                                  gui=bool(extra.get("gui")))
    finally:
        for s_ in shooters:
            s_.cancel()
        if killer:
            killer.cancel()
        if srv is not None and srv.poll() is None:
            r3.stop_server(srv, PORT)
        if ground is not None:
            ground["stop"]()
    (out / "webots.log").write_text(log, encoding="utf-8")
    meta = {"scenario": name, "inference_mode": mode, "faults": faults, "extra": extra, "webots_exit": code,
            "wall_s": round(time.time() - t0, 1), "server_killed_wall_s": killed.get("wall_s"), "screenshots": shots}
    (out / "scenario.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{name}: exit {code} ({meta['wall_s']} s)", flush=True)
    return meta


def analyse_safety(root: Path) -> dict:
    from r4_safety_analysis import analyse_safety as _a
    return _a(root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("measure", "analyse", "map", "analyse-map", "safety-run", "safety", "analyse-safety", "fusion",
                                    "analyse-fusion", "monitor", "validate", "telemetry"))
    ap.add_argument("--only", nargs="*", help="safety-run / safety: subset of scenarios")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--no-inference", action="store_true")
    ap.add_argument("--inference", action="store_true", help="map: run with YOLO on (default for map: off)")
    a = ap.parse_args()
    if a.cmd == "measure":
        run = WB / "output" / f"r4-measure-{time.strftime('%Y%m%d-%H%M%S')}"
        tour = measure(run, inference=not a.no_inference)
        res = analyse(run)
        res["tour_stops"] = {k: tour.get(k) for k in ("max_abs_x_err_mm", "max_abs_yaw_err_deg", "min_clearance_along_true_path_m", "inference")}
    elif a.cmd == "map":                                   # R4.2: tour with YOLO disabled, then the grid evaluation
        run = WB / "output" / f"r4-map-{time.strftime('%Y%m%d-%H%M%S')}"
        tour = measure(run, inference=a.inference)
        res = analyse_map(run)
        res = {k: v for k, v in res.items() if k != "objects"}
    elif a.cmd in ("safety-run", "safety"):                # R4.3: all scenarios (about 2 min each), then the analysis
        root = WB / "output" / f"r4-safety-{time.strftime('%Y%m%d-%H%M%S')}"
        for name in (a.only or SAFETY_SCENARIOS):
            run_scenario(root, name)
        res = analyse_safety(root) if a.cmd == "safety" else {"dir": root.relative_to(ROOT).as_posix()}
    elif a.cmd == "fusion":                                # R4.4: fusion vs depth-only, delayed and disconnected inference
        root = WB / "output" / f"r4-fusion-{time.strftime('%Y%m%d-%H%M%S')}"
        for name in (a.only or FUSION_SCENARIOS):
            run_scenario(root, name)
        from r4_fusion_analysis import analyse_fusion
        res = analyse_fusion(root)
    elif a.cmd == "telemetry":                             # R5.2: Webots -> UDP -> ground receiver, live
        root = WB / "output" / f"r5-telemetry-{time.strftime('%Y%m%d-%H%M%S')}"
        for name in (a.only or TELEMETRY_SCENARIOS):
            run_scenario(root, name)
        from r5_telemetry_analysis import analyse_telemetry
        res = analyse_telemetry(root)
    elif a.cmd == "validate":                              # R4.6: safety + fusion + monitor on the same code, Webots memory sampled
        res = validate(WB / "output" / f"r4-validate-{time.strftime('%Y%m%d-%H%M%S')}")
    elif a.cmd == "monitor":                               # R4.5 display part: panels on/off timing, no YOLO, GUI screenshots
        root = WB / "output" / f"r4-monitor-{time.strftime('%Y%m%d-%H%M%S')}"
        for name in (a.only or MONITOR_SCENARIOS):
            run_scenario(root, name)
        from r4_monitor_analysis import analyse_monitor
        res = analyse_monitor(root)
    elif a.cmd == "analyse-fusion":
        from r4_fusion_analysis import analyse_fusion
        res = analyse_fusion(Path(a.path).resolve())
    elif a.cmd == "analyse-safety":
        res = analyse_safety(Path(a.path).resolve())
    elif a.cmd == "analyse-map":
        res = {k: v for k, v in analyse_map(Path(a.path).resolve()).items() if k != "objects"}
    else:
        res = analyse(Path(a.path).resolve())
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
