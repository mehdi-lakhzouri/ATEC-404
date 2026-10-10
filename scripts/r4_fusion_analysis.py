"""R4.4 offline analysis (ML environment): YOLO + depth fusion vs depth-only mapping, obstacle tracks vs ground truth,
latency, occlusion, delayed / disconnected inference, figures. Ground truth is used ONLY here.

Called by ``scripts/run_webots_r4.py fusion`` / ``analyse-fusion <dir>``; <dir> holds one sub-folder per scenario.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from artemis_x.evaluation import map_eval as me  # noqa: E402
import make_world  # noqa: E402  (map_eval puts tools/ on sys.path)
import r4_safety_analysis as sa  # noqa: E402

WB = ROOT / "artemis-webots"
FIG = ROOT / "docs" / "figures" / "webots"


def rock_footprints(man: dict, wg: me.WorldGrid) -> dict:
    """World cells under each rock (any vertex > 5 mm above the terrain) - also for rocks below the 5 cm threshold."""
    out = {}
    for k, rk in enumerate(man["rocks"]):
        V = np.array(make_world.ph_place(**rk["placement"])["vertices"])
        th = np.array([make_world.terrain_height(x, y) for x, y in V[:, :2]]) + 0.002
        i, j, ok = wg.ij(V[:, 0], V[:, 1])
        sel = ok & (V[:, 2] - th > 0.005)
        out[k] = (j[sel], i[sel])
    return out


def map_report(run: dict, man: dict, wg: me.WorldGrid, gt: dict, zone: np.ndarray, foot: dict) -> dict:
    snaps = sa.load_cost_snaps(run)
    al = me.alignment(run["cyc"][0])
    acc = me.accumulate(snaps, al, wg)
    cacc = sa.accumulate_cost(snaps, al, wg)
    haz = gt["pos"] | gt["neg"]
    m_haz = (acc == me.OCCUPIED) | (acc == me.NEGATIVE)
    view = {}
    G = run["tel"]["gt"]
    for o in gt["objects"]:
        if o["kind"] == "crater":
            continue
        d = np.hypot(o["x"] - G[:, 0], o["y"] - G[:, 1]) - o["width_m"] / 2
        b = np.abs(np.angle(np.exp(1j * (np.arctan2(o["y"] - G[:, 1], o["x"] - G[:, 0]) - G[:, 2]))))
        view[o["id"]] = bool(((d <= 4.0) & (b <= 0.5 + 0.3)).any())
    det = {}
    for o in gt["objects"]:
        if o["kind"] == "crater":
            continue
        if o["kind"] == "rock":
            jj, ii = foot[o["id"]]
            cells = np.zeros_like(haz); cells[jj, ii] = True
        else:
            cells = gt["owner"] == o["id"]
        det[o["id"]] = bool((m_haz & sa_dilate(cells)).any()) if cells.any() else False
    bins = {"h >= 0.10 m": lambda h: h >= 0.10, "0.05-0.10 m": lambda h: 0.05 <= h < 0.10, "< 0.05 m": lambda h: h < 0.05}
    rec = {}
    for name, f in bins.items():
        xs = [o for o in gt["objects"] if o["kind"] != "crater" and view.get(o["id"]) and f(o["height_m"])]
        rec[name] = {"in_view": len(xs), "mapped": sum(det[o["id"]] for o in xs),
                     "missed": [f"{o['name']} ({o['zone']}, h {o['height_m']})" for o in xs if not det[o["id"]]]}
    return {"recall": rec, "occupied_cells": int(m_haz.sum()),
            "occupied_precision_strict": round(float((m_haz & haz).sum() / max(1, m_haz.sum())), 4),
            "occupied_precision_incl_low_rocks": round(float((m_haz & (haz | sa_dilate(gt["low"]))).sum() / max(1, m_haz.sum())), 4),
            "false_free_hazard_cells": int(((acc == me.FREE) & haz).sum()),
            "cost_vs_truth": sa.cost_coverage(cacc, acc, zone, gt), "_acc": acc, "_cacc": cacc, "_al": al, "_det": det}


def sa_dilate(m: np.ndarray, r: int = 1) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def track_report(run: dict, man: dict, al: tuple) -> dict:
    fus = run["summ"].get("mapping", {}).get("fusion")
    if not fus:
        return {}
    tracks = [t for t in fus["tracks_final"] if t["status"] != "tentative"]
    objs = [{"name": r["name"], "zone": r["zone"], "kind": "rock", "x": r["x"], "y": r["y"], "r": r["width_m"] / 2} for r in man["rocks"]]
    objs += [{"name": d["name"], "zone": d["zone"], "kind": "engineered object", "x": d["x"], "y": d["y"], "r": 0.3} for d in man["distractors"]]
    O = np.array([[o["x"], o["y"], o["r"]] for o in objs])
    rows, by_obj = [], {}
    for t in tracks:
        w = me.to_world(np.array([[t["x"], t["y"]]]), al)[0]
        d = np.hypot(O[:, 0] - w[0], O[:, 1] - w[1])
        k = int(np.argmin(d - O[:, 2]))
        matched = d[k] <= O[k, 2] + 0.35
        cr = min(abs(math.hypot(w[0] - c["x"], w[1] - c["y"]) - c["radius"]) for c in man["craters"])
        rows.append({**t, "world": [round(float(w[0]), 3), round(float(w[1]), 3)], "matched": objs[k]["name"] if matched else None,
                     "kind": objs[k]["kind"] if matched else ("crater rim" if cr < 0.35 else "terrain"),
                     "centre_error_m": round(float(d[k]), 3) if matched else None})
        if matched:
            by_obj.setdefault(objs[k]["name"], []).append(t["id"])
    m = [r for r in rows if r["matched"] and r["kind"] == "rock"]
    err = [r["centre_error_m"] for r in m]
    dup = {k: v for k, v in by_obj.items() if len(v) > 1}
    by_status = {s: sum(1 for t in tracks if t["status"] == s) for s in ("confirmed", "lost")}
    occl = [r for r in rows if r["matched"] == "crs_rock_14"]
    return {"tracks_confirmed_or_lost": len(tracks), "by_status": by_status, "tentative_left": sum(1 for t in fus["tracks_final"] if t["status"] == "tentative"),
            "matched_to_rocks": len(m), "rocks_tracked": len({r["matched"] for r in m}),
            "matched_to_engineered_objects_labelled_rock": sum(1 for r in rows if r["kind"] == "engineered object"),
            "unmatched_tracks": {k: sum(1 for r in rows if not r["matched"] and r["kind"] == k) for k in ("crater rim", "terrain")},
            "centre_error_m": {"median": sa.q(err, 0.5), "p95": sa.q(err, 0.95), "max": round(max(err), 3) if err else None},
            "duplicates": {"objects_with_more_than_one_track": len(dup), "rate": round(len(dup) / max(1, len(by_obj)), 4), "detail": dup},
            "occluded_rock_14": [{k: r[k] for k in ("id", "centre_error_m", "hits", "bimodal_hits", "status")} for r in occl],
            "fusion_stats": {k: fus.get(k) for k in ("results", "detections", "measurements", "rejected", "bimodal", "small_rock_hits", "footprint_hits", "latency_s", "fuse_ms")},
            "tracker_stats": fus["tracker"], "_rows": rows}


def analyse_fusion(root: Path) -> dict:
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    wg = me.WorldGrid()
    gt = me.ground_truth(man, wg)
    zone = sa.gt_collision_zone(gt, wg, sa.CFG.collision_radius_m)
    foot = rock_footprints(man, wg)
    runs = {d.name: sa.load_run(d) for d in sorted(root.iterdir()) if (d / "tour" / "summary.json").exists()}
    out = {"label": "R4.4 YOLO + depth fusion - Webots course tour; ground truth used offline here only",
           "root": root.relative_to(ROOT).as_posix(), "scenarios": {}}
    full = {}
    for n, run in runs.items():
        mp = run["summ"]["mapping"]
        mr = map_report(run, man, wg, gt, zone, foot)
        tr = track_report(run, man, mr["_al"])
        t, S, R = sa.reason_series(run["cyc"])
        res = {"scenario": n, "inference_mode": run["meta"].get("inference_mode"),
               "tour": {"completed_moves": run["summ"]["final"]["completed_moves"], "planned_moves": run["summ"]["final"]["planned_moves"]},
               "inference": run["summ"].get("inference", {}) and {k: run["summ"]["inference"].get(k) for k in ("submitted", "ok", "failed", "stale", "evicted", "latency_ms_p50", "latency_ms_p95")},
               "timing_ms": {k: mp["timing_ms"].get(k) for k in ("scan_ms", "grid_ms", "cost_ms", "cycle_ms", "step_max_ms")},
               "health_time_in_state_s": mp["health"]["time_in_state_s"], "reasons": {r: round(float(v.mean()), 4) for r, v in R.items()},
               "map": {k: v for k, v in mr.items() if not k.startswith("_")}, "tracks": {k: v for k, v in tr.items() if not k.startswith("_")}}
        if run["meta"].get("server_killed_wall_s"):
            tk = sa.wall_to_sim(run["cyc"], run["meta"]["server_killed_wall_s"])
            res["disconnect"] = {"kill_sim_s": tk, **sa.detect(t, R, "inference_unavailable", tk, t[-1]),
                                 "confirmed_tracks_at_kill": next((c["tracks"]["confirmed"] for c in run["cyc"] if c["stamp_s"] >= tk), None),
                                 "confirmed_tracks_at_end": run["cyc"][-1]["tracks"]["confirmed"],
                                 "cycles_after_kill": sum(1 for c in run["cyc"] if c["stamp_s"] > tk)}
        out["scenarios"][n] = res
        full[n] = {"mr": mr, "tr": tr, "run": run}
    (root / "r4_4_fusion_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figures(root, full, man, wg)
    return out


def figures(root: Path, full: dict, man: dict, wg: me.WorldGrid) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    FIG.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(root / name); fig.savefig(FIG / name); plt.close(fig)
    from run_webots_r4 import _rgb
    ext = [wg.x0, wg.x0 + wg.nx * wg.res, wg.y0, wg.y0 + wg.ny * wg.res]
    if "fusion" in full:
        f = full["fusion"]
        rows = f["tr"]["_rows"]
        fig, axs = plt.subplots(2, 1, figsize=(17, 8.2), dpi=110)
        for ax, key, title in ((axs[0], "depth_only", "depth only (R4.2/R4.3)"), (axs[1], "fusion", "YOLO + depth fusion (R4.4) with confirmed tracks")):
            if key not in full:
                continue
            ax.imshow(_rgb(full[key]["mr"]["_acc"]), origin="lower", extent=ext, interpolation="nearest")
            sa._outlines(ax, man, 0.6)
            ax.set_title(title, fontsize=10); ax.set_xlim(-1.5, 39.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        for r in rows:
            col = {"rock": "#2ecc71", "engineered object": "#e67e22", "crater rim": "#9b59b6", "terrain": "#e74c3c"}[r["kind"]]
            axs[1].add_patch(plt.Circle(tuple(r["world"]), max(0.05, r["radius_m"]), fill=False, ec=col, lw=1.4))
            axs[1].text(r["world"][0] + 0.08, r["world"][1] + 0.08, str(r["id"]), fontsize=5.5, color=col)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(fill=False, ec="#2ecc71", label="track on a true rock"), Patch(fill=False, ec="#e67e22", label="track on an engineered object (label error)"),
                            Patch(fill=False, ec="#9b59b6", label="track on a crater rim"), Patch(fill=False, ec="#e74c3c", label="track on bare terrain (false)"),
                            Patch(fill=False, ec="#19a7c9", label="true rock")], loc="lower center", ncol=5, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, "r4_4_tracks_and_maps.png")
        # zoom on the small-rock zone
        fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), dpi=110)
        for ax, key in zip(axs, ("depth_only", "fusion")):
            if key not in full:
                continue
            ax.imshow(_rgb(full[key]["mr"]["_acc"]), origin="lower", extent=ext, interpolation="nearest")
            sa._outlines(ax, man, 1.0)
            for rk in man["rocks"]:
                if rk["zone"] == "Z2_small_scatter":
                    ax.text(rk["x"] + 0.12, rk["y"] + 0.08, rk["name"][-2:], fontsize=7, color="#19a7c9")
            if key == "fusion":
                for r in rows:
                    ax.add_patch(plt.Circle(tuple(r["world"]), max(0.05, r["radius_m"]), fill=False, ec="#2ecc71" if r["kind"] == "rock" else "#e74c3c", lw=1.4))
            ax.set_xlim(5.3, 8.0); ax.set_ylim(-1.7, 2.0); ax.set_aspect("equal"); ax.set_title(f"Z2 small rocks - {key.replace('_', ' ')}", fontsize=10)
        fig.tight_layout()
        save(fig, "r4_4_small_rocks_zoom.png")
        # annotated camera frames at the observation stops (detections + track ids from the controller's own log)
        tour = f["run"]["dir"] / "tour"
        dets = {d["frame_id"]: d for d in me.load_jsonl(tour / "inference" / "detections.jsonl") if d.get("detections") is not None}
        frames = sorted(tour.glob("frames/*_rgb.png"), key=lambda p: int(p.name.split("_")[0][5:]))
        tiles = []
        for p in frames:
            idx = int(p.name.split("_")[0][5:])
            cand = [k for k in dets if abs(k - idx) <= 2 and dets[k].get("association", "").startswith("same capture")]
            im = Image.open(p).convert("RGB"); dr = ImageDraw.Draw(im)
            if cand:
                rec = dets[min(cand, key=lambda k: abs(k - idx))]
                for d in rec["detections"]:
                    x0, y0, x1, y1 = d["xyxy"]
                    tid = d.get("track_id")
                    dr.rectangle([x0, y0, x1, y1], outline=(255, 0, 255) if tid else (255, 160, 0), width=2)
                    lbl = (f"T{tid} " if tid else "") + f"{d['conf']:.2f}" + (f" {d['range_m']:.1f}m" if d.get("valid") else "")
                    dr.text((x0 + 2, max(0, y0 - 11)), lbl, fill=(255, 255, 0))
            dr.rectangle([0, 0, 719, 14], fill=(0, 0, 0)); dr.text((3, 2), p.stem.split("_", 1)[1].replace("_rgb", ""), fill=(255, 255, 255))
            tiles.append(im.resize((480, 320)))
        if tiles:
            cols = 3; rws = (len(tiles) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 480, rws * 320), (20, 20, 20))
            for k, t in enumerate(tiles):
                sheet.paste(t, ((k % cols) * 480, (k // cols) * 320))
            for d in (root, FIG):
                sheet.save(d / "r4_4_annotated_stops.jpg", quality=88)
