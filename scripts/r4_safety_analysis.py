"""R4.3 offline analysis (ML environment): health / fault timelines, corridor decisions vs ground truth, odometry
uncertainty (measured vs model), cost-layer coverage of true hazards, figures. Ground truth is used ONLY here.

Called by ``scripts/run_webots_r4.py analyse-safety <dir>`` (or ``safety``), where <dir> holds one sub-folder per scenario.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.evaluation import map_eval as me  # noqa: E402
from artemis_x.sim.mapping.config import MappingConfig  # noqa: E402
from artemis_x.sim.mapping.uncertainty import UncertaintyModel  # noqa: E402

WB = ROOT / "artemis-webots"
FIG = ROOT / "docs" / "figures" / "webots"
CFG = MappingConfig()
EXPECTED = {"depth_dropout": "depth_degraded", "depth_freeze": "depth_frozen", "depth_invalid": "depth_invalid",
            "cycle_overrun": "cycle_overrun"}


def q(a, p):
    a = np.asarray([x for x in a if x is not None], float)
    return round(float(np.quantile(a, p)), 3) if len(a) else None


def load_run(d: Path) -> dict:
    tour = d / "tour"
    cyc = me.load_jsonl(tour / "mapping" / "mapping.jsonl")
    summ = json.loads((tour / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((tour / "telemetry.csv").open(encoding="utf-8")))
    tel = {"t": np.array([float(r["sim_time_s"]) for r in rows]),
           "est": np.array([[float(r["odo_x"]), float(r["odo_y"]), float(r["odo_yaw"])] for r in rows]),
           "gt": np.array([[float(r["gt_x"]), float(r["gt_y"]), float(r["gt_yaw"])] for r in rows])}
    meta = json.loads((d / "scenario.json").read_text(encoding="utf-8")) if (d / "scenario.json").exists() else {"scenario": d.name}
    return {"dir": d, "cyc": cyc, "summ": summ, "tel": tel, "meta": meta}


# ---- health ------------------------------------------------------------------------------------------------------------
def reason_series(cyc: list) -> tuple:
    t = np.array([c["stamp_s"] for c in cyc])
    names = sorted({r for c in cyc for r in c["health"]["reasons"]})
    R = {r: np.array([r in c["health"]["reasons"] for c in cyc]) for r in names}
    S = np.array([c["health"]["state"] for c in cyc])
    return t, S, R


def detect(t, R, reason, t0, t1) -> dict:
    """Latency from fault onset t0 to the reason being active; recovery from fault end t1 to the reason clearing."""
    act = R.get(reason, np.zeros(len(t), bool))
    on = np.nonzero(act & (t >= t0 - 1e-9))[0]
    if not len(on) or t[on[0]] > t1 + 2.0:
        return {"reason": reason, "detected": False}
    first = on[0]
    after = np.nonzero(~act & (t >= t1 - 1e-9))[0]
    return {"reason": reason, "detected": True, "latency_s": round(float(t[first] - t0), 3),
            "recovery_s": round(float(t[after[0]] - t1), 3) if len(after) else None,
            "active_before_fault": bool(act[(t < t0) & (t > t0 - 2.0)].any())}


def wall_to_sim(cyc: list, wall: float) -> float | None:
    W = np.array([c["wall_s"] for c in cyc]); T = np.array([c["stamp_s"] for c in cyc])
    if wall < W[0] or wall > W[-1]:
        return None
    return float(np.interp(wall, W, T))


# ---- ground truth corridor -----------------------------------------------------------------------------------------
def gt_collision_zone(gt: dict, wg: me.WorldGrid, radius: float) -> np.ndarray:
    """Cells whose CENTRE lies within ``radius`` of any true hazard cell (positive or negative)."""
    haz = gt["pos"] | gt["neg"]
    r = int(math.ceil(radius / wg.res))
    out = np.zeros_like(haz)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if math.hypot(dx, dy) * wg.res <= radius + 1e-9:
                out |= np.roll(np.roll(haz, dy, 0), dx, 1)
    return out


def corridor_truth(run: dict, zone: np.ndarray, wg: me.WorldGrid) -> dict:
    """Per cycle: does the TRUE 1 m corridor ahead of the TRUE pose cross the true collision zone? Compared with the
    advisory decision (obstacle / unobserved / stale reasons). Dangerous = truth blocked while the system says clear."""
    tp = fp = fn = tn = dangerous = 0
    for c in run["cyc"]:
        g = c.get("gt_pose_eval_only")
        if not g:
            continue
        ts = np.arange(0, CFG.corridor_length_m + 1e-9, CFG.corridor_step_m)
        xs, ys = g["x"] + np.cos(g["yaw"]) * ts, g["y"] + np.sin(g["yaw"]) * ts
        i, j, ok = wg.ij(xs, ys)
        blocked = bool((zone[j[ok], i[ok]]).any())
        rs = c["health"]["reasons"]
        said_obs = "obstacle_in_corridor" in rs or "obstacle_in_footprint" in rs
        said_clear = not (said_obs or "unobserved_corridor" in rs or "stale_corridor" in rs)
        tp += blocked and said_obs; fn += blocked and not said_obs; fp += (not blocked) and said_obs; tn += (not blocked) and not said_obs
        dangerous += blocked and said_clear
    n = tp + fp + fn + tn
    return {"cycles": n, "true_blocked": tp + fn, "obstacle_reason_tp": tp, "obstacle_reason_fn": fn, "obstacle_reason_fp": fp,
            "dangerous_clear_while_blocked": dangerous,
            "obstacle_reason_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "obstacle_reason_precision": round(tp / (tp + fp), 4) if tp + fp else None}


# ---- uncertainty ---------------------------------------------------------------------------------------------------
def relative_errors(run: dict) -> list:
    """Measured relative odometry error between cycle pairs (est vs ground truth, each in its own start frame) next to
    the model's sigma_rel for the same distance / turn / lever arm."""
    cyc = [c for c in run["cyc"] if c.get("gt_pose_eval_only")]
    m = UncertaintyModel(CFG)
    # the estimate AT the capture instant (pose_est), the same instant as the logged ground truth; pose_now is one step later
    E = np.array([[c["pose_est"]["x"], c["pose_est"]["y"], c["pose_est"]["yaw"]] for c in cyc])
    G = np.array([[c["gt_pose_eval_only"]["x"], c["gt_pose_eval_only"]["y"], c["gt_pose_eval_only"]["yaw"]] for c in cyc])
    T = np.array([c["stamp_s"] for c in cyc])
    tel = run["tel"]
    # cumulative distance / turn from the ESTIMATE (what the model sees)
    d_est = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(tel["est"][:, :2], axis=0).T))])
    a_est = np.concatenate([[0], np.cumsum(np.abs(np.angle(np.exp(1j * np.diff(tel["est"][:, 2])))))])
    D = np.interp(T, tel["t"], d_est); A = np.interp(T, tel["t"], a_est)
    out = []
    for i in range(0, len(cyc), 8):
        for j in range(i + 1, len(cyc), 4):
            dD = D[j] - D[i]
            if dD > 10.5:
                break
            def rel(P, a, b):
                c, s = math.cos(P[a, 2]), math.sin(P[a, 2])
                dx, dy = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1]
                return np.array([c * dx + s * dy, -s * dx + c * dy])
            re, rg = rel(E, i, j), rel(G, i, j)
            err = float(np.linalg.norm(re - rg))
            rho = float(np.linalg.norm(rg))
            out.append((dD, A[j] - A[i], rho, err, m.sigma(dD, A[j] - A[i], rho)))
    return out


def rel_summary(rel: list) -> dict:
    if not rel:
        return {}
    R = np.array(rel)
    bins = [(0, 1), (1, 2), (2, 4), (4, 7), (7, 10.5)]
    res = {}
    for a, b in bins:
        s = (R[:, 0] >= a) & (R[:, 0] < b)
        if s.any():
            res[f"{a}-{b} m"] = {"pairs": int(s.sum()), "measured_err_p50_m": round(float(np.median(R[s, 3])), 4),
                                 "measured_err_p95_m": round(float(np.quantile(R[s, 3], 0.95)), 4),
                                 "model_sigma_p50_m": round(float(np.median(R[s, 4])), 4),
                                 "covered_by_2sigma": round(float((R[s, 3] <= 2 * R[s, 4] + 1e-6).mean()), 4)}
    return res


# ---- cost layer vs truth -------------------------------------------------------------------------------------------
def accumulate_cost(snaps: list, al: tuple, wg: me.WorldGrid) -> np.ndarray:
    acc = np.full((wg.ny, wg.nx), 255, np.uint8)
    for s in snaps:
        i, j, ok = me.snapshot_to_world(s, al, wg)
        seen = (s["obs_age_ds"].ravel() != 65535) & ok
        acc[j[seen], i[seen]] = s["cost"].ravel()[seen]
    return acc


def load_cost_snaps(run: dict) -> list:
    snaps = me.load_snapshots(run["dir"] / "tour" / "mapping" / "snapshots")
    for s in snaps:
        h = s["hdr"]
        blob = (run["dir"] / "tour" / "mapping" / "snapshots" / h["binary"]).read_bytes()
        lay = next(L for L in h["layers"] if L["name"] == "cost")
        s["cost"] = np.frombuffer(blob, np.uint8, count=lay["bytes"], offset=lay["offset"]).reshape(h["height"], h["width"])
    return snaps


def cost_coverage(cost_acc: np.ndarray, acc: np.ndarray, zone: np.ndarray, gt: dict) -> dict:
    """Cells inside the TRUE collision zone (centre within 0.25 m of a true hazard) that the cost layer would let a rover
    centre occupy (cost 0-252, i.e. observed FREE and not inflated enough). 0 is the safety goal."""
    observed = acc != me.UNKNOWN
    unsafe = zone & observed & (cost_acc < 253)
    core = zone & observed & ~(gt["pos"] | gt["neg"])
    owner = gt["owner"]
    names = {o["id"]: o["name"] for o in gt["objects"]}
    by_obj: dict = {}
    for j, i in zip(*np.nonzero(unsafe)):                  # attribute each unsafe cell to the nearest owned hazard (<= 3 cells)
        win = owner[max(0, j - 3):j + 4, max(0, i - 3):i + 4]
        ids = win[win >= 0]
        key = names.get(int(np.bincount(ids).argmax()), "?") if ids.size else "terrain (rim / bowl edge)"
        by_obj[key] = by_obj.get(key, 0) + 1
    in_hazard = unsafe & (gt["pos"] | gt["neg"])
    return {"collision_zone_cells_observed": int((zone & observed).sum()), "unsafe_cells": int(unsafe.sum()),
            "unsafe_cells_inside_hazard": int(in_hazard.sum()), "unsafe_by_object": dict(sorted(by_obj.items(), key=lambda kv: -kv[1])),
            "unsafe_fraction": round(float(unsafe.sum() / max(1, (zone & observed).sum())), 4),
            "inscribed_or_lethal_cells": int((cost_acc >= 253)[observed & (cost_acc != 255)].sum()),
            "free_cells_cost0": int(((cost_acc == 0) & observed).sum()),
            "graded_cells": int(((cost_acc > 0) & (cost_acc < 253)).sum()), "zone_cells_not_hazard": int(core.sum())}


def local_consistency(run: dict, snaps: list, zone: np.ndarray, wg: me.WorldGrid, radius_m: float = 3.0, every: int = 3) -> dict:
    """What safety actually needs: the map RELATIVE TO THE ROVER. For each snapshot the map is placed with the rover's
    TRUE pose at that instant (ground truth, offline), and cells within ``radius_m`` of the rover that lie in the true
    collision zone are checked. 'unsafe' = observed and cost < 253. The same check with the uncertainty margin removed
    (mapped hazards dilated by the collision radius only) shows what the 2 sigma_rel inflation adds."""
    tel = run["tel"]
    r_col = CFG.collision_radius_m + CFG.resolution_m * math.sqrt(2) / 2
    rc = int(math.ceil(r_col / CFG.resolution_m))
    disc = [(dx, dy) for dy in range(-rc, rc + 1) for dx in range(-rc, rc + 1) if max(0.0, math.hypot(dx, dy) * CFG.resolution_m - CFG.resolution_m * math.sqrt(2) / 2) <= CFG.collision_radius_m]
    tot = {"snapshots": 0, "zone_cells_observed": 0, "unsafe_with_uncertainty_margin": 0, "unsafe_without_uncertainty_margin": 0}
    for s in [x for x in snaps if not x["final"]][::every]:
        h = s["hdr"]
        k = int(np.clip(np.searchsorted(tel["t"], h["stamp_s"]), 0, len(tel["t"]) - 1))
        g, e = tel["gt"][k], h["pose"]
        th = g[2] - e["yaw"]
        al = (g[0] - (math.cos(th) * e["x"] - math.sin(th) * e["y"]), g[1] - (math.sin(th) * e["x"] + math.cos(th) * e["y"]), th)
        i, j, ok = me.snapshot_to_world(s, al, wg)
        n = h["width"]
        st, cost = s["state"].ravel(), s["cost"].ravel()
        hz = ((st == me.OCCUPIED) | (st == me.NEGATIVE)).reshape(n, n)
        nomargin = np.zeros_like(hz)
        for dx, dy in disc:
            nomargin |= np.roll(np.roll(hz, dy, 0), dx, 1)
        rr, cc = np.divmod(np.arange(n * n), n)
        xo = h["origin_xy"][0] + (cc + 0.5) * h["resolution_m"]; yo = h["origin_xy"][1] + (rr + 0.5) * h["resolution_m"]
        near = (np.hypot(xo - e["x"], yo - e["y"]) <= radius_m) & ok
        inz = near.copy()
        inz[near] = zone[j[near], i[near]]
        free_obs = (st == me.FREE)
        tot["snapshots"] += 1
        tot["zone_cells_observed"] += int((inz & (st != me.UNKNOWN)).sum())
        tot["unsafe_with_uncertainty_margin"] += int((inz & (st != me.UNKNOWN) & (cost < 253)).sum())
        tot["unsafe_without_uncertainty_margin"] += int((inz & free_obs & ~nomargin.ravel()).sum())
    z = max(1, tot["zone_cells_observed"])
    tot["unsafe_rate_with_margin"] = round(tot["unsafe_with_uncertainty_margin"] / z, 4)
    tot["unsafe_rate_without_margin"] = round(tot["unsafe_without_uncertainty_margin"] / z, 4)
    return tot


# ---- per scenario --------------------------------------------------------------------------------------------------
def analyse_run(run: dict, man: dict, wg: me.WorldGrid, gt: dict, zone: np.ndarray) -> dict:
    cyc, summ, meta = run["cyc"], run["summ"], run["meta"]
    mp = summ.get("mapping", {})
    t, S, R = reason_series(cyc)
    res = {"scenario": meta.get("scenario"), "inference_mode": meta.get("inference_mode"), "faults": meta.get("faults"),
           "tour": {"completed_moves": summ["final"]["completed_moves"], "planned_moves": summ["final"]["planned_moves"]},
           "cycles": len(cyc), "frozen_frames_rejected": mp.get("frozen_frames_rejected"),
           "timing_ms": {k: mp.get("timing_ms", {}).get(k) for k in ("scan_ms", "grid_ms", "cost_ms", "health_ms", "cycle_ms", "step_max_ms")},
           "memory_mb": mp.get("process_memory_mb", {}).get("at_end"), "costmap": mp.get("costmap"),
           "health": {"time_in_state_s": mp.get("health", {}).get("time_in_state_s"), "transitions": mp.get("health", {}).get("transitions"),
                      "reason_active_fraction": {r: round(float(v.mean()), 4) for r, v in R.items()}}}
    # true path clearance (ground truth, offline)
    import make_world  # noqa: E402  (already on path through map_eval)
    obs = make_world.course_obstacles(man["rocks"], man["distractors"])
    G = run["tel"]["gt"]
    O = np.array(obs)
    sub = G[::4]
    dmin = np.min(np.hypot(sub[:, None, 0] - O[None, :, 0], sub[:, None, 1] - O[None, :, 1]) - O[None, :, 2])
    res["true_path_min_clearance_m"] = round(float(dmin), 3)
    # odometry: final absolute error
    al = me.alignment(cyc[0])
    E = np.array([[c["pose_now"]["x"], c["pose_now"]["y"]] for c in cyc]); Gc = np.array([[c["gt_pose_eval_only"]["x"], c["gt_pose_eval_only"]["y"]] for c in cyc])
    Ew = me.to_world(E, al)
    res["odometry_abs_err_m"] = {"final": round(float(np.hypot(*(Ew[-1] - Gc[-1]))), 3), "max": round(float(np.max(np.hypot(*(Ew - Gc).T))), 3)}
    res["sigma_abs_final_m"] = cyc[-1]["pose_now"]["sigma_xy"]
    # faults
    det = []
    for name, f in (meta.get("faults") or {}).items():
        if name in EXPECTED:
            det.append({"fault": name, "window_s": [f["t0"], f["t1"]], **detect(t, R, EXPECTED[name], f["t0"], f["t1"])})
    if meta.get("inference_mode") == "absent":
        det.append({"fault": "yolo_unavailable_from_start", **detect(t, R, "inference_unavailable", t[0], t[-1])})
    if meta.get("inference_mode") == "kill" and meta.get("server_killed_wall_s"):
        tk = wall_to_sim(cyc, meta["server_killed_wall_s"])
        det.append({"fault": "yolo_disconnect", "kill_sim_s": None if tk is None else round(tk, 3),
                    **(detect(t, R, "inference_unavailable", tk, t[-1]) if tk is not None else {"detected": False})})
    if meta.get("inference_mode") == "delay":
        det.append({"fault": "yolo_stale_results_1500ms", **detect(t, R, "inference_stale", t[0], t[-1])})
    res["fault_detection"] = det
    # map / cost integrity vs truth
    snaps = load_cost_snaps(run)
    acc = me.accumulate(snaps, al, wg)
    cacc = accumulate_cost(snaps, al, wg)
    haz = gt["pos"] | gt["neg"]
    res["map"] = {"false_free_hazard_cells": int(((acc == me.FREE) & haz).sum()),
                  "occupied_precision_strict": round(float((((acc == me.OCCUPIED) | (acc == me.NEGATIVE)) & haz).sum()
                                                           / max(1, ((acc == me.OCCUPIED) | (acc == me.NEGATIVE)).sum())), 4),
                  "alignment": me.best_shift(acc, haz)}
    res["cost_vs_truth"] = cost_coverage(cacc, acc, zone, gt)
    res["local_consistency_3m"] = local_consistency(run, snaps, zone, wg)
    res["corridor_vs_truth"] = corridor_truth(run, zone, wg)
    res["_acc"], res["_cacc"], res["_snaps"], res["_al"] = acc, cacc, snaps, al
    return res


def analyse_safety(root: Path) -> dict:
    man = json.loads((WB / "worlds" / "artemis_test_course.json").read_text(encoding="utf-8"))
    wg = me.WorldGrid()
    gt = me.ground_truth(man, wg)
    zone = gt_collision_zone(gt, wg, CFG.collision_radius_m)
    runs = {d.name: load_run(d) for d in sorted(root.iterdir()) if (d / "tour" / "summary.json").exists()}
    out = {"label": "R4.3 safety layers - Webots course tour scenarios; ground truth used offline here only",
           "root": root.relative_to(ROOT).as_posix(), "scenarios": {}}
    full = {}
    for name, run in runs.items():
        r = analyse_run(run, man, wg, gt, zone)
        full[name] = r
        out["scenarios"][name] = {k: v for k, v in r.items() if not k.startswith("_")}
    for name in ("baseline", "odo_scale_p3", "odo_scale_m3", "odo_heading"):
        if name in runs:
            out["scenarios"][name]["relative_odometry_error"] = rel_summary(relative_errors(runs[name]))
    (root / "r4_3_safety_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figures(root, runs, full, man, wg, gt)
    return out


# ---- figures -------------------------------------------------------------------------------------------------------
def _cost_rgb(c: np.ndarray) -> np.ndarray:
    img = np.zeros(c.shape + (3,))
    g = (c > 0) & (c < 253)
    img[c == 0] = (0.95, 0.94, 0.90)
    img[g] = np.stack([np.full(g.sum(), 1.0), 1.0 - c[g] / 300.0, 0.55 - c[g] / 520.0], 1)
    img[c == 253] = (0.98, 0.45, 0.0)
    img[c == 254] = (0.10, 0.10, 0.12)
    img[c == 255] = (0.50, 0.50, 0.50)
    return img


def _outlines(ax, man, lw=0.7):
    import matplotlib.pyplot as plt
    for rk in man["rocks"]:
        ax.add_patch(plt.Circle((rk["x"], rk["y"]), rk["width_m"] / 2, fill=False, ec="#19a7c9", lw=lw))
    for d in man["distractors"]:
        ax.add_patch(plt.Circle((d["x"], d["y"]), 0.3, fill=False, ec="#2a6fdb", lw=lw, ls="--"))
    for cr in man["craters"]:
        ax.add_patch(plt.Circle((cr["x"], cr["y"]), cr["radius"], fill=False, ec="#9b59b6", lw=lw, ls=":"))


def figures(root: Path, runs: dict, full: dict, man: dict, wg: me.WorldGrid, gt: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Wedge, Circle
    FIG.mkdir(parents=True, exist_ok=True)
    ext = [wg.x0, wg.x0 + wg.nx * wg.res, wg.y0, wg.y0 + wg.ny * wg.res]
    leg = [Patch(color=(0.95, 0.94, 0.90), label="cost 0 (free)"), Patch(color=(1.0, 0.75, 0.4), label="1-252 graded"),
           Patch(color=(0.98, 0.45, 0.0), label="253 inscribed"), Patch(color=(0.1, 0.1, 0.12), label="254 lethal"),
           Patch(color=(0.5, 0.5, 0.5), label="255 unknown"), Patch(fill=False, ec="#19a7c9", label="true rock"),
           Patch(fill=False, ec="#9b59b6", ls=":", label="true crater")]

    def save(fig, name):
        fig.savefig(root / name); fig.savefig(FIG / name); plt.close(fig)

    if "baseline" in full:
        b = full["baseline"]
        from run_webots_r4 import _rgb  # occupancy colours of R4.2
        # 1-2: occupancy vs costmap, whole course
        fig, axs = plt.subplots(2, 1, figsize=(17, 7.6), dpi=110)
        axs[0].imshow(_rgb(b["_acc"]), origin="lower", extent=ext, interpolation="nearest"); axs[0].set_title("occupancy grid (R4.2) - before inflation")
        axs[1].imshow(_cost_rgb(b["_cacc"]), origin="lower", extent=ext, interpolation="nearest")
        axs[1].set_title("cost layer (R4.3) - collision radius 0.25 m (+ 2 sigma_rel), safety band 0.15 m; unknown stays 255")
        for ax in axs:
            _outlines(ax, man); ax.set_xlim(-1.5, 39.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.legend(handles=leg, loc="lower center", ncol=7, fontsize=8); fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, "r4_3_occupancy_vs_cost.png")
        # 3-4: zooms (boulders, crater)
        zooms = [("Z3 boulder (1.4 m)", 10.0, 0.4), ("Z8 boulder field", 31.0, 0.6), ("Z6 boulder + shadowed rocks", 20.3, 0.4),
                 ("Z7 craters (bowl = lethal negative hazard)", 24.8, 0.0)]
        fig, axs = plt.subplots(1, 4, figsize=(18, 5.2), dpi=110)
        for ax, (title, cx, cy) in zip(axs, zooms):
            ax.imshow(_cost_rgb(b["_cacc"]), origin="lower", extent=ext, interpolation="nearest")
            _outlines(ax, man, 1.0)
            ax.set_xlim(cx - 2.2, cx + 2.2); ax.set_ylim(cy - 2.2, cy + 2.2); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
        fig.legend(handles=leg, loc="lower center", ncol=7, fontsize=8); fig.tight_layout(rect=(0, 0.07, 1, 1))
        save(fig, "r4_3_cost_zooms.png")
        # 5: unknown regions and the blind zone around the rover (live window at three moments)
        snaps = [s for s in b["_snaps"] if not s["final"]]
        picks = [snaps[2], snaps[int(len(snaps) * 0.37)], snaps[int(len(snaps) * 0.62)]]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), dpi=110)
        al = b["_al"]
        for ax, s in zip(axs, picks):
            h = s["hdr"]; n, r = h["width"], h["resolution_m"]
            x0, y0 = h["origin_xy"][0] + al[0], h["origin_xy"][1] + al[1]
            ax.imshow(_cost_rgb(s["cost"]), origin="lower", extent=[x0, x0 + n * r, y0, y0 + n * r], interpolation="nearest")
            p = h["pose"]; pw = me.to_world(np.array([[p["x"], p["y"]]]), al)[0]; yaw = p["yaw"] + al[2]
            ax.add_patch(Wedge((pw[0] + 0.16 * math.cos(yaw), pw[1] + 0.16 * math.sin(yaw)), 6.0, math.degrees(yaw) - 28.6, math.degrees(yaw) + 28.6,
                               fill=False, ec="#1db954", lw=1.0, ls="--"))
            ax.add_patch(Wedge((pw[0], pw[1]), 0.64, math.degrees(yaw) - 28.6, math.degrees(yaw) + 28.6, width=0.45, fc=(0.9, 0.1, 0.4, 0.35), ec="none"))
            ax.add_patch(Circle((pw[0], pw[1]), CFG.footprint_radius_measured_m, fill=False, ec="#00a0ff", lw=1.4))
            ax.plot([pw[0], pw[0] + CFG.corridor_length_m * math.cos(yaw)], [pw[1], pw[1] + CFG.corridor_length_m * math.sin(yaw)], c="#00a0ff", lw=2)
            hh = h.get("health") or {}
            ax.set_title(f"t = {h['stamp_s']:.0f} s   health {hh.get('state')}\n{', '.join(hh.get('reasons', []))}", fontsize=8)
            ax.set_xlim(pw[0] - 3, pw[0] + 4); ax.set_ylim(pw[1] - 3, pw[1] + 4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.legend(handles=leg[:5] + [Patch(fc=(0.9, 0.1, 0.4, 0.35), label="camera blind zone (0.19-0.64 m)"),
                                      Patch(fill=False, ec="#1db954", ls="--", label="camera FOV"),
                                      Patch(fill=False, ec="#00a0ff", label="footprint r 0.21 m + 1 m corridor")],
                   loc="lower center", ncol=8, fontsize=8)
        fig.suptitle("live cost window around the rover: UNKNOWN (grey) is never traversable; the blind zone is only known if seen earlier")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        save(fig, "r4_3_unknown_blind_zone.png")
    # 6: odometry uncertainty - costmap and measured vs predicted error
    odo = [n for n in ("baseline", "odo_scale_p3", "odo_scale_m3", "odo_heading") if n in full]
    if len(odo) > 1:
        fig, axs = plt.subplots(len(odo), 1, figsize=(17, 2.9 * len(odo)), dpi=100)
        for ax, n in zip(np.atleast_1d(axs), odo):
            r = full[n]
            ax.imshow(_cost_rgb(r["_cacc"]), origin="lower", extent=ext, interpolation="nearest")
            _outlines(ax, man, 0.6)
            cv = r["cost_vs_truth"]
            ax.set_title(f"{n}: unsafe cells (true collision zone mapped < 253) = {cv['unsafe_cells']}, "
                         f"final odometry error {r['odometry_abs_err_m']['final']} m, map best shift {r['map']['alignment']['best_shift_cells_xy']} cells", fontsize=9)
            ax.set_xlim(-1.5, 39.5); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        fig.tight_layout()
        save(fig, "r4_3_cost_under_odometry_faults.png")
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
        cols = {"baseline": "#2c3e50", "odo_scale_p3": "#e67e22", "odo_scale_m3": "#16a085", "odo_heading": "#8e44ad"}
        for n in odo:
            Rr = np.array(relative_errors(runs[n]))
            if len(Rr):
                ax.scatter(Rr[:, 0], Rr[:, 3], s=3, alpha=0.35, c=cols[n], label=f"{n}: measured relative error")
        Rb = np.array(relative_errors(runs[odo[0]]))
        ax.scatter(Rb[:, 0], 2 * Rb[:, 4], s=2, c="#bbbbbb", alpha=0.25, zorder=0, label="model 2 sigma_rel per pair (depends on turns and lever arm)")
        ax.set_xlabel("distance driven between the two poses (m)"); ax.set_ylabel("relative position error (m)"); ax.set_ylim(0, 0.8)
        ax.set_title("odometry: measured relative error (ground truth, offline) vs the model's 2-sigma envelope")
        ax.legend(fontsize=7, markerscale=3); fig.tight_layout()
        save(fig, "r4_3_uncertainty_measured_vs_model.png")
    # 7: health timelines
    tl = [n for n in ("depth_faults", "yolo_disconnect", "yolo_off_start", "yolo_stale", "baseline") if n in runs]
    if tl:
        fig, axs = plt.subplots(len(tl), 1, figsize=(17, 2.6 * len(tl)), dpi=100, sharex=True)
        colors = {"OK": "#2ecc71", "DEGRADED": "#f1c40f", "STOP": "#e74c3c"}
        for ax, n in zip(np.atleast_1d(axs), tl):
            t, S, R = reason_series(runs[n]["cyc"])
            for st, c in colors.items():
                ax.fill_between(t, -1.2, -0.4, where=S == st, color=c, step="post", lw=0)
            names = sorted(R)
            for k, rname in enumerate(names):
                ax.fill_between(t, k, k + 0.8, where=R[rname], color="#34495e", step="post", lw=0)
            ax.set_yticks([-0.8] + [k + 0.4 for k in range(len(names))]); ax.set_yticklabels(["state"] + names, fontsize=7)
            meta = runs[n]["meta"]
            for f, spec in (meta.get("faults") or {}).items():
                if "t0" in spec:
                    ax.axvspan(spec["t0"], spec["t1"], color="#3498db", alpha=0.25)
                    ax.text(spec["t0"], len(names) + 0.1, f, fontsize=7, color="#2471a3")
            if meta.get("server_killed_wall_s"):
                tk = wall_to_sim(runs[n]["cyc"], meta["server_killed_wall_s"])
                if tk is not None:
                    ax.axvline(tk, color="#3498db", lw=1.5); ax.text(tk, len(names) + 0.1, "YOLO server killed", fontsize=7, color="#2471a3")
            ax.set_title(f"{n}: health (green OK / yellow DEGRADED / red STOP) and active reasons", fontsize=9)
        np.atleast_1d(axs)[-1].set_xlabel("simulation time (s)")
        fig.tight_layout()
        save(fig, "r4_3_health_timelines.png")
