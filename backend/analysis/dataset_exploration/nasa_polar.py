"""A3 - NASA POLAR stereo analysis (images, exposure brackets, calibration, LiDAR ground truth)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .common import LOG, ErrorLog, Paths, describe, save_fig, tone_map_12bit

TERRAINS = ("Terrain01_LargeOnly", "Terrain06_Smooth", "Terrain11_FreshCrater")
PATH_RE = re.compile(
    r"Terrain(?P<terrain>\d{2})_(?P<tname>[^/]+)/Pos(?P<pos>[A-Z])_(?P<dist>\d+)_L(?P<light>on|off)/"
    r"(?P<sun>NoSun|Sun_(?P<az>\d+))/Cam(?P<cam>[LR])_(?P<ts>\d+)_(?P<exp>\d+)\.png$"
)
THUMB = (121, 76)  # (w, h) thumbnail for similarity analysis
SAT = 4090        # 12-bit effective range; values >= SAT treated as clipped


def parse_path(rel: str) -> dict[str, Any] | None:
    m = PATH_RE.search(rel)
    if not m:
        return None
    g = m.groupdict()
    return {
        "terrain": g["terrain"], "terrain_name": g["tname"], "pos": g["pos"], "pos_number": int(g["dist"]),
        "light": g["light"], "sun": "no" if g["sun"] == "NoSun" else g["az"], "cam": g["cam"],
        "timestamp": int(g["ts"]), "exposure": g["exp"], "exposure_ms": int(g["exp"]),
    }


def analyse_image(path: str) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path}
    try:
        with Image.open(path) as im:
            out.update(mode=im.mode, size_wh=list(im.size), comment=str(im.info.get("Comment", ""))[:200])
            a = np.asarray(im)
    except Exception as e:
        out["error"] = repr(e)
        return out
    out["dtype"] = str(a.dtype)
    out["channels"] = 1 if a.ndim == 2 else a.shape[2]
    a = a.astype(np.float32)
    p = np.percentile(a, [1, 50, 99])
    out.update(
        min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
        p01=float(p[0]), p50=float(p[1]), p99=float(p[2]),
        sat_frac=float((a >= SAT).mean()),
        well_exposed_frac=float(((a > 600) & (a < 3800)).mean()),
        uses_low_4_bits=bool((a.astype(np.uint16) & 0xF).any()),
    )
    t = cv2.resize(a, THUMB, interpolation=cv2.INTER_AREA)
    out["thumb"] = t
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def parse_calibration(p: Path) -> dict[str, Any]:
    kv = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = [float(x) for x in v.split()]
    K_l = np.array(kv["CAMERA_MATRIX_LEFT"]).reshape(3, 3)
    K_r = np.array(kv["CAMERA_MATRIX_RIGHT"]).reshape(3, 3)
    R = np.array(kv["ROTATION_MATRIX"]).reshape(3, 3)
    T = np.array(kv["TRANSLATION_VECTOR"])
    Rl = np.array(kv["CAM_LIDAR_ROTATION"]).reshape(3, 3)
    Tl = np.array(kv["CAM_LIDAR_TRANSLATION"])
    w, h = (int(x) for x in kv["IMAGE_WIDTH_HEIGHT"])
    ang = lambda M: float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))  # noqa: E731
    return {
        "keys": sorted(kv),
        "image_wh": [w, h],
        "K_left": K_l, "K_right": K_r,
        "dist_left_k1k2p1p2": kv["DISTORTION_COEFFICIENTS_LEFT"],
        "dist_right_k1k2p1p2": kv["DISTORTION_COEFFICIENTS_RIGHT"],
        "R_stereo": R, "T_stereo": T, "R_cam_lidar": Rl, "T_cam_lidar": Tl,
        "derived": {
            "baseline_m_assuming_metres": float(np.linalg.norm(T)),
            "fx_fy_left": [K_l[0, 0], K_l[1, 1]], "cx_cy_left": [K_l[0, 2], K_l[1, 2]],
            "hfov_deg_left": float(np.degrees(2 * np.arctan(w / 2 / K_l[0, 0]))),
            "vfov_deg_left": float(np.degrees(2 * np.arctan(h / 2 / K_l[1, 1]))),
            "stereo_rotation_angle_deg": ang(R),
            "stereo_R_orthonormality_err": float(np.abs(R @ R.T - np.eye(3)).max()),
            "cam_lidar_R_orthonormality_err": float(np.abs(Rl @ Rl.T - np.eye(3)).max()),
            "cam_lidar_R_det": float(np.linalg.det(Rl)),
            "disparity_px_at_depth_m": {str(z): float(K_l[0, 0] * np.linalg.norm(T) / z) for z in (2, 4, 6, 8, 10)},
        },
    }


def _load_xyz(p: Path) -> np.ndarray:
    df = pd.read_csv(p, sep=r"\s+", header=None, engine="c", dtype=np.float64)
    return df.to_numpy()


def analyse_xyz(p: Path, calib: dict[str, Any], errors: ErrorLog) -> tuple[dict[str, Any], np.ndarray | None]:
    try:
        a = _load_xyz(p)
    except Exception as e:
        errors.add("nasa_polar.xyz", p.as_posix(), repr(e))
        return {"readable": False}, None
    out: dict[str, Any] = {"readable": True, "rows": int(a.shape[0]), "columns": int(a.shape[1]),
                           "nan_rows": int(np.isnan(a).any(axis=1).sum())}
    names = ["c0_int", "c1_int", "x", "y", "z", "c5_int"][: a.shape[1]]
    out["column_stats"] = {n: {"min": a[:, i].min(), "max": a[:, i].max(), "mean": a[:, i].mean(),
                               "integer_valued": bool(np.all(a[:, i] == np.round(a[:, i])))}
                           for i, n in enumerate(names)}
    if a.shape[1] >= 5:
        xyz = a[:, 2:5]
        out["range_m_from_origin"] = describe(np.linalg.norm(xyz[:: 50], axis=1))
        # roughness evidence: residual std after least-squares plane fit z = ax + by + c (subsampled)
        s = xyz[:: 20]
        A = np.c_[s[:, 0], s[:, 1], np.ones(len(s))]
        # the LiDAR "up" axis is unknown a priori; fit along the axis of least variance
        cov = np.cov((s - s.mean(0)).T)
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]
        resid = (s - s.mean(0)) @ normal
        out["plane_fit"] = {
            "normal": normal, "explained_var_smallest_axis": float(evals[0] / evals.sum()),
            "residual_std_m": float(resid.std()), "residual_abs_p99_m": float(np.percentile(np.abs(resid), 99)),
            "residual_abs_p999_m": float(np.percentile(np.abs(resid), 99.9)),
        }
        del A
        # Which direction is CAM_LIDAR_*? test both against the left intrinsics.
        K = calib["K_left"]; w, h = calib["image_wh"]
        Rl, Tl = calib["R_cam_lidar"], calib["T_cam_lidar"]
        sub = xyz[:: 25]
        cand = {"lidar_to_cam: Xc = R X + t": sub @ Rl.T + Tl, "cam_to_lidar inverted: Xc = R^T (X - t)": (sub - Tl) @ Rl}
        tests = {}
        for name, pc in cand.items():
            z = pc[:, 2]
            front = z > 0.1
            u = K[0, 0] * pc[front, 0] / z[front] + K[0, 2]
            v = K[1, 1] * pc[front, 1] / z[front] + K[1, 2]
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            tests[name] = {"front_frac": float(front.mean()), "in_image_frac": float(inside.sum() / len(pc)),
                           "depth_m_in_image": describe(z[front][inside])}
        out["projection_test"] = tests
        out["projection_selected"] = max(tests, key=lambda k: tests[k]["in_image_frac"])
        return out, a
    return out, None


def run(paths: Paths, errors: ErrorLog, workers: int) -> dict[str, Any]:
    root = paths.nasa_polar
    LOG.info("A3 NASA POLAR: %s", root)
    res: dict[str, Any] = {"root": paths.rel(root), "terrains": {}}
    files = sorted(p for t in TERRAINS for p in (root / t).rglob("*.png"))
    meta = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        m = parse_path(rel)
        if m is None:
            errors.add("nasa_polar.parse", rel, "path does not follow Terrain/Pos/Sun/Cam convention")
            continue
        m["rel"] = rel
        meta.append(m)
    res["unparsed_paths"] = len(files) - len(meta)
    LOG.info("decoding %d NASA images", len(meta))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats = list(ex.map(analyse_image, [str(root / m["rel"]) for m in meta], chunksize=8))
    for m, s in zip(meta, stats):
        if "error" in s:
            errors.add("nasa_polar.decode", m["rel"], s["error"])
        m.update({k: v for k, v in s.items() if k != "path"})

    res["image_properties"] = {
        "total_images": len(meta),
        "decode_failures": [m["rel"] for m in meta if "error" in m],
        "mode_size_dtype": dict(Counter(f"{m.get('mode')}|{m.get('size_wh')}|{m.get('dtype')}|ch={m.get('channels')}" for m in meta)),
        "global_max_value": max(m["max"] for m in meta if "max" in m),
        "global_min_value": min(m["min"] for m in meta if "min" in m),
        "effective_bit_depth": int(np.ceil(np.log2(max(m["max"] for m in meta if "max" in m) + 1))),
        "images_using_low_4_bits": sum(m.get("uses_low_4_bits", False) for m in meta),
        "png_comment_examples": sorted({m.get("comment", "") for m in meta})[:5],
    }
    calib = {}
    for t in TERRAINS:
        cp = root / t / "stereo.calibration"
        if cp.exists():
            calib[t] = parse_calibration(cp)
        else:
            errors.add("nasa_polar.calibration", t, "stereo.calibration missing")

    by_terrain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in meta:
        by_terrain[f"Terrain{m['terrain']}_{m['terrain_name']}"].append(m)

    xyz_cache: dict[str, dict[str, np.ndarray]] = {}
    for t in TERRAINS:
        ms = by_terrain[t]
        info: dict[str, Any] = {"images": len(ms)}
        info["by_position"] = dict(Counter(f"Pos{m['pos']}_{m['pos_number']}" for m in ms))
        info["by_light"] = dict(Counter(m["light"] for m in ms))
        info["by_sun"] = dict(Counter(m["sun"] for m in ms))
        info["by_camera"] = dict(Counter(m["cam"] for m in ms))
        info["by_exposure_ms"] = dict(sorted(Counter(m["exposure"] for m in ms).items()))
        cond = defaultdict(set)
        for m in ms:
            cond[f"Pos{m['pos']}_L{m['light']}"].add(m["sun"])
        info["acquisition_folders"] = {k: sorted(v) for k, v in sorted(cond.items())}
        expected = {f"Pos{p}_L{l}" for p in "ABC" for l in ("on", "off")}
        info["missing_position_light_folders"] = sorted(expected - set(cond))
        # stereo pairing: same folder, same timestamp+exposure, L and R
        grp = defaultdict(dict)
        for m in ms:
            grp[(m["pos"], m["light"], m["sun"], m["timestamp"], m["exposure"])][m["cam"]] = m
        info["stereo_pairs_complete"] = sum(1 for g in grp.values() if set(g) == {"L", "R"})
        info["stereo_unpaired"] = [next(iter(g.values()))["rel"] for g in grp.values() if set(g) != {"L", "R"}]
        bracket = defaultdict(list)
        for m in ms:
            bracket[(m["pos"], m["light"], m["sun"], m["cam"])].append(m)
        info["brackets"] = len(bracket)
        info["bracket_sizes"] = dict(Counter(len(v) for v in bracket.values()))
        dup_exp = [k for k, v in bracket.items() if len({m["exposure"] for m in v}) != len(v)]
        info["brackets_with_repeated_exposure"] = [list(k) for k in dup_exp]
        info["distinct_camera_viewpoints"] = len({(m["pos"], m["cam"]) for m in ms})
        info["distinct_scene_lighting_states"] = len({(m["pos"], m["light"], m["sun"]) for m in ms})
        # exposure behaviour
        exp_tab = {}
        for e in sorted({m["exposure"] for m in ms}):
            sel = [m for m in ms if m["exposure"] == e and "p50" in m]
            exp_tab[e] = {"n": len(sel), "median_p50": float(np.median([m["p50"] for m in sel])),
                          "median_sat_frac": float(np.median([m["sat_frac"] for m in sel])),
                          "median_well_exposed_frac": float(np.median([m["well_exposed_frac"] for m in sel]))}
        info["exposure_response"] = exp_tab
        light_tab = {}
        for key in sorted({(m["light"], m["sun"]) for m in ms}):
            sel = [m for m in ms if (m["light"], m["sun"]) == key and "p50" in m]
            light_tab[f"L{key[0]}_sun{key[1]}"] = {"n": len(sel), "median_p50": float(np.median([m["p50"] for m in sel])),
                                                  "median_sat_frac": float(np.median([m["sat_frac"] for m in sel]))}
        info["lighting_response"] = light_tab
        # best exposure per bracket = maximum well-exposed fraction
        best = Counter()
        for v in bracket.values():
            b = max((m for m in v if "well_exposed_frac" in m), key=lambda m: (m["well_exposed_frac"], -m["exposure_ms"]), default=None)
            if b:
                best[b["exposure"]] += 1
        info["best_exposure_histogram"] = dict(sorted(best.items()))
        # redundancy: similarity of thumbnails (log domain, per-image standardised)
        def th(m):
            return np.log1p(np.maximum(m["thumb"] - m["p01"], 0))
        adj, lr, sun = [], [], []
        for v in bracket.values():
            v = sorted((m for m in v if "thumb" in m), key=lambda m: m["exposure_ms"])
            adj += [_corr(th(a), th(b)) for a, b in zip(v, v[1:]) if min(a["well_exposed_frac"], b["well_exposed_frac"]) > 0.2]
        for g in grp.values():
            if set(g) == {"L", "R"} and "thumb" in g["L"] and g["L"]["well_exposed_frac"] > 0.2:
                lr.append(_corr(th(g["L"]), th(g["R"])))
        by_view = defaultdict(list)
        for m in ms:
            if "thumb" in m:
                by_view[(m["pos"], m["cam"], m["exposure"], m["light"])].append(m)
        for v in by_view.values():
            v = sorted(v, key=lambda m: m["sun"])
            sun += [_corr(th(a), th(b)) for a, b in zip(v, v[1:]) if min(a["well_exposed_frac"], b["well_exposed_frac"]) > 0.2]
        info["redundancy_thumbnail_correlation"] = {
            "adjacent_exposures_same_bracket": describe(adj),
            "left_vs_right_same_shot": describe(lr),
            "different_sun_azimuth_same_view_exposure": describe(sun),
            "note": "correlation of 121x76 log thumbnails; pairs with <20% well-exposed pixels excluded",
        }
        # calibration + ground truth
        if t in calib:
            c = calib[t]
            info["calibration"] = c
            info["calibration_image_size_matches_images"] = all(m.get("size_wh") == c["image_wh"] for m in ms)
        gt = {}
        xyz_cache[t] = {}
        for gp in sorted((root / t / "GroundTruth_ascii").glob("*.xyz")):
            LOG.info("reading %s", gp.name)
            g, arr = analyse_xyz(gp, calib[t], errors)
            gt[gp.name] = g
            if arr is not None:
                xyz_cache[t][gp.stem] = arr[:: 25, 2:5]
        info["ground_truth_xyz"] = gt
        info["ground_truth_positions_vs_image_positions"] = {
            "xyz_files": sorted(p.stem for p in (root / t / "GroundTruth_ascii").glob("*.xyz")),
            "image_positions": sorted({f"Pos{m['pos']}" for m in ms}),
        }
        res["terrains"][t] = info

    if len(calib) > 1:
        ref = calib[TERRAINS[0]]
        res["calibration_differences_vs_terrain01"] = {
            t: {k: bool(np.allclose(np.asarray(ref[k]), np.asarray(c[k]))) for k in
                ("K_left", "K_right", "R_stereo", "T_stereo", "R_cam_lidar", "T_cam_lidar")}
            for t, c in calib.items() if t != TERRAINS[0]
        }
    inverted = {(t, p[:-4]): g["projection_selected"].startswith("cam_to_lidar")
                for t, info in res["terrains"].items() for p, g in info["ground_truth_xyz"].items() if g.get("readable")}
    res["figures"], res["samples"] = _figures(paths, root, meta, by_terrain, calib, xyz_cache, inverted)
    res["image_index"] = [
        {k: m.get(k) for k in ("rel", "terrain", "pos", "light", "sun", "cam", "timestamp", "exposure",
                               "p50", "sat_frac", "well_exposed_frac")}
        for m in meta
    ]
    return res


def _best_image(ms: list[dict[str, Any]], **sel: Any) -> dict[str, Any] | None:
    c = [m for m in ms if all(m[k] == v for k, v in sel.items()) and "well_exposed_frac" in m]
    return max(c, key=lambda m: (m["well_exposed_frac"], -m["exposure_ms"]), default=None)


def _figures(paths, root, meta, by_terrain, calib, xyz_cache, inverted) -> tuple[list[str], list[str]]:
    F, S = paths.figures, paths.samples
    figs, samples = [], []
    colors = {"Terrain01_LargeOnly": "#4C72B0", "Terrain06_Smooth": "#55A868", "Terrain11_FreshCrater": "#C44E52"}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for t, ms in by_terrain.items():
        exps = sorted({m["exposure_ms"] for m in ms})
        ax[0].plot(exps, [np.median([m["p50"] for m in ms if m["exposure_ms"] == e]) for e in exps], "o-", label=t, color=colors.get(t))
        ax[1].plot(exps, [np.median([m["sat_frac"] for m in ms if m["exposure_ms"] == e]) for e in exps], "o-", label=t, color=colors.get(t))
        ax[2].plot(exps, [np.median([m["well_exposed_frac"] for m in ms if m["exposure_ms"] == e]) for e in exps], "o-", label=t, color=colors.get(t))
    for a, ttl in zip(ax, ("Median pixel value (12-bit)", "Saturated fraction (>=4090)", "Well-exposed fraction (600-3800)")):
        a.set_xscale("log", base=2); a.set_xlabel("exposure (ms, from filename)"); a.set_title(ttl); a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    figs.append(save_fig(fig, F / "nasa_exposure_response.png"))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for a, (t, ms) in zip(ax, sorted(by_terrain.items())):
        keys = sorted({(m["light"], m["sun"]) for m in ms})
        vals = [[m["p50"] for m in ms if (m["light"], m["sun"]) == k and m["exposure"] in ("0128", "0256")] for k in keys]
        a.boxplot(vals, showfliers=False); a.set_xticks(range(1, len(keys) + 1), [f"L{l}\nsun {s}" for l, s in keys], fontsize=8)
        a.set_title(f"{t}\nmedian pixel @128-256 ms", fontsize=10)
    figs.append(save_fig(fig, F / "nasa_lighting_conditions.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    labels, bottoms = [], {}
    conds = sorted({f"Pos{m['pos']}_L{m['light']}" for m in meta})
    terr = sorted(by_terrain)
    bottom = np.zeros(len(terr))
    for c in conds:
        v = np.array([sum(1 for m in by_terrain[t] if f"Pos{m['pos']}_L{m['light']}" == c) for t in terr])
        ax.bar(terr, v, bottom=bottom, label=c); bottom += v
    ax.set(title="NASA POLAR images per terrain by position / rover light", ylabel="images"); ax.legend(fontsize=8, ncol=3)
    figs.append(save_fig(fig, F / "nasa_image_counts.png"))

    # samples per terrain: each position, best-exposed left image, two lighting states
    for t, ms in sorted(by_terrain.items()):
        poss = sorted({m["pos"] for m in ms})
        fig, ax = plt.subplots(len(poss), 2, figsize=(14, 4.4 * len(poss)))
        ax = np.atleast_2d(ax)
        for r, p in enumerate(poss):
            for c, (light, sun) in enumerate([("off", "180"), ("on", "no")]):
                m = _best_image(ms, pos=p, light=light, sun=sun, cam="L") or _best_image(ms, pos=p, cam="L")
                a = np.asarray(Image.open(root / m["rel"]))
                ax[r, c].imshow(tone_map_12bit(a), cmap="gray")
                ax[r, c].set_title(f"{m['rel']}\nexp {m['exposure']} ms, well-exposed {m['well_exposed_frac']:.2f}", fontsize=8)
                ax[r, c].axis("off")
        fig.suptitle(f"{t} - left camera, display-stretched (0.5-99.5 percentile)")
        samples.append(save_fig(fig, S / f"nasa_{t}.png"))

    # one complete exposure bracket
    t = "Terrain11_FreshCrater"
    ms = sorted((m for m in by_terrain.get(t, []) if m["pos"] == "A" and m["light"] == "off" and m["sun"] == "180" and m["cam"] == "L"),
                key=lambda m: m["exposure_ms"])
    if ms:
        fig, ax = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 2.6))
        for a, m in zip(np.atleast_1d(ax), ms):
            a.imshow(np.asarray(Image.open(root / m["rel"])) / 4095.0, cmap="gray", vmin=0, vmax=1)
            a.set_title(f"{m['exposure']} ms\nsat {m['sat_frac']:.2f}", fontsize=8); a.axis("off")
        fig.suptitle(f"Exposure bracket (raw linear scale): {t} PosA Loff Sun_180 CamL")
        samples.append(save_fig(fig, S / "nasa_exposure_bracket.png"))

    # LiDAR -> left camera projection overlay (feasibility evidence for spatial localisation)
    for t, ms in sorted(by_terrain.items()):
        if t not in calib or not xyz_cache.get(t):
            continue
        c = calib[t]; K = c["K_left"]; w, h = c["image_wh"]
        poss = sorted(xyz_cache[t])
        fig, ax = plt.subplots(1, len(poss), figsize=(7 * len(poss), 4.6))
        for a, pstem in zip(np.atleast_1d(ax), poss):
            p = pstem.replace("Pos", "")
            m = _best_image(ms, pos=p, light="on", sun="no", cam="L") or _best_image(ms, pos=p, cam="L")
            if m is None:
                a.set_title(f"{pstem}: no image"); a.axis("off"); continue
            img = tone_map_12bit(np.asarray(Image.open(root / m["rel"])))
            pts = xyz_cache[t][pstem]
            if inverted.get((t, pstem), True):
                pc, label = (pts - c["T_cam_lidar"]) @ c["R_cam_lidar"], "Xc = R^T (X - t)"
            else:
                pc, label = pts @ c["R_cam_lidar"].T + c["T_cam_lidar"], "Xc = R X + t"
            f = pc[:, 2] > 0.1
            u = K[0, 0] * pc[f, 0] / pc[f, 2] + K[0, 2]; v = K[1, 1] * pc[f, 1] / pc[f, 2] + K[1, 2]
            ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            a.imshow(img, cmap="gray")
            sc = a.scatter(u[ins][::4], v[ins][::4], c=pc[f, 2][ins][::4], s=0.3, cmap="turbo", alpha=0.5)
            fig.colorbar(sc, ax=a, fraction=0.03, label="depth (m)")
            a.set_title(f"{t} {pstem} on {Path(m['rel']).name}: LiDAR projected, {label}\n(selected by projection test; no distortion; every 100th point)", fontsize=8)
            a.set_xlim(0, w); a.set_ylim(h, 0); a.axis("off")
        samples.append(save_fig(fig, S / f"nasa_lidar_projection_{t}.png"))
    return figs, samples
