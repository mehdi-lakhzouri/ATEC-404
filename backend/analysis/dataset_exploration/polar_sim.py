"""A4 - POLAR-Sim annotation analysis (YOLO detection boxes and YOLO segmentation polygons)."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .common import LOG, ErrorLog, Paths, describe, save_fig

IMG_W, IMG_H = 1936, 1216  # NASA POLAR image size (from stereo.calibration and decoded PNGs)
CANON_RE = re.compile(r"^(?P<terrain>\d{2})_(?P<pos>[A-Z])_(?P<light>on|off)_(?P<sun>no|\d+)_(?P<cam>[LR])_(?P<exp>\d{4})\.txt$")
PREFIX_RE = re.compile(r"^(?P<key>\d{2}_[A-Z]_(?:on|off)_(?:no|\d+)_[LR]_\d{4})")
DET_CLASSES_README = {0: "rock", 1: "shadow"}
SEG_CLASSES_README = {0: "background", 1: "ground", 2: "rocks", 3: "shadows"}
EPS = 1e-6


def classify_file(p: Path) -> dict[str, Any]:
    name = p.name
    if name == "classes.txt":
        return {"kind": "metadata_classes"}
    if name.startswith(".") or p.suffix.lower() != ".txt":
        return {"kind": "system_or_non_txt"}
    m = CANON_RE.match(name)
    if m:
        return {"kind": "canonical", "key": name[:-4], **m.groupdict()}
    m = PREFIX_RE.match(name)
    if m:
        return {"kind": "variant_copy", "key": m.group("key")}
    return {"kind": "unrecognised_name"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_det_file(p: Path, n_classes: int) -> dict[str, Any]:
    txt = p.read_text(encoding="utf-8", errors="replace")
    boxes, issues = [], []
    lines = txt.splitlines()
    for ln, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            issues.append({"line": ln, "issue": "blank_line"})
            continue
        tok = s.split()
        if len(tok) != 5:
            issues.append({"line": ln, "issue": f"expected 5 tokens, got {len(tok)}", "raw": s[:120]})
            continue
        try:
            c = int(tok[0])
            cx, cy, w, h = (float(t) for t in tok[1:])
        except ValueError:
            issues.append({"line": ln, "issue": "non_numeric_token", "raw": s[:120]})
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            issues.append({"line": ln, "issue": "non_finite"}); continue
        bad = []
        if not 0 <= c < n_classes:
            bad.append("class_out_of_range")
        if w <= 0 or h <= 0:
            bad.append("nonpositive_size")
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            bad.append("centre_out_of_[0,1]")
        if cx - w / 2 < -EPS or cy - h / 2 < -EPS or cx + w / 2 > 1 + EPS or cy + h / 2 > 1 + EPS:
            bad.append("box_extends_outside_image")
        if w * IMG_W < 2 or h * IMG_H < 2:
            bad.append("tiny_lt_2px")
        for b in bad:
            issues.append({"line": ln, "issue": b, "raw": s[:120]})
        boxes.append({"cls": c, "cx": cx, "cy": cy, "w": w, "h": h, "flags": bad})
    dup = [k for k, v in Counter(l.strip() for l in lines if l.strip()).items() if v > 1]
    for d in dup:
        issues.append({"issue": "duplicate_line", "raw": d[:120]})
    return {"boxes": boxes, "issues": issues, "empty": not txt.strip(), "lines": len(lines)}


def _poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def parse_seg_file(p: Path, raster_scale: float = 0.25) -> dict[str, Any]:
    txt = p.read_text(encoding="utf-8", errors="replace")
    polys, issues = [], []
    rw, rh = int(IMG_W * raster_scale), int(IMG_H * raster_scale)
    label = np.full((rh, rw), 255, np.uint8)  # 255 = no polygon
    for ln, line in enumerate(txt.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        try:
            c = int(tok[0])
            v = np.array([float(t) for t in tok[1:]])
        except ValueError:
            issues.append({"line": ln, "issue": "non_numeric_token", "raw": s[:80]}); continue
        if len(v) % 2:
            issues.append({"line": ln, "issue": "odd_coordinate_count", "n": len(v)}); continue
        pts = v.reshape(-1, 2)
        bad = []
        if not 0 <= c <= 3:
            bad.append("class_out_of_range")
        if len(pts) < 3:
            bad.append("fewer_than_3_vertices")
        if not np.all(np.isfinite(pts)):
            bad.append("non_finite")
        elif (pts < -EPS).any() or (pts > 1 + EPS).any():
            bad.append("coordinate_out_of_[0,1]")
        px = pts * [IMG_W, IMG_H]
        uniq = len(np.unique(np.round(px, 3), axis=0))
        area = _poly_area(px) if len(pts) >= 3 else 0.0
        if uniq < 3 or area < 1.0:
            bad.append("degenerate_area_lt_1px")
        for b in bad:
            issues.append({"line": ln, "issue": b})
        polys.append({"cls": c, "n_vertices": len(pts), "area_px": area, "flags": bad})
        if "non_finite" not in bad and len(pts) >= 3:
            cv2.fillPoly(label, [np.round(px * raster_scale).astype(np.int32)], int(c))
    cover = {SEG_CLASSES_README.get(k, str(k)): float((label == k).mean()) for k in range(4)}
    cover["unlabelled"] = float((label == 255).mean())
    return {"polys": polys, "issues": issues, "empty": not txt.strip(), "coverage_last_write_wins": cover}


def _scan(dir_: Path) -> list[Path]:
    return sorted(p for p in dir_.rglob("*") if p.is_file() and ".git" not in p.parts)


def run(paths: Paths, errors: ErrorLog) -> dict[str, Any]:
    root = paths.polar_sim
    LOG.info("A4 POLAR-Sim: %s", root)
    res: dict[str, Any] = {"root": paths.rel(root), "class_metadata_readme": {"detection": DET_CLASSES_README, "segmentation": SEG_CLASSES_README}}

    # ------------------------------------------------------------------ detection
    det_root = root / "Labels"
    det_files = _scan(det_root)
    kinds = Counter()
    classes_txt = {}
    canonical: dict[str, dict[str, Any]] = {}
    variants: dict[str, list[str]] = defaultdict(list)
    other = []
    folder_mismatch = []
    for p in det_files:
        info = classify_file(p)
        kinds[info["kind"]] += 1
        rel = p.relative_to(root).as_posix()
        if info["kind"] == "metadata_classes":
            classes_txt[rel] = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        elif info["kind"] == "canonical":
            if p.parent.name != f"Terrain{info['terrain']}":
                folder_mismatch.append(rel)
            canonical[info["key"]] = {"path": p, "rel": rel, **info}
        elif info["kind"] == "variant_copy":
            variants[info["key"]].append(rel)
        else:
            other.append(rel)
    class_lists = {tuple(v) for v in classes_txt.values()}
    det_names = list(next(iter(class_lists))) if len(class_lists) == 1 else None
    if len(class_lists) != 1:
        errors.add("polar_sim.det", "classes.txt", f"inconsistent class lists: {class_lists}")
    n_cls = len(det_names) if det_names else 2

    parsed: dict[str, dict[str, Any]] = {}
    for key, c in sorted(canonical.items()):
        try:
            parsed[key] = parse_det_file(c["path"], n_cls)
        except OSError as e:
            errors.add("polar_sim.det.read", c["rel"], repr(e))
    variant_report = []
    for key, rels in sorted(variants.items()):
        base = canonical.get(key)
        entry = {"key": key, "variants": rels, "canonical_exists": base is not None}
        if base:
            h0 = _sha(base["path"])
            entry["identical_to_canonical"] = [_sha(root / r) == h0 for r in rels]
        variant_report.append(entry)

    per_terrain = defaultdict(lambda: Counter())
    per_terrain_files = Counter()
    sizes = defaultdict(list)
    per_file_counts = []
    by_exposure = defaultdict(list)
    invalid_files = {}
    issue_types = Counter()
    for key, pr in parsed.items():
        c = canonical[key]
        per_terrain_files[c["terrain"]] += 1
        for b in pr["boxes"]:
            per_terrain[c["terrain"]][b["cls"]] += 1
            sizes[b["cls"]].append((b["w"] * IMG_W, b["h"] * IMG_H))
        per_file_counts.append(len(pr["boxes"]))
        by_exposure[c["exp"]].append(sum(1 for b in pr["boxes"] if b["cls"] == 0))
        real_issues = [i for i in pr["issues"] if i["issue"] not in ("tiny_lt_2px",)]
        for i in pr["issues"]:
            issue_types[i["issue"]] += 1
        if real_issues:
            invalid_files[c["rel"]] = real_issues[:10]
    empty_files = sorted(canonical[k]["rel"] for k, pr in parsed.items() if pr["empty"])
    # Are labels re-used across the exposures of one bracket (annotated once per HDR scene)?
    brackets = defaultdict(list)
    rock_sets = defaultdict(list)
    for key, c in canonical.items():
        b = key.rsplit("_", 1)[0]
        brackets[b].append(_sha(c["path"]))
        if key in parsed:
            rock_sets[b].append(tuple(sorted((round(x["cx"], 4), round(x["cy"], 4), round(x["w"], 4), round(x["h"], 4))
                                             for x in parsed[key]["boxes"] if x["cls"] == 0)))
    multi = {k: v for k, v in brackets.items() if len(v) > 1}
    rmulti = {k: v for k, v in rock_sets.items() if len(v) > 1}
    bracket_reuse = {
        "brackets_with_ge2_label_files": len(multi),
        "brackets_all_files_identical": sum(len(set(v)) == 1 for v in multi.values()),
        "brackets_with_distinct_files": sum(len(set(v)) > 1 for v in multi.values()),
        "distinct_contents_per_bracket": describe(len(set(v)) for v in multi.values()),
        "brackets_rock_boxes_identical_all_exposures": sum(len(set(v)) == 1 for v in rmulti.values()),
        "brackets_rock_boxes_distinct_sets": describe(len(set(v)) for v in rmulti.values()),
        "note": "rock boxes compared at 1e-4 normalised precision; shadows may differ between exposures",
    }
    res["detection"] = {
        "directory": paths.rel(det_root),
        "terrain_folders": sorted(p.name for p in det_root.iterdir() if p.is_dir()),
        "file_kind_counts": dict(kinds),
        "class_metadata_files": classes_txt,
        "class_mapping_verified": {i: n for i, n in enumerate(det_names)} if det_names else None,
        "class_mapping_matches_readme": det_names == [DET_CLASSES_README[0], DET_CLASSES_README[1]] if det_names else False,
        "canonical_label_files": len(parsed),
        "canonical_files_in_wrong_terrain_folder": folder_mismatch,
        "variant_copy_files": sum(len(v) for v in variants.values()),
        "variant_copy_report": variant_report,
        "unrecognised_or_system_files": other + [p.relative_to(root).as_posix() for p in det_files if classify_file(p)["kind"] == "system_or_non_txt"],
        "empty_label_files": len(empty_files),
        "empty_label_file_examples": empty_files[:30],
        "files_per_terrain": dict(sorted(per_terrain_files.items())),
        "annotations_per_terrain_and_class": {t: {det_names[k] if det_names else k: v for k, v in sorted(c.items())} for t, c in sorted(per_terrain.items())},
        "annotations_total_per_class": {det_names[k] if det_names else k: sum(c[k] for c in per_terrain.values()) for k in range(n_cls)},
        "objects_per_file": describe(per_file_counts),
        "box_width_px_per_class": {det_names[k] if det_names else k: describe(w for w, _ in v) for k, v in sorted(sizes.items())},
        "box_height_px_per_class": {det_names[k] if det_names else k: describe(h for _, h in v) for k, v in sorted(sizes.items())},
        "rock_boxes_per_file_by_exposure": {e: describe(v) for e, v in sorted(by_exposure.items())},
        "label_reuse_across_exposure_bracket": bracket_reuse,
        "issue_type_counts": dict(issue_types),
        "files_with_issues": len(invalid_files),
        "files_with_issues_detail": invalid_files,
        "pixel_size_assumption": [IMG_W, IMG_H],
    }

    # ------------------------------------------------------------------ segmentation
    seg_root = root / "SegmentationLabels"
    seg_files = _scan(seg_root)
    skinds = Counter()
    seg_canon: dict[str, dict[str, Any]] = {}
    seg_dups: dict[str, list[str]] = defaultdict(list)
    seg_misfiled = []
    seg_other = []
    for p in seg_files:
        info = classify_file(p)
        rel = p.relative_to(root).as_posix()
        if info["kind"] == "unrecognised_name" and p.suffix == ".txt":
            # Terrain13 uses positions D/E; still parse with the generic prefix rule
            info = {"kind": "canonical_extended", "key": p.stem}
        skinds[info["kind"]] += 1
        if info["kind"] in ("canonical", "canonical_extended"):
            folder = re.search(r"terrain(\d{2})", rel)
            if folder and folder.group(1) != p.stem[:2]:
                seg_misfiled.append(rel)
            if info["key"] in seg_canon:
                seg_dups[info["key"]].append(rel)
                errors.add("polar_sim.seg", rel, "duplicate key (first file in sorted order kept as canonical)")
                continue  # deterministic: keep first in sorted path order
            seg_canon[info["key"]] = {"path": p, "rel": rel, "terrain": p.stem[:2], **info}
        else:
            seg_other.append(rel)
    sp: dict[str, dict[str, Any]] = {}
    for key, c in sorted(seg_canon.items()):
        try:
            sp[key] = parse_seg_file(c["path"])
        except OSError as e:
            errors.add("polar_sim.seg.read", c["rel"], repr(e))
    s_terr = defaultdict(Counter)
    s_files = Counter()
    verts = defaultdict(list)
    areas = defaultdict(list)
    s_issue = Counter()
    s_invalid = {}
    cover = defaultdict(list)
    classes_seen = Counter()
    for key, pr in sp.items():
        t = seg_canon[key]["terrain"]
        s_files[t] += 1
        for q in pr["polys"]:
            s_terr[t][q["cls"]] += 1
            classes_seen[q["cls"]] += 1
            verts[q["cls"]].append(q["n_vertices"])
            areas[q["cls"]].append(q["area_px"])
        for i in pr["issues"]:
            s_issue[i["issue"]] += 1
        if pr["issues"]:
            s_invalid[seg_canon[key]["rel"]] = pr["issues"][:10]
        for k, v in pr["coverage_last_write_wins"].items():
            cover[k].append(v)
    cname = lambda k: SEG_CLASSES_README.get(k, str(k))  # noqa: E731
    files_with_class = {cname(k): sum(1 for pr in sp.values() if any(q["cls"] == k for q in pr["polys"])) for k in range(4)}
    res["segmentation"] = {
        "directory": paths.rel(seg_root),
        "terrain_folders": sorted(p.name for p in seg_root.iterdir() if p.is_dir()),
        "file_kind_counts": dict(skinds),
        "non_label_files": seg_other,
        "files_in_folder_of_other_terrain": seg_misfiled,
        "duplicate_keys": [
            {"key": k, "kept": seg_canon[k]["rel"], "duplicates": v,
             "identical_to_kept": [_sha(root / r) == _sha(seg_canon[k]["path"]) for r in v]}
            for k, v in sorted(seg_dups.items())
        ],
        "class_metadata_files_present": [p.relative_to(root).as_posix() for p in seg_files if p.suffix.lower() in (".yaml", ".yml") or p.name == "classes.txt"],
        "class_mapping_source": "README.md + yolo2bitmap.py (no metadata file in SegmentationLabels)",
        "class_ids_observed": dict(sorted(classes_seen.items())),
        "label_files": len(sp),
        "empty_label_files": sum(pr["empty"] for pr in sp.values()),
        "files_per_terrain": dict(sorted(s_files.items())),
        "positions_observed": dict(Counter(k.split("_")[1] for k in sp)),
        "polygons_per_terrain_and_class": {t: {cname(k): v for k, v in sorted(c.items())} for t, c in sorted(s_terr.items())},
        "polygons_total_per_class": {cname(k): v for k, v in sorted(classes_seen.items())},
        "files_containing_class": files_with_class,
        "vertices_per_polygon": {cname(k): describe(v) for k, v in sorted(verts.items())},
        "polygon_area_px": {cname(k): describe(v) for k, v in sorted(areas.items())},
        "polygons_per_file": describe(len(pr["polys"]) for pr in sp.values()),
        "pixel_coverage_quarter_res": {k: describe(v) for k, v in sorted(cover.items())},
        "issue_type_counts": dict(s_issue),
        "files_with_issues": len(s_invalid),
        "files_with_issues_detail": dict(list(sorted(s_invalid.items()))[:60]),
    }
    res["figures"] = _figures(paths, res, sizes, det_names, s_terr)
    res["_index"] = {"det": {k: {"rel": v["rel"]} for k, v in canonical.items()},
                     "det_variants": dict(variants), "seg": {k: {"rel": v["rel"]} for k, v in seg_canon.items()}}
    res["_parsed_det"] = parsed
    res["_parsed_seg"] = sp
    return res


def _figures(paths: Paths, res, sizes, det_names, s_terr) -> list[str]:
    F = paths.figures
    figs = []
    det = res["detection"]["annotations_per_terrain_and_class"]
    terr = sorted(det)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    names = det_names or ["0", "1"]
    for k, n in enumerate(names):
        ax[0].bar(np.arange(len(terr)) + k * 0.4, [det[t].get(n, 0) for t in terr], width=0.4, label=n)
    ax[0].set_xticks(np.arange(len(terr)) + 0.2, terr); ax[0].set(title="POLAR-Sim detection boxes per terrain", xlabel="terrain", ylabel="boxes"); ax[0].legend()
    st = sorted(s_terr)
    for k in range(4):
        ax[1].bar(np.arange(len(st)) + k * 0.2, [s_terr[t].get(k, 0) for t in st], width=0.2, label=SEG_CLASSES_README[k])
    ax[1].set_xticks(np.arange(len(st)) + 0.3, st); ax[1].set(title="POLAR-Sim segmentation polygons per terrain", xlabel="terrain", ylabel="polygons"); ax[1].legend()
    figs.append(save_fig(fig, F / "polarsim_class_distribution.png"))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, v in sorted(sizes.items()):
        a = np.array(v)
        ax[0].hist(np.log10(a[:, 0] * a[:, 1]), bins=60, alpha=.6, label=names[k])
        ax[1].hist(np.clip(a[:, 0] / a[:, 1], 0, 6), bins=60, alpha=.6, label=names[k])
    ax[0].set(title="Box area log10(px²) @1936x1216", xlabel="log10 area"); ax[0].legend()
    ax[1].set(title="Aspect w/h (clipped 6)", xlabel="w/h"); ax[1].legend()
    ex = res["detection"]["rock_boxes_per_file_by_exposure"]
    keys = sorted(ex)
    ax[2].plot([int(k) for k in keys], [ex[k]["mean"] for k in keys], "o-")
    ax[2].set_xscale("log", base=2); ax[2].set(title="Mean rock boxes per label file vs exposure", xlabel="exposure (ms)", ylabel="rock boxes"); ax[2].grid(alpha=.3)
    figs.append(save_fig(fig, F / "polarsim_box_distributions.png"))
    return figs
