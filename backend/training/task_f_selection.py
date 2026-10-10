"""Task F: validation-only model selection, confidence-threshold analysis and false-negative analysis.

Reuses the cached E.1c ignore-aware predictions (conf >= 0.001, imgsz 736, NMS IoU 0.6, max_det 300) on the VAL split.
Never reads the test split. Writes only into the output folder.

Usage (from backend/):
    uv run --no-sync python -m training.task_f_selection --stage analyse     # thresholds, sizes, FN categories, scenes
    uv run --no-sync python -m training.task_f_selection --stage freeze      # write the frozen selection
    uv run --no-sync python -m training.task_f_selection --stage inference   # live inference parity + qualitative check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from analysis.evaluation.ignore_eval import ImageEval, _area, iou, load_split, match_image

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"
RUN = ROOT / "experiments/task-e/full-training-config-a-200-s42"
OUT = ROOT / "experiments/task-f/model-selection-001"
PRIMARY_IGNORE = [DS / "manifests/eval_ignore_regions.json"]        # approved primary policy: REVIEW regions only
CANDIDATES = {"best.pt": "best", "epoch_completed_120.pt": "epoch_completed_120", "epoch_completed_130.pt": "epoch_completed_130",
              "epoch_completed_140.pt": "epoch_completed_140"}
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
MATCH_IOU, IOA_THR, MIN_PRED_AREA = 0.5, 0.5, 50.0
PREDICT = {"imgsz": 736, "iou": 0.6, "max_det": 300}
SIZE_BINS = {"small <400": (0, 400), "medium 400-2500": (400, 2500), "large 2500-10000": (2500, 10000), "very large >=10000": (10000, float("inf")),
             "large (all) >=2500": (2500, float("inf"))}
# Selection rules, fixed in code before any threshold metric was computed:
RULES = {
    "checkpoint": "Keep best.pt (provisional primary, user decision) unless a comparison candidate exceeds its primary ignore-aware "
                  "AP50 by more than 0.005 (below that, differences are within single-seed checkpoint-to-checkpoint noise).",
    "threshold": "Recall-oriented: the LOWEST threshold in the grid whose F1 is within 0.02 of the best F1 on the grid "
                 "(recall gained at negligible F1 cost; a missed obstacle costs more than a false alarm).",
    "nms": "Unchanged from the evaluation that produced the candidates: NMS IoU 0.6, max_det 300, imgsz 736, plus min box area 50 px^2.",
    "evaluation_policy": "Primary = REVIEW regions ignored (eval_ignore_regions.json), IoU 0.5, IoA 0.5, 101-pt AP, min_pred_area 50. "
                         "Sub-threshold PROPOSED ignores reported as secondary only (not approved).",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _w(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8", newline="\n")


def _images(stem: str) -> dict[str, ImageEval]:
    return load_split(DS, "val", RUN / "ignore_aware_val" / stem / "val" / "labels", PRIMARY_IGNORE)


def gt_match(img: ImageEval, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching of predictions >= thr (same rule as match_image). Returns (gt_matched mask, kept preds)."""
    p = img.pred[(img.pred[:, 4] >= thr)] if len(img.pred) else img.pred.reshape(0, 5)
    p = p[_area(p[:, :4]) >= MIN_PRED_AREA] if len(p) else p
    p = p[np.argsort(-p[:, 4], kind="stable")]
    gt = img.gt.reshape(-1, 4)
    used = np.zeros(len(gt), bool)
    ov = iou(p[:, :4], gt)
    for k in range(len(p)):
        if len(gt):
            cand = np.where(~used, ov[k], -1.0)
            if cand.max() >= MATCH_IOU:
                used[int(np.argmax(cand))] = True
    return used, p


def threshold_table(images: dict[str, ImageEval]) -> list[dict[str, Any]]:
    per = [match_image(im, MATCH_IOU, IOA_THR, MIN_PRED_AREA) for im in images.values()]
    s = np.concatenate([x[0] for x in per]); l = np.concatenate([x[1] for x in per]); n = sum(x[2] for x in per)
    rows = []
    for t in THRESHOLDS:
        k = s >= t
        tp, fp = int((l[k] == 1).sum()), int((l[k] == 0).sum()); fn = n - tp
        P, R = tp / max(tp + fp, 1), tp / max(n, 1)
        f = lambda b: (1 + b * b) * P * R / max(b * b * P + R, 1e-12)  # noqa: E731
        rows.append({"conf": t, "tp": tp, "fp": fp, "fn": fn, "gt": n, "precision": round(P, 4), "recall": round(R, 4),
                     "f1": round(f(1), 4), "f2": round(f(2), 4), "fp_per_image": round(fp / len(images), 2)})
    return rows


def size_recall(images: dict[str, ImageEval]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, rng in SIZE_BINS.items():
        per = [match_image(im, MATCH_IOU, IOA_THR, MIN_PRED_AREA, rng) for im in images.values()]
        s = np.concatenate([x[0] for x in per]); l = np.concatenate([x[1] for x in per]); n = sum(x[2] for x in per)
        out[name] = {"gt": n, **{f"recall@{t}": round(int((l[s >= t] == 1).sum()) / max(n, 1), 4) for t in THRESHOLDS},
                     **{f"precision@{t}": round(int((l[s >= t] == 1).sum()) / max(int((s >= t).sum()), 1), 4) for t in THRESHOLDS}}
    return out


def max_det_check(images: dict[str, ImageEval]) -> dict[str, Any]:
    """Filtering cached conf>=0.001 predictions at t equals predicting at t unless max_det truncated boxes scoring >= t."""
    capped = {n: float(im.pred[:, 4].min()) for n, im in images.items() if len(im.pred) >= PREDICT["max_det"]}
    return {"images_at_max_det": len(capped),
            **{f"images_where_cap_affects_conf_{t}": sum(1 for v in capped.values() if v >= t) for t in THRESHOLDS}}


def fn_analysis(images: dict[str, ImageEval], thr: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cats = {"found_below_threshold": 0, "localisation_iou_0.1_0.5": 0, "near_miss_iou_lt_0.1": 0}
    by_size = {k: dict(cats) for k in SIZE_BINS if k != "large (all) >=2500"}
    per_img = []
    for name, im in images.items():
        used, _ = gt_match(im, thr)
        used_all, _ = gt_match(im, 0.001)
        gt = im.gt.reshape(-1, 4)
        ga = _area(gt)
        p_thr = im.pred[im.pred[:, 4] >= thr][:, :4] if len(im.pred) else np.zeros((0, 4))
        best_iou = iou(gt, p_thr).max(axis=1) if len(p_thr) and len(gt) else np.zeros(len(gt))
        fn_idx = np.where(~used)[0]
        large_fn = 0
        for j in fn_idx:
            c = "found_below_threshold" if used_all[j] else ("localisation_iou_0.1_0.5" if best_iou[j] >= 0.1 else "near_miss_iou_lt_0.1")
            cats[c] += 1
            sz = next(k for k, (a, b) in SIZE_BINS.items() if k != "large (all) >=2500" and a <= ga[j] < b)
            by_size[sz][c] += 1
            large_fn += int(ga[j] >= 2500)
        per_img.append({"image": name, "gt": len(gt), "fn": len(fn_idx), "fn_large": large_fn, "gt_large": int((ga >= 2500).sum()),
                        "recall": round(1 - len(fn_idx) / max(len(gt), 1), 3), "review_regions": len(im.ignore)})
    return {"total_fn": sum(cats.values()), "categories": cats, "by_size": by_size}, per_img


def render(images: dict[str, ImageEval], names: list[str], thr: float, out_dir: Path, tag: str, split: str = "val") -> Path:
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name in names:
        im = images[name]
        used, p = gt_match(im, thr)
        img = Image.open(DS / "images" / split / name).convert("RGB"); d = ImageDraw.Draw(img)
        for b in im.ignore:
            d.rectangle(list(b), outline=(255, 200, 0), width=1)
        for b, u in zip(im.gt, used):
            d.rectangle(list(b), outline=(0, 220, 0) if u else (255, 0, 0), width=1 if u else 2)
        for b in p:
            d.rectangle(list(b[:4]), outline=(255, 0, 255), width=1)
        d.rectangle([0, 0, 719, 14], fill=(0, 0, 0))
        d.text((3, 2), f"{name} conf>={thr} | GT {len(im.gt)}: matched green, MISSED red | preds magenta {len(p)} | REVIEW yellow {len(im.ignore)}",
               fill=(255, 255, 255))
        img.save(out_dir / f"{tag}_{Path(name).stem}.jpg", quality=92)
        tiles.append(img)
    cols = 2; rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (720 * cols, 480 * rows), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 720, (i // cols) * 480))
    p = out_dir / f"{tag}_contact_sheet.jpg"
    sheet.save(p, quality=90)
    return p


def analyse() -> dict[str, Any]:
    prior = json.loads((RUN / "ignore_aware_val.json").read_text(encoding="utf-8"))["checkpoints"]
    res: dict[str, Any] = {"split": "val (923 images) - test split not read", "rules": RULES, "candidates": {}}
    imgs_by = {}
    for ck, stem in CANDIDATES.items():
        images = _images(stem); imgs_by[ck] = images
        p = prior[ck]["primary_review_ignores"]
        res["candidates"][ck] = {"sha256": _sha(RUN / "weights" / ck),
                                 "AP50": round(p["AP50"], 4), "AP50_95": round(p["AP50_95"], 4),
                                 "AP50_by_size": {k: round(v["AP50"], 4) for k, v in p["by_size"].items()},
                                 "AP50_secondary_subthreshold_PROPOSED": round(prior[ck]["secondary_review_plus_subthreshold_PROPOSED"]["AP50"], 4),
                                 "AP50_reference_no_ignores": round(prior[ck]["reference_no_ignores"]["AP50"], 4),
                                 "thresholds": threshold_table(images), "size_recall": size_recall(images),
                                 "max_det_check": max_det_check(images)}
        # consistency: our threshold table at 0.25 must equal the evaluator's @conf numbers
        r25 = res["candidates"][ck]["thresholds"][-1]
        assert (r25["tp"], r25["fp"]) == (p["tp@conf"], p["fp@conf"]), (ck, r25, p["tp@conf"], p["fp@conf"])
    # checkpoint rule
    base = res["candidates"]["best.pt"]["AP50"]
    better = {k: v["AP50"] for k, v in res["candidates"].items() if v["AP50"] - base > 0.005}
    chosen = max(better, key=better.get) if better else "best.pt"
    # threshold rule
    tab = res["candidates"][chosen]["thresholds"]
    f1max = max(r["f1"] for r in tab)
    thr = min(r["conf"] for r in tab if r["f1"] >= f1max - 0.02)
    res["selection"] = {"checkpoint": chosen, "candidates_beating_best_by_gt_0.005": better, "f1_max_on_grid": f1max,
                        "f1_argmax_conf": max(tab, key=lambda r: r["f1"])["conf"], "conf": thr}
    fn, per_img = fn_analysis(imgs_by[chosen], thr)
    res["fn_analysis_selected"] = fn
    with (OUT / "per_image_fn_selected.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_img[0])); w.writeheader(); w.writerows(per_img)
    hard_large = [r["image"] for r in sorted(per_img, key=lambda r: (-r["fn_large"], -r["fn"]))[:6]]
    hard_rate = [r["image"] for r in sorted([r for r in per_img if r["gt"] >= 5], key=lambda r: (r["recall"], -r["gt"]))[:6]]
    res["difficult_scenes"] = {"most_missed_large_rocks": [r for r in per_img if r["image"] in hard_large],
                               "lowest_recall_ge5_gt": [r for r in per_img if r["image"] in hard_rate],
                               "images_with_any_missed_large_rock": sum(1 for r in per_img if r["fn_large"]),
                               "images_with_large_rocks": sum(1 for r in per_img if r["gt_large"])}
    render(imgs_by[chosen], hard_large, thr, OUT / "fn_scenes", "large_fn")
    render(imgs_by[chosen], hard_rate, thr, OUT / "fn_scenes", "low_recall")
    _plots(res)
    _w(OUT / "selection_analysis.json", res)
    return res


def _plots(res: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for ck, v in res["candidates"].items():
        t = v["thresholds"]
        ax[0].plot([r["recall"] for r in t], [r["precision"] for r in t], "o-", label=ck)
        ax[1].plot([r["conf"] for r in t], [r["f1"] for r in t], "o-", label=ck)
        ax[2].plot([r["conf"] for r in t], [r["fp_per_image"] for r in t], "o-", label=ck)
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("P/R at conf 0.05..0.25")
    ax[1].set_xlabel("conf"); ax[1].set_title("F1"); ax[2].set_xlabel("conf"); ax[2].set_title("false positives / image")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Task F threshold analysis - VAL only, primary ignore-aware policy (PROVISIONAL automated labels)")
    fig.tight_layout(); fig.savefig(OUT / "threshold_analysis.png", dpi=110); plt.close(fig)


def freeze() -> dict[str, Any]:
    import shutil
    a = json.loads((OUT / "selection_analysis.json").read_text(encoding="utf-8"))
    sel = a["selection"]; ck = sel["checkpoint"]
    src = RUN / "weights" / ck
    dst = OUT / "model" / "artemis_rock_yolo11n_e1c_s42.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(src, dst)                                    # copy; the run's checkpoints are never moved or deleted
    assert _sha(dst) == _sha(src)
    from ultralytics import YOLO
    m = YOLO(str(dst))
    cfg = {"frozen": True, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "selected_on": "validation split only (923 images)",
           "test_split_used": False,
           "model": {"file": dst.relative_to(ROOT).as_posix(), "sha256": _sha(dst), "source_checkpoint": src.relative_to(ROOT).as_posix(),
                     "source_sha256": _sha(src), "architecture": "YOLO11n", "params": sum(p.numel() for p in m.model.parameters()),
                     "class_names": {int(k): v for k, v in m.names.items()}, "training_run": "E.1c full-training-config-a-200-s42",
                     "ultralytics_best_epoch": 111},
           "inference": {"imgsz": PREDICT["imgsz"], "conf": sel["conf"], "nms_iou": PREDICT["iou"], "max_det": PREDICT["max_det"],
                         "min_box_area_px2": MIN_PRED_AREA, "classes": [0], "agnostic_nms": False, "input": "native 720x480 render, RGB",
                         "note": "min_box_area is a post-filter applied after Ultralytics predict (boxes < 50 px^2 dropped)"},
           "evaluation_policy": {"primary": RULES["evaluation_policy"], "match_iou": MATCH_IOU, "ignore_ioa": IOA_THR,
                                 "ap": "101-point, AP50 and AP50-95 (IoU 0.50:0.95:0.05)", "size_bins_px2": {"small": "<400", "medium": "400-2500", "large": ">=2500"},
                                 "ignore_files": [{"file": p.relative_to(ROOT).as_posix(), "sha256": _sha(p)} for p in PRIMARY_IGNORE],
                                 "labels": "PROVISIONAL automated S2-H VALID boxes (not human-verified)"},
           "dataset": {"name": "yolo-rock-v1.1", "status": "PROVISIONAL, NOT FROZEN", "split_manifest_sha256": _sha(DS / "manifests/split_manifest.csv")},
           "selection_rules": RULES, "selection_result": sel,
           "val_metrics_at_frozen_config": next(r for r in a["candidates"][ck]["thresholds"] if r["conf"] == sel["conf"]),
           "val_AP": {"AP50": a["candidates"][ck]["AP50"], "AP50_95": a["candidates"][ck]["AP50_95"]}}
    _w(OUT / "FROZEN_MODEL_CONFIG.json", cfg)
    return cfg


def inference() -> dict[str, Any]:
    """Live inference with the frozen config: (1) parity with the cached-prediction analysis on 50 val images, (2) overlays."""
    from ultralytics import YOLO
    cfg = json.loads((OUT / "FROZEN_MODEL_CONFIG.json").read_text(encoding="utf-8"))
    inf = cfg["inference"]
    model = YOLO(str(ROOT / cfg["model"]["file"]))
    stem = CANDIDATES[cfg["selection_result"]["checkpoint"]]
    cached = _images(stem)
    names = sorted(cached)[::18][:50]
    diffs, t_inf = [], []
    for n in names:
        t = time.perf_counter()
        r = model.predict(str(DS / "images/val" / n), imgsz=inf["imgsz"], conf=inf["conf"], iou=inf["nms_iou"], max_det=inf["max_det"],
                          device=0, verbose=False)[0]
        t_inf.append(time.perf_counter() - t)
        live = r.boxes.xyxy.cpu().numpy(); live = live[_area(live) >= inf["min_box_area_px2"]]
        c = cached[n].pred; c = c[(c[:, 4] >= inf["conf"])] if len(c) else c.reshape(0, 5); c = c[_area(c[:, :4]) >= inf["min_box_area_px2"]] if len(c) else c
        m = iou(live, c[:, :4]).max(axis=1) if len(live) and len(c) else np.zeros(len(live))
        diffs.append({"image": n, "live": len(live), "cached": len(c), "live_matched_iou_gt_0.99": int((m > 0.99).sum())})
    from .smoke_report_data import SCENES
    man = {int(r["frame_id"]): r["split"] for r in csv.DictReader((DS / "manifests/split_manifest.csv").open(encoding="utf-8"))}
    scenes = [f"render{fid:04d}.png" for fid, _ in SCENES if man.get(fid) == "val"]
    sheet = render(cached, scenes, inf["conf"], OUT / "qualitative", "val_scene")
    res = {"parity_images": len(names), "parity_exact_count": sum(d["live"] == d["cached"] for d in diffs),
           "parity_boxes_live": sum(d["live"] for d in diffs), "parity_boxes_cached": sum(d["cached"] for d in diffs),
           "parity_boxes_matched": sum(d["live_matched_iou_gt_0.99"] for d in diffs), "per_image": diffs,
           "latency_ms_per_image_incl_io": {"median": round(1000 * float(np.median(t_inf[1:])), 1), "p95": round(1000 * float(np.percentile(t_inf[1:], 95)), 1)},
           "qualitative_scenes": scenes, "contact_sheet": sheet.relative_to(ROOT).as_posix(), "model_names": {int(k): v for k, v in model.names.items()}}
    _w(OUT / "inference_check.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("analyse", "freeze", "inference"), required=True)
    st = ap.parse_args().stage
    out = {"analyse": analyse, "freeze": freeze, "inference": inference}[st]()
    print(json.dumps(out, indent=1, default=float)[:9000])
