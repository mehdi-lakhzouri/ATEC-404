"""Task F: ONCE-ONLY final evaluation of the frozen model on the reserved TEST split (917 images).

Everything comes from experiments/task-f/model-selection-001/FROZEN_MODEL_CONFIG.json; nothing is tuned here.
Refuses to run if the frozen config, model, ignore files or split manifest differ from their recorded SHA-256, or if a
previous test run left its lock file. Predictions use the same protocol as validation: conf 0.001 (for AP), NMS IoU 0.6,
max_det 300, imgsz 736; the operating-point metrics use the frozen conf 0.20 and the 50 px^2 post-filter.

Usage (from backend/):  uv run --no-sync python -m training.task_f_test_eval --authorized-once
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

from analysis.evaluation.ignore_eval import IOU_THRS, ImageEval, _area, ap_101, evaluate, iou, ioa, load_split, match_image
from .preflight import PROTECTED, _tree_meta
from .task_f_selection import SIZE_BINS, fn_analysis, render

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"
SEL = ROOT / "experiments/task-f/model-selection-001"
OUT = ROOT / "experiments/task-f/test-evaluation-001"
LOCK = OUT / "TEST_RUN_LOCK.json"
IGN_PRIMARY = DS / "manifests/eval_ignore_regions.json"
IGN_SUB = DS / "manifests/eval_ignore_regions_subthreshold_PROPOSED.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _w(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8", newline="\n")


def verify_frozen() -> dict[str, Any]:
    cfg_p = SEL / "FROZEN_MODEL_CONFIG.json"
    recorded = {ln.split()[1].lstrip("*./"): ln.split()[0] for ln in (SEL / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()}
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    model = ROOT / cfg["model"]["file"]
    checks = {
        "frozen_config_sha_matches_checksums": _sha(cfg_p) == recorded["FROZEN_MODEL_CONFIG.json"],
        "config_marked_frozen_and_test_unused": cfg["frozen"] is True and cfg["test_split_used"] is False,
        "model_sha_matches_frozen": _sha(model) == cfg["model"]["sha256"],
        "model_sha_matches_checksums": _sha(model) == recorded["model/artemis_rock_yolo11n_e1c_s42.pt"],
        "primary_ignore_sha_matches_frozen": all(_sha(ROOT / f["file"]) == f["sha256"] for f in cfg["evaluation_policy"]["ignore_files"]),
        "split_manifest_sha_matches_frozen": _sha(DS / "manifests/split_manifest.csv") == cfg["dataset"]["split_manifest_sha256"],
        "test_images_917": len(list((DS / "images/test").glob("*.png"))) == 917,
        "test_labels_917": len(list((DS / "labels/test").glob("*.txt"))) == 917,
        "no_previous_test_run": not LOCK.exists(),
    }
    return {"config": cfg, "config_sha256": _sha(cfg_p), "model": model, "checks": checks}


def classify_preds(img: ImageEval, conf: float, min_area: float, thr: float = 0.5, ioa_thr: float = 0.5) -> dict[str, int]:
    """Mirror of ignore_eval.match_image (full size range) at the operating point, with FP categories."""
    p = img.pred[(img.pred[:, 4] >= conf)] if len(img.pred) else img.pred.reshape(0, 5)
    p = p[_area(p[:, :4]) >= min_area] if len(p) else p
    p = p[np.argsort(-p[:, 4], kind="stable")]
    gt = img.gt.reshape(-1, 4); ign = img.ignore.reshape(-1, 4)
    ov = iou(p[:, :4], gt); oi, oa = iou(p[:, :4], ign), ioa(p[:, :4], ign)
    used = np.zeros(len(gt), bool)
    c = {"tp": 0, "ignored": 0, "fp_duplicate": 0, "fp_localisation": 0, "fp_background": 0}
    for k in range(len(p)):
        if len(gt):
            cand = np.where(~used, ov[k], -1.0)
            if cand.max() >= thr:
                used[int(np.argmax(cand))] = True; c["tp"] += 1; continue
        if len(ign) and (oi[k].max() >= thr or oa[k].max() >= ioa_thr):
            c["ignored"] += 1; continue
        m = ov[k].max() if len(gt) else 0.0
        c["fp_duplicate" if m >= thr else "fp_localisation" if m >= 0.1 else "fp_background"] += 1
    return c


def policy_metrics(images: dict[str, ImageEval], conf: float, min_area: float) -> dict[str, Any]:
    r = evaluate(images, conf=conf, ioa_thr=0.5, min_pred_area=min_area)
    P, R = r["precision@conf"], r["recall@conf"]
    out = {"images": r["images"], "gt": r["gt"], "tp": r["tp@conf"], "fp": r["fp@conf"], "fn": r["fn@conf"],
           "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-12), "fp_per_image": r["fp@conf"] / r["images"],
           "AP50": r["AP50"], "AP50_95": r["AP50_95"], "AP50_by_size_standard_bins": {k: v["AP50"] for k, v in r["by_size"].items()}}
    out["recall_by_size"] = {}
    for name, rng in SIZE_BINS.items():
        per = [match_image(im, 0.5, 0.5, min_area, rng) for im in images.values()]
        s = np.concatenate([x[0] for x in per]); l = np.concatenate([x[1] for x in per]); n = sum(x[2] for x in per)
        k = s >= conf
        out["recall_by_size"][name] = {"gt": n, "recall": int((l[k] == 1).sum()) / max(n, 1),
                                       "precision": int((l[k] == 1).sum()) / max(int(k.sum()), 1), "AP50": ap_101(s, l, n)["ap"]}
    cats = {}
    for im in images.values():
        for kk, v in classify_preds(im, conf, min_area).items():
            cats[kk] = cats.get(kk, 0) + v
    assert (cats["tp"], cats["fp_duplicate"] + cats["fp_localisation"] + cats["fp_background"]) == (out["tp"], out["fp"]), (cats, out)
    out["prediction_breakdown"] = cats
    return out


def pr_curve(images: dict[str, ImageEval], min_area: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per = [match_image(im, 0.5, 0.5, min_area) for im in images.values()]
    s = np.concatenate([x[0] for x in per]); l = np.concatenate([x[1] for x in per]); n = sum(x[2] for x in per)
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(l[o] == 1); fp = np.cumsum(l[o] == 0)
    return tp / n, tp / np.maximum(tp + fp, 1), s[o]


def run() -> dict[str, Any]:
    v = verify_frozen()
    if not all(v["checks"].values()):
        raise SystemExit(f"REFUSED: frozen-configuration verification failed: {v['checks']}")
    cfg, inf = v["config"], v["config"]["inference"]
    OUT.mkdir(parents=True, exist_ok=True)
    fp_before = {k: _tree_meta(d) for k, d in PROTECTED.items()}
    _w(LOCK, {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "frozen_config_sha256": v["config_sha256"], "model_sha256": cfg["model"]["sha256"],
              "note": "once-only test evaluation; this lock prevents a second run"})
    from ultralytics import YOLO
    model = YOLO(str(v["model"]))
    assert {int(k): n for k, n in model.names.items()} == {0: "rock"}
    t0 = time.time()
    model.predict(source=str(DS / "images/test"), imgsz=inf["imgsz"], conf=0.001, iou=inf["nms_iou"], max_det=inf["max_det"], device=0,
                  save=False, save_txt=True, save_conf=True, project=str(OUT / "predictions"), name="test", exist_ok=True, verbose=False)
    res = score_and_report("test", OUT / "predictions/test/labels", OUT, v)
    res["predict_seconds"] = round(time.time() - t0, 1)
    fp_after = {k: _tree_meta(d) for k, d in PROTECTED.items()}
    res["protected_trees_unchanged"] = {k: fp_before[k] == fp_after[k] for k in PROTECTED}
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _w(OUT / "test_results.json", res)
    lock = json.loads(LOCK.read_text(encoding="utf-8")); lock["finished"] = res["finished"]; _w(LOCK, lock)
    return res


def score_and_report(split: str, pdir: Path, out: Path, v: dict[str, Any]) -> dict[str, Any]:
    cfg, inf = v["config"], v["config"]["inference"]
    conf, area = inf["conf"], inf["min_box_area_px2"]
    out.mkdir(parents=True, exist_ok=True)
    sets = {"PRIMARY_review_ignores": [IGN_PRIMARY], "SECONDARY_review_plus_subthreshold_PROPOSED": [IGN_PRIMARY, IGN_SUB], "REFERENCE_no_ignores": []}
    imgs = {k: load_split(DS, split, pdir, f) for k, f in sets.items()}
    res: dict[str, Any] = {
        "LABELS": "AUTOMATED S2-H boxes, NOT human-verified. All metrics are PROVISIONAL.",
        "split": f"{split} ({len(next(iter(imgs.values())))} images)" + (", evaluated once with the frozen configuration; no tuning" if split == "test" else " - PIPELINE SELF-TEST"),
        "frozen_config": {"file": "experiments/task-f/model-selection-001/FROZEN_MODEL_CONFIG.json", "sha256": v["config_sha256"]},
        "model": cfg["model"], "inference": inf, "verification_before": v["checks"],
        "protocol": "predict conf 0.001 / NMS IoU 0.6 / max_det 300 / imgsz 736 (as validation); operating point conf 0.20 + 50 px^2 post-filter; "
                    "match IoU 0.5, ignore IoA 0.5, 101-pt AP",
        }
    res["policies"] = {k: policy_metrics(im, conf, area) for k, im in imgs.items()}
    capped = [float(im.pred[:, 4].min()) for im in imgs["REFERENCE_no_ignores"].values() if len(im.pred) >= inf["max_det"]]
    res["max_det_check"] = {"images_at_max_det": len(capped), "images_where_cap_affects_conf_0.20": sum(1 for x in capped if x >= conf)}
    fn, per_img = fn_analysis(imgs["PRIMARY_review_ignores"], conf)
    res["fn_analysis_primary"] = fn
    with (out / f"per_image_{split}_primary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_img[0])); w.writeheader(); w.writerows(per_img)
    res["images_with_large_rocks"] = sum(1 for r in per_img if r["gt_large"])
    res["images_missing_any_large_rock"] = sum(1 for r in per_img if r["fn_large"])
    worst = [r["image"] for r in sorted(per_img, key=lambda r: (-r["fn_large"], -r["fn"]))[:6]]
    res["most_missed_large_scenes"] = [r for r in per_img if r["image"] in worst]
    render(imgs["PRIMARY_review_ignores"], worst, conf, out / "fn_scenes", f"{split}_large_fn", split=split)
    sel = json.loads((SEL / "selection_analysis.json").read_text(encoding="utf-8"))["candidates"]["best.pt"]
    vt = next(r for r in sel["thresholds"] if r["conf"] == conf)
    res["validation_reference_same_config"] = {"precision": vt["precision"], "recall": vt["recall"], "f1": vt["f1"], "fp_per_image": vt["fp_per_image"],
                                               "AP50": sel["AP50"], "AP50_95": sel["AP50_95"],
                                               "recall_by_size": {k: s[f"recall@{conf}"] for k, s in sel["size_recall"].items()}}
    _figures(res, imgs["PRIMARY_review_ignores"], area, conf, out, split)
    return res


def _figures(res: dict[str, Any], primary: dict[str, ImageEval], area: float, conf: float, out: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    rec, prec, sc = pr_curve(primary, area)
    ax[0].plot(rec, prec, lw=1.2, label="PRIMARY (REVIEW ignored)")
    p0 = res["policies"]["PRIMARY_review_ignores"]
    ax[0].plot([p0["recall"]], [p0["precision"]], "ro", label=f"frozen conf {conf}: P {p0['precision']:.3f} R {p0['recall']:.3f}")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title(f"{split.upper()} PR curve, IoU 0.5 (PROVISIONAL labels)"); ax[0].legend(fontsize=8)
    names = list(p0["recall_by_size"])
    x = np.arange(len(names)); wdt = 0.38
    ax[1].bar(x - wdt / 2, [res["validation_reference_same_config"]["recall_by_size"][n] for n in names], wdt, label="val")
    ax[1].bar(x + wdt / 2, [p0["recall_by_size"][n]["recall"] for n in names], wdt, label=split)
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8); ax[1].set_ylim(0, 1)
    ax[1].set_title(f"recall by size @ conf {conf} (PRIMARY)"); ax[1].legend()
    pol = list(res["policies"]); m = ["precision", "recall", "f1", "AP50", "AP50_95"]
    for j, k in enumerate(pol):
        ax[2].bar(np.arange(len(m)) + (j - 1) * 0.27, [res["policies"][k][q] for q in m], 0.27, label=k.split("_")[0])
    ax[2].set_xticks(np.arange(len(m))); ax[2].set_xticklabels(m); ax[2].set_ylim(0, 1); ax[2].legend(fontsize=8)
    ax[2].set_title(f"{split.upper()}: primary vs secondary vs no-ignore")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"Artemis-X frozen YOLO11n (E.1c best.pt) - {split.upper()} evaluation - automated labels, PROVISIONAL")
    fig.tight_layout(); fig.savefig(out / f"{split}_evaluation_figures.png", dpi=110); plt.close(fig)
    b = p0["prediction_breakdown"]; fa = res["fn_analysis_primary"]["categories"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(list(b), list(b.values()), color=["tab:green", "tab:gray", "tab:orange", "tab:red", "tab:purple"])
    ax[0].set_title(f"predictions @ conf {conf} (PRIMARY)"); ax[0].tick_params(axis="x", labelsize=8)
    ax[1].bar(list(fa), list(fa.values()), color=["tab:blue", "tab:orange", "tab:red"]); ax[1].set_title("false-negative categories (PRIMARY)")
    ax[1].tick_params(axis="x", labelsize=8)
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out / f"{split}_confusion_analysis.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authorized-once", action="store_true", help="explicit flag: the user authorised exactly one test evaluation")
    ap.add_argument("--verify-only", action="store_true", help="check the frozen configuration and exit (does not read test predictions)")
    ap.add_argument("--selftest-val", type=Path, metavar="OUT_DIR", help="dry-run the scoring/report pipeline on the cached VAL predictions")
    a = ap.parse_args()
    if a.verify_only:
        print(json.dumps(verify_frozen()["checks"], indent=1)); raise SystemExit(0)
    if a.selftest_val:
        vv = verify_frozen()
        r = score_and_report("val", ROOT / "experiments/task-e/full-training-config-a-200-s42/ignore_aware_val/best/val/labels", a.selftest_val, vv)
        _w(a.selftest_val / "selftest_val.json", r)
        print(json.dumps({k: {q: r["policies"][k][q] for q in ("precision", "recall", "f1", "fp_per_image", "AP50", "AP50_95")} for k in r["policies"]}, indent=1))
        raise SystemExit(0)
    if not a.authorized_once:
        raise SystemExit("REFUSED: pass --authorized-once (the test split is evaluated exactly once, after explicit approval)")
    r = run()
    print(json.dumps({k: r[k] for k in ("policies", "max_det_check", "fn_analysis_primary", "protected_trees_unchanged")}, indent=1, default=float))
