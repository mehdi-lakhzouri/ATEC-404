"""Ignore-aware evaluation of a trained checkpoint (prepared for Task E; needs a trained model to run).

Predictions are produced with Ultralytics (`save_txt`, `save_conf`, conf=0.001) on the UNMASKED val/test images and
scored with analysis.evaluation.ignore_eval. Ultralytics' own val mAP is reported as PROVISIONAL reference only.

Usage (from backend/):
    uv run --no-sync python -m training.evaluate --weights <run>/weights/best.pt --split val [--subthreshold]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"


def predict(weights: Path, split: str, out: Path, imgsz: int = 736) -> Path:  # pragma: no cover - needs a model
    from ultralytics import YOLO
    YOLO(str(weights)).predict(source=str(DS / "images" / split), imgsz=imgsz, conf=0.001, iou=0.6, max_det=300,
                               save=False, save_txt=True, save_conf=True, project=str(out), name=split, exist_ok=True, verbose=False)
    return out / split / "labels"


def score(pred_dir: Path, split: str, subthreshold: bool, min_pred_area: float = 50.0) -> dict:
    from analysis.evaluation.ignore_eval import evaluate, load_split
    ign = [DS / "manifests/eval_ignore_regions.json"]
    if subthreshold:
        ign.append(DS / "manifests/eval_ignore_regions_subthreshold_PROPOSED.json")
    r = evaluate(load_split(DS, split, pred_dir, ign), min_pred_area=min_pred_area)
    r["ignore_files"] = [p.name for p in ign]
    r["labels"] = "PROVISIONAL automated S2-H VALID boxes (not human-verified)"
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--subthreshold", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/task-e/eval")
    a = ap.parse_args()
    pd = predict(a.weights, a.split, a.out)
    res = score(pd, a.split, a.subthreshold)
    (a.out / f"ignore_aware_{a.split}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))
