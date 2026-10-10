"""Post-run evidence for a smoke test: checkpoint checks, sample inference, provisional metrics, monitoring summary.

Reads only the dataset; writes only into the run folder.
Usage (from backend/):  uv run --no-sync python -m training.smoke_report_data --run-dir ../experiments/task-e/smoke-test-001
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"
SCENES = [(189, "sparse"), (115, "sparse, many REVIEW"), (9433, "dense (111 VALID)"), (5589, "dense"), (575, "hard: dark/shadow"),
          (586, "hard: dark/shadow"), (4357, "difficult: 155 REVIEW"), (6352, "difficult: shadowed boulders"), (5311, "no records (dim-mask rocks)")]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _f(x: str) -> float | None:
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def monitoring(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    p = run / "nvidia_smi.csv"
    if p.exists():
        rows = list(csv.reader(p.open(encoding="utf-8")))
        head = [h.strip() for h in rows[0]]
        data = [dict(zip(head, [c.strip() for c in r])) for r in rows[1:] if len(r) == len(head)]
        # newer drivers rename clocks_throttle_reasons.* -> clocks_event_reasons.* and clocks.sm -> clocks.current.sm
        col = lambda suffix: next((h for h in head if h.endswith(suffix)), None)  # noqa: E731
        num = lambda key: [v for v in (_f(d[key].split()[0]) for d in data if key and d.get(key)) if v is not None]  # noqa: E731
        temp, mem, util, pw, clk = (num(col(k)) for k in ("temperature.gpu", "memory.used [MiB]", "utilization.gpu [%]", "power.draw [W]", ".sm [MHz]"))
        hw, sw, pc = col("hw_thermal_slowdown"), col("sw_thermal_slowdown"), col("sw_power_cap")
        missing = [n for n, c in (("hw_thermal_slowdown", hw), ("sw_thermal_slowdown", sw), ("sw_power_cap", pc)) if c is None]
        if missing:
            raise KeyError(f"nvidia-smi columns not found: {missing}; header={head}")
        active = lambda d, c: d.get(c, "").strip() == "Active"  # noqa: E731  ("Not Active" must not match)
        thermal = [d for d in data if active(d, hw) or active(d, sw)]
        powercap = [d for d in data if active(d, pc)]
        out["nvidia_smi"] = {"samples": len(data), "interval_s": 2,
                             "temp_c": {"max": max(temp, default=None), "mean": round(float(np.mean(temp)), 1) if temp else None},
                             "vram_used_mib": {"max": max(mem, default=None), "total": data[0].get("memory.total [MiB]") if data else None},
                             "util_pct": {"mean": round(float(np.mean(util)), 1) if util else None, "max": max(util, default=None)},
                             "power_w": {"max": max(pw, default=None), "mean": round(float(np.mean(pw)), 1) if pw else None},
                             "sm_clock_mhz": {"max": max(clk, default=None), "min": min(clk, default=None)},
                             "thermal_slowdown_samples": len(thermal), "power_cap_samples": len(powercap)}
    p = run / "resources.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        ram = [float(r["ram_used_gb"]) for r in rows if r["ram_used_gb"]]
        out["ram_used_gb"] = {"min": min(ram, default=None), "max": max(ram, default=None)}
    return out


def results_csv(run: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8")))
    return [{k.strip(): (_f(v) if k.strip() != "epoch" else int(float(v))) for k, v in r.items()} for r in rows]


def checkpoints(run: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    res = {}
    for n in ("best.pt", "last.pt"):
        p = run / "weights" / n
        if not p.exists():
            res[n] = {"exists": False}; continue
        # trusted: a checkpoint produced by this run; Ultralytics checkpoints hold pickled train args (YOLO() loads them the same way)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = YOLO(str(p))
        res[n] = {"exists": True, "bytes": p.stat().st_size, "sha256": _sha(p), "epoch_field": ck.get("epoch"),
                  "best_fitness": float(ck["best_fitness"]) if ck.get("best_fitness") is not None else None,
                  "names": m.names, "params": sum(x.numel() for x in m.model.parameters()),
                  "all_params_finite": all(torch.isfinite(x).all().item() for x in m.model.parameters())}
    return res


def inference(run: Path, scenes: list[tuple[int, str]] = SCENES, imgsz: int = 640) -> dict[str, Any]:
    from ultralytics import YOLO
    from analysis.evaluation.ignore_eval import ImageEval, match_image
    man = {int(r["frame_id"]): r for r in csv.DictReader((DS / "manifests/split_manifest.csv").open(encoding="utf-8"))}
    ign = json.loads((DS / "manifests/eval_ignore_regions.json").read_text(encoding="utf-8"))["regions"]
    model = YOLO(str(run / "weights/best.pt"))
    out_dir = run / "inference"; out_dir.mkdir(exist_ok=True)
    rows, tiles = [], []
    for fid, kind in scenes:
        split = man[fid]["split"]
        img_p = DS / "images" / split / f"render{fid:04d}.png"
        r = model.predict(str(img_p), imgsz=imgsz, conf=0.25, device=0, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy()
        gt = np.array([[float(v) for v in ln.split()[1:]] for ln in (DS / "labels" / split / f"render{fid:04d}.txt").read_text().splitlines()]).reshape(-1, 4)
        gt = np.c_[(gt[:, 0] - gt[:, 2] / 2) * 720, (gt[:, 1] - gt[:, 3] / 2) * 480, (gt[:, 0] + gt[:, 2] / 2) * 720, (gt[:, 1] + gt[:, 3] / 2) * 480] if len(gt) else gt
        rv = np.array([x[:4] for x in ign[split].get(f"render{fid:04d}.png", [])]).reshape(-1, 4)
        sc, lab, ngt = match_image(ImageEval(gt, np.c_[b, s] if len(b) else np.zeros((0, 5)), rv), 0.5)
        im = Image.open(img_p).convert("RGB"); d = ImageDraw.Draw(im)
        for g in gt:
            d.rectangle([g[0], g[1], g[2] - 1, g[3] - 1], outline=(0, 200, 0), width=1)
        for (x0, y0, x1, y1), c in zip(b, s):
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 255), width=2)
            d.text((x0 + 2, max(0, y0 - 11)), f"{c:.2f}", fill=(255, 0, 255))
        d.rectangle([0, 0, 719, 15], fill=(0, 0, 0))
        d.text((3, 2), f"f{fid:04d} {split} {kind} | pred(conf>=.25, magenta)={len(b)} | VALID GT (green)={len(gt)} | REVIEW={len(rv)} | "
                       f"TP {int((lab == 1).sum())} FP {int((lab == 0).sum())} ignored {len(b) - len(lab)}", fill=(255, 255, 255))
        im.save(out_dir / f"pred_{fid:04d}.jpg", quality=92)
        tiles.append(im)
        rows.append({"frame_id": fid, "split": split, "kind": kind, "predictions": int(len(b)), "valid_gt": int(len(gt)), "review": int(len(rv)),
                     "tp_iou50": int((lab == 1).sum()), "fp": int((lab == 0).sum()), "ignored_on_review": int(len(b) - len(lab)),
                     "conf_max": round(float(s.max()), 3) if len(s) else None})
    W, H = 720, 480
    sheet = Image.new("RGB", (W * 3, H * 3), (20, 20, 20))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 3) * W, (k // 3) * H))
    sheet.resize((W * 3 // 2, H * 3 // 2), Image.BILINEAR).save(out_dir / "inference_contact_sheet.jpg", quality=90)
    return {"conf": 0.25, "imgsz": imgsz, "model_names": {int(k): v for k, v in model.names.items()}, "images": rows}


def ignore_aware_val(run: Path) -> dict[str, Any]:
    from ultralytics import YOLO
    from analysis.evaluation.ignore_eval import evaluate, load_split
    pred_root = run / "ignore_aware_val"
    YOLO(str(run / "weights/best.pt")).predict(source=str(DS / "images/val"), imgsz=640, conf=0.001, iou=0.6, max_det=300, device=0,
                                               save=False, save_txt=True, save_conf=True, project=str(pred_root), name="val", exist_ok=True, verbose=False)
    res = {}
    for name, files in (("review_ignores", [DS / "manifests/eval_ignore_regions.json"]),
                        ("review_plus_subthreshold_PROPOSED", [DS / "manifests/eval_ignore_regions.json", DS / "manifests/eval_ignore_regions_subthreshold_PROPOSED.json"]),
                        ("no_ignores_naive", [])):
        r = evaluate(load_split(DS, "val", pred_root / "val" / "labels", files), min_pred_area=50.0)
        res[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "by_size"}
        res[name]["by_size_AP50"] = {k: round(v["AP50"], 4) for k, v in r["by_size"].items()}
    res["note"] = "PROVISIONAL: 3-epoch smoke model; labels are automated, not human-verified"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    run = ap.parse_args().run_dir.resolve()
    data = {"results_csv": results_csv(run), "checkpoints": checkpoints(run), "monitoring": monitoring(run),
            "inference": inference(run), "ignore_aware_val": ignore_aware_val(run)}
    (run / "smoke_report_data.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(json.dumps(data, indent=1, default=str)[:7000])
