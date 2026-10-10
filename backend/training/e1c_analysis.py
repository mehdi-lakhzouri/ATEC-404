"""Post-run analysis for a full Task E training run (non-training; reads the dataset, writes only into the run folder).

Stages (from backend/):
    uv run --no-sync python -m training.e1c_analysis --run-dir <run> --stage checkpoints   # integrity of every .pt
    uv run --no-sync python -m training.e1c_analysis --run-dir <run> --stage curves        # results.csv + telemetry plots, milestones
    uv run --no-sync python -m training.e1c_analysis --run-dir <run> --stage ignore-eval   # D.2 ignore-aware eval, VAL split only
    uv run --no-sync python -m training.e1c_analysis --run-dir <run> --stage inference     # compatibility check on val scenes

The ignore-aware stage uses training/evaluate.py unchanged (imgsz 736, conf 0.001, NMS IoU 0.6, max_det 300,
match IoU 0.5, min_pred_area 50 px^2) for every checkpoint. The test split is never touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/processed/yolo-rock-v1.1"
MILESTONES = (50, 100, 150, 200)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _w(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8", newline="\n")


def _ckpt_epoch(name: str, ck: dict) -> int | None:
    """Completed epochs held by a checkpoint (Ultralytics stores the 0-based index; stripped finals store -1)."""
    if name.startswith("epoch_completed_"):
        return int(name[16:19])
    e = ck.get("epoch")
    return None if e is None or e < 0 else int(e) + 1


def checkpoints(run: Path) -> dict[str, Any]:
    import torch
    out = {}
    for p in sorted((run / "weights").glob("*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)   # trusted: produced by this run
        m = ck["model"] if ck.get("model") is not None else ck["ema"]
        params = list(m.parameters())
        names = {int(k): v for k, v in m.names.items()}
        out[p.name] = {"bytes": p.stat().st_size, "sha256": _sha(p), "completed_epochs": _ckpt_epoch(p.name, ck),
                       "stored_epoch_field": ck.get("epoch"), "names": names, "names_ok": names == {0: "rock"},
                       "params": sum(x.numel() for x in params),
                       "all_params_finite": all(torch.isfinite(x.float()).all().item() for x in params),
                       "has_optimizer_state": ck.get("optimizer") is not None, "has_scaler_state": ck.get("scaler") is not None,
                       "has_ema": ck.get("ema") is not None, "updates": ck.get("updates"),
                       "resumable_training_state": ck.get("optimizer") is not None and ck.get("ema") is not None and ck.get("epoch", -1) is not None and ck.get("epoch", -1) >= 0,
                       "train_metrics": {k: float(v) for k, v in (ck.get("train_metrics") or {}).items()}}
    return out


def _results(run: Path) -> list[dict[str, float]]:
    rows = list(csv.DictReader((run / "results.csv").open(encoding="utf-8")))
    return [{k.strip(): float(v) for k, v in r.items()} for r in rows]


def _smi(run: Path) -> list[dict[str, Any]]:
    rows = list(csv.reader((run / "nvidia_smi.csv").open(encoding="utf-8")))
    head = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if len(r) != len(head) or r[0].strip() == head[0]:
            continue
        d = dict(zip(head, (c.strip() for c in r)))
        try:
            num = lambda k: float(d[k].split()[0])  # noqa: E731
            out.append({"t": datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                        "util": num("utilization.gpu [%]"), "mem_util": num("utilization.memory [%]"),
                        "vram": num("memory.used [MiB]"), "temp": num("temperature.gpu"), "power": num("power.draw [W]"),
                        "sm": num("clocks.current.sm [MHz]"),
                        "thermal": "Active" in (d["clocks_event_reasons.hw_thermal_slowdown"], d["clocks_event_reasons.sw_thermal_slowdown"]),
                        "powercap": d["clocks_event_reasons.sw_power_cap"] == "Active"})
        except (KeyError, ValueError):
            continue
    return out


def _stats(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 1), "median": round(float(np.median(a)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "max": round(float(a.max()), 1), "min": round(float(a.min()), 1)}


def curves(run: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = _results(run)
    meta = json.loads((run / "train_config.json").read_text(encoding="utf-8"))
    eps = [e for e in meta["epochs"] if "epoch_total_s" in e]
    smi = _smi(run)
    n_train = 1903
    # per-epoch telemetry: training window [t_start, t_start+train_s] and val+save window [.., t_end]
    per_epoch = []
    for e in eps:
        t0, t1, t2 = e["t_start"], e["t_start"] + e["train_s"], e["t_end"]
        tr = [s for s in smi if t0 <= s["t"] <= t1]
        va = [s for s in smi if t1 < s["t"] <= t2]
        per_epoch.append({"epoch": e["epoch"], "train_s": e["train_s"], "epoch_total_s": e["epoch_total_s"],
                          "val_save_s": round(e["epoch_total_s"] - e["train_s"], 2),
                          "img_per_s_epoch": round(n_train / e["train_s"], 1),
                          "img_per_s_batches": round(e["n_batches"] * 16 / e["train_batches_s"], 1) if e.get("train_batches_s") else None,
                          "gpu_util_train_median": float(np.median([s["util"] for s in tr])) if tr else None,
                          "gpu_util_train_mean": round(float(np.mean([s["util"] for s in tr])), 1) if tr else None,
                          "vram_max_mib": max((s["vram"] for s in tr + va), default=None),
                          "temp_max_c": max((s["temp"] for s in tr + va), default=None),
                          "peak_alloc_mb": e.get("peak_alloc_mb"), "peak_reserved_mb": e.get("peak_reserved_mb"),
                          "ram_available_gb_end": e.get("ram_available_gb")})
    # resources.csv: t_s is relative to the monitor start (~ first nvidia-smi sample; alignment +-2 s)
    rrows = list(csv.DictReader((run / "resources.csv").open(encoding="utf-8")))
    t_origin = smi[0]["t"] if smi else 0
    ram_used = [float(r["ram_used_gb"]) for r in rrows if r["ram_used_gb"]]
    ram_av = [float(r["ram_available_gb"]) for r in rrows if r.get("ram_available_gb")]
    workers = [int(r["dataloader_workers"]) for r in rrows if r["dataloader_workers"]]
    cpu = [float(r["cpu_pct"]) for r in rrows if r["cpu_pct"]]
    alloc = [float(r["cuda_alloc_mb"]) for r in rrows if r["cuda_alloc_mb"]]
    resv = [float(r["cuda_reserved_mb"]) for r in rrows if r["cuda_reserved_mb"]]
    train_windows = [(e["t_start"], e["t_start"] + e["train_s"]) for e in eps]
    in_train = lambda t: any(a <= t <= b for a, b in train_windows)  # noqa: E731
    smi_train = [s for s in smi if in_train(s["t"])]
    smi_val = [s for s in smi if not in_train(s["t"]) and eps and eps[0]["t_start"] <= s["t"] <= eps[-1]["t_end"]]
    tel = {"nvidia_smi_samples": len(smi),
           "gpu_util_train_window": _stats([s["util"] for s in smi_train]),
           "gpu_util_val_save_window": _stats([s["util"] for s in smi_val]),
           "gpu_mem_util_train_window": _stats([s["mem_util"] for s in smi_train]),
           "vram_mib": _stats([s["vram"] for s in smi]), "temp_c": _stats([s["temp"] for s in smi]),
           "power_w_train": _stats([s["power"] for s in smi_train]), "power_w_all": _stats([s["power"] for s in smi]),
           "sm_clock_mhz_train": _stats([s["sm"] for s in smi_train]),
           "thermal_slowdown_samples": sum(s["thermal"] for s in smi), "power_cap_samples": sum(s["powercap"] for s in smi),
           "cuda_alloc_mb_max": max(alloc, default=None), "cuda_reserved_mb_max": max(resv, default=None),
           "cpu_pct": _stats(cpu), "dataloader_workers": {"min": min(workers, default=None), "max": max(workers, default=None)},
           "ram_used_gb": _stats(ram_used), "ram_available_gb": _stats(ram_av),
           "ram_available_lt_2gb_samples": sum(1 for v in ram_av if v < 2), "ram_available_lt_4gb_samples": sum(1 for v in ram_av if v < 4)}
    # convergence / milestones
    col = lambda k: [r[k] for r in res]  # noqa: E731
    keys = {"box": "train/box_loss", "cls": "train/cls_loss", "dfl": "train/dfl_loss", "vbox": "val/box_loss", "vcls": "val/cls_loss",
            "vdfl": "val/dfl_loss", "P": "metrics/precision(B)", "R": "metrics/recall(B)", "m50": "metrics/mAP50(B)", "m5095": "metrics/mAP50-95(B)"}
    milestones = {}
    for m in MILESTONES:
        if len(res) < m:
            continue
        cur = res[m - 1]
        win = res[max(0, m - 20):m]
        slope = lambda k: float(np.polyfit([r["epoch"] for r in win], [r[keys[k]] for r in win], 1)[0]) * 10  # noqa: E731  per 10 epochs
        best_m = max(res[:m], key=lambda r: r[keys["m5095"]])
        milestones[m] = {**{k: round(cur[v], 4) for k, v in keys.items()},
                         "lr_pg0": cur["lr/pg0"],
                         "slope_per10ep_last20": {k: round(slope(k), 5) for k in ("box", "cls", "vbox", "vcls", "m50", "m5095")},
                         "best_mAP50_95_so_far": {"epoch": int(best_m["epoch"]), "value": round(best_m[keys["m5095"]], 4)},
                         "train_minus_val_cls_loss": round(cur[keys["cls"]] - cur[keys["vcls"]], 4)}
    best_row = max(res, key=lambda r: 0.1 * r[keys["m50"]] + 0.9 * r[keys["m5095"]]) if res else None   # Ultralytics fitness
    # plots
    out = run / "analysis"; out.mkdir(exist_ok=True)
    ep = col("epoch")
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, (tk, vk, t) in enumerate((("box", "vbox", "box loss"), ("cls", "vcls", "cls loss"), ("dfl", "vdfl", "DFL loss"))):
        ax[0, i].plot(ep, col(keys[tk]), label="train"); ax[0, i].plot(ep, col(keys[vk]), label="val"); ax[0, i].set_title(t)
        ax[0, i].axvline(len(ep) - 10 + 0.5 if len(ep) >= 10 else 0, color="gray", ls=":", lw=1, label="close_mosaic"); ax[0, i].legend()
    ax[1, 0].plot(ep, col(keys["P"]), label="precision"); ax[1, 0].plot(ep, col(keys["R"]), label="recall"); ax[1, 0].set_title("P / R (PROVISIONAL)"); ax[1, 0].legend()
    ax[1, 1].plot(ep, col(keys["m50"]), label="mAP50"); ax[1, 1].plot(ep, col(keys["m5095"]), label="mAP50-95"); ax[1, 1].set_title("Ultralytics val mAP (PROVISIONAL, no ignore regions)"); ax[1, 1].legend()
    for g in ("lr/pg0", "lr/pg1", "lr/pg2"):
        ax[1, 2].plot(ep, col(g), label=g)
    ax[1, 2].set_title("learning rate"); ax[1, 2].legend()
    for a in ax.flat:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.suptitle("E.1c YOLO11n Config A seed 42 - training curves (labels automated, PROVISIONAL)"); fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=110); plt.close(fig)
    if smi:
        t0 = smi[0]["t"]; tm = [(s["t"] - t0) / 60 for s in smi]
        fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
        ax[0].plot(tm, [s["util"] for s in smi], lw=.3); ax[0].set_ylabel("GPU util %")
        ax[1].plot(tm, [s["vram"] for s in smi], lw=.6); ax[1].set_ylabel("VRAM MiB")
        ax[2].plot(tm, [s["temp"] for s in smi], lw=.6, color="tab:red"); ax[2].set_ylabel("temp C")
        ax2 = ax[2].twinx(); ax2.plot(tm, [s["power"] for s in smi], lw=.3, color="tab:orange", alpha=.6); ax2.set_ylabel("power W")
        rt = [t_origin / 60 + float(r["t_s"]) / 60 - t0 / 60 for r in rrows]
        ax[3].plot(rt, [float(r["ram_used_gb"]) for r in rrows], lw=.6, label="RAM used GB")
        if ram_av:
            ax[3].plot(rt[:len(ram_av)], ram_av, lw=.6, label="RAM available GB")
        ax[3].set_ylabel("GB"); ax[3].legend(); ax[3].set_xlabel("minutes since start")
        for a in ax:
            a.grid(alpha=.3)
        fig.suptitle("E.1c hardware telemetry (1 s sampling)"); fig.tight_layout(); fig.savefig(out / "hardware_telemetry.png", dpi=110); plt.close(fig)
    if per_epoch:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        pe = [p["epoch"] for p in per_epoch]
        ax[0].plot(pe, [p["train_s"] for p in per_epoch], label="train"); ax[0].plot(pe, [p["val_save_s"] for p in per_epoch], label="val + save")
        ax[0].set_title("epoch duration (s)"); ax[0].legend()
        ax[1].plot(pe, [p["img_per_s_epoch"] for p in per_epoch], label="epoch-level"); ax[1].plot(pe, [p["img_per_s_batches"] for p in per_epoch], label="within batches")
        ax[1].set_title("training throughput (img/s)"); ax[1].legend()
        ax[2].plot(pe, [p["gpu_util_train_median"] for p in per_epoch], label="GPU util median (train)"); ax[2].set_title("GPU util per epoch (%)")
        for a in ax:
            a.set_xlabel("epoch"); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "epoch_performance.png", dpi=110); plt.close(fig)
    tot = [p["epoch_total_s"] for p in per_epoch]
    summary = {"epochs_in_results_csv": len(res), "epochs_recorded": len(eps), "status": meta.get("status"),
               "started": meta.get("started"), "finished": meta.get("finished"),
               "wall_clock_h_epochs": round(sum(tot) / 3600, 3) if tot else None,
               "epoch_duration_s": _stats(tot), "train_s": _stats([p["train_s"] for p in per_epoch]),
               "val_save_s": _stats([p["val_save_s"] for p in per_epoch]),
               "img_per_s_epoch": _stats([p["img_per_s_epoch"] for p in per_epoch]),
               "img_per_s_batches": _stats([p["img_per_s_batches"] for p in per_epoch if p["img_per_s_batches"]]),
               "final_eval_best_pt": meta.get("final_eval_best_pt"), "final_eval_s": meta.get("final_eval_s"),
               "ultralytics_best_epoch_by_fitness": int(best_row["epoch"]) if best_row else None,
               "resource_warnings": meta.get("resource_warnings"), "telemetry": tel, "milestones": milestones,
               "last_row": res[-1] if res else None, "max_mAP50": max(col(keys["m50"]), default=None),
               "max_mAP50_95": max(col(keys["m5095"]), default=None)}
    with (out / "per_epoch_performance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_epoch[0])); w.writeheader(); w.writerows(per_epoch)
    return summary


def ignore_eval(run: Path) -> dict[str, Any]:
    from .evaluate import predict, score
    from analysis.evaluation.ignore_eval import evaluate, load_split
    out = run / "ignore_aware_val"; out.mkdir(exist_ok=True)
    cks = sorted((run / "weights").glob("epoch_completed_*.pt")) + [run / "weights/best.pt", run / "weights/last.pt"]
    res = {"policy": {"split": "val (923 images)", "predict": "imgsz 736, conf 0.001, NMS IoU 0.6, max_det 300 (training/evaluate.py)",
                      "match": "IoU 0.5 greedy by score; ignore = COCO-crowd IoA >= 0.5 on REVIEW regions; 101-pt AP; min_pred_area 50 px^2",
                      "primary": "review_ignores", "labels": "PROVISIONAL automated S2-H VALID boxes (not human-verified)"},
           "checkpoints": {}}
    for ck in cks:
        t = time.time()
        pd = predict(ck, "val", out / ck.stem)
        r = {"primary_review_ignores": score(pd, "val", subthreshold=False),
             "secondary_review_plus_subthreshold_PROPOSED": score(pd, "val", subthreshold=True)}
        r["reference_no_ignores"] = evaluate(load_split(DS, "val", pd, []), min_pred_area=50.0)
        r["seconds"] = round(time.time() - t, 1)
        res["checkpoints"][ck.name] = r
        p = r["primary_review_ignores"]
        print(ck.name, {k: round(p[k], 4) for k in p if isinstance(p[k], float)}, flush=True)
        _w(run / "ignore_aware_val.json", res)
    return res


def inference(run: Path) -> dict[str, Any]:
    from .smoke_report_data import SCENES, inference as infer
    man = {int(r["frame_id"]): r["split"] for r in csv.DictReader((DS / "manifests/split_manifest.csv").open(encoding="utf-8"))}
    val_scenes = [s for s in SCENES if man.get(s[0]) == "val"]
    return infer(run, scenes=val_scenes, imgsz=736)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=("checkpoints", "curves", "ignore-eval", "inference"), required=True)
    a = ap.parse_args()
    run = a.run_dir.resolve()
    fn = {"checkpoints": checkpoints, "curves": curves, "ignore-eval": ignore_eval, "inference": inference}[a.stage]
    data = fn(run)
    name = {"checkpoints": "checkpoint_inventory.json", "curves": "analysis/summary.json", "ignore-eval": "ignore_aware_val.json",
            "inference": "inference_check.json"}[a.stage]
    _w(run / name, data)
    print(json.dumps(data, indent=1, default=float)[:6000])
