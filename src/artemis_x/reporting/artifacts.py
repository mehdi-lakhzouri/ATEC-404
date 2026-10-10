"""Read-only loaders for the published scientific artifacts (``artifacts/``).

Every number shown in the notebooks comes from these files, which are verified copies of the historical
experiment outputs (see ``artifacts/provenance/ARTIFACT_MANIFEST.json``). Nothing here trains, evaluates or
reads the held-out test images.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.paths import ARTIFACTS, ROOT

SIZE_ORDER = ["small <400", "medium 400-2500", "large 2500-10000", "very large >=10000", "large (all) >=2500"]


def artifact(rel_path: str) -> Path:
    p = ARTIFACTS / rel_path
    if not p.exists():
        raise FileNotFoundError(f"artifacts/{rel_path} missing - run scripts/export_artifacts.py on a machine with the experiments/ folder")
    return p


@lru_cache(maxsize=None)
def load_json(rel_path: str) -> Any:
    return json.loads(artifact(rel_path).read_text(encoding="utf-8"))


def manifest() -> dict:
    return load_json("provenance/ARTIFACT_MANIFEST.json")


def provenance(rel_paths: list[str]) -> pd.DataFrame:
    """Which historical file each published artifact came from, with its SHA-256 (verified by scripts/verify_artifacts.py)."""
    files = {f["path"]: f for f in manifest()["files"]}
    rows = []
    for r in rel_paths:
        f = files["artifacts/" + r] if "artifacts/" + r in files else files[r]
        rows.append({"artifact": f["path"], "mode": f["mode"], "source (local experiments/)": f["source"], "sha256": f["sha256"][:16] + "..."})
    return pd.DataFrame(rows)


def frozen_config() -> dict:
    return load_json("provenance/FROZEN_MODEL_CONFIG.json")


def versions() -> dict:
    return load_json("provenance/VERSIONS.json")


# --- Task A-D ---------------------------------------------------------------------------------------------------

def dataset_overview() -> pd.DataFrame:
    a = load_json("data/artificial_lunar_analysis.json")
    st = load_json("data/split_statistics_v1.1.json")
    rows = [("Artificial Lunar scenes (render / ground / clean)", f"{a['files']['ids_present_in_all_three']:,} x 3"),
            ("Image size", "720 x 480 px, 8-bit RGB"),
            ("Original CSV rock boxes (large rocks only)", f"{a['bounding_boxes']['area_px']['count']:,}"),
            ("Near-duplicate render pairs (dHash <= 3)", f"{a['near_duplicates']['pairs']:,}"),
            ("Real lunar photos (PCAM/TCAM, qualitative only)", f"{a['real_moon_images']['counts']['images']}"),
            ("Frames eligible after anomaly exclusion", f"{st['split_validation']['frames']:,}")]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def quality_issues() -> pd.DataFrame:
    q = load_json("data/data_quality_report.json")
    return pd.DataFrame([{"id": i["id"], "severity": i["severity"], "source": i.get("dataset", ""), "issue": i["title"]} for i in q["issues"]])


def curated_counts() -> pd.DataFrame:
    v = versions()["versions"]
    rows = []
    for name in ("curated-dataset-v1", "curated-dataset-v1.1"):
        c = v[name]["counts"]
        rows.append({"version": name, "policy": v[name]["policy_version"], "status": v[name]["status"].split(" (")[0], **c})
    return pd.DataFrame(rows)


def split_table() -> pd.DataFrame:
    st = load_json("data/split_statistics_v1.1.json")
    rows = []
    for s in ("train", "val", "test"):
        d = st["splits"][s]
        rows.append({"split": s, "frames": d["frames"], "fraction": d["fraction"], "VALID boxes": d["valid_boxes"],
                     "REVIEW boxes": d["review_boxes"], "VALID / image": round(d["valid_per_image_mean"], 1),
                     "hard scenes": d["difficulty"].get("hard", 0)})
    a = st["configs"]["A_conservative"]
    rows.append({"split": "train, Config A (trained)", "frames": a["scenes"], "fraction": None, "VALID boxes": a["valid_boxes"],
                 "REVIEW boxes": a["review_boxes_in_scenes"], "VALID / image": round(a["valid_per_image_mean"], 1),
                 "hard scenes": a["difficulty"].get("hard", 0)})
    return pd.DataFrame(rows)


# --- Task E -----------------------------------------------------------------------------------------------------

def training_history() -> pd.DataFrame:
    df = pd.read_csv(artifact("training/results.csv"))
    df.columns = [c.strip() for c in df.columns]
    return df


def epoch_performance() -> pd.DataFrame:
    return pd.read_csv(artifact("training/per_epoch_performance.csv"))


def training_summary() -> dict:
    return load_json("training/e1c_summary.json")


def checkpoint_inventory() -> pd.DataFrame:
    inv = load_json("training/checkpoint_inventory.json")
    rows = [{"file": k, "completed_epochs": v["completed_epochs"], "MB": round(v["bytes"] / 2**20, 2), "names_ok": v["names_ok"],
             "finite": v["all_params_finite"], "resumable": v["resumable_training_state"], "sha256": v["sha256"][:12] + "..."}
            for k, v in inv.items()]
    return pd.DataFrame(rows).sort_values(["completed_epochs", "file"], na_position="last").reset_index(drop=True)


# --- Task F -----------------------------------------------------------------------------------------------------

def checkpoint_comparison() -> pd.DataFrame:
    d = load_json("evaluation/ignore_aware_val_22_checkpoints.json")["checkpoints"]
    rows = []
    for ck, v in d.items():
        p, s, n = v["primary_review_ignores"], v["secondary_review_plus_subthreshold_PROPOSED"], v["reference_no_ignores"]
        ep = int(ck[16:19]) if ck.startswith("epoch_completed_") else {"best.pt": 111, "last.pt": 200}.get(ck)
        rows.append({"checkpoint": ck, "epoch": ep, "AP50 primary": p["AP50"], "AP50-95 primary": p["AP50_95"],
                     "P@0.25": p["precision@conf"], "R@0.25": p["recall@conf"], "AP50 secondary": s["AP50"], "AP50 no-ignores": n["AP50"]})
    return pd.DataFrame(rows).sort_values(["epoch", "checkpoint"]).reset_index(drop=True)


def threshold_table(checkpoint: str = "best.pt") -> pd.DataFrame:
    return pd.DataFrame(load_json("evaluation/selection_analysis.json")["candidates"][checkpoint]["thresholds"])


def candidate_summary(conf: float = 0.20) -> pd.DataFrame:
    a = load_json("evaluation/selection_analysis.json")["candidates"]
    rows = []
    for ck, v in a.items():
        t = next(r for r in v["thresholds"] if r["conf"] == conf)
        rows.append({"checkpoint": ck, "AP50": v["AP50"], "AP50-95": v["AP50_95"], "precision": t["precision"], "recall": t["recall"],
                     "F1": t["f1"], "FP/image": t["fp_per_image"], "large recall": v["size_recall"]["large (all) >=2500"][f"recall@{conf}"],
                     "very-large recall": v["size_recall"]["very large >=10000"][f"recall@{conf}"]})
    return pd.DataFrame(rows)


def selection_rules() -> dict:
    return load_json("evaluation/selection_analysis.json")["rules"]


def test_results() -> dict:
    return load_json("evaluation/test_results.json")


def test_policy_table() -> pd.DataFrame:
    r = test_results()
    rows = []
    for name, p in r["policies"].items():
        rows.append({"policy": name, "precision": p["precision"], "recall": p["recall"], "F1": p["f1"], "FP/image": p["fp_per_image"],
                     "AP50": p["AP50"], "AP50-95": p["AP50_95"], "TP": p["tp"], "FP": p["fp"], "FN": p["fn"]})
    v = r["validation_reference_same_config"]
    rows.append({"policy": "VALIDATION (primary, same frozen config)", "precision": v["precision"], "recall": v["recall"], "F1": v["f1"],
                 "FP/image": v["fp_per_image"], "AP50": v["AP50"], "AP50-95": v["AP50_95"], "TP": None, "FP": None, "FN": None})
    return pd.DataFrame(rows)


def test_size_table() -> pd.DataFrame:
    r = test_results()
    p = r["policies"]["PRIMARY_review_ignores"]["recall_by_size"]
    v = r["validation_reference_same_config"]["recall_by_size"]
    return pd.DataFrame([{"size bin (px^2)": k, "GT (test)": p[k]["gt"], "recall test": p[k]["recall"], "recall val": v[k],
                          "precision test": p[k]["precision"], "AP50 test": p[k]["AP50"]} for k in SIZE_ORDER])


def repo_relative(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.name
