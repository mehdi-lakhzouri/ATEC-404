"""Strategy metrics, quality findings and decision file. Separates full-dataset statistics from sampled review."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..dataset_exploration.common import save_fig
from .config import AREA_THRESHOLDS, PRIMARY_VARIANT, PROPOSED_MIN_AREA, VARIANTS


def _rate(c: Counter, good: set[str]) -> dict[str, Any]:
    n = sum(c.values())
    return {"reviewed": n, "counts": dict(sorted(c.items())), "pass": sum(v for k, v in c.items() if k in good),
            "pass_rate": round(sum(v for k, v in c.items() if k in good) / n, 4) if n else None}


def review_metrics(assistant: dict[str, dict[str, dict[str, str]]], cand: list[dict[str, Any]], unc: list[dict[str, Any]],
                   manifest: list[dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {"reviewer": "AI assistant visual assessment (NOT human validation)"}
    a = assistant["candidates"]
    by = defaultdict(Counter); by_proxy = defaultdict(Counter); by_cls_thr = defaultdict(Counter)
    for c in cand:
        v = a.get(c["candidate_id"], {}).get("verdict")
        if not v:
            continue
        by[c["stratum"]][v] += 1
        by_proxy[c["visibility_proxy"]][v] += 1
        if c["stratum"].startswith(("small", "large_new")):
            for T in AREA_THRESHOLDS:
                if c["area_px"] >= T:
                    by_cls_thr[f"additional_ge_{T}px"][v] += 1
    good = {"VISIBLE_ROCK"}
    res["candidates"] = {
        "definition": "pass = VISIBLE_ROCK (a distinct rock visible in the render at native resolution, box tight within ~2px). "
                      "Denominator = reviewed candidates in the stratum. Strata are NOT proportional to population; do not average them.",
        "by_stratum": {k: _rate(v, good) for k, v in sorted(by.items())},
        "by_visibility_proxy": {k: _rate(v, good) for k, v in sorted(by_proxy.items())},
        "additional_candidates_by_min_area": {k: _rate(v, good) for k, v in sorted(by_cls_thr.items())},
        "split_merge_artifact_counts": dict(Counter(a[c["candidate_id"]]["verdict"] for c in cand if c["candidate_id"] in a)),
    }
    sc = assistant["scenes"]
    clean = {str(m["frame_id"]) for m in manifest if m["inclusion_status"] == "candidate_subset"}
    agg = {}
    for scope, ids in (("candidate_subset", [k for k in sc if k in clean]), ("anomaly_listed", [k for k in sc if k not in clean])):
        agg[scope] = {"scenes_assessed": len(ids)}
        for q in ("s1_misses_visible_rocks", "s2_recovers", "s2_tight", "s2_splits", "s2_merges", "s2_artifacts", "small_rocks_visible", "preferred"):
            agg[scope][q] = dict(Counter(sc[k].get(q, "") for k in ids))
    res["scenes"] = agg
    ua = assistant["uncertain"]
    byu = defaultdict(Counter)
    for u in unc:
        v = ua.get(u["uncertain_id"], {}).get("verdict")
        if v:
            byu[u["stratum"]][v] += 1
            byu["_all"][v] += 1
    res["uncertain_boxes"] = {k: dict(v) for k, v in sorted(byu.items())}
    res["terrain06"] = dict(Counter(r.get("verdict") for r in assistant["t06"].values()))
    ex = defaultdict(Counter)
    for r in assistant["exposure"].values():
        ex[r.get("exposure", "?")][r.get("verdict")] += 1
    res["exposure"] = {k: dict(v) for k, v in sorted(ex.items())}
    return res


def figures(out: Path, al: dict[str, Any], expo: dict[str, Any]) -> list[str]:
    F = out / "figures"
    files = []
    s2 = al["s2"]["candidate_subset"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    names = [v.name for v in VARIANTS]
    ax[0].bar(names, [al["variants"][n]["candidate_subset"]["match_rate"] for n in names], color=["#999", "#6a9", "#2a7"])
    ax[0].set(title="S1 CSV boxes matched by an S2 large component (IoU>=0.5)\ncandidate subset", ylabel="fraction of CSV boxes", ylim=(0, 1))
    for k, n in enumerate(names):
        ax[0].text(k, al["variants"][n]["candidate_subset"]["match_rate"] + 0.02, f"{al['variants'][n]['candidate_subset']['match_rate']:.3f}", ha="center")
    ax[1].bar([str(t) for t in AREA_THRESHOLDS], [s2[f"min_area_{t}"]["additional_small_medium_per_image"]["mean"] for t in AREA_THRESHOLDS], color="#3a3", label="small/medium")
    ax[1].bar([str(t) for t in AREA_THRESHOLDS], [s2[f"min_area_{t}"]["additional_large_without_csv_per_image"]["mean"] for t in AREA_THRESHOLDS],
              bottom=[s2[f"min_area_{t}"]["additional_small_medium_per_image"]["mean"] for t in AREA_THRESHOLDS], color="#85f", label="large w/o CSV")
    ax[1].axhline(al["candidate_subset"]["csv_boxes"] / al["candidate_subset"]["frames"], color="c", ls="--", label="S1 CSV boxes/image")
    ax[1].set(title="Mean additional S2 candidates per image\nvs minimum area (px, native 720x480)", xlabel="min area", ylabel="boxes / image"); ax[1].legend()
    bins = s2["small_area_bins"]
    ax[2].bar(list(bins), list(bins.values()), color="#3a3")
    ax[2].set(title="Small/medium (green) component areas\ncandidate subset, all frames", xlabel="area px", ylabel="components"); ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    files.append(save_fig(fig, F / "s1_vs_s2_counts.png"))

    be = expo["by_exposure"]
    exps = list(be)
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(exps))
    for cls, col in (("visible_fixed_linear", "#2a7"), ("visible_only_enhanced", "#eb3"), ("not_identifiable", "#d33")):
        v = np.array([be[e].get(cls, 0) for e in exps])
        ax.bar(exps, v, bottom=bottom, label=cls, color=col); bottom += v
    ax.set(title="POLAR-Sim rock boxes on matched NASA images: automated contrast proxy by exposure", xlabel="exposure (ms)", ylabel="fraction of rock boxes")
    ax.legend(fontsize=8)
    files.append(save_fig(fig, F / "nasa_rock_box_visibility_by_exposure.png"))
    return files


def decision(out: Path, evidence: dict[str, Any], human: dict[str, Any]) -> dict[str, Any]:
    p = out / "assistant_review" / "decision.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    approved = False  # only a human can approve; this code never sets it
    return {
        "task": "AI-01.B",
        "status": "PENDING_HUMAN_REVIEW",
        "recommended_strategy": d.get("recommended_strategy", "BLOCKED"),
        "target_ontology": d.get("target_ontology", "undecided"),
        "confidence": d.get("confidence", "low"),
        "s2_extraction_policy": d.get("s2_extraction_policy", {"variant": PRIMARY_VARIANT, "min_area_px": PROPOSED_MIN_AREA}),
        "evaluation_ontology": d.get("evaluation_ontology", ""),
        "evidence_summary": d.get("evidence_summary", []),
        "key_numbers": evidence,
        "limitations": d.get("limitations", []),
        "required_human_reviews": d.get("required_human_reviews", []),
        "decision_author": "AI assistant recommendation (not human-approved)",
        "human_review_status": human,
        "approved_for_training": approved,
        "next_task": "AI-01.C",
    }
