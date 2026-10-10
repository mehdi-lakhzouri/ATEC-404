"""Leakage-free, stratified 80/10/10 split at leakage-group level (deterministic)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
UNCERTAINTY_A = 0.25   # config A keeps scenes with REVIEW / (VALID + REVIEW) <= this


def density_bin(n_valid: int) -> str:
    return "d0" if n_valid == 0 else "d1_1-15" if n_valid <= 15 else "d2_16-35" if n_valid <= 35 else "d3_36+"


def scene_props(scene: dict[str, Any]) -> dict[str, Any]:
    v, r, x = int(scene["valid"]), int(scene["review_required"]), int(scene["rejected"])
    flags = scene["quality_flags"].split(";") if scene["quality_flags"] else []
    u = r / (v + r) if v + r else 0.0
    return {"n_valid": v, "n_review": r, "n_rejected": x, "uncertainty": u,
            "difficulty": "hard" if ("very_dark" in flags or "heavy_shadow" in flags) else "normal",
            "density": density_bin(v), "uncertainty_bin": "u_low" if u <= UNCERTAINTY_A else "u_high",
            "annotation_state": "has_valid" if v else ("records_no_valid" if r + x else "no_records")}


class _UF:
    def __init__(self) -> None:
        self.p: dict[Any, Any] = {}

    def find(self, a: Any) -> Any:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def leakage_groups(frames: list[int], task_c_group: dict[int, str], edges: list[tuple[int, int, int, str]]) -> dict[int, str]:
    uf = _UF()
    for f in frames:
        uf.find(f)
    by_c = defaultdict(list)
    for f in frames:
        by_c[task_c_group[f]].append(f)
    for members in by_c.values():
        for m in members[1:]:
            uf.union(members[0], m)
    for a, b, _, _ in edges:
        uf.union(a, b)
    comp = defaultdict(list)
    for f in frames:
        comp[uf.find(f)].append(f)
    out = {}
    for root, members in comp.items():
        name = f"G{min(members):04d}" if len(members) > 1 else f"S{members[0]:04d}"
        for m in members:
            out[m] = name
    return out


def assign(frames: list[int], props: dict[int, dict[str, Any]], groups: dict[int, str], seed: int) -> dict[int, str]:
    """Within each stratum, shuffle groups and give each to the split with the largest relative deficit."""
    gmembers = defaultdict(list)
    for f in frames:
        gmembers[groups[f]].append(f)
    # a multi-frame group takes the stratum of its lowest frame id (deterministic)
    gstrat = {g: "|".join(props[min(m)][k] for k in ("difficulty", "density", "uncertainty_bin")) for g, m in gmembers.items()}
    rng = np.random.default_rng(seed)
    split_of: dict[int, str] = {}
    for stratum in sorted(set(gstrat.values())):
        gs = sorted(g for g in gmembers if gstrat[g] == stratum)
        gs = [gs[k] for k in rng.permutation(len(gs))]
        got = Counter()
        for g in gs:
            n = sum(got.values()) + len(gmembers[g])
            s = max(FRACTIONS, key=lambda k: (FRACTIONS[k] * n - got[k], -list(FRACTIONS).index(k)))
            got[s] += len(gmembers[g])
            for f in gmembers[g]:
                split_of[f] = s
    return split_of


def stats(frames: list[int], split_of: dict[int, str], props: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in FRACTIONS:
        fs = [f for f in frames if split_of[f] == s]
        v = np.array([props[f]["n_valid"] for f in fs]); r = np.array([props[f]["n_review"] for f in fs])
        out[s] = {"frames": len(fs), "fraction": round(len(fs) / len(frames), 4), "valid_boxes": int(v.sum()), "review_boxes": int(r.sum()),
                  "valid_per_image_mean": round(float(v.mean()), 3), "uncertainty_mean": round(float(np.mean([props[f]["uncertainty"] for f in fs])), 4),
                  **{k: dict(sorted(Counter(props[f][k] for f in fs).items())) for k in ("difficulty", "density", "uncertainty_bin", "annotation_state")}}
    return out
