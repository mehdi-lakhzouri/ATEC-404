"""Shared helpers: paths, logging, deterministic JSON output, small geometry utils."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SEED = 20260101
LOG = logging.getLogger("dataset_exploration")


@dataclass(frozen=True)
class Paths:
    project_root: Path
    out_dir: Path

    @property
    def dataset_root(self) -> Path:
        return self.project_root / "dataset"

    @property
    def artificial_lunar(self) -> Path:
        return self.dataset_root / "artificial-lunar"

    @property
    def nasa_polar(self) -> Path:
        return self.dataset_root / "nasa-polar"

    @property
    def polar_sim(self) -> Path:
        return self.dataset_root / "polar-sim-source"

    @property
    def figures(self) -> Path:
        return self.out_dir / "figures"

    @property
    def samples(self) -> Path:
        return self.out_dir / "samples"

    @property
    def logs(self) -> Path:
        return self.out_dir / "logs"

    def ensure(self) -> None:
        for p in (self.out_dir, self.figures, self.samples, self.logs):
            p.mkdir(parents=True, exist_ok=True)

    def rel(self, p: Path) -> str:
        """Project-relative POSIX path, so outputs are machine-independent."""
        try:
            return p.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()


@dataclass
class ErrorLog:
    """Collects non-fatal failures so nothing is silently discarded."""

    items: list[dict[str, str]] = field(default_factory=list)

    def add(self, where: str, path: str, error: str) -> None:
        LOG.warning("%s | %s | %s", where, path, error)
        self.items.append({"stage": where, "path": path, "error": error})


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        seq = sorted(o) if isinstance(o, set) else o
        return [_jsonable(v) for v in seq]
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path: Path, data: Any) -> None:
    """Deterministic JSON: sorted keys, rounded floats, no timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def describe(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(a.size),
        "min": a.min(),
        "max": a.max(),
        "mean": a.mean(),
        "std": a.std(),
        "p01": q[0], "p05": q[1], "p25": q[2], "median": q[3],
        "p75": q[4], "p95": q[5], "p99": q[6],
    }


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes in x0,y0,x1,y1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """Deterministic greedy one-to-one matching by descending IoU."""
    pairs: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return pairs
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None, kind="stable"), iou.shape))[0]
    used_a: set[int] = set()
    used_b: set[int] = set()
    for i, j in order:
        v = iou[i, j]
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        pairs.append((int(i), int(j), float(v)))
    return pairs


def tone_map_12bit(a: np.ndarray) -> np.ndarray:
    """Percentile stretch of a 16-bit (12-bit effective) image to uint8 for viewing only."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None, "CreationDate": None})
    plt.close(fig)
    return path.name
