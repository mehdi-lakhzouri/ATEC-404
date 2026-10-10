"""The frozen Artemis-X rock detector: configuration and checksum-verified weights.

FROZEN_MODEL_CONFIG.json (Task F) is the single source of truth. This module only reads it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import paths


class ModelIntegrityError(RuntimeError):
    """Raised when the weights on disk do not match the frozen SHA-256."""


@dataclass(frozen=True)
class FrozenModelSpec:
    weights: Path
    sha256: str
    imgsz: int
    conf: float
    nms_iou: float
    max_det: int
    min_box_area_px2: float
    class_names: dict[int, str]
    source_checkpoint: str
    training_run: str
    epoch: int
    config_path: Path

    def predict_kwargs(self) -> dict:
        """Keyword arguments for ``ultralytics.YOLO.predict`` (the area post-filter is applied separately)."""
        return {"imgsz": self.imgsz, "conf": self.conf, "iou": self.nms_iou, "max_det": self.max_det}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _locate_weights(cfg: dict) -> Path:
    published = paths.FROZEN_MODEL
    if published.exists():
        return published
    return paths.ROOT / cfg["model"]["file"]          # historical location inside experiments/


def load_frozen_spec(config_path: Path | None = None, verify: bool = True) -> FrozenModelSpec:
    cfg_path = Path(config_path) if config_path else paths.frozen_config_path()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not cfg.get("frozen"):
        raise ModelIntegrityError(f"{paths.rel(cfg_path)} is not marked frozen")
    inf, model = cfg["inference"], cfg["model"]
    weights = _locate_weights(cfg)
    if verify:
        if not weights.exists():
            raise FileNotFoundError(f"frozen weights not found: {paths.rel(weights)}")
        got = sha256_file(weights)
        if got != model["sha256"]:
            raise ModelIntegrityError(f"{paths.rel(weights)} sha256 {got} != frozen {model['sha256']}")
    return FrozenModelSpec(weights=weights, sha256=model["sha256"], imgsz=int(inf["imgsz"]), conf=float(inf["conf"]),
                           nms_iou=float(inf["nms_iou"]), max_det=int(inf["max_det"]), min_box_area_px2=float(inf["min_box_area_px2"]),
                           class_names={int(k): v for k, v in model["class_names"].items()}, source_checkpoint=model["source_checkpoint"],
                           training_run=model["training_run"], epoch=int(model["ultralytics_best_epoch"]), config_path=cfg_path)
