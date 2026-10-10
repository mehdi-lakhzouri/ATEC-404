"""Inference with the frozen rock detector, exactly as configured in Task F.

Ultralytics ``predict`` with the frozen imgsz / conf / NMS IoU / max_det, followed by the frozen post-filter that
drops boxes smaller than ``min_box_area_px2``. Runs on CUDA when available, otherwise on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .model_registry import FrozenModelSpec, load_frozen_spec


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    class_name: str = "rock"

    @property
    def area(self) -> float:
        return max(self.x1 - self.x0, 0.0) * max(self.y1 - self.y0, 0.0)


@dataclass
class DetectionResult:
    detections: list[Detection]
    image_hw: tuple[int, int]
    inference_ms: float
    device: str
    removed_small_boxes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        """(N, 5) array: x0, y0, x1, y1, confidence."""
        return np.array([[d.x0, d.y0, d.x1, d.y1, d.confidence] for d in self.detections], dtype=np.float64).reshape(-1, 5)


def default_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present in this project
        return "cpu"


class RockDetector:
    """Frozen YOLO11n rock detector. ``RockDetector().predict(image)`` -> DetectionResult."""

    def __init__(self, spec: FrozenModelSpec | None = None, device: str | None = None):
        from ultralytics import YOLO
        self.spec = spec or load_frozen_spec()
        self.device = device or default_device()
        self.model = YOLO(str(self.spec.weights))
        names = {int(k): v for k, v in self.model.names.items()}
        if names != self.spec.class_names:
            raise RuntimeError(f"model class names {names} != frozen {self.spec.class_names}")

    def predict_rgb(self, rgb: np.ndarray) -> DetectionResult:
        """HxWx3 uint8 array in RGB order (PIL, Webots after BGRA->RGB). Converted to the BGR order Ultralytics expects."""
        return self.predict(np.ascontiguousarray(rgb[..., ::-1]))

    def predict(self, image: str | Path | np.ndarray) -> DetectionResult:
        """Image path, or HxWx3 uint8 array in **BGR** order (OpenCV convention, as Ultralytics assumes for numpy input).
        Use ``predict_rgb`` for RGB arrays."""
        src = str(image) if isinstance(image, (str, Path)) else image
        t0 = time.perf_counter()
        r = self.model.predict(src, device=self.device, verbose=False, **self.spec.predict_kwargs())[0]
        ms = (time.perf_counter() - t0) * 1000
        xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        dets = [Detection(*map(float, b), confidence=float(c)) for b, c in zip(xyxy, conf)]
        kept = [d for d in dets if d.area >= self.spec.min_box_area_px2]
        return DetectionResult(kept, tuple(r.orig_shape), ms, self.device, removed_small_boxes=len(dets) - len(kept),
                               meta={"speed_ms": dict(r.speed)})

    def warmup(self, hw: tuple[int, int] = (480, 720)) -> None:
        self.predict(np.zeros((*hw, 3), dtype=np.uint8))
