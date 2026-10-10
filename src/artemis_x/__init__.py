"""Artemis-X perception: lunar rock detection with a frozen YOLO11n model.

Public, lightweight entry points. The research pipeline that produced the model (dataset analysis, cleaning,
YOLO export, guarded training, ignore-aware evaluation) lives in ``backend/`` and is wrapped, not duplicated.
"""

__version__ = "0.1.0"

from .ml.model_registry import FrozenModelSpec, load_frozen_spec  # noqa: E402,F401
