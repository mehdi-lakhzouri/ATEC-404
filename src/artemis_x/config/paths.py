"""Portable repository paths (pathlib only, no machine-specific locations).

Override with environment variables when data live elsewhere:
    ARTEMIS_ROOT       repository root (default: three levels above this file)
    ARTEMIS_DATA_ROOT  folder that contains the dataset versions (default: <root>/dataset)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARTEMIS_ROOT", Path(__file__).resolve().parents[3]))
BACKEND = ROOT / "backend"                       # historical research pipeline (Tasks A-F), uv project + lock
DATA_ROOT = Path(os.environ.get("ARTEMIS_DATA_ROOT", ROOT / "dataset"))
PROCESSED = DATA_ROOT / "processed"
YOLO_DATASET = PROCESSED / "yolo-rock-v1.1"      # canonical YOLO dataset (PROVISIONAL, not frozen)
CURATED_DATASET = PROCESSED / "curated-dataset-v1.1"
RAW_RENDERS = DATA_ROOT / "artificial-lunar" / "images" / "render"

EXPERIMENTS = ROOT / "experiments"               # local-only historical outputs (not published)
E1C_RUN = EXPERIMENTS / "task-e" / "full-training-config-a-200-s42"
SELECTION_DIR = EXPERIMENTS / "task-f" / "model-selection-001"
TEST_EVAL_DIR = EXPERIMENTS / "task-f" / "test-evaluation-001"

ARTIFACTS = ROOT / "artifacts"                   # published, pixel-free copies of the scientific evidence
MODELS = ROOT / "models"
FROZEN_MODEL = MODELS / "artemis_rock_yolo11n.pt"
FROZEN_CONFIG = ARTIFACTS / "provenance" / "FROZEN_MODEL_CONFIG.json"
AUTHORITATIVE_FROZEN_CONFIG = SELECTION_DIR / "FROZEN_MODEL_CONFIG.json"


def frozen_config_path() -> Path:
    """The authoritative frozen config when the local experiment folder exists, else its published copy."""
    return AUTHORITATIVE_FROZEN_CONFIG if AUTHORITATIVE_FROZEN_CONFIG.exists() else FROZEN_CONFIG


def ensure_backend_importable() -> None:
    """Make the proven research modules (``analysis.*``, ``training.*``) importable without moving them."""
    p = str(BACKEND)
    if BACKEND.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def rel(path: Path | str) -> str:
    """Repository-relative POSIX path for display (never print absolute local paths)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.name


def local_dataset_available() -> bool:
    return (YOLO_DATASET / "images" / "val").is_dir()
