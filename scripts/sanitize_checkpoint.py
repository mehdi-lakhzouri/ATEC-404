"""Produce a DISTRIBUTABLE copy of the frozen checkpoint without embedded local filesystem paths, and prove equivalence.

The historical frozen model (sha256 413d0481...) is only READ. The sanitised copy is written to --out (never into
models/ unless explicitly requested and approved). Only metadata strings change:
    train_args/model.args: data, project, save_dir  -> repository-relative placeholders
    git.root                                         -> None
Weights (model, ema), class names and every inference-relevant argument are untouched; the SHA-256 of the file changes
because the serialised metadata bytes change.

Equivalence checks (all must pass):
    1. every tensor in model/ema state_dict is bit-identical
    2. class names and inference-relevant args identical
    3. predictions identical (boxes, scores) on N validation images with the frozen settings (when the dataset is present)
    4. no local path remains in any metadata string

Usage:  uv run --project backend --no-sync python scripts/sanitize_checkpoint.py --out <dir>/artemis_rock_yolo11n_public.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.config import paths  # noqa: E402
from artemis_x.ml.model_registry import load_frozen_spec  # noqa: E402

PLACEHOLDERS = {"data": "experiments/task-e/full-training-config-a-200-s42/data_view/data_view.yaml",
                "project": "experiments/task-e", "save_dir": "experiments/task-e/full-training-config-a-200-s42"}
LOCAL = re.compile(r"[A-Za-z]:[\\/]Users|" + re.escape(Path.home().name))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _strings(o, seen=None):
    seen = seen or set()
    if id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v, seen)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v, seen)
    elif isinstance(o, str):
        yield o
    elif hasattr(o, "__dict__") and not isinstance(o, torch.Tensor):
        for k, v in vars(o).items():
            if k not in ("_parameters", "_buffers"):
                yield from _strings(v, seen)


def sanitise(src: Path, dst: Path) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)   # trusted: the project's own frozen checkpoint
    for holder in (ck.get("train_args"), getattr(ck.get("model"), "args", None), getattr(ck.get("ema"), "args", None)):
        if isinstance(holder, dict):
            for k, v in PLACEHOLDERS.items():
                if k in holder:
                    holder[k] = v
    if isinstance(ck.get("git"), dict):
        ck["git"] = {k: (None if k == "root" else v) for k, v in ck["git"].items()}
    ck["sanitised_from_sha256"] = _sha(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    return ck


def equivalence(src: Path, dst: Path, n_images: int) -> dict:
    a = torch.load(src, map_location="cpu", weights_only=False)
    b = torch.load(dst, map_location="cpu", weights_only=False)
    res = {}
    for key in ("model", "ema"):
        if a.get(key) is None:
            continue
        sa, sb = a[key].state_dict(), b[key].state_dict()
        res[f"{key}_tensors_identical"] = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    res["names_identical"] = a["model"].names == b["model"].names
    keep = [k for k in a["train_args"] if k not in PLACEHOLDERS]
    res["inference_args_identical"] = all(a["train_args"][k] == b["train_args"][k] for k in keep)
    res["no_local_paths_left"] = not any(LOCAL.search(s) for s in _strings({k: v for k, v in b.items() if k not in ("model", "ema")}))
    res["no_local_paths_left"] &= not any(LOCAL.search(s) for s in _strings(getattr(b["model"], "args", {})))
    if paths.local_dataset_available() and n_images:
        from ultralytics import YOLO
        spec = load_frozen_spec()
        ma, mb = YOLO(str(src)), YOLO(str(dst))
        imgs = sorted((paths.YOLO_DATASET / "images" / "val").glob("*.png"))[::max(1, 923 // n_images)][:n_images]
        same = 0
        for im in imgs:
            ra = ma.predict(str(im), verbose=False, **spec.predict_kwargs())[0].boxes
            rb = mb.predict(str(im), verbose=False, **spec.predict_kwargs())[0].boxes
            same += int(torch.equal(ra.xyxy.cpu(), rb.xyxy.cpu()) and torch.equal(ra.conf.cpu(), rb.conf.cpu()))
        res["predictions_identical"] = f"{same}/{len(imgs)}"
        res["predictions_ok"] = same == len(imgs)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=int, default=20)
    a = ap.parse_args()
    if a.out.resolve() in (paths.FROZEN_MODEL.resolve(),):
        raise SystemExit("refusing to overwrite the frozen model")
    src = load_frozen_spec().weights               # sha-verified frozen model
    before = _sha(src)
    sanitise(src, a.out)
    eq = equivalence(src, a.out, a.images)
    assert _sha(src) == before, "frozen model changed"
    out = {"frozen_sha256": before, "sanitised_sha256": _sha(a.out), "sanitised_bytes": a.out.stat().st_size, **eq}
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if all(v for k, v in eq.items() if isinstance(v, bool)) else 1)
