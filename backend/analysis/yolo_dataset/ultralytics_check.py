"""Independent check with the consumer's own code: Ultralytics image->label path mapping and label verifier.

Uses only offline helpers (no check_det_dataset: it may download fonts). No training, no cache files written.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml


def _verify(args: tuple[str, str]) -> tuple[int, int, int, int, int, str]:
    from ultralytics.data.utils import verify_image_label
    r = verify_image_label((args[0], args[1], "", False, 1, 0, 0, False))
    _, lb, shape, _, _, nm, nf, ne, nc, msg = r
    return nm, nf, ne, nc, 0 if lb is None else len(lb), msg


def run(out: Path, workers: int) -> dict[str, Any]:
    import ultralytics
    from ultralytics.data.utils import img2label_paths
    res: dict[str, Any] = {"ultralytics_version": ultralytics.__version__}
    for cfg in ("A", "B"):
        d = yaml.safe_load((out / f"data_{cfg}.yaml").read_text(encoding="utf-8"))
        base = Path(d["path"])
        for split in ("train", "val", "test"):
            imgs = sorted(str(p) for p in (base / d[split]).glob("*.png"))
            labs = img2label_paths(imgs)
            if cfg == "B" and split != "train":
                res[f"{cfg}_{split}"] = "same files as config A"
                continue
            with ProcessPoolExecutor(workers) as ex:
                rr = list(ex.map(_verify, zip(imgs, labs), chunksize=64))
            res[f"{cfg}_{split}"] = {"images": len(imgs), "labels_found": sum(r[1] for r in rr), "labels_missing": sum(r[0] for r in rr),
                                     "empty_labels": sum(r[2] for r in rr), "corrupt": sum(r[3] for r in rr), "boxes": sum(r[4] for r in rr),
                                     "messages": sorted({r[5] for r in rr if r[5]})[:5],
                                     "label_paths_resolve_into_export": all(Path(p).exists() for p in labs)}
    return res
