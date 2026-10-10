"""Run the frozen Artemis-X rock detector on one or more images and print (optionally save) the detections.

    uv run --project backend --no-sync python scripts/demo_inference.py path/to/frame.png [more.png ...] [--device cpu] [--save out.json]

Uses exactly the frozen settings (imgsz 736, conf 0.20, NMS IoU 0.6, max_det 300, boxes < 50 px^2 removed) and verifies
the model SHA-256 before loading. Do not point it at the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artemis_x.ml.inference import RockDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: CUDA when available)")
    ap.add_argument("--save", type=Path, help="write all detections to this JSON file")
    a = ap.parse_args()
    if any("images/test" in p.as_posix() for p in a.images):
        print("refusing: the held-out test split is reserved for the once-only evaluation", file=sys.stderr)
        return 2
    det = RockDetector(device=a.device)
    out = []
    for p in a.images:
        r = det.predict(p)
        print(f"{p.name}: {len(r.detections)} rocks (conf >= {det.spec.conf}), {r.inference_ms:.1f} ms on {r.device}")
        out.append({"image": p.name, "image_hw": list(r.image_hw), "inference_ms": round(r.inference_ms, 2), "device": r.device,
                    "detections": [{"xyxy": [round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1)], "confidence": round(d.confidence, 4),
                                    "class": d.class_name} for d in r.detections]})
    if a.save:
        a.save.write_text(json.dumps({"model_sha256": det.spec.sha256, "results": out}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
