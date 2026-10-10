"""Print the software/hardware environment relevant to Artemis-X (read-only, no GPU kernels launched).

    uv run --project backend --no-sync python scripts/inspect_environment.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artemis_x.config import paths  # noqa: E402


def main() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 2), capability=f"sm_{p.major}{p.minor}",
                        arch_in_build=f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list())
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    info.update(frozen_model_present=paths.FROZEN_MODEL.exists(), local_dataset_present=paths.local_dataset_available(),
                local_experiments_present=paths.SELECTION_DIR.exists())
    return info


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
