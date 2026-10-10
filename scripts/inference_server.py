"""Start the Artemis-X R3 inference server (frozen YOLO11n, 127.0.0.1:5761) in the ML environment.

    uv run --project backend --no-sync python scripts/inference_server.py [--port 5761] [--device cpu]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artemis_x.sim.inference_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
