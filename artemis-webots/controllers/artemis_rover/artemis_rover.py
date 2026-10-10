"""Webots controller for the Artemis-X rover (Koala). Thin entry point: the logic lives in src/artemis_x/sim.

Webots starts this file with its configured Python ("python" by default). Only the standard library and the Webots
``controller`` module are required.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from artemis_x.sim.rover import main  # noqa: E402

if __name__ == "__main__":
    main()
