"""Artemis-X ground station backend (R5.5).

Decoded telemetry comes ONLY from the R5.4 ``GroundReceiver`` (validated, CRC-checked, ACKed, reconciled by
``GroundMap``). This package adds persistence (MongoDB), versioned REST and WebSocket APIs, and local network controls
for the R5.3 emulator. It never reads Webots ground truth or rover-side state.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:          # reuse the rover-side telemetry package (protocol, receiver, map)
    sys.path.insert(0, str(REPO / "src"))

API_VERSION = "v1"
__version__ = "0.5.5"
