"""Artemis-X ground station receiver (R5.2): UDP telemetry -> validated, decoded JSONL. Standard library only.

    python scripts/ground_receiver.py --out artemis-webots/output/ground/telemetry.jsonl            # until Ctrl+C
    python scripts/ground_receiver.py --port 5790 --duration 600 --out <file> --stats <file.json>
    python scripts/ground_receiver.py --out <file> --ack-to 127.0.0.1:5794 --map-out <map.json>    # R5.4: ACKs + reconciled map

Every datagram becomes one JSON line: reception wall time, session, sequence, type, observation time, seq_flag
(ok / first / gap:N / duplicate / late) and the decoded body - or ``invalid`` with the reason. Stats are printed and
written at exit (Ctrl+C included).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.telemetry.transport import DEFAULT_HOST, DEFAULT_PORT, GroundReceiver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", type=Path, required=True, help="JSONL output (appended)")
    ap.add_argument("--stats", type=Path, help="JSON stats written at exit")
    ap.add_argument("--duration", type=float, help="seconds; default: until Ctrl+C")
    ap.add_argument("--ack-to", help="R5.4: send selective ACKs to host:port (the emulator's reverse port 5794, or the rover's 5793)")
    ap.add_argument("--map-out", type=Path, help="R5.4: write the reconciled ground obstacle map (JSON) at exit")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ack = None
    if a.ack_to:
        h, port = a.ack_to.rsplit(":", 1)
        ack = (h, int(port))
    rx = GroundReceiver(a.host, a.port, a.out, ack_to=ack, reconcile=a.map_out is not None)
    print(f"[ground] listening on udp://{a.host}:{a.port} -> {a.out}", flush=True)
    try:
        stats = rx.run(a.duration)
    except KeyboardInterrupt:
        stats = rx.close()
    gmap = stats.pop("ground_map", None)
    if a.map_out and gmap is not None:
        a.map_out.write_text(json.dumps(gmap, indent=1), encoding="utf-8")
    if a.stats:
        a.stats.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "sessions"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
