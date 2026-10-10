"""Artemis-X ground station receiver (R5.2): UDP telemetry -> validated, decoded JSONL. Standard library only.

    python scripts/ground_receiver.py --out artemis-webots/output/ground/telemetry.jsonl            # until Ctrl+C
    python scripts/ground_receiver.py --port 5790 --duration 600 --out <file> --stats <file.json>

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
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rx = GroundReceiver(a.host, a.port, a.out)
    print(f"[ground] listening on udp://{a.host}:{a.port} -> {a.out}", flush=True)
    try:
        stats = rx.run(a.duration)
    except KeyboardInterrupt:
        stats = rx.close()
    if a.stats:
        a.stats.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "sessions"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
