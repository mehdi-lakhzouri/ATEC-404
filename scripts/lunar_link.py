"""Artemis-X lunar network emulator (R5.3): run it, and change the link live without restarting Webots.

    python scripts/lunar_link.py serve --log artemis-webots/output/link/link.jsonl --stats artemis-webots/output/link/stats.json
        type at its prompt: NORMAL | DEGRADED | SEVERE | BLACKOUT | RESTORE | SET loss=0.1 latency_ms=50 | STATUS | QUIT
    python scripts/lunar_link.py ctl SEVERE            # from any other terminal (UDP control port 5792)
    python scripts/lunar_link.py ctl "SET loss=0.1 jitter_ms=20 bandwidth_Bps=2000"
    python scripts/lunar_link.py ctl STATUS

Rover telemetry must target the emulator: $env:ARTEMIS_TELEMETRY_PORT = "5791" (the ground receiver keeps 5790).
R5.4 store-and-forward: add --reverse here, --ack-to 127.0.0.1:5794 on the receiver, and $env:ARTEMIS_TELEMETRY_DTN = "1"
on the rover (ACKs come back on udp 5793).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artemis_x.telemetry.emulator import (CONTROL_PORT, GROUND_PORT, PRESETS, REVERSE_PORT, ROVER_ACK_PORT, ROVER_PORT,  # noqa: E402
                                          LunarLinkEmulator, send_control)

SHOW = ("preset", "link_state", "received", "forwarded", "dropped_loss", "dropped_blackout", "dropped_queue", "delivery_ratio",
        "blackout_total_s", "throughput_Bps", "delay_ms")


def serve(a) -> int:
    if a.log:
        a.log.parent.mkdir(parents=True, exist_ok=True)
    rev = (("127.0.0.1", REVERSE_PORT), ("127.0.0.1", ROVER_ACK_PORT)) if a.reverse else (None, None)
    em = LunarLinkEmulator(("127.0.0.1", a.listen), ("127.0.0.1", a.ground), ("127.0.0.1", a.control), preset=a.preset.upper(),
                           seed=a.seed, log_path=a.log, reverse_listen=rev[0], rover_ack=rev[1])
    stop = threading.Event()
    print(f"[link] rover udp:{a.listen} -> ground udp:{a.ground} | control udp:{a.control} | preset {a.preset.upper()} | seed {a.seed}")
    print(f"[link] commands: {' | '.join(PRESETS)} | SET k=v ... | STATUS | QUIT", flush=True)

    def console():
        for line in sys.stdin:
            cmd = line.strip()
            if not cmd:
                continue
            if cmd.upper() in ("QUIT", "EXIT"):
                stop.set()
                return
            try:
                if cmd.upper() in PRESETS:
                    em.apply(cmd)
                elif cmd.upper().startswith("SET "):
                    send_control(cmd, ("127.0.0.1", a.control))
                s = em.stats()
                print(json.dumps({k: s[k] for k in SHOW}), flush=True)
            except (ValueError, KeyError, OSError) as e:
                print(f"[link] error: {e}", flush=True)
    threading.Thread(target=console, daemon=True).start()
    try:
        stats = em.run(stop, a.duration)
    except KeyboardInterrupt:
        stats = em.close()
    if a.stats:
        a.stats.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in SHOW}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--listen", type=int, default=ROVER_PORT)
    s.add_argument("--ground", type=int, default=GROUND_PORT)
    s.add_argument("--control", type=int, default=CONTROL_PORT)
    s.add_argument("--preset", default="NORMAL", choices=[p for p in PRESETS] + [p.lower() for p in PRESETS])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--log", type=Path)
    s.add_argument("--stats", type=Path)
    s.add_argument("--duration", type=float)
    s.add_argument("--reverse", action="store_true", help="R5.4: also carry ground ACKs udp 5794 -> rover udp 5793 (same impairments)")
    c = sub.add_parser("ctl")
    c.add_argument("command", nargs="+")
    c.add_argument("--control", type=int, default=CONTROL_PORT)
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a)
    reply = send_control(" ".join(a.command), ("127.0.0.1", a.control))
    print(json.dumps({k: reply.get(k) for k in ("error",) + SHOW if k in reply}, indent=1))
    return 1 if "error" in reply else 0


if __name__ == "__main__":
    raise SystemExit(main())
