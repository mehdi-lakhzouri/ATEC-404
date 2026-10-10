"""Capture the ground station's WebSocket stream to JSONL (R5.5 validation; also a minimal client example).

    uv run --project ground_station python scripts/r5_ws_capture.py --url "ws://127.0.0.1:8000/ws/v1/missions/*" --out ws.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets


async def capture(url: str, out: str, duration: float | None) -> None:
    end = None if duration is None else time.time() + duration
    with open(out, "a", encoding="utf-8") as f:
        while end is None or time.time() < end:
            try:
                async with websockets.connect(url, open_timeout=5) as ws:   # one connection; reconnect only on failure
                    while end is None or time.time() < end:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except (asyncio.TimeoutError, TimeoutError):
                            continue                        # a quiet second keeps the connection
                        f.write(json.dumps({"client_rx_wall_s": round(time.time(), 6), "event": json.loads(msg)}) + "\n")
                        f.flush()
            except (OSError, websockets.exceptions.WebSocketException):
                await asyncio.sleep(0.5)                    # backend not up yet / restarting: reconnect


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float)
    a = ap.parse_args()
    try:
        asyncio.run(capture(a.url, a.out, a.duration))
    except KeyboardInterrupt:
        pass
