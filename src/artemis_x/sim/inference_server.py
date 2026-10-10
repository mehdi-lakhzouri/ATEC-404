"""Artemis-X R3 inference server: frozen YOLO11n over the local TCP protocol (runs in the ML environment, backend/.venv).

* Loads ``RockDetector`` with the authoritative frozen configuration; ``load_frozen_spec`` refuses weights whose SHA-256
  differs from the frozen value (historical checkpoint 413d0481..., never the sanitised copy).
* Settings are the frozen ones (imgsz 736, conf 0.20, NMS IoU 0.60, max_det 300, boxes < 50 px^2 removed) - not configurable.
* Binds 127.0.0.1 only; serves one client at a time; malformed/oversized messages get an error reply or a closed connection.
* ``--delay-ms`` adds artificial latency (responsiveness tests only).

Start:  uv run --project backend --no-sync python scripts/inference_server.py [--port 5761] [--device cpu|cuda:0]
"""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

from ..ml.inference import RockDetector
from ..ml.model_registry import load_frozen_spec
from . import protocol


class InferenceServer:
    def __init__(self, host: str = protocol.DEFAULT_HOST, port: int = protocol.DEFAULT_PORT, device: str | None = None,
                 delay_ms: float = 0.0, idle_timeout_s: float = 600.0):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("the inference server is local-only (127.0.0.1)")
        self.spec = load_frozen_spec()                       # verifies the frozen SHA-256
        self.det = RockDetector(self.spec, device=device)
        self.det.warmup()
        self.delay_ms, self.idle_timeout_s = delay_ms, idle_timeout_s
        self.sock = socket.create_server((host, port), reuse_port=False)
        self.port = self.sock.getsockname()[1]
        self.stats = {"frames": 0, "errors": 0, "clients": 0, "inference_ms": []}
        self._stop = False

    def hello(self) -> dict:
        return {"type": "hello", "protocol": protocol.VERSION, "model_sha256": self.spec.sha256,
                "inference": {"imgsz": self.spec.imgsz, "conf": self.spec.conf, "nms_iou": self.spec.nms_iou,
                              "max_det": self.spec.max_det, "min_box_area_px2": self.spec.min_box_area_px2},
                "class_names": self.spec.class_names, "device": self.det.device}

    def handle(self, conn: socket.socket) -> None:
        conn.settimeout(self.idle_timeout_s)
        conn.sendall(protocol.encode(self.hello()))
        while not self._stop:
            try:
                header, payload = protocol.read_message(conn)
            except (protocol.ConnectionClosed, socket.timeout, OSError):
                return
            except protocol.ProtocolError as e:              # framing is unreliable after this: reply and drop the client
                self.stats["errors"] += 1
                try:
                    conn.sendall(protocol.encode({"type": "error", "frame_id": None, "error": str(e)}))
                except OSError:
                    pass
                return
            kind = header.get("type")
            if kind == "bye":
                return
            if kind == "shutdown":
                self._stop = True
                return
            if kind != "frame":
                conn.sendall(protocol.encode({"type": "error", "frame_id": header.get("frame_id"), "error": f"unexpected type {kind!r}"}))
                continue
            try:
                protocol.validate_frame(header, payload)
            except protocol.ProtocolError as e:
                self.stats["errors"] += 1
                conn.sendall(protocol.encode({"type": "error", "frame_id": header.get("frame_id"), "error": str(e)}))
                continue
            rgb = np.frombuffer(payload, np.uint8).reshape(int(header["height"]), int(header["width"]), 3)
            t0 = time.perf_counter()
            res = self.det.predict_rgb(rgb)
            if self.delay_ms:
                time.sleep(self.delay_ms / 1000)
            ms = (time.perf_counter() - t0) * 1000
            self.stats["frames"] += 1
            self.stats["inference_ms"].append(round(ms, 2))
            conn.sendall(protocol.encode({
                "type": "detections", "frame_id": int(header["frame_id"]), "sim_time_s": header["sim_time_s"],
                "detections": [{"xyxy": [round(d.x0, 2), round(d.y0, 2), round(d.x1, 2), round(d.y1, 2)], "conf": round(d.confidence, 5)}
                               for d in res.detections],
                "removed_small_boxes": res.removed_small_boxes, "inference_ms": round(ms, 2), "model_ms": res.meta.get("speed_ms"),
                "device": res.device, "model_sha256": self.spec.sha256}))

    def serve_forever(self) -> None:
        self.sock.settimeout(0.5)
        print(json.dumps({"listening": f"127.0.0.1:{self.port}", **self.hello()}), flush=True)
        try:
            while not self._stop:
                try:
                    conn, _ = self.sock.accept()
                except socket.timeout:
                    continue
                self.stats["clients"] += 1
                with conn:
                    self.handle(conn)
        finally:
            self.sock.close()

    def stop(self) -> None:
        self._stop = True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--delay-ms", type=float, default=0.0, help="artificial extra latency (tests only)")
    a = ap.parse_args(argv)
    srv = InferenceServer(port=a.port, device=a.device, delay_ms=a.delay_ms)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(json.dumps({"stopped": True, "frames": srv.stats["frames"], "errors": srv.stats["errors"]}), flush=True)
    return 0
