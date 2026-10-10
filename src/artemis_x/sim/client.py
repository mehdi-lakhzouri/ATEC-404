"""Non-blocking inference client for the Webots controller (stdlib only).

Threading model: the Webots simulation thread only calls ``submit()`` and ``poll()``; it never waits on the network
and never shares Webots objects with the worker. A single background worker owns the socket.

Backpressure: one pending slot (a newer frame replaces an older unsent one -> counted as ``dropped``) and at most one
request in flight. Replies are matched on ``frame_id``; replies for any other id are discarded (``mismatched``).
Timeouts close the connection (a late reply can then never be confused with a newer frame) and the worker reconnects
with backoff. If the server is unavailable the controller simply receives no detections and the status says why.
"""

from __future__ import annotations

import collections
import socket
import threading
import time

from . import protocol


class DetectionClient:
    def __init__(self, host: str = protocol.DEFAULT_HOST, port: int = protocol.DEFAULT_PORT, request_timeout_s: float = 5.0,
                 connect_timeout_s: float = 1.0, retry_s: float = 1.0, expected_sha256: str | None = None, max_results: int = 16):
        self.addr = (host, port)
        self.request_timeout_s, self.connect_timeout_s, self.retry_s = request_timeout_s, connect_timeout_s, retry_s
        self.expected_sha256 = expected_sha256
        self._pending = None                                   # (frame_id, sim_time, w, h, rgb, t_submit)
        self._results = collections.deque(maxlen=max_results)
        self._cv = threading.Condition()
        self._stop = False
        self._sock = None
        self.hello = None
        self.status = {"connected": False, "sent": 0, "received": 0, "dropped": 0, "timeouts": 0, "mismatched": 0,
                       "server_errors": 0, "connect_failures": 0, "reconnects": 0, "last_error": None, "errors": collections.deque(maxlen=20)}
        self._thread = threading.Thread(target=self._run, name="artemis-inference-client", daemon=True)

    # ---- simulation-thread API ------------------------------------------------------------------------------------
    def start(self) -> "DetectionClient":
        self._thread.start()
        return self

    def submit(self, frame_id: int, sim_time_s: float, width: int, height: int, rgb: bytes, encoding: str = "rgb8") -> None:
        with self._cv:
            if self._pending is not None:
                self.status["dropped"] += 1
            self._pending = (frame_id, sim_time_s, width, height, rgb, time.perf_counter(), encoding)
            self._cv.notify()

    def poll(self) -> list[dict]:
        out = []
        while self._results:
            out.append(self._results.popleft())
        return out

    def close(self, timeout_s: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._close_socket(send_bye=True)
        self._thread.join(timeout_s)

    def snapshot(self) -> dict:
        s = dict(self.status)
        s["errors"] = list(self.status["errors"])
        s["model_sha256"] = (self.hello or {}).get("model_sha256")
        return s

    # ---- worker ------------------------------------------------------------------------------------------------------
    def _error(self, msg: str) -> None:
        self.status["last_error"] = msg
        self.status["errors"].append(f"{time.strftime('%H:%M:%S')} {msg}")

    def _close_socket(self, send_bye: bool = False) -> None:
        s, self._sock = self._sock, None
        self.status["connected"] = False
        if s is not None:
            try:
                if send_bye:
                    s.sendall(protocol.encode({"type": "bye"}))
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    def _connect(self) -> bool:
        try:
            s = socket.create_connection(self.addr, timeout=self.connect_timeout_s)
            s.settimeout(self.request_timeout_s)
            hello, _ = protocol.read_message(s)
            if hello.get("type") != "hello":
                raise protocol.ProtocolError(f"expected hello, got {hello.get('type')}")
            if self.expected_sha256 and hello.get("model_sha256") != self.expected_sha256:
                s.close()
                raise protocol.ProtocolError(f"server model sha256 {hello.get('model_sha256')} != expected frozen {self.expected_sha256}")
            self._sock, self.hello = s, hello
            self.status["connected"] = True
            return True
        except (OSError, protocol.ProtocolError, protocol.ConnectionClosed) as e:
            self.status["connect_failures"] += 1
            self._error(f"connect: {type(e).__name__}: {e}")
            return False

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._pending = self._pending, None
            if self._sock is None and not self._connect():
                self._results.append({"frame_id": job[0], "sim_time_s": job[1], "ok": False, "error": "server unavailable",
                                      "latency_ms": (time.perf_counter() - job[5]) * 1000})
                time.sleep(self.retry_s)
                continue
            fid, t_sim, w, h, rgb, t0, enc = job
            try:
                self._sock.sendall(protocol.frame_message(fid, t_sim, w, h, rgb, enc))
                self.status["sent"] += 1
                while True:
                    header, _ = protocol.read_message(self._sock)
                    if header.get("frame_id") == fid:
                        break
                    self.status["mismatched"] += 1                  # never attach a reply to the wrong frame
                lat = (time.perf_counter() - t0) * 1000
                if header.get("type") == "detections":
                    self.status["received"] += 1
                    self._results.append({"frame_id": fid, "sim_time_s": t_sim, "ok": True, "detections": header.get("detections", []),
                                          "inference_ms": header.get("inference_ms"), "device": header.get("device"),
                                          "model_sha256": header.get("model_sha256"), "latency_ms": lat})
                else:
                    self.status["server_errors"] += 1
                    self._results.append({"frame_id": fid, "sim_time_s": t_sim, "ok": False, "error": header.get("error", "server error"),
                                          "latency_ms": lat})
            except socket.timeout:
                self.status["timeouts"] += 1
                self._error(f"timeout after {self.request_timeout_s}s on frame {fid}")
                self._results.append({"frame_id": fid, "sim_time_s": t_sim, "ok": False, "error": "timeout", "latency_ms": (time.perf_counter() - t0) * 1000})
                self._close_socket(); self.status["reconnects"] += 1
            except (OSError, protocol.ConnectionClosed, protocol.ProtocolError) as e:
                self._error(f"request: {type(e).__name__}: {e}")
                self._results.append({"frame_id": fid, "sim_time_s": t_sim, "ok": False, "error": f"{type(e).__name__}: {e}",
                                      "latency_ms": (time.perf_counter() - t0) * 1000})
                self._close_socket(); self.status["reconnects"] += 1
