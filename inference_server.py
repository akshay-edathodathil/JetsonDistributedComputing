"""
inference_server.py — Lightweight TCP inference server for Jetson workers.

Runs on each Jetson using only system-installed packages (torch, ultralytics,
opencv, numpy — all already present). No Ray or new installs needed.

The GPU PC connects to this server and sends frames; results come back over
the same socket connection.

Wire protocol (both directions):
    4-byte big-endian length prefix | pickle.dumps(payload)

Usage:
    python3 inference_server.py --config config.yaml
    python3 inference_server.py --model-type yolov8 --model-path models/yolo.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any

# models.py is rsync'd alongside this file
from models import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("inference_server")


# ── GPU utilization ───────────────────────────────────────────────────────────

def _gpu_util() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return -1.0


# ── Socket helpers ────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _send_payload(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_payload(sock: socket.socket) -> Any | None:
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    data = _recv_exact(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


# ── Client handler ────────────────────────────────────────────────────────────

class ClientHandler(threading.Thread):
    def __init__(self, conn: socket.socket, addr: tuple, model: ModelLoader,
                 stats: dict, stats_lock: threading.Lock) -> None:
        super().__init__(daemon=True)
        self._conn = conn
        self._addr = addr
        self._model = model
        self._stats = stats
        self._stats_lock = stats_lock

    def run(self) -> None:
        logger.info("Client connected: %s", self._addr)
        try:
            while True:
                request = _recv_payload(self._conn)
                if request is None:
                    break

                frame = request["frame"]
                frame_id = request["frame_id"]
                capture_ts = request["capture_ts"]

                result = self._model.infer_timed(frame)
                result.frame_id = frame_id

                response = {
                    "frame_id": frame_id,
                    "detections": [
                        {
                            "bbox": list(d.bbox),
                            "confidence": d.confidence,
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "keypoints": d.keypoints,
                        }
                        for d in result.detections
                    ],
                    "inference_ms": result.inference_ms,
                    "capture_ts": capture_ts,
                    "result_ts": time.time(),
                    "gpu_util_pct": _gpu_util(),
                }

                _send_payload(self._conn, pickle.dumps(response))

                with self._stats_lock:
                    self._stats["frames"] += 1
                    self._stats["total_ms"] += result.inference_ms

        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        except Exception as exc:
            logger.error("Handler error: %s", exc)
        finally:
            self._conn.close()
            logger.info("Client disconnected: %s", self._addr)


# ── Inference server ──────────────────────────────────────────────────────────

class InferenceServer:
    def __init__(self, host: str, port: int, model: ModelLoader) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._stats: dict = {"frames": 0, "total_ms": 0.0, "start_time": time.time()}
        self._stats_lock = threading.Lock()
        self._server: socket.socket | None = None
        self._running = False

    def start(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self._host, self._port))
        self._server.listen(16)
        self._running = True

        # Start stats printer thread
        threading.Thread(target=self._print_stats, daemon=True).start()

        logger.info("Inference server listening on %s:%d", self._host, self._port)

        try:
            while self._running:
                try:
                    conn, addr = self._server.accept()
                    handler = ClientHandler(
                        conn, addr, self._model, self._stats, self._stats_lock
                    )
                    handler.start()
                except OSError:
                    break
        finally:
            self._server.close()

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()

    def _print_stats(self) -> None:
        while self._running:
            time.sleep(10)
            with self._stats_lock:
                elapsed = time.time() - self._stats["start_time"]
                frames = self._stats["frames"]
                avg_ms = self._stats["total_ms"] / frames if frames > 0 else 0
                fps = frames / elapsed if elapsed > 0 else 0
            logger.info(
                "Stats: %d frames | %.1f fps | avg inference %.1f ms | GPU %.0f%%",
                frames, fps, avg_ms, _gpu_util(),
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson inference server")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address")
    parser.add_argument("--port", type=int, default=None,
                        help="Port (overrides config)")
    parser.add_argument("--model-type", default=None,
                        help="Model type: yolov8|rtdetr|rtmpose (overrides config)")
    parser.add_argument("--model-path", default=None,
                        help="Path to model weights (overrides config)")
    args = parser.parse_args()

    # Load config
    config: dict[str, Any] = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config not found at %s, using defaults.", args.config)

    model_cfg = config.get("model", {}).copy()
    if args.model_type:
        model_cfg["type"] = args.model_type
    if args.model_path:
        model_cfg["path"] = args.model_path

    port = args.port or config.get("cluster", {}).get("inference_port", 9100)

    logger.info("Loading model: type=%s path=%s", model_cfg.get("type"), model_cfg.get("path"))
    model = ModelLoader(model_cfg)
    model.load()
    model.warmup()
    logger.info("Model ready.")

    server = InferenceServer(args.host, port, model)

    def _handle_signal(sig, frame):
        logger.info("Shutting down (signal %d)...", sig)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    server.start()


if __name__ == "__main__":
    main()
