"""
ray_worker.py — JetsonInferenceWorker Ray actor.

Each Jetson runs one instance of this actor. The actor keeps the model loaded
in GPU memory between frames, so there is no per-frame model loading overhead.

Deploy via NodeAffinitySchedulingStrategy in ray_head.py to pin each actor to
a specific Jetson node.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any

import numpy as np
import ray

from models import Detection, ModelLoader

# ── Structured JSON logger ─────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "worker_id": getattr(record, "worker_id", None),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra_keys = {"frame_id", "inference_ms", "gpu_util_pct", "model_type"}
        for k in extra_keys:
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload)


def _setup_logger(worker_id: int, log_path: str | None = None) -> logging.Logger:
    log = logging.getLogger(f"worker.{worker_id}")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log

    handler: logging.Handler
    if log_path:
        handler = logging.FileHandler(log_path, mode="a")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    log.addHandler(handler)
    return log


# ── GPU utilization helper ────────────────────────────────────────────────────

def _gpu_utilization() -> float:
    """Return GPU utilization percent (0-100). Returns -1 on failure."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return -1.0


# ── Ray actor ─────────────────────────────────────────────────────────────────

@ray.remote(
    num_gpus=1,
    num_cpus=2,
    resources={"jetson": 1},
    max_restarts=3,
    max_task_retries=2,
)
class JetsonInferenceWorker:
    """
    Stateful inference actor that lives on one Jetson node.

    Lifecycle:
        worker = JetsonInferenceWorker.options(...).remote(worker_id, config)
        ray.get(worker.is_alive.remote())          # health check
        result = ray.get(worker.infer.remote(...)) # run inference
        ray.get(worker.shutdown.remote())           # cleanup
    """

    def __init__(self, worker_id: int, config: dict) -> None:
        self._worker_id = worker_id
        self._config = config

        log_path = config.get("output", {}).get("log_path", None)
        if log_path:
            log_path = log_path.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S"))
        self._log = _setup_logger(worker_id, log_path)
        self._log.info("Initializing worker %d", worker_id, extra={"worker_id": worker_id})

        self._loader = ModelLoader(config.get("model", {}))
        self._loader.load()
        self._loader.warmup()

        self._frame_count: int = 0
        self._total_latency_ms: float = 0.0
        self._start_time: float = time.time()

        self._log.info(
            "Worker %d ready. Model: %s",
            worker_id,
            config.get("model", {}).get("type", "unknown"),
            extra={"worker_id": worker_id},
        )

    # ── Core inference ────────────────────────────────────────────────────────

    def infer(
        self,
        frame_ref: ray.ObjectRef,
        frame_id: int,
        capture_ts: float,
    ) -> dict:
        """
        Run inference on a frame stored in the Ray object store.

        Args:
            frame_ref:  ObjectRef pointing to the frame (numpy array, BGR).
            frame_id:   Monotonically increasing frame sequence number.
            capture_ts: Unix timestamp when the frame was captured.

        Returns:
            dict with keys: frame_id, worker_id, detections, inference_ms,
                            capture_ts, result_ts, gpu_util_pct
        """
        frame: np.ndarray = ray.get(frame_ref)

        result = self._loader.infer_timed(frame)
        result.frame_id = frame_id
        ms = result.inference_ms

        self._frame_count += 1
        self._total_latency_ms += ms

        gpu_util = _gpu_utilization()

        self._log.debug(
            "frame done",
            extra={
                "worker_id": self._worker_id,
                "frame_id": frame_id,
                "inference_ms": round(ms, 2),
                "gpu_util_pct": gpu_util,
                "model_type": result.model_type,
            },
        )

        return {
            "frame_id": frame_id,
            "worker_id": self._worker_id,
            "detections": [self._detection_to_dict(d) for d in result.detections],
            "inference_ms": ms,
            "capture_ts": capture_ts,
            "result_ts": time.time(),
            "gpu_util_pct": gpu_util,
        }

    # ── Stats & health ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return performance stats for this worker."""
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        avg_lat = (self._total_latency_ms / self._frame_count
                   if self._frame_count > 0 else 0.0)
        return {
            "worker_id": self._worker_id,
            "fps": round(fps, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "total_frames": self._frame_count,
            "gpu_util_pct": _gpu_utilization(),
            "uptime_s": round(elapsed, 1),
        }

    def is_alive(self) -> bool:
        return True

    def shutdown(self) -> None:
        stats = self.get_stats()
        self._log.info(
            "Shutting down. Final stats: %s", stats,
            extra={"worker_id": self._worker_id},
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detection_to_dict(d: Detection) -> dict:
        return {
            "bbox": list(d.bbox),
            "confidence": d.confidence,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "keypoints": d.keypoints,
        }
