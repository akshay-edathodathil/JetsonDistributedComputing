"""
ray_head.py — Distributed frame distribution and result aggregation.

Runs on the GPU PC head node. Reads video frames, distributes them to
JetsonInferenceWorker actors via Ray, collects and reorders results, then
renders/saves output and writes per-frame metrics to CSV.

Usage:
    python ray_head.py --source test_video.mp4 --workers 4 --model yolov8
    python ray_head.py --source rtsp://192.168.1.200/stream --workers 2
    python ray_head.py --source 0  # webcam
"""

from __future__ import annotations

import argparse
import collections
import csv
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ray
import yaml
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ray_worker import JetsonInferenceWorker

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ray_head")


# ── Worker pool creation ──────────────────────────────────────────────────────

def _get_jetson_nodes(head_ip: str) -> list[dict]:
    """Return Ray cluster nodes that are not the head node."""
    nodes = ray.nodes()
    jetson_nodes = [
        n for n in nodes
        if n.get("Alive") and n.get("NodeManagerAddress", "") != head_ip
    ]
    return jetson_nodes


def create_workers(
    num_workers: int,
    config: dict,
) -> list[ray.ObjectRef]:
    """
    Spawn JetsonInferenceWorker actors, one per Jetson node, using
    NodeAffinitySchedulingStrategy to pin each actor to a specific node.
    """
    head_ip = config["cluster"]["head_ip"]
    jetson_nodes = _get_jetson_nodes(head_ip)

    if not jetson_nodes:
        raise RuntimeError(
            "No Jetson worker nodes visible in Ray cluster. "
            "Run ./cluster_setup.sh start before launching ray_head.py."
        )

    available = len(jetson_nodes)
    if num_workers > available:
        logger.warning(
            "Requested %d workers but only %d Jetson nodes available. Using %d.",
            num_workers, available, available,
        )
        num_workers = available

    workers = []
    for i, node in enumerate(jetson_nodes[:num_workers]):
        node_id = node["NodeID"]
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        worker = JetsonInferenceWorker.options(
            scheduling_strategy=strategy,
            name=f"worker_{i}",
        ).remote(i, config)
        workers.append(worker)
        logger.info(
            "Spawned worker %d on node %s (%s)",
            i, node_id[:8], node.get("NodeManagerAddress", "?"),
        )

    # Verify all workers are alive before processing starts
    alive = ray.get([w.is_alive.remote() for w in workers])
    assert all(alive), "Some workers failed to initialize."
    logger.info("All %d workers ready.", num_workers)
    return workers


# ── Frame distributor ─────────────────────────────────────────────────────────

class FrameDistributor:
    """
    Reads frames from a video source and distributes them to workers.

    Tracks in-flight frame count per worker for backpressure.
    Routes new frames to the least-busy worker.
    """

    def __init__(
        self,
        workers: list,
        config: dict,
        result_queue: collections.deque,
        stop_event: threading.Event,
    ) -> None:
        self._workers = workers
        self._pipeline = config.get("pipeline", {})
        self._input_cfg = config.get("input", {})
        self._max_in_flight = int(self._pipeline.get("max_in_flight_per_worker", 4))
        self._drop_policy = self._pipeline.get("frame_drop_policy", "drop")
        self._fps_cap = float(self._input_cfg.get("fps_cap", 30))
        self._result_queue = result_queue
        self._stop_event = stop_event

        # {future: (worker_idx, frame_id)} — maps pending futures back to metadata
        self._pending: dict[ray.ObjectRef, tuple[int, int]] = {}
        # in-flight count per worker
        self._in_flight: list[int] = [0] * len(workers)
        self._dropped_frames: int = 0
        self._sent_frames: int = 0

    def run(self, source: str) -> None:
        cap = self._open_source(source)
        frame_id = 0
        frame_interval = 1.0 / self._fps_cap if self._fps_cap > 0 else 0.0
        last_frame_time = 0.0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video source exhausted at frame %d.", frame_id)
                    break

                # FPS cap
                now = time.perf_counter()
                if frame_interval > 0:
                    elapsed = now - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                last_frame_time = time.perf_counter()

                # Collect any completed futures before routing new frame
                self._collect_done()

                worker_idx = self._select_worker()
                if worker_idx is None:
                    self._dropped_frames += 1
                    if self._dropped_frames % 100 == 1:
                        logger.warning(
                            "Dropped %d frames (all workers at capacity).",
                            self._dropped_frames,
                        )
                    frame_id += 1
                    continue

                frame_ref = ray.put(frame)
                capture_ts = time.time()
                future = self._workers[worker_idx].infer.remote(frame_ref, frame_id, capture_ts)
                self._pending[future] = (worker_idx, frame_id)
                self._in_flight[worker_idx] += 1
                self._sent_frames += 1
                frame_id += 1

        finally:
            cap.release()
            logger.info(
                "Distributor done. Sent=%d Dropped=%d In-flight=%d",
                self._sent_frames, self._dropped_frames, sum(self._in_flight),
            )
            # Drain remaining in-flight frames
            while self._pending:
                self._collect_done(timeout=1.0)

    def _collect_done(self, timeout: float = 0.001) -> None:
        if not self._pending:
            return
        futures = list(self._pending.keys())
        done, _ = ray.wait(futures, num_returns=min(len(futures), 8), timeout=timeout)
        for future in done:
            worker_idx, frame_id = self._pending.pop(future)
            self._in_flight[worker_idx] = max(0, self._in_flight[worker_idx] - 1)
            try:
                result = ray.get(future)
                self._result_queue.append(result)
            except ray.exceptions.RayActorError as exc:
                logger.error(
                    "Worker %d died processing frame %d: %s. Ray will restart it.",
                    worker_idx, frame_id, exc,
                )
            except Exception as exc:
                logger.error("Frame %d inference error: %s", frame_id, exc)

    def _select_worker(self) -> int | None:
        """Return index of least-busy worker, or None if all are at capacity."""
        if self._drop_policy == "block":
            # Block until a slot opens on any worker
            while True:
                idx = min(range(len(self._workers)), key=lambda i: self._in_flight[i])
                if self._in_flight[idx] < self._max_in_flight:
                    return idx
                self._collect_done(timeout=0.01)
        else:
            idx = min(range(len(self._workers)), key=lambda i: self._in_flight[i])
            return idx if self._in_flight[idx] < self._max_in_flight else None

    @staticmethod
    def _open_source(source: str) -> cv2.VideoCapture:
        src: int | str = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        logger.info(
            "Opened source %r — %dx%d @ %.1f fps",
            source,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )
        return cap


# ── Result aggregator ─────────────────────────────────────────────────────────

class ResultAggregator:
    """
    Consumes results from result_queue, reorders them by frame_id,
    renders bounding boxes, writes CSV metrics, and optionally displays output.
    """

    def __init__(
        self,
        config: dict,
        workers: list,
        stop_event: threading.Event,
        source: str,
    ) -> None:
        self._output_cfg = config.get("output", {})
        self._pipeline_cfg = config.get("pipeline", {})
        self._reorder_buffer = int(self._pipeline_cfg.get("result_reorder_buffer", 16))
        self._display = bool(self._output_cfg.get("display", True))
        self._save_video = bool(self._output_cfg.get("save_video", False))
        self._stop_event = stop_event
        self._workers = workers

        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self._output_cfg.get("csv_path", "results/metrics_{timestamp}.csv").replace(
            "{timestamp}", ts
        )
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "frame_id", "worker_id", "inference_ms", "e2e_latency_ms",
                "num_detections", "capture_ts", "result_ts", "gpu_util_pct",
            ],
        )
        self._csv_writer.writeheader()
        logger.info("Writing metrics to %s", csv_path)

        self._video_writer: cv2.VideoWriter | None = None
        if self._save_video:
            output_path = self._output_cfg.get("output_path", "results/output.mp4").replace(
                "{timestamp}", ts
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self._video_out_path = output_path

        # Buffer for temporal reordering: {frame_id: result_dict}
        self._buffer: dict[int, dict] = {}
        self._next_frame_id: int = 0
        self._processed: int = 0
        self._t_start = time.time()

    def run(self, result_queue: collections.deque) -> None:
        last_stats_time = time.time()

        while not self._stop_event.is_set() or result_queue:
            if not result_queue:
                time.sleep(0.005)
                continue

            result = result_queue.popleft()
            self._buffer[result["frame_id"]] = result

            # Flush all contiguous frames from next_frame_id
            while self._next_frame_id in self._buffer:
                self._flush_frame(self._buffer.pop(self._next_frame_id))
                self._next_frame_id += 1

            # If buffer is over the reorder limit, force-flush the oldest
            if len(self._buffer) > self._reorder_buffer:
                oldest = min(self._buffer.keys())
                self._flush_frame(self._buffer.pop(oldest))
                self._next_frame_id = oldest + 1

            # Log throughput every 5 seconds
            now = time.time()
            if now - last_stats_time >= 5.0:
                elapsed = now - self._t_start
                fps = self._processed / elapsed if elapsed > 0 else 0
                logger.info("Throughput: %.1f fps | Processed: %d frames", fps, self._processed)
                last_stats_time = now

        self._cleanup()

    def _flush_frame(self, result: dict) -> None:
        e2e_ms = (result["result_ts"] - result["capture_ts"]) * 1000.0
        self._csv_writer.writerow({
            "frame_id": result["frame_id"],
            "worker_id": result["worker_id"],
            "inference_ms": round(result["inference_ms"], 2),
            "e2e_latency_ms": round(e2e_ms, 2),
            "num_detections": len(result["detections"]),
            "capture_ts": result["capture_ts"],
            "result_ts": result["result_ts"],
            "gpu_util_pct": result["gpu_util_pct"],
        })
        self._processed += 1

    def _cleanup(self) -> None:
        self._csv_file.flush()
        self._csv_file.close()
        if self._video_writer is not None:
            self._video_writer.release()
        if self._display:
            cv2.destroyAllWindows()
        logger.info("Aggregator done. Total frames processed: %d", self._processed)


# ── Render helpers ────────────────────────────────────────────────────────────

_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]


def render_detections(frame: np.ndarray, detections: list[dict], worker_id: int) -> np.ndarray:
    color = _COLORS[worker_id % len(_COLORS)]
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if det.get("keypoints"):
            for kx, ky, ks in det["keypoints"]:
                if ks > 0.3:
                    cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)
    cv2.putText(frame, f"W{worker_id}", (10, 20 + worker_id * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _shutdown_workers(workers: list) -> None:
    logger.info("Shutting down %d workers...", len(workers))
    try:
        ray.get([w.shutdown.remote() for w in workers], timeout=10)
    except Exception:
        pass
    for w in workers:
        try:
            ray.kill(w)
        except Exception:
            pass
    logger.info("Workers shut down.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed inference head node")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="Video file, RTSP URL, or camera index (overrides config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of Jetson workers to use")
    parser.add_argument("--model", choices=["yolov8", "rtdetr", "rtmpose"],
                        help="Model type (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if args.source:
        config.setdefault("input", {})["source"] = args.source
    if args.model:
        config.setdefault("model", {})["type"] = args.model

    source = config["input"]["source"]
    num_workers = args.workers or len(config["cluster"]["jetson_ips"])

    ray.init(address="auto", ignore_reinit_error=True)
    logger.info("Connected to Ray cluster. Resources: %s", ray.cluster_resources())

    workers = create_workers(num_workers, config)

    result_queue: collections.deque = collections.deque()
    stop_event = threading.Event()

    distributor = FrameDistributor(workers, config, result_queue, stop_event)
    aggregator = ResultAggregator(config, workers, stop_event, source)

    agg_thread = threading.Thread(target=aggregator.run, args=(result_queue,), daemon=True)
    agg_thread.start()

    try:
        distributor.run(source)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        agg_thread.join(timeout=15)
        _shutdown_workers(workers)
        ray.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()
