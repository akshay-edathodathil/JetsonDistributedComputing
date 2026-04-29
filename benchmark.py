"""
benchmark.py — Throughput and latency benchmarking for the distributed inference system.

Tests 1, 2, or 4 Jetsons and produces a summary table + per-run CSV.
Supports fault injection: kills one worker mid-run to verify fault tolerance.

Usage:
    python benchmark.py --video test_video.mp4 --workers 1,2,4 --frames 300
    python benchmark.py --video test_video.mp4 --workers 4 --frames 300 --fault-inject
    python benchmark.py --video test_video.mp4 --workers 2 --frames 100 --model rtdetr
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ray
import yaml
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ray_worker import JetsonInferenceWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("benchmark")


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    frame_id: int
    worker_id: int
    inference_ms: float
    e2e_latency_ms: float
    num_detections: int
    gpu_util_pct: float


@dataclass
class RunResult:
    num_workers: int
    total_frames: int
    dropped_frames: int
    elapsed_s: float
    records: list[FrameRecord] = field(default_factory=list)

    @property
    def throughput_fps(self) -> float:
        return self.total_frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def latency_percentile(self, pct: float) -> float:
        if not self.records:
            return 0.0
        vals = sorted(r.e2e_latency_ms for r in self.records)
        idx = min(int(len(vals) * pct / 100), len(vals) - 1)
        return vals[idx]

    def inference_percentile(self, pct: float) -> float:
        if not self.records:
            return 0.0
        vals = sorted(r.inference_ms for r in self.records)
        idx = min(int(len(vals) * pct / 100), len(vals) - 1)
        return vals[idx]


# ── Worker helpers ────────────────────────────────────────────────────────────

def _get_jetson_nodes(head_ip: str) -> list[dict]:
    return [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("NodeManagerAddress", "") != head_ip
    ]


def _spawn_workers(num_workers: int, config: dict) -> list:
    head_ip = config["cluster"]["head_ip"]
    jetson_nodes = _get_jetson_nodes(head_ip)
    if not jetson_nodes:
        raise RuntimeError("No Jetson nodes found. Is the cluster running?")

    workers = []
    for i, node in enumerate(jetson_nodes[:num_workers]):
        strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        w = JetsonInferenceWorker.options(
            scheduling_strategy=strategy,
            name=f"bench_worker_{i}",
        ).remote(i, config)
        workers.append(w)

    ray.get([w.is_alive.remote() for w in workers])
    logger.info("Spawned %d workers.", num_workers)
    return workers


def _kill_workers(workers: list) -> None:
    try:
        ray.get([w.shutdown.remote() for w in workers], timeout=5)
    except Exception:
        pass
    for w in workers:
        try:
            ray.kill(w)
        except Exception:
            pass


# ── Single benchmark run ──────────────────────────────────────────────────────

def run_benchmark(
    video_path: str,
    num_workers: int,
    num_frames: int,
    config: dict,
    fault_inject: bool = False,
    fault_at_frame: int = 100,
) -> RunResult:
    logger.info(
        "Starting benchmark: %d workers, %d frames, fault_inject=%s",
        num_workers, num_frames, fault_inject,
    )

    workers = _spawn_workers(num_workers, config)
    max_in_flight = int(config.get("pipeline", {}).get("max_in_flight_per_worker", 4))
    fps_cap = float(config.get("input", {}).get("fps_cap", 30))
    frame_interval = 1.0 / fps_cap if fps_cap > 0 else 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path!r}")

    in_flight: list[int] = [0] * num_workers
    pending: dict[ray.ObjectRef, tuple[int, int, float]] = {}  # future → (wid, fid, cap_ts)
    records: list[FrameRecord] = []
    dropped = 0
    frame_id = 0
    fault_injected = False
    last_frame_time = 0.0
    t_start = time.perf_counter()

    def collect(timeout: float = 0.001) -> None:
        if not pending:
            return
        done, _ = ray.wait(list(pending.keys()), num_returns=min(len(pending), 8), timeout=timeout)
        for fut in done:
            wid, fid, cap_ts = pending.pop(fut)
            in_flight[wid] = max(0, in_flight[wid] - 1)
            try:
                result = ray.get(fut)
                records.append(FrameRecord(
                    frame_id=fid,
                    worker_id=result["worker_id"],
                    inference_ms=result["inference_ms"],
                    e2e_latency_ms=(result["result_ts"] - cap_ts) * 1000.0,
                    num_detections=len(result["detections"]),
                    gpu_util_pct=result["gpu_util_pct"],
                ))
            except ray.exceptions.RayActorError as exc:
                logger.warning("Worker %d failed on frame %d: %s", wid, fid, exc)
            except Exception as exc:
                logger.warning("Frame %d error: %s", fid, exc)

    try:
        while frame_id < num_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                ret, frame = cap.read()
                if not ret:
                    break

            # FPS cap
            now = time.perf_counter()
            if frame_interval > 0:
                wait = frame_interval - (now - last_frame_time)
                if wait > 0:
                    time.sleep(wait)
            last_frame_time = time.perf_counter()

            collect()

            # Fault injection: kill worker 0 at the specified frame
            if fault_inject and not fault_injected and frame_id >= fault_at_frame:
                logger.warning("FAULT INJECTION: killing worker 0 at frame %d", frame_id)
                ray.kill(workers[0])
                fault_injected = True
                in_flight[0] = 0
                # Remove pending futures for worker 0
                for fut in list(pending.keys()):
                    if pending[fut][0] == 0:
                        del pending[fut]

            # Select least-busy worker (skip dead worker 0 after fault injection)
            available = list(range(num_workers))
            if fault_inject and fault_injected:
                available = [i for i in available if i != 0]
            if not available:
                break

            wid = min(available, key=lambda i: in_flight[i])
            if in_flight[wid] >= max_in_flight:
                dropped += 1
                frame_id += 1
                continue

            frame_ref = ray.put(frame)
            cap_ts = time.time()
            fut = workers[wid].infer.remote(frame_ref, frame_id, cap_ts)
            pending[fut] = (wid, frame_id, cap_ts)
            in_flight[wid] += 1
            frame_id += 1

        # Drain remaining futures
        while pending:
            collect(timeout=2.0)

    finally:
        cap.release()
        _kill_workers(workers)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Run complete: %d frames in %.2fs (%.1f fps), dropped %d",
        len(records), elapsed, len(records) / elapsed if elapsed > 0 else 0, dropped,
    )
    return RunResult(
        num_workers=num_workers,
        total_frames=len(records),
        dropped_frames=dropped,
        elapsed_s=elapsed,
        records=records,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results: list[RunResult]) -> None:
    print("\n" + "═" * 75)
    print(f"{'Workers':>8} {'Frames':>8} {'Dropped':>8} {'FPS':>8} "
          f"{'E2E p50':>10} {'E2E p90':>10} {'E2E p99':>10}")
    print("─" * 75)
    for r in results:
        print(
            f"{r.num_workers:>8} {r.total_frames:>8} {r.dropped_frames:>8} "
            f"{r.throughput_fps:>8.1f} "
            f"{r.latency_percentile(50):>10.1f} "
            f"{r.latency_percentile(90):>10.1f} "
            f"{r.latency_percentile(99):>10.1f}"
        )
    print("═" * 75)
    print("  E2E latency values in milliseconds (capture → result)\n")


def print_latency_histogram(run: RunResult, bins: int = 10) -> None:
    if not run.records:
        return
    vals = sorted(r.e2e_latency_ms for r in run.records)
    lo, hi = vals[0], vals[-1]
    width = (hi - lo) / bins if hi > lo else 1.0
    print(f"\nLatency histogram ({run.num_workers}-worker run):")
    for b in range(bins):
        low = lo + b * width
        high = lo + (b + 1) * width
        count = sum(1 for v in vals if low <= v < high)
        bar = "█" * max(1, count * 40 // len(vals)) if count else ""
        print(f"  {low:6.1f}–{high:6.1f} ms | {bar} {count}")


def save_results(results: list[RunResult], results_dir: str) -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"benchmark_{ts}_summary.csv")
    detail_path = os.path.join(results_dir, f"benchmark_{ts}_detail.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_workers", "total_frames", "dropped_frames", "elapsed_s",
            "throughput_fps", "e2e_p50_ms", "e2e_p90_ms", "e2e_p99_ms",
            "infer_p50_ms", "infer_p90_ms", "infer_p99_ms",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "num_workers": r.num_workers,
                "total_frames": r.total_frames,
                "dropped_frames": r.dropped_frames,
                "elapsed_s": round(r.elapsed_s, 3),
                "throughput_fps": round(r.throughput_fps, 2),
                "e2e_p50_ms": round(r.latency_percentile(50), 2),
                "e2e_p90_ms": round(r.latency_percentile(90), 2),
                "e2e_p99_ms": round(r.latency_percentile(99), 2),
                "infer_p50_ms": round(r.inference_percentile(50), 2),
                "infer_p90_ms": round(r.inference_percentile(90), 2),
                "infer_p99_ms": round(r.inference_percentile(99), 2),
            })
    logger.info("Summary saved to %s", summary_path)

    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_workers_config", "frame_id", "worker_id", "inference_ms",
            "e2e_latency_ms", "num_detections", "gpu_util_pct",
        ])
        writer.writeheader()
        for r in results:
            for rec in r.records:
                writer.writerow({
                    "num_workers_config": r.num_workers,
                    "frame_id": rec.frame_id,
                    "worker_id": rec.worker_id,
                    "inference_ms": round(rec.inference_ms, 2),
                    "e2e_latency_ms": round(rec.e2e_latency_ms, 2),
                    "num_detections": rec.num_detections,
                    "gpu_util_pct": rec.gpu_util_pct,
                })
    logger.info("Detail saved to %s", detail_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed inference benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", help="Video file path (overrides config)")
    parser.add_argument(
        "--workers",
        default=None,
        help="Comma-separated worker counts to test, e.g. '1,2,4'",
    )
    parser.add_argument("--frames", type=int, default=None, help="Frames per run (overrides config)")
    parser.add_argument("--model", choices=["yolov8", "rtdetr", "rtmpose"],
                        help="Model type (overrides config)")
    parser.add_argument(
        "--fault-inject", action="store_true",
        help="Kill worker 0 mid-run to test fault tolerance",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    bench_cfg = config.get("benchmark", {})

    video = args.video or config.get("input", {}).get("source", "test_video.mp4")
    num_frames = args.frames or bench_cfg.get("num_frames", 300)
    fault_at = bench_cfg.get("fault_inject_at_frame", 100)
    results_dir = bench_cfg.get("results_dir", "results")

    if args.workers:
        worker_counts = [int(x.strip()) for x in args.workers.split(",")]
    else:
        worker_counts = bench_cfg.get("worker_counts", [1, 2, 4])

    if args.model:
        config.setdefault("model", {})["type"] = args.model

    if args.fault_inject and len(worker_counts) != 1:
        logger.warning("--fault-inject runs a single config; using first worker count: %d", worker_counts[0])
        worker_counts = [worker_counts[0]]

    ray.init(address="auto", ignore_reinit_error=True)
    logger.info("Ray cluster resources: %s", ray.cluster_resources())

    all_results: list[RunResult] = []

    for nw in worker_counts:
        try:
            result = run_benchmark(
                video_path=video,
                num_workers=nw,
                num_frames=num_frames,
                config=config,
                fault_inject=args.fault_inject,
                fault_at_frame=fault_at,
            )
            all_results.append(result)
            if args.fault_inject:
                logger.info(
                    "Fault injection result: %d frames processed despite worker failure.",
                    result.total_frames,
                )
        except Exception as exc:
            logger.error("Run with %d workers failed: %s", nw, exc)

    if all_results:
        print_summary(all_results)
        print_latency_histogram(all_results[-1])
        save_results(all_results, results_dir)

    ray.shutdown()


if __name__ == "__main__":
    main()
