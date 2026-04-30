"""
jetson_pipeline.py — Per-Jetson offline batch inference pipeline.

Runs autonomously on each Jetson. No Ray, no network coordination.
Pipeline per frame:
  1. RT-DETR (.pt via ultralytics) → monkey bboxes
  2. RTMPoseOnnx → 17 COCO keypoints per bbox
  3. IDModelOnnx → monkey identity (Elm / Jok) per bbox
Output: flat CSV, one row per detected monkey per frame.

Usage:
    python3 jetson_pipeline.py \\
        --video ~/video.mp4 \\
        --rtdetr ~/JetsonDistributedComp/converted_models/RT-DETR_640.pt \\
        --rtmpose ~/JetsonDistributedComp/converted_models/rtmpose_homecage_best_size256.onnx \\
        --id-model ~/JetsonDistributedComp/converted_models/ID_model_ElmJok_Apr2026.onnx \\
        --output ~/results/cam117_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import time
from pathlib import Path

import cv2

from onnx_runner import COCO_KEYPOINT_NAMES, IDModelOnnx, RTDETROnnx, RTMPoseOnnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("jetson_pipeline")

# ── CSV header ─────────────────────────────────────────────────────────────────

_KP_COLS: list[str] = []
for _n in COCO_KEYPOINT_NAMES:
    _KP_COLS += [f"{_n}_x", f"{_n}_y", f"{_n}_confidence"]

CSV_HEADER: list[str] = [
    "frame_number",
    "monkey_ID",
    "monkey_name",
    "monkey_id_confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_confidence",
] + _KP_COLS


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    monkey_names: list[str] = [n.strip() for n in args.monkey_names.split(",")]

    # Load RT-DETR.
    # If the user passes a .pt, keep using ultralytics RTDETR even if a sibling
    # .engine exists. If the user passes a .engine, use TRT directly.
    rtdetr_path = Path(args.rtdetr)
    _use_rtdetr_trt = rtdetr_path.suffix.lower() == ".engine"
    if _use_rtdetr_trt:
        logger.info("RT-DETR → TRT engine: %s", args.rtdetr)
        rtdetr_trt = RTDETROnnx(args.rtdetr)
        detector = None
    else:
        from ultralytics import RTDETR
        logger.info("RT-DETR → ultralytics .pt: %s", args.rtdetr)
        detector = RTDETR(args.rtdetr)
        rtdetr_trt = None

    input_size = tuple(int(x) for x in args.rtmpose_input_size.split(","))  # W, H
    logger.info("Loading RTMPose from %s (input %dx%d)", args.rtmpose, *input_size)
    pose_model = RTMPoseOnnx(args.rtmpose, input_size=input_size)

    logger.info("Loading ID model from %s", args.id_model)
    id_model = IDModelOnnx(args.id_model)

    # Warmup — first CUDA call initialises kernels (~6 s on ultralytics path)
    import numpy as _np
    logger.info("Warming up RT-DETR...")
    _dummy = _np.zeros((640, 640, 3), dtype=_np.uint8)
    if _use_rtdetr_trt:
        rtdetr_trt.infer(_dummy, conf=0.99)   # TRT warmup (fast)
    else:
        detector.predict(_dummy, verbose=False)  # ultralytics CUDA init (~6s)
    logger.info("Warmup complete.")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info("Video: %d frames @ %.2f fps → %s", total_frames, fps, args.output)

    # Prepare output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(out_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(CSV_HEADER)

    # Graceful interrupt
    _stop = [False]

    def _sig(sig, _frame):
        logger.info("Signal %d received — flushing and stopping.", sig)
        _stop[0] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Stats
    frame_num = 0
    processed = 0
    total_det_ms = 0.0
    total_pose_ms = 0.0
    total_id_ms = 0.0
    total_detections = 0
    t_wall = time.time()

    try:
        while True:
            if _stop[0]:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            # Optionally skip frames to trade recall for speed
            if args.skip_frames > 1 and (frame_num % args.skip_frames) != 1:
                continue

            # ── Detection ──────────────────────────────────────────────────────
            t0 = time.perf_counter()
            bboxes: list[tuple] = []
            det_confs: list[float] = []
            if _use_rtdetr_trt:
                for bbox, c in rtdetr_trt.infer(frame, conf=args.conf):
                    bboxes.append(bbox)
                    det_confs.append(c)
            else:
                results = detector.predict(frame, conf=args.conf, verbose=False)
                if results and results[0].boxes is not None:
                    for box, c in zip(
                        results[0].boxes.xyxy.cpu().numpy(),
                        results[0].boxes.conf.cpu().numpy(),
                    ):
                        bboxes.append(tuple(float(v) for v in box))
                        det_confs.append(float(c))
            total_det_ms += (time.perf_counter() - t0) * 1000.0

            # ── Per-detection: pose + ID ───────────────────────────────────────
            for bbox, det_conf in zip(bboxes, det_confs):
                t0 = time.perf_counter()
                keypoints = pose_model.infer(frame, bbox)
                pose_t = (time.perf_counter() - t0) * 1000.0
                total_pose_ms += pose_t

                t0 = time.perf_counter()
                monkey_id, id_conf = id_model.infer(frame, bbox)
                id_t = (time.perf_counter() - t0) * 1000.0
                total_id_ms += id_t
                
                # Log first few detections for debugging
                if total_detections < 5:
                    logger.info(f"Detection {total_detections}: pose={pose_t:.2f}ms, id={id_t:.2f}ms")

                monkey_name = (
                    monkey_names[monkey_id] if monkey_id < len(monkey_names)
                    else str(monkey_id)
                )
                x1, y1, x2, y2 = bbox

                row: list = [
                    frame_num,
                    monkey_id,
                    monkey_name,
                    f"{id_conf:.4f}",
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{det_conf:.4f}",
                ]
                for kx, ky, ks in keypoints:
                    row += [f"{kx:.2f}", f"{ky:.2f}", f"{ks:.4f}"]
                writer.writerow(row)
                total_detections += 1

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t_wall
                pct = 100.0 * frame_num / total_frames if total_frames > 0 else 0.0
                nd = total_detections or 1
                logger.info(
                    "Frame %d/%d (%.0f%%) | %.1f fps | "
                    "det %.1fms | pose %.1fms | id %.1fms | dets/frame %.1f",
                    frame_num, total_frames, pct,
                    processed / elapsed,
                    total_det_ms / processed,
                    total_pose_ms / nd,
                    total_id_ms / nd,
                    total_detections / processed,
                )

    finally:
        cap.release()
        csv_fh.flush()
        csv_fh.close()

    elapsed = time.time() - t_wall
    nd = total_detections or 1

    # ── Write summary file ─────────────────────────────────────────────────────
    summary_path = out_path.with_suffix(".summary.txt")
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), elapsed % 60
    summary_lines = [
        f"video:              {args.video}",
        f"output_csv:         {out_path}",
        f"frames_processed:   {processed}",
        f"total_frames:       {total_frames}",
        f"total_detections:   {total_detections}",
        f"avg_detections_per_frame: {total_detections / processed:.2f}" if processed else "avg_detections_per_frame: 0",
        f"",
        f"total_time:         {h:02d}h {m:02d}m {s:05.2f}s  ({elapsed:.1f}s)",
        f"processing_fps:     {processed / elapsed:.2f}" if elapsed else "processing_fps: 0",
        f"",
        f"avg_det_ms:         {total_det_ms / processed:.1f}" if processed else "avg_det_ms: 0",
        f"avg_pose_ms:        {total_pose_ms / nd:.1f}",
        f"avg_id_ms:          {total_id_ms / nd:.1f}",
        f"avg_total_ms_per_frame: {(total_det_ms + total_pose_ms / max(total_detections/processed, 1) + total_id_ms / max(total_detections/processed, 1)) / processed:.1f}" if processed else "avg_total_ms_per_frame: 0",
        f"",
        f"interrupted:        {'yes' if _stop[0] else 'no'}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")

    logger.info(
        "Finished: %d frames | %d detections | %02dh%02dm%05.2fs | %.2f fps",
        processed, total_detections, h, m, s, processed / elapsed if elapsed else 0,
    )
    logger.info("Summary written to %s", summary_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Jetson per-video monkey detection + pose + ID pipeline"
    )
    p.add_argument("--video", required=True,
                   help="Path to input video file")
    p.add_argument("--rtdetr", required=True,
                   help="Path to RT-DETR_640.pt (ultralytics)")
    p.add_argument("--rtmpose", required=True,
                   help="Path to RTMPose .onnx model")
    p.add_argument("--id-model", required=True, dest="id_model",
                   help="Path to ID model .onnx")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: <video_stem>_results.csv next to video)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="RT-DETR detection confidence threshold (default: 0.5)")
    p.add_argument("--skip-frames", type=int, default=1, dest="skip_frames",
                   help="Process every Nth frame (1=all, 2=every other, …)")
    p.add_argument("--rtmpose-input-size", default="256,256", dest="rtmpose_input_size",
                   help="RTMPose model input width,height (default: 256,256)")
    p.add_argument("--monkey-names", default="Elm,Jok", dest="monkey_names",
                   help="Comma-separated names ordered by class ID (default: Elm,Jok)")
    args = p.parse_args()

    if args.output is None:
        stem = Path(args.video).stem
        args.output = str(Path(args.video).parent / f"{stem}_results.csv")

    run_pipeline(args)


if __name__ == "__main__":
    main()
