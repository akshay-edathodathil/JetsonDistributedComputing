"""
onnx_runner.py — Inference wrappers for RTMPose and ID model.

Backend auto-selected per model:
  • TRT  — if a .engine file exists alongside the .onnx  (GPU, fastest)
  • ORT  — onnxruntime 1.15.1 CPU                        (fallback)

TRT memory management uses torch CUDA tensors — no pycuda needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",    "right_eye",
    "left_ear",    "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow",  "right_elbow",
    "left_wrist",  "right_wrist",
    "left_hip",    "right_hip",
    "left_knee",   "right_knee",
    "left_ankle",  "right_ankle",
]
NUM_KP = len(COCO_KEYPOINT_NAMES)  # 17


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _engine_path(onnx_path: str) -> str | None:
    """Return the .engine path if it exists alongside the .onnx, else None."""
    ep = str(Path(onnx_path).with_suffix(".engine"))
    return ep if os.path.exists(ep) else None


def _letterbox(
    img: np.ndarray, target_wh: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    tw, th = target_wh
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_l = (tw - nw) // 2
    pad_t = (th - nh) // 2
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, (pad_l, pad_t)


def _to_nchw(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[np.newaxis]


def _safe_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _simcc_decode(
    x_logits: np.ndarray,  # [K, BINS]
    y_logits: np.ndarray,  # [K, BINS]
    bbox: tuple,
    scale: float,
    pad_l: int, pad_t: int,
    input_w: int, input_h: int,
    bins: int = 512,
) -> list[tuple[float, float, float]]:
    x1, y1 = int(bbox[0]), int(bbox[1])
    keypoints: list[tuple[float, float, float]] = []
    for k in range(min(x_logits.shape[0], NUM_KP)):
        xi = int(np.argmax(x_logits[k]))
        yi = int(np.argmax(y_logits[k]))
        ex = np.exp(x_logits[k] - x_logits[k].max())
        ey = np.exp(y_logits[k] - y_logits[k].max())
        score = float(ex[xi] / ex.sum()) * float(ey[yi] / ey.sum())
        xc = (xi * input_w / bins - pad_l) / scale
        yc = (yi * input_h / bins - pad_t) / scale
        keypoints.append((float(xc + x1), float(yc + y1), score))
    while len(keypoints) < NUM_KP:
        keypoints.append((0.0, 0.0, 0.0))
    return keypoints


# ── ORT loader ────────────────────────────────────────────────────────────────

def _load_ort(model_path: str):
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        logger.warning("CUDA ORT unavailable for %s — using CPU.", model_path)
    logger.info("ORT: %s | providers: %s", model_path, sess.get_providers())
    return sess


# ── TRT runner (torch CUDA buffers — no pycuda needed) ───────────────────────

class TRTRunner:
    """
    Wraps a TensorRT .engine file.
    GPU I/O via torch CUDA tensors, so pycuda is not required.
    Pass input_shape for engines with dynamic input dimensions.
    """

    def __init__(self, engine_path: str, input_shape: tuple[int, ...] | None = None) -> None:
        import tensorrt as trt
        import torch

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        if input_shape is not None:
            self._ctx.set_binding_shape(0, input_shape)

        self._bufs: list[torch.Tensor] = []
        self._ptrs: list[int] = []
        self._n_in = 0

        for i in range(self._engine.num_bindings):
            shape = tuple(self._ctx.get_binding_shape(i))
            if any(d <= 0 for d in shape):
                raise RuntimeError(
                    f"Binding {i} has unresolved shape {shape}. "
                    "Provide input_shape= to TRTRunner."
                )
            buf = torch.zeros(shape, dtype=torch.float32, device="cuda")
            self._bufs.append(buf)
            self._ptrs.append(buf.data_ptr())
            if self._engine.binding_is_input(i):
                self._n_in += 1

        self._n_out = len(self._bufs) - self._n_in
        logger.info("TRT loaded: %s (in=%d out=%d)", engine_path, self._n_in, self._n_out)

    def run(self, *inputs: np.ndarray) -> list[np.ndarray]:
        import torch
        for i, inp in enumerate(inputs):
            self._bufs[i].copy_(torch.from_numpy(np.ascontiguousarray(inp)))
        self._ctx.execute_v2(self._ptrs)
        torch.cuda.synchronize()
        return [self._bufs[self._n_in + i].cpu().numpy() for i in range(self._n_out)]


# ── RT-DETR (TRT only — ultralytics .pt used as fallback by pipeline) ─────────

class RTDETROnnx:
    """
    RT-DETR inference via TRT engine.
    Used when RT-DETR_640.engine exists; otherwise jetson_pipeline.py falls
    back to ultralytics RTDETR(.pt) which already runs on GPU.

    Output format from the no-NMS ONNX export:
      output0: [1, 300, 5]  →  x1, y1, x2, y2, confidence  (letterboxed 640×640)
    Detections are TopK-300 by score (no true NMS). Filter by conf threshold.
    """

    def __init__(self, engine_path: str) -> None:
        self._trt = TRTRunner(engine_path)
        logger.info("RTDETROnnx → TRT: %s", engine_path)

    def infer(
        self,
        frame_bgr: np.ndarray,
        conf: float = 0.5,
    ) -> list[tuple[tuple, float]]:
        """
        Returns list of (bbox, confidence):
          bbox: (x1, y1, x2, y2) in original frame pixel coords
        """
        fh, fw = frame_bgr.shape[:2]
        padded, scale, (pad_l, pad_t) = _letterbox(frame_bgr, (640, 640))
        inp = _to_nchw(padded)

        outputs = self._trt.run(inp)
        # RT-DETR TRT engine produces 5 outputs; output[4] contains [1, 300, 5]: bbox + conf
        dets = outputs[4][0]  # Shape: [300, 5]

        results = []
        for det in dets:
            # Handle different output shapes gracefully
            if len(det) < 5:
                continue

            # Some TRT runs produce NaN/Inf confidences (binding/shape mismatch).
            # Treat non-finite confidences as low-confidence and skip them.
            c = float(det[4])
            if not np.isfinite(c):
                continue

            if c < conf:
                continue
            x1 = max(0.0, (float(det[0]) - pad_l) / scale)
            y1 = max(0.0, (float(det[1]) - pad_t) / scale)
            x2 = min(float(fw), (float(det[2]) - pad_l) / scale)
            y2 = min(float(fh), (float(det[3]) - pad_t) / scale)
            if x2 > x1 and y2 > y1:
                results.append(((x1, y1, x2, y2), c))
        return results


# ── RTMPose ────────────────────────────────────────────────────────────────────

class RTMPoseOnnx:
    """
    RTMPose SimCC inference.
    Automatically uses TRT (.engine) if it exists alongside the .onnx,
    otherwise falls back to onnxruntime CPU.
    """
    SIMCC_BINS = 512

    def __init__(self, model_path: str, input_size: tuple[int, int] = (256, 256)) -> None:
        self._input_w, self._input_h = input_size
        ep = _engine_path(model_path)
        if ep:
            logger.info("RTMPose → TRT: %s", ep)
            self._trt = TRTRunner(ep, input_shape=(1, 3, self._input_h, self._input_w))
            self._use_trt = True
        else:
            self._sess = _load_ort(model_path)
            self._inp_name = self._sess.get_inputs()[0].name
            self._out_x   = self._sess.get_outputs()[0].name
            self._out_y   = self._sess.get_outputs()[1].name
            self._use_trt = False
            logger.info("RTMPose → ORT CPU (no .engine found next to %s)", model_path)

    def infer(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> list[tuple[float, float, float]]:
        crop = _safe_crop(frame_bgr, bbox)
        if crop is None:
            return [(0.0, 0.0, 0.0)] * NUM_KP

        padded, scale, (pad_l, pad_t) = _letterbox(crop, (self._input_w, self._input_h))
        inp = _to_nchw(padded)

        if self._use_trt:
            x_logits, y_logits = self._trt.run(inp)
        else:
            x_logits, y_logits = self._sess.run(
                [self._out_x, self._out_y], {self._inp_name: inp}
            )
        return _simcc_decode(
            x_logits[0], y_logits[0], bbox,
            scale, pad_l, pad_t, self._input_w, self._input_h,
        )


# ── ID model ───────────────────────────────────────────────────────────────────

class IDModelOnnx:
    """
    Individual monkey identification.
    Automatically uses TRT (.engine) if it exists alongside the .onnx,
    otherwise falls back to onnxruntime CPU.
    """

    def __init__(self, model_path: str) -> None:
        ep = _engine_path(model_path)
        expected_engine = str(Path(model_path).with_suffix(".engine"))
        logger.debug(f"Looking for ID engine at: {expected_engine}")
        if ep:
            logger.info(f"✓ ID model → TRT ENGINE: {ep}")
            self._trt = TRTRunner(ep, input_shape=(1, 3, 640, 640))
            self._use_trt = True
        else:
            logger.warning(f"✗ ID model → ORT CPU (engine not found at {expected_engine})")
            self._sess = _load_ort(model_path)
            self._inp_name = self._sess.get_inputs()[0].name
            self._out_name = self._sess.get_outputs()[0].name
            self._use_trt = False
            logger.info("Loading ONNX from: %s", model_path)

    def infer(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[int, float]:
        import time
        t0 = time.perf_counter()
        crop = _safe_crop(frame_bgr, bbox)
        if crop is None:
            return 0, 0.0
        t1 = time.perf_counter()
        padded, _, _ = _letterbox(crop, (640, 640))
        t2 = time.perf_counter()
        inp = _to_nchw(padded)
        t3 = time.perf_counter()
        if self._use_trt:
            logits = self._trt.run(inp)[0][0]
        else:
            logits = self._sess.run([self._out_name], {self._inp_name: inp})[0][0]
        t4 = time.perf_counter()
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        monkey_id = int(np.argmax(probs))
        t5 = time.perf_counter()
        # Log breakdown on every 50th call (reduce spam)
        if not hasattr(self, '_infer_count'):
            self._infer_count = 0
        self._infer_count += 1
        if self._infer_count % 50 == 0:
            logger.debug(f"ID infer breakdown: crop={1000*(t1-t0):.2f}ms letterbox={1000*(t2-t1):.2f}ms to_nchw={1000*(t3-t2):.2f}ms infer={1000*(t4-t3):.2f}ms softmax={1000*(t5-t4):.2f}ms total={1000*(t5-t0):.2f}ms")
        return monkey_id, float(probs[monkey_id])
