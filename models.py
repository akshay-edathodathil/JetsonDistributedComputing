"""
models.py — Unified model loading and inference abstraction.

Supports YOLOv8, RT-DETR (via ultralytics), and RTMPose (via mmpose/rtmlib).
All models accept BGR numpy arrays and return a list of Detection objects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Normalized detection result from any model type."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    confidence: float
    class_id: int
    class_name: str
    keypoints: list[tuple[float, float, float]] | None = None  # (x, y, score) for pose


@dataclass
class InferenceResult:
    """Full result for one frame."""
    frame_id: int
    detections: list[Detection]
    inference_ms: float
    model_type: str
    image_shape: tuple[int, int]  # H, W


class ModelLoader:
    """
    Loads and runs one of: yolov8, rtdetr, rtmpose.
    Call load() before infer(). Call warmup() after load() to initialize CUDA kernels.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model_type: str = config.get("type", "yolov8").lower()
        self._model_path: str = config.get("path", "")
        self._confidence: float = float(config.get("confidence", 0.5))
        self._iou: float = float(config.get("iou_threshold", 0.45))
        self._img_size: int = int(config.get("img_size", 640))
        self._warmup_iters: int = int(config.get("warmup_iters", 3))
        self._device: str = self._select_device()
        self._model: Any = None
        self._class_names: list[str] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights into memory. Must be called before infer()."""
        logger.info("Loading %s from %s on %s", self._model_type, self._model_path, self._device)
        dispatch = {
            "yolov8": self._load_yolov8,
            "rtdetr": self._load_rtdetr,
            "rtmpose": self._load_rtmpose,
        }
        if self._model_type not in dispatch:
            raise ValueError(f"Unknown model type: {self._model_type!r}. Choose from {list(dispatch)}")
        dispatch[self._model_type]()
        logger.info("%s loaded successfully.", self._model_type)

    def warmup(self) -> None:
        """Run dummy forward passes to initialize CUDA kernels."""
        if self._model is None:
            raise RuntimeError("Call load() before warmup().")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.info("Warming up %s (%d iters)...", self._model_type, self._warmup_iters)
        for _ in range(self._warmup_iters):
            self.infer(dummy)
        logger.info("Warmup complete.")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.
        Returns a list of Detection objects (empty list if nothing detected).
        """
        if self._model is None:
            raise RuntimeError("Call load() before infer().")
        dispatch = {
            "yolov8": self._infer_yolov8,
            "rtdetr": self._infer_rtdetr,
            "rtmpose": self._infer_rtmpose,
        }
        return dispatch[self._model_type](frame)

    def infer_timed(self, frame: np.ndarray) -> InferenceResult:
        """Run infer() and return an InferenceResult with timing info."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        detections = self.infer(frame)
        ms = (time.perf_counter() - t0) * 1000.0
        return InferenceResult(
            frame_id=-1,  # caller fills this in
            detections=detections,
            inference_ms=ms,
            model_type=self._model_type,
            image_shape=(h, w),
        )

    # ── YOLOv8 ────────────────────────────────────────────────────────────────

    def _load_yolov8(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._model_path)
        self._model.to(self._device)
        self._class_names = list(self._model.names.values()) if hasattr(self._model, "names") else []

    def _infer_yolov8(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._img_size,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
                detections.append(Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=name,
                ))
        return detections

    # ── RT-DETR ───────────────────────────────────────────────────────────────

    def _load_rtdetr(self) -> None:
        # RT-DETR is supported natively in ultralytics >= 8.0
        from ultralytics import RTDETR
        self._model = RTDETR(self._model_path)
        self._model.to(self._device)
        self._class_names = list(self._model.names.values()) if hasattr(self._model, "names") else []

    def _infer_rtdetr(self, frame: np.ndarray) -> list[Detection]:
        # RT-DETR uses the same predict API as YOLO in ultralytics
        results = self._model.predict(
            frame,
            conf=self._confidence,
            imgsz=self._img_size,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
                detections.append(Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=name,
                ))
        return detections

    # ── RTMPose ───────────────────────────────────────────────────────────────

    def _load_rtmpose(self) -> None:
        rtmpose_cfg = self._config.get("rtmpose", {})
        try:
            # Prefer rtmlib (lightweight wrapper around RTMPose/RTMDet)
            from rtmlib import Body
            self._model = Body(
                det=rtmpose_cfg.get("det_model_path", ""),
                pose=rtmpose_cfg.get("pose_model_path", ""),
                to_openpose=False,
                backend="onnxruntime",
                device=self._device,
            )
            self._rtmpose_backend = "rtmlib"
            logger.info("RTMPose loaded via rtmlib.")
        except ImportError:
            # Fall back to mmpose inference pipeline
            self._load_rtmpose_mmpose(rtmpose_cfg)

    def _load_rtmpose_mmpose(self, rtmpose_cfg: dict) -> None:
        from mmpose.apis import init_model
        from mmdet.apis import init_detector

        det_cfg = rtmpose_cfg.get("det_config", "")
        det_ckpt = rtmpose_cfg.get("det_model_path", "")
        pose_cfg = rtmpose_cfg.get("pose_config", "")
        pose_ckpt = rtmpose_cfg.get("pose_model_path", "")

        self._det_model = init_detector(det_cfg, det_ckpt, device=self._device)
        self._model = init_model(pose_cfg, pose_ckpt, device=self._device)
        self._rtmpose_backend = "mmpose"
        logger.info("RTMPose loaded via mmpose.")

    def _infer_rtmpose(self, frame: np.ndarray) -> list[Detection]:
        if self._rtmpose_backend == "rtmlib":
            return self._infer_rtmpose_rtmlib(frame)
        return self._infer_rtmpose_mmpose(frame)

    def _infer_rtmpose_rtmlib(self, frame: np.ndarray) -> list[Detection]:
        keypoints_list, scores_list = self._model(frame)
        detections: list[Detection] = []
        for kps, scores in zip(keypoints_list, scores_list):
            mean_score = float(np.mean(scores))
            if mean_score < self._confidence:
                continue
            kp_tuples = [(float(x), float(y), float(s)) for (x, y), s in zip(kps, scores)]
            xs = [k[0] for k in kp_tuples]
            ys = [k[1] for k in kp_tuples]
            detections.append(Detection(
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                confidence=mean_score,
                class_id=0,
                class_name="person",
                keypoints=kp_tuples,
            ))
        return detections

    def _infer_rtmpose_mmpose(self, frame: np.ndarray) -> list[Detection]:
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown

        det_result = inference_detector(self._det_model, frame)
        bboxes = det_result.pred_instances.bboxes.cpu().numpy()
        det_scores = det_result.pred_instances.scores.cpu().numpy()

        detections: list[Detection] = []
        for bbox, score in zip(bboxes, det_scores):
            if score < self._confidence:
                continue
            pose_results = inference_topdown(self._model, frame, [{"bbox": bbox}])
            if not pose_results:
                continue
            kps = pose_results[0].pred_instances.keypoints[0].cpu().numpy()
            kp_scores = pose_results[0].pred_instances.keypoint_scores[0].cpu().numpy()
            kp_tuples = [(float(x), float(y), float(s)) for (x, y), s in zip(kps, kp_scores)]
            detections.append(Detection(
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                confidence=float(score),
                class_id=0,
                class_name="person",
                keypoints=kp_tuples,
            ))
        return detections

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> str:
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
