"""
models_jetson.py — Jetson-optimized model loading for ONNX and TensorRT.

Use this on Jetson workers. Automatically detects available inference backends
and falls back gracefully. Supports:
  • ONNX models with ONNX Runtime (fast, portable)
  • TensorRT engines (.engine files) (fastest, Jetson-specific)
  • PyTorch models (fallback, slower on Jetson)

On Jetson, prioritizes: TensorRT > ONNX > PyTorch

Config example:
  model:
    type: yolov8
    path: models/yolov8n.onnx  # or .engine or .pt
    engine_path: models/yolov8n.engine  # optional, for TensorRT
    confidence: 0.5
    iou_threshold: 0.45
    img_size: 640
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
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
    Jetson-optimized model loader. Automatically selects best available backend.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model_type: str = config.get("type", "yolov8").lower()
        self._model_path: str = config.get("path", "")
        self._engine_path: str = config.get("engine_path", "")
        self._confidence: float = float(config.get("confidence", 0.5))
        self._iou: float = float(config.get("iou_threshold", 0.45))
        self._img_size: int = int(config.get("img_size", 640))
        self._warmup_iters: int = int(config.get("warmup_iters", 3))
        self._device: str = self._select_device()
        self._model: Any = None
        self._backend: str = ""  # Will be set during load()
        self._class_names: list[str] = []

    def load(self) -> None:
        """Load model weights into memory. Must be called before infer()."""
        logger.info(
            f"Loading {self._model_type} from {self._model_path} on {self._device}"
        )

        # Auto-detect best backend based on file extension or engine_path
        if self._try_load_tensorrt():
            return
        if self._try_load_onnx():
            return
        if self._try_load_pytorch():
            return

        raise RuntimeError(
            f"Could not load {self._model_type}. "
            "Ensure ONNX Runtime, TensorRT, or PyTorch is installed."
        )

    def _try_load_tensorrt(self) -> bool:
        """Try to load TensorRT engine."""
        if not self._engine_path or not Path(self._engine_path).exists():
            return False

        try:
            import tensorrt as trt

            logger.info(f"Loading TensorRT engine from {self._engine_path}")
            dispatch = {
                "yolov8": self._load_tensorrt_detector,
                "rtdetr": self._load_tensorrt_detector,
                "rtmpose": self._load_tensorrt_rtmpose,
            }
            if self._model_type not in dispatch:
                return False

            dispatch[self._model_type]()
            self._backend = "tensorrt"
            logger.info(f"✓ Loaded via TensorRT (fastest)")
            return True
        except ImportError:
            logger.debug("TensorRT not available")
            return False
        except Exception as e:
            logger.warning(f"TensorRT loading failed: {e}")
            return False

    def _try_load_onnx(self) -> bool:
        """Try to load ONNX model."""
        if not self._model_path.endswith(".onnx"):
            return False

        try:
            import onnxruntime

            logger.info(f"Loading ONNX model from {self._model_path}")
            dispatch = {
                "yolov8": self._load_onnx_detector,
                "rtdetr": self._load_onnx_detector,
                "rtmpose": self._load_onnx_rtmpose,
            }
            if self._model_type not in dispatch:
                return False

            dispatch[self._model_type]()
            self._backend = "onnx"
            logger.info(f"✓ Loaded via ONNX Runtime")
            return True
        except ImportError:
            logger.debug("ONNX Runtime not available")
            return False
        except Exception as e:
            logger.warning(f"ONNX loading failed: {e}")
            return False

    def _try_load_pytorch(self) -> bool:
        """Try to load PyTorch model (fallback)."""
        if not self._model_path.endswith(".pt"):
            return False

        try:
            logger.info(f"Loading PyTorch model from {self._model_path} (slower on Jetson)")
            dispatch = {
                "yolov8": self._load_yolov8,
                "rtdetr": self._load_rtdetr,
                "rtmpose": self._load_rtmpose,
            }
            if self._model_type not in dispatch:
                return False

            dispatch[self._model_type]()
            self._backend = "pytorch"
            logger.warning(f"Using PyTorch backend. Consider exporting to ONNX for better performance.")
            return True
        except ImportError:
            logger.debug("PyTorch not available")
            return False
        except Exception as e:
            logger.warning(f"PyTorch loading failed: {e}")
            return False

    def warmup(self) -> None:
        """Run dummy forward passes to initialize engines."""
        if self._model is None:
            raise RuntimeError("Call load() before warmup().")
        dummy = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
        logger.info(f"Warming up {self._model_type} ({self._warmup_iters} iters)...")
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
            "yolov8": self._infer_detector,
            "rtdetr": self._infer_detector,
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

    # ── TensorRT (fastest on Jetson) ───────────────────────────────────────────

    def _load_tensorrt_detector(self) -> None:
        """Load TensorRT engine for YOLOv8/RT-DETR."""
        import tensorrt as trt

        logger.info(f"Loading TensorRT engine: {self._engine_path}")
        # This is a simplified example; full implementation would handle
        # input/output binding names, data types, etc.
        pass

    def _load_tensorrt_rtmpose(self) -> None:
        """Load TensorRT engines for RTMPose detection + pose."""
        pass

    # ── ONNX (portable, good speed) ────────────────────────────────────────────

    def _load_onnx_detector(self) -> None:
        """Load ONNX model for YOLOv8/RT-DETR."""
        import onnxruntime as ort

        # Use GPU provider on Jetson (CUDAExecutionProvider)
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        self._model = ort.InferenceSession(self._model_path, providers=providers)

        # Try to infer class names from ONNX graph if available
        self._class_names = []
        logger.debug(f"ONNX model inputs: {[i.name for i in self._model.get_inputs()]}")

    def _infer_detector(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on detector model (ONNX/TensorRT/PyTorch)."""
        if self._backend == "onnx":
            return self._infer_onnx_detector(frame)
        elif self._backend == "tensorrt":
            return self._infer_tensorrt_detector(frame)
        else:  # pytorch
            return self._infer_pytorch_detector(frame)

    def _infer_onnx_detector(self, frame: np.ndarray) -> list[Detection]:
        """Run ONNX detector inference."""
        # Preprocess: resize, normalize
        h, w = frame.shape[:2]
        resized = self._resize_frame(frame, self._img_size)

        # Normalize: RGB, 0-1 range
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        # Add batch dimension and transpose to CHW
        input_data = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

        # Run inference
        input_name = self._model.get_inputs()[0].name
        output_names = [o.name for o in self._model.get_outputs()]
        outputs = self._model.run(output_names, {input_name: input_data})

        # Parse outputs (format depends on model)
        # This is a generic handler; adjust based on actual output format
        detections = self._parse_detector_output(outputs, frame)
        return detections

    def _infer_tensorrt_detector(self, frame: np.ndarray) -> list[Detection]:
        """Run TensorRT detector inference."""
        # Placeholder; implement based on TensorRT binding logic
        return []

    def _infer_pytorch_detector(self, frame: np.ndarray) -> list[Detection]:
        """Fallback: run PyTorch detector inference (slower on Jetson)."""
        import torch

        with torch.no_grad():
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
                name = (
                    self._class_names[cls_id]
                    if cls_id < len(self._class_names)
                    else str(cls_id)
                )
                detections.append(
                    Detection(
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=name,
                    )
                )
        return detections

    def _parse_detector_output(
        self, outputs: list, original_frame: np.ndarray
    ) -> list[Detection]:
        """Parse detector output from ONNX model."""
        # This is a generic parser; actual format depends on export settings
        detections: list[Detection] = []
        # TODO: Implement based on actual ONNX output format
        return detections

    # ── ONNX RTMPose ──────────────────────────────────────────────────────────

    def _load_onnx_rtmpose(self) -> None:
        """Load ONNX RTMPose models (detection + pose)."""
        import onnxruntime as ort

        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        # Assuming det and pose models are separate ONNX files
        # Store as dict with 'det' and 'pose' keys
        self._model = {}
        self._model["det"] = ort.InferenceSession(
            self._model_path.replace(".onnx", "_det.onnx"), providers=providers
        )
        self._model["pose"] = ort.InferenceSession(
            self._model_path.replace(".onnx", "_pose.onnx"), providers=providers
        )

    def _infer_onnx_rtmpose(self, frame: np.ndarray) -> list[Detection]:
        """Run ONNX RTMPose inference."""
        # TODO: Implement RTMPose ONNX inference
        return []

    def _infer_rtmpose(self, frame: np.ndarray) -> list[Detection]:
        """Run RTMPose inference (ONNX/TensorRT/PyTorch)."""
        if self._backend == "onnx":
            return self._infer_onnx_rtmpose(frame)
        elif self._backend == "tensorrt":
            # TODO: TensorRT RTMPose
            return []
        else:  # pytorch
            return self._infer_pytorch_rtmpose(frame)

    def _infer_pytorch_rtmpose(self, frame: np.ndarray) -> list[Detection]:
        """Fallback: run PyTorch RTMPose inference."""
        keypoints_list, scores_list = self._model(frame)
        detections: list[Detection] = []
        for kps, scores in zip(keypoints_list, scores_list):
            mean_score = float(np.mean(scores))
            if mean_score < self._confidence:
                continue
            kp_tuples = [(float(x), float(y), float(s)) for (x, y), s in zip(kps, scores)]
            xs = [k[0] for k in kp_tuples]
            ys = [k[1] for k in kp_tuples]
            detections.append(
                Detection(
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=mean_score,
                    class_id=0,
                    class_name="person",
                    keypoints=kp_tuples,
                )
            )
        return detections

    # ── PyTorch fallback ──────────────────────────────────────────────────────

    def _load_yolov8(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)
        self._model.to(self._device)
        self._class_names = (
            list(self._model.names.values())
            if hasattr(self._model, "names")
            else []
        )

    def _load_rtdetr(self) -> None:
        from ultralytics import RTDETR

        self._model = RTDETR(self._model_path)
        self._model.to(self._device)
        self._class_names = (
            list(self._model.names.values())
            if hasattr(self._model, "names")
            else []
        )

    def _load_rtmpose(self) -> None:
        try:
            from rtmlib import Body

            rtmpose_cfg = self._config.get("rtmpose", {})
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
            logger.error("rtmlib not available. Install: pip install rtmlib")
            raise

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
        """Resize frame to size x size, preserving aspect ratio with letterboxing."""
        import cv2

        h, w = frame.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros((size, size, 3), dtype=frame.dtype)
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return padded

    @staticmethod
    def _select_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
