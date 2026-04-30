"""
onnx_inference_utils.py — Helper utilities for ONNX model inference.

Handles:
  • Preprocessing (resize, normalize, format)
  • Output parsing for YOLOv8/RT-DETR exported via ultralytics
  • Coordinate transformation
  • NMS (non-maximum suppression)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

# YOLOv8/RT-DETR ONNX output format (from ultralytics export):
# - Input: (1, 3, 640, 640) normalized float32 [0, 1]
# - Output: (1, 84, 8400) for YOLOv8n
#   Format per detection: [x, y, w, h, conf, cls0_prob, cls1_prob, ...]


def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 640,
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess frame for ONNX model inference.

    Args:
        frame: BGR input frame (H, W, 3)
        target_size: Target size (square)
        normalize: If True, normalize to [0, 1]; if False, keep [0, 255]

    Returns:
        (input_tensor, scale_factor): Tensor ready for ONNX (1, 3, 640, 640),
                                     scale factor for coordinate conversion
    """
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)

    # Resize with aspect ratio preservation
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Letterbox padding
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    # Convert BGR to RGB and normalize
    rgb = padded[:, :, ::-1]
    if normalize:
        rgb = rgb.astype(np.float32) / 255.0

    # Transpose to CHW and add batch dimension
    input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :].astype(
        np.float32
    )

    return input_tensor, scale


def postprocess_yolov8(
    outputs: list[np.ndarray],
    frame_shape: Tuple[int, int],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    target_size: int = 640,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse YOLOv8 ONNX output to detections.

    YOLOv8 exports as: output shape (1, 84, 8400)
    - 84 = 4 (bbox) + 1 (conf) + 80 (classes) for COCO
    - 8400 = feature map points

    Args:
        outputs: List with one element, shape (1, 84, N)
        frame_shape: Original frame (H, W)
        conf_threshold: Confidence threshold
        iou_threshold: NMS IOU threshold
        target_size: Model input size (640)

    Returns:
        (boxes, confidences, class_ids): NumPy arrays
        - boxes: (N, 4) with [x1, y1, x2, y2] in original image coords
        - confidences: (N,)
        - class_ids: (N,)
    """
    output = outputs[0][0]  # (84, 8400)
    num_classes = output.shape[0] - 5  # 80 for COCO

    # Extract predictions
    predictions = output.T  # (8400, 84)

    # Filter by objectness score
    objectness = predictions[:, 4]
    mask = objectness > conf_threshold
    predictions = predictions[mask]

    if predictions.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    # Extract components
    boxes_cxcy = predictions[:, :2]  # Center x, y
    boxes_wh = predictions[:, 2:4]   # Width, height
    confidences = predictions[:, 4]
    class_probs = predictions[:, 5:]

    # Convert to x1, y1, x2, y2
    x1 = boxes_cxcy[:, 0] - boxes_wh[:, 0] / 2
    y1 = boxes_cxcy[:, 1] - boxes_wh[:, 1] / 2
    x2 = boxes_cxcy[:, 0] + boxes_wh[:, 0] / 2
    y2 = boxes_cxcy[:, 1] + boxes_wh[:, 1] / 2
    boxes = np.column_stack([x1, y1, x2, y2])

    # Get class IDs and scores
    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_ids)), class_ids]

    # Apply NMS
    keep_idx = nms(boxes, confidences * class_scores, iou_threshold)
    boxes = boxes[keep_idx]
    confidences = confidences[keep_idx]
    class_ids = class_ids[keep_idx]

    # Scale boxes to original image coordinates
    h, w = frame_shape
    scale_x = w / target_size
    scale_y = h / target_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Clip to image boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

    return boxes, confidences, class_ids


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Non-Maximum Suppression.

    Args:
        boxes: (N, 4) with [x1, y1, x2, y2]
        scores: (N,)
        iou_threshold: IOU threshold for suppression

    Returns:
        indices of kept boxes
    """
    x1, y1, x2, y2 = boxes.T

    areas = (x2 - x1) * (y2 - y1)
    sorted_idx = np.argsort(-scores)

    keep = []
    while len(sorted_idx) > 0:
        current = sorted_idx[0]
        keep.append(current)

        if len(sorted_idx) == 1:
            break

        rest = sorted_idx[1:]
        x1_inter = np.maximum(x1[current], x1[rest])
        y1_inter = np.maximum(y1[current], y1[rest])
        x2_inter = np.minimum(x2[current], x2[rest])
        y2_inter = np.minimum(y2[current], y2[rest])

        w_inter = np.maximum(0, x2_inter - x1_inter)
        h_inter = np.maximum(0, y2_inter - y1_inter)
        inter_area = w_inter * h_inter

        union_area = areas[current] + areas[rest] - inter_area
        iou = inter_area / union_area

        sorted_idx = rest[iou < iou_threshold]

    return np.array(keep)


def postprocess_rtdetr(
    outputs: list[np.ndarray],
    frame_shape: Tuple[int, int],
    conf_threshold: float = 0.5,
    target_size: int = 640,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse RT-DETR ONNX output.

    RT-DETR typically exports with different output format than YOLOv8.
    This is a placeholder; adapt based on actual export format.

    Common RT-DETR output:
    - boxes: (batch, num_queries, 4) in normalized coords
    - scores: (batch, num_queries, num_classes)
    """
    # TODO: Implement based on your RT-DETR export configuration
    # For now, use generic detection parsing
    raise NotImplementedError(
        "RT-DETR ONNX parsing not yet implemented. "
        "Adapt based on your specific export format."
    )


# Example usage:
if __name__ == "__main__":
    import onnxruntime as ort

    # Load ONNX model and test
    model_path = "models/exported/yolov8n.onnx"
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # Load test image
    test_image = cv2.imread("test_image.jpg")
    if test_image is not None:
        # Preprocess
        input_tensor, scale = preprocess_frame(test_image)

        # Infer
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names, {input_name: input_tensor})

        # Postprocess
        boxes, scores, class_ids = postprocess_yolov8(
            outputs,
            test_image.shape[:2],
            conf_threshold=0.5,
        )

        print(f"Detected {len(boxes)} objects")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            print(
                f"  {i+1}. Class {class_id}: confidence {score:.2f}, "
                f"bbox {box.astype(int)}"
            )
