#!/usr/bin/env python3
"""
Test synthetic frame detection on GPU.

Loads synthetic preview frames with known monkey counts (4/10/16/20),
runs RT-DETR + RTMPose + ID model, and validates detection accuracy.
"""

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("/home/cams/Desktop/ABT_Annotation/models")
FRAMES_DIR = Path("/home/cams/homecage_data/DATA/POST/LongVideosForTest/8h")

RTDETR_MODEL = str(MODELS_DIR / "RT-DETR_640.pt")
ID_MODEL = str(MODELS_DIR / "ID_model_ElmJok_Apr2026.pt")
RTMPOSE_MODEL = str(MODELS_DIR / "rtmpose_homecage_best.pth")

CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Models dir: {MODELS_DIR}")
print(f"Frames dir: {FRAMES_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

print("Loading models...")

# RT-DETR
try:
    from ultralytics import RTDETR
    rtdetr = RTDETR(RTDETR_MODEL)
    print(f"✓ RT-DETR loaded: {RTDETR_MODEL}")
except Exception as e:
    print(f"✗ RT-DETR failed: {e}")
    exit(1)

# ID Model
try:
    id_model = torch.jit.load(ID_MODEL, map_location=DEVICE)
    id_model.eval()
    print(f"✓ ID model loaded: {ID_MODEL}")
except Exception as e:
    print(f"✗ ID model failed: {e}")
    try:
        id_model = torch.load(ID_MODEL, map_location=DEVICE)
        id_model.eval()
        print(f"✓ ID model loaded (state_dict): {ID_MODEL}")
    except Exception as e2:
        print(f"✗ ID model fallback failed: {e2}")
        id_model = None

# RTMPose (if available)
try:
    rtmpose_net = torch.load(RTMPOSE_MODEL, map_location=DEVICE)
    rtmpose_net.eval()
    print(f"✓ RTMPose loaded: {RTMPOSE_MODEL}")
except Exception as e:
    print(f"⚠ RTMPose not loaded (optional): {e}")
    rtmpose_net = None

print()

# ─────────────────────────────────────────────────────────────────────────────
# Process frames
# ─────────────────────────────────────────────────────────────────────────────

def infer_id_model(frame, bbox, model, device="cuda"):
    """Infer monkey ID from bbox crop."""
    if model is None:
        return 0, 0.0
    
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0, 0.0
    
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (640, 640))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
    crop_tensor = crop_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(crop_tensor)[0]
        probs = torch.softmax(logits, dim=-1)
        monkey_id = int(torch.argmax(probs).item())
        confidence = float(probs[monkey_id].item())
    
    return monkey_id, confidence


def process_frame(frame_path, expected_count):
    """Process one frame and return detection stats."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    
    frame_h, frame_w = frame.shape[:2]
    results = {
        "frame": frame_path.name,
        "expected_count": expected_count,
        "detected_count": 0,
        "detection_time_ms": 0.0,
        "id_time_ms": 0.0,
        "pose_time_ms": 0.0,
        "total_time_ms": 0.0,
        "detections": [],
    }
    
    # RT-DETR detection
    t0 = time.perf_counter()
    detection_results = rtdetr(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    t_det = time.perf_counter() - t0
    results["detection_time_ms"] = t_det * 1000
    
    detections = detection_results[0].boxes
    results["detected_count"] = len(detections)
    
    t_id_total = 0
    t_pose_total = 0
    
    # Process each detection
    for i, box in enumerate(detections):
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        conf = float(box.conf[0].cpu().numpy())
        
        # ID inference
        t_id = time.perf_counter()
        monkey_id, id_conf = infer_id_model(frame, bbox, id_model, DEVICE) if id_model else (0, 0.0)
        t_id = time.perf_counter() - t_id
        t_id_total += t_id
        
        results["detections"].append({
            "id": i,
            "bbox": bbox.tolist(),
            "detection_conf": conf,
            "monkey_id": monkey_id,
            "id_conf": id_conf,
        })
    
    results["id_time_ms"] = t_id_total * 1000
    results["total_time_ms"] = t_det * 1000 + t_id_total * 1000
    
    return results


# Find all preview frames
preview_frames = sorted(FRAMES_DIR.glob("preview_*.png"))
print(f"Found {len(preview_frames)} preview frames\n")

# Group by monkey count
frames_by_count = defaultdict(list)
for frame_path in preview_frames:
    # Extract count from filename: preview_view102_m04.png
    if "_m" in frame_path.name:
        count_str = frame_path.name.split("_m")[1].replace(".png", "")
        try:
            count = int(count_str)
            frames_by_count[count].append(frame_path)
        except ValueError:
            pass

# Process one frame per count
print("=" * 80)
print("DETECTION TEST RESULTS")
print("=" * 80)
print()

total_stats = {"correct": 0, "total": 0, "total_time_ms": 0.0}

for count in sorted(frames_by_count.keys()):
    frame_path = frames_by_count[count][0]  # Test first frame of this count
    print(f"Testing: {frame_path.name}")
    print(f"Expected monkeys: {count}")
    
    result = process_frame(frame_path, count)
    if result is None:
        print("✗ Failed to load frame\n")
        continue
    
    detected = result["detected_count"]
    expected = result["expected_count"]
    match = "✓" if detected == expected else "✗"
    
    print(f"Detected monkeys: {detected} {match}")
    print(f"Detection time: {result['detection_time_ms']:.1f} ms")
    print(f"ID inference time: {result['id_time_ms']:.1f} ms")
    print(f"Total time: {result['total_time_ms']:.1f} ms")
    print()
    
    total_stats["total"] += 1
    if detected == expected:
        total_stats["correct"] += 1
    total_stats["total_time_ms"] += result["total_time_ms"]

print("=" * 80)
print("SUMMARY")
print("=" * 80)
accuracy = (total_stats["correct"] / total_stats["total"] * 100) if total_stats["total"] > 0 else 0
print(f"Detection accuracy: {total_stats['correct']}/{total_stats['total']} ({accuracy:.0f}%)")
print(f"Average time per frame: {total_stats['total_time_ms'] / total_stats['total']:.1f} ms")
print()
