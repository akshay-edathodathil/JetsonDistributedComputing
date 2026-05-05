#!/usr/bin/env python3
"""Generate synthetic long benchmark videos for Jetson pipeline testing.

This script composes one synthetic frame per (view, monkey_count) using:
- one true-negative background per camera view
- one Elm crop (placed on left half)
- one Jok crop (placed on right half)

Then it repeats the composed frame into videos for each requested duration at
5 fps (default), using ffmpeg when available and OpenCV as a fallback.

It writes:
- frames/*.png
- videos/*.mp4
- synthetic_benchmark_manifest.json

Example:
  python3 generate_synthetic_benchmark_videos.py \
    --background view102=/data/view102_empty.png \
    --background view108=/data/view108_empty.png \
    --background view113=/data/view113_empty.png \
    --elm-crop /data/elm_crop.png \
    --jok-crop /data/jok_crop.png \
    --output-dir /home/cams/homecage_data/DATA/POST/250708/jetson_synth_benchmark
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_key_value(items: List[str], name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid {name} entry '{raw}', expected key=path")
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_image_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim != 3:
        raise RuntimeError(f"Image must be HxWxC: {path}")
    if img.shape[2] == 4:
        return img
    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate([img, alpha], axis=2)
    raise RuntimeError(f"Unsupported channel count in {path}: {img.shape[2]}")


def resize_by_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = max(1, target_h)
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def alpha_paste(dst_bgr: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> None:
    fh, fw = fg_rgba.shape[:2]
    H, W = dst_bgr.shape[:2]

    if x >= W or y >= H or x + fw <= 0 or y + fh <= 0:
        return

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + fw)
    y1 = min(H, y + fh)

    fx0 = x0 - x
    fy0 = y0 - y
    fx1 = fx0 + (x1 - x0)
    fy1 = fy0 + (y1 - y0)

    roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    fg = fg_rgba[fy0:fy1, fx0:fx1, :3].astype(np.float32)
    alpha = fg_rgba[fy0:fy1, fx0:fx1, 3:4].astype(np.float32) / 255.0

    dst_bgr[y0:y1, x0:x1] = (fg * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 6) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def place_on_side(
    frame_w: int,
    frame_h: int,
    crops: List[np.ndarray],
    side: str,
    rng: random.Random,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Place crops on left or right side using deterministic grid layout.
    Guarantees no overlap by calculating cell sizes from actual crop dimensions.
    """
    assert side in ("left", "right")
    placed: List[Tuple[np.ndarray, int, int]] = []
    
    if not crops:
        return placed
    
    side_x_min = 0 if side == "left" else frame_w // 2
    side_x_max = frame_w // 2 if side == "left" else frame_w
    side_width = side_x_max - side_x_min
    
    y_min = int(frame_h * 0.10)
    y_max = int(frame_h * 0.95)
    available_height = y_max - y_min
    
    n_crops = len(crops)
    
    # Find max crop dimensions to calculate grid
    max_crop_w = max(c.shape[1] for c in crops)
    max_crop_h = max(c.shape[0] for c in crops)
    
    # Calculate how many columns we can fit (with padding)
    cols_per_row = max(1, side_width // (max_crop_w + 20))  # 20px padding per crop
    rows_needed = (n_crops + cols_per_row - 1) // cols_per_row
    
    # Calculate cell dimensions
    cell_w = side_width // cols_per_row
    cell_h = available_height // max(1, rows_needed)
    
    # Place crops in grid
    for idx, crop in enumerate(crops):
        row = idx // cols_per_row
        col = idx % cols_per_row
        
        # Center crop within cell
        cell_x = side_x_min + col * cell_w
        cell_y = y_min + row * cell_h
        
        crop_h, crop_w = crop.shape[:2]
        x = cell_x + (cell_w - crop_w) // 2
        y = cell_y + (cell_h - crop_h) // 2
        
        # Clamp to bounds
        x = max(side_x_min, min(x, side_x_max - crop_w))
        y = max(y_min, min(y, y_max - crop_h))
        
        placed.append((crop, x, y))
    
    return placed


def compose_frame(bg_bgr: np.ndarray, elm: np.ndarray, jok: np.ndarray, monkey_count: int, seed: int) -> np.ndarray:
    out = bg_bgr.copy()
    H, _W = out.shape[:2]
    rng = random.Random(seed)

    # Scale crops based on monkey count for realistic detection variability
    # Fewer monkeys = larger crops (easier detection), more monkeys = smaller crops (harder detection)
    monkey_count_to_scale = {
        4: 1.2,    # Larger crops, easier detection
        10: 1.0,   # Baseline size
        16: 0.85,  # Smaller crops
        20: 0.7    # Smallest crops, harder detection
    }
    scale_factor = monkey_count_to_scale.get(monkey_count, 1.0)
    height_min = 0.16 * scale_factor
    height_max = 0.28 * scale_factor

    elm_n = monkey_count // 2
    jok_n = monkey_count - elm_n

    elm_crops = []
    for _ in range(elm_n):
        h = int(rng.uniform(height_min, height_max) * H)
        elm_crops.append(resize_by_height(elm, h))

    jok_crops = []
    for _ in range(jok_n):
        h = int(rng.uniform(height_min, height_max) * H)
        jok_crops.append(resize_by_height(jok, h))

    for crop, x, y in place_on_side(out.shape[1], H, elm_crops, "left", rng):
        alpha_paste(out, crop, x, y)

    for crop, x, y in place_on_side(out.shape[1], H, jok_crops, "right", rng):
        alpha_paste(out, crop, x, y)

    return out


def write_video_ffmpeg(still_png: Path, out_mp4: Path, fps: int, duration_h: int) -> None:
    total_seconds = duration_h * 3600
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(still_png),
        "-t",
        str(total_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def write_video_cv(still_bgr: np.ndarray, out_mp4: Path, fps: int, duration_h: int) -> None:
    total_frames = duration_h * 3600 * fps
    h, w = still_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {out_mp4}")
    for _ in range(total_frames):
        writer.write(still_bgr)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark videos and manifest.")
    parser.add_argument(
        "--background",
        action="append",
        required=True,
        help="Background mapping in view=path format. Provide 3 entries.",
    )
    parser.add_argument(
        "--host-map",
        action="append",
        default=["view102=j003", "view108=j002", "view113=j004"],
        help="View to host map in view=host format.",
    )
    parser.add_argument("--elm-crop", required=True, help="Path to Elm crop PNG/JPG.")
    parser.add_argument("--jok-crop", required=True, help="Path to Jok crop PNG/JPG.")
    parser.add_argument("--counts", default="4,10,16,20", help="Monkey counts, comma-separated.")
    parser.add_argument("--durations-h", default="8,10,12", help="Video lengths in hours, comma-separated.")
    parser.add_argument("--fps", type=int, default=5, help="Output FPS (default: 5).")
    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--seed", type=int, default=250708, help="Random seed for deterministic placements.")
    args = parser.parse_args()

    backgrounds = parse_key_value(args.background, "background")
    host_map = parse_key_value(args.host_map, "host-map")
    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    durations_h = [int(x) for x in args.durations_h.split(",") if x.strip()]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    elm_rgba = load_image_rgba(Path(args.elm_crop))
    jok_rgba = load_image_rgba(Path(args.jok_crop))

    ffmpeg_exists = shutil.which("ffmpeg") is not None

    manifest = {
        "fps": args.fps,
        "counts": counts,
        "durations_h": durations_h,
        "host_video_map": {"j002": [], "j003": [], "j004": []},
        "videos": [],
    }

    for view, bg_path_s in backgrounds.items():
        bg_path = Path(bg_path_s)
        host = host_map.get(view, "")
        if host not in manifest["host_video_map"]:
            continue

        bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg_bgr is None:
            raise RuntimeError(f"Failed to load background: {bg_path}")

        for count in counts:
            composed = compose_frame(bg_bgr, elm_rgba, jok_rgba, count, args.seed + hash((view, count)) % 100000)
            frame_name = f"{view}_m{count:02d}.png"

            for dh in durations_h:
                # Create subdirectories per duration: LongVideosForTest/8h/, /10h/, /12h/
                duration_dir = out_root / f"{dh}h"
                duration_dir.mkdir(parents=True, exist_ok=True)
                frames_dir = duration_dir / "frames"
                videos_dir = duration_dir / "videos"
                frames_dir.mkdir(parents=True, exist_ok=True)
                videos_dir.mkdir(parents=True, exist_ok=True)

                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), composed)

                video_name = f"{view}_m{count:02d}_{dh}h_{args.fps}fps.mp4"
                video_path = videos_dir / video_name

                if ffmpeg_exists:
                    write_video_ffmpeg(frame_path, video_path, args.fps, dh)
                else:
                    write_video_cv(composed, video_path, args.fps, dh)

                item = {
                    "host": host,
                    "view": view,
                    "monkey_count": count,
                    "duration_h": dh,
                    "fps": args.fps,
                    "video_name": video_name,
                    "video_path": str(video_path),
                }
                manifest["videos"].append(item)
                manifest["host_video_map"][host].append(item)

    manifest_path = out_root / "synthetic_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Generated videos: {len(manifest['videos'])}")


if __name__ == "__main__":
    main()
