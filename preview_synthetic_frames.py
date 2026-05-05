#!/usr/bin/env python3
"""Generate preview frames to verify crop scaling before generating full videos."""

import cv2
import numpy as np
from pathlib import Path
from generate_synthetic_benchmark_videos import (
    load_image_rgba,
    compose_frame,
)


def main():
    # File paths
    bg_paths = {
        "view102": Path("/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view102_empty.jpg"),
        "view108": Path("/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view108_empty.jpg"),
        "view117": Path("/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view_117empty.jpg"),
    }
    elm_crop_path = Path("/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/elm_crop.png")
    jok_crop_path = Path("/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/jok_crop.png")
    
    output_root = Path("/home/cams/homecage_data/DATA/POST/LongVideosForTest")
    durations_h = [8, 10, 12]
    counts = [4, 10, 16, 20]
    seed = 250708
    
    # Load crops once
    elm_rgba = load_image_rgba(elm_crop_path)
    jok_rgba = load_image_rgba(jok_crop_path)
    
    # Create duration directories
    for dh in durations_h:
        (output_root / f"{dh}h").mkdir(parents=True, exist_ok=True)
    
    # Generate preview frames for each view and monkey count
    for view, bg_path in bg_paths.items():
        print(f"\nProcessing {view}...")
        bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg_bgr is None:
            print(f"  ✗ Failed to load {bg_path}")
            continue
        
        for count in counts:
            # Generate one composite frame for this view/count
            composed = compose_frame(bg_bgr, elm_rgba, jok_rgba, count, seed + hash((view, count)) % 100000)
            frame_name = f"preview_{view}_m{count:02d}.png"
            
            # Save to each duration folder (same frame in all folders, just for preview)
            for dh in durations_h:
                output_path = output_root / f"{dh}h" / frame_name
                cv2.imwrite(str(output_path), composed)
                print(f"  ✓ {frame_name} → {dh}h/")
    
    print("\n✅ Preview frames saved!")
    print(f"   View in: /home/cams/homecage_data/DATA/POST/LongVideosForTest/{{8h,10h,12h}}/")
    print("\n   Crop scaling per monkey count:")
    print("     4 monkeys  → 1.2× crop size (easy detection)")
    print("     10 monkeys → 1.0× crop size (baseline)")
    print("     16 monkeys → 0.85× crop size")
    print("     20 monkeys → 0.7× crop size (hard detection)")


if __name__ == "__main__":
    main()
