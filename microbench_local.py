#!/usr/bin/env python3
"""Local GPU microbenchmark for RT-DETR using ultralytics.
Usage: python3 microbench_local.py --model /path/to/RT-DETR_640.pt --image /path/to/preview.png --iters 30
"""
import argparse, time, cv2, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()

    from ultralytics import RTDETR
    import statistics

    det = RTDETR(args.model)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'Failed to load image: {args.image}')

    # Warmup
    for _ in range(args.warmup):
        _ = det.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False)

    times = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        _ = det(img, conf=0.5, verbose=False)
        times.append((time.perf_counter()-t0)*1000.0)
    
    print('times_ms:', times)
    print('median_ms:', statistics.median(times))
    print('mean_ms:', statistics.mean(times))
    print('min_ms:', min(times), 'max_ms:', max(times))

if __name__ == '__main__':
    main()
