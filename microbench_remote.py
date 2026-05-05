#!/usr/bin/env python3
"""Remote Jetson microbenchmark. Uses ultralytics RTDETR .pt on Jetson GPU.
Saves JSON results to /home/nvidia/test_frame/microbench_results.json
"""
import argparse, time, cv2, json, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
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

    res = {
        'times_ms': times,
        'median_ms': float(statistics.median(times)),
        'mean_ms': float(statistics.mean(times)),
        'min_ms': float(min(times)),
        'max_ms': float(max(times)),
    }

    outp = '/home/nvidia/test_frame/microbench_results.json'
    with open(outp, 'w') as fh:
        json.dump(res, fh)
    print('WROTE', outp)
    print(res)

if __name__ == '__main__':
    main()
