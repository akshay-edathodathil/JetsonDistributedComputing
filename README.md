# Jetson Distributed Inference System

Distributed multi-object detection and pose estimation across a Ray cluster:  
**1× GPU PC** (head node) + **4× NVIDIA Jetson** devices (worker nodes).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU PC — 192.168.1.1                        │
│                                                                 │
│  Video / RTSP ──► FrameDistributor ──► ray.put(frame)          │
│                         │                                       │
│              ┌──────────┼──────────────────┐                   │
│              │ least-busy routing          │                   │
│              ▼          ▼        ▼         ▼                   │
│         [future]   [future]  [future]  [future]               │
│              │          │        │         │                   │
│              └──────────┴────────┴─────────┘                   │
│                         │                                       │
│               ResultAggregator                                  │
│         (reorder by frame_id → CSV + display)                  │
│                                                                 │
│  Ray Dashboard ► http://192.168.1.1:8265                       │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
  │  Jetson 1  │ │  Jetson 2  │ │  Jetson 3  │ │  Jetson 4  │
  │ .101       │ │ .102       │ │ .103       │ │ .104       │
  │            │ │            │ │            │ │            │
  │  Worker 0  │ │  Worker 1  │ │  Worker 2  │ │  Worker 3  │
  │  YOLOv8 /  │ │  YOLOv8 /  │ │  YOLOv8 /  │ │  YOLOv8 /  │
  │  RT-DETR / │ │  RT-DETR / │ │  RT-DETR / │ │  RT-DETR / │
  │  RTMPose   │ │  RTMPose   │ │  RTMPose   │ │  RTMPose   │
  └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

**Key design decisions:**
- `ray.put(frame)` stores frames in the plasma object store; workers receive zero-copy `ObjectRef`s
- Actors use `NodeAffinitySchedulingStrategy` — one actor per Jetson, model stays in GPU RAM
- `max_restarts=3` on each actor — Ray automatically restarts a crashed worker
- Results buffered and reordered by `frame_id` before writing/display

---

## Prerequisites

| Component | Requirement |
|---|---|
| GPU PC | Python 3.10, CUDA 12.1, `.venv_mmpose` active |
| Jetsons | JetPack 5.x or 6.x, SSH access from GPU PC |
| Network | All nodes on same LAN, SSH key auth configured |
| Models | `.pt` / `.pth` files placed in `models/` (see below) |

---

## Quick Start

### 1 — Install Ray on the GPU PC

```bash
source /home/cams/Desktop/Akshay/.venv_mmpose/bin/activate
pip install "ray[default]>=2.9.0,<3.0.0"
```

### 2 — First-time cluster setup (syncs code + creates venv on each Jetson)

```bash
cd ~/JetsonDistributedComp   # or wherever the project lives on GPU PC
./cluster_setup.sh setup
```

### 3 — Start the cluster

```bash
./cluster_setup.sh start        # all 4 Jetsons
./cluster_setup.sh start 1      # only 1 Jetson (for testing)
./cluster_setup.sh status       # verify nodes are visible
```

### 4 — Run inference

```bash
# Video file
python ray_head.py --source test_video.mp4 --workers 4 --model yolov8

# RTSP stream
python ray_head.py --source rtsp://192.168.1.200/stream --workers 4

# Webcam on GPU PC
python ray_head.py --source 0 --workers 2 --model rtdetr
```

### 5 — Benchmark

```bash
# Throughput scaling: 1, 2, 4 workers
python benchmark.py --video test_video.mp4 --workers 1,2,4 --frames 300

# Fault tolerance demo
python benchmark.py --video test_video.mp4 --workers 4 --frames 300 --fault-inject
```

---

## Model Setup

Place model files in the `models/` directory and update `config.yaml`:

```yaml
model:
  type: "yolov8"           # yolov8 | rtdetr | rtmpose
  path: "models/yolov8n.pt"
```

| Model | File | Notes |
|---|---|---|
| YOLOv8 | `models/yolov8*.pt` | Standard ultralytics export |
| RT-DETR | `models/rtdetr*.pt` | ultralytics RTDETR class |
| RTMPose | `models/rtmdet*.pth` + `models/rtmpose*.pth` | Needs `rtmpose.det_model_path` and `rtmpose.pose_model_path` in config |

Model conversion from custom `.pth` files is handled separately (not part of this pipeline).

---

## Configuration Reference (`config.yaml`)

```yaml
cluster:
  head_ip: "192.168.1.1"        # GPU PC IP
  ray_port: 6379
  dashboard_port: 8265
  object_store_memory_gb: 4     # Plasma store size on head
  jetson_ips: [...]             # Worker IPs
  username: "nvidia"            # SSH user on Jetsons
  project_dir: "~/JetsonDistributedComp"

model:
  type: yolov8                  # yolov8 | rtdetr | rtmpose
  path: models/yolov8n.pt
  confidence: 0.5
  iou_threshold: 0.45
  img_size: 640
  warmup_iters: 3

pipeline:
  max_in_flight_per_worker: 4   # Backpressure threshold per Jetson
  frame_drop_policy: drop       # drop | block
  result_reorder_buffer: 16     # Frames buffered for temporal reordering

input:
  source: test_video.mp4        # File path, rtsp:// URL, or camera index
  fps_cap: 30

output:
  display: true                 # Show cv2 window on GPU PC
  save_video: false
  csv_path: results/metrics_{timestamp}.csv
```

---

## File Structure

```
JetsonDistributedComp/
├── config.yaml               # All cluster and pipeline parameters
├── cluster_setup.sh          # Cluster control: sync / setup / start / stop / status
├── setup_worker_env.sh       # Jetson-side venv creation (run once per Jetson)
├── models.py                 # Unified ModelLoader: YOLOv8, RT-DETR, RTMPose
├── ray_worker.py             # JetsonInferenceWorker Ray actor
├── ray_head.py               # FrameDistributor + ResultAggregator (GPU PC)
├── benchmark.py              # Throughput/latency tests + fault injection
├── requirements_head.txt     # GPU PC deps (ray[default] only — rest in .venv_mmpose)
├── requirements_worker.txt   # Jetson deps (ray, ultralytics, rtmlib — no torch)
├── models/                   # Place .pt / .pth files here
├── configs/                  # mmpose/mmdet config files (RTMPose only)
└── results/                  # Auto-created: CSVs, benchmark outputs
```

---

## Monitoring

- **Ray Dashboard**: http://192.168.1.1:8265 — actor status, resource utilization, task throughput
- **Per-frame metrics CSV**: `results/metrics_YYYYMMDD_HHMMSS.csv`
  - Columns: `frame_id`, `worker_id`, `inference_ms`, `e2e_latency_ms`, `num_detections`, `gpu_util_pct`
- **Worker JSON logs**: `results/worker_logs_YYYYMMDD_HHMMSS.jsonl`

---

## Troubleshooting

**Ray workers not visible after `./cluster_setup.sh start`**
- Run `./cluster_setup.sh status` to check SSH connectivity
- Verify Jetsons can reach GPU PC: `ssh nvidia@192.168.1.101 ping -c1 192.168.1.1`
- Check Ray port: `ss -tlnp | grep 6379` on GPU PC

**`CUDA not available` on Jetson**
- Verify JetPack torch: `python3 -c "import torch; print(torch.cuda.is_available())"`
- If False, the `--system-site-packages` venv may not be inheriting JetPack's torch
- Try: `source venv/bin/activate && python -c "import torch; print(torch.__file__)"`

**Model file not found**
- Paths in `config.yaml` are relative to the directory where you run `ray_head.py`
- Ensure `models/` is synced to Jetsons: `./cluster_setup.sh sync`

**High dropped frame count**
- Increase `pipeline.max_in_flight_per_worker` in `config.yaml`
- Or reduce `input.fps_cap`
- Or switch `pipeline.frame_drop_policy` to `block` (adds backpressure to source)

**RTMPose: `No module named rtmlib`**
- On Jetson: `source venv/bin/activate && pip install rtmlib`
- If rtmlib unavailable, the mmpose fallback is used automatically

---

## Expected Performance (approximate)

| Workers | Throughput | E2E Latency p50 |
|---|---|---|
| 1 Jetson | ~8–15 fps | ~80–120 ms |
| 2 Jetsons | ~16–28 fps | ~80–120 ms |
| 4 Jetsons | ~30–55 fps | ~80–120 ms |

*Numbers depend on model size, input resolution, and Jetson model (Nano / Xavier / Orin).*
