#!/bin/bash
#
# run_benchmark_pipeline.sh
#
# Unattended script that:
# 1. Generates all synthetic benchmark videos (36 total: 3 views × 4 counts × 3 durations)
# 2. Runs the 3-Jetson benchmark playbook
# 3. Logs all output to a timestamped file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/benchmark_run_${TIMESTAMP}.log"

echo "=== Synthetic Benchmark Pipeline ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Generate synthetic videos
# ─────────────────────────────────────────────────────────────────────────────

echo "[1/2] Generating synthetic videos..." | tee -a "$LOG_FILE"
echo "This will create 36 videos (8h/10h/12h durations × 4 monkey counts × 3 views)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 "$SCRIPT_DIR/generate_synthetic_benchmark_videos.py" \
  --background view102=/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view102_empty.jpg \
  --background view108=/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view108_empty.jpg \
  --background view117=/home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/view_117empty.jpg \
  --elm-crop /home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/elm_crop.png \
  --jok-crop /home/cams/Desktop/Akshay/JetsonDistributedComp/LongVideoTests/jok_crop.png \
  --output-dir /home/cams/homecage_data/DATA/POST/LongVideosForTest \
  --host-map view102=j003 \
  --host-map view108=j002 \
  --host-map view117=j004 \
  --counts 4,10,16,20 \
  --durations-h 8,10,12 \
  --fps 5 \
  --seed 250708 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[✓] Video generation complete at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run 3-Jetson benchmark playbook
# ─────────────────────────────────────────────────────────────────────────────

echo "[2/2] Running 3-Jetson benchmark playbook..." | tee -a "$LOG_FILE"
echo "Benchmarking j002, j003, j004 (j001 excluded)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
ansible-playbook -i inventory.yaml run_synthetic_benchmark_3jetsons.yaml 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[✓] Benchmark complete at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "=== Results ===" | tee -a "$LOG_FILE"
echo "Benchmark output: /home/cams/homecage_data/DATA/POST/LongVideosForTest/results/" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
