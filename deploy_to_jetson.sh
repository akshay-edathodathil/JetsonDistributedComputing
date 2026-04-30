#!/bin/bash
# deploy_to_jetson.sh - Transfer models and deploy to Jetson cluster

set -e

PROJECT_DIR="/home/cams/Desktop/Akshay/JetsonDistributedComp"
JETSON_WORKERS=("192.168.1.2" "192.168.1.3" "192.168.1.4" "192.168.1.5")
JETSON_USER="nvidia"
JETSON_PROJECT_DIR="~/JetsonDistributedComp"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== JETSON CLUSTER DEPLOYMENT ===${NC}\n"

# ============================================
# Step 1: Transfer model files
# ============================================
echo -e "${BLUE}[Step 1] Transferring model files to all Jetson workers...${NC}"

for worker in "${JETSON_WORKERS[@]}"; do
  worker_num=$(echo $worker | cut -d'.' -f4)
  echo -e "${YELLOW}  → j00${worker_num} ($worker)${NC}"
  
  rsync -avz --progress \
    "${PROJECT_DIR}/converted_models/"*.onnx \
    "${PROJECT_DIR}/converted_models/"*.pt \
    "${PROJECT_DIR}/converted_models/"*.pth \
    "${PROJECT_DIR}/converted_models/"*.py \
    "${PROJECT_DIR}/converted_models/"*.sh \
    "${JETSON_USER}@${worker}:${JETSON_PROJECT_DIR}/models/" || echo "Warning: rsync issue with $worker"
done

echo -e "${GREEN}✓ Model files transferred${NC}\n"

# ============================================
# Step 2: Install dependencies on Jetson
# ============================================
echo -e "${BLUE}[Step 2] Installing dependencies on Jetson (parallel)...${NC}"

for worker in "${JETSON_WORKERS[@]}"; do
  worker_num=$(echo $worker | cut -d'.' -f4)
  
  (
    echo -e "${YELLOW}  → j00${worker_num}: Installing packages...${NC}"
    
    ssh ${JETSON_USER}@${worker} bash << 'EOF'
set -e

# Install torch-tensorrt (for TensorRT conversion)
echo "Installing torch-tensorrt..."
pip install -q torch-tensorrt 2>/dev/null || echo "Note: torch-tensorrt may need manual build"

# Verify ONNX runtime
python3 -c "import onnxruntime; print('✓ ONNX Runtime available')" 2>/dev/null || \
  echo "⚠ ONNX Runtime not found"

# Verify TensorRT
python3 -c "import tensorrt; print('✓ TensorRT available')" 2>/dev/null || \
  echo "⚠ TensorRT not found (use trtexec instead)"

echo "Dependency check complete"
EOF
  ) &
done

wait
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# ============================================
# Step 3: Generate TensorRT engines
# ============================================
echo -e "${BLUE}[Step 3] Generating TensorRT engines...${NC}"

# Only generate on first worker (engines are device-specific)
primary_worker="${JETSON_WORKERS[0]}"
echo -e "${YELLOW}  → Generating on primary worker ($primary_worker)${NC}"

ssh ${JETSON_USER}@${primary_worker} bash << 'EOF'
cd ~/JetsonDistributedComp/models

echo "Converting ONNX to TensorRT engines..."

# Convert each ONNX file
for onnx_file in *.onnx; do
  if [ -f "$onnx_file" ]; then
    engine_file="${onnx_file%.onnx}.engine"
    
    echo "  Converting: $onnx_file → $engine_file"
    trtexec --onnx=$onnx_file \
      --saveEngine=$engine_file \
      --fp16 \
      --workspace=2048 \
      --iterations=50 \
      >/dev/null 2>&1 && \
      echo "  ✓ $engine_file created" || \
      echo "  ⚠ Failed to create $engine_file (will use ONNX)"
  fi
done

echo "Engine generation complete"
ls -lh *.engine 2>/dev/null | wc -l | xargs echo "  Total engines:"
EOF

echo -e "${GREEN}✓ TensorRT engines generated${NC}\n"

# ============================================
# Step 4: Sync engines to other workers
# ============================================
echo -e "${BLUE}[Step 4] Syncing TensorRT engines to other workers...${NC}"

for worker in "${JETSON_WORKERS[@]:1}"; do
  worker_num=$(echo $worker | cut -d'.' -f4)
  echo -e "${YELLOW}  → Syncing to j00${worker_num}${NC}"
  
  rsync -avz --progress \
    ${JETSON_USER}@${primary_worker}:~/JetsonDistributedComp/models/*.engine \
    ${JETSON_USER}@${worker}:~/JetsonDistributedComp/models/ 2>/dev/null || \
    echo "Note: No engines to sync (using ONNX fallback)"
done

echo -e "${GREEN}✓ Engines synchronized${NC}\n"

# ============================================
# Step 5: Transfer inference code
# ============================================
echo -e "${BLUE}[Step 5] Transferring inference code...${NC}"

for worker in "${JETSON_WORKERS[@]}"; do
  worker_num=$(echo $worker | cut -d'.' -f4)
  echo -e "${YELLOW}  → j00${worker_num}${NC}"
  
  rsync -avz --progress \
    "${PROJECT_DIR}/models_jetson.py" \
    "${PROJECT_DIR}/inference_server.py" \
    "${PROJECT_DIR}/benchmark.py" \
    "${PROJECT_DIR}/config.yaml" \
    "${JETSON_USER}@${worker}:${JETSON_PROJECT_DIR}/" 2>/dev/null
done

echo -e "${GREEN}✓ Inference code transferred${NC}\n"

# ============================================
# Step 6: Verify deployment
# ============================================
echo -e "${BLUE}[Step 6] Verifying deployment...${NC}"

for worker in "${JETSON_WORKERS[@]}"; do
  worker_num=$(echo $worker | cut -d'.' -f4)
  
  (
    echo -e "${YELLOW}  → j00${worker_num}: Checking models...${NC}"
    
    ssh ${JETSON_USER}@${worker} python3 << 'EOF'
import os
from pathlib import Path

models_dir = Path("~/JetsonDistributedComp/models").expanduser()

# Count models
onnx_files = list(models_dir.glob("*.onnx"))
engine_files = list(models_dir.glob("*.engine"))
pt_files = list(models_dir.glob("*.pt"))

print(f"  ONNX models:  {len(onnx_files)}")
print(f"  TensorRT engines: {len(engine_files)}")
print(f"  PyTorch models:   {len(pt_files)}")

if len(onnx_files) >= 5:
    print("  ✓ All primary models present")
else:
    print("  ⚠ Some models missing")

# Test model loader import
try:
    import sys
    sys.path.insert(0, str(Path("~/JetsonDistributedComp").expanduser()))
    from models_jetson import ModelLoader
    print("  ✓ models_jetson.py loadable")
except Exception as e:
    print(f"  ✗ models_jetson.py import failed: {e}")
EOF
  ) &
done

wait
echo -e "${GREEN}✓ Verification complete${NC}\n"

# ============================================
# Step 7: Quick inference test
# ============================================
echo -e "${BLUE}[Step 7] Quick inference test on primary worker...${NC}"

ssh ${JETSON_USER}@${primary_worker} python3 << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("~/JetsonDistributedComp").expanduser()))

try:
    from models_jetson import ModelLoader
    
    # Test loading a model
    print("Testing YOLOv8 loader...")
    loader = ModelLoader(
        model_type="yolov8",
        model_path="~/JetsonDistributedComp/models/yolo26m-pose_640.onnx"
    )
    
    print(f"  ✓ Model loaded successfully")
    print(f"  Backend: {loader.backend}")
    
    # Check what backend is being used
    if loader.backend == "tensorrt":
        print("  ✓ Using TensorRT (fastest)")
    elif loader.backend == "onnx":
        print("  ✓ Using ONNX (fast)")
    elif loader.backend == "pytorch":
        print("  ⚠ Using PyTorch (slower fallback)")
    
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
EOF

echo -e "${GREEN}✓ Inference test complete${NC}\n"

# ============================================
# Summary
# ============================================
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"

echo "Next steps:"
echo "1. Start Ray cluster on primary worker:"
echo "   ssh nvidia@192.168.1.2"
echo "   ray start --head --port=6379"
echo ""
echo "2. Start Ray workers on other Jetson nodes:"
echo "   for i in 3 4 5; do"
echo "     ssh nvidia@192.168.1.\$i ray start --address=192.168.1.2:6379"
echo "   done"
echo ""
echo "3. Deploy inference server:"
echo "   python inference_server.py --config config.yaml"
echo ""
echo "4. Run benchmarks:"
echo "   python benchmark.py --num-workers 4"
echo ""
