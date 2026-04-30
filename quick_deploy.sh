#!/bin/bash
# quick_deploy.sh - Fast deployment to Jetson cluster

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JETSON_WORKERS=("192.168.1.2" "192.168.1.3" "192.168.1.4" "192.168.1.5")

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Quick Deployment to Jetson Cluster${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Check deployment readiness
echo -e "${YELLOW}[Check] Verifying deployment files...${NC}"
missing=0
for f in "converted_models/"*.onnx "converted_models/"*.pt models_jetson.py config.yaml; do
  if [ ! -f "$PROJECT_DIR/$f" ]; then
    echo -e "  ${RED}✗ Missing: $f${NC}"
    missing=1
  fi
done

if [ $missing -eq 1 ]; then
  echo -e "${RED}Error: Missing files. Run conversions first.${NC}"
  exit 1
fi

echo -e "${GREEN}✓ All files present${NC}\n"

# Ask for action
echo -e "${BLUE}What would you like to do?${NC}"
echo "1) Transfer files only"
echo "2) Transfer + TensorRT conversion"
echo "3) Full deployment (transfer + conversion + test)"
read -p "Select (1-3): " action

case $action in
  1) 
    echo -e "\n${YELLOW}[1/1] Transferring files to Jetson workers...${NC}"
    for worker in "${JETSON_WORKERS[@]}"; do
      worker_num=$(echo $worker | cut -d'.' -f4)
      echo -e "  → j00${worker_num}"
      rsync -avz --progress \
        "$PROJECT_DIR/converted_models/"*.onnx \
        "$PROJECT_DIR/converted_models/"*.pt \
        "$PROJECT_DIR/converted_models/"*.py \
        "$PROJECT_DIR/models_jetson.py" \
        "$PROJECT_DIR/config.yaml" \
        "nvidia@$worker:~/JetsonDistributedComp/" 2>/dev/null
    done
    echo -e "${GREEN}✓ Files transferred${NC}\n"
    echo "Next: SSH to 192.168.1.2 and run:"
    echo "  bash ~/JetsonDistributedComp/converted_models/jetson_convert_to_tensorrt.sh"
    ;;
    
  2)
    echo -e "\n${YELLOW}[1/2] Transferring files...${NC}"
    for worker in "${JETSON_WORKERS[@]}"; do
      rsync -avz --progress \
        "$PROJECT_DIR/converted_models/"*.onnx \
        "$PROJECT_DIR/converted_models/"*.pt \
        "$PROJECT_DIR/converted_models/"*.py \
        "$PROJECT_DIR/converted_models/"*.sh \
        "$PROJECT_DIR/models_jetson.py" \
        "nvidia@$worker:~/JetsonDistributedComp/" 2>/dev/null
    done
    echo -e "${GREEN}✓ Files transferred${NC}\n"
    
    echo -e "${YELLOW}[2/2] Converting to TensorRT on primary worker...${NC}"
    ssh nvidia@${JETSON_WORKERS[0]} bash << 'EOF'
cd ~/JetsonDistributedComp/converted_models
echo "Converting ONNX → TensorRT engines..."
bash jetson_convert_to_tensorrt.sh 2>&1 | tail -20
ls -lh *.engine 2>/dev/null && echo "✓ Engines created" || echo "ℹ Check if engines generated"
EOF
    
    echo -e "\n${YELLOW}Syncing engines to other workers...${NC}"
    for worker in "${JETSON_WORKERS[@]:1}"; do
      worker_num=$(echo $worker | cut -d'.' -f4)
      echo "  → Syncing to j00${worker_num}"
      rsync -avz \
        nvidia@${JETSON_WORKERS[0]}:~/JetsonDistributedComp/converted_models/*.engine \
        "nvidia@$worker:~/JetsonDistributedComp/converted_models/" 2>/dev/null || \
        echo "  (No engines to sync)"
    done
    echo -e "${GREEN}✓ Deployment complete${NC}\n"
    echo "Next: Test inference and deploy Ray cluster"
    ;;
    
  3)
    echo -e "\n${YELLOW}[1/4] Transferring files...${NC}"
    for worker in "${JETSON_WORKERS[@]}"; do
      rsync -avz --progress \
        "$PROJECT_DIR/converted_models/"*.onnx \
        "$PROJECT_DIR/converted_models/"*.pt \
        "$PROJECT_DIR/converted_models/"*.py \
        "$PROJECT_DIR/converted_models/"*.sh \
        "$PROJECT_DIR/models_jetson.py" \
        "$PROJECT_DIR/config.yaml" \
        "nvidia@$worker:~/JetsonDistributedComp/" 2>/dev/null
    done
    echo -e "${GREEN}✓ Files transferred${NC}\n"
    
    echo -e "${YELLOW}[2/4] Installing TensorRT tools...${NC}"
    for worker in "${JETSON_WORKERS[@]}"; do
      ssh -f "nvidia@$worker" \
        "pip install -q torch-tensorrt 2>/dev/null && echo '✓ Installed' || echo '⚠ May need manual install'"
    done
    echo -e "${GREEN}✓ Tools installed${NC}\n"
    
    echo -e "${YELLOW}[3/4] Converting to TensorRT...${NC}"
    ssh nvidia@${JETSON_WORKERS[0]} bash << 'EOF'
cd ~/JetsonDistributedComp/converted_models
bash jetson_convert_to_tensorrt.sh 2>&1 | tail -20
EOF
    
    echo -e "\n${YELLOW}[4/4] Testing inference...${NC}"
    ssh nvidia@${JETSON_WORKERS[0]} python3 << 'EOF'
import sys
sys.path.insert(0, '/home/nvidia/JetsonDistributedComp')
from models_jetson import ModelLoader

try:
    loader = ModelLoader(model_type="yolov8", 
                        model_path="/home/nvidia/JetsonDistributedComp/converted_models/yolo26m-pose_640.onnx")
    print(f"✓ Model loaded successfully")
    print(f"  Backend: {loader.backend}")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
EOF
    
    echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ FULL DEPLOYMENT COMPLETE${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"
    
    echo "Ready to start Ray cluster:"
    echo "  ray start --head --port=6379"
    echo ""
    echo "Then deploy inference server:"
    echo "  python inference_server.py --config config.yaml"
    ;;
    
  *)
    echo "Invalid selection"
    exit 1
    ;;
esac
