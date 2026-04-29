#!/usr/bin/env bash
# setup_worker_env.sh — Run on each Jetson to create the worker Python environment.
# JetPack provides PyTorch for ARM64; we inherit it via --system-site-packages.
# Do NOT pip install torch here.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
REQUIREMENTS="${PROJECT_DIR}/requirements_worker.txt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# ── Detect system Python ───────────────────────────────────────────────────────
# JetPack 5.x ships Python 3.8; JetPack 6.x ships Python 3.10.
# Prefer python3.10 → python3.8 → python3.
PYTHON=""
for py in python3.10 python3.8 python3; do
    if command -v "$py" &>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "No Python 3 found. Is JetPack installed?"
    exit 1
fi

log "Using Python: $PYTHON ($(${PYTHON} --version))"

# ── Verify JetPack torch is accessible ────────────────────────────────────────
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    err "System torch is not accessible or CUDA is not available."
    err "Ensure JetPack is installed and you're running on a Jetson with GPU."
    err "Continuing anyway — torch check can be re-run after venv creation."
fi

# ── Create venv ───────────────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    log "Venv already exists at $VENV_DIR. Skipping creation."
else
    log "Creating venv at $VENV_DIR with --system-site-packages..."
    $PYTHON -m venv "$VENV_DIR" --system-site-packages
    log "Venv created."
fi

PYTHON_VENV="${VENV_DIR}/bin/python"
PIP_VENV="${VENV_DIR}/bin/pip"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
log "Upgrading pip..."
"$PIP_VENV" install --upgrade pip --quiet

# ── Install worker dependencies ────────────────────────────────────────────────
log "Installing requirements from $REQUIREMENTS..."
# Use --no-deps for torch-adjacent packages to avoid overwriting JetPack torch
"$PIP_VENV" install -r "$REQUIREMENTS"
log "Dependencies installed."

# ── Create results directory ──────────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/configs"

# ── Verify installation ───────────────────────────────────────────────────────
log "Verifying installation..."
"$PYTHON_VENV" - <<'VERIFY'
import sys
errors = []

try:
    import ray
    print(f"  ✓ ray {ray.__version__}")
except ImportError as e:
    errors.append(f"  ✗ ray: {e}")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  ✓ torch {torch.__version__} | CUDA: {cuda_ok} | Device: {torch.cuda.get_device_name(0) if cuda_ok else 'N/A'}")
except ImportError as e:
    errors.append(f"  ✗ torch: {e}")

try:
    import ultralytics
    print(f"  ✓ ultralytics {ultralytics.__version__}")
except ImportError as e:
    errors.append(f"  ✗ ultralytics: {e}")

try:
    import cv2
    print(f"  ✓ opencv {cv2.__version__}")
except ImportError as e:
    errors.append(f"  ✗ opencv: {e}")

try:
    import yaml
    print(f"  ✓ pyyaml")
except ImportError as e:
    errors.append(f"  ✗ pyyaml: {e}")

if errors:
    print("\nFailed imports:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\nAll checks passed.")
VERIFY

log "Worker environment setup complete."
log "Activate with: source ${VENV_DIR}/bin/activate"
