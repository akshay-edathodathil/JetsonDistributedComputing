"""
pytorch_to_tensorrt.py — Convert PyTorch models directly to TensorRT for Jetson.

Approaches to convert RT-DETR and RTMPose when ONNX export fails:

1. ONNX via TorchScript intermediate (most reliable)
2. ONNX with different opset versions
3. Direct torch2tensorrt on Jetson (Jetson-only, requires TensorRT)
4. PyTorch backend on Jetson (acceptable performance)

Usage:

# Approach 1: Export PyTorch to TorchScript first (most reliable)
python pytorch_to_tensorrt.py \
  --model-path /path/to/model.pt \
  --model-type rtdetr \
  --method torchscript \
  --output-dir converted_models

# Approach 2: Try ONNX with different opset versions
python pytorch_to_tensorrt.py \
  --model-path /path/to/model.pt \
  --model-type rtdetr \
  --method onnx_opset \
  --opset-version 14 \
  --output-dir converted_models

# Approach 3: Export RTMPose with correct input size
python pytorch_to_tensorrt.py \
  --model-path /path/to/rtmpose.pth \
  --model-type rtmpose \
  --config-path /path/to/config.py \
  --input-size 384 \
  --method onnx_correct_size \
  --output-dir converted_models
"""

from __future__ import annotations

import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pytorch_to_tensorrt")


def export_via_torchscript(
    model_path: str,
    model_type: str,
    output_dir: str,
    img_size: int = 640,
) -> None:
    """
    Export PyTorch to TorchScript (.ts), then to ONNX.
    
    This works around ONNX export bugs by using TorchScript as intermediate.
    Works for: RT-DETR, YOLOv8, RTMPose
    
    Jetson can convert .ts → TensorRT with torch2tensorrt package.
    """
    from ultralytics import YOLO, RTDETR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_obj = Path(model_path)
    
    logger.info(f"Loading {model_type} from {model_path}")
    
    if model_type.lower() == "rtdetr":
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    
    model.eval()
    
    # Step 1: Trace to TorchScript
    logger.info(f"Tracing {model_type} to TorchScript...")
    try:
        dummy_input = torch.randn(1, 3, img_size, img_size)
        scripted_model = torch.jit.trace(model.model, dummy_input)
        
        ts_path = output_dir / f"{model_path_obj.stem}.ts"
        torch.jit.save(scripted_model, str(ts_path))
        logger.info(f"✓ TorchScript saved: {ts_path}")
        
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        logger.warning("Fallback: Keep PyTorch model on Jetson")
        return
    
    # Step 2: Try ONNX from TorchScript
    logger.info(f"Exporting TorchScript to ONNX...")
    try:
        dummy_input = torch.randn(1, 3, img_size, img_size)
        onnx_path = output_dir / f"{model_path_obj.stem}_from_ts.onnx"
        
        torch.onnx.export(
            scripted_model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        logger.info(f"✓ ONNX from TorchScript: {onnx_path}")
        
    except Exception as e:
        logger.warning(f"ONNX export from TorchScript failed: {e}")
        logger.info(f"TorchScript file still available: {ts_path}")


def export_onnx_different_opsets(
    model_path: str,
    model_type: str,
    output_dir: str,
    img_size: int = 640,
    opset_versions: list[int] = None,
) -> None:
    """
    Try exporting ONNX with different opset versions.
    Some operations are supported in different opsets.
    
    Commonly working: opset 11, 12, 14
    """
    from ultralytics import YOLO, RTDETR
    
    if opset_versions is None:
        opset_versions = [11, 12, 14, 16]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_obj = Path(model_path)
    
    logger.info(f"Loading {model_type} from {model_path}")
    
    if model_type.lower() == "rtdetr":
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)
    
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    for opset_version in opset_versions:
        logger.info(f"\n{'='*70}")
        logger.info(f"Trying opset version: {opset_version}")
        logger.info(f"{'='*70}")
        
        try:
            onnx_path = output_dir / f"{model_path_obj.stem}_opset{opset_version}.onnx"
            
            torch.onnx.export(
                model.model if hasattr(model, 'model') else model,
                dummy_input,
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
            )
            
            # Verify ONNX
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"✓ ONNX opset {opset_version} SUCCESS: {onnx_path}")
            logger.info(f"  Size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
            
        except Exception as e:
            error_msg = str(e)[:100]
            logger.warning(f"✗ Opset {opset_version} failed: {error_msg}")


def export_rtmpose_correct_size(
    model_path: str,
    config_path: str,
    output_dir: str,
    input_size: int = 384,
) -> None:
    """
    Export RTMPose with the correct input size.
    
    The issue is usually shape mismatch because dummy_input doesn't match
    what the model was trained on.
    """
    from mmpose.apis import init_model
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_obj = Path(model_path)
    
    logger.info(f"Loading RTMPose from {model_path} with config {config_path}")
    
    try:
        model = init_model(config_path, model_path, device="cuda:0")
        model.eval()
        
        logger.info(f"Exporting RTMPose with input size: {input_size}x{input_size}")
        
        # Create dummy input with correct size
        dummy_input = torch.randn(1, 3, input_size, input_size).to("cuda:0")
        
        onnx_path = output_dir / f"{model_path_obj.stem}_size{input_size}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=["image"],
                output_names=["output"],
                dynamic_axes={
                    "image": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size"},
                },
                opset_version=12,
                do_constant_folding=True,
                verbose=False,
            )
        
        # Verify
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"✓ RTMPose ONNX (size {input_size}): {onnx_path}")
        logger.info(f"  Size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
        
    except Exception as e:
        logger.error(f"RTMPose export failed: {e}")
        logger.warning("Keep .pth file as fallback on Jetson")


def create_tensorrt_conversion_script(output_dir: str) -> None:
    """
    Create a script for Jetson to convert models to TensorRT.
    This script will be run ON Jetson with torch2tensorrt package.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = output_dir / "jetson_convert_to_tensorrt.sh"
    
    script_content = """#!/bin/bash
# Run this script ON Jetson to convert ONNX/TorchScript to TensorRT
# Requires: torch2tensorrt package

set -e

echo "========================================================================="
echo "TensorRT Conversion for Jetson"
echo "========================================================================="

# For ONNX models - use trtexec (standard tool)
for onnx in *.onnx; do
  echo "Converting ONNX: $onnx"
  trtexec --onnx=$onnx \\
    --saveEngine=${onnx%.onnx}.engine \\
    --fp16 \\
    --workspace=2048 \\
    --minShapes=input:1x3x640x640 \\
    --optShapes=input:1x3x640x640 \\
    --maxShapes=input:1x3x640x640 \\
    --iterations=100
done

# For TorchScript models - use torch2tensorrt (if available)
if command -v python3 &> /dev/null; then
  python3 << 'PYTHON_SCRIPT'
try:
    import torch_tensorrt as torch2trt
    import glob
    
    for ts_file in glob.glob("*.ts"):
        print(f"Converting TorchScript: {ts_file}")
        model = torch.jit.load(ts_file)
        
        # Convert to TensorRT
        trt_model = torch2trt.compile(
            model,
            inputs=[torch.ones(1, 3, 640, 640).cuda()],
            enabled_precisions={torch.float, torch.half},
            workspace_size=2 << 30  # 2GB
        )
        
        output_file = ts_file.replace(".ts", ".ts_trt")
        torch.jit.save(trt_model, output_file)
        print(f"✓ Saved: {output_file}")
        
except ImportError:
    print("torch_tensorrt not installed. Install with:")
    print("  pip install torch-tensorrt")
PYTHON_SCRIPT
fi

echo "========================================================================="
echo "✓ Conversion complete!"
echo "========================================================================="
"""
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    logger.info(f"✓ Created Jetson conversion script: {script_path}")
    logger.info(f"  Copy to Jetson and run: bash jetson_convert_to_tensorrt.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to TensorRT via alternative methods"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model (.pt or .pth)",
    )
    parser.add_argument(
        "--model-type",
        choices=["rtdetr", "yolov8", "rtmpose"],
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Config path (required for RTMPose)",
    )
    parser.add_argument(
        "--method",
        choices=["torchscript", "onnx_opset", "onnx_correct_size"],
        default="torchscript",
        help="Export method",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        nargs="+",
        default=[11, 12, 14, 16],
        help="ONNX opset versions to try",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="converted_models",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    try:
        if args.method == "torchscript":
            export_via_torchscript(
                args.model_path,
                args.model_type,
                args.output_dir,
                args.input_size,
            )
        elif args.method == "onnx_opset":
            export_onnx_different_opsets(
                args.model_path,
                args.model_type,
                args.output_dir,
                args.input_size,
                args.opset_version,
            )
        elif args.method == "onnx_correct_size":
            if not args.config_path:
                parser.error("--config-path required for onnx_correct_size")
            export_rtmpose_correct_size(
                args.model_path,
                args.config_path,
                args.output_dir,
                args.input_size,
            )
        
        # Create Jetson conversion script
        create_tensorrt_conversion_script(args.output_dir)
        
        logger.info("\n" + "="*70)
        logger.info("📌 Next steps:")
        logger.info("="*70)
        logger.info("1. Transfer any generated .onnx/.ts files to Jetson")
        logger.info("2. Transfer jetson_convert_to_tensorrt.sh to Jetson")
        logger.info("3. On Jetson: bash jetson_convert_to_tensorrt.sh")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
