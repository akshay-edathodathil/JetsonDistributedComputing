"""
convert_models.py — Convert PyTorch models to ONNX + TensorRT for Jetson.

Supports:
  • YOLOv8 (.pt) → ONNX + TensorRT engine
  • RT-DETR (.pt) → ONNX + TensorRT engine
  • RTMPose (.pth) → ONNX (RTMPose requires Jetson for TensorRT)
  • Generic detection models → Auto-detected and converted

Usage (GPU PC with training environment):
  # Convert single model to both ONNX and TensorRT
  python convert_models.py \
    --model-path /path/to/model.pt \
    --output-dir converted_models \
    --img-size 640

  # Batch convert all models
  python convert_models.py \
    --batch \
    --model-dir /home/cams/Desktop/ABT_Annotation/models \
    --output-dir converted_models \
    --skip-folders _old_models

  # TensorRT workflow:
  # 1. GPU PC: convert to ONNX (this script)
  # 2. Transfer ONNX to Jetson
  # 3. Jetson: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

Model Storage:
  • GPU PC: Keep both ONNX + TensorRT (for distribution/backup)
  • Jetson: Needs at least ONE format (ONNX recommended for portability)
  • models_jetson.py auto-selects best available: TensorRT > ONNX > PyTorch
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("convert_models")


def convert_yolov8_or_rtdetr(
    model_path: str,
    output_dir: str,
    img_size: int = 640,
    model_type: Optional[str] = None,
) -> None:
    """
    Export YOLOv8 or RT-DETR to ONNX.
    Auto-detects model type from weights if not specified.
    TensorRT conversion done separately on Jetson with trtexec.
    """
    from ultralytics import YOLO, RTDETR
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_obj = Path(model_path)

    logger.info(f"Loading model from {model_path}")
    
    # Auto-detect model type if not specified
    if model_type is None:
        try:
            # Try loading as RTDETR first
            model = RTDETR(model_path)
            model_type = "rtdetr"
            logger.info("✓ Detected as RT-DETR")
        except Exception as e:
            try:
                model = YOLO(model_path)
                model_type = "yolov8"
                logger.info("✓ Detected as YOLOv8")
            except Exception as e2:
                logger.error(f"Could not auto-detect model type: {e2}")
                raise
    else:
        model_type = model_type.lower()
        if model_type == "yolov8":
            model = YOLO(model_path)
        elif model_type == "rtdetr":
            model = RTDETR(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Export to ONNX with error handling for RT-DETR
    logger.info(f"Exporting {model_type} to ONNX...")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=img_size,
            half=False,  # fp32 for better compatibility
            device=0,
            simplify=True,
        )
        logger.info(f"✓ ONNX exported to: {onnx_path}")

        # Create symlink with cleaner name in output_dir
        onnx_output = output_dir / f"{model_path_obj.stem}.onnx"
        if Path(onnx_path) != onnx_output:
            import shutil
            shutil.copy2(onnx_path, onnx_output)
            logger.info(f"✓ Copied to: {onnx_output}")

        # Print TensorRT conversion instructions
        logger.info("\n" + "="*70)
        logger.info("📌 To create TensorRT engine on Jetson:")
        logger.info("="*70)
        logger.info(f"  trtexec --onnx={onnx_output.name} \\")
        logger.info(f"    --saveEngine={onnx_output.stem}.engine \\")
        logger.info(f"    --fp16 --iterations=100")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        if "get_pool_ceil_padding" in str(e) and model_type == "rtdetr":
            logger.warning(f"⚠️  RT-DETR ONNX export failed due to PyTorch/ONNX compatibility")
            logger.warning(f"    RT-DETR has operations (get_pool_ceil_padding) not supported by ONNX")
            logger.warning(f"    Workaround: Keep the .pt model on Jetson for now")
            logger.warning(f"    Or: Use PyTorch backend in models_jetson.py")
            logger.warning(f"    Full error: {str(e)[:200]}...")
            raise
        else:
            raise


def convert_rtmpose(
    model_path: str,
    output_dir: str,
) -> None:
    """
    Export RTMPose (detection or pose) to ONNX.
    Requires mmpose to be installed.
    Auto-detects and finds config file (.py) if available.
    """
    from mmpose.apis import init_model
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_obj = Path(model_path)

    # Try to find config file (.py with same name)
    model_dir = model_path_obj.parent
    config_candidates = [
        model_dir / f"{model_path_obj.stem}.py",
        model_dir / f"{model_path_obj.stem}.json",
        model_dir / f"{model_path_obj.stem}.yml",
        model_dir / f"{model_path_obj.stem}.yaml",
    ]
    
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = str(candidate)
            logger.info(f"✓ Found config: {candidate.name}")
            break
    
    if not config_path:
        logger.error(f"RTMPose config not found! Looked for:")
        for c in config_candidates:
            logger.error(f"  - {c.name}")
        logger.error(f"RTMPose requires a config file alongside the model weights.")
        raise FileNotFoundError(f"No config found for RTMPose model: {model_path}")

    logger.info(f"Loading RTMPose model from {model_path} with config {config_path}")
    
    try:
        model = init_model(config_path, model_path, device="cuda:0")
        logger.info("✓ RTMPose model loaded")
        
        # Export to ONNX
        logger.info("Exporting RTMPose to ONNX...")
        dummy_input = torch.randn(1, 3, 640, 640).to("cuda:0")
        
        onnx_output = output_dir / f"{model_path_obj.stem}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_output),
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
        logger.info(f"✓ ONNX exported to: {onnx_output}")
        
        # Validate ONNX
        import onnx
        onnx_model = onnx.load(str(onnx_output))
        onnx.checker.check_model(onnx_model)
        logger.info(f"✓ ONNX model validated")
        
    except FileNotFoundError as e:
        raise
    except Exception as e:
        logger.error(f"RTMPose conversion failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX for Jetson deployment"
    )
    
    # Single model conversion
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to PyTorch model file to convert",
    )
    parser.add_argument(
        "--model-type",
        choices=["yolov8", "rtdetr", "rtmpose", "auto"],
        default="auto",
        help="Model type (auto-detected if not specified)",
    )
    
    # Batch conversion
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch conversion mode (convert all models in directory)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing models to convert (for batch mode)",
    )
    parser.add_argument(
        "--skip-folders",
        type=str,
        nargs="+",
        default=["_old_models"],
        help="Folders to skip (default: _old_models)",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="converted_models",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size for YOLOv8/RT-DETR",
    )

    args = parser.parse_args()

    try:
        if args.batch:
            # Batch conversion
            if not args.model_dir:
                parser.error("--model-dir required for batch conversion")
            batch_convert(args.model_dir, args.output_dir, args.skip_folders, args.img_size)
        else:
            # Single model conversion
            if not args.model_path:
                parser.error("--model-path required or use --batch mode")
            
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model not found: {args.model_path}")
                return 1
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine model type
            suffix = model_path.suffix.lower()
            if suffix not in [".pt", ".pth"]:
                logger.error(f"Unsupported model format: {suffix}")
                return 1
            
            model_type = args.model_type
            if model_type == "auto":
                # Auto-detect based on filename and model structure
                name_lower = model_path.stem.lower()
                if "rtmpose" in name_lower or suffix == ".pth":
                    model_type = "rtmpose"
                elif "rtdetr" in name_lower:
                    model_type = "rtdetr"
                else:
                    model_type = "yolov8"  # Default for .pt files
            
            if model_type == "rtmpose":
                convert_rtmpose(str(model_path), args.output_dir)
            else:
                convert_yolov8_or_rtdetr(
                    str(model_path),
                    args.output_dir,
                    args.img_size,
                    model_type,
                )
        
        logger.info("\n✓ Conversion complete!")
        logger.info(f"✓ Output directory: {Path(args.output_dir).resolve()}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1

    return 0


def batch_convert(
    model_dir: str,
    output_dir: str,
    skip_folders: list[str],
    img_size: int = 640,
) -> None:
    """
    Batch convert all models in a directory.
    
    Args:
        model_dir: Directory containing models
        output_dir: Output directory for converted models
        skip_folders: List of folder names to skip (e.g., ["_old_models"])
        img_size: Image size for YOLOv8/RT-DETR
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_found = []
    
    # Find all .pt and .pth files, excluding skip folders
    for pattern in ["*.pt", "*.pth"]:
        for model_path in model_dir.glob(pattern):
            # Check if in skip folder
            is_skipped = any(skip in model_path.parts for skip in skip_folders)
            if not is_skipped:
                models_found.append(model_path)
    
    if not models_found:
        logger.warning(f"No models found in {model_dir}")
        return
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Found {len(models_found)} models to convert")
    logger.info(f"{'='*70}\n")
    
    for i, model_path in enumerate(models_found, 1):
        logger.info(f"\n[{i}/{len(models_found)}] Converting: {model_path.name}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024**2):.1f} MB")
        
        try:
            # Auto-detect model type
            name_lower = model_path.stem.lower()
            if "rtmpose" in name_lower or model_path.suffix == ".pth":
                model_type = "rtmpose"
            elif "rtdetr" in name_lower:
                model_type = "rtdetr"
            else:
                model_type = "yolov8"
            
            logger.info(f"  Detected type: {model_type}")
            
            if model_type == "rtmpose":
                convert_rtmpose(str(model_path), str(output_dir))
            else:
                convert_yolov8_or_rtdetr(
                    str(model_path),
                    str(output_dir),
                    img_size,
                    model_type,
                )
                
            logger.info(f"  ✓ Success")
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("Conversion Summary")
    logger.info(f"{'='*70}")
    onnx_files = list(output_dir.glob("*.onnx"))
    logger.info(f"ONNX files created: {len(onnx_files)}")
    for onnx_file in sorted(onnx_files):
        logger.info(f"  • {onnx_file.name} ({onnx_file.stat().st_size / (1024**2):.1f} MB)")
    logger.info(f"\n📌 Next steps:")
    logger.info(f"  1. Transfer ONNX files to Jetson")
    logger.info(f"  2. (Optional) On Jetson, convert to TensorRT with trtexec")
    logger.info(f"  3. Update config.yaml to reference .onnx files")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    exit(main())
