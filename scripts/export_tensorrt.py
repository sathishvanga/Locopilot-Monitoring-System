#!/usr/bin/env python3
"""
TensorRT Model Export Script - Phase 2.1 Performance Optimization

Exports YOLO11 models to TensorRT format for 2-4x inference speedup on NVIDIA GPUs.

Usage:
    python scripts/export_tensorrt.py

Requirements:
    - NVIDIA GPU with CUDA support
    - TensorRT installed (pip install tensorrt)
    - ultralytics package

Output:
    - yolo11m.engine (object detection)
    - yolo11m-pose.engine (pose estimation)

After export, update environment variables:
    export YOLO_WEIGHTS_PRELOAD=yolo11m.engine
    export YOLO_POSE_WEIGHTS=yolo11m-pose.engine
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_cuda_available():
    """Check if CUDA is available for TensorRT export."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. TensorRT export requires an NVIDIA GPU.")
        print("Please run this script on a machine with CUDA support.")
        return False

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    return True


def export_model_to_tensorrt(model_path: str, imgsz: int = 640, half: bool = True):
    """
    Export a YOLO model to TensorRT format.

    Args:
        model_path: Path to the .pt model file
        imgsz: Input image size for the engine
        half: Enable FP16 precision (recommended for 2x speedup)

    Returns:
        Path to the exported .engine file
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Exporting: {model_path}")
    print(f"Image Size: {imgsz}")
    print(f"FP16 Precision: {half}")
    print(f"{'='*60}")

    # Load model
    model = YOLO(model_path)

    # Export to TensorRT
    # The export function returns the path to the exported model
    engine_path = model.export(
        format='engine',       # TensorRT format
        imgsz=imgsz,          # Input size
        half=half,            # FP16 precision
        device=0,             # GPU device
        simplify=True,        # Simplify ONNX model
        workspace=4,          # Workspace size in GB
        verbose=True          # Show export progress
    )

    print(f"\nExport complete: {engine_path}")

    # Verify the exported model
    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"Engine file size: {size_mb:.1f} MB")

    return engine_path


def benchmark_model(engine_path: str, iterations: int = 100):
    """
    Benchmark inference speed of the exported model.

    Args:
        engine_path: Path to the .engine file
        iterations: Number of inference iterations for benchmarking
    """
    import time
    import numpy as np
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_path}")
    print(f"{'='*60}")

    # Load TensorRT model
    model = YOLO(engine_path)

    # Create dummy input
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        model(dummy_frame, verbose=False)

    # Benchmark
    print(f"Running {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        model(dummy_frame, verbose=False)

    elapsed_time = time.time() - start_time
    avg_time_ms = (elapsed_time / iterations) * 1000
    fps = iterations / elapsed_time

    print(f"\nResults:")
    print(f"  Average inference time: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")


def main():
    """Main export function."""
    print("="*60)
    print("TensorRT Model Export Script")
    print("Phase 2.1 Performance Optimization")
    print("="*60)

    # Check CUDA availability
    if not check_cuda_available():
        sys.exit(1)

    # Model configurations
    models_to_export = [
        {
            'path': 'yolo11m.pt',
            'imgsz': 640,
            'description': 'Object Detection (YOLO11m)'
        },
        {
            'path': 'yolo11m-pose.pt',
            'imgsz': 640,
            'description': 'Pose Estimation (YOLO11m-pose)'
        }
    ]

    exported_models = []

    for model_config in models_to_export:
        model_path = model_config['path']

        # Check if source model exists
        if not os.path.exists(model_path):
            print(f"\nWARNING: Model not found: {model_path}")
            print("The model will be downloaded automatically during export.")

        try:
            engine_path = export_model_to_tensorrt(
                model_path=model_path,
                imgsz=model_config['imgsz'],
                half=True
            )
            exported_models.append(engine_path)
        except Exception as e:
            print(f"\nERROR exporting {model_path}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)

    if exported_models:
        print("\nSuccessfully exported models:")
        for path in exported_models:
            print(f"  - {path}")

        print("\nTo use TensorRT models, set environment variables:")
        print("  export YOLO_WEIGHTS_PRELOAD=yolo11m.engine")
        print("  export YOLO_POSE_WEIGHTS=yolo11m-pose.engine")

        # Optionally benchmark
        print("\n" + "-"*60)
        benchmark_choice = input("Run benchmark on exported models? (y/n): ").lower().strip()
        if benchmark_choice == 'y':
            for engine_path in exported_models:
                benchmark_model(engine_path)
    else:
        print("No models were exported successfully.")


if __name__ == "__main__":
    main()
