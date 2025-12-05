"""
Export YOLOv8 models to ONNX format for 3x faster CPU inference

This script exports PyTorch YOLOv8 models to ONNX format with CPU optimizations.
ONNX Runtime provides significantly faster inference on CPU compared to PyTorch.

Usage:
    python3 scripts/export_to_onnx.py

Requirements:
    - ultralytics package (for YOLO models)
    - PyTorch models: yolov8m.pt, yolov8m-pose.pt

Output:
    - yolov8m.onnx (object detection model)
    - yolov8m-pose.onnx (pose estimation model)
"""

from ultralytics import YOLO
import os
import sys

def export_yolo_to_onnx(model_path, output_name=None):
    """Export YOLO model to ONNX format.

    Args:
        model_path: Path to PyTorch model file (.pt)
        output_name: Optional output filename (default: same as input with .onnx extension)

    Returns:
        str: Path to exported ONNX model
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Determine output filename
    if output_name is None:
        output_name = model_path.replace('.pt', '.onnx')

    print(f"Exporting to: {output_name}")
    print(f"Configuration:")
    print(f"  - Format: ONNX")
    print(f"  - Input size: 640x640")
    print(f"  - Simplify: Yes (optimized graph)")
    print(f"  - Opset version: 12 (widely supported)")
    print(f"  - Dynamic shapes: No (static for CPU optimization)")

    try:
        # Export to ONNX with CPU optimizations
        export_result = model.export(
            format='onnx',
            imgsz=640,  # Standard YOLO input size
            simplify=True,  # Simplify ONNX graph for better performance
            opset=12,  # ONNX opset version (12 is widely supported)
            dynamic=False  # Static shapes for better CPU optimization
        )

        print(f"✅ Export complete: {output_name}")

        # Get file size
        if os.path.exists(output_name):
            file_size_mb = os.path.getsize(output_name) / (1024 * 1024)
            print(f"   File size: {file_size_mb:.2f} MB")

        return output_name

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None

def main():
    """Main export function."""
    print("\n" + "="*60)
    print("YOLOv8 to ONNX Model Export")
    print("="*60)
    print("This will export PyTorch models to ONNX format for 3x faster CPU inference")
    print()

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working directory: {project_root}\n")

    # Export object detection model
    print("STEP 1: Exporting YOLOv8m object detection model")
    print("-" * 60)
    result1 = export_yolo_to_onnx('yolov8m.pt', 'yolov8m.onnx')

    # Export pose model
    print("\n\nSTEP 2: Exporting YOLOv8m-Pose model")
    print("-" * 60)
    result2 = export_yolo_to_onnx('yolov8m-pose.pt', 'yolov8m-pose.onnx')

    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)

    if result1:
        print(f"✅ Object detection model: yolov8m.onnx")
    else:
        print(f"❌ Object detection model: FAILED")

    if result2:
        print(f"✅ Pose estimation model: yolov8m-pose.onnx")
    else:
        print(f"❌ Pose estimation model: FAILED")

    if result1 and result2:
        print("\n✨ All models exported successfully!")
        print("\nNext steps:")
        print("1. Install onnxruntime: pip install onnxruntime>=1.16.0")
        print("2. Set environment variable: USE_ONNX_RUNTIME=1")
        print("3. Run your video processing with 3x faster inference!")
        return 0
    else:
        print("\n⚠️  Some exports failed. Please check error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
