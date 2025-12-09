"""
Export YOLO models to OpenVINO format

This script exports YOLOv11 PyTorch models to OpenVINO format for optimized CPU inference.
OpenVINO provides 2-3x speedup through Intel/AMD CPU optimizations.

Usage:
    python scripts/export_models_to_openvino.py
"""

from ultralytics import YOLO
import os
import sys


def export_model(model_path, output_name):
    """
    Export YOLO model to OpenVINO format (FP32 only)

    Args:
        model_path: Path to PyTorch model file (.pt)
        output_name: Name for output directory
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return False

    try:
        print(f"\n📦 Exporting {model_path} to OpenVINO...")
        print(f"   Target: {output_name}/")

        model = YOLO(model_path)

        # Export to OpenVINO format
        model.export(
            format='openvino',
            half=False,  # FP32 for accuracy (no quantization)
            dynamic=False,  # Static shape for better optimization
            imgsz=640,  # Match current config
        )

        print(f"✅ Successfully exported to {output_name}/")
        print(f"   Generated files: model.xml, model.bin")
        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def verify_export(output_dir):
    """Verify that exported model files exist"""
    xml_file = os.path.join(output_dir, "model.xml")
    bin_file = os.path.join(output_dir, "model.bin")

    if os.path.exists(xml_file) and os.path.exists(bin_file):
        xml_size = os.path.getsize(xml_file) / (1024 * 1024)  # MB
        bin_size = os.path.getsize(bin_file) / (1024 * 1024)  # MB
        print(f"   ✓ {xml_file} ({xml_size:.2f} MB)")
        print(f"   ✓ {bin_file} ({bin_size:.2f} MB)")
        return True
    else:
        print(f"   ❌ Missing files in {output_dir}")
        return False


def main():
    print("=" * 70)
    print("YOLO Model Export to OpenVINO")
    print("=" * 70)

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print(f"Working directory: {os.getcwd()}\n")

    # Export both models
    models_to_export = [
        ('yolo11s.pt', 'yolo11s_openvino_model'),
        ('yolo11s-pose.pt', 'yolo11s-pose_openvino_model'),
    ]

    results = {}

    for model_path, output_name in models_to_export:
        success = export_model(model_path, output_name)
        results[output_name] = success

    # Verify exports
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    all_success = True
    for output_name, success in results.items():
        if success:
            print(f"\n{output_name}:")
            if not verify_export(output_name):
                all_success = False
        else:
            print(f"\n{output_name}: ❌ Export failed")
            all_success = False

    # Summary
    print("\n" + "=" * 70)
    if all_success:
        print("✅ All models exported successfully!")
        print("\nNext steps:")
        print("1. Verify .xml and .bin files exist in output directories")
        print("2. Test loading models with YOLO()")
        print("3. Continue with backend integration")
    else:
        print("❌ Some exports failed. Please check errors above.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
