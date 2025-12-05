"""
Quantize ONNX models to INT8 for 2-3x CPU speedup

This script performs INT8 quantization on YOLOv8 ONNX models using
static quantization with calibration data for best accuracy.

Requirements:
    - onnxruntime >= 1.16.0
    - calibration_data/ with calibration frames (run create_calibration_dataset.py first)

Usage:
    python scripts/quantize_to_int8.py [--dynamic]

    --dynamic: Use dynamic quantization (faster, slightly lower accuracy)
               Default: static quantization with calibration data
"""

import argparse
import glob
import os
import cv2
import numpy as np
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        print(f"✅ onnxruntime version: {onnxruntime.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install onnxruntime>=1.16.0")
        return False


def create_calibration_data_reader(calibration_dir: str, input_name: str = 'images'):
    """
    Create a calibration data reader for static quantization.

    Args:
        calibration_dir: Directory containing calibration images
        input_name: Name of model input tensor

    Returns:
        CalibrationDataReader instance
    """
    from onnxruntime.quantization import CalibrationDataReader

    class YOLOCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calib_dir: str, input_name: str):
            self.calib_images = sorted(glob.glob(f'{calib_dir}/*.jpg'))
            if not self.calib_images:
                raise FileNotFoundError(f"No calibration images found in {calib_dir}")

            self.current_idx = 0
            self.input_name = input_name
            print(f"   Loaded {len(self.calib_images)} calibration images")

        def get_next(self):
            """Get next calibration sample."""
            if self.current_idx >= len(self.calib_images):
                return None

            # Load and preprocess image
            img_path = self.calib_images[self.current_idx]
            img = cv2.imread(img_path)
            self.current_idx += 1

            if img is None:
                return self.get_next()  # Skip invalid images

            # YOLO preprocessing
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)   # Add batch dimension

            return {self.input_name: img}

    return YOLOCalibrationDataReader(calibration_dir, input_name)


def quantize_yolo_static(
    model_path: str,
    output_path: str,
    calibration_dir: str = 'calibration_data'
):
    """
    Perform static INT8 quantization with calibration data.

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path for quantized output model
        calibration_dir: Directory with calibration images
    """
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

    print(f"\n🔧 Static INT8 Quantization: {Path(model_path).name}")
    print(f"   Input:  {model_path}")
    print(f"   Output: {output_path}")

    # Check if calibration data exists
    calib_images = glob.glob(f'{calibration_dir}/*.jpg')
    if not calib_images:
        raise FileNotFoundError(
            f"No calibration data found in {calibration_dir}/\n"
            f"Run 'python scripts/create_calibration_dataset.py' first"
        )

    print(f"   Calibration data: {len(calib_images)} images")

    # Create calibration reader
    calibration_reader = create_calibration_data_reader(calibration_dir)

    # Quantize with static calibration
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,  # Quantize-Dequantize format (better for CPU)
        per_channel=True,               # Per-channel quantization (better accuracy)
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True             # Apply additional optimizations
    )

    # Check output file size
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"   ✅ Quantization complete!")
    print(f"   Original size: {original_size:.1f} MB")
    print(f"   Quantized size: {quantized_size:.1f} MB ({reduction:.1f}% reduction)")


def quantize_yolo_dynamic(model_path: str, output_path: str):
    """
    Perform dynamic INT8 quantization (faster but less accurate).

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path for quantized output model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\n🔧 Dynamic INT8 Quantization: {Path(model_path).name}")
    print(f"   Input:  {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Note: Dynamic quantization is faster but may have lower accuracy")

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )

    # Check output file size
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"   ✅ Quantization complete!")
    print(f"   Original size: {original_size:.1f} MB")
    print(f"   Quantized size: {quantized_size:.1f} MB ({reduction:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description='Quantize YOLOv8 ONNX models to INT8')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic quantization')
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 ONNX INT8 Quantization Script")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return

    # Check if ONNX models exist
    models_to_quantize = []
    if os.path.exists('yolov8m.onnx'):
        models_to_quantize.append(('yolov8m.onnx', 'yolov8m_int8.onnx'))
    if os.path.exists('yolov8m-pose.onnx'):
        models_to_quantize.append(('yolov8m-pose.onnx', 'yolov8m-pose_int8.onnx'))

    if not models_to_quantize:
        print("\n❌ No ONNX models found (yolov8m.onnx, yolov8m-pose.onnx)")
        print("   Run 'python scripts/export_to_onnx.py' first")
        return

    # Quantize each model
    for model_path, output_path in models_to_quantize:
        try:
            if args.dynamic:
                quantize_yolo_dynamic(model_path, output_path)
            else:
                quantize_yolo_static(model_path, output_path)
        except Exception as e:
            print(f"\n❌ Error quantizing {model_path}: {e}")
            continue

    print("\n" + "=" * 60)
    print("✅ All quantization tasks complete!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("   1. Test accuracy: Compare INT8 vs FP32 on sample videos")
    print("   2. Enable INT8: Set USE_INT8_QUANTIZATION=1 in .env")
    print("   3. Benchmark: Measure inference speedup")


if __name__ == "__main__":
    main()
