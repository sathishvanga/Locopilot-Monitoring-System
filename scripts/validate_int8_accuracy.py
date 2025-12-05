"""
Validate INT8 quantization accuracy against FP32 baseline

Compares detection results between FP32 and INT8 models to ensure
accuracy degradation is within acceptable limits (<5%).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.onnx_yolo_wrapper import ONNXYOLODetector


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    intersection = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def compare_detections(fp32_results, int8_results, iou_threshold: float = 0.5) -> Dict:
    """
    Compare detection results between FP32 and INT8 models.

    Args:
        fp32_results: Results from FP32 model
        int8_results: Results from INT8 model
        iou_threshold: IoU threshold for matching detections

    Returns:
        Dictionary with comparison metrics
    """
    fp32_boxes = [box.xyxy[0] for box in fp32_results[0].boxes]
    int8_boxes = [box.xyxy[0] for box in int8_results[0].boxes]

    fp32_confs = [box.conf[0] for box in fp32_results[0].boxes]
    int8_confs = [box.conf[0] for box in int8_results[0].boxes]

    # Calculate metrics
    matches = 0
    conf_diffs = []

    for i, fp32_box in enumerate(fp32_boxes):
        best_iou = 0
        best_match = -1

        for j, int8_box in enumerate(int8_boxes):
            iou = calculate_iou(fp32_box, int8_box)
            if iou > best_iou:
                best_iou = iou
                best_match = j

        if best_iou >= iou_threshold and best_match >= 0:
            matches += 1
            conf_diff = abs(float(fp32_confs[i]) - float(int8_confs[best_match]))
            conf_diffs.append(conf_diff)

    precision = matches / len(int8_boxes) if int8_boxes else 0
    recall = matches / len(fp32_boxes) if fp32_boxes else 0
    avg_conf_diff = np.mean(conf_diffs) if conf_diffs else 0

    return {
        'fp32_count': len(fp32_boxes),
        'int8_count': len(int8_boxes),
        'matches': matches,
        'precision': precision,
        'recall': recall,
        'avg_conf_diff': avg_conf_diff
    }


def extract_test_frames(video_dir: str, num_frames: int = 50) -> List[np.ndarray]:
    """
    Extract test frames from videos in a directory.

    Args:
        video_dir: Directory containing test videos
        num_frames: Number of frames to extract

    Returns:
        List of extracted frames
    """
    # Find video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(Path(video_dir).glob(ext))

    if not video_paths:
        print(f"  No videos found in {video_dir}")
        return []

    # Extract frames evenly from first video
    video_path = str(video_paths[0])
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = min(num_frames, total_frames)

    indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def validate_quantization(test_images_dir: str = 'sample_videos', num_samples: int = 50):
    """
    Validate INT8 quantization accuracy.

    Args:
        test_images_dir: Directory with test images/videos
        num_samples: Number of frames to test
    """
    print("=" * 60)
    print("INT8 Quantization Accuracy Validation")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    try:
        fp32_model = ONNXYOLODetector('yolov8m.onnx')
        print("  ✅ FP32 model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load FP32 model: {e}")
        return False

    try:
        int8_model = ONNXYOLODetector('yolov8m_int8.onnx')
        print("  ✅ INT8 model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load INT8 model: {e}")
        print(f"     Run 'python scripts/quantize_to_int8.py' first")
        return False

    # Extract test frames
    print(f"\nExtracting {num_samples} test frames...")
    test_frames = extract_test_frames(test_images_dir, num_samples)

    if not test_frames:
        print("  ❌ No test frames found")
        print(f"     Add videos to {test_images_dir}/ directory")
        return False

    print(f"  ✅ Extracted {len(test_frames)} frames")

    # Compare detections
    print("\nComparing detections...")
    all_metrics = []

    for idx, frame in enumerate(test_frames):
        try:
            fp32_result = fp32_model(frame)
            int8_result = int8_model(frame)

            metrics = compare_detections(fp32_result, int8_result)
            all_metrics.append(metrics)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(test_frames)} frames")
        except Exception as e:
            print(f"  Warning: Error processing frame {idx}: {e}")
            continue

    if not all_metrics:
        print("  ❌ No metrics collected")
        return False

    # Aggregate results
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_conf_diff = np.mean([m['avg_conf_diff'] for m in all_metrics])
    total_fp32 = sum([m['fp32_count'] for m in all_metrics])
    total_int8 = sum([m['int8_count'] for m in all_metrics])

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Frames tested:     {len(all_metrics)}")
    print(f"FP32 detections:   {total_fp32}")
    print(f"INT8 detections:   {total_int8}")
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall:    {avg_recall:.2%}")
    print(f"Avg Conf Diff:     {avg_conf_diff:.4f}")

    # Determine pass/fail
    if avg_precision >= 0.95 and avg_recall >= 0.95:
        print("\n✅ PASSED: INT8 quantization maintains >95% accuracy")
        print("   Safe to use in production with USE_INT8_QUANTIZATION=1")
        return True
    else:
        print("\n⚠️  WARNING: Accuracy below 95% threshold")
        print("   Consider:")
        print("   - Using dynamic quantization (--dynamic flag)")
        print("   - Adding more diverse calibration data")
        print("   - Sticking with FP32 ONNX (USE_INT8_QUANTIZATION=0)")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate INT8 quantization accuracy')
    parser.add_argument('--video-dir', default='sample_videos', help='Directory with test videos')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of frames to test')
    args = parser.parse_args()

    validate_quantization(args.video_dir, args.num_samples)
