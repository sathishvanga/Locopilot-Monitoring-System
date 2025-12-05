#!/usr/bin/env python3
"""
Verification script for Phase 1.2: Frame Resolution Reduction for Detection

This script verifies that:
1. Configuration settings are properly loaded
2. Frame preprocessing works correctly
3. Coordinate scaling is accurate
4. Integration with detection pipeline is functional
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import get_settings


def test_configuration():
    """Test that Phase 1.2 configuration is loaded correctly"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)

    settings = get_settings()

    print(f"✓ Detection Width: {settings.detection_width}px")
    print(f"✓ Detection Height: {settings.detection_height}px")
    print(f"✓ Detection Resolution: {settings.detection_resolution}")

    assert settings.detection_width == 640, "Detection width should be 640px"
    assert settings.detection_height == 480, "Detection height should be 480px"
    assert settings.detection_resolution == (640, 480), "Detection resolution tuple mismatch"

    print("\n✅ Configuration test PASSED\n")


def test_frame_preprocessing():
    """Test frame preprocessing and coordinate scaling"""
    print("=" * 60)
    print("TEST 2: Frame Preprocessing")
    print("=" * 60)

    # Create mock monitor instance
    from locopilot_monitor import LocopilotActivityMonitor

    # Create a dummy video path (we won't actually process it)
    monitor = None
    try:
        # Use a minimal initialization (will fail but we can still test methods)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_video = f.name

        # Create a minimal black video for testing
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 1.0, (1280, 720))
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out.write(black_frame)
        out.release()

        monitor = LocopilotActivityMonitor(
            video_path=temp_video,
            create_run_dir=False
        )

        # Test frame preprocessing
        original_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        target_size = (640, 480)

        print(f"Original frame size: {original_frame.shape[:2]} (H x W)")
        print(f"Target detection size: {target_size[1]}x{target_size[0]} (H x W)")

        resized_frame, scale_factors = monitor.preprocess_frame_for_detection(
            original_frame, target_size
        )

        print(f"✓ Resized frame size: {resized_frame.shape[:2]} (H x W)")
        print(f"✓ Scale factors: {scale_factors} (scale_x, scale_y)")

        assert resized_frame.shape[:2] == (480, 640), "Resized frame dimensions incorrect"
        assert scale_factors == (2.0, 1.5), f"Scale factors should be (2.0, 1.5), got {scale_factors}"

        print("\n✅ Frame preprocessing test PASSED\n")

    finally:
        if monitor:
            # Cleanup
            try:
                os.unlink(temp_video)
            except:
                pass


def test_coordinate_scaling():
    """Test coordinate scaling accuracy"""
    print("=" * 60)
    print("TEST 3: Coordinate Scaling")
    print("=" * 60)

    from locopilot_monitor import LocopilotActivityMonitor
    import tempfile
    import cv2

    # Create temporary video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_video = f.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 1.0, (1280, 720))
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out.write(black_frame)
    out.release()

    try:
        monitor = LocopilotActivityMonitor(
            video_path=temp_video,
            create_run_dir=False
        )

        # Test bounding box scaling
        scale_factors = (2.0, 1.5)  # 1280/640=2.0, 720/480=1.5

        # Detection bbox in reduced resolution (640x480)
        detection_bbox = np.array([100, 100, 200, 200])

        # Expected bbox in original resolution (1280x720)
        expected_bbox = np.array([200, 150, 400, 300])

        scaled_bbox = monitor.scale_detection_coordinates(detection_bbox, scale_factors)

        print(f"Detection bbox (640x480): {detection_bbox}")
        print(f"Scaled bbox (1280x720): {scaled_bbox}")
        print(f"Expected bbox: {expected_bbox}")

        # Check accuracy (allow ±0.1 pixel tolerance for floating point)
        np.testing.assert_allclose(scaled_bbox, expected_bbox, atol=0.1)

        # Test with list format
        detection_bbox_list = [100, 100, 200, 200]
        scaled_bbox_list = monitor.scale_detection_coordinates(detection_bbox_list, scale_factors)

        assert isinstance(scaled_bbox_list, list), "Should return list when input is list"
        print(f"✓ List format scaling: {detection_bbox_list} → {scaled_bbox_list}")

        # Test with tuple format
        detection_bbox_tuple = (100, 100, 200, 200)
        scaled_bbox_tuple = monitor.scale_detection_coordinates(detection_bbox_tuple, scale_factors)

        assert isinstance(scaled_bbox_tuple, tuple), "Should return tuple when input is tuple"
        print(f"✓ Tuple format scaling: {detection_bbox_tuple} → {scaled_bbox_tuple}")

        print("\n✅ Coordinate scaling test PASSED\n")

    finally:
        try:
            os.unlink(temp_video)
        except:
            pass


def test_accuracy_tolerance():
    """Test that coordinate scaling is accurate to ±5 pixels"""
    print("=" * 60)
    print("TEST 4: Accuracy Tolerance (±5 pixels)")
    print("=" * 60)

    from locopilot_monitor import LocopilotActivityMonitor
    import tempfile
    import cv2

    # Create temporary video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_video = f.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 1.0, (1280, 720))
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out.write(black_frame)
    out.release()

    try:
        monitor = LocopilotActivityMonitor(
            video_path=temp_video,
            create_run_dir=False
        )

        scale_factors = (2.0, 1.5)

        # Test various bbox positions
        test_cases = [
            (np.array([50, 50, 150, 150]), np.array([100, 75, 300, 225])),
            (np.array([320, 240, 400, 300]), np.array([640, 360, 800, 450])),
            (np.array([0, 0, 100, 100]), np.array([0, 0, 200, 150])),
        ]

        max_error = 0
        for detection, expected in test_cases:
            scaled = monitor.scale_detection_coordinates(detection, scale_factors)
            error = np.max(np.abs(scaled - expected))
            max_error = max(max_error, error)

            print(f"Detection: {detection} → Scaled: {scaled}")
            print(f"  Expected: {expected}, Error: {error:.2f}px")

            assert error <= 5.0, f"Error {error:.2f}px exceeds ±5px tolerance"

        print(f"\n✓ Maximum error: {max_error:.2f}px (within ±5px tolerance)")
        print("\n✅ Accuracy tolerance test PASSED\n")

    finally:
        try:
            os.unlink(temp_video)
        except:
            pass


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("PHASE 1.2: Frame Resolution Reduction - Verification")
    print("=" * 60 + "\n")

    try:
        test_configuration()
        test_frame_preprocessing()
        test_coordinate_scaling()
        test_accuracy_tolerance()

        print("=" * 60)
        print("✅ ALL TESTS PASSED - Phase 1.2 is working correctly!")
        print("=" * 60)
        print("\nExpected Performance Impact:")
        print("  • 25-40% faster YOLO inference")
        print("  • Bounding box accuracy: ±5 pixels")
        print("  • Automatic coordinate scaling to original resolution")
        print("\nUsage:")
        print("  • Set DETECTION_WIDTH=640 in .env (default)")
        print("  • Set DETECTION_HEIGHT=480 in .env (default)")
        print("  • Resolution reduction is applied automatically")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
