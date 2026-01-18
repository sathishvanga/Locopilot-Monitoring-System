"""
Unit tests for MotionDetectionService

Tests cover:
- Static frames should be skipped
- Moving frames should be processed
- Consecutive skip limit is respected (safety valve)
- Adaptive calibration works correctly
- Statistics tracking is accurate

Can be run with pytest if available, or directly with python:
    python tests/test_motion_detection_service.py
"""

import numpy as np
import cv2
import sys
import os
import importlib.util

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Direct import to avoid package hierarchy issues with config.py
spec = importlib.util.spec_from_file_location(
    'motion_detection_service',
    os.path.join(project_root, 'app', 'services', 'motion_detection_service.py')
)
motion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion_module)

MotionDetectionService = motion_module.MotionDetectionService
MotionMetrics = motion_module.MotionMetrics
MotionDetectionStats = motion_module.MotionDetectionStats

# Try to import pytest, fall back to simple test runner
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# ============================================================================
# Test Helpers
# ============================================================================

def create_service():
    """Create a fresh MotionDetectionService instance"""
    return MotionDetectionService(
        motion_threshold=0.01,
        min_contour_area=500,
        blur_kernel_size=5,
        binary_threshold=25,
        adaptive_calibration=False,  # Disable for predictable tests
        max_consecutive_skips=5
    )


def create_static_frame():
    """Create a static test frame (solid gray)"""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


def create_moving_frame():
    """Create a frame with significant motion (white rectangle on gray)"""
    frame = create_static_frame()
    # Draw a large white rectangle (significant change)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), -1)
    return frame


# ============================================================================
# Test Classes (compatible with pytest)
# ============================================================================

class TestMotionMetrics:
    """Tests for MotionMetrics dataclass"""

    def test_default_values(self):
        """Test that default values are sensible"""
        metrics = MotionMetrics()
        assert metrics.motion_score == 0.0
        assert metrics.motion_area == 0
        assert metrics.motion_percentage == 0.0
        assert metrics.contour_count == 0
        assert metrics.has_significant_motion is True  # Default: process frame
        assert metrics.processing_time_ms == 0.0


class TestMotionDetectionStats:
    """Tests for MotionDetectionStats dataclass"""

    def test_skip_rate_calculation(self):
        """Test skip rate percentage calculation"""
        stats = MotionDetectionStats(
            frames_analyzed=100,
            frames_skipped=30,
            frames_processed=70
        )
        assert stats.skip_rate == 30.0

    def test_skip_rate_zero_division(self):
        """Test skip rate handles zero frames analyzed"""
        stats = MotionDetectionStats()
        assert stats.skip_rate == 0.0

    def test_avg_processing_time(self):
        """Test average processing time calculation"""
        stats = MotionDetectionStats(
            frames_analyzed=10,
            total_processing_time_ms=50.0
        )
        assert stats.avg_processing_time_ms == 5.0


class TestMotionDetectionService:
    """Tests for MotionDetectionService"""

    def test_first_frame_always_processed(self):
        """First frame should always be processed (no previous frame to compare)"""
        service = create_service()
        static_frame = create_static_frame()
        should_skip, metrics = service.should_skip_frame(static_frame)

        assert should_skip is False
        assert metrics.has_significant_motion is True

    def test_static_frames_skipped(self):
        """Identical frames should be skipped"""
        service = create_service()
        static_frame = create_static_frame()

        # First frame - always processed
        should_skip1, _ = service.should_skip_frame(static_frame)
        assert should_skip1 is False

        # Second identical frame - should be skipped
        should_skip2, metrics = service.should_skip_frame(static_frame.copy())
        assert should_skip2 is True
        assert metrics.motion_percentage == 0.0

    def test_moving_frames_processed(self):
        """Frames with significant motion should be processed"""
        service = create_service()
        static_frame = create_static_frame()
        moving_frame = create_moving_frame()

        # First frame
        service.should_skip_frame(static_frame)

        # Frame with motion - should NOT be skipped
        should_skip, metrics = service.should_skip_frame(moving_frame)

        assert should_skip is False
        assert metrics.motion_percentage > 0.0
        assert metrics.has_significant_motion is True

    def test_consecutive_skip_limit_respected(self):
        """Safety valve: max consecutive skips is respected"""
        service = create_service()
        static_frame = create_static_frame()

        # Process first frame
        service.should_skip_frame(static_frame)

        # Submit max_consecutive_skips identical frames
        for i in range(service.max_consecutive_skips - 1):
            should_skip, _ = service.should_skip_frame(static_frame.copy())
            assert should_skip is True, f"Frame {i+2} should be skipped"

        # Next frame should be forced through (safety valve)
        should_skip, _ = service.should_skip_frame(static_frame.copy())
        assert should_skip is False, "Safety valve should force processing"

    def test_statistics_tracking(self):
        """Statistics should be accurately tracked"""
        service = create_service()
        static_frame = create_static_frame()
        moving_frame = create_moving_frame()

        # Process a mix of frames
        service.should_skip_frame(static_frame)  # First frame - processed
        service.should_skip_frame(static_frame.copy())  # Static - skipped
        service.should_skip_frame(moving_frame)  # Motion - processed
        service.should_skip_frame(static_frame.copy())  # Static - skipped

        stats = service.get_statistics_summary()

        assert stats['frames_analyzed'] == 4
        assert stats['frames_skipped'] == 2
        assert stats['frames_processed'] == 2
        assert stats['skip_rate_percent'] == 50.0

    def test_reset_clears_state(self):
        """Reset should clear all state"""
        service = create_service()
        static_frame = create_static_frame()

        # Process some frames
        service.should_skip_frame(static_frame)
        service.should_skip_frame(static_frame.copy())

        # Reset
        service.reset()

        # Verify state is cleared
        assert service._prev_frame_gray is None
        assert service._frame_count == 0
        assert service.stats.frames_analyzed == 0

    def test_blur_kernel_must_be_odd(self):
        """Blur kernel size should be odd"""
        # Even kernel should be converted to odd
        service = MotionDetectionService(blur_kernel_size=4)
        assert service.blur_kernel_size == 5

        # Odd kernel should remain unchanged
        service2 = MotionDetectionService(blur_kernel_size=7)
        assert service2.blur_kernel_size == 7


class TestMotionDetectionWithAdaptiveCalibration:
    """Tests for adaptive calibration feature"""

    def test_calibration_learns_baseline(self):
        """Calibration should learn baseline motion from first N frames"""
        service = MotionDetectionService(
            motion_threshold=0.001,
            adaptive_calibration=True,
            calibration_frames=5,
            max_consecutive_skips=10
        )

        # Create frames with slight noise (simulating camera shake)
        base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Submit calibration frames with slight variations
        for i in range(service.calibration_frames + 1):
            frame = base_frame.copy()
            # Add small random noise
            noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            service.should_skip_frame(frame)

        # Verify calibration occurred
        stats = service.get_statistics_summary()
        assert stats['is_calibrated'] is True
        assert stats['baseline_motion'] is not None
        assert stats['baseline_motion'] >= 0.0

    def test_effective_threshold_uses_max(self):
        """Effective threshold should be max of configured and calibrated"""
        service = MotionDetectionService(
            motion_threshold=0.001,
            adaptive_calibration=True,
            calibration_frames=5,
            max_consecutive_skips=10
        )
        base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Before calibration, should use configured threshold
        assert service._get_effective_threshold() == service.motion_threshold

        # After calibration with some motion, baseline should be considered
        for i in range(service.calibration_frames + 1):
            frame = base_frame.copy()
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            service.should_skip_frame(frame)

        effective = service._get_effective_threshold()
        assert effective >= service.motion_threshold


class TestMotionDetectionRealWorldScenarios:
    """Tests simulating real-world video scenarios"""

    def test_gradual_lighting_change(self):
        """Gradual lighting changes should not trigger motion"""
        service = MotionDetectionService()
        base_value = 100
        frames_processed = 0
        frames_skipped = 0

        # First frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * base_value
        should_skip, _ = service.should_skip_frame(frame)
        if not should_skip:
            frames_processed += 1

        # Gradually change brightness
        for i in range(10):
            brightness = base_value + i  # +1 brightness per frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * brightness
            should_skip, _ = service.should_skip_frame(frame)
            if should_skip:
                frames_skipped += 1
            else:
                frames_processed += 1

        # Most frames with gradual change should be skipped
        # (binary threshold of 25 filters out small changes)
        assert frames_skipped > frames_processed

    def test_person_moving_triggers_motion(self):
        """A person moving through the frame should trigger motion detection"""
        service = MotionDetectionService()

        # Background frame
        background = np.ones((480, 640, 3), dtype=np.uint8) * 128
        service.should_skip_frame(background)

        # Person enters frame (simulated as moving rectangle)
        frame_with_person = background.copy()
        cv2.rectangle(frame_with_person, (200, 100), (350, 400), (80, 80, 80), -1)

        should_skip, metrics = service.should_skip_frame(frame_with_person)

        assert should_skip is False
        assert metrics.motion_percentage > 0.01  # Significant motion

    def test_processing_time_is_fast(self):
        """Motion detection should be fast (< 10ms per frame)"""
        service = MotionDetectionService()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warmup
        service.should_skip_frame(frame)

        # Measure time
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, metrics = service.should_skip_frame(frame2)

        # Motion detection should be very fast
        assert metrics.processing_time_ms < 10.0  # Should be < 1ms typically


# ============================================================================
# Simple Test Runner (for when pytest is not available)
# ============================================================================

def run_tests_without_pytest():
    """Simple test runner when pytest is not available"""
    test_classes = [
        TestMotionMetrics,
        TestMotionDetectionStats,
        TestMotionDetectionService,
        TestMotionDetectionWithAdaptiveCalibration,
        TestMotionDetectionRealWorldScenarios,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print("=" * 60)
    print("Running Motion Detection Service Tests")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        # Find all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            method = getattr(instance, method_name)

            try:
                method()
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  [ERROR] {method_name}: {type(e).__name__}: {e}")
                failed_tests.append((test_class.__name__, method_name, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return 1

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    if HAS_PYTEST:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        sys.exit(run_tests_without_pytest())
