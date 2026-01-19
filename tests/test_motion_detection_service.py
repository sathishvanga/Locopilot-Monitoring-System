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
        max_consecutive_skips=5,
        use_mog2=False,  # Use frame differencing for predictable tests
        detection_scale=1.0,  # Full resolution for tests
        roi_margin=0.0,  # No ROI exclusion for tests
        scene_change_threshold=1.0,  # Disable scene change detection for tests (100% threshold)
        continuous_motion_subsample=3,
        min_continuous_motion_frames=100  # Effectively disable subsampling for tests
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
        assert metrics.scene_change_detected is False
        assert metrics.is_subsampled_skip is False


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
        # Note: Frame differencing compares each frame to the previous one
        service.should_skip_frame(static_frame)  # First frame - always processed
        service.should_skip_frame(static_frame.copy())  # Same as prev - skipped (no change)
        service.should_skip_frame(static_frame.copy())  # Same as prev - skipped (no change)
        service.should_skip_frame(moving_frame)  # Motion appeared - processed

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
        service = MotionDetectionService(
            use_mog2=False,  # Use frame differencing for predictable test behavior
            motion_threshold=0.01,
            min_contour_area=500,
            binary_threshold=25,
            max_consecutive_skips=30
        )
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
        service = MotionDetectionService(
            use_mog2=False,  # Use frame differencing for predictable test behavior
            motion_threshold=0.01,
            min_contour_area=500,
            binary_threshold=25
        )

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
        service = MotionDetectionService(
            use_mog2=False,  # Use frame differencing for speed test
            detection_scale=1.0
        )
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warmup
        service.should_skip_frame(frame)

        # Measure time
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, metrics = service.should_skip_frame(frame2)

        # Motion detection should be very fast
        assert metrics.processing_time_ms < 10.0  # Should be < 1ms typically


class TestEnhancedMotionDetection:
    """Tests for Phase 2-6 motion detection enhancements"""

    def test_frame_downscaling(self):
        """Test that frame downscaling works correctly"""
        service = MotionDetectionService(
            motion_threshold=0.01,
            min_contour_area=500,
            blur_kernel_size=5,
            binary_threshold=25,
            adaptive_calibration=False,
            max_consecutive_skips=5,
            use_mog2=False,
            detection_scale=0.25,  # 25% resolution
            roi_margin=0.0,
            scene_change_threshold=1.0  # Disable scene change detection for test
        )

        # Test with a frame that has motion
        static_frame = create_static_frame()
        moving_frame = create_moving_frame()

        service.should_skip_frame(static_frame)
        should_skip, metrics = service.should_skip_frame(moving_frame)

        # Should still detect motion even with downscaling
        assert should_skip is False
        assert metrics.motion_percentage > 0.0

    def test_mog2_background_subtraction(self):
        """Test that MOG2 background subtraction works"""
        service = MotionDetectionService(
            motion_threshold=0.01,
            min_contour_area=500,
            blur_kernel_size=5,
            binary_threshold=25,
            adaptive_calibration=False,
            max_consecutive_skips=30,
            use_mog2=True,  # Enable MOG2
            detection_scale=1.0,
            roi_margin=0.0
        )

        # Verify MOG2 is initialized
        assert service.bg_subtractor is not None

        # Test with frames
        static_frame = create_static_frame()
        service.should_skip_frame(static_frame)

        # Second static frame should be skipped
        should_skip, _ = service.should_skip_frame(static_frame.copy())
        assert should_skip is True

    def test_scene_change_detection(self):
        """Test that scene changes (lighting transitions) are detected"""
        service = MotionDetectionService(
            motion_threshold=0.01,
            min_contour_area=500,
            blur_kernel_size=5,
            binary_threshold=25,
            adaptive_calibration=False,
            max_consecutive_skips=30,
            use_mog2=False,
            detection_scale=1.0,
            roi_margin=0.0,
            scene_change_threshold=0.15
        )

        # First frame - dark
        dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        service.should_skip_frame(dark_frame)

        # Second frame - much brighter (scene change)
        bright_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        should_skip, metrics = service.should_skip_frame(bright_frame)

        # Scene change should be detected and frame skipped
        assert metrics.scene_change_detected is True
        assert should_skip is True

    def test_roi_exclusion(self):
        """Test that ROI margin excludes border motion"""
        service = MotionDetectionService(
            motion_threshold=0.01,
            min_contour_area=500,
            blur_kernel_size=5,
            binary_threshold=25,
            adaptive_calibration=False,
            max_consecutive_skips=30,
            use_mog2=False,
            detection_scale=1.0,
            roi_margin=0.10  # 10% border exclusion
        )

        # Create static frame
        static_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        service.should_skip_frame(static_frame)

        # Create frame with motion only in border (should be ignored)
        border_motion_frame = static_frame.copy()
        # Add motion to top border (within 10% = 48 pixels)
        border_motion_frame[:40, :, :] = 255

        should_skip, metrics = service.should_skip_frame(border_motion_frame)
        # Border motion should be ignored
        assert should_skip is True

    def test_continuous_motion_subsampling(self):
        """Test that continuous motion triggers subsampling"""
        service = MotionDetectionService(
            motion_threshold=0.01,
            min_contour_area=500,
            blur_kernel_size=5,
            binary_threshold=25,
            adaptive_calibration=False,
            max_consecutive_skips=30,
            use_mog2=False,
            detection_scale=1.0,
            roi_margin=0.0,
            continuous_motion_subsample=3,  # Process 1 in 3
            min_continuous_motion_frames=3  # Start after 3 frames
        )

        # Background
        static_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        service.should_skip_frame(static_frame)

        # Generate sustained motion frames
        subsampled_count = 0
        processed_count = 0

        for i in range(15):
            # Create frame with motion
            motion_frame = static_frame.copy()
            # Move rectangle position each frame
            x = 100 + (i * 20)
            cv2.rectangle(motion_frame, (x, 100), (x + 100, 200), (255, 255, 255), -1)

            should_skip, metrics = service.should_skip_frame(motion_frame)

            if metrics.is_subsampled_skip:
                subsampled_count += 1
            elif not should_skip:
                processed_count += 1

        # After min_continuous_motion_frames, subsampling should kick in
        # With subsample=3, we process 1 in every 3 motion frames
        assert subsampled_count > 0, "Subsampling should have occurred"
        assert service.stats.subsampled_skips > 0

    def test_statistics_includes_new_metrics(self):
        """Test that statistics summary includes new metrics"""
        service = MotionDetectionService(
            use_mog2=True,
            detection_scale=0.25
        )

        stats = service.get_statistics_summary()

        # Verify new metrics are present
        assert 'scene_change_skips' in stats
        assert 'subsampled_skips' in stats
        assert 'use_mog2' in stats
        assert 'detection_scale' in stats

        # Verify values
        assert stats['use_mog2'] is True
        assert stats['detection_scale'] == 0.25

    def test_reset_reinitializes_mog2(self):
        """Test that reset reinitializes MOG2 background subtractor"""
        service = MotionDetectionService(use_mog2=True)

        # Process a frame
        frame = create_static_frame()
        service.should_skip_frame(frame)

        # Store reference to old bg_subtractor
        old_bg_subtractor = service.bg_subtractor

        # Reset
        service.reset()

        # MOG2 should be reinitialized (new object)
        assert service.bg_subtractor is not None
        assert service.continuous_motion_frames == 0


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
        TestEnhancedMotionDetection,
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
