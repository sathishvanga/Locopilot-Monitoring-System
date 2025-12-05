#!/usr/bin/env python3
"""
Test script for Phase 3.1: Async Frame Reader Integration

This script verifies that the async frame reader is properly integrated
into the main processing pipeline and multiprocessing workers.

Tests:
1. Configuration loading (from environment variables)
2. Main pipeline integration (process_video)
3. Multiprocessing integration (process_video_range)
4. Fallback to synchronous mode on errors
5. Performance comparison (optional)
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import get_settings
from locopilot_monitor import LocopilotActivityMonitor


def create_test_video(output_path, duration_sec=10, fps=30, width=640, height=480):
    """Create a simple test video for testing"""
    print(f"Creating test video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create a simple frame with changing colors
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add some variation
        color_value = int((i / total_frames) * 255)
        frame[:, :] = [color_value, 255 - color_value, 128]

        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Test video created: {total_frames} frames at {fps} FPS")


def test_config_loading():
    """Test 1: Verify configuration loads correctly"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Loading")
    print("="*60)

    settings = get_settings()

    print(f"✓ use_async_frame_reader: {settings.use_async_frame_reader}")
    print(f"✓ async_buffer_size: {settings.async_buffer_size}")

    # Test with environment variable override
    os.environ['USE_ASYNC_FRAME_READER'] = '1'
    os.environ['ASYNC_BUFFER_SIZE'] = '20'

    # Reload settings
    from app.utils.config import Settings
    from functools import lru_cache

    # Clear cache and reload
    get_settings.cache_clear()
    settings = get_settings()

    print(f"\nWith environment variables:")
    print(f"✓ use_async_frame_reader: {settings.use_async_frame_reader}")
    print(f"✓ async_buffer_size: {settings.async_buffer_size}")

    assert settings.use_async_frame_reader == True, "Async reader should be enabled"
    assert settings.async_buffer_size == 20, "Buffer size should be 20"

    print("\n✓ Configuration loading test PASSED")
    return True


def test_async_frame_reader_import():
    """Test 2: Verify AsyncFrameReader can be imported and used"""
    print("\n" + "="*60)
    print("TEST 2: AsyncFrameReader Import and Basic Usage")
    print("="*60)

    try:
        from app.utils.async_frame_reader import AsyncFrameReader
        print("✓ AsyncFrameReader imported successfully")

        # Create test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            test_video_path = f.name

        create_test_video(test_video_path, duration_sec=5, fps=30)

        try:
            # Test basic usage
            print("\nTesting AsyncFrameReader basic usage...")
            frame_count = 0

            with AsyncFrameReader(test_video_path, buffer_size=10, sample_fps=1.0) as reader:
                while True:
                    frame_data = reader.get_frame()
                    if frame_data is None:
                        break
                    sample_idx, timestamp_sec, frame, frame_idx = frame_data
                    frame_count += 1

            print(f"✓ Read {frame_count} frames successfully")
            print("✓ AsyncFrameReader basic usage test PASSED")

            return True
        finally:
            # Clean up test video
            if os.path.exists(test_video_path):
                os.remove(test_video_path)

    except Exception as e:
        print(f"✗ AsyncFrameReader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitor_integration():
    """Test 3: Verify integration with LocopilotActivityMonitor"""
    print("\n" + "="*60)
    print("TEST 3: LocopilotActivityMonitor Integration")
    print("="*60)

    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        test_video_path = f.name

    create_test_video(test_video_path, duration_sec=5, fps=30)

    try:
        # Create output directory
        with tempfile.TemporaryDirectory() as output_dir:
            print(f"\nTest video: {test_video_path}")
            print(f"Output directory: {output_dir}")

            # Ensure async is enabled
            os.environ['USE_ASYNC_FRAME_READER'] = '1'
            os.environ['ASYNC_BUFFER_SIZE'] = '15'

            # Reload settings
            get_settings.cache_clear()

            # Test with async reader enabled
            print("\nTesting with ASYNC frame reader enabled...")
            monitor = LocopilotActivityMonitor(
                video_path=test_video_path,
                output_dir=output_dir,
                save_annotated_frames=False,
                sample_fps=1.0
            )

            # Verify settings are loaded
            assert monitor.settings.use_async_frame_reader == True
            print(f"✓ Monitor settings loaded correctly")
            print(f"  - use_async_frame_reader: {monitor.settings.use_async_frame_reader}")
            print(f"  - async_buffer_size: {monitor.settings.async_buffer_size}")

            print("\n✓ Monitor integration test PASSED")
            return True

    except Exception as e:
        print(f"✗ Monitor integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_synchronous_fallback():
    """Test 4: Verify synchronous fallback works"""
    print("\n" + "="*60)
    print("TEST 4: Synchronous Fallback")
    print("="*60)

    # Disable async reader
    os.environ['USE_ASYNC_FRAME_READER'] = '0'
    get_settings.cache_clear()

    settings = get_settings()
    print(f"✓ Async reader disabled: {settings.use_async_frame_reader}")

    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        test_video_path = f.name

    create_test_video(test_video_path, duration_sec=3, fps=30)

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            print("\nTesting with SYNCHRONOUS frame reader...")
            monitor = LocopilotActivityMonitor(
                video_path=test_video_path,
                output_dir=output_dir,
                save_annotated_frames=False,
                sample_fps=1.0
            )

            assert monitor.settings.use_async_frame_reader == False
            print(f"✓ Monitor using synchronous mode")

            print("\n✓ Synchronous fallback test PASSED")
            return True

    except Exception as e:
        print(f"✗ Synchronous fallback test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_multiprocessing_integration():
    """Test 5: Verify multiprocessing worker integration"""
    print("\n" + "="*60)
    print("TEST 5: Multiprocessing Worker Integration")
    print("="*60)

    # Enable async reader
    os.environ['USE_ASYNC_FRAME_READER'] = '1'
    os.environ['ASYNC_BUFFER_SIZE'] = '10'
    get_settings.cache_clear()

    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        test_video_path = f.name

    create_test_video(test_video_path, duration_sec=5, fps=30)

    try:
        with tempfile.TemporaryDirectory() as output_dir:
            print(f"\nTest video: {test_video_path}")

            # Create monitor
            monitor = LocopilotActivityMonitor(
                video_path=test_video_path,
                output_dir=output_dir,
                save_annotated_frames=False,
                sample_fps=1.0
            )

            # Test process_video_range (used by multiprocessing workers)
            print("\nTesting process_video_range with async reader...")

            # Get total frames
            cap = cv2.VideoCapture(test_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"Total frames: {total_frames}")

            # Process a range
            start_frame = 0
            end_frame = total_frames // 2

            print(f"Processing range: {start_frame} - {end_frame}")

            activities = monitor.process_video_range(
                start_frame=start_frame,
                end_frame=end_frame,
                save_clips=False
            )

            print(f"✓ Processed frame range successfully")
            print(f"  - Activities detected: {len(activities)}")

            print("\n✓ Multiprocessing integration test PASSED")
            return True

    except Exception as e:
        print(f"✗ Multiprocessing integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 3.1: ASYNC FRAME READER INTEGRATION TESTS")
    print("="*60)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("AsyncFrameReader Import", test_async_frame_reader_import),
        ("Monitor Integration", test_monitor_integration),
        ("Synchronous Fallback", test_synchronous_fallback),
        ("Multiprocessing Integration", test_multiprocessing_integration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, result in results if result)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
