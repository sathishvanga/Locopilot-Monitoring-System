"""
Unit tests for ARCH-03: widen mp_overlap_seconds to cover temporal windows.

These tests verify:

1. ``calculate_frame_ranges`` honours the ``overlap_seconds`` parameter such
   that every non-first chunk's ``overlap_start_frame`` precedes its canonical
   ``start_frame`` by at least ``overlap_seconds * native_fps`` frames.

2. ``Settings`` fails fast (ValueError) when the configured
   ``mp_overlap_seconds`` is smaller than either of the two temporal windows
   it must cover (sleep baseline calibration, hand gesture coordination).

3. ``Settings`` accepts a valid configuration where
   ``mp_overlap_seconds >= max(sleep_baseline_calibration_window,
   hand_gesture_coordination_window)``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the repository root is on sys.path so ``app.utils.*`` imports resolve
# whether pytest is invoked from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeVideoCapture:
    """Minimal stand-in for ``cv2.VideoCapture`` used by the frame-range test.

    ``calculate_frame_ranges`` only touches ``isOpened``, ``get`` (for FPS and
    frame count), and ``release``.  A real video file is unnecessary.
    """

    def __init__(self, fps: float, total_frames: int):
        self._fps = float(fps)
        self._total_frames = int(total_frames)

    # cv2.VideoCapture interface --------------------------------------------

    def isOpened(self) -> bool:  # noqa: N802 (cv2 name)
        return True

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._total_frames)
        return 0.0

    def release(self) -> None:
        return None


@pytest.fixture
def fake_video(monkeypatch):
    """Patch cv2.VideoCapture inside video_multiprocessing with a fake."""
    import cv2

    from app.utils import video_multiprocessing as vmp

    fps = 30.0
    total_frames = int(fps * 300)  # 300-second video

    def _factory(_path):
        return _FakeVideoCapture(fps, total_frames)

    monkeypatch.setattr(vmp.cv2, "VideoCapture", _factory)
    # Touch cv2 to keep it imported.
    assert cv2 is not None
    return {"fps": fps, "total_frames": total_frames}


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Ensure Settings is rebuilt fresh between tests."""
    from app.utils import config as config_module

    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# calculate_frame_ranges: overlap frames land before canonical start_frame
# ---------------------------------------------------------------------------


def test_calculate_frame_ranges_overlap_covers_twelve_seconds(fake_video):
    """Every non-first chunk must start at least 12*fps frames before its
    canonical start_frame when ``overlap_seconds=12.0``."""
    from app.utils.video_multiprocessing import calculate_frame_ranges

    fps = fake_video["fps"]
    overlap_seconds = 12.0
    chunk_duration = 30.0  # generous so we get multiple chunks

    frame_ranges, total_frames, native_fps = calculate_frame_ranges(
        video_path="/nonexistent/fake.mp4",
        sample_fps=0.5,
        chunk_duration=chunk_duration,
        min_chunk_duration=2.0,
        overlap_seconds=overlap_seconds,
    )

    assert native_fps == fps
    assert total_frames == fake_video["total_frames"]
    assert len(frame_ranges) >= 2, "need at least two chunks to test overlap"

    expected_overlap_frames = int(overlap_seconds * native_fps)

    # First chunk: overlap_start_frame must equal start_frame (no prior chunk).
    first = frame_ranges[0]
    assert first.overlap_start_frame == first.start_frame

    # Every subsequent chunk: overlap extends back by >= expected_overlap_frames.
    for fr in frame_ranges[1:]:
        assert fr.overlap_start_frame < fr.start_frame, (
            f"chunk {fr.range_id}: overlap_start_frame "
            f"{fr.overlap_start_frame} must be < start_frame {fr.start_frame}"
        )
        gap = fr.start_frame - fr.overlap_start_frame
        assert gap >= expected_overlap_frames, (
            f"chunk {fr.range_id}: overlap gap {gap} frames "
            f"< required {expected_overlap_frames} (12s * {native_fps}fps)"
        )


def test_calculate_frame_ranges_default_overlap_is_twelve_seconds(fake_video):
    """The function's default value of ``overlap_seconds`` is now 12.0.

    Callers that do not pass the parameter explicitly should still receive
    a 12-second warm-up region.
    """
    from app.utils.video_multiprocessing import calculate_frame_ranges

    fps = fake_video["fps"]

    frame_ranges, _, native_fps = calculate_frame_ranges(
        video_path="/nonexistent/fake.mp4",
        sample_fps=0.5,
        chunk_duration=30.0,
        min_chunk_duration=2.0,
        # overlap_seconds intentionally omitted — should default to 12.0
    )

    assert len(frame_ranges) >= 2
    expected_overlap_frames = int(12.0 * native_fps)
    for fr in frame_ranges[1:]:
        gap = fr.start_frame - fr.overlap_start_frame
        assert gap >= expected_overlap_frames


# ---------------------------------------------------------------------------
# Settings validator: too-small overlap raises
# ---------------------------------------------------------------------------


def test_settings_rejects_too_small_overlap():
    """Settings must fail startup when overlap < max(sleep, gesture) windows."""
    from pydantic import ValidationError

    from app.utils.config import Settings

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        Settings(
            mp_overlap_seconds=5.0,
            sleep_baseline_calibration_window=10.0,
            hand_gesture_coordination_window=10.0,
        )

    msg = str(exc_info.value)
    assert "mp_overlap_seconds" in msg
    assert "10" in msg  # required value appears in the message


def test_settings_accepts_matching_overlap():
    """Settings must accept overlap >= max(sleep, gesture) windows."""
    from app.utils.config import Settings

    settings = Settings(
        mp_overlap_seconds=12.0,
        sleep_baseline_calibration_window=10.0,
        hand_gesture_coordination_window=10.0,
    )

    assert settings.mp_overlap_seconds == 12.0
    assert settings.sleep_baseline_calibration_window == 10.0
    assert settings.hand_gesture_coordination_window == 10.0


def test_settings_accepts_overlap_equal_to_largest_window():
    """Boundary case: overlap exactly equal to the largest window is valid."""
    from app.utils.config import Settings

    settings = Settings(
        mp_overlap_seconds=10.0,
        sleep_baseline_calibration_window=8.0,
        hand_gesture_coordination_window=10.0,
    )
    assert settings.mp_overlap_seconds == 10.0


def test_settings_rejects_overlap_smaller_than_gesture_window_only():
    """Regression guard: overlap must cover gesture window too, not just sleep."""
    from pydantic import ValidationError

    from app.utils.config import Settings

    with pytest.raises((ValueError, ValidationError)):
        Settings(
            mp_overlap_seconds=5.0,
            sleep_baseline_calibration_window=2.0,
            hand_gesture_coordination_window=10.0,
        )
