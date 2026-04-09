"""
Unit tests for ``app.services.video_reader``.

These tests mock ``cv2.VideoCapture`` so no real video is required.  They
cover the three behaviours required by ARCH-04 / task 0004:

1. ``VideoReader`` opens a path **exactly once** and reuses the capture across
   multiple ``read_frames_near`` calls.
2. ``close()`` releases the capture.
3. The context manager protocol opens on ``__enter__`` and releases on
   ``__exit__``.
4. ``VideoReaderLRU.get_or_create`` returns the same instance on repeated
   calls for the same path and evicts beyond its max size.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# cv2 constants we reference below.  Importing cv2 in tests is fine because
# the module is already a dependency of the repo (and we patch VideoCapture).
import cv2  # noqa: E402

from app.services.video_reader import VideoReader, VideoReaderLRU  # noqa: E402


def _make_fake_capture(
    fps: float = 25.0,
    total_frames: int = 100,
    frame_shape=(4, 4, 3),
    is_opened: bool = True,
):
    """Build a ``unittest.mock.MagicMock`` that quacks like ``cv2.VideoCapture``."""
    fake_cap = mock.MagicMock(name="cv2.VideoCapture")
    fake_cap.isOpened.return_value = is_opened

    def _get(prop):
        if prop == cv2.CAP_PROP_FPS:
            return fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return total_frames
        return 0.0

    fake_cap.get.side_effect = _get

    # Return a tiny synthetic BGR frame on every read
    frame = np.zeros(frame_shape, dtype=np.uint8)
    fake_cap.read.return_value = (True, frame)
    return fake_cap


class TestVideoReaderReuse:
    """The capture must be opened exactly once and reused across calls."""

    def test_multiple_read_frames_near_calls_reuse_single_capture(self):
        fake_cap = _make_fake_capture(fps=25.0, total_frames=200)
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap) as patched:
            reader = VideoReader("/fake/video.mp4")

            frames_a = reader.read_frames_near(timestamp_sec=1.0, num_frames=5)
            frames_b = reader.read_frames_near(timestamp_sec=3.0, num_frames=5)
            frames_c = reader.read_frames_near(timestamp_sec=5.0, num_frames=5)

            # Exactly ONE VideoCapture instantiation across three calls
            assert patched.call_count == 1
            patched.assert_called_with("/fake/video.mp4")

            # Each call returns the expected number of frames
            assert len(frames_a) == 5
            assert len(frames_b) == 5
            assert len(frames_c) == 5

            # fps / total_frames were cached on open, not re-queried per call
            fps_calls = [c for c in fake_cap.get.call_args_list if c.args == (cv2.CAP_PROP_FPS,)]
            assert len(fps_calls) == 1
            frame_count_calls = [
                c for c in fake_cap.get.call_args_list if c.args == (cv2.CAP_PROP_FRAME_COUNT,)
            ]
            assert len(frame_count_calls) == 1

    def test_read_frames_near_seeks_to_centered_window(self):
        fake_cap = _make_fake_capture(fps=10.0, total_frames=1000)
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/fake/video.mp4")
            reader.read_frames_near(timestamp_sec=5.0, num_frames=10)

            # center_frame = 5 * 10 = 50, start_frame = 50 - 5 = 45
            fake_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 45)

    def test_read_frames_near_clamps_to_total_frames(self):
        fake_cap = _make_fake_capture(fps=10.0, total_frames=60)
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/fake/video.mp4")
            # timestamp past end -> must clamp start to max(0, total-num_frames)
            reader.read_frames_near(timestamp_sec=1000.0, num_frames=10)
            fake_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 50)

    def test_read_frames_near_stops_on_failed_read(self):
        fake_cap = _make_fake_capture(fps=25.0, total_frames=100)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        # 3 good frames then EOF
        fake_cap.read.side_effect = [(True, frame), (True, frame), (True, frame), (False, None)]
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/fake/video.mp4")
            frames = reader.read_frames_near(timestamp_sec=1.0, num_frames=10)
            assert len(frames) == 3

    def test_failed_open_returns_empty_list(self):
        fake_cap = _make_fake_capture(is_opened=False)
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/does/not/exist.mp4")
            assert reader.is_open() is False
            frames = reader.read_frames_near(timestamp_sec=1.0, num_frames=5)
            assert frames == []


class TestVideoReaderClose:
    """``close()`` must release the underlying capture."""

    def test_close_releases_capture(self):
        fake_cap = _make_fake_capture()
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/fake/video.mp4")
            reader.close()
            fake_cap.release.assert_called_once()

    def test_close_is_idempotent(self):
        fake_cap = _make_fake_capture()
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            reader = VideoReader("/fake/video.mp4")
            reader.close()
            # After first release, isOpened flips to False
            fake_cap.isOpened.return_value = False
            reader.close()  # must not raise
            # release should only have been called once (second close skips it)
            assert fake_cap.release.call_count == 1


class TestVideoReaderContextManager:
    """``with VideoReader(...) as r:`` opens once, closes on exit."""

    def test_context_manager_opens_and_closes(self):
        fake_cap = _make_fake_capture()
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap) as patched:
            with VideoReader("/fake/video.mp4") as reader:
                frames = reader.read_frames_near(1.0, 5)
                assert len(frames) == 5
                assert patched.call_count == 1
            fake_cap.release.assert_called_once()

    def test_context_manager_releases_on_exception(self):
        fake_cap = _make_fake_capture()
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap):
            with pytest.raises(RuntimeError):
                with VideoReader("/fake/video.mp4") as reader:
                    assert reader.is_open()
                    raise RuntimeError("boom")
            fake_cap.release.assert_called_once()


class TestVideoReaderLRU:
    """LRU cache wrapping ``VideoReader`` instances."""

    def test_get_or_create_returns_same_instance(self):
        fake_cap = _make_fake_capture()
        with mock.patch("app.services.video_reader.cv2.VideoCapture", return_value=fake_cap) as patched:
            lru = VideoReaderLRU(max_size=2)
            r1 = lru.get_or_create("/video/a.mp4")
            r2 = lru.get_or_create("/video/a.mp4")
            assert r1 is r2
            assert patched.call_count == 1

    def test_lru_evicts_when_full(self):
        fake_caps = [_make_fake_capture() for _ in range(3)]
        with mock.patch(
            "app.services.video_reader.cv2.VideoCapture",
            side_effect=fake_caps,
        ):
            lru = VideoReaderLRU(max_size=2)
            r_a = lru.get_or_create("/video/a.mp4")
            r_b = lru.get_or_create("/video/b.mp4")
            r_c = lru.get_or_create("/video/c.mp4")  # should evict r_a

            assert len(lru) == 2
            assert "/video/a.mp4" not in lru
            assert "/video/b.mp4" in lru
            assert "/video/c.mp4" in lru
            # The evicted capture's release should have been called
            fake_caps[0].release.assert_called_once()
            # The surviving captures should still be open
            fake_caps[1].release.assert_not_called()
            fake_caps[2].release.assert_not_called()
            # Keep r_a and r_b referenced so the eviction is the only reason
            # release was called on fake_caps[0].
            assert r_a is not None
            assert r_b is not None
            assert r_c is not None

    def test_close_all_releases_everything(self):
        fake_caps = [_make_fake_capture() for _ in range(2)]
        with mock.patch(
            "app.services.video_reader.cv2.VideoCapture",
            side_effect=fake_caps,
        ):
            lru = VideoReaderLRU(max_size=4)
            lru.get_or_create("/video/a.mp4")
            lru.get_or_create("/video/b.mp4")
            lru.close_all()
            for fake_cap in fake_caps:
                fake_cap.release.assert_called_once()
            assert len(lru) == 0

    def test_rejects_invalid_max_size(self):
        with pytest.raises(ValueError):
            VideoReaderLRU(max_size=0)

    def test_stale_entry_is_reopened(self):
        """If a cached reader was externally closed, get_or_create opens a new one."""
        fake_caps = [_make_fake_capture() for _ in range(2)]
        with mock.patch(
            "app.services.video_reader.cv2.VideoCapture",
            side_effect=fake_caps,
        ):
            lru = VideoReaderLRU(max_size=2)
            reader_a1 = lru.get_or_create("/video/a.mp4")
            # Simulate external close: flip isOpened flag
            fake_caps[0].isOpened.return_value = False
            reader_a2 = lru.get_or_create("/video/a.mp4")
            assert reader_a2 is not reader_a1


class TestVotingServiceIntegration:
    """VotingVerificationService must accept and use a video_reader_getter."""

    def test_voting_service_accepts_getter_and_delegates_extraction(self):
        """
        End-to-end wiring check: constructing the voting service with a
        ``video_reader_getter`` and calling the private frame-extraction
        helper should delegate to the reader without opening a second
        ``cv2.VideoCapture`` per call.

        ``app.services.video_reader`` and
        ``app.services.voting_verification_service`` both do ``import cv2``
        so they share the same module object.  Patching ``cv2.VideoCapture``
        once is sufficient to intercept both paths.
        """
        from app.services.voting_verification_service import VotingVerificationService

        fake_cap = _make_fake_capture(fps=25.0, total_frames=500)
        with mock.patch(
            "cv2.VideoCapture", return_value=fake_cap
        ) as patched_capture:
            lru = VideoReaderLRU(max_size=2)
            svc = VotingVerificationService(
                yolo_model=None,
                yolo_pose_model=None,
                video_reader_getter=lru.get_or_create,
            )

            # Call the private helper twice (same path, different timestamps)
            frames_a = svc._extract_native_frames(
                "/fake/video.mp4", timestamp_sec=1.0, num_frames=5, activity_type="test"
            )
            frames_b = svc._extract_native_frames(
                "/fake/video.mp4", timestamp_sec=3.0, num_frames=5, activity_type="test"
            )

            # The cv2.VideoCapture constructor is hit *once* total despite
            # the two extraction calls: that's the whole point of ARCH-04.
            assert patched_capture.call_count == 1
            assert len(frames_a) == 5
            assert len(frames_b) == 5

    def test_voting_service_without_getter_uses_direct_capture(self):
        """Single-process fallback: no getter -> direct cv2.VideoCapture()."""
        from app.services.voting_verification_service import VotingVerificationService

        fake_cap = _make_fake_capture(fps=25.0, total_frames=100)
        with mock.patch(
            "cv2.VideoCapture", return_value=fake_cap
        ) as patched:
            svc = VotingVerificationService(
                yolo_model=None, yolo_pose_model=None, video_reader_getter=None
            )
            frames = svc._extract_native_frames(
                "/fake/video.mp4", timestamp_sec=1.0, num_frames=5, activity_type="test"
            )
            # Fallback path DOES open cv2.VideoCapture directly
            assert patched.call_count == 1
            assert len(frames) == 5
            fake_cap.release.assert_called_once()
