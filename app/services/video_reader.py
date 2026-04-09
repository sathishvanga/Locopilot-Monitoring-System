"""
Long-lived VideoReader wrapper for voting verification frame extraction.

This module exists to eliminate the per-call ``cv2.VideoCapture`` open/seek
overhead in the voting verification hot loop (ARCH-04 / task 0004).  The
original ``_extract_native_frames_near_timestamp`` implementation in
``voting_verification_service.py`` opened a fresh capture, seeked, read N
frames, and released on every call.  At 0.5 fps sampling over a 30-minute
video that is hundreds of container opens per worker -- and H.264 seeks are
not cheap.

The ``VideoReader`` here caches:

- a single ``cv2.VideoCapture`` instance per video path,
- the video's fps and total frame count (queried once at open),

and exposes a single ``read_frames_near(timestamp_sec, num_frames)`` method
that seeks and reads without ever re-opening the container.  Instances are
meant to be cached per worker process via ``_worker_models['video_readers']``
in ``app/utils/video_multiprocessing.py`` as a small LRU keyed by path.

The class is intentionally dependency-light: no settings, no logger injection,
no batching -- it is a thin wrapper over ``cv2.VideoCapture`` that preserves
the exact windowing semantics of the original helper so behaviour is
byte-identical when the feature flag is enabled or disabled.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Thin wrapper around ``cv2.VideoCapture`` that keeps the capture open
    across multiple ``read_frames_near`` calls.

    Usage::

        reader = VideoReader("/path/to/video.mp4")
        try:
            frames = reader.read_frames_near(timestamp_sec=42.5, num_frames=10)
        finally:
            reader.close()

    Or as a context manager::

        with VideoReader("/path/to/video.mp4") as reader:
            frames = reader.read_frames_near(42.5, 10)

    The ``fps`` and ``total_frames`` attributes are populated at construction
    time from the underlying capture.  If the capture fails to open the
    reader still constructs successfully (``is_open`` returns False) and
    subsequent ``read_frames_near`` calls return an empty list -- this matches
    the fallback behaviour of the original extract helper.
    """

    def __init__(self, video_path: str) -> None:
        """
        Open the video container once and cache fps/total_frames.

        Args:
            video_path: Absolute path to the video file to open.
        """
        self.path: str = video_path
        self.cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.warning("[VideoReader] Failed to open video: %s", video_path)
            self.fps: float = 25.0
            self.total_frames: int = 0
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(
            "[VideoReader] opened %s (fps=%.2f, total_frames=%d)",
            video_path, self.fps, self.total_frames,
        )

    def is_open(self) -> bool:
        """Return True if the underlying capture is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def read_frames_near(
        self,
        timestamp_sec: float,
        num_frames: int,
    ) -> List[np.ndarray]:
        """
        Seek to a frame centered around ``timestamp_sec`` and read
        ``num_frames`` consecutive BGR frames.

        This mirrors the windowing behaviour of the original
        ``_extract_native_frames_near_timestamp`` helper:

        - ``center_frame = int(timestamp_sec * fps)``
        - ``start_frame  = max(0, center_frame - num_frames // 2)``
        - clamp so the window never runs past ``total_frames``.

        Args:
            timestamp_sec: Center timestamp in seconds.
            num_frames: Number of frames to read.

        Returns:
            List of BGR ``numpy.ndarray`` frames (length may be less than
            ``num_frames`` if EOF is reached or reads fail).
        """
        if not self.is_open():
            logger.warning(
                "[VideoReader] read_frames_near on closed capture: %s",
                self.path,
            )
            return []

        # Calculate frame window centered around timestamp
        center_frame = int(timestamp_sec * self.fps)
        start_frame = max(0, center_frame - num_frames // 2)

        # Ensure we don't go past the end
        if self.total_frames > 0 and start_frame + num_frames > self.total_frames:
            start_frame = max(0, self.total_frames - num_frames)

        # Seek to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: List[np.ndarray] = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)

        return frames

    def close(self) -> None:
        """Release the underlying capture.  Idempotent."""
        cap = getattr(self, "cap", None)
        if cap is None:
            return
        try:
            if cap.isOpened():
                cap.release()
                logger.debug("[VideoReader] released capture: %s", self.path)
        except Exception as exc:  # pragma: no cover - cv2 release never fails in practice
            logger.warning("[VideoReader] error releasing capture %s: %s", self.path, exc)
        finally:
            # Drop the reference so subsequent close() calls are no-ops even
            # if the underlying capture mock/object still reports isOpened().
            self.cap = None

    # ----- Context manager protocol --------------------------------------------------

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - destructor timing is nondeterministic
        try:
            self.close()
        except Exception:
            pass


class VideoReaderLRU:
    """
    Tiny LRU cache of ``VideoReader`` instances keyed by video path.

    Designed to live on ``_worker_models['video_readers']`` in the
    multiprocessing worker state.  Typical production usage is a single
    entry (one video per chunk), but a max size of 2 is used so that a
    transition between videos on the same worker does not thrash.

    Not thread-safe on purpose: each worker process owns its own instance.
    """

    def __init__(self, max_size: int = 2) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        # Insertion-ordered dict used as a cheap LRU.
        from collections import OrderedDict
        self._cache: "OrderedDict[str, VideoReader]" = OrderedDict()

    def get_or_create(self, video_path: str) -> VideoReader:
        """
        Return the cached reader for ``video_path``, opening a new one if
        missing.  Touches the LRU so the returned entry is most-recently-used.
        """
        reader = self._cache.get(video_path)
        if reader is not None and reader.is_open():
            self._cache.move_to_end(video_path)
            return reader

        # Either not cached or the cached capture was released; replace it.
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
            self._cache.pop(video_path, None)

        reader = VideoReader(video_path)
        self._cache[video_path] = reader
        self._cache.move_to_end(video_path)

        # Evict LRU entries beyond max_size
        while len(self._cache) > self._max_size:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass

        return reader

    def close_all(self) -> None:
        """Release every cached capture and empty the LRU."""
        for reader in list(self._cache.values()):
            try:
                reader.close()
            except Exception:
                pass
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache
