"""Frame sampling generator extracted from ``locopilot_monitor.py`` (T6).

Lifts ``LocopilotActivityMonitor.sample_video_frames`` (lines 861-930) into
a pure module-level function. ``sample_fps`` and ``logger`` were previously
``self.*`` reads; they are now keyword arguments. The body is otherwise a
byte-identical copy so the rewire in Section 3 (TR) can replace the method
body with a thin forwarder and operators continue to find the same
``[Frame Sampling]`` log strings in production logs.

Imports ``video_capture_context`` from the new shared
``app.core.utils.video_io`` module (also created in T6) instead of
re-defining it locally.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Tuple

import cv2

from app.core.utils.video_io import video_capture_context


def sample_video_frames(
    video_path: str,
    *,
    sample_fps: float,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Iterator[Tuple[int, float, Any, int]]:
    """Sample frames at fixed intervals based on sample_fps.

    Yields tuples: (sample_index, timestamp_sec, frame_bgr, frame_idx)

    Args:
        video_path: Path to video file
        sample_fps: Target sample rate (frames per second).
        start_frame: Optional starting frame index (for range processing)
        end_frame: Optional ending frame index (for range processing)
        logger: Logger to emit ``[Frame Sampling]`` debug lines on. Defaults
            to a module-level logger; callers should pass the monitor's
            ``self.logger`` so log lines end up in the same hierarchy as
            before the extraction.

    Yields:
        sample_index: Sequential index of sampled frames (0, 1, 2, ...)
        timestamp_sec: Timestamp in seconds from video start
        frame_bgr: BGR frame from OpenCV
        frame_idx: Original frame index in the video
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    with video_capture_context(video_path) as cap:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Determine frame range
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else total_frames

        # Calculate stride: how many frames to skip between samples
        step = max(1, int(round(native_fps / max(1e-6, float(sample_fps)))))

        logger.debug(f"[Frame Sampling] Native FPS: {native_fps:.2f}, Sample FPS: {sample_fps}")
        logger.debug(f"[Frame Sampling] Step: {step} (sampling 1 frame every {step} frames)")
        logger.debug(f"[Frame Sampling] Frame range: {start_frame} - {end_frame}")
        logger.debug(f"[Frame Sampling] Expected sampled frames: ~{((end_frame - start_frame) // step)}")

        sampled_idx = 0
        # Start from the beginning of the range, aligned to step
        first_sample_frame = start_frame + (step - (start_frame % step)) % step

        # Single seek to the first sample, then sequential grab()/retrieve().
        # grab() skips full decode where the H.264 bitstream allows (P/B
        # frames can be parsed without YUV conversion), while cap.set()
        # per-sample forces a seek to the preceding keyframe and
        # re-decodes forward — O(keyframe_distance) per sample on
        # typical ~2s GOPs. This gives an order of magnitude speedup
        # for the frame-sampling phase on long H.264 files.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_sample_frame))
        current_frame = first_sample_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = current_frame / native_fps
            yield sampled_idx, timestamp, frame, current_frame
            sampled_idx += 1

            # Advance by (step - 1) grabs to position for the next sample.
            # Bail out if we run past end_frame while grabbing.
            next_sample = current_frame + step
            if next_sample >= end_frame:
                break
            for _ in range(step - 1):
                if not cap.grab():
                    next_sample = end_frame  # force outer loop to exit
                    break
            current_frame = next_sample

        logger.debug(f"[Frame Sampling] Completed sampling, total samples: {sampled_idx}")


__all__ = ["sample_video_frames"]
