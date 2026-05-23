"""Video clip writer + H.264 reencoder lifted from the monolith.

Both functions are stateless. ``save_video_clip`` writes frames using
OpenCV's ``mp4v`` codec, then re-encodes to H.264 (browser-compatible)
via ``reencode_to_h264``. Behavior is byte-identical to the original
``LocopilotActivityMonitor.save_video_clip`` and ``_reencode_to_h264``
methods — log strings, ffmpeg flags, and timeouts are preserved verbatim.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, List, Optional

import cv2


_DEFAULT_LOGGER = logging.getLogger(__name__)


def reencode_to_h264(
    input_path: str,
    *,
    ffmpeg_path: str = '/usr/bin/ffmpeg',
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Re-encode video to H.264 for browser compatibility.

    OpenCV's mp4v codec (MPEG-4 Part 2) doesn't play in browsers.
    This re-encodes to H.264 which has universal browser support.

    Args:
        input_path: Path to the video file to re-encode

    Returns:
        True if re-encoding succeeded, False otherwise
    """
    log = logger if logger is not None else _DEFAULT_LOGGER
    temp_path = input_path + ".temp.mp4"
    try:
        result = subprocess.run([
            ffmpeg_path, '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-loglevel', 'error',
            temp_path
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(temp_path):
            os.replace(temp_path, input_path)
            log.debug(f"Re-encoded to H.264: {input_path}")
            return True
        else:
            stderr = result.stderr.decode() if result.stderr else ""
            log.warning(f"H.264 re-encoding failed (code {result.returncode}): {stderr}")
    except FileNotFoundError:
        log.warning("ffmpeg not found - videos will use mp4v codec (may not play in browsers)")
    except subprocess.TimeoutExpired:
        log.warning(f"H.264 re-encoding timed out for: {input_path}")
    except Exception as e:
        log.warning(f"H.264 re-encoding failed: {e}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return False


def save_video_clip(
    frames: List[Any],
    output_path: str,
    fps: float,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save frames as video clip at sample FPS for full-duration playback.

    Args:
        frames: List of frames to save
        output_path: Path to save video
        fps: FPS to use for video (should be sample_fps for real-time duration)
    """
    if len(frames) == 0:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Use the provided FPS (sample_fps) to create full-duration clips
    # Example: 13 frames @ 0.5 FPS = 26 seconds (real-time)
    # instead of: 13 frames @ 30 FPS = 0.43 seconds (fast-motion)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    # Re-encode to H.264 for browser compatibility
    # (mp4v codec from OpenCV doesn't play in browsers)
    reencode_to_h264(output_path, logger=logger)
