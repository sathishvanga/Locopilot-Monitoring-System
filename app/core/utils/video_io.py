"""Video I/O helpers extracted from ``locopilot_monitor.py`` (T6).

This module owns the small ``video_capture_context`` context manager that
guarantees ``cv2.VideoCapture.release()`` runs even on exceptions. It is a
direct lift of the helper at lines 145-156 of ``locopilot_monitor.py``;
behavior must remain byte-identical so the rewire (Section 3 / TR) can
delete the local definition and import from here without changing any log
strings or runtime behavior.
"""
from __future__ import annotations

import contextlib

import cv2


@contextlib.contextmanager
def video_capture_context(video_path):
    """
    Context manager to ensure VideoCapture is always released.
    This prevents memory leaks from unclosed video captures.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        if cap.isOpened():
            cap.release()


__all__ = ["video_capture_context"]
