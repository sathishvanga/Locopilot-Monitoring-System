"""Image encoding helpers extracted from ``vlm_verification_service``.

Owns: ROI detection (cropping around the relevant person/object), the
two crop variants used by the strip stitcher, and base64 encoding for
the OpenAI-compatible payload.

ROI selection prefers the coord-driven path (``_roi_from_bboxes``) which
reads Pipeline-1's stored person/object bboxes off the activity dict.
The legacy HSV pixel-recovery path (``_detect_roi``) is retained as a
fallback for activities written before the ``bboxes`` field existed and
for activity types without a meaningful object bbox.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _encode_image(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _roi_from_bboxes(
    bboxes: Optional[Dict[str, Any]],
    frame_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Union of Pipeline-1's stored person + object bboxes, clamped to frame.

    ``bboxes`` is the activity dict's ``bboxes`` field, shaped as::

        {"person": [x1,y1,x2,y2] | None,
         "object": [x1,y1,x2,y2] | None,
         "frame_size": [W, H]   | None}

    Returns ``(x0, y0, x1, y1)`` or ``None`` when no usable rect is
    present. Caller is responsible for the 95%-coverage and padding
    logic (lives in :func:`_crop_to_roi`).

    If ``frame_size`` is recorded and disagrees with the actual keyframe
    dimensions (e.g. the JPEG was resized after detection), bboxes are
    rescaled proportionally so they still land on the right region.
    """
    if not bboxes:
        return None
    rects = [r for r in (bboxes.get("person"), bboxes.get("object")) if r]
    if not rects:
        return None
    h, w = frame_shape
    src = bboxes.get("frame_size")
    if src and isinstance(src, (list, tuple)) and len(src) == 2 and src[0] and src[1]:
        src_w, src_h = src
        if src_w != w or src_h != h:
            sx, sy = w / float(src_w), h / float(src_h)
            rects = [
                [r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy] for r in rects
            ]
    x0 = max(0, int(min(r[0] for r in rects)))
    y0 = max(0, int(min(r[1] for r in rects)))
    x1 = min(w, int(max(r[2] for r in rects)))
    y1 = min(h, int(max(r[3] for r in rects)))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    return (x0, y0, x1, y1)


def _detect_roi(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Fallback ROI: recover Pipeline-1's bboxes from painted pixel colors.

    Used when the activity dict has no ``bboxes`` field (e.g. activities
    written before that field was added, or activity types where the
    coord-driven path returns None). Pipeline-1 draws person bboxes in
    bright GREEN and object bboxes in ORANGE/YELLOW; we HSV-mask both
    and take the union.

    Prefer :func:`_roi_from_bboxes`; this exists only for backward
    compatibility and will be removed once object-bbox plumbing is
    universal (see Task #7).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (45, 150, 150), (75, 255, 255))
    orange = cv2.inRange(hsv, (15, 150, 150), (35, 255, 255))
    mask = cv2.bitwise_or(green, orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 30 or hh < 30:
            continue
        rects.append((x, y, x + ww, y + hh))
    if not rects:
        return None
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _crop_to_roi(
    img: np.ndarray,
    bboxes: Optional[Dict[str, Any]] = None,
    padding: int = 30,
) -> np.ndarray:
    """Crop the keyframe to the Pipeline-1 person+object ROI (with padding).

    Selects ROI via :func:`_roi_from_bboxes` first; falls back to
    :func:`_detect_roi` when no stored bboxes are provided. Falls back
    to the original image when no usable ROI is detected by either
    path, or when the ROI already covers most of the frame.
    """
    h, w = img.shape[:2]
    roi = _roi_from_bboxes(bboxes, (h, w)) if bboxes else None
    if roi is None:
        roi = _detect_roi(img)
    if roi is None:
        return img
    x0, y0, x1, y1 = roi
    if (x1 - x0) >= 0.95 * w and (y1 - y0) >= 0.95 * h:
        return img
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return img[y0:y1, x0:x1]


