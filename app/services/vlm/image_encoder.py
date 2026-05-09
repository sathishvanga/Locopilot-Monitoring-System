"""Image encoding helpers extracted from ``vlm_verification_service``.

Owns: ROI detection (cropping around the relevant person/object), the
two crop variants used by the strip stitcher, and base64 encoding for
the OpenAI-compatible payload. No behaviour changes — these functions
were copied verbatim from the original monolithic file.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _encode_image(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def _detect_roi(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Compute a tight ROI containing all person + object bboxes that
    Pipeline-1 has rendered into the keyframe.

    Pipeline-1 draws each detected person bbox in bright GREEN and each
    object (book, cup, bottle, phone, bag, etc.) bbox in ORANGE/YELLOW.
    Returns the union of those rectangles as ``(x0, y0, x1, y1)`` so the
    VLM only sees the hand + book/cup region, not the whole cabin.

    Returns None when no qualifying bbox is found (caller should fall back
    to the full frame).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Bright green (Pipeline-1 person bbox): hue ~60, high S+V
    green = cv2.inRange(hsv, (45, 150, 150), (75, 255, 255))
    # Orange/yellow (Pipeline-1 object bbox): hue 15..35
    orange = cv2.inRange(hsv, (15, 150, 150), (35, 255, 255))
    mask = cv2.bitwise_or(green, orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rects: List[Tuple[int, int, int, int]] = []
    for c in contours:
        if cv2.contourArea(c) < 300:  # ignore noise (text, dots)
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        # Reject very thin/long rects (text labels, skeleton lines)
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


def _crop_to_roi(img: np.ndarray, padding: int = 30) -> np.ndarray:
    """Crop the keyframe to the Pipeline-1-bboxes union (with padding).

    Falls back to the original image when no usable ROI is detected.
    """
    h, w = img.shape[:2]
    roi = _detect_roi(img)
    if roi is None:
        return img
    x0, y0, x1, y1 = roi
    # Reject ROIs that already cover most of the image — no benefit cropping
    if (x1 - x0) >= 0.95 * w and (y1 - y0) >= 0.95 * h:
        return img
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return img[y0:y1, x0:x1]


