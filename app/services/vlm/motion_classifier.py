"""Window-region frame-diff motion classifier.

Pipeline-1's vibration-based motion detector is fooled by diesel-idle
vibration at this trainset's stations: it reports RUNNING during station
halts where the engine idles at high amplitude. The VLM is also unreliable
because writing/eating/etc. activities CROP the keyframes to the
person+object ROI before sending — the cabin window (the only motion cue
visible to the VLM) is removed by that crop.

This module fixes that gap by computing pixel-diff in the cabin's WINDOW
region directly from the *uncropped* keyframes the VLM service already
loaded. If the window region shows little change between consecutive
keyframes, the train is stationary — return a synthetic STOPPED verdict
that the existing structural rules will demote to FALSE_POSITIVE/UNCERTAIN.

The window position differs by camera, so we use EasyOCR on the
in-frame text overlay (e.g. ``CAB 1 ALP camera 3`` / ``CAB 1 LP camera 2``)
to identify the camera and look up its window ROI.

Camera detection runs once per video and is cached by source-video
filename, so subsequent activities for the same video pay no OCR cost.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton EasyOCR reader for camera-overlay detection. Used to live in the
# old OCRTimestampService; inlined here when that service was deleted so we
# don't load the model twice.
# ---------------------------------------------------------------------------
_easyocr_reader = None
_easyocr_reader_lock = threading.Lock()


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        with _easyocr_reader_lock:
            if _easyocr_reader is None:
                try:
                    import easyocr
                except ImportError:
                    logger.warning(
                        "[motion_classifier] easyocr not installed; "
                        "camera detection disabled"
                    )
                    return None
                logger.info(
                    "[motion_classifier] Initializing EasyOCR reader "
                    "(this may take a moment)..."
                )
                _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("[motion_classifier] EasyOCR reader initialized")
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Per-camera window ROI table.
#
# ROI coordinates are (x1, y1, x2, y2) in the camera's native 1280x720 frame.
# Validated against the 7-activity batch on 2026-05-10 (5 stationary FPs + 1
# running TP); chosen so the cabin window grille / doorway falls within the
# ROI but the dashboard / persons do not.
# ---------------------------------------------------------------------------
CAMERA_WINDOW_ROIS: Dict[str, Tuple[int, int, int, int]] = {
    # ALP camera 3 (CAB 1): window grille on the upper-right
    "ALP_CAM3": (850, 0, 960, 400),
    # LP camera 2 (CAB 1): window on the upper-left
    "LP_CAM2": (0, 0, 200, 400),
}

# Direct ``cameraAngle`` (from /analyze) -> ROI lookup. Used when the
# caller already knows which side it is, bypassing OCR entirely. The
# OCR path stays as a fallback for legacy callers / when cameraAngle is
# unknown. Values mirror CAMERA_WINDOW_ROIS so a fleet retuning of one
# table propagates to the other.
_ROI_BY_CAMERA_ANGLE: Dict[int, Tuple[int, int, int, int]] = {
    1: CAMERA_WINDOW_ROIS["LP_CAM2"],   # 1 = LP  (ch03)
    2: CAMERA_WINDOW_ROIS["ALP_CAM3"],  # 2 = ALP (ch02)
}


# Threshold tuned on the 7-clip batch: the lowest motion score for the
# only TP (vid11, running) was 13.15; the highest score for the 5 actually-
# posted FPs (excluding vid06a which was already filtered by Pipeline-1)
# was 7.55. A threshold of 10.0 sits cleanly between the two.
#
# This is a CONSERVATIVE threshold — when in doubt (no camera detected,
# OCR failure, ROI extraction failure), we DO NOT override. Falls back to
# the existing VLM verdict pipeline.
MOTION_DIFF_STOPPED_THRESHOLD: float = 10.0


# Per-source-video cache for the detected camera ID. Many activities share
# a source video; OCR is expensive (~150 ms) so we pay it once and reuse.
_camera_cache: Dict[str, str] = {}
_camera_cache_lock = threading.Lock()


# Recognised camera-overlay text patterns. The CCTV overlay shows e.g.
# ``CAB 1 ALP camera 3`` or ``CAB 1 LP camera 2``. EasyOCR is unreliable:
# observed mangled outputs include ``CAB amera 3`` (missing 'c' from
# 'camera'), ``CAU 1 ALR era 3`` (CAB→CAU, ALP→ALR, camera→era), and
# bare ``3`` / ``2`` tokens. We match loosely on the trailing digit
# token next to a 'camera'-like word (with optional missing letters).
_CAMERA_WORD = r"(?:c?am(?:e?r?a?)|era)"  # camera, amera, era, am, etc.

_ALP_TOKENS = re.compile(
    r"\bALR\b|\bA[Ll]P\b|"        # explicit ALP role token (or its OCR variants)
    rf"{_CAMERA_WORD}\s*3\b|"     # camera 3 / amera 3 / era 3 / etc.
    r"\bcamera_3\b",
    re.IGNORECASE,
)
_LP_TOKENS = re.compile(
    rf"{_CAMERA_WORD}\s*2\b|"     # camera 2 / amera 2 / era 2 / etc.
    r"\bcamera_2\b|"
    r"(?<![A-Za-z])LP(?![A-Za-z])",  # bare 'LP' (not part of ALP)
    re.IGNORECASE,
)


def _detect_camera_from_text(text: str) -> Optional[str]:
    """Match an OCR'd text fragment against the known camera overlays.

    Strategy: ALP tokens take priority because ``ALP`` contains ``LP``,
    so a frame that has ``ALP camera 3`` would otherwise also match the
    LP rule. Once we see any ALP/camera-3 evidence, we commit to ALP_CAM3
    and ignore other matches in the same blob.
    """
    if not text:
        return None
    if _ALP_TOKENS.search(text):
        return "ALP_CAM3"
    if _LP_TOKENS.search(text):
        return "LP_CAM2"
    return None


def detect_camera_id(jpg_path: Path) -> Optional[str]:
    """OCR the camera-identifier overlay from a single keyframe.

    Returns the canonical camera id (e.g. ``ALP_CAM3``) or ``None`` if
    OCR is unavailable / the overlay cannot be matched.

    The text overlay is expected in the bottom-right region of the frame
    (per the CCTV stamping convention on this trainset). We crop to that
    region first so EasyOCR has fewer distractors and runs faster.
    """
    img = cv2.imread(str(jpg_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    # Bottom-right text overlay region. Validated against the deployed
    # CCTV overlay (text observed at y=558-590, x=924-1217 on a 1280x720
    # frame). We take a generous 35%-wide × 25%-tall slab so OCR has
    # whitespace margin around the line.
    x0, y0 = int(w * 0.65), int(h * 0.75)
    roi = img[y0:h, x0:w]

    try:
        reader = _get_easyocr_reader()
        if reader is None:
            return None
        # detail=0 returns just the text strings, fast-paragraph option
        # OFF because the overlay is one short line.
        results = reader.readtext(roi, detail=0, paragraph=False)
    except Exception as e:
        logger.debug("[motion_classifier] OCR failed for %s: %s", jpg_path, e)
        return None

    text_blob = " ".join(results) if results else ""
    cam_id = _detect_camera_from_text(text_blob)
    if cam_id is None:
        logger.debug(
            "[motion_classifier] could not match camera overlay in %s "
            "(OCR text=%r)", jpg_path.name, text_blob[:80],
        )
    return cam_id


def detect_camera_for_video(
    video_filename: str, keyframes,
) -> Optional[str]:
    """Return the cached camera id for ``video_filename`` or detect it now.

    Tries OCR on EACH keyframe in turn until one succeeds — different
    frames have different camera-overlay clarity (LP body sometimes
    occludes the text), so a single-frame attempt is unreliable.

    Subsequent calls with the same ``video_filename`` are O(1).
    """
    if not video_filename:
        return None
    with _camera_cache_lock:
        cached = _camera_cache.get(video_filename)
        if cached is not None:
            return cached if cached != "__none__" else None

    # Iterate keyframes until one yields a recognisable camera id.
    cam_id: Optional[str] = None
    if keyframes:
        for kf in keyframes:
            cam_id = detect_camera_id(Path(kf))
            if cam_id is not None:
                break

    with _camera_cache_lock:
        # Cache negatives too so we don't re-OCR on every activity for a
        # video where OCR can't read the overlay.
        _camera_cache[video_filename] = cam_id or "__none__"
    return cam_id


def _window_roi_diff(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     roi: Tuple[int, int, int, int]) -> float:
    """Mean absolute pixel diff inside ``roi`` between two grayscale frames."""
    x1, y1, x2, y2 = roi
    h, w = prev_gray.shape
    # Clamp ROI to actual frame bounds.
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = prev_gray[y1:y2, x1:x2]
    b = curr_gray[y1:y2, x1:x2]
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(a, b)))


def compute_window_motion_score(
    jpg_paths,
    camera_id: str,
    roi_override: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Dict[str, float]]:
    """Compute median window-ROI pixel diff across consecutive keyframes.

    ``camera_id`` is used to look up the ROI in :data:`CAMERA_WINDOW_ROIS`
    when ``roi_override`` is not supplied. Pass ``roi_override`` from the
    cameraAngle-direct path so OCR-derived ``camera_id`` is bypassable.

    Returns a dict ``{score, threshold, stopped, n_frames, latency_sec}``
    or ``None`` if the camera is unknown / inputs are insufficient.
    """
    roi = roi_override if roi_override is not None else CAMERA_WINDOW_ROIS.get(camera_id)
    if roi is None or len(jpg_paths) < 2:
        return None

    t0 = time.time()
    grays = []
    for p in jpg_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if len(grays) < 2:
        return None

    diffs = [
        _window_roi_diff(grays[i], grays[i + 1], roi)
        for i in range(len(grays) - 1)
    ]
    score = float(np.median(diffs))
    return {
        "score": score,
        "threshold": MOTION_DIFF_STOPPED_THRESHOLD,
        "stopped": score < MOTION_DIFF_STOPPED_THRESHOLD,
        "n_frames": len(grays),
        "n_pairs": len(diffs),
        "latency_sec": time.time() - t0,
        "camera_id": camera_id,
        "roi": roi,
    }


def _parse_roi(text: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        parts = [int(v.strip()) for v in text.split(",")]
        return (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None
    except Exception:
        return None


def parse_roi_overrides_from_settings(
    settings,
) -> Optional[Dict[int, Tuple[int, int, int, int]]]:
    """Read per-camera ROI overrides for the window classifier.

    Reads ``window_motion_roi_lp`` and ``window_motion_roi_alp`` from a
    pydantic Settings object. Returns ``None`` if neither is set.
    """
    overrides: Dict[int, Tuple[int, int, int, int]] = {}
    lp = _parse_roi(getattr(settings, "window_motion_roi_lp", "") or "")
    alp = _parse_roi(getattr(settings, "window_motion_roi_alp", "") or "")
    if lp:
        overrides[1] = lp
    if alp:
        overrides[2] = alp
    return overrides or None


def classify_motion(
    video_filename: str,
    keyframes,
    camera_angle: Optional[int] = None,
    roi_overrides: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> Optional[Dict[str, float]]:
    """Top-level entry: detect camera + compute window-motion score.

    Resolution order for the ROI:

      1. ``camera_angle`` (1=LP, 2=ALP) if provided — direct lookup from
         the /analyze endpoint, bypasses OCR entirely. ``roi_overrides``
         can swap out the default per-angle ROI without code edits.
      2. EasyOCR on the in-frame text overlay (legacy path). Used only
         when ``camera_angle`` is missing — keeps backwards compat with
         old code paths and old recordings whose overlay still matches
         the historical regex.

    Returns the same dict as :func:`compute_window_motion_score`, or
    ``None`` when classification cannot be made (no camera / no frames).
    Callers should treat ``None`` as "no motion override" and continue
    with the existing VLM pipeline.
    """
    if not keyframes:
        logger.info("[motion_classifier] called with empty keyframes; returning None")
        return None

    logger.info(
        "[motion_classifier] enter video=%s n_frames=%d camera_angle=%s",
        os.path.basename(video_filename or ""), len(keyframes), camera_angle,
    )

    # Path 1: cameraAngle from /analyze. No OCR.
    if camera_angle in (1, 2):
        roi = None
        if roi_overrides and camera_angle in roi_overrides:
            roi = roi_overrides[camera_angle]
        else:
            roi = _ROI_BY_CAMERA_ANGLE.get(camera_angle)
        if roi is None:
            return None
        cam_id_label = "LP_CAM2" if camera_angle == 1 else "ALP_CAM3"
        return compute_window_motion_score(
            keyframes, cam_id_label, roi_override=roi,
        )

    # Path 2: OCR fallback (legacy callers without camera_angle).
    cam_id = detect_camera_for_video(video_filename, keyframes)
    if cam_id is None:
        return None
    return compute_window_motion_score(keyframes, cam_id)
