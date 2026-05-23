"""Speedometer-region frame-diff motion classifier.

Sibling to ``motion_classifier.py``. That module uses the cabin window to
detect train motion; this one uses the analog speedometer dial. Both are
needed because the existing gates fail in different ways:

  * Pipeline-1 vibration is fooled by diesel idle (engine vibrates while
    parked at a station).
  * The window-region classifier requires a recognisable camera-overlay
    pattern (CCTV text like ``CAB 1 LP camera 2``). When the overlay
    format differs (``IPCamera 03``, etc.) the camera lookup fails and
    the gate stays silent.
  * The VLM cannot reliably read the speedometer needle on these
    foreshortened overhead crops (small/quantised model, oblique angle).

Frame-diff in the speedometer ROI sidesteps all three. When the train is
stationary, the analog needle does not move and the per-pixel diff
across consecutive keyframes inside the dial ROI is small. When the
train is running, the needle moves between keyframes and the diff rises.

The per-camera ROI is selected by ``camera_angle`` (1=LP, 2=ALP) which
the ``/analyze`` endpoint already plumbs through ``video_controller`` →
``video_processing_service`` and (after this change) the VLM verifier.

Scope
-----
This is an additional signal, not a replacement for the existing
window-motion classifier. Both gates run independently; if either fires
STOPPED the activity is dropped. Fail-open: any missing input
(insufficient keyframes, unreadable frame, unknown camera) returns
``None`` and the caller continues with the normal VLM pipeline.

Threshold provenance
--------------------
Tuned on run_20260511_191710 (LP-side, 28-minute trip, 42 keyframes
grouped into activities of 5 frames each). The clearest stationary
periods showed median pairwise diff <= 3.5; running periods exceeded 8.
The default threshold (5.0) sits in the middle of that gap. ALP defaults
are inherited from the same threshold; tune per camera once a labelled
ALP corpus exists.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Default per-camera ROIs in original 1280x720 frame coords.
# Tuned visually 2026-05-12 against:
#   LP  (camera_angle=1): run_20260511_191710 frame 19850 daylight view.
#                         Tight box around the dashboard gauge cluster's
#                         largest visible dial.
#   ALP (camera_angle=2): ch02_20260325112031 daylight wide_001. Gauge
#                         cluster sits upper-centre (different mount).
DEFAULT_ROIS: Dict[int, Tuple[int, int, int, int]] = {
    1: (960, 50, 1080, 180),   # LP : tight box on the speedometer dial
    2: (430, 0, 870, 160),     # ALP: cluster band (tighten once labelled)
}

# Default decision threshold. See module docstring "Threshold provenance".
DEFAULT_STOPPED_THRESHOLD: float = 5.0

# Minimum keyframes required for a meaningful diff. Two would technically
# work but a single noisy pair can swing the verdict; three pairs (=4
# frames) gives the median some headroom. Caller can override.
DEFAULT_MIN_FRAMES: int = 3


def _parse_roi(text: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse 'x1,y1,x2,y2' into a tuple. Returns None on malformed input."""
    try:
        parts = [int(v.strip()) for v in text.split(",")]
        if len(parts) != 4:
            return None
        return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        return None


def _roi_for_camera(
    camera_angle: int,
    overrides: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Resolve the active ROI for a camera, with optional override map."""
    if overrides and camera_angle in overrides:
        return overrides[camera_angle]
    return DEFAULT_ROIS.get(camera_angle)


def _diff_in_roi(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> float:
    """Mean abs pixel diff inside ``roi`` between two grayscale frames."""
    x1, y1, x2, y2 = roi
    h, w = prev_gray.shape
    x2, y2 = min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = prev_gray[y1:y2, x1:x2]
    b = curr_gray[y1:y2, x1:x2]
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(a, b)))


def compute_speedometer_motion_score(
    jpg_paths: List[Path],
    camera_angle: int,
    threshold: float = DEFAULT_STOPPED_THRESHOLD,
    min_frames: int = DEFAULT_MIN_FRAMES,
    roi_overrides: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> Optional[Dict[str, float]]:
    """Compute median pairwise pixel diff inside the speedometer ROI.

    Returns a dict ``{score, threshold, stopped, n_frames, n_pairs,
    latency_sec, camera_angle, roi}`` or ``None`` if the camera is
    unknown or there are not enough usable frames. Fail-open semantics:
    callers should treat ``None`` as "no override" and continue.
    """
    roi = _roi_for_camera(camera_angle, roi_overrides)
    if roi is None:
        logger.debug(
            "[speedometer] no ROI configured for camera_angle=%s; "
            "skipping classification",
            camera_angle,
        )
        return None
    if len(jpg_paths) < min_frames:
        return None

    t0 = time.time()
    grays: List[np.ndarray] = []
    for p in jpg_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if len(grays) < min_frames:
        return None

    diffs: List[float] = []
    for i in range(len(grays) - 1):
        if grays[i].shape != grays[i + 1].shape:
            continue
        diffs.append(_diff_in_roi(grays[i], grays[i + 1], roi))
    if not diffs:
        return None
    score = float(np.median(diffs))
    return {
        "score": score,
        "threshold": float(threshold),
        "stopped": score < threshold,
        "n_frames": len(grays),
        "n_pairs": len(diffs),
        "latency_sec": time.time() - t0,
        "camera_angle": int(camera_angle),
        "roi": list(roi),
    }


def classify_motion(
    keyframes: List[Path],
    camera_angle: Optional[int],
    threshold: float = DEFAULT_STOPPED_THRESHOLD,
    roi_overrides: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> Optional[Dict[str, float]]:
    """Top-level entry point matching the shape of ``motion_classifier.classify_motion``.

    ``camera_angle`` should be the 1/2 LP/ALP value already plumbed from
    the analyze endpoint. ``None`` (camera unknown) returns ``None``
    immediately — no override.
    """
    if not keyframes or camera_angle not in (1, 2):
        return None
    return compute_speedometer_motion_score(
        keyframes,
        camera_angle=camera_angle,
        threshold=threshold,
        roi_overrides=roi_overrides,
    )


def parse_roi_overrides_from_settings(
    settings,
) -> Optional[Dict[int, Tuple[int, int, int, int]]]:
    """Read per-camera ROI overrides from a pydantic Settings object.

    Reads ``speedometer_roi_lp`` and ``speedometer_roi_alp`` (both
    optional strings ``'x1,y1,x2,y2'``). Returns ``None`` if neither
    override is set (caller will fall back to ``DEFAULT_ROIS``).
    """
    overrides: Dict[int, Tuple[int, int, int, int]] = {}
    lp_raw = getattr(settings, "speedometer_roi_lp", "") or ""
    alp_raw = getattr(settings, "speedometer_roi_alp", "") or ""
    lp = _parse_roi(lp_raw) if lp_raw else None
    alp = _parse_roi(alp_raw) if alp_raw else None
    if lp:
        overrides[1] = lp
    if alp:
        overrides[2] = alp
    return overrides or None
