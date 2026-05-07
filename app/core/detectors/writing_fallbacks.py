"""Writing-detection fallbacks lifted from ``locopilot_monitor.py``.

T1 of the locopilot-refactor plan. The two functions below were previously
methods on ``LocopilotActivityMonitor``. They are extracted verbatim with the
only allowed delta being:

- ``self.activity_detector`` becomes an injected ``activity_detector`` arg.
- ``self.wrist_proximity_tracking`` becomes the ``wrist_proximity_tracking``
  dict argument (mutated in place; the rewire passes the live dict so
  cleanup logic still sees the same object).
- ``self.MAX_WRIST_DISTANCE`` etc. come from ``WritingFallbackThresholds``.
- ``self.logger`` becomes the ``logger`` argument.

Log message strings are preserved byte-identical (operators grep production
logs for them).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from app.core.utils.geometry import bbox_overlap_with_margin


@dataclass
class WritingFallbackThresholds:
    """Numeric thresholds for the two writing fallback detectors.

    The monitor builds one of these once in ``__init__`` from its
    ``self.MAX_WRIST_DISTANCE`` / ``self.WRITING_REQUIRED_CONSECUTIVE`` /
    ``self.PERSON_BOOK_OVERLAP_MARGIN`` etc. constants.
    """

    max_wrist_distance: int
    max_single_wrist_distance: int
    max_elbow_distance: int
    writing_required_consecutive: int
    writing_min_duration: float
    person_book_overlap_margin: int
    book_posture_required_consecutive: int
    book_posture_min_duration: float


def detect_writing_by_wrist_proximity(
    *,
    pose_landmarks: Any,
    frame_shape: Tuple[int, ...],
    person_idx: int,
    timestamp_sec: float,
    activity_detector: Any,
    wrist_proximity_tracking: dict,
    thresholds: WritingFallbackThresholds,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Detect writing activity based on wrist/elbow proximity + head posture heuristic.

    When both wrists (or elbows as fallback) are close together AND head is tilted down
    (typical writing posture) for a sustained duration, it indicates writing activity.
    This method serves as a fallback when book detection doesn't trigger but person is clearly writing.

    Required Conditions (both must be true):
    1. Wrist/Elbow proximity: Left and right wrists within 300px (or elbows within 450px)
    2. Head posture: Head tilted down (nose below eye line)
    3. Temporal: Sustained for 1+ seconds across 2+ consecutive frames

    Args:
        pose_landmarks: MediaPipe pose landmarks (must include wrist/elbow + head keypoints)
        frame_shape: Tuple of (height, width) of the frame
        person_idx: Index of the person being analyzed
        timestamp_sec: Current timestamp in seconds
        activity_detector: Instance exposing ``calculate_wrist_distance`` and
            ``detect_head_looking_down`` (the existing
            ``app.core.detectors.activity_detector`` instance).
        wrist_proximity_tracking: Per-person tracking dict; mutated in place.
            The same dict object the monitor stores on
            ``self.wrist_proximity_tracking``.
        thresholds: Numeric thresholds bundle.
        logger: Optional logger.

    Returns:
        bool: True if writing detected by pose-based heuristic, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Calculate distance between wrists (with elbow fallback)
    distance_result = activity_detector.calculate_wrist_distance(pose_landmarks, frame_shape)

    # Handle tuple return (distance, source)
    if isinstance(distance_result, tuple):
        distance, source = distance_result
    else:
        # Backward compatibility if function returns single value
        distance = distance_result
        source = 'wrist' if distance is not None else None

    # DEBUG: Log distance calculation
    if distance is None:
        logger.debug(f"Person {person_idx}: Wrist/elbow distance = None (landmarks missing)")
        return False
    else:
        logger.debug(f"Person {person_idx}: {source.capitalize()} distance = {distance:.1f}px")

    # Initialize tracking for this person if needed
    if person_idx not in wrist_proximity_tracking:
        wrist_proximity_tracking[person_idx] = {
            'start_time': None,
            'duration': 0.0,
            'consecutive_frames': 0
        }

    # Configurable thresholds - different for wrist vs elbow

    # Select threshold based on detection source
    if source == 'wrist':
        max_distance = thresholds.max_wrist_distance
    elif source == 'single_wrist':
        max_distance = thresholds.max_single_wrist_distance
    else:
        max_distance = thresholds.max_elbow_distance

    person_tracking = wrist_proximity_tracking[person_idx]

    # Check if distance is within threshold
    if distance <= max_distance:
        # NEW: Check if head is looking down (required for writing posture)
        head_looking_down = activity_detector.detect_head_looking_down(pose_landmarks)

        # DEBUG: Log head state
        logger.debug(f"Person {person_idx}: Head looking down = {head_looking_down} (source={source})")

        if not head_looking_down:
            # Head not down - reset tracking (not a writing posture)
            if person_tracking['start_time'] is not None:
                # Was tracking, now stopped because head is up
                logger.debug(
                    f"Person {person_idx}: {source.capitalize()}s close ({distance:.1f}px) but head not down - "
                    f"resetting writing tracker"
                )
            person_tracking['start_time'] = None
            person_tracking['duration'] = 0.0
            person_tracking['consecutive_frames'] = 0
            return False

        # BOTH conditions met: distance close AND head down
        # Update tracking
        if person_tracking['start_time'] is None:
            # Start new proximity event
            person_tracking['start_time'] = timestamp_sec
            person_tracking['consecutive_frames'] = 1
            logger.debug(
                f"Person {person_idx}: Writing posture started ({source}) - dist={distance:.1f}px, head=down"
            )
        else:
            # Continue existing proximity event
            person_tracking['duration'] = timestamp_sec - person_tracking['start_time']
            person_tracking['consecutive_frames'] += 1
            logger.debug(
                f"Person {person_idx}: Writing posture continuing ({source}) - dist={distance:.1f}px, "
                f"head=down, frames={person_tracking['consecutive_frames']}, "
                f"duration={person_tracking['duration']:.1f}s"
            )

        # Check if thresholds are met
        if (person_tracking['consecutive_frames'] >= thresholds.writing_required_consecutive and
            person_tracking['duration'] >= thresholds.writing_min_duration):
            logger.info(
                f"Person {person_idx}: WRITING CONFIRMED via {source} - distance close + head down for "
                f"{person_tracking['duration']:.1f}s ({person_tracking['consecutive_frames']} frames)"
            )
            return True
    else:
        # Distance too far apart - reset tracking
        if person_tracking['start_time'] is not None:
            logger.debug(
                f"Person {person_idx}: Writing posture lost - {source}s too far ({distance:.1f}px) - "
                f"resetting tracker"
            )
        person_tracking['start_time'] = None
        person_tracking['duration'] = 0.0
        person_tracking['consecutive_frames'] = 0

    return False


def detect_writing_by_book_and_posture(
    *,
    pose_landmarks: Any,
    person_bbox: List[int],
    book_bboxes: List[List[int]],
    person_idx: int,
    timestamp_sec: float,
    activity_detector: Any,
    wrist_proximity_tracking: dict,
    thresholds: WritingFallbackThresholds,
    bbox_overlap_with_margin_fn: Callable = bbox_overlap_with_margin,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Fallback writing detection when wrists are not visible.

    Detects writing based on:
    1. Book detected in person's region
    2. Head looking down (reading/writing posture)
    3. Sustained for minimum duration

    This is a fallback when wrist/elbow detection fails.

    Args:
        pose_landmarks: Pose landmarks for this person
        person_bbox: [x1, y1, x2, y2] bounding box of person
        book_bboxes: List of book bounding boxes detected in frame
        person_idx: Index of person being analyzed
        timestamp_sec: Current timestamp in seconds
        activity_detector: Instance exposing ``detect_head_looking_down``.
        wrist_proximity_tracking: Per-person tracking dict; mutated in place.
            Reuses ``tracking_key=f"book_posture_{person_idx}"``.
        thresholds: Numeric thresholds bundle.
        bbox_overlap_with_margin_fn: Overlap predicate; defaults to
            ``app.core.utils.geometry.bbox_overlap_with_margin``.
        logger: Optional logger.

    Returns:
        bool: True if writing detected via book+posture, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not book_bboxes or len(book_bboxes) == 0:
        return False

    # Initialize tracking if needed
    tracking_key = f"book_posture_{person_idx}"
    if tracking_key not in wrist_proximity_tracking:
        wrist_proximity_tracking[tracking_key] = {
            'start_time': None,
            'duration': 0.0,
            'consecutive_frames': 0
        }

    person_tracking = wrist_proximity_tracking[tracking_key]

    # Check if any book is in person's region
    person_book_margin = thresholds.person_book_overlap_margin  # Same margin used elsewhere
    book_in_region = False
    for book_bbox in book_bboxes:
        if bbox_overlap_with_margin_fn(book_bbox, person_bbox, person_book_margin):
            book_in_region = True
            break

    if not book_in_region:
        person_tracking['start_time'] = None
        person_tracking['duration'] = 0.0
        person_tracking['consecutive_frames'] = 0
        return False

    # Check head posture (must be looking down toward book)
    head_looking_down = activity_detector.detect_head_looking_down(pose_landmarks)

    if not head_looking_down:
        if person_tracking['start_time'] is not None:
            logger.debug(
                f"Person {person_idx}: Book in region but head not down - resetting book+posture tracker"
            )
        person_tracking['start_time'] = None
        person_tracking['duration'] = 0.0
        person_tracking['consecutive_frames'] = 0
        return False

    # Both conditions met: book in region + head down

    if person_tracking['start_time'] is None:
        person_tracking['start_time'] = timestamp_sec
        person_tracking['consecutive_frames'] = 1
        logger.debug(
            f"Person {person_idx}: Book+posture writing started - book in region, head down"
        )
    else:
        person_tracking['duration'] = timestamp_sec - person_tracking['start_time']
        person_tracking['consecutive_frames'] += 1
        logger.debug(
            f"Person {person_idx}: Book+posture continuing - frames={person_tracking['consecutive_frames']}, "
            f"duration={person_tracking['duration']:.1f}s"
        )

    if (person_tracking['consecutive_frames'] >= thresholds.book_posture_required_consecutive and
        person_tracking['duration'] >= thresholds.book_posture_min_duration):
        logger.info(
            f"Person {person_idx}: WRITING CONFIRMED via book+posture fallback - "
            f"book in region + head down for {person_tracking['duration']:.1f}s"
        )
        return True

    return False
