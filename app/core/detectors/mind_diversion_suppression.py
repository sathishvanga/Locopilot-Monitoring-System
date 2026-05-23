"""Mind-diversion suppression rules.

Lifted byte-identically from ``LocopilotActivityMonitor.should_suppress_mind_diversion``
(``locopilot_monitor.py`` lines 3875-3937 in the pre-refactor monolith) so the
mind-diversion detector logic can live outside the 5400-line class.

The function is pure: it reads ``recent_person_activities`` but never mutates it.
Operators grep production logs for the exact reason strings returned here, so
the string values must remain byte-identical to the source.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def should_suppress_mind_diversion(
    *,
    person_idx: int,
    person_activities: Dict[str, Any],
    pose_landmarks: Any,
    detections: Dict[str, List[Any]],
    frame_shape: Tuple[int, ...],
    current_time: Optional[float],
    settings: Any,
    recent_person_activities: Dict[int, Dict[str, Any]],
    get_keypoint,
) -> Tuple[bool, Optional[str]]:
    """Suppress mind diversion if person is doing legitimate work activity.

    This function checks multiple conditions to prevent false positives when the LP
    is legitimately working on documents (logbook, papers, etc.).

    Args:
        person_idx: Index of the person being checked
        person_activities: Dict of detected activities for this person
        pose_landmarks: Pose landmarks for the person
        detections: YOLO detections dict (may contain 'book', etc.)
        frame_shape: (height, width) of the frame
        current_time: Current timestamp (optional, for recent activity check)
        settings: Settings object exposing
            ``mind_diversion_suppress_with_writing``,
            ``mind_diversion_writing_grace_seconds``,
            ``mind_diversion_wrist_distance_threshold``.
        recent_person_activities: Per-person recent-activity timestamps map
            (read-only here; the monitor owns the dict).
        get_keypoint: Callable equivalent to ``self.get_keypoint`` —
            ``get_keypoint(pose_landmarks, name)`` returns a keypoint with
            ``.x``, ``.y``, ``.visibility`` attributes.

    Returns:
        tuple: (should_suppress: bool, reason: str or None)
    """
    h, w = frame_shape[:2]

    # 1. WRITING ACTIVITY SUPPRESSION
    if settings.mind_diversion_suppress_with_writing:
        if person_activities.get('writing', False):
            return True, "suppressed_writing_active"

        # Check recent writing (within grace period)
        if current_time is not None and recent_person_activities is not None:
            writing_timestamp = recent_person_activities.get(person_idx, {}).get('writing')
            if writing_timestamp and (current_time - writing_timestamp) < settings.mind_diversion_writing_grace_seconds:
                return True, "suppressed_recent_writing"

    # 2. BOOK DETECTION SUPPRESSION
    if detections and 'book' in detections and len(detections.get('book', [])) > 0:
        return True, "suppressed_book_detected"

    # 3. HAND POSITION HEURISTIC (Critical for camera angle)
    # If both wrists visible and close together in lap area → likely document work
    if pose_landmarks:
        try:
            left_wrist = get_keypoint(pose_landmarks, 'left_wrist')
            right_wrist = get_keypoint(pose_landmarks, 'right_wrist')
            nose = get_keypoint(pose_landmarks, 'nose')

            if left_wrist.visibility > 0.3 and right_wrist.visibility > 0.3:
                # Calculate wrist positions
                left_wrist_coords = np.array([left_wrist.x * w, left_wrist.y * h])
                right_wrist_coords = np.array([right_wrist.x * w, right_wrist.y * h])
                wrist_distance = np.linalg.norm(left_wrist_coords - right_wrist_coords)

                # Check if wrists are in "lap area" (below nose, in front of body)
                nose_y = nose.y * h
                avg_wrist_y = (left_wrist_coords[1] + right_wrist_coords[1]) / 2
                wrists_below_face = avg_wrist_y > nose_y

                # If wrists close together AND below face → writing pose
                if wrist_distance < settings.mind_diversion_wrist_distance_threshold and wrists_below_face:
                    return True, "suppressed_writing_pose_detected"
        except (AttributeError, IndexError):
            pass  # Landmarks not available, continue without suppression

    # 4. NO SUPPRESSION - Allow detection
    return False, None
