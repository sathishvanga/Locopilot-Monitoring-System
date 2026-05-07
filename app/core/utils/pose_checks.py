"""Small pure pose / object proximity helpers.

Lifted from ``locopilot_monitor.py`` (T1 of the locopilot-refactor plan).

This module intentionally has no class state - the two helpers are pure
functions extracted verbatim so behavior remains byte-identical with the
monolith implementation. The only delta from the original methods is that
``self.get_keypoint`` and ``self.logger`` become explicit arguments.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple


def check_hands_below_shoulders(
    pose_landmarks: Any,
    get_keypoint: Callable[[Any, str], Any],
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Check if both hands are below shoulder level.

    A more relaxed check than "hands in lap" for various camera angles.

    Args:
        pose_landmarks: YoloPoseLandmarks or MediaPipe pose landmarks
        get_keypoint: Callable that resolves a named keypoint from
            ``pose_landmarks`` (matches the monitor's
            ``self.get_keypoint`` / ``self._get_keypoint_by_name``).
        logger: Optional logger; defaults to this module's logger.

    Returns:
        bool: True if both hands are below shoulders, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        left_shoulder = get_keypoint(pose_landmarks, 'left_shoulder')
        right_shoulder = get_keypoint(pose_landmarks, 'right_shoulder')
        left_wrist = get_keypoint(pose_landmarks, 'left_wrist')
        right_wrist = get_keypoint(pose_landmarks, 'right_wrist')

        if any(p is None for p in [left_shoulder, right_shoulder, left_wrist, right_wrist]):
            return False

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        return left_wrist.y > shoulder_y and right_wrist.y > shoulder_y

    except Exception as e:
        logger.debug(f"Exception in check_hands_below_shoulders: {e}")
        return False


def check_hand_object_interaction(
    hand_coords: Optional[Tuple[float, float]],
    object_bbox: Optional[List[int]],
    margin: int = 50,
) -> bool:
    """Check if hand is interacting with an object

    Args:
        hand_coords: (x, y) coordinates of hand
        object_bbox: [x1, y1, x2, y2] bounding box of object
        margin: proximity margin in pixels (default 50, use 30 for tighter checks)
    """
    if hand_coords is None or object_bbox is None:
        return False

    hx, hy = hand_coords
    x1, y1, x2, y2 = object_bbox
    return (x1 - margin <= hx <= x2 + margin and
            y1 - margin <= hy <= y2 + margin)
