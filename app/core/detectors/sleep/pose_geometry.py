"""Pose geometry helpers for SleepDetector.

Pure module-level functions that take a ``SleepDetector`` instance as
their first argument. Each function is a verbatim move of the
corresponding method body from the original ``sleep_detector.py``,
with ``self`` rebound to ``detector``. No logic changes.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.utils.pose_utils import (
    calculate_wrist_distance as _calculate_wrist_distance,
    get_keypoint as _get_keypoint,
)


def get_keypoint(detector: Any, landmarks: Any, keypoint_name: str) -> Any:
    """Get a keypoint from landmarks by name.

    Delegates to the canonical ``get_keypoint`` in
    ``app.core.utils.pose_utils``.

    Args:
        landmarks: Landmark list (either .landmark attribute or direct list)
        keypoint_name: String name like 'nose', 'left_wrist', etc.

    Returns:
        Landmark object with x, y, z, visibility attributes
    """
    return _get_keypoint(landmarks, keypoint_name)


def validate_pose_landmarks(
    detector: Any,
    pose_landmarks: Any,
    min_landmarks: Optional[int] = None,
    min_visibility: Optional[float] = None
) -> bool:
    """Validate that pose landmarks are valid and usable for detection.

    Args:
        pose_landmarks: Pose landmarks object
        min_landmarks: Minimum number of landmarks required
        min_visibility: Minimum average visibility score

    Returns:
        bool: True if landmarks are valid, False otherwise
    """
    if min_landmarks is None:
        min_landmarks = detector.MIN_POSE_LANDMARKS
    if min_visibility is None:
        min_visibility = detector.MIN_POSE_VISIBILITY
    if pose_landmarks is None:
        return False

    # Support both YoloPoseLandmarks (has .landmark) and plain list
    landmark_list = (pose_landmarks.landmark if hasattr(pose_landmarks, 'landmark')
                    else pose_landmarks if isinstance(pose_landmarks, list) else None)
    if landmark_list is None or len(landmark_list) < min_landmarks:
        return False

    # Validate coordinates are within valid range (0-1 for normalized)
    valid_count = 0
    total_visibility = 0.0

    for landmark in landmark_list:
        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            valid_count += 1
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            total_visibility += visibility

    if valid_count < min_landmarks:
        return False

    avg_visibility = total_visibility / valid_count if valid_count > 0 else 0.0
    if avg_visibility < min_visibility:
        return False

    return True


def calculate_head_tilt_angle(detector: Any, landmarks: Any) -> Optional[float]:
    """Calculate head tilt angle from pose landmarks.

    The angle is calculated from the nose position relative to the neck
    (midpoint of shoulders). A negative angle indicates head tilted
    forward/down (sleeping position).

    Args:
        landmarks: Pose landmarks with nose and shoulder keypoints

    Returns:
        float: Head tilt angle in degrees (0 = upright, negative = forward/down)
               None if calculation fails
    """
    try:
        nose = detector.get_keypoint(landmarks, 'nose')
        left_shoulder = detector.get_keypoint(landmarks, 'left_shoulder')
        right_shoulder = detector.get_keypoint(landmarks, 'right_shoulder')

        # Calculate neck position (midpoint between shoulders)
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2

        # Calculate angle from vertical
        delta_y = nose.y - neck_y
        delta_x = nose.x - neck_x

        # Negative angle = head tilted forward/down
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi - 90
        # Normalize to [-180, 180] so values near +/-180 don't produce
        # spurious 300+ deg jumps when downstream code computes deltas
        # against a baseline that wrapped the other way.
        # See CLAUDE.md "Head tilt angle wrapping" FP pattern.
        angle = (angle + 180) % 360 - 180

        return angle

    except Exception as e:
        detector.logger.debug(f"Exception in calculate_head_tilt_angle: {e}")
        return None


def calculate_movement_score(
    detector: Any,
    current_landmarks: Any,
    previous_landmarks: Any
) -> float:
    """Calculate movement score between two sets of pose landmarks.

    Tracks movement of upper body keypoints (nose, shoulders, elbows, wrists)
    between consecutive frames.

    Args:
        current_landmarks: Current frame landmarks
        previous_landmarks: Previous frame landmarks

    Returns:
        float: Movement score (0 = no movement, higher = more movement)
    """
    if previous_landmarks is None:
        return 0.0

    try:
        key_landmark_names = [
            'nose', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]

        total_movement = 0.0
        for landmark_name in key_landmark_names:
            curr = detector.get_keypoint(current_landmarks, landmark_name)
            prev = detector.get_keypoint(previous_landmarks, landmark_name)

            distance = np.sqrt(
                (curr.x - prev.x) ** 2 +
                (curr.y - prev.y) ** 2
            )
            total_movement += distance

        movement_score = total_movement / len(key_landmark_names)
        return movement_score

    except Exception as e:
        detector.logger.debug(f"Exception in calculate_movement_score: {e}")
        return 0.0


def calculate_wrist_distance(
    detector: Any,
    pose_landmarks: Any,
    frame_shape: Tuple[int, ...]
) -> Tuple[Optional[float], Optional[str]]:
    """Calculate Euclidean distance between left and right wrists.

    Falls back to elbow distance or single wrist-to-shoulder if wrists
    are not both visible. Delegates to the shared utility in
    ``app.core.utils.pose_utils.calculate_wrist_distance``.

    Args:
        pose_landmarks: Pose landmarks object
        frame_shape: Tuple of (height, width) of the frame

    Returns:
        tuple: (distance in pixels, source) where source is 'wrist', 'elbow',
               'single_wrist', or (None, None) if not detectable
    """
    if not pose_landmarks:
        return None, None

    if not detector.validate_pose_landmarks(pose_landmarks):
        return None, None

    return _calculate_wrist_distance(
        pose_landmarks,
        frame_shape,
        get_keypoint_func=detector.get_keypoint,
        wrist_visibility_threshold=detector.WRIST_VISIBILITY_THRESHOLD,
        elbow_visibility_threshold=detector.ELBOW_VISIBILITY_THRESHOLD,
    )


def _calculate_body_movement(
    detector: Any,
    landmarks: Any,
    tracking: Dict[str, Any],
    body_indices: List[int]
) -> Optional[float]:
    """Calculate movement from body keypoints across frames.

    Used for IR forward-lean detection where only body keypoints
    (shoulders, elbows, hips) are visible.

    Args:
        landmarks: Pose landmarks object with .landmark attribute
        tracking: Per-person IR forward lean tracking dict
        body_indices: List of YOLO keypoint indices for body parts

    Returns:
        float: Mean movement in normalized coords (0-1 scale),
               or None if no previous frame
    """
    body_vis_threshold = getattr(detector.settings, 'ir_forward_lean_body_vis_threshold', 0.2) if detector.settings else 0.2
    current_kps = []
    for idx in body_indices:
        lm = landmarks.landmark[idx]
        if getattr(lm, 'visibility', 0) > body_vis_threshold:
            current_kps.append((lm.x, lm.y))
        else:
            current_kps.append(None)

    prev_kps = tracking.get('previous_body_keypoints')
    tracking['previous_body_keypoints'] = current_kps

    if prev_kps is None:
        return None

    movements = []
    for curr, prev in zip(current_kps, prev_kps):
        if curr is not None and prev is not None:
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            movements.append((dx ** 2 + dy ** 2) ** 0.5)

    if not movements:
        return None
    return sum(movements) / len(movements)
