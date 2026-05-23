"""Pure pose-landmark validators lifted from ``locopilot_monitor.py``.

These three helpers gate gesture detection (and other pose-driven activity
detectors) by rejecting landmark sets that are missing, anatomically
implausible, or jittering between frames. They are direct lifts of the
methods that previously lived on ``LocopilotActivityMonitor`` -- behavior
must remain byte-identical so log strings and numeric thresholds are
preserved verbatim.

Public API:

* ``validate_pose_landmarks`` - count + visibility check on a single frame.
* ``validate_anatomical_consistency`` - shoulder/arm/torso sanity rules.
* ``check_landmark_stability`` - 3-frame shoulder-jump smoothness gate;
  mutates a caller-owned ``history`` dict in place.
"""
from collections import deque
from typing import Any, Callable, Dict, Tuple


def validate_pose_landmarks(
    pose_landmarks: Any,
    *,
    min_landmarks: int = 10,
    min_visibility: float = 0.3,
) -> bool:
    """Validate that pose landmarks are valid and usable for activity detection.

    Args:
        pose_landmarks: MediaPipe pose landmarks
        min_landmarks: Minimum number of landmarks required (default: MIN_POSE_LANDMARKS)
        min_visibility: Minimum average visibility score (default: MIN_POSE_VISIBILITY)

    Returns:
        bool: True if landmarks are valid, False otherwise
    """
    if pose_landmarks is None:
        return False

    # Support both YoloPoseLandmarks (has .landmark) and plain list
    landmark_list = pose_landmarks.landmark if hasattr(pose_landmarks, 'landmark') else pose_landmarks if isinstance(pose_landmarks, list) else None
    if landmark_list is None or len(landmark_list) < min_landmarks:
        return False

    # Validate coordinates are within valid range (0-1 for normalized)
    valid_count = 0
    total_visibility = 0.0

    for landmark in landmark_list:
        # Check if coordinates are valid (normalized 0-1)
        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            valid_count += 1
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            total_visibility += visibility

    # Check we have enough valid landmarks
    if valid_count < min_landmarks:
        return False

    # Check average visibility
    avg_visibility = total_visibility / valid_count if valid_count > 0 else 0.0
    if avg_visibility < min_visibility:
        return False

    return True


def validate_anatomical_consistency(
    pose_landmarks: Any,
    frame_shape: Tuple[int, ...],
    *,
    get_keypoint: Callable[[Any, str], Any],
) -> Tuple[bool, str]:
    """
    Validate that pose landmarks follow anatomical rules.
    Rejects physically impossible configurations.

    Returns: (is_valid: bool, reason: str)
    """
    h, w = frame_shape[:2]

    try:
        # Get key landmarks
        right_shoulder = get_keypoint(pose_landmarks, 'right_shoulder')
        left_shoulder = get_keypoint(pose_landmarks, 'left_shoulder')
        right_elbow = get_keypoint(pose_landmarks, 'right_elbow')
        left_elbow = get_keypoint(pose_landmarks, 'left_elbow')
        right_wrist = get_keypoint(pose_landmarks, 'right_wrist')
        left_wrist = get_keypoint(pose_landmarks, 'left_wrist')
        right_hip = get_keypoint(pose_landmarks, 'right_hip')
        left_hip = get_keypoint(pose_landmarks, 'left_hip')
        nose = get_keypoint(pose_landmarks, 'nose')

        # Rule 1: Shoulders should be roughly horizontal (±30 degrees)
        shoulder_y_diff = abs(right_shoulder.y - left_shoulder.y) * h
        shoulder_x_diff = abs(right_shoulder.x - left_shoulder.x) * w
        if shoulder_x_diff > 0:
            shoulder_slope = shoulder_y_diff / shoulder_x_diff
            if shoulder_slope > 0.6:  # ~30 degrees
                return False, "Shoulders not horizontal (slope too steep)"

        # Rule 2: Shoulder-elbow-wrist distances must be reasonable
        # Forearm (elbow-wrist) typically 80-120% of upper arm (shoulder-elbow)
        def distance(lm1, lm2):
            dx = (lm1.x - lm2.x) * w
            dy = (lm1.y - lm2.y) * h
            return (dx**2 + dy**2)**0.5

        right_upper_arm = distance(right_shoulder, right_elbow)
        right_forearm = distance(right_elbow, right_wrist)
        left_upper_arm = distance(left_shoulder, left_elbow)
        left_forearm = distance(left_elbow, left_wrist)

        # Check proportions (forearm should be 50-150% of upper arm length)
        if right_upper_arm > 10:  # Avoid division by very small numbers
            right_ratio = right_forearm / right_upper_arm
            if right_ratio < 0.5 or right_ratio > 1.5:
                return False, f"Right arm proportions invalid ({right_ratio:.2f})"

        if left_upper_arm > 10:
            left_ratio = left_forearm / left_upper_arm
            if left_ratio < 0.5 or left_ratio > 1.5:
                return False, f"Left arm proportions invalid ({left_ratio:.2f})"

        # Rule 3: Nose should be above shoulders (not inverted person)
        avg_shoulder_y = (right_shoulder.y + left_shoulder.y) / 2
        if nose.y > avg_shoulder_y + 0.1:  # Nose more than 10% below shoulders
            return False, "Nose below shoulders (inverted detection)"

        # Rule 4: Hips should be below shoulders
        avg_hip_y = (right_hip.y + left_hip.y) / 2
        if avg_hip_y < avg_shoulder_y:
            return False, "Hips above shoulders (inverted detection)"

        # Rule 5: High visibility required for key landmarks
        if (right_shoulder.visibility < 0.5 or left_shoulder.visibility < 0.5 or
            nose.visibility < 0.5):
            return False, "Low visibility for critical landmarks"

        return True, "Valid"

    except (IndexError, AttributeError) as e:
        return False, f"Missing landmarks: {e}"


def check_landmark_stability(
    person_idx: int,
    pose_landmarks: Any,
    frame_shape: Tuple[int, ...],
    *,
    history: Dict[int, Dict[str, Any]],
    max_jump_threshold: float = 100.0,
    get_keypoint: Callable[[Any, str], Any],
) -> Tuple[bool, float]:
    """
    Check if landmarks are stable over time (not jumping erratically).
    Erratic jumps indicate poor detection quality.

    Returns: (is_stable: bool, max_jump: float)
    """
    h, w = frame_shape[:2]

    if person_idx not in history:
        history[person_idx] = {
            'right_shoulder': deque(maxlen=3),
            'left_shoulder': deque(maxlen=3)
        }

    person_history = history[person_idx]

    # Get current shoulder positions (most stable body parts)
    right_shoulder = get_keypoint(pose_landmarks, 'right_shoulder')
    left_shoulder = get_keypoint(pose_landmarks, 'left_shoulder')

    right_pos = (int(right_shoulder.x * w), int(right_shoulder.y * h))
    left_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))

    person_history['right_shoulder'].append(right_pos)
    person_history['left_shoulder'].append(left_pos)

    # Need at least 2 positions to check stability
    if len(person_history['right_shoulder']) < 2:
        return True, 0  # Not enough data, assume stable

    # Calculate maximum jump between consecutive frames
    max_jump = 0
    for key in ['right_shoulder', 'left_shoulder']:
        positions = list(person_history[key])
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            jump = (dx**2 + dy**2)**0.5
            max_jump = max(max_jump, jump)

    # Shoulders shouldn't jump more than 100px between frames (person sitting, camera static)
    is_stable = max_jump < max_jump_threshold

    return is_stable, max_jump
