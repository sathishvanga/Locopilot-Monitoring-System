"""Pose utility functions for shared pose-landmark calculations.

This module provides reusable pose-related utility functions that are shared
across multiple detector modules (e.g., SleepDetector, ActivityDetector,
GestureDetector, MindDiversionDetector) to avoid code duplication.

Canonical ``get_keypoint`` lives here so every detector resolves keypoint
names through one implementation that includes MediaPipe fallback mappings.
"""
from typing import Any, Callable, Optional, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)

# YOLO Keypoint indices (COCO format) for direct access
YOLO_KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# MediaPipe-specific keypoints that don't exist in YOLO, mapped to fallbacks
_FALLBACK_MAP = {
    'left_index': 'left_wrist',
    'right_index': 'right_wrist',
    'left_pinky': 'left_wrist',
    'right_pinky': 'right_wrist',
    'left_thumb': 'left_wrist',
    'right_thumb': 'right_wrist',
}


def get_keypoint(landmarks: Any, keypoint_name: str) -> Any:
    """Get a keypoint from landmarks by name.

    Supports both YoloPoseLandmarks (has ``.landmark`` attribute) and plain
    landmark lists.  Handles MediaPipe-specific keypoint names (e.g.
    ``left_index``) by falling back to the corresponding wrist keypoint.

    Args:
        landmarks: Landmark list (either object with ``.landmark`` attribute
                   or a direct list of landmark objects).
        keypoint_name: String name like ``'nose'``, ``'left_wrist'``, etc.

    Returns:
        Landmark object with ``x``, ``y``, ``z``, ``visibility`` attributes.

    Raises:
        ValueError: If *keypoint_name* is not a recognised keypoint.
    """
    # Support both YoloPoseLandmarks (has .landmark) and plain list
    landmark_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

    name_lower = keypoint_name.lower()

    if name_lower in YOLO_KEYPOINT_INDICES:
        idx = YOLO_KEYPOINT_INDICES[name_lower]
        return landmark_list[idx]

    # Handle MediaPipe-specific keypoints that don't exist in YOLO
    if name_lower in _FALLBACK_MAP:
        fallback_name = _FALLBACK_MAP[name_lower]
        idx = YOLO_KEYPOINT_INDICES[fallback_name]
        return landmark_list[idx]

    raise ValueError(f"Unknown keypoint: {keypoint_name}")


def calculate_wrist_distance(
    pose_landmarks: Any,
    frame_shape: Tuple[int, ...],
    get_keypoint_func: Callable[[Any, str], Any],
    wrist_visibility_threshold: float = 0.3,
    elbow_visibility_threshold: float = 0.25,
    single_wrist_visibility: float = 0.5,
    shoulder_visibility: float = 0.3,
) -> Tuple[Optional[float], Optional[str]]:
    """Calculate Euclidean distance between left and right wrists.

    Falls back to elbow distance or single wrist-to-shoulder midpoint
    if wrists are not both visible.

    Args:
        pose_landmarks: Pose landmarks object (YoloPoseLandmarks or similar).
            Must support attribute access for .landmark or be a direct list.
        frame_shape: Tuple of (height, width, ...) of the frame.
        get_keypoint_func: Callable that retrieves a keypoint from landmarks
            by name. Signature: (landmarks, keypoint_name: str) -> landmark.
        wrist_visibility_threshold: Minimum visibility for both wrists to
            use the wrist-based distance (default 0.3).
        elbow_visibility_threshold: Minimum visibility for both elbows to
            use the elbow-based fallback (default 0.25).
        single_wrist_visibility: Minimum visibility for a single wrist in
            the single-wrist-to-shoulder fallback (default 0.5).
        shoulder_visibility: Minimum visibility for shoulders in the
            single-wrist-to-shoulder fallback (default 0.3).

    Returns:
        tuple: (distance_in_pixels, source) where source is one of:
            - 'wrist': both wrists visible, distance between them
            - 'elbow': both elbows visible, distance between them
            - 'single_wrist': one wrist visible, distance to shoulder midpoint
            - (None, None): not enough keypoints visible
    """
    if not pose_landmarks:
        return None, None

    try:
        landmarks = (
            pose_landmarks.landmark
            if hasattr(pose_landmarks, 'landmark')
            else pose_landmarks
        )
        h, w = frame_shape[:2]

        right_wrist = get_keypoint_func(landmarks, 'right_wrist')
        left_wrist = get_keypoint_func(landmarks, 'left_wrist')

        # Try wrists first (primary method)
        if (right_wrist.visibility >= wrist_visibility_threshold and
                left_wrist.visibility >= wrist_visibility_threshold):
            right_wrist_px = (right_wrist.x * w, right_wrist.y * h)
            left_wrist_px = (left_wrist.x * w, left_wrist.y * h)

            distance = np.sqrt(
                (right_wrist_px[0] - left_wrist_px[0])**2 +
                (right_wrist_px[1] - left_wrist_px[1])**2
            )
            return distance, 'wrist'

        # Fallback: elbows
        right_elbow = get_keypoint_func(landmarks, 'right_elbow')
        left_elbow = get_keypoint_func(landmarks, 'left_elbow')

        if (right_elbow.visibility >= elbow_visibility_threshold and
                left_elbow.visibility >= elbow_visibility_threshold):
            right_elbow_px = (right_elbow.x * w, right_elbow.y * h)
            left_elbow_px = (left_elbow.x * w, left_elbow.y * h)
            distance = np.sqrt(
                (right_elbow_px[0] - left_elbow_px[0])**2 +
                (right_elbow_px[1] - left_elbow_px[1])**2
            )
            return distance, 'elbow'

        # Fallback: single wrist to shoulder midpoint
        right_shoulder = get_keypoint_func(landmarks, 'right_shoulder')
        left_shoulder = get_keypoint_func(landmarks, 'left_shoulder')

        visible_wrist = None
        if (right_wrist.visibility >= single_wrist_visibility and
                left_wrist.visibility < single_wrist_visibility):
            visible_wrist = right_wrist
        elif (left_wrist.visibility >= single_wrist_visibility and
                right_wrist.visibility < single_wrist_visibility):
            visible_wrist = left_wrist

        if (visible_wrist is not None and
                right_shoulder.visibility >= shoulder_visibility and
                left_shoulder.visibility >= shoulder_visibility):
            wrist_px = (visible_wrist.x * w, visible_wrist.y * h)
            shoulder_mid_px = (
                (right_shoulder.x + left_shoulder.x) / 2 * w,
                (right_shoulder.y + left_shoulder.y) / 2 * h
            )
            distance = np.sqrt(
                (wrist_px[0] - shoulder_mid_px[0])**2 +
                (wrist_px[1] - shoulder_mid_px[1])**2
            )
            return distance, 'single_wrist'

        return None, None
    except Exception as e:
        logger.debug(f"Exception in calculate_wrist_distance: {e}")
        return None, None
