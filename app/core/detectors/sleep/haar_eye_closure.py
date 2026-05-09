"""Haar cascade eye closure detection.

Verbatim move of ``detect_eye_closure_haar`` and ``_load_haar_cascades``
from the original ``sleep_detector.py`` with ``self`` rebound to
``detector``.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _load_haar_cascades(detector: Any, eye_cascade_path: Optional[str]) -> None:
    """Load Haar cascade classifiers for face and eye detection."""
    try:
        detector.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        detector.profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        if eye_cascade_path:
            detector.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        else:
            detector.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        detector.logger.debug("Haar cascades loaded successfully")
    except Exception as e:
        detector.logger.warning(f"Failed to load Haar cascades: {e}")
        detector.face_cascade = None
        detector.profile_face_cascade = None
        detector.eye_cascade = None


def detect_eye_closure_haar(
    detector: Any,
    frame: Any,
    landmarks: Any,
    person_idx: int,
    bbox: List[int],
    timestamp_sec: float
) -> Dict[str, Any]:
    """Detect closed eyes using Haar cascade within pose-estimated face ROI.

    Strategy:
    1. Use YOLO pose keypoints (nose, eyes, ears) to estimate face bounding box
    2. Crop -> grayscale -> histogram equalize (for low-light)
    3. Try Haar frontal face, then profile face as fallback
    4. Within detected face, run eye cascade
    5. Face found but NO eyes -> eyes closed

    Args:
        frame: BGR image (numpy array)
        landmarks: Pose landmarks with face keypoints
        person_idx: Person index for tracking
        bbox: Person bounding box [x1, y1, x2, y2]
        timestamp_sec: Current timestamp in seconds

    Returns:
        dict with keys: eyes_closed, face_detected, num_eyes, is_microsleep,
                       is_sleep, duration, consecutive_closed
    """
    result = {
        'eyes_closed': False,
        'face_detected': False,
        'num_eyes': 0,
        'is_microsleep': False,
        'is_sleep': False,
        'duration': 0.0,
        'consecutive_closed': 0,
    }

    if frame is None or landmarks is None or detector.eye_cascade is None or detector.settings is None:
        return result

    if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
        return result

    tracking = detector._get_per_person_sleep_tracking(person_idx)
    h, w = frame.shape[:2]
    settings = detector.settings

    # --- Build face ROI from YOLO pose keypoints ---
    face_kp_indices = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
    face_points_x = []
    face_points_y = []

    for idx in face_kp_indices:
        if idx < len(landmarks.landmark):
            lm = landmarks.landmark[idx]
            if lm.visibility > 0.1:
                px = int(lm.x * w)
                py = int(lm.y * h)
                face_points_x.append(px)
                face_points_y.append(py)

    if len(face_points_x) < 2:
        return result

    # Compute bounding box with padding
    cx = int(np.mean(face_points_x))
    cy = int(np.mean(face_points_y))
    spread_x = max(max(face_points_x) - min(face_points_x), 30)
    spread_y = max(max(face_points_y) - min(face_points_y), 30)
    padding = getattr(settings, 'haar_eye_roi_padding', 0.5)

    roi_x1 = max(0, int(cx - spread_x * (0.5 + padding)))
    roi_y1 = max(0, int(cy - spread_y * (0.5 + padding)))
    roi_x2 = min(w, int(cx + spread_x * (0.5 + padding)))
    roi_y2 = min(h, int(cy + spread_y * (0.5 + padding)))

    # Clip to person bbox if available
    if bbox is not None and len(bbox) >= 4:
        bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        roi_x1 = max(roi_x1, bx1)
        roi_y1 = max(roi_y1, by1)
        roi_x2 = min(roi_x2, bx2)
        roi_y2 = min(roi_y2, by2)

    if roi_x2 - roi_x1 < 20 or roi_y2 - roi_y1 < 20:
        return result

    # Crop face ROI and convert to grayscale
    face_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if face_roi.size == 0:
        return result

    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
    gray_roi = cv2.equalizeHist(gray_roi)

    # --- Detect face within ROI ---
    scale_factor = getattr(settings, 'haar_eye_scale_factor', 1.1)
    min_neighbors = getattr(settings, 'haar_eye_min_neighbors', 3)

    faces = detector.face_cascade.detectMultiScale(
        gray_roi, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Fallback to profile face
    if len(faces) == 0:
        faces = detector.profile_face_cascade.detectMultiScale(
            gray_roi, scaleFactor=scale_factor, minNeighbors=max(1, min_neighbors - 1),
            minSize=(30, 30)
        )
        if len(faces) == 0:
            flipped_roi = cv2.flip(gray_roi, 1)
            faces = detector.profile_face_cascade.detectMultiScale(
                flipped_roi, scaleFactor=scale_factor, minNeighbors=max(1, min_neighbors - 1),
                minSize=(30, 30)
            )

    if len(faces) == 0:
        tracking['haar_face_detected_in_roi'] = False
        return result

    result['face_detected'] = True
    tracking['haar_face_detected_in_roi'] = True

    # Use the largest detected face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = largest_face

    # Extract upper half of face (eye region)
    eye_region_y1 = fy + int(fh * 0.15)
    eye_region_y2 = fy + int(fh * 0.55)
    eye_roi = gray_roi[eye_region_y1:eye_region_y2, fx:fx + fw]

    if eye_roi.size == 0:
        return result

    # --- Detect eyes within face region ---
    eyes = detector.eye_cascade.detectMultiScale(
        eye_roi, scaleFactor=1.1, minNeighbors=2,
        minSize=(15, 15)
    )
    num_eyes = len(eyes)
    result['num_eyes'] = num_eyes

    # --- Temporal tracking ---
    consecutive_threshold = getattr(settings, 'haar_eye_closed_consecutive_frames', 3)

    if num_eyes >= 1:
        tracking['haar_eyes_open_count'] += 1
        tracking['haar_eyes_closed_count'] = 0
        tracking['haar_eyes_closed_start'] = None
    else:
        tracking['haar_eyes_closed_count'] += 1
        tracking['haar_eyes_open_count'] = 0
        if tracking['haar_eyes_closed_start'] is None:
            tracking['haar_eyes_closed_start'] = timestamp_sec

    result['consecutive_closed'] = tracking['haar_eyes_closed_count']

    if tracking['haar_eyes_closed_count'] >= consecutive_threshold:
        result['eyes_closed'] = True

        if tracking['haar_eyes_closed_start'] is not None:
            duration = timestamp_sec - tracking['haar_eyes_closed_start']
            result['duration'] = duration

            sleep_duration = getattr(settings, 'haar_eye_sleep_duration', 10.0)
            microsleep_duration = getattr(settings, 'haar_eye_microsleep_duration', 3.0)

            if duration >= sleep_duration:
                result['is_sleep'] = True
            elif duration >= microsleep_duration:
                result['is_microsleep'] = True

    return result
