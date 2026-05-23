"""IR forward-lean sleep detection fallback.

Verbatim move of ``detect_ir_forward_lean_sleep`` from the original
``sleep_detector.py`` with ``self`` rebound to ``detector``.
"""

from typing import Any, Dict, List, Tuple


def detect_ir_forward_lean_sleep(
    detector: Any,
    landmarks: Any,
    bbox: List[int],
    timestamp_sec: float,
    person_idx: int,
    frame_shape: Tuple[int, ...]
) -> Tuple[bool, bool, Dict[str, Any]]:
    """Detect forward-lean sleep posture using body-only keypoints in IR/dark frames.

    Uses shoulders, elbows, and hips (which remain visible in IR) plus the
    disappearance of head keypoints as a strong signal of forward lean.

    MANDATORY gate: Head keypoints must be invisible. If head is visible,
    person is not in a forward-lean sleep posture.

    Scoring signals (after head_invisible gate passes):
    - Head keypoints invisible (+2): All 5 head keypoints below visibility threshold
    - Shoulders high in bbox (+1): Shoulder midpoint in upper 40% of person bbox
    - Bbox aspect ratio squashed (+1): height/width < 1.2
    - Low body movement (+1): Body keypoints stable across frames
    - Elbows below shoulders (+1): Arms hanging/braced = forward lean posture

    Args:
        landmarks: Pose landmarks (may have low-visibility head keypoints)
        bbox: Person bounding box [x1, y1, x2, y2]
        timestamp_sec: Current timestamp in seconds
        person_idx: Person index for per-person tracking
        frame_shape: Frame shape tuple (height, width, channels)

    Returns:
        tuple: (is_sleeping, is_microsleeping, debug_info)
    """
    if landmarks is None or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
        return False, False, {}

    if len(landmarks.landmark) < detector.YOLO_MIN_KEYPOINTS:
        return False, False, {
            'ir_forward_lean': False,
            'reason': 'insufficient_landmarks',
            'landmark_count': len(landmarks.landmark)
        }

    # Safe settings access with defaults
    head_vis_thresh = getattr(detector.settings, 'ir_forward_lean_head_vis_threshold', 0.15) if detector.settings else 0.15
    body_vis_thresh = getattr(detector.settings, 'ir_forward_lean_body_vis_threshold', 0.2) if detector.settings else 0.2
    min_body_kps = getattr(detector.settings, 'ir_forward_lean_min_body_keypoints', 3) if detector.settings else 3
    score_threshold = getattr(detector.settings, 'ir_forward_lean_score_threshold', 4) if detector.settings else 4
    min_duration = getattr(detector.settings, 'ir_forward_lean_min_duration', 5.0) if detector.settings else 5.0
    sleep_duration = getattr(detector.settings, 'ir_forward_lean_sleep_duration', 10.0) if detector.settings else 10.0

    h, w = frame_shape[:2]

    head_indices = detector.YOLO_HEAD_INDICES
    body_indices = detector.YOLO_BODY_INDICES

    # Count visible body keypoints
    visible_body_count = 0
    for idx in body_indices:
        if getattr(landmarks.landmark[idx], 'visibility', 0) > body_vis_thresh:
            visible_body_count += 1

    if visible_body_count < min_body_kps:
        return False, False, {
            'ir_forward_lean': False,
            'reason': 'insufficient_body_keypoints',
            'visible_body': visible_body_count
        }

    # --- Scoring ---
    score = 0
    score_breakdown = {}

    # Signal 1: Head keypoints invisible (MANDATORY gate + 2 points)
    head_visible = 0
    for idx in head_indices:
        if getattr(landmarks.landmark[idx], 'visibility', 0) >= head_vis_thresh:
            head_visible += 1
    if head_visible == 0:
        score += 2
        score_breakdown['head_invisible'] = 2
    else:
        score_breakdown['head_invisible'] = 0
        tracking = detector._get_ir_forward_lean_tracking(person_idx)
        tracking['start_time'] = None
        tracking['sub_threshold_streak'] = 0
        return False, False, {
            'ir_forward_lean': False,
            'reason': 'head_visible',
            'head_visible': head_visible,
            'score_breakdown': score_breakdown,
        }

    # Signal 2: Shoulders high in bbox (+1)
    left_shoulder = landmarks.landmark[5]
    right_shoulder = landmarks.landmark[6]
    if (getattr(left_shoulder, 'visibility', 0) > body_vis_thresh and
            getattr(right_shoulder, 'visibility', 0) > body_vis_thresh):
        shoulder_mid_y_px = ((left_shoulder.y + right_shoulder.y) / 2) * h
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        shoulder_relative = (shoulder_mid_y_px - y1) / bbox_height if bbox_height > 0 else 0.5
        if shoulder_relative < detector.IR_SHOULDER_RELATIVE_THRESHOLD:
            score += 1
            score_breakdown['shoulders_high'] = 1
        else:
            score_breakdown['shoulders_high'] = 0
    else:
        score_breakdown['shoulders_high'] = 0

    # Signal 3: Bbox aspect ratio squashed (+1)
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 1.5
    if aspect_ratio < detector.IR_BBOX_ASPECT_RATIO_THRESHOLD:
        score += 1
        score_breakdown['squashed_bbox'] = 1
    else:
        score_breakdown['squashed_bbox'] = 0
    score_breakdown['aspect_ratio'] = round(aspect_ratio, 2)

    # Signal 4: Low body movement (+1)
    tracking = detector._get_ir_forward_lean_tracking(person_idx)
    body_movement = detector._calculate_body_movement(landmarks, tracking, body_indices)
    if body_movement is not None and body_movement < detector.IR_LOW_MOVEMENT_THRESHOLD:
        score += 1
        score_breakdown['low_movement'] = 1
    else:
        score_breakdown['low_movement'] = 0
    score_breakdown['body_movement'] = round(body_movement, 4) if body_movement is not None else None

    # Signal 5: Elbows below shoulders (+1)
    left_elbow = landmarks.landmark[7]
    right_elbow = landmarks.landmark[8]
    elbows_below = 0
    elbows_checked = 0
    for elbow, shoulder in [(left_elbow, left_shoulder), (right_elbow, right_shoulder)]:
        if (getattr(elbow, 'visibility', 0) > body_vis_thresh and
                getattr(shoulder, 'visibility', 0) > body_vis_thresh):
            elbows_checked += 1
            if elbow.y > shoulder.y:
                elbows_below += 1
    if elbows_checked > 0 and elbows_below == elbows_checked:
        score += 1
        score_breakdown['elbows_below'] = 1
    else:
        score_breakdown['elbows_below'] = 0

    # --- Duration gating ---
    debug_info = {
        'ir_forward_lean': True,
        'score': score,
        'threshold': score_threshold,
        'score_breakdown': score_breakdown,
        'visible_body': visible_body_count,
        'head_visible': head_visible,
    }

    if score >= score_threshold:
        tracking['sub_threshold_streak'] = 0
        if tracking['start_time'] is None:
            tracking['start_time'] = timestamp_sec

        duration = timestamp_sec - tracking['start_time']
        debug_info['duration'] = round(duration, 1)

        if duration >= sleep_duration:
            detector.logger.info(
                f"[IR FORWARD LEAN] Person {person_idx}: SLEEP detected "
                f"(score={score}/{score_threshold}, duration={duration:.1f}s)"
            )
            return True, False, debug_info
        elif duration >= min_duration:
            detector.logger.info(
                f"[IR FORWARD LEAN] Person {person_idx}: MICROSLEEP detected "
                f"(score={score}/{score_threshold}, duration={duration:.1f}s)"
            )
            return False, True, debug_info
        else:
            detector.logger.debug(
                f"[IR FORWARD LEAN] Person {person_idx}: forward lean detected but duration too short "
                f"(score={score}, duration={duration:.1f}s, need {min_duration:.0f}s)"
            )
    else:
        tracking['sub_threshold_streak'] = tracking.get('sub_threshold_streak', 0) + 1
        if tracking['sub_threshold_streak'] >= detector.SUB_THRESHOLD_STREAK_LIMIT:
            tracking['start_time'] = None
            tracking['sub_threshold_streak'] = 0
        debug_info['duration'] = 0
        debug_info['sub_threshold_streak'] = tracking['sub_threshold_streak']

    return False, False, debug_info
