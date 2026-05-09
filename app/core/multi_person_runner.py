"""Multi-person activity runner extracted from ``LocopilotActivityMonitor``.

Task 0008 (architecture cleanup): the per-frame multi-person dispatcher
``process_all_persons_activities`` previously lived as a ~1,200-line method on
the monolith. The body has been moved here verbatim with only one mechanical
change -- every ``self.<attr>`` access now reads the originating monitor as
``monitor.<attr>``. The runner is stateless; all mutable per-person tracking
state remains on the monitor.
"""

from typing import Any, Dict, List, Optional

import cv2
import math
import numpy as np

from app.core.utils.geometry import calculate_iou, bbox_overlap_with_margin


class MultiPersonActivityRunner:
    """Stateless orchestrator for per-frame multi-person activity detection.

    The :meth:`run` method is the relocated body of
    ``LocopilotActivityMonitor.process_all_persons_activities``. It receives
    the host monitor and reads every detector / tracking attribute through
    that reference, so nothing on the monitor's side needs to be re-routed.
    """

    def run(self, monitor, frame: Any, detections: Dict[str, List[Any]],
            person_roles: Dict[int, Dict[str, Any]], timestamp_sec: float,
            face_results: Any = None, frame_number: Optional[int] = None,
            precomputed_pose_results: Optional[Any] = None,
            precomputed_sleep_pose_results: Optional[Any] = None,
            is_dark_frame: Optional[bool] = None) -> Dict[str, Any]:
        """Process all detected persons for ALL activity detections (mind diversion, sleep, etc.)

        This is the MAIN multi-person processing method that:
        1. Runs YOLO26-Pose once to get all persons with keypoints (or uses precomputed results)
        2. Matches YOLO detections to person_roles by bounding box IoU
        3. Detects ALL activities for EACH person (mind diversion, sleep, cell phone, writing, etc.)
        4. Returns aggregated results for all persons

        Args:
            frame: The full frame image (BGR format)
            detections: YOLO detections dictionary containing 'person', 'cell_phone', 'book', etc.
            person_roles: Dictionary of person roles from identify_person_roles()
            timestamp_sec: Current timestamp in seconds
            face_results: MediaPipe face mesh results (optional, for mind diversion detection)
            frame_number: Frame number for logging/debugging (optional)
            precomputed_pose_results: Pre-computed YOLO pose results (optional, for GPU batch optimization)
            precomputed_sleep_pose_results: Low-confidence YOLO pose results for sleep detection fallback (optional)
            is_dark_frame: Whether the frame is dark/IR (optional, computed from brightness if None)

        Returns:
            dict: {
                'persons': {
                    person_idx: {
                        'pose_landmarks': translated landmarks,
                        'role': 'LP'/'ALP'/etc.,
                        'bbox': [x1, y1, x2, y2],
                        'activities': {
                            'mind_diversion': bool,
                            'sleep': bool,
                            'microsleep': bool,
                            'cell_phone': bool,
                            'writing': bool,
                            'packing_bags': bool,
                            'lp_hand_gesture': bool,
                            'alp_hand_gesture': bool
                        },
                        'debug_info': {
                            'head_pose': {...},
                            'sleep_info': {...},
                            'gesture_debug': {...}
                        }
                    }
                },
                'aggregated': {
                    'mind_diversion_detected': bool,
                    'sleep_detected': bool,
                    'microsleep_detected': bool,
                    'cell_phone_detected': bool,
                    'writing_detected': bool,
                    'packing_detected': bool,
                    'lp_hand_gesture_detected': bool,
                    'alp_hand_gesture_detected': bool,
                    'performing_person': int (person_idx who performed the activity, or -1 for aggregated)
                }
            }
        """
        if not person_roles or len(person_roles) == 0:
            # No persons detected, return empty results
            return {
                'persons': {},
                'aggregated': {
                    'mind_diversion_detected': False,
                    'sleep_detected': False,
                    'microsleep_detected': False,
                    'cell_phone_detected': False,
                    'writing_detected': False,
                    'packing_detected': False,
                    'lp_hand_gesture_detected': False,
                    'alp_hand_gesture_detected': False,
                    'performing_person': -1
                }
            }

        h, w = frame.shape[:2]
        persons_data = {}

        # ============ YOLO26-POSE: Single inference for all persons ============
        # Run YOLO26-Pose once on the full frame to get all persons with keypoints
        # This replaces the per-person MediaPipe cropping loop for better performance
        # If precomputed_pose_results is provided (from GPU batch inference), use it directly
        if precomputed_pose_results is not None:
            yolo_pose_results = precomputed_pose_results
        else:
            yolo_pose_results = monitor.yolo_pose.process(frame)

        # Match YOLO pose detections to person_roles by bounding box IoU
        matched_poses = monitor._match_pose_to_roles(yolo_pose_results, person_roles)

        # Match low-confidence sleep poses as fallback for persons not found at normal confidence
        matched_sleep_poses = {}
        if precomputed_sleep_pose_results is not None and precomputed_sleep_pose_results:
            matched_sleep_poses = monitor._match_pose_to_roles(precomputed_sleep_pose_results, person_roles)

        # Dark frame flag for IR forward lean detection (compute if not passed by caller)
        if is_dark_frame is None:
            is_dark_frame = False
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                frame_brightness = float(np.mean(gray)) / 255.0
                is_dark_frame = frame_brightness < monitor.settings.yolo_dark_frame_brightness_threshold
            except Exception as e:
                monitor.logger.debug(f"[DARK FRAME] Failed to check frame brightness: {e}")

        # Filter out static cell phone detections (panel instruments) before per-person processing
        detections['cell_phone'] = monitor._update_static_phone_tracking(detections['cell_phone'])

        # PRE-PASS: batch pose-guided ROI detection across ALL matched persons
        # in a single YOLO call. Downstream per-keypoint visibility check in
        # ``_collect_person_rois`` already drops low-visibility keypoints, so
        # no pre-filter is needed here. Persons the main loop later rejects
        # (torso-out-of-bbox etc.) silently contribute no valid ROIs to the
        # batch — cheap no-op, no correctness impact. Eliminates per-person
        # kernel-launch overhead (N persons -> 1 YOLO call, not N).
        precomputed_rois_by_person: Dict[int, Dict[str, List[Any]]] = {}
        if matched_poses:
            try:
                precomputed_rois_by_person = monitor.object_detector.detect_objects_multi_persons_rois(
                    frame, matched_poses
                )
            except Exception as _e:
                # Safe fallback: main loop calls per-person detection directly.
                monitor.logger.debug(f"[MULTI-PERSON ROI] batched call failed, falling back: {_e}")
                precomputed_rois_by_person = {}

        # Process each person individually
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue

            bbox = person_data['bbox']  # [x1, y1, x2, y2]

            # Get matched pose keypoints for this person
            # YOLO26-Pose provides full-frame coordinates directly (no cropping/translation needed)
            try:
                has_pose = True
                translated_landmarks = None

                if person_idx not in matched_poses:
                    has_pose = False
                else:
                    # Get the matched YoloPoseLandmarks (MediaPipe-compatible interface)
                    translated_landmarks = matched_poses[person_idx]

                    # Validate landmarks are valid before using for activity detection
                    if translated_landmarks is None or len(translated_landmarks.landmark) == 0:
                        has_pose = False
                        translated_landmarks = None
                    else:
                        # Check if at least some keypoints have good visibility
                        visible_count = sum(1 for lm in translated_landmarks.landmark if lm.visibility > 0.3)
                        if visible_count < 5:
                            has_pose = False
                            translated_landmarks = None

                h, w = frame.shape[:2]

                if has_pose and translated_landmarks is not None:
                    # ============ KEYPOINT CONSISTENCY VALIDATION ============
                    # Verify that torso center falls within (or near) the person's bbox
                    # This catches cases where pose matching assigned wrong skeleton to person
                    left_shoulder = translated_landmarks.landmark[5]  # YOLO index
                    right_shoulder = translated_landmarks.landmark[6]  # YOLO index

                    # Calculate torso center in pixel coords
                    torso_center_x = ((left_shoulder.x + right_shoulder.x) / 2) * w
                    torso_center_y = ((left_shoulder.y + right_shoulder.y) / 2) * h

                    # Check if torso center is within expanded bbox (1.5x margin)
                    bbox_margin = 0.5  # 50% expansion
                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    expanded_x1 = x1 - bbox_width * bbox_margin
                    expanded_x2 = x2 + bbox_width * bbox_margin
                    expanded_y1 = y1 - bbox_height * bbox_margin
                    expanded_y2 = y2 + bbox_height * bbox_margin

                    torso_in_bbox = (
                        expanded_x1 <= torso_center_x <= expanded_x2 and
                        expanded_y1 <= torso_center_y <= expanded_y2
                    )

                    if not torso_in_bbox:
                        monitor.logger.warning(
                            f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                            f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) outside expanded bbox "
                            f"[{expanded_x1:.0f}-{expanded_x2:.0f}, {expanded_y1:.0f}-{expanded_y2:.0f}] - SKIPPING"
                        )
                        has_pose = False
                        translated_landmarks = None
                    else:
                        monitor.logger.debug(
                            f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                            f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) VALID within bbox"
                        )

                # ============ NO-POSE PATH: Limited detection for persons without pose ============
                if not has_pose:
                    # Run object-based eating/drinking detection and no-pose sleep tracking
                    no_pose_activities = {
                        'mind_diversion': False,
                        'sleep': False,
                        'microsleep': False,
                        'cell_phone': False,
                        'writing': False,
                        'packing_bags': False,
                        'lp_hand_gesture': False,
                        'alp_hand_gesture': False,
                        'eating_drinking': False
                    }
                    no_pose_debug = {
                        'head_pose': {},
                        'sleep_info': {'no_pose': True},
                        'gesture_debug': {}
                    }

                    # --- Object-based eating/drinking (cup directly overlaps person bbox) ---
                    if getattr(monitor.settings, 'eating_drinking_detection_enabled', True):
                        cup_bottle_bboxes = []
                        cup_conf_threshold = getattr(monitor.settings, 'eating_drinking_cup_confidence', 0.25)
                        for roi_det in detections.get('roi_detections', []):
                            if roi_det['class'] in ('cup', 'bottle') and roi_det['confidence'] > cup_conf_threshold:
                                det_bbox = roi_det['bbox']
                                if bbox_overlap_with_margin(det_bbox, bbox, 50):
                                    cup_bottle_bboxes.append(det_bbox)
                        for cb_xyxy in detections.get('cup_bottle', []):
                            cb_bbox = [float(cb_xyxy[0]), float(cb_xyxy[1]), float(cb_xyxy[2]), float(cb_xyxy[3])]
                            if bbox_overlap_with_margin(cb_bbox, bbox, 50):
                                cup_bottle_bboxes.append(cb_bbox)

                        if cup_bottle_bboxes:
                            # Cup overlaps person bbox directly → eating/drinking (no hand-face check possible)
                            no_pose_activities['eating_drinking'] = True
                            no_pose_debug['head_pose']['sub_type'] = 'eating_drinking'
                            no_pose_debug['head_pose']['detected'] = True
                            no_pose_debug['head_pose']['method'] = 'no_pose_cup_overlap'
                            monitor.logger.info(
                                f"[NO-POSE EATING/DRINKING] Person {person_idx}: cup/bottle overlaps bbox, "
                                f"no pose available - flagging eating/drinking"
                            )

                    # --- Low-confidence pose fallback for sleep detection ---
                    # When normal confidence misses a sleeping person, try low-confidence poses
                    if matched_sleep_poses and person_idx in matched_sleep_poses:
                        sleep_fallback_landmarks = matched_sleep_poses[person_idx]
                        if sleep_fallback_landmarks is not None and len(sleep_fallback_landmarks.landmark) > 0:
                            visible_kps = sum(1 for lm in sleep_fallback_landmarks.landmark if lm.visibility > 0.3)
                            if visible_kps >= 5:
                                sleep_det, microsleep_det, sleep_info = monitor.sleep_detector.detect_pose_based_sleep(
                                    sleep_fallback_landmarks, timestamp_sec, person_idx=person_idx,
                                    frame_shape=frame.shape
                                )
                                if sleep_det:
                                    no_pose_activities['sleep'] = True
                                    no_pose_debug['sleep_info'] = sleep_info
                                    no_pose_debug['sleep_info']['method'] = 'low_conf_pose_fallback'
                                if microsleep_det:
                                    no_pose_activities['microsleep'] = True

                    # --- IR forward-lean sleep detection (body-only keypoints in dark frames) ---
                    # CR-NEW-003: Safe settings access with default
                    ir_fl_enabled = getattr(monitor.settings, 'ir_forward_lean_enabled', True) if monitor.settings else True
                    if is_dark_frame and ir_fl_enabled and not no_pose_activities.get('sleep', False):
                        # Try to use low-confidence sleep landmarks for body-only analysis
                        ir_landmarks = None
                        if matched_sleep_poses and person_idx in matched_sleep_poses:
                            ir_landmarks = matched_sleep_poses[person_idx]
                        elif person_idx in matched_poses:
                            ir_landmarks = matched_poses[person_idx]

                        if ir_landmarks is not None and hasattr(ir_landmarks, 'landmark') and len(ir_landmarks.landmark) > 0:
                            ir_sleep, ir_microsleep, ir_info = monitor.sleep_detector.detect_ir_forward_lean_sleep(
                                ir_landmarks, bbox, timestamp_sec, person_idx, frame.shape
                            )
                            if ir_sleep:
                                no_pose_activities['sleep'] = True
                                no_pose_debug['sleep_info'] = ir_info
                                no_pose_debug['sleep_info']['method'] = 'ir_forward_lean'
                            elif ir_microsleep:
                                no_pose_activities['microsleep'] = True
                                no_pose_debug['sleep_info'] = ir_info
                                no_pose_debug['sleep_info']['method'] = 'ir_forward_lean'

                    # --- No-pose sleep detection (bbox stability tracking) ---
                    sleep_no_pose_enabled = getattr(monitor.settings, 'sleep_no_pose_enabled', True)
                    if sleep_no_pose_enabled:
                        # Use shorter duration for IR/dark frames (15s vs 30s)
                        if is_dark_frame:
                            min_duration = getattr(monitor.settings, 'ir_sleep_no_pose_min_duration', 15.0)
                        else:
                            min_duration = getattr(monitor.settings, 'sleep_no_pose_min_duration', 30.0)
                        stability_threshold = getattr(monitor.settings, 'sleep_no_pose_bbox_stability_threshold', 0.15)

                        if person_idx not in monitor.no_pose_sleep_tracking:
                            monitor.no_pose_sleep_tracking[person_idx] = {
                                'first_seen': timestamp_sec,
                                'last_bbox': list(bbox),
                                'stable_since': timestamp_sec
                            }
                        else:
                            tracker = monitor.no_pose_sleep_tracking[person_idx]
                            # Calculate IoU between current and last bbox to measure stability
                            # CR-NEW-001: Use consolidated calculate_iou method
                            last_bbox = tracker['last_bbox']
                            iou = calculate_iou(bbox, last_bbox)
                            bbox_change = 1.0 - iou  # How much the bbox moved

                            if bbox_change > stability_threshold:
                                # Person moved significantly, reset stability timer
                                tracker['stable_since'] = timestamp_sec

                            tracker['last_bbox'] = list(bbox)

                            # Check if person has been stable (not moving) for min_duration
                            stable_duration = timestamp_sec - tracker['stable_since']
                            if stable_duration >= min_duration:
                                no_pose_activities['sleep'] = True
                                no_pose_debug['sleep_info']['no_pose_sleep'] = True
                                no_pose_debug['sleep_info']['stable_duration'] = stable_duration
                                monitor.logger.info(
                                    f"[NO-POSE SLEEP] Person {person_idx}: no pose for {stable_duration:.1f}s "
                                    f"with stable bbox (change={bbox_change:.3f}) - flagging sleep"
                                )

                    persons_data[person_idx] = {
                        'pose_landmarks': None,
                        'role': person_data.get('role', 'UNKNOWN'),
                        'role_name': person_data.get('role_name', 'Unknown'),
                        'bbox': bbox,
                        'activities': no_pose_activities,
                        'debug_info': no_pose_debug
                    }
                    continue

                # Initialize activity detection results for this person
                person_activities = {
                    'mind_diversion': False,
                    'sleep': False,
                    'microsleep': False,
                    'cell_phone': False,
                    'writing': False,
                    'packing_bags': False,
                    'lp_hand_gesture': False,
                    'alp_hand_gesture': False,
                    'eating_drinking': False
                }

                person_debug_info = {
                    'head_pose': {},
                    'sleep_info': {},
                    'gesture_debug': {}
                }

                # ============ PER-PERSON OBJECT DETECTION (ROI ONLY) ============
                # CR-006: Run ONLY pose-guided ROI detection for this person's keypoints.
                # Full-frame YOLO inference (Stage 1) is already done via the 'detections'
                # parameter passed into this method -- no need to re-run it per person.
                # PERF: consume the pre-pass batched result when available (single
                # multi-person YOLO call); fall back to per-person when not.
                if person_idx in precomputed_rois_by_person:
                    person_detections = precomputed_rois_by_person[person_idx]
                else:
                    person_detections = monitor.object_detector.detect_objects_person_rois(frame, translated_landmarks)

                # FIX C-01: Create per-person scoped detection lists instead of mutating
                # the shared 'detections' dict. Using .extend() on the shared dict causes
                # cross-person contamination: person 0's ROI detections leak into person 1's
                # checks because the list grows cumulatively across loop iterations.
                person_cell_phones = detections['cell_phone'] + person_detections['cell_phone']
                person_books = detections['book'] + person_detections['book']

                # DEBUG: Log per-person ROI detection results
                if person_detections['cell_phone']:
                    monitor.logger.info(f"[MULTI-PERSON ROI] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): Found {len(person_detections['cell_phone'])} cell phone(s)")

                # ============ ACTIVITY DETECTION FOR THIS PERSON ============

                # 1. MIND DIVERSION DETECTION
                head_pose_info = monitor.calculate_head_pose_angles(
                    translated_landmarks,
                    face_results,
                    frame.shape
                )
                person_debug_info['head_pose'] = head_pose_info
                mind_diversion_detected = head_pose_info.get('detected', False)
                person_activities['mind_diversion'] = mind_diversion_detected

                # NOTE: Mind diversion book suppression moved to AFTER writing detection
                # This allows us to check both book presence AND writing activity

                # 2. SLEEP / MICROSLEEP DETECTION (pose-based)
                # Run Haar eye detection once (shared between score boost and fallback)
                haar_eye_result = None
                # CR-NEW-003: Safe settings access with default
                haar_enabled = getattr(monitor.settings, 'haar_eye_detection_enabled', True) if monitor.settings else True
                if (haar_enabled and monitor.eye_cascade is not None):
                    haar_eye_result = monitor.sleep_detector.detect_eye_closure_haar(
                        frame, translated_landmarks, person_idx, bbox, timestamp_sec
                    )
                    person_debug_info['haar_eye_info'] = haar_eye_result

                pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = monitor.sleep_detector.detect_pose_based_sleep(
                    translated_landmarks, timestamp_sec, person_idx=person_idx,
                    frame_shape=frame.shape, haar_result=haar_eye_result
                )
                person_debug_info['sleep_info'] = pose_sleep_info
                person_activities['sleep'] = pose_sleep_detected
                person_activities['microsleep'] = pose_microsleep_detected

                # 2b. IR FORWARD LEAN FALLBACK (when dark frame and normal sleep detection missed)
                # CR-NEW-003: Safe settings access with default
                ir_fl_enabled = getattr(monitor.settings, 'ir_forward_lean_enabled', True) if monitor.settings else True
                if (is_dark_frame and ir_fl_enabled and
                        not pose_sleep_detected and not pose_microsleep_detected):
                    ir_sleep, ir_microsleep, ir_info = monitor.sleep_detector.detect_ir_forward_lean_sleep(
                        translated_landmarks, bbox, timestamp_sec, person_idx, frame.shape
                    )
                    if ir_sleep:
                        person_activities['sleep'] = True
                        person_debug_info['sleep_info'] = ir_info
                        person_debug_info['sleep_info']['method'] = 'ir_forward_lean_pose_fallback'
                    elif ir_microsleep:
                        person_activities['microsleep'] = True
                        person_debug_info['sleep_info'] = ir_info
                        person_debug_info['sleep_info']['method'] = 'ir_forward_lean_pose_fallback'

                # 2c. HAAR EYE CLOSURE FALLBACK (reuse result from step 2 — no second call)
                if (haar_eye_result is not None and
                        not person_activities.get('sleep') and not person_activities.get('microsleep')):
                    if haar_eye_result.get('is_sleep'):
                        person_activities['sleep'] = True
                        person_debug_info['sleep_info'] = {'method': 'haar_eye_closure', **haar_eye_result}
                    elif haar_eye_result.get('is_microsleep'):
                        person_activities['microsleep'] = True
                        person_debug_info['sleep_info'] = {'method': 'haar_eye_closure', **haar_eye_result}

                # 3. CELL PHONE DETECTION (check if hand near phone in THIS person's region)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                _cell_phone_fired = False
                if len(person_cell_phones) > 0:
                    # DEBUG: Log when cell phones are detected
                    if monitor.consecutive_detections.get('cell_phone', 0) == 0:
                        monitor.logger.info(f"[DEBUG CELL PHONE] {len(person_cell_phones)} phone(s) detected in frame")
                    right_hand = monitor.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = monitor.get_keypoint(translated_landmarks, 'left_wrist')

                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))

                    # STRICTER MARGIN: Reduced from default to ensure phone is really near hand
                    margin = 100  # Reduced from activity_thresholds margin to be more strict

                    for phone_bbox in person_cell_phones:
                        # Check if phone bbox overlaps with person bbox (with margin)
                        phone_in_person_region = bbox_overlap_with_margin(phone_bbox, bbox, margin)

                        if phone_in_person_region:
                            # Check if hand is near the phone (stricter check)
                            right_hand_near = monitor.check_hand_object_interaction(right_hand_coords, phone_bbox, margin)
                            left_hand_near = monitor.check_hand_object_interaction(left_hand_coords, phone_bbox, margin)

                            if right_hand_near or left_hand_near:
                                # Use per-person temporal filtering (2-3 frame requirement)
                                should_trigger = monitor.update_per_person_detection(
                                    person_idx, 'cell_phone', True, timestamp_sec
                                )
                                person_activities['cell_phone'] = should_trigger
                                _cell_phone_fired = True
                                break

                # 3a. CELL PHONE POSE FALLBACK (added 2026-04-20)
                # YOLO v8 often misses phones when the hand occludes them at the
                # ear — seen in all_activities.mp4 where 3 distinct phone-to-ear
                # events went unflagged and v8 produced only a single 0.49-conf
                # detection in 38 min of footage.
                # This fallback fires cell_phone when a wrist is sustainedly
                # close to an ear keypoint (signature pose: phone-to-ear).
                # Gated by per-person temporal filter (``cell_phone`` requires
                # ≥2 consecutive frames) so single-frame scratches/face-touches
                # don't leak through. The wrist-near-ear distance must be
                # < 20% of bbox height — tight enough that hand-on-forehead
                # or hand-on-chin won't match.
                _pose_fallback_enabled = getattr(monitor.settings, 'cell_phone_pose_fallback_enabled', True) if monitor.settings else True
                if _pose_fallback_enabled and not _cell_phone_fired:
                    try:
                        _bbox_h = max(1, bbox[3] - bbox[1])
                        # Distance ratio tightened 2026-04-20 from 0.20 → 0.15
                        # after all_activities.mp4 produced 26 phone detections
                        # (ground-truth: 4). 0.15 * bbox_h keeps phone-to-ear
                        # detectable (wrist actually at ear, ~70-100px typical)
                        # but rejects generic hand-to-face gestures (wrist
                        # near forehead / chin / nose = 120+ px).
                        _near_ear_thresh_px = max(30, int(0.15 * _bbox_h))
                        _r_wrist = monitor.get_keypoint(translated_landmarks, 'right_wrist')
                        _l_wrist = monitor.get_keypoint(translated_landmarks, 'left_wrist')
                        _r_ear = monitor.get_keypoint(translated_landmarks, 'right_ear')
                        _l_ear = monitor.get_keypoint(translated_landmarks, 'left_ear')

                        def _dist_px(a, b):
                            dx = (a.x - b.x) * w
                            dy = (a.y - b.y) * h
                            return (dx * dx + dy * dy) ** 0.5

                        _any_near_ear = False
                        _debug_pair = None
                        # Pair each visible wrist with each visible ear on the
                        # same side (natural phone-to-ear pose). Crossed pairs
                        # (right wrist to left ear) are also accepted since LP
                        # often holds phone to opposite ear.
                        _pairs = [
                            ('right_wrist', _r_wrist, 'right_ear', _r_ear),
                            ('left_wrist',  _l_wrist, 'left_ear',  _l_ear),
                            ('right_wrist', _r_wrist, 'left_ear',  _l_ear),
                            ('left_wrist',  _l_wrist, 'right_ear', _r_ear),
                        ]
                        for _wn, _w, _en, _e in _pairs:
                            if _w.visibility < 0.3 or _e.visibility < 0.3:
                                continue
                            _d = _dist_px(_w, _e)
                            if _d < _near_ear_thresh_px:
                                _any_near_ear = True
                                _debug_pair = (_wn, _en, _d)
                                break

                        if _any_near_ear:
                            _should_trigger = monitor.update_per_person_detection(
                                person_idx, 'cell_phone', True, timestamp_sec
                            )
                            person_activities['cell_phone'] = _should_trigger
                            if _should_trigger and monitor.consecutive_detections.get('cell_phone', 0) in (0, 1):
                                monitor.logger.info(
                                    f"[CELL PHONE POSE FALLBACK] Person {person_idx}: "
                                    f"wrist-near-ear ({_debug_pair[0]} <-> {_debug_pair[1]}, "
                                    f"dist={_debug_pair[2]:.0f}px < {_near_ear_thresh_px}px) "
                                    f"— no YOLO phone detection this frame, firing on pose"
                                )
                    except Exception as _e:
                        monitor.logger.debug(f"[CELL PHONE POSE FALLBACK] skip: {_e}")

                # 3b. EATING/DRINKING DETECTION (cup/bottle near face = mind diversion)
                eating_drinking_detected = False
                if getattr(monitor.settings, 'eating_drinking_detection_enabled', True) and not person_activities.get('mind_diversion', False):
                    # Check if cup/bottle detected in ROI near this person
                    cup_bottle_bboxes = []
                    cup_conf_threshold = getattr(monitor.settings, 'eating_drinking_cup_confidence', 0.25)
                    for roi_det in detections.get('roi_detections', []):
                        if roi_det['class'] in ('cup', 'bottle') and roi_det['confidence'] > cup_conf_threshold:
                            det_bbox = roi_det['bbox']
                            if bbox_overlap_with_margin(det_bbox, bbox, 100):
                                cup_bottle_bboxes.append(det_bbox)

                    # Also check full-frame cup/bottle detections
                    for cb_xyxy in detections.get('cup_bottle', []):
                        cb_bbox = [float(cb_xyxy[0]), float(cb_xyxy[1]), float(cb_xyxy[2]), float(cb_xyxy[3])]
                        if bbox_overlap_with_margin(cb_bbox, bbox, 100):
                            cup_bottle_bboxes.append(cb_bbox)

                    if cup_bottle_bboxes:
                        # Check if hand is near face level AND holding cup/bottle
                        right_wrist = monitor.get_keypoint(translated_landmarks, 'right_wrist')
                        left_wrist = monitor.get_keypoint(translated_landmarks, 'left_wrist')
                        nose = monitor.get_keypoint(translated_landmarks, 'nose')

                        hand_face_margin = getattr(monitor.settings, 'eating_drinking_hand_face_margin', 80)
                        hand_obj_margin = getattr(monitor.settings, 'eating_drinking_hand_object_margin', 150)

                        if nose and nose.visibility > 0.3:
                            nose_y = nose.y * h
                            for wrist in [right_wrist, left_wrist]:
                                if wrist and wrist.visibility > 0.3:
                                    wrist_y = wrist.y * h
                                    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                                    # Hand is at or above shoulder/chin level (drinking position)
                                    if wrist_y < nose_y + hand_face_margin:
                                        for cb_bbox in cup_bottle_bboxes:
                                            if monitor.check_hand_object_interaction(wrist_coords, cb_bbox, hand_obj_margin):
                                                eating_drinking_detected = True
                                                break
                                if eating_drinking_detected:
                                    break

                        # Fallback: cup directly overlaps person bbox AND at least one wrist visible
                        # Handles overhead camera angles where hand-face proximity is unreliable
                        if not eating_drinking_detected:
                            any_wrist_visible = any(
                                w_kp and w_kp.visibility > 0.3
                                for w_kp in [right_wrist, left_wrist]
                            )
                            if any_wrist_visible:
                                for cb_bbox in cup_bottle_bboxes:
                                    # Cup must directly overlap person bbox (no margin = stricter spatial check)
                                    if bbox_overlap_with_margin(cb_bbox, bbox, 0):
                                        eating_drinking_detected = True
                                        monitor.logger.info(
                                            f"[EATING/DRINKING FALLBACK] Cup/bottle directly overlaps person {person_idx} bbox "
                                            f"with wrist visible - flagging eating/drinking"
                                        )
                                        break

                    if eating_drinking_detected:
                        person_activities['eating_drinking'] = True
                        person_debug_info['head_pose']['sub_type'] = 'eating_drinking'
                        person_debug_info['head_pose']['detected'] = True
                        person_debug_info['head_pose']['method'] = 'object_proximity'

                # 3c. EATING/DRINKING POSE-ONLY FALLBACK — REMOVED (2026-04-26)
                # Was firing on every frame where a wrist landed in the
                # (±100 px, -40 to +80 px) box around the nose, with no gate
                # against an actual cup/bottle ever being detected. On the
                # TV22 production batch this produced 99 eating/drinking
                # activities across 10 videos (vs 4 from the cup/bottle
                # paths), all with stationary wrists parked on control
                # levers — almost certainly false positives. Eating/drinking
                # now requires YOLO to actually see a cup or bottle (paths
                # 3a / 3b above).

                # 4. WRITING DETECTION — calibrated across TV22.3–TV22.9 GT
                # frames on 2026-04-23. Two detection paths are OR-combined.
                # Role is no longer restricted: both LP and ALP can write
                # (TV22.6 and TV22.8 GTs are LP writing events).
                #
                # PRIMARY — both wrists INSIDE book bbox (+10 px pad):
                #   High-precision, ~fires on ~25 % of writing-GT samples.
                #   Measured hits on GT windows (both_in=True):
                #     TV22.4 3/5, TV22.5_0447 1/5, TV22.7_0321 1/5,
                #     TV22.7_0932 2/5, TV22.7_2611 1/5, TV22.9 1/5
                #
                # FALLBACK — pose-only "hands held together in the lap/chest zone"
                # Fires when the book class is missed by YOLO (observed in
                # 40-100 % of GT samples for TV22.6, TV22.7_1906, TV22.8).
                # Criteria measured across all 10 writing GTs:
                #   both wrists visible
                #   wrist_dist < 80 px  (hands together holding paper)
                #   both wrists BELOW shoulders (wrist_y > shoulder_y + 40 px)
                #   both wrists in lower-2/3 of person bbox (avoid raised-hand FPs)
                writing_detected_raw = False
                if True:  # no role filter — LP can write too (TV22.6, TV22.8)
                    right_hand = monitor.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = monitor.get_keypoint(translated_landmarks, 'left_wrist')
                    rv = right_hand.visibility
                    lv = left_hand.visibility
                    min_wrist_vis = getattr(
                        monitor.settings, 'writing_min_wrist_visibility', 0.3
                    )
                    allow_single_wrist = getattr(
                        monitor.settings, 'writing_allow_single_wrist', True
                    )
                    # LOG-BOOK ROI MASK: drop book detections whose centre falls
                    # outside the desk/lap interaction zone. Most FPs in the
                    # TV22 batch came from YOLO mis-classifying small control-
                    # panel devices as "book" or from the wrist drifting near
                    # an unrelated desk object during routine cab activity.
                    # Format: ``WRITING_BOOK_ROI=x1,y1,x2,y2`` normalized to
                    # frame size. Empty (default) keeps current behaviour.
                    roi_str = getattr(monitor.settings, 'writing_book_roi', '') or ''
                    if roi_str and person_books:
                        try:
                            r_parts = [float(v) for v in roi_str.split(',')]
                            if len(r_parts) == 4:
                                rx1 = r_parts[0] * w
                                ry1 = r_parts[1] * h
                                rx2 = r_parts[2] * w
                                ry2 = r_parts[3] * h
                                kept_books = []
                                for _b in person_books:
                                    bcx = (_b[0] + _b[2]) * 0.5
                                    bcy = (_b[1] + _b[3]) * 0.5
                                    if rx1 <= bcx <= rx2 and ry1 <= bcy <= ry2:
                                        kept_books.append(_b)
                                _filtered_n = len(person_books) - len(kept_books)
                                if _filtered_n > 0:
                                    monitor.logger.info(
                                        f"[WRITING:roi-filter] P{person_idx} "
                                        f"f{frame_number} t={timestamp_sec:.1f}s: "
                                        f"dropped {_filtered_n}/{len(person_books)} "
                                        f"book(s) outside ROI=[{r_parts[0]:.2f},"
                                        f"{r_parts[1]:.2f},{r_parts[2]:.2f},"
                                        f"{r_parts[3]:.2f}]"
                                    )
                                person_books = kept_books
                        except (ValueError, IndexError):
                            pass  # malformed; ignore
                    has_book = len(person_books) > 0
                    if rv >= min_wrist_vis and lv >= min_wrist_vis:
                        rwx, rwy = right_hand.x * w, right_hand.y * h
                        lwx, lwy = left_hand.x * w, left_hand.y * h
                        wr_dist = math.hypot(rwx - lwx, rwy - lwy)

                        # PRIMARY PATH — relaxed wrist-vs-book rule.
                        # Original "BOTH wrists inside book bbox" missed real
                        # writing where one hand holds the page (inside bbox)
                        # and the writing hand sits a few pixels off the page
                        # edge, or where the page extends below the YOLO bbox.
                        # Diagnostic logs across TV22.4/.7/.8/.9 GT windows
                        # showed wrist→bbox edge distances of 4-44 px on the
                        # "outside" wrist when one wrist was inside.
                        # New rule: at least one wrist inside AND the other
                        # within ``writing_other_wrist_max_dist`` px of the
                        # nearest bbox edge (default 50).
                        if has_book:
                            monitor._writing_last_book_seen[person_idx] = timestamp_sec
                            bbox_pad = 10
                            other_wrist_max = getattr(
                                monitor.settings, 'writing_other_wrist_max_dist', 50
                            )
                            def _edge_dist(x, y, b1, b2, b3, b4):
                                dx = max(b1 - x, 0, x - b3)
                                dy = max(b2 - y, 0, y - b4)
                                return (dx * dx + dy * dy) ** 0.5
                            best_miss = None  # (book_bbox, r_in, l_in, dR, dL)
                            for book_bbox in person_books:
                                bx1, by1, bx2, by2 = book_bbox[:4]
                                r_in = (bx1 - bbox_pad <= rwx <= bx2 + bbox_pad) and \
                                       (by1 - bbox_pad <= rwy <= by2 + bbox_pad)
                                l_in = (bx1 - bbox_pad <= lwx <= bx2 + bbox_pad) and \
                                       (by1 - bbox_pad <= lwy <= by2 + bbox_pad)
                                dR = _edge_dist(rwx, rwy, bx1, by1, bx2, by2)
                                dL = _edge_dist(lwx, lwy, bx1, by1, bx2, by2)
                                fired_strict = r_in and l_in
                                fired_relaxed = (
                                    (r_in and dL <= other_wrist_max)
                                    or (l_in and dR <= other_wrist_max)
                                )
                                if fired_strict or fired_relaxed:
                                    writing_detected_raw = True
                                    rule = 'strict' if fired_strict else 'relaxed'
                                    monitor.logger.info(
                                        f"[WRITING:book] P{person_idx} f{frame_number} "
                                        f"t={timestamp_sec:.1f}s rule={rule}: "
                                        f"wrist_dist={wr_dist:.0f} dR={dR:.0f} dL={dL:.0f} "
                                        f"r_in={r_in} l_in={l_in} book_bbox={book_bbox}"
                                    )
                                    break
                                if best_miss is None or (r_in or l_in):
                                    best_miss = (book_bbox, r_in, l_in, dR, dL)
                            if not writing_detected_raw and best_miss is not None:
                                bb, r_in, l_in, dR, dL = best_miss
                                bx1, by1, bx2, by2 = (int(v) for v in bb[:4])
                                monitor.logger.info(
                                    f"[WRITING:miss-bbox] P{person_idx} f{frame_number} "
                                    f"t={timestamp_sec:.1f}s: books={len(person_books)} "
                                    f"R=({int(rwx)},{int(rwy)})@{rv:.2f} r_in={r_in} dR={dR:.0f} "
                                    f"L=({int(lwx)},{int(lwy)})@{lv:.2f} l_in={l_in} dL={dL:.0f} "
                                    f"book=[{bx1},{by1},{bx2},{by2}]"
                                )
                    elif has_book:
                        # SINGLE-WRIST FALLBACK — when one wrist is occluded by
                        # the writing hand (visibility < min_wrist_vis) we lose
                        # the dual-wrist signal. The visible wrist alone, if
                        # confidently INSIDE the book bbox, is enough evidence
                        # that this person is interacting with the log book.
                        # Calibrated 2026-05-06 from TV22.5 4:47 (rv=0.99
                        # lv=0.03), TV22.7 9:32 (rv=0.99 lv=0.03-0.21).
                        # Stricter than the dual rule: visible wrist must be
                        # fully inside the bbox (no edge-distance slack), so
                        # we don't fire on a hand that just brushes the book
                        # area while reaching for controls.
                        bbox_pad = 10
                        if allow_single_wrist and (rv >= 0.5) ^ (lv >= 0.5):
                            if rv >= 0.5:
                                wx, wy, wname = right_hand.x * w, right_hand.y * h, 'R'
                                wvis = rv
                            else:
                                wx, wy, wname = left_hand.x * w, left_hand.y * h, 'L'
                                wvis = lv
                            for book_bbox in person_books:
                                bx1, by1, bx2, by2 = book_bbox[:4]
                                if (bx1 - bbox_pad <= wx <= bx2 + bbox_pad and
                                        by1 - bbox_pad <= wy <= by2 + bbox_pad):
                                    writing_detected_raw = True
                                    monitor.logger.info(
                                        f"[WRITING:book] P{person_idx} f{frame_number} "
                                        f"t={timestamp_sec:.1f}s rule=single_wrist: "
                                        f"{wname}=({int(wx)},{int(wy)})@{wvis:.2f} "
                                        f"inside book_bbox={book_bbox}"
                                    )
                                    monitor._writing_last_book_seen[person_idx] = timestamp_sec
                                    break
                        if not writing_detected_raw:
                            monitor.logger.info(
                                f"[WRITING:miss-vis] P{person_idx} f{frame_number} "
                                f"t={timestamp_sec:.1f}s: books={len(person_books)} "
                                f"rv={rv:.2f} lv={lv:.2f} (need >={min_wrist_vis:.1f} each, "
                                f"or single >=0.5 inside bbox)"
                            )

                        # FALLBACK PATH — REMOVED (2026-04-26)
                        # The pose-only "wrists-together-in-lap, book seen
                        # within last 30 s" path was producing 61 of 62
                        # writing flags on the most recent test video, vs
                        # 1 from the primary "wrists inside book bbox"
                        # path. The 30 s recency gate was too loose: any
                        # transient book sighting (paper waved, control
                        # panel sticker, side reading material) opened a
                        # 30 s window in which any seated posture with
                        # hands held together would trigger writing.
                        # Writing now requires YOLO to actually detect a
                        # book and BOTH wrists to fall inside its bbox
                        # (primary path above). _writing_last_book_seen
                        # tracking is left in place but no longer
                        # consumed by any rule path; harmless to keep.

                should_trigger = monitor.update_per_person_detection(
                    person_idx, 'writing', writing_detected_raw, timestamp_sec
                )
                person_activities['writing'] = should_trigger
                person_debug_info['writing_method'] = 'book_hand' if writing_detected_raw else 'none'

                # SUPPRESS MIND DIVERSION IF LEGITIMATE WORK ACTIVITY DETECTED
                # Uses comprehensive suppression logic that checks:
                # 1. Writing activity detected
                # 2. Recent writing activity (within grace period)
                # 3. Book/document present in frame
                # 4. Hands in writing position (wrists close together, below face)
                if person_activities['mind_diversion']:
                    # Don't suppress eating/drinking detections — they are object-based, not head-pose
                    sub_type = person_debug_info.get('head_pose', {}).get('sub_type')
                    if sub_type != 'eating_drinking':
                        # FIX C-01: Pass per-person scoped detections to avoid
                        # cross-person book contamination in suppression logic
                        person_scoped_detections = {**detections, 'book': person_books, 'cell_phone': person_cell_phones}
                        should_suppress, suppress_reason = monitor.should_suppress_mind_diversion(
                            person_idx=person_idx,
                            person_activities=person_activities,
                            pose_landmarks=translated_landmarks,
                            detections=person_scoped_detections,
                            frame_shape=frame.shape,
                            current_time=timestamp_sec
                        )

                        if should_suppress:
                            person_activities['mind_diversion'] = False
                            person_debug_info['head_pose']['suppressed'] = True
                            person_debug_info['head_pose']['suppressed_reason'] = suppress_reason
                        else:
                            person_debug_info['head_pose']['suppressed'] = False
                            person_debug_info['head_pose']['suppressed_reason'] = None

                # 5. PACKING DETECTION (check if hand near backpack in THIS person's region)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                # FP-FIX: Filter out static backpacks (cabin fixtures) before detection
                active_backpacks = monitor._update_static_backpack_tracking(detections['backpack'])
                if len(active_backpacks) > 0:
                    right_hand = monitor.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = monitor.get_keypoint(translated_landmarks, 'left_wrist')

                    # Check wrist visibility - only use if visible enough
                    right_wrist_visible = right_hand.visibility > 0.3
                    left_wrist_visible = left_hand.visibility > 0.3

                    # Use smoothed hand positions to reduce pose estimation noise (only if visible)
                    right_hand_coords = None
                    left_hand_coords = None

                    if right_wrist_visible:
                        right_hand_coords = monitor._get_smoothed_hand_position(
                            person_idx, 'right', right_hand, w, h, timestamp_sec
                        )
                    elif left_wrist_visible:
                        # Fallback: if right wrist not visible, try using right elbow as approximation
                        right_elbow = monitor.get_keypoint(translated_landmarks, 'right_elbow')
                        if right_elbow.visibility > 0.3:
                            right_hand_coords = (int(right_elbow.x * w), int(right_elbow.y * h))

                    if left_wrist_visible:
                        left_hand_coords = monitor._get_smoothed_hand_position(
                            person_idx, 'left', left_hand, w, h, timestamp_sec
                        )
                    elif right_wrist_visible:
                        # Fallback: if left wrist not visible, try using left elbow as approximation
                        left_elbow = monitor.get_keypoint(translated_landmarks, 'left_elbow')
                        if left_elbow.visibility > 0.3:
                            left_hand_coords = (int(left_elbow.x * w), int(left_elbow.y * h))

                    # Separate margins: region overlap vs. hand proximity
                    region_margin = monitor.activity_thresholds['packing_bags'].get('region_margin', 100)
                    proximity_margin = monitor.activity_thresholds['packing_bags']['margin']

                    # ============ SIMPLIFIED PACKING DETECTION ============
                    # Core logic: If wrist is inside/near backpack bbox -> Packing detected!
                    # M-01 FIX: Track the best match across ALL backpacks instead of
                    # stopping at the first match. Priority: wrist-inside > motion-confirmed.
                    # Within the same priority, prefer the closest backpack (smallest distance).
                    packing_motion_analysis = None
                    packing_detected_simple = False
                    best_pack_type = None        # 'wrist_inside' | 'motion' | None
                    best_pack_distance = float('inf')
                    best_pack_bbox = None
                    best_pack_motion = None
                    best_pack_debug = None

                    for backpack_bbox in active_backpacks:
                        # Check if backpack is in this person's region (wider margin)
                        backpack_in_person_region = bbox_overlap_with_margin(
                            backpack_bbox, bbox, region_margin
                        )

                        if not backpack_in_person_region:
                            continue

                        # ===== SIMPLIFIED CHECK: Is wrist INSIDE backpack bbox? =====
                        right_inside, right_dist = monitor.activity_detector.is_wrist_inside_backpack(
                            right_hand_coords, backpack_bbox, margin=40
                        )
                        left_inside, left_dist = monitor.activity_detector.is_wrist_inside_backpack(
                            left_hand_coords, backpack_bbox, margin=40
                        )

                        wrist_inside_backpack = right_inside or left_inside
                        closest_distance = min(right_dist, left_dist)

                        cur_debug = {
                            'right_wrist_inside': right_inside,
                            'left_wrist_inside': left_inside,
                            'right_dist': right_dist,
                            'left_dist': left_dist,
                            'closest_distance': closest_distance,
                            'backpack_bbox': list(backpack_bbox[:4])
                        }

                        # ===== PRIMARY: Wrist inside backpack bbox =====
                        # FP-FIX 2026-04-11 (v2): earlier patch delegated to analyze_packing_hand_motion
                        # which accepts direction_changes>=1 OR sustained_proximity. Fidgeting trivially
                        # produces direction_changes=1, and a seated LP next to a stationary bag is always
                        # in sustained_proximity — so both branches leaked (6 FPs in run_094731).
                        # The primary path now requires the AND of:
                        #   - direction_changes >= 2  (not a single back-and-forth twitch)
                        #   - sustained_proximity      (hand genuinely inside the bag region)
                        #   - time_span >= 6s          (not a momentary pass-through)
                        #   - velocity in 15–200 px/s  (real hand motion, not noise)
                        # Real packing (reach in → manipulate → pull out) passes all four.
                        # Fidgeting, sitting near a bag, or reaching past a bag for something else fails.
                        if wrist_inside_backpack:
                            primary_motion = monitor.analyze_packing_hand_motion(
                                person_idx, translated_landmarks, frame.shape, timestamp_sec, backpack_bbox
                            )
                            strict_packing = (
                                primary_motion.get('direction_changes', 0) >= 2
                                and primary_motion.get('sustained_proximity', False)
                                and primary_motion.get('time_span', 0) >= 6.0
                                and 15 <= primary_motion.get('avg_velocity', 0) <= 200
                            )
                            if not strict_packing:
                                continue  # wrist inside but no sustained packing motion — reject

                            if best_pack_type != 'wrist_inside' or closest_distance < best_pack_distance:
                                best_pack_type = 'wrist_inside'
                                best_pack_distance = closest_distance
                                best_pack_bbox = backpack_bbox
                                best_pack_motion = primary_motion
                                best_pack_debug = cur_debug
                            continue  # Check remaining backpacks for a closer match

                        # ===== FALLBACK: Hand near backpack with motion analysis =====
                        # Only consider if we have not found a wrist-inside match yet
                        if best_pack_type == 'wrist_inside':
                            continue

                        hand_near_backpack = (
                            monitor.check_hand_object_interaction(right_hand_coords, backpack_bbox, proximity_margin) or
                            monitor.check_hand_object_interaction(left_hand_coords, backpack_bbox, proximity_margin)
                        )

                        if hand_near_backpack:
                            cur_motion = monitor.analyze_packing_hand_motion(
                                person_idx, translated_landmarks, frame.shape, timestamp_sec, backpack_bbox
                            )
                            motion_confirmed = cur_motion['packing_motion_detected']
                            sustained_proximity = cur_motion.get('sustained_proximity', False) and \
                                                 cur_motion.get('sustained_proximity_time', False)

                            if motion_confirmed or sustained_proximity:
                                if best_pack_type != 'motion' or closest_distance < best_pack_distance:
                                    best_pack_type = 'motion'
                                    best_pack_distance = closest_distance
                                    best_pack_bbox = backpack_bbox
                                    best_pack_motion = cur_motion
                                    best_pack_debug = cur_debug

                    # ===== APPLY BEST MATCH RESULT AFTER LOOP =====
                    if best_pack_debug is not None:
                        person_debug_info['packing_wrist_check'] = best_pack_debug

                    # FP-FIX: Wrist motion gate — require actual hand movement for packing
                    # Stationary hands near controls should not trigger packing detection
                    if best_pack_type is not None:
                        wrist_has_motion = monitor._check_wrist_motion_for_packing(person_idx, timestamp_sec)
                        if not wrist_has_motion:
                            monitor.logger.debug(
                                f"[PACKING WRIST MOTION GATE] Person {person_idx}: "
                                f"suppressed packing (wrists stationary)"
                            )
                            best_pack_type = None  # Suppress the detection

                    if best_pack_type == 'wrist_inside':
                        packing_detected_simple = True
                        monitor.logger.info(
                            f"PACKING DETECTED (SIMPLE): Wrist inside backpack bbox! "
                            f"Distance: {best_pack_distance:.0f}px, "
                            f"Backpack: {list(best_pack_bbox[:4])}"
                        )
                        should_trigger = monitor.update_per_person_detection(
                            person_idx, 'packing_bags', True, timestamp_sec
                        )
                        person_activities['packing_bags'] = should_trigger
                        if person_idx not in monitor.recent_person_activities:
                            monitor.recent_person_activities[person_idx] = {}
                        monitor.recent_person_activities[person_idx]['packing_bags'] = timestamp_sec

                    elif best_pack_type == 'motion':
                        packing_motion_analysis = best_pack_motion
                        person_debug_info['packing_motion'] = packing_motion_analysis
                        should_trigger = monitor.update_per_person_detection(
                            person_idx, 'packing_bags', True, timestamp_sec
                        )
                        person_activities['packing_bags'] = should_trigger
                        if person_idx not in monitor.recent_person_activities:
                            monitor.recent_person_activities[person_idx] = {}
                        monitor.recent_person_activities[person_idx]['packing_bags'] = timestamp_sec

                    else:
                        # No match found across all backpacks - reset counter
                        should_trigger = monitor.update_per_person_detection(
                            person_idx, 'packing_bags', False, timestamp_sec
                        )
                        person_activities['packing_bags'] = should_trigger

                # UPDATE TEMPORAL HISTORY for writing and cell phone too
                if person_activities.get('writing', False):
                    if person_idx not in monitor.recent_person_activities:
                        monitor.recent_person_activities[person_idx] = {}
                    monitor.recent_person_activities[person_idx]['writing'] = timestamp_sec

                if person_activities.get('cell_phone', False):
                    if person_idx not in monitor.recent_person_activities:
                        monitor.recent_person_activities[person_idx] = {}
                    monitor.recent_person_activities[person_idx]['cell_phone'] = timestamp_sec

                # 6. HAND GESTURE DETECTION (LP/ALP)
                # CRITICAL: This runs AFTER packing/writing/phone detection for context-aware filtering
                # Pass person_activities, backpack detections, person_idx, and timestamp for full suppression
                single_person_roles = {person_idx: person_data}
                lp_gesture, alp_gesture, gesture_debug = monitor.detect_hand_gesture(
                    translated_landmarks,
                    frame.shape,
                    single_person_roles,
                    yolo_person_boxes=None,
                    person_activities=person_activities,
                    backpack_detections=detections.get('backpack', []),
                    person_idx=person_idx,
                    current_timestamp=timestamp_sec,
                    frame_number=frame_number
                )
                person_debug_info['gesture_debug'] = gesture_debug

                person_activities['lp_hand_gesture'] = lp_gesture
                person_activities['alp_hand_gesture'] = alp_gesture

                # Track hand raise timestamps for temporal coordination window
                if person_activities['lp_hand_gesture']:
                    if person_idx not in monitor.recent_person_activities:
                        monitor.recent_person_activities[person_idx] = {}
                    monitor.recent_person_activities[person_idx]['lp_hand_raise'] = timestamp_sec

                if person_activities['alp_hand_gesture']:
                    if person_idx not in monitor.recent_person_activities:
                        monitor.recent_person_activities[person_idx] = {}
                    monitor.recent_person_activities[person_idx]['alp_hand_raise'] = timestamp_sec

                # Store this person's data
                persons_data[person_idx] = {
                    'pose_landmarks': translated_landmarks,
                    'role': person_data.get('role', 'UNKNOWN'),
                    'role_name': person_data.get('role_name', 'Unknown'),
                    'bbox': bbox,
                    'activities': person_activities,
                    'debug_info': person_debug_info
                }

            except Exception as e:
                monitor.logger.error(f"Error processing person {person_idx}: {e}", exc_info=True)
                continue

        # ============ CLEAN UP STALE PER-PERSON TRACKING (CR-012) ============
        # Remove tracking for persons no longer detected to prevent stale state
        active_person_indices = set(persons_data.keys())
        monitor._cleanup_stale_person_tracking(active_person_indices)

        # Also clear no-pose tracking for persons that now have pose (they moved to the pose path)
        for person_idx in list(monitor.no_pose_sleep_tracking.keys()):
            if person_idx in persons_data and persons_data[person_idx].get('pose_landmarks') is not None:
                del monitor.no_pose_sleep_tracking[person_idx]

        # ============ AGGREGATE RESULTS ACROSS ALL PERSONS ============
        aggregated = {
            'mind_diversion_detected': False,
            'sleep_detected': False,
            'microsleep_detected': False,
            'cell_phone_detected': False,
            'writing_detected': False,
            'packing_detected': False,
            'lp_hand_gesture_detected': False,
            'alp_hand_gesture_detected': False,
            'eating_drinking_detected': False,
            'performing_person': -1,
            'performing_persons': [],  # List of person indices who performed activities
            # F1 (2026-04-06): per-activity triggering-person map so evidence
            # attribution picks the actual person_idx that raised each flag
            # (previously `min(person_roles.keys())` always attributed to LP).
            'triggering_persons_by_activity': {
                'mind_diversion': [],
                'sleep': [],
                'microsleep': [],
                'cell_phone': [],
                'writing': [],
                'packing_bags': [],
                'lp_hand_gesture': [],
                'alp_hand_gesture': [],
                'eating_drinking': [],
            }
        }
        triggering_map = aggregated['triggering_persons_by_activity']

        # Aggregate: if ANY person has an activity, mark it as detected
        # Per-person state machine gate (H-02 fix): only aggregate sleep/microsleep
        # if THAT SPECIFIC person's state machine is in DROWSY or beyond.
        # This prevents person 0's SLEEPING state from letting person 1's
        # microsleep bypass the gate.
        for person_idx, person_data in persons_data.items():
            activities = person_data['activities']

            if activities['mind_diversion']:
                aggregated['mind_diversion_detected'] = True
                aggregated['performing_persons'].append(person_idx)
                triggering_map['mind_diversion'].append(person_idx)

            # Per-person state machine gate for sleep/microsleep
            person_sleep_info = person_data.get('debug_info', {}).get('sleep_info', {})
            person_sleep_state = person_sleep_info.get('sleep_state', 'ALERT')
            person_state_machine_ready = person_sleep_state in ('DROWSY', 'MICROSLEEP', 'SLEEPING')

            if activities['sleep']:
                if person_state_machine_ready:
                    aggregated['sleep_detected'] = True
                    triggering_map['sleep'].append(person_idx)
                else:
                    # Suppress this person's sleep - state machine not ready
                    activities['sleep'] = False
            if activities['microsleep']:
                if person_state_machine_ready:
                    aggregated['microsleep_detected'] = True
                    triggering_map['microsleep'].append(person_idx)
                else:
                    # Suppress this person's microsleep - state machine not ready
                    activities['microsleep'] = False
            if activities['cell_phone']:
                aggregated['cell_phone_detected'] = True
                triggering_map['cell_phone'].append(person_idx)
            if activities['writing']:
                aggregated['writing_detected'] = True
                triggering_map['writing'].append(person_idx)
            if activities['packing_bags']:
                aggregated['packing_detected'] = True
                triggering_map['packing_bags'].append(person_idx)
            if activities['lp_hand_gesture']:
                aggregated['lp_hand_gesture_detected'] = True
                triggering_map['lp_hand_gesture'].append(person_idx)
            if activities['alp_hand_gesture']:
                aggregated['alp_hand_gesture_detected'] = True
                triggering_map['alp_hand_gesture'].append(person_idx)
            if activities.get('eating_drinking'):
                aggregated['eating_drinking_detected'] = True
                triggering_map['eating_drinking'].append(person_idx)

        # Set performing_person to the first detected person (for backward compatibility)
        if aggregated['performing_persons']:
            aggregated['performing_person'] = aggregated['performing_persons'][0]

        return {
            'persons': persons_data,
            'aggregated': aggregated
        }
