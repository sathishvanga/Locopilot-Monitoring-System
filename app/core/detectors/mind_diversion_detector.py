"""Mind diversion detection using head pose analysis.

This module provides detection of mind diversion (attention lapses) through
head pose angle analysis. It uses both pose landmarks (nose, shoulders) and
optional face mesh landmarks for improved accuracy.

Detects three sub-types of mind diversion:
1. looking_sideways - Head turned significantly to the side (> yaw threshold)
2. looking_down_distracted - Head looking down (> pitch threshold)
3. looking_away_combined - Head turned AND looking down (compound threshold)

The detector accounts for camera placement (behind-right of crew) where
negative yaw indicates looking toward the track (legitimate work).
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.utils.pose_utils import get_keypoint as _canonical_get_keypoint
from app.utils.logger import get_logger


class MindDiversionDetector:
    """Detects mind diversion through head pose angles.

    This detector analyzes head pose (yaw and pitch angles) to identify
    when a person's attention may have diverted from their primary task.
    It supports both YOLO pose landmarks and optional MediaPipe face mesh
    for improved accuracy.

    The detector is configured for a camera placement behind-right of the
    crew, so negative yaw (looking left toward the track) is considered
    legitimate forward-looking behavior.

    Attributes:
        yaw_sideways: Yaw threshold for looking_sideways detection (degrees)
        yaw_combined: Yaw threshold for looking_away_combined detection (degrees)
        pitch_down: Pitch threshold for looking_down_distracted detection (degrees)
        pitch_combined: Pitch threshold for looking_away_combined detection (degrees)
        yaw_max_for_down: Max yaw allowed for looking_down_distracted (degrees)
        exempt_forward_looking: Whether to exempt negative yaw (looking at track)
        suppress_with_writing: Whether to suppress detection when writing is active
        writing_grace_seconds: Grace period after writing ends (seconds)
        wrist_distance_threshold: Max wrist distance for writing pose detection (pixels)
    """

    def __init__(self, settings: Optional[Any] = None, logger: Optional[logging.Logger] = None) -> None:
        """Initialize the mind diversion detector.

        Args:
            settings: Optional settings object with threshold configurations.
                     If None, default values are used.
            logger: Optional logger instance. If None, creates a file-only logger.
        """
        self.logger = logger or get_logger('MindDiversionDetector')
        self.settings = settings

        # Yaw/Pitch thresholds (from settings or defaults).
        # pitch_down default lowered 45 -> 30 on 2026-05-20 after audit found
        # the previous default was dead code: pitch_angle is np.clip(..., -45, 45)
        # so the comparison `pitch_angle > pitch_down=45` could never fire,
        # making looking_down_distracted unreachable on the pose path.
        self.yaw_sideways = getattr(settings, 'mind_diversion_yaw_sideways', 78) if settings else 78
        self.yaw_combined = getattr(settings, 'mind_diversion_yaw_combined', 58) if settings else 58
        self.pitch_down = getattr(settings, 'mind_diversion_pitch_down', 30) if settings else 30
        self.pitch_combined = getattr(settings, 'mind_diversion_pitch_combined', 35) if settings else 35
        self.yaw_max_for_down = getattr(settings, 'mind_diversion_yaw_max_for_down', 55) if settings else 55

        # Forward-looking exemption
        self.exempt_forward_looking = getattr(settings, 'mind_diversion_exempt_forward_looking', True) if settings else True

        # Writing suppression settings
        self.suppress_with_writing = getattr(settings, 'mind_diversion_suppress_with_writing', True) if settings else True
        self.writing_grace_seconds = getattr(settings, 'mind_diversion_writing_grace_seconds', 5.0) if settings else 5.0
        self.wrist_distance_threshold = getattr(settings, 'mind_diversion_wrist_distance_threshold', 200) if settings else 200

        # State tracking for sustained detection
        self._recent_person_activities: Dict[int, Dict[str, float]] = {}

        self.logger.info(
            f"MindDiversionDetector initialized: yaw_sideways={self.yaw_sideways}, "
            f"pitch_down={self.pitch_down}, exempt_forward={self.exempt_forward_looking}"
        )

    def _get_keypoint(self, landmarks: Any, keypoint_name: str) -> Optional[Any]:
        """Get a keypoint from landmarks by name.

        Thin wrapper around the canonical ``get_keypoint`` in
        ``app.core.utils.pose_utils`` that returns ``None`` instead of
        raising on unknown/missing keypoints (preserving this detector's
        existing error-handling contract).

        Args:
            landmarks: Pose landmarks object with .landmark attribute
            keypoint_name: Name of keypoint (e.g., 'nose', 'left_shoulder')

        Returns:
            Landmark object with x, y, visibility attributes, or None if not found
        """
        if landmarks is None:
            return None

        try:
            return _canonical_get_keypoint(landmarks, keypoint_name)
        except (ValueError, IndexError, AttributeError):
            return None

    def calculate_head_pose_angles(
        self,
        pose_landmarks: Any,
        face_landmarks: Any,
        frame_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Calculate head pose angles (yaw, pitch) to detect mind diversion.

        Detects three types of mind diversion:
        1. looking_sideways - head turned > threshold (configurable)
        2. looking_away_combined - head turned > threshold AND down > threshold
        3. looking_down_distracted - head down > threshold (configurable)

        Uses both pose landmarks (nose, shoulders) and face mesh landmarks for accuracy.

        Args:
            pose_landmarks: Pose landmarks (YoloPoseLandmarks or MediaPipe)
            face_landmarks: MediaPipe face mesh landmarks (can be None)
            frame_shape: (height, width) or (height, width, channels) of frame

        Returns:
            dict: {
                'yaw': float,       # Side turn angle in degrees (-90 to +90)
                'pitch': float,     # Up/down tilt angle in degrees (-90 to +90)
                'detected': bool,   # True if mind diversion detected
                'sub_type': str,    # 'looking_sideways'|'looking_down_distracted'|'looking_away_combined'|None
                'method': str       # Detection method used ('pose_landmarks'|'face_mesh'|'ear_asymmetry'|'none')
            }
        """
        h, w = frame_shape[:2]
        result = {'yaw': 0, 'pitch': 0, 'detected': False, 'sub_type': None, 'method': 'none'}

        if not pose_landmarks:
            return result

        try:
            # Get pose landmarks
            nose = self._get_keypoint(pose_landmarks, 'nose')
            left_shoulder = self._get_keypoint(pose_landmarks, 'left_shoulder')
            right_shoulder = self._get_keypoint(pose_landmarks, 'right_shoulder')
            left_ear = self._get_keypoint(pose_landmarks, 'left_ear')
            right_ear = self._get_keypoint(pose_landmarks, 'right_ear')

            # Check visibility
            if not nose or nose.visibility < 0.5:
                # FALLBACK: When nose not visible, use ear asymmetry for yaw estimation
                result = self._estimate_yaw_from_ear_asymmetry(
                    left_ear, right_ear, result
                )
                return result

            # Validate shoulder and ear visibility before using them
            min_shoulder_vis = 0.3
            min_ear_vis = 0.3
            if (not left_shoulder or getattr(left_shoulder, 'visibility', 0) < min_shoulder_vis or
                    not right_shoulder or getattr(right_shoulder, 'visibility', 0) < min_shoulder_vis):
                return result  # Cannot compute angles without visible shoulders

            if (not left_ear or getattr(left_ear, 'visibility', 0) < min_ear_vis or
                    not right_ear or getattr(right_ear, 'visibility', 0) < min_ear_vis):
                # Ears not visible -- fall back to ear asymmetry estimation
                result = self._estimate_yaw_from_ear_asymmetry(
                    left_ear, right_ear, result
                )
                return result

            # Convert to pixel coordinates
            nose_coords = np.array([nose.x * w, nose.y * h])
            left_shoulder_coords = np.array([left_shoulder.x * w, left_shoulder.y * h])
            right_shoulder_coords = np.array([right_shoulder.x * w, right_shoulder.y * h])
            left_ear_coords = np.array([left_ear.x * w, left_ear.y * h])
            right_ear_coords = np.array([right_ear.x * w, right_ear.y * h])

            # Calculate shoulder midpoint
            shoulder_midpoint = (left_shoulder_coords + right_shoulder_coords) / 2
            shoulder_width = np.linalg.norm(right_shoulder_coords - left_shoulder_coords)

            # METHOD 1: Calculate YAW (side turning) using nose offset from shoulder midpoint
            nose_offset_x = nose_coords[0] - shoulder_midpoint[0]

            # Normalize by shoulder width and convert to angle
            # Positive = turned right, Negative = turned left
            yaw_normalized = nose_offset_x / (shoulder_width / 2) if shoulder_width > 0 else 0
            yaw_angle = np.clip(yaw_normalized * 45, -90, 90)  # Scale to degrees

            # METHOD 2: Calculate PITCH (up/down tilt) using nose position relative to ears
            ear_midpoint = (left_ear_coords + right_ear_coords) / 2
            nose_offset_y = nose_coords[1] - ear_midpoint[1]

            # Normalize by head size (ear-to-nose distance) and convert to angle
            # Positive = looking down, Negative = looking up
            head_height = shoulder_midpoint[1] - ear_midpoint[1]
            if head_height > 0:
                pitch_normalized = nose_offset_y / head_height
                pitch_angle = np.clip(pitch_normalized * 30, -45, 45)  # Scale to degrees
            else:
                pitch_angle = 0

            result['yaw'] = yaw_angle
            result['pitch'] = pitch_angle
            result['method'] = 'pose_landmarks'

            # Apply detection logic
            self._apply_detection_logic(result, yaw_angle, pitch_angle)

            # Use face mesh if available for more accurate detection
            if face_landmarks and hasattr(face_landmarks, 'multi_face_landmarks') and face_landmarks.multi_face_landmarks:
                self._refine_with_face_mesh(result, face_landmarks, frame_shape)

            return result

        except (IndexError, AttributeError, ZeroDivisionError) as e:
            self.logger.debug(f"Exception in calculate_head_pose_angles: {e}")
            return {'yaw': 0, 'pitch': 0, 'detected': False, 'sub_type': None, 'method': 'error'}

    def _estimate_yaw_from_ear_asymmetry(
        self,
        left_ear: Optional[Any],
        right_ear: Optional[Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate yaw angle from ear visibility asymmetry.

        When the nose is not visible, ear asymmetry can indicate head turn:
        - Right ear hidden = turned right (looking away from track)
        - Left ear hidden = turned left (looking toward track - legitimate)

        Args:
            left_ear: Left ear landmark (may be None)
            right_ear: Right ear landmark (may be None)
            result: Result dictionary to update

        Returns:
            Updated result dictionary
        """
        left_ear_vis = left_ear.visibility if left_ear else 0
        right_ear_vis = right_ear.visibility if right_ear else 0

        if left_ear_vis > 0.5 and right_ear_vis < 0.3:
            # Right ear hidden = turned right (looking away from track)
            yaw_angle = 60  # Estimate significant right turn
            result['yaw'] = yaw_angle
            result['method'] = 'ear_asymmetry'

            # Positive yaw (looking right/away) - always check threshold
            if yaw_angle > self.yaw_sideways:
                result['detected'] = True
                result['sub_type'] = 'looking_sideways'

        elif right_ear_vis > 0.5 and left_ear_vis < 0.3:
            # Left ear hidden = turned left (looking toward track - LEGITIMATE)
            yaw_angle = -60  # Estimate significant left turn
            result['yaw'] = yaw_angle
            result['method'] = 'ear_asymmetry'

            # Negative yaw (looking left/forward toward track)
            # Only trigger if forward-looking exemption is disabled
            if not self.exempt_forward_looking and abs(yaw_angle) > self.yaw_sideways:
                result['detected'] = True
                result['sub_type'] = 'looking_sideways'
            # If exempt_forward_looking is True, negative yaw does NOT trigger detection

        return result

    def _apply_detection_logic(
        self,
        result: Dict[str, Any],
        yaw_angle: float,
        pitch_angle: float
    ) -> None:
        """Apply multi-scenario mind diversion detection logic.

        Sub-type detection with priority order:
        1. looking_sideways - head turned significantly to side (HIGH CONFIDENCE)
        2. looking_away_combined - head turned AND down (HIGH CONFIDENCE)
        3. looking_down_distracted - only head down, not sideways (MEDIUM CONFIDENCE)

        Args:
            result: Result dictionary to update in-place
            yaw_angle: Calculated yaw angle in degrees
            pitch_angle: Calculated pitch angle in degrees
        """
        sub_type = None

        # Forward-looking exemption: Camera is behind-right of crew
        # - Negative yaw = looking LEFT toward track/window (LEGITIMATE WORK)
        # - Positive yaw = looking RIGHT away from track (POTENTIAL DIVERSION)
        # When exempt_forward_looking is True, only positive yaw triggers detection
        if self.exempt_forward_looking:
            effective_yaw = yaw_angle if yaw_angle > 0 else 0  # Only positive yaw triggers
        else:
            effective_yaw = abs(yaw_angle)  # Both directions trigger (original behavior)

        # Scenario 1: looking_sideways (head turned > threshold, regardless of pitch)
        if effective_yaw > self.yaw_sideways:
            sub_type = 'looking_sideways'
            result['detected'] = True
        # Scenario 2: looking_away_combined (turned AND down)
        elif effective_yaw > self.yaw_combined and pitch_angle > self.pitch_combined:
            sub_type = 'looking_away_combined'
            result['detected'] = True
        # Scenario 3: looking_down_distracted (only down, not sideways)
        elif pitch_angle > self.pitch_down and effective_yaw < self.yaw_max_for_down:
            sub_type = 'looking_down_distracted'
            result['detected'] = True

        result['sub_type'] = sub_type

    def _refine_with_face_mesh(
        self,
        result: Dict[str, Any],
        face_landmarks: Any,
        frame_shape: Tuple[int, ...]
    ) -> None:
        """Refine head pose angles using face mesh landmarks.

        Face mesh provides more accurate head pose estimation than pose
        landmarks alone. This method recalculates yaw and pitch using
        facial feature points and re-evaluates detection thresholds.

        Args:
            result: Result dictionary to update in-place
            face_landmarks: MediaPipe face mesh result
            frame_shape: (height, width) of frame
        """
        try:
            h, w = frame_shape[:2]
            # Use first detected face
            face_lm = face_landmarks.multi_face_landmarks[0].landmark

            # Bounds check: MediaPipe face mesh should have 468 landmarks.
            # The highest index we access is 454; verify the list is large enough.
            required_landmark_count = 468
            if len(face_lm) < required_landmark_count:
                self.logger.debug(
                    f"Face mesh has only {len(face_lm)} landmarks "
                    f"(need {required_landmark_count}), skipping refinement"
                )
                return

            # Key face mesh landmarks for 3D pose estimation
            nose_tip = face_lm[1]  # Nose tip
            left_face_edge = face_lm[234]  # Left face edge
            right_face_edge = face_lm[454]  # Right face edge
            left_eye = face_lm[33]  # Left eye outer corner
            right_eye = face_lm[263]  # Right eye outer corner

            # Convert to pixel coordinates
            nose_tip_coords = np.array([nose_tip.x * w, nose_tip.y * h])
            left_edge_coords = np.array([left_face_edge.x * w, left_face_edge.y * h])
            right_edge_coords = np.array([right_face_edge.x * w, right_face_edge.y * h])
            left_eye_coords = np.array([left_eye.x * w, left_eye.y * h])
            right_eye_coords = np.array([right_eye.x * w, right_eye.y * h])

            # Calculate face width and nose offset for YAW
            face_width = np.linalg.norm(right_edge_coords - left_edge_coords)
            face_center_x = (left_edge_coords[0] + right_edge_coords[0]) / 2
            nose_offset_x_face = nose_tip_coords[0] - face_center_x

            # YAW angle from face mesh (more accurate)
            if face_width > 0:
                yaw_face = (nose_offset_x_face / (face_width / 2)) * 60  # Scale to degrees
                result['yaw'] = np.clip(yaw_face, -90, 90)

            # Calculate PITCH using nose tip and eye line
            eye_midpoint = (left_eye_coords + right_eye_coords) / 2
            nose_to_eye_dist = np.linalg.norm(nose_tip_coords - eye_midpoint)
            nose_below_eyes = nose_tip_coords[1] - eye_midpoint[1]

            # PITCH angle from face mesh
            if nose_to_eye_dist > 0:
                pitch_face = (nose_below_eyes / nose_to_eye_dist) * 45
                result['pitch'] = np.clip(pitch_face, -45, 45)

            result['method'] = 'face_mesh'

            # Re-evaluate detection with face mesh data
            yaw_fm = result['yaw']
            pitch_fm = result['pitch']

            # Forward-looking exemption (same logic as pose-based detection)
            if self.exempt_forward_looking:
                effective_yaw_fm = yaw_fm if yaw_fm > 0 else 0
            else:
                effective_yaw_fm = abs(yaw_fm)

            # Reset and re-evaluate
            sub_type = None
            result['detected'] = False

            # Scenario 1: looking_sideways
            if effective_yaw_fm > self.yaw_sideways:
                sub_type = 'looking_sideways'
                result['detected'] = True
            # Scenario 2: looking_away_combined
            elif effective_yaw_fm > self.yaw_combined and pitch_fm > self.pitch_combined:
                sub_type = 'looking_away_combined'
                result['detected'] = True
            # Scenario 3: looking_down_distracted
            elif pitch_fm > self.pitch_down and effective_yaw_fm < self.yaw_max_for_down:
                sub_type = 'looking_down_distracted'
                result['detected'] = True

            result['sub_type'] = sub_type

        except Exception as e:
            # If face mesh processing fails, keep pose-based result
            self.logger.debug(f"Exception in _refine_with_face_mesh: {e}")

    def detect_mind_diversion(
        self,
        pose_landmarks: Any,
        face_landmarks: Any,
        frame_shape: Tuple[int, ...],
        writing_active: bool = False,
        person_idx: int = 0,
        current_time: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Detect mind diversion with optional writing suppression.

        This is the main entry point for mind diversion detection. It
        calculates head pose angles and applies suppression logic for
        legitimate work activities.

        Args:
            pose_landmarks: Pose landmarks object
            face_landmarks: Optional face mesh landmarks
            frame_shape: (height, width) of frame
            writing_active: Whether writing activity is currently detected
            person_idx: Index of the person being checked
            current_time: Current timestamp for grace period checks

        Returns:
            Tuple of (detected, sub_type, debug_info):
                - detected: True if mind diversion detected (not suppressed)
                - sub_type: Detection sub-type or None
                - debug_info: Dictionary with angles, method, and suppression info
        """
        # Calculate head pose angles
        head_pose = self.calculate_head_pose_angles(
            pose_landmarks, face_landmarks, frame_shape
        )

        debug_info = {
            'yaw': head_pose['yaw'],
            'pitch': head_pose['pitch'],
            'method': head_pose['method'],
            'raw_detected': head_pose['detected'],
            'raw_sub_type': head_pose['sub_type'],
            'suppressed': False,
            'suppression_reason': None
        }

        # Check suppression if detection was positive
        if head_pose['detected'] and self.suppress_with_writing:
            should_suppress, reason = self._check_writing_suppression(
                writing_active, person_idx, current_time
            )
            if should_suppress:
                debug_info['suppressed'] = True
                debug_info['suppression_reason'] = reason
                return False, None, debug_info

        return head_pose['detected'], head_pose['sub_type'], debug_info

    def _check_writing_suppression(
        self,
        writing_active: bool,
        person_idx: int,
        current_time: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """Check if mind diversion should be suppressed due to writing activity.

        Args:
            writing_active: Whether writing is currently active
            person_idx: Person index for tracking recent activities
            current_time: Current timestamp

        Returns:
            Tuple of (should_suppress, reason)
        """
        # Active writing suppression
        if writing_active:
            # Update recent activity timestamp
            if current_time is not None:
                if person_idx not in self._recent_person_activities:
                    self._recent_person_activities[person_idx] = {}
                self._recent_person_activities[person_idx]['writing'] = current_time
            return True, "suppressed_writing_active"

        # Recent writing grace period suppression
        if current_time is not None and person_idx in self._recent_person_activities:
            writing_timestamp = self._recent_person_activities[person_idx].get('writing')
            if writing_timestamp and (current_time - writing_timestamp) < self.writing_grace_seconds:
                return True, "suppressed_recent_writing"

        return False, None

    def check_forward_looking_exemption(self, yaw: float) -> bool:
        """Check if the given yaw angle qualifies for forward-looking exemption.

        Camera is behind-right of crew, so negative yaw (looking left toward
        the track) is considered legitimate forward-looking behavior.

        Args:
            yaw: Yaw angle in degrees (negative = left, positive = right)

        Returns:
            True if the yaw angle is exempt from diversion detection
        """
        if not self.exempt_forward_looking:
            return False
        return yaw < 0  # Negative yaw = looking toward track = exempt

    def should_suppress_mind_diversion(
        self,
        person_idx: int,
        person_activities: Dict[str, Any],
        pose_landmarks: Any,
        detections: Dict[str, List[Any]],
        frame_shape: Tuple[int, ...],
        current_time: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Suppress mind diversion if person is doing legitimate work activity.

        This function checks multiple conditions to prevent false positives when
        the LP is legitimately working on documents (logbook, papers, etc.).

        Args:
            person_idx: Index of the person being checked
            person_activities: Dict of detected activities for this person
            pose_landmarks: Pose landmarks for the person
            detections: YOLO detections dict (may contain 'book', etc.)
            frame_shape: (height, width) of the frame
            current_time: Current timestamp (optional, for recent activity check)

        Returns:
            tuple: (should_suppress: bool, reason: str or None)
        """
        h, w = frame_shape[:2]

        # 1. WRITING ACTIVITY SUPPRESSION
        if self.suppress_with_writing:
            if person_activities.get('writing', False):
                return True, "suppressed_writing_active"

            # Check recent writing (within grace period)
            if current_time is not None and person_idx in self._recent_person_activities:
                writing_timestamp = self._recent_person_activities[person_idx].get('writing')
                if writing_timestamp and (current_time - writing_timestamp) < self.writing_grace_seconds:
                    return True, "suppressed_recent_writing"

        # 2. BOOK DETECTION SUPPRESSION
        if detections and 'book' in detections and len(detections.get('book', [])) > 0:
            return True, "suppressed_book_detected"

        # 3. HAND POSITION HEURISTIC (Critical for camera angle)
        # If both wrists visible and close together in lap area -> likely document work
        if pose_landmarks:
            try:
                left_wrist = self._get_keypoint(pose_landmarks, 'left_wrist')
                right_wrist = self._get_keypoint(pose_landmarks, 'right_wrist')
                nose = self._get_keypoint(pose_landmarks, 'nose')

                if (left_wrist and right_wrist and nose and
                    left_wrist.visibility > 0.3 and right_wrist.visibility > 0.3):
                    # Calculate wrist positions
                    left_wrist_coords = np.array([left_wrist.x * w, left_wrist.y * h])
                    right_wrist_coords = np.array([right_wrist.x * w, right_wrist.y * h])
                    wrist_distance = np.linalg.norm(left_wrist_coords - right_wrist_coords)

                    # Check if wrists are in "lap area" (below nose, in front of body)
                    nose_y = nose.y * h
                    avg_wrist_y = (left_wrist_coords[1] + right_wrist_coords[1]) / 2
                    wrists_below_face = avg_wrist_y > nose_y

                    # If wrists close together AND below face -> writing pose
                    if wrist_distance < self.wrist_distance_threshold and wrists_below_face:
                        return True, "suppressed_writing_pose_detected"
            except (AttributeError, IndexError):
                pass  # Landmarks not available, continue without suppression

        # 4. NO SUPPRESSION - Allow detection
        return False, None

    def update_activity_timestamp(
        self,
        person_idx: int,
        activity_type: str,
        timestamp: float
    ) -> None:
        """Update the timestamp for a person's activity.

        Used to track recent activities for suppression grace periods.

        Args:
            person_idx: Person index
            activity_type: Type of activity (e.g., 'writing')
            timestamp: Timestamp of the activity
        """
        if person_idx not in self._recent_person_activities:
            self._recent_person_activities[person_idx] = {}
        self._recent_person_activities[person_idx][activity_type] = timestamp

    def clear_activity_history(self, person_idx: Optional[int] = None) -> None:
        """Clear activity history for a person or all persons.

        Args:
            person_idx: Person index to clear, or None to clear all
        """
        if person_idx is not None:
            self._recent_person_activities.pop(person_idx, None)
        else:
            self._recent_person_activities.clear()

    def reset(self) -> None:
        """Clear all per-video state.

        Wipes ``_recent_person_activities``, the only state-holding dict on
        this detector, so the next video begins with no carryover writing
        grace-period entries from the previous video.
        """
        self._recent_person_activities.clear()

    def on_suppressed(self, person_idx: Optional[int], activity_name: str) -> None:
        """Hook invoked when an activity for a person is suppressed by the
        train-stopped gate.

        For ``mind_diversion`` we drop the per-person writing-grace cache
        entry so a writing event suppressed during the STOPPED window does
        not extend its grace period into the resume window.

        Args:
            person_idx: Per-person index whose state should be cleared.
            activity_name: Suppressed activity key (only acts on
                ``'mind_diversion'`` and ``'writing'``).
        """
        if person_idx is None:
            return
        if activity_name in ('mind_diversion', 'writing'):
            self._recent_person_activities.pop(person_idx, None)
