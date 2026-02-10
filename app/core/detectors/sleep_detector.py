"""Sleep and microsleep detection using pose analysis.

This module extracts sleep detection logic from locopilot_monitor.py,
providing a reusable SleepDetector class for detecting sleep, microsleep,
and drowsiness states using pose landmarks and eye analysis.

Detection Methods:
1. Pose-based sleep detection (head drop, body posture analysis)
2. IR forward-lean detection (for dark/IR frames)
3. Haar cascade eye closure detection

State Machine:
ALERT -> DROWSY -> MICROSLEEP -> SLEEP
With LOOKING_DOWN_WORKING as a parallel state for active work detection.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
import logging
import numpy as np
import cv2


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
    'right_ankle': 16
}


class SleepDetector:
    """Detects sleep and microsleep using pose landmarks and eye analysis.

    This class provides comprehensive sleep detection through multiple methods:
    - Pose-based detection using head position, body movement, and posture
    - IR forward-lean detection for dark/IR camera conditions
    - Haar cascade eye closure detection

    The detector maintains per-person tracking state for multi-person scenarios
    and uses a temporal state machine for robust detection.

    Attributes:
        settings: Configuration settings object (optional)
        sample_fps: Frame sampling rate for history buffer sizing
        per_person_tracking: Per-person sleep tracking state
        ir_forward_lean_tracking: Per-person IR detection tracking state
    """

    # YOLO keypoint indices
    YOLO_HEAD_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
    YOLO_BODY_INDICES = [5, 6, 7, 8, 11, 12]  # left/right shoulders, elbows, hips
    YOLO_MIN_KEYPOINTS = 13  # Minimum landmarks required (indices 0-12)

    def __init__(
        self,
        settings: Optional[Any] = None,
        eye_cascade_path: Optional[str] = None,
        sample_fps: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the SleepDetector.

        Args:
            settings: Configuration settings object with sleep detection thresholds.
                     If None, uses default values.
            eye_cascade_path: Path to Haar cascade XML file for eye detection.
                            If None, uses OpenCV's default haarcascade_eye.xml.
            sample_fps: Frame sampling rate (frames per second) for sizing
                       history buffers. Default 0.5 (1 frame every 2 seconds).
            logger: Optional logger instance. If None, creates a new one.
        """
        self.settings = settings
        self.sample_fps = sample_fps
        self.logger = logger or logging.getLogger(__name__)

        # Per-person tracking dictionaries
        self.per_person_tracking: Dict[int, Dict[str, Any]] = defaultdict(self._create_tracking_dict)
        self.ir_forward_lean_tracking: Dict[int, Dict[str, Any]] = defaultdict(dict)

        # Initialize thresholds from settings or defaults
        self._init_thresholds()

        # Load Haar cascades for eye detection
        self._load_haar_cascades(eye_cascade_path)

    def _init_thresholds(self) -> None:
        """Initialize detection thresholds from settings or use defaults."""
        s = self.settings

        # Sleep score and duration thresholds
        self.SLEEP_STRONG_SCORE = getattr(s, 'sleep_strong_score', 4) if s else 4
        self.SLEEP_STRONG_DURATION = getattr(s, 'sleep_strong_duration', 2) if s else 2
        self.SLEEP_MODERATE_DURATION = getattr(s, 'sleep_moderate_duration', 4) if s else 4
        self.SLEEP_MICROSLEEP_DURATION = getattr(s, 'sleep_microsleep_duration', 2) if s else 2

        # Movement and posture thresholds
        self.MINIMAL_MOVEMENT_THRESHOLD = getattr(s, 'minimal_movement_threshold', 0.15) if s else 0.15
        self.STABLE_POSTURE_VARIANCE = getattr(s, 'stable_posture_variance', 100) if s else 100
        self.EYES_NOT_VISIBLE_THRESHOLD = getattr(s, 'eyes_not_visible_threshold', 0.4) if s else 0.4

        # Baseline calibration
        self.SLEEP_BASELINE_ENABLED = getattr(s, 'sleep_baseline_enabled', True) if s else True
        self.SLEEP_BASELINE_CALIBRATION_WINDOW = getattr(s, 'sleep_baseline_calibration_window', 10.0) if s else 10.0
        self.SLEEP_BASELINE_MIN_SAMPLES = getattr(s, 'sleep_baseline_min_samples', 5) if s else 5

        # Baseline delta thresholds
        self.SLEEP_BASELINE_NOSE_BELOW_DELTA = getattr(s, 'sleep_baseline_nose_below_delta', 40) if s else 40
        self.SLEEP_BASELINE_HEAD_TILT_DELTA = getattr(s, 'sleep_baseline_head_tilt_delta', 25) if s else 25
        self.SLEEP_BASELINE_TORSO_HEIGHT_DELTA = getattr(s, 'sleep_baseline_torso_height_delta', 40) if s else 40
        self.SLEEP_BASELINE_SHOULDER_WIDTH_DELTA = getattr(s, 'sleep_baseline_shoulder_width_delta', 20) if s else 20

        # Sustained signal thresholds
        self.SLEEP_SUSTAINED_STILLNESS_THRESHOLD = getattr(s, 'sleep_sustained_stillness_threshold', 0.02) if s else 0.02
        self.SLEEP_SUSTAINED_STILLNESS_FRAMES = getattr(s, 'sleep_sustained_stillness_frames', 3) if s else 3
        self.SLEEP_HANDS_CLASPED_THRESHOLD = getattr(s, 'sleep_hands_clasped_threshold', 100) if s else 100
        self.SLEEP_HANDS_CLASPED_FRAMES = getattr(s, 'sleep_hands_clasped_frames', 3) if s else 3
        self.SLEEP_SUSTAINED_LOW_EYE_FRAMES = getattr(s, 'sleep_sustained_low_eye_frames', 3) if s else 3
        self.SLEEP_HANDS_SPREAD_THRESHOLD = getattr(s, 'sleep_hands_spread_threshold', 180) if s else 180

        # Head bob detection
        self.SLEEP_HEAD_BOB_DRIFT_MAX_RATE = getattr(s, 'sleep_head_bob_drift_max_rate', 15.0) if s else 15.0
        self.SLEEP_HEAD_BOB_JERK_MIN_RATE = getattr(s, 'sleep_head_bob_jerk_min_rate', 20.0) if s else 20.0
        self.SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES = getattr(s, 'sleep_head_bob_min_drift_frames', 2) if s else 2
        self.SLEEP_HEAD_BOB_MIN_AMPLITUDE = getattr(s, 'sleep_head_bob_min_amplitude', 10.0) if s else 10.0
        self.SLEEP_HEAD_BOB_SCORE_BONUS = getattr(s, 'sleep_head_bob_score_bonus', 2) if s else 2
        self.SLEEP_HEAD_BOB_BYPASS_EYE_GATE = getattr(s, 'sleep_head_bob_bypass_eye_gate', True) if s else True

        # Wrist velocity thresholds — Fix 11: scale by sqrt(fps_ratio) for low-fps sampling.
        # At 0.5fps (2s between frames), YOLO keypoint jitter exceeds the reference 30fps threshold.
        base_wrist_vel_still = getattr(s, 'sleep_wrist_velocity_still_threshold', 0.005) if s else 0.005
        base_wrist_vel_active = getattr(s, 'sleep_wrist_velocity_active_threshold', 0.03) if s else 0.03
        frame_interval = 1.0 / max(0.1, self.sample_fps)
        reference_interval = 1.0 / 30.0
        fps_scale = max(1.0, (frame_interval / reference_interval) ** 0.5)
        self.SLEEP_WRIST_VEL_STILL = base_wrist_vel_still * fps_scale
        self.SLEEP_WRIST_VEL_ACTIVE = base_wrist_vel_active * fps_scale
        self.SLEEP_WRIST_VEL_STILL_FRAMES = getattr(s, 'sleep_wrist_velocity_still_frames', 2) if s else 2

        # State machine thresholds
        self.SLEEP_STATE_MACHINE_ENABLED = getattr(s, 'sleep_state_machine_enabled', True) if s else True
        self.SLEEP_STATE_HAND_ACTIVITY_THRESHOLD = getattr(s, 'sleep_state_hand_activity_threshold', 0.02) if s else 0.02
        self.SLEEP_DROWSY_TO_MICROSLEEP_SEC = getattr(s, 'sleep_state_drowsy_to_microsleep_sec', 2.0) if s else 2.0
        self.SLEEP_MICROSLEEP_TO_SLEEP_SEC = getattr(s, 'sleep_state_microsleep_to_sleep_sec', 4.0) if s else 4.0

        # Shoulder slump detection
        self.SLEEP_SHOULDER_SLUMP_RATE_THRESHOLD = getattr(s, 'sleep_shoulder_slump_rate_threshold', 0.005) if s else 0.005
        self.SLEEP_SHOULDER_SLUMP_MIN_FRAMES = getattr(s, 'sleep_shoulder_slump_min_frames', 3) if s else 3

        # IR forward lean thresholds
        self.IR_SHOULDER_RELATIVE_THRESHOLD = getattr(s, 'ir_shoulder_relative_threshold', 0.4) if s else 0.4
        self.IR_BBOX_ASPECT_RATIO_THRESHOLD = getattr(s, 'ir_bbox_aspect_ratio_threshold', 1.2) if s else 1.2
        self.IR_LOW_MOVEMENT_THRESHOLD = getattr(s, 'ir_low_movement_threshold', 0.02) if s else 0.02
        self.SUB_THRESHOLD_STREAK_LIMIT = getattr(s, 'sub_threshold_streak_limit', 3) if s else 3

        # Visibility thresholds
        self.WRIST_VISIBILITY_THRESHOLD = getattr(s, 'wrist_visibility_threshold', 0.3) if s else 0.3
        self.ELBOW_VISIBILITY_THRESHOLD = getattr(s, 'elbow_visibility_threshold', 0.25) if s else 0.25
        self.MIN_POSE_LANDMARKS = getattr(s, 'min_pose_landmarks', 10) if s else 10
        self.MIN_POSE_VISIBILITY = getattr(s, 'min_pose_visibility', 0.3) if s else 0.3

    def _load_haar_cascades(self, eye_cascade_path: Optional[str]) -> None:
        """Load Haar cascade classifiers for face and eye detection."""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.profile_face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            if eye_cascade_path:
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            else:
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'
                )
            self.logger.debug("Haar cascades loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load Haar cascades: {e}")
            self.face_cascade = None
            self.profile_face_cascade = None
            self.eye_cascade = None

    def _create_tracking_dict(self) -> Dict[str, Any]:
        """Create a new per-person tracking dictionary with all required fields."""
        return {
            'pose_sleep_start': None,
            'pose_sleep_duration': 0,
            'previous_landmarks': None,
            'movement_history': deque(maxlen=int(30 * self.sample_fps)),
            'head_tilt_history': deque(maxlen=int(10 * self.sample_fps)),
            'nose_distance_history': deque(maxlen=int(10 * self.sample_fps)),
            'torso_height_history': deque(maxlen=int(10 * self.sample_fps)),
            'nose_y_norm_history': deque(maxlen=int(10 * self.sample_fps)),
            'nose_above_shoulders_history': deque(maxlen=int(10 * self.sample_fps)),
            # Baseline calibration
            'baseline_calibrating': self.SLEEP_BASELINE_ENABLED,
            'baseline_start_time': None,
            'baseline_samples': {
                'nose_below_px': [], 'head_tilt': [], 'torso_height_px': [],
                'shoulder_width_px': [], 'nose_above_shoulders_norm': [],
                'nose_y_normalized': [], 'movement': [], 'eye_vis': [],
            },
            'baseline': None,  # Dict of medians once calibration completes
            # Sustained-signal counters
            'sustained_stillness_count': 0,
            'hands_clasped_count': 0,
            'low_eye_vis_count': 0,
            'wrist_distance_history': deque(maxlen=int(10 * self.sample_fps)),
            'face_not_visible_count': 0,
            # Head bob
            'head_tilt_deltas': deque(maxlen=int(10 * self.sample_fps)),
            'head_bob_count': 0,
            # Wrist velocity
            'previous_wrist_positions': None,
            'wrist_velocity_history': deque(maxlen=int(10 * self.sample_fps)),
            # State machine
            'sleep_state': 'ALERT',
            'state_enter_time': None,
            'state_history': deque(maxlen=10),
            # Shoulder slump
            'shoulder_y_history': deque(maxlen=int(10 * self.sample_fps)),
            'shoulder_y_timestamps': deque(maxlen=int(10 * self.sample_fps)),
            # Haar cascade eye closure tracking
            'haar_eyes_closed_start': None,
            'haar_eyes_closed_count': 0,
            'haar_eyes_open_count': 0,
            'haar_face_detected_in_roi': False,
        }

    def _get_per_person_sleep_tracking(self, person_idx: int) -> Dict[str, Any]:
        """Get or initialize per-person sleep tracking state.

        Args:
            person_idx: Person index for tracking

        Returns:
            Dict containing all tracking state for this person
        """
        if person_idx not in self.per_person_tracking:
            self.per_person_tracking[person_idx] = self._create_tracking_dict()
        return self.per_person_tracking[person_idx]

    def _get_ir_forward_lean_tracking(self, person_idx: int) -> Dict[str, Any]:
        """Get or initialize per-person IR forward-lean sleep tracking state.

        Args:
            person_idx: Person index for tracking

        Returns:
            Dict containing IR forward lean tracking state for this person
        """
        if person_idx not in self.ir_forward_lean_tracking:
            self.ir_forward_lean_tracking[person_idx] = {
                'start_time': None,
                'previous_body_keypoints': None,
                'sub_threshold_streak': 0,  # Consecutive frames below score threshold
            }
        return self.ir_forward_lean_tracking[person_idx]

    def get_keypoint(self, landmarks: Any, keypoint_name: str) -> Any:
        """Get a keypoint from landmarks by name.

        Args:
            landmarks: Landmark list (either .landmark attribute or direct list)
            keypoint_name: String name like 'nose', 'left_wrist', etc.

        Returns:
            Landmark object with x, y, z, visibility attributes
        """
        # Support both YoloPoseLandmarks (has .landmark) and plain list
        landmark_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks

        name_lower = keypoint_name.lower()

        if name_lower in YOLO_KEYPOINT_INDICES:
            idx = YOLO_KEYPOINT_INDICES[name_lower]
            return landmark_list[idx]

        # Handle MediaPipe-specific keypoints that don't exist in YOLO
        fallback_map = {
            'left_index': 'left_wrist',
            'right_index': 'right_wrist',
            'left_pinky': 'left_wrist',
            'right_pinky': 'right_wrist',
            'left_thumb': 'left_wrist',
            'right_thumb': 'right_wrist',
        }

        if name_lower in fallback_map:
            fallback_name = fallback_map[name_lower]
            idx = YOLO_KEYPOINT_INDICES[fallback_name]
            return landmark_list[idx]

        raise ValueError(f"Unknown keypoint: {keypoint_name}")

    def validate_pose_landmarks(
        self,
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
            min_landmarks = self.MIN_POSE_LANDMARKS
        if min_visibility is None:
            min_visibility = self.MIN_POSE_VISIBILITY
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

    def calculate_head_tilt_angle(self, landmarks: Any) -> Optional[float]:
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
            nose = self.get_keypoint(landmarks, 'nose')
            left_shoulder = self.get_keypoint(landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(landmarks, 'right_shoulder')

            # Calculate neck position (midpoint between shoulders)
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2

            # Calculate angle from vertical
            delta_y = nose.y - neck_y
            delta_x = nose.x - neck_x

            # Negative angle = head tilted forward/down
            angle = np.arctan2(delta_y, delta_x) * 180 / np.pi - 90

            return angle

        except Exception as e:
            self.logger.debug(f"Exception in calculate_head_tilt_angle: {e}")
            return None

    def calculate_movement_score(
        self,
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
                curr = self.get_keypoint(current_landmarks, landmark_name)
                prev = self.get_keypoint(previous_landmarks, landmark_name)

                distance = np.sqrt(
                    (curr.x - prev.x) ** 2 +
                    (curr.y - prev.y) ** 2
                )
                total_movement += distance

            movement_score = total_movement / len(key_landmark_names)
            return movement_score

        except Exception as e:
            self.logger.debug(f"Exception in calculate_movement_score: {e}")
            return 0.0

    def calculate_wrist_distance(
        self,
        pose_landmarks: Any,
        frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[float], Optional[str]]:
        """Calculate Euclidean distance between left and right wrists.

        Falls back to elbow distance or single wrist-to-shoulder if wrists
        are not both visible.

        Args:
            pose_landmarks: Pose landmarks object
            frame_shape: Tuple of (height, width) of the frame

        Returns:
            tuple: (distance in pixels, source) where source is 'wrist', 'elbow',
                   'single_wrist', or (None, None) if not detectable
        """
        if not pose_landmarks:
            return None, None

        if not self.validate_pose_landmarks(pose_landmarks):
            return None, None

        try:
            landmarks = pose_landmarks.landmark if hasattr(pose_landmarks, 'landmark') else pose_landmarks
            h, w = frame_shape[:2]

            right_wrist = self.get_keypoint(landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')

            # Try wrists first
            if (right_wrist.visibility >= self.WRIST_VISIBILITY_THRESHOLD and
                    left_wrist.visibility >= self.WRIST_VISIBILITY_THRESHOLD):
                right_wrist_px = (right_wrist.x * w, right_wrist.y * h)
                left_wrist_px = (left_wrist.x * w, left_wrist.y * h)

                distance = np.sqrt(
                    (right_wrist_px[0] - left_wrist_px[0])**2 +
                    (right_wrist_px[1] - left_wrist_px[1])**2
                )
                return distance, 'wrist'

            # Fallback: elbows
            right_elbow = self.get_keypoint(landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(landmarks, 'left_elbow')

            if (right_elbow.visibility >= self.ELBOW_VISIBILITY_THRESHOLD and
                    left_elbow.visibility >= self.ELBOW_VISIBILITY_THRESHOLD):
                right_elbow_px = (right_elbow.x * w, right_elbow.y * h)
                left_elbow_px = (left_elbow.x * w, left_elbow.y * h)
                distance = np.sqrt(
                    (right_elbow_px[0] - left_elbow_px[0])**2 +
                    (right_elbow_px[1] - left_elbow_px[1])**2
                )
                return distance, 'elbow'

            # Fallback: single wrist to shoulder midpoint
            right_shoulder = self.get_keypoint(landmarks, 'right_shoulder')
            left_shoulder = self.get_keypoint(landmarks, 'left_shoulder')
            SINGLE_WRIST_VIS = 0.5
            SHOULDER_VIS = 0.3

            visible_wrist = None
            if right_wrist.visibility >= SINGLE_WRIST_VIS and left_wrist.visibility < SINGLE_WRIST_VIS:
                visible_wrist = right_wrist
            elif left_wrist.visibility >= SINGLE_WRIST_VIS and right_wrist.visibility < SINGLE_WRIST_VIS:
                visible_wrist = left_wrist

            if (visible_wrist is not None and
                    right_shoulder.visibility >= SHOULDER_VIS and
                    left_shoulder.visibility >= SHOULDER_VIS):
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
            self.logger.debug(f"Exception in calculate_wrist_distance: {e}")
            return None, None

    def _calculate_body_movement(
        self,
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
        body_vis_threshold = getattr(self.settings, 'ir_forward_lean_body_vis_threshold', 0.2) if self.settings else 0.2
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

    def _update_sleep_state_machine(
        self,
        tracking: Dict[str, Any],
        timestamp_sec: float,
        is_head_down: bool,
        is_sustained_low_eyes: bool,
        is_minimal_movement: bool,
        head_bob_detected: bool,
        avg_wrist_velocity: float
    ) -> str:
        """Temporal state machine for sleep detection.

        States: ALERT, LOOKING_DOWN_WORKING, DROWSY, MICROSLEEP, SLEEPING

        Args:
            tracking: Per-person sleep tracking dict
            timestamp_sec: Current timestamp in seconds
            is_head_down: Whether head is tilted down
            is_sustained_low_eyes: Whether eyes have been low-visibility for sustained frames
            is_minimal_movement: Whether body movement is minimal
            head_bob_detected: Whether a head bob pattern was detected
            avg_wrist_velocity: Average wrist velocity from recent history

        Returns:
            str: Current sleep state after transition
        """
        current_state = tracking.get('sleep_state', 'ALERT')
        has_hand_activity = avg_wrist_velocity > self.SLEEP_STATE_HAND_ACTIVITY_THRESHOLD

        # Calculate time in current state
        if tracking.get('state_enter_time') is not None:
            time_in_state = timestamp_sec - tracking['state_enter_time']
        else:
            time_in_state = 0.0
            tracking['state_enter_time'] = timestamp_sec

        new_state = current_state

        if current_state == 'ALERT':
            if is_head_down and has_hand_activity:
                new_state = 'LOOKING_DOWN_WORKING'
            elif ((is_sustained_low_eyes and is_minimal_movement and not has_hand_activity)
                  or head_bob_detected):
                new_state = 'DROWSY'

        elif current_state == 'LOOKING_DOWN_WORKING':
            if not is_head_down or not has_hand_activity:
                new_state = 'ALERT'

        elif current_state == 'DROWSY':
            if has_hand_activity or (not is_sustained_low_eyes and not head_bob_detected):
                new_state = 'ALERT'
            elif time_in_state >= self.SLEEP_DROWSY_TO_MICROSLEEP_SEC:
                new_state = 'MICROSLEEP'

        elif current_state == 'MICROSLEEP':
            if has_hand_activity or (not is_sustained_low_eyes and not head_bob_detected):
                new_state = 'ALERT'
            elif time_in_state >= self.SLEEP_MICROSLEEP_TO_SLEEP_SEC:
                new_state = 'SLEEPING'

        elif current_state == 'SLEEPING':
            if has_hand_activity:
                new_state = 'ALERT'

        # Handle state transition
        if new_state != current_state:
            tracking['state_history'].append((current_state, new_state, timestamp_sec))
            tracking['sleep_state'] = new_state
            tracking['state_enter_time'] = timestamp_sec
            self.logger.debug(
                f"[Sleep State Machine] {current_state} -> {new_state} "
                f"(time_in_prev={time_in_state:.1f}s, hand_activity={has_hand_activity}, "
                f"head_bob={head_bob_detected})"
            )

        return tracking['sleep_state']

    def detect_pose_based_sleep(
        self,
        landmarks: Any,
        timestamp_sec: float,
        person_idx: int,
        frame_shape: Tuple[int, ...],
        haar_result: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Detect sleep/microsleep from pose landmarks.

        Uses a weighted scoring approach with detection paths for:
        - Forward head drop (legacy, frontal/side cameras)
        - Reclined posture (overhead/behind cameras)

        Common signals include eye visibility, movement level, and posture stability.

        Args:
            landmarks: Pose landmarks object (YoloPoseLandmarks or similar)
            timestamp_sec: Current timestamp in seconds
            person_idx: Person index for per-person tracking
            frame_shape: Frame shape tuple (height, width, channels)
            haar_result: Optional pre-computed Haar cascade result

        Returns:
            tuple: (sleep_detected, microsleep_detected, debug_info)
        """
        if not landmarks:
            return False, False, {}

        if not self.validate_pose_landmarks(landmarks):
            return False, False, {}

        # Get per-person tracking state
        tracking = self._get_per_person_sleep_tracking(person_idx)

        # Calculate head tilt angle
        landmark_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        head_tilt = self.calculate_head_tilt_angle(landmark_list)

        # Calculate movement score
        movement_score = self.calculate_movement_score(
            landmark_list,
            tracking['previous_landmarks']
        )

        # --- Nose-to-shoulder signals ---
        nose_below_px = None
        nose_above_shoulders_norm = None
        frame_height = frame_shape[0] if frame_shape is not None else None
        frame_width = frame_shape[1] if frame_shape is not None and len(frame_shape) > 1 else None

        try:
            nose = self.get_keypoint(landmark_list, 'nose')
            left_shoulder = self.get_keypoint(landmark_list, 'left_shoulder')
            right_shoulder = self.get_keypoint(landmark_list, 'right_shoulder')
            if nose and left_shoulder and right_shoulder:
                shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
                if frame_height is not None:
                    nose_below_px = (nose.y - shoulder_mid_y) * frame_height
                else:
                    nose_below_px = (nose.y - shoulder_mid_y) * 720
                nose_above_shoulders_norm = shoulder_mid_y - nose.y
        except Exception:
            self.logger.debug("[POSE SLEEP] Failed to extract nose/shoulder keypoints")

        # --- Reclined posture signals ---
        torso_height_px = None
        nose_y_normalized = None
        shoulder_width_px = None
        try:
            nose = self.get_keypoint(landmark_list, 'nose')
            left_shoulder = self.get_keypoint(landmark_list, 'left_shoulder')
            right_shoulder = self.get_keypoint(landmark_list, 'right_shoulder')
            left_hip = self.get_keypoint(landmark_list, 'left_hip')
            right_hip = self.get_keypoint(landmark_list, 'right_hip')

            h_ref = frame_height if frame_height is not None else 720
            w_ref = frame_width if frame_width is not None else 1280

            # Torso height
            if (left_hip and right_hip and left_shoulder and right_shoulder and
                    getattr(left_hip, 'visibility', 0) > 0.2 and getattr(right_hip, 'visibility', 0) > 0.2 and
                    getattr(left_shoulder, 'visibility', 0) > 0.2 and getattr(right_shoulder, 'visibility', 0) > 0.2):
                sh_mid_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_mid_y = (left_hip.y + right_hip.y) / 2
                torso_height_px = abs(hip_mid_y - sh_mid_y) * h_ref

            # Nose Y normalized
            if nose and getattr(nose, 'visibility', 0) > 0.3:
                nose_y_normalized = nose.y

            # Shoulder width
            if (left_shoulder and right_shoulder and
                    getattr(left_shoulder, 'visibility', 0) > 0.2 and getattr(right_shoulder, 'visibility', 0) > 0.2):
                shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * w_ref
        except Exception as e:
            self.logger.debug(f"[POSE SLEEP] Failed to calculate reclined signals: {e}")

        # --- Eye visibility ---
        avg_eye_vis = None
        try:
            left_eye = self.get_keypoint(landmark_list, 'left_eye')
            right_eye = self.get_keypoint(landmark_list, 'right_eye')
            left_vis = getattr(left_eye, 'visibility', 0.0) or 0.0
            right_vis = getattr(right_eye, 'visibility', 0.0) or 0.0
            avg_eye_vis = (left_vis + right_vis) / 2
        except Exception as e:
            self.logger.debug(f"[POSE SLEEP] Failed to calculate eye visibility: {e}")

        # --- Wrist distance ---
        wrist_dist = None
        wrist_below_shoulder = False
        try:
            wd, _wd_source = self.calculate_wrist_distance(landmarks, frame_shape)
            if wd is not None:
                wrist_dist = wd
                tracking['wrist_distance_history'].append(wrist_dist)

            # Check if hands are at console level
            rw = self.get_keypoint(landmark_list, 'right_wrist')
            lw = self.get_keypoint(landmark_list, 'left_wrist')
            rs = self.get_keypoint(landmark_list, 'right_shoulder')
            ls = self.get_keypoint(landmark_list, 'left_shoulder')
            if (rw and lw and rs and ls and
                    getattr(rw, 'visibility', 0) > 0.2 and getattr(lw, 'visibility', 0) > 0.2 and
                    getattr(rs, 'visibility', 0) > 0.2 and getattr(ls, 'visibility', 0) > 0.2):
                avg_wrist_y = (rw.y + lw.y) / 2
                avg_shoulder_y = (rs.y + ls.y) / 2
                wrist_below_shoulder = (avg_wrist_y - avg_shoulder_y) > 0.08
        except Exception as e:
            self.logger.debug(f"[POSE SLEEP] Failed to check wrist-shoulder position: {e}")

        # Initialize variables used in debug_info
        wrist_velocity = 0.0
        avg_wrist_velocity = 0.0
        is_wrists_still = False
        is_wrists_active = False
        head_bob_detected = False
        shoulder_slump_rate = 0.0
        is_shoulder_slumping = False
        current_sleep_state = 'ALERT'
        is_face_not_visible = False
        is_face_gone_with_body_signals = False

        # --- Wrist velocity tracking ---
        try:
            rw = self.get_keypoint(landmark_list, 'right_wrist')
            lw = self.get_keypoint(landmark_list, 'left_wrist')
            if (rw and lw and
                    getattr(rw, 'visibility', 0) > 0.2 and getattr(lw, 'visibility', 0) > 0.2):
                current_wrist_pos = ((lw.x, lw.y), (rw.x, rw.y))
                prev_wrist_pos = tracking.get('previous_wrist_positions')
                if prev_wrist_pos is not None:
                    left_disp = ((current_wrist_pos[0][0] - prev_wrist_pos[0][0]) ** 2 +
                                 (current_wrist_pos[0][1] - prev_wrist_pos[0][1]) ** 2) ** 0.5
                    right_disp = ((current_wrist_pos[1][0] - prev_wrist_pos[1][0]) ** 2 +
                                  (current_wrist_pos[1][1] - prev_wrist_pos[1][1]) ** 2) ** 0.5
                    wrist_velocity = (left_disp + right_disp) / 2.0
                tracking['wrist_velocity_history'].append(wrist_velocity)
                tracking['previous_wrist_positions'] = current_wrist_pos
        except Exception as e:
            self.logger.debug(f"[POSE SLEEP] Failed to calculate wrist velocity: {e}")

        # Update history
        if head_tilt is not None:
            tracking['head_tilt_history'].append(head_tilt)
            if len(tracking['head_tilt_history']) >= 2:
                tilt_list = tracking['head_tilt_history']
                delta = tilt_list[-1] - tilt_list[-2]
                tracking['head_tilt_deltas'].append(delta)

        tracking['movement_history'].append(movement_score)

        if nose_below_px is not None:
            tracking['nose_distance_history'].append(nose_below_px)

        if torso_height_px is not None:
            tracking['torso_height_history'].append(torso_height_px)

        if nose_y_normalized is not None:
            tracking['nose_y_norm_history'].append(nose_y_normalized)

        if nose_above_shoulders_norm is not None:
            tracking['nose_above_shoulders_history'].append(nose_above_shoulders_norm)

        # --- Shoulder Y tracking ---
        try:
            ls_sh = self.get_keypoint(landmark_list, 'left_shoulder')
            rs_sh = self.get_keypoint(landmark_list, 'right_shoulder')
            if (ls_sh and rs_sh and
                    getattr(ls_sh, 'visibility', 0) > 0.2 and getattr(rs_sh, 'visibility', 0) > 0.2):
                avg_sh_y = (ls_sh.y + rs_sh.y) / 2.0
                tracking['shoulder_y_history'].append(avg_sh_y)
                tracking['shoulder_y_timestamps'].append(timestamp_sec)
        except Exception as e:
            self.logger.debug(f"[POSE SLEEP] Failed to update shoulder history: {e}")

        # Store current landmarks for next frame
        tracking['previous_landmarks'] = landmark_list

        # --- Baseline calibration logic ---
        if tracking.get('baseline_calibrating') and self.SLEEP_BASELINE_ENABLED:
            if tracking['baseline_start_time'] is None:
                tracking['baseline_start_time'] = timestamp_sec

            bs = tracking['baseline_samples']
            if nose_below_px is not None:
                bs['nose_below_px'].append(nose_below_px)
            if head_tilt is not None:
                bs['head_tilt'].append(head_tilt)
            if torso_height_px is not None:
                bs['torso_height_px'].append(torso_height_px)
            if shoulder_width_px is not None:
                bs['shoulder_width_px'].append(shoulder_width_px)
            if nose_above_shoulders_norm is not None:
                bs['nose_above_shoulders_norm'].append(nose_above_shoulders_norm)
            if nose_y_normalized is not None:
                bs['nose_y_normalized'].append(nose_y_normalized)
            if movement_score is not None:
                bs['movement'].append(movement_score)
            if avg_eye_vis is not None:
                bs['eye_vis'].append(avg_eye_vis)

            elapsed = timestamp_sec - tracking['baseline_start_time']
            min_samples_met = len(bs['nose_below_px']) >= self.SLEEP_BASELINE_MIN_SAMPLES
            if elapsed >= self.SLEEP_BASELINE_CALIBRATION_WINDOW and min_samples_met:
                tracking['baseline'] = {
                    k: float(np.median(v)) if len(v) > 0 else None
                    for k, v in bs.items()
                }
                tracking['baseline_calibrating'] = False
                self.logger.debug(
                    f"[POSE SLEEP] Person {person_idx}: baseline established after {elapsed:.1f}s "
                    f"({len(bs['nose_below_px'])} samples) - {tracking['baseline']}"
                )

        # --- Sustained-signal counter updates ---
        # Sustained stillness
        if movement_score is not None and movement_score < self.SLEEP_SUSTAINED_STILLNESS_THRESHOLD:
            tracking['sustained_stillness_count'] = tracking.get('sustained_stillness_count', 0) + 1
        else:
            tracking['sustained_stillness_count'] = 0

        # Hands clasped
        if wrist_dist is not None and wrist_dist < self.SLEEP_HANDS_CLASPED_THRESHOLD:
            tracking['hands_clasped_count'] = tracking.get('hands_clasped_count', 0) + 1
        else:
            tracking['hands_clasped_count'] = 0

        # Sustained low eye visibility
        eyes_not_in_frame_thresh = getattr(self.settings, 'eyes_not_in_frame_threshold', 0.15) if self.settings else 0.15
        overhead_nose_y_thresh = getattr(self.settings, 'sleep_overhead_nose_y_threshold', 0.10) if self.settings else 0.10
        is_overhead_camera = nose_y_normalized is not None and nose_y_normalized < overhead_nose_y_thresh
        if (avg_eye_vis is not None and avg_eye_vis >= eyes_not_in_frame_thresh
                and avg_eye_vis < self.EYES_NOT_VISIBLE_THRESHOLD
                and not is_overhead_camera):
            tracking['low_eye_vis_count'] = tracking.get('low_eye_vis_count', 0) + 1
        else:
            tracking['low_eye_vis_count'] = 0

        # Face-not-visible counter
        face_gone_threshold = getattr(self.settings, 'sleep_face_gone_threshold', 0.25) if self.settings else 0.25
        is_face_not_visible = (avg_eye_vis is not None
                               and avg_eye_vis < face_gone_threshold
                               and not is_overhead_camera)
        if is_face_not_visible:
            tracking['face_not_visible_count'] = tracking.get('face_not_visible_count', 0) + 1
        else:
            tracking['face_not_visible_count'] = 0

        # Derive sustained-signal booleans
        is_sustained_stillness = tracking['sustained_stillness_count'] >= self.SLEEP_SUSTAINED_STILLNESS_FRAMES
        is_hands_clasped = tracking['hands_clasped_count'] >= self.SLEEP_HANDS_CLASPED_FRAMES
        is_sustained_low_eyes = tracking['low_eye_vis_count'] >= self.SLEEP_SUSTAINED_LOW_EYE_FRAMES

        # --- Head Bob Detection ---
        head_bob_detected = False
        deltas = list(tracking['head_tilt_deltas'])
        if len(deltas) >= (self.SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES + 1):
            for i in range(len(deltas) - 1, self.SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES - 1, -1):
                if deltas[i] >= self.SLEEP_HEAD_BOB_JERK_MIN_RATE:
                    drift_ok = True
                    total_drift = 0.0
                    for j in range(i - 1, max(i - 1 - self.SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES, -1), -1):
                        if j < 0:
                            drift_ok = False
                            break
                        if deltas[j] > 0 or abs(deltas[j]) > self.SLEEP_HEAD_BOB_DRIFT_MAX_RATE:
                            drift_ok = False
                            break
                        total_drift += abs(deltas[j])
                    if drift_ok and total_drift >= self.SLEEP_HEAD_BOB_MIN_AMPLITUDE:
                        head_bob_detected = True
                        tracking['head_bob_count'] = tracking.get('head_bob_count', 0) + 1
                        break

        # --- Wrist Velocity Flags ---
        vel_hist = list(tracking['wrist_velocity_history'])
        avg_wrist_velocity = np.mean(vel_hist) if len(vel_hist) > 0 else 0.0
        is_wrists_still = False
        if len(vel_hist) >= self.SLEEP_WRIST_VEL_STILL_FRAMES:
            recent_vels = vel_hist[-self.SLEEP_WRIST_VEL_STILL_FRAMES:]
            is_wrists_still = all(v < self.SLEEP_WRIST_VEL_STILL for v in recent_vels)
        is_wrists_active = avg_wrist_velocity > self.SLEEP_WRIST_VEL_ACTIVE

        # Need sufficient history
        min_samples = max(2, int(2 * self.sample_fps))

        if len(tracking['movement_history']) < min_samples:
            return False, False, {
                'head_tilt': head_tilt,
                'movement': movement_score,
                'nose_below_px': nose_below_px,
                'nose_above_shoulders_norm': nose_above_shoulders_norm,
                'torso_height_px': torso_height_px,
                'nose_y_normalized': nose_y_normalized,
                'shoulder_width_px': shoulder_width_px,
                'avg_eye_vis': avg_eye_vis,
                'status': 'building_history'
            }

        # Calculate averages
        avg_head_tilt = np.mean(list(tracking['head_tilt_history'])) if len(tracking['head_tilt_history']) > 0 else 0
        avg_movement = np.mean(list(tracking['movement_history']))
        head_tilt_variance = np.var(list(tracking['head_tilt_history'])) if len(tracking['head_tilt_history']) > 0 else 999
        avg_nose_below_px = np.mean(list(tracking['nose_distance_history'])) if len(tracking['nose_distance_history']) > 0 else 0
        avg_torso_height = np.mean(list(tracking['torso_height_history'])) if len(tracking['torso_height_history']) > 0 else 0
        avg_nose_y_norm = np.mean(list(tracking['nose_y_norm_history'])) if len(tracking['nose_y_norm_history']) > 0 else 0.5
        avg_nose_above_shoulders = np.mean(list(tracking['nose_above_shoulders_history'])) if len(tracking['nose_above_shoulders_history']) > 0 else 0.0

        # --- Shoulder Slump Rate ---
        shoulder_slump_rate = 0.0
        is_shoulder_slumping = False
        sh_y_hist = list(tracking['shoulder_y_history'])
        sh_t_hist = list(tracking['shoulder_y_timestamps'])
        if len(sh_y_hist) >= self.SLEEP_SHOULDER_SLUMP_MIN_FRAMES and len(sh_t_hist) == len(sh_y_hist):
            t_arr = np.array(sh_t_hist)
            y_arr = np.array(sh_y_hist)
            t_mean = np.mean(t_arr)
            y_mean = np.mean(y_arr)
            t_diff = t_arr - t_mean
            y_diff = y_arr - y_mean
            denom = np.sum(t_diff ** 2)
            if denom > 1e-12:
                shoulder_slump_rate = float(np.sum(t_diff * y_diff) / denom)
                is_shoulder_slumping = shoulder_slump_rate > self.SLEEP_SHOULDER_SLUMP_RATE_THRESHOLD

        # --- Compute sleep indicators ---
        baseline = tracking.get('baseline')
        has_baseline = baseline is not None

        head_tilt_thresh = getattr(self.settings, 'sleep_head_tilt_threshold', -155) if self.settings else -155
        nose_below_thresh = getattr(self.settings, 'sleep_nose_below_px_threshold', -55) if self.settings else -55

        # Fix 9: Soften absolute thresholds by ~15% during pre-calibration window.
        # Gradual sleep onset during the first 10s baseline may be missed otherwise.
        if not has_baseline:
            head_tilt_thresh *= 0.85  # e.g. -155 → -131.75 (softer)
            nose_below_thresh *= 0.85  # e.g. -55 → -46.75 (softer)

        if has_baseline:
            baseline_nose = baseline.get('nose_below_px')
            is_nose_drooping = (nose_below_px is not None and baseline_nose is not None and
                                nose_below_px < baseline_nose - self.SLEEP_BASELINE_NOSE_BELOW_DELTA)

            baseline_tilt = baseline.get('head_tilt')
            is_head_down = (baseline_tilt is not None and
                            avg_head_tilt < baseline_tilt - self.SLEEP_BASELINE_HEAD_TILT_DELTA)

            baseline_torso = baseline.get('torso_height_px')
            is_torso_elongated = (torso_height_px is not None and baseline_torso is not None and
                                  torso_height_px > baseline_torso + self.SLEEP_BASELINE_TORSO_HEIGHT_DELTA)

            baseline_shoulder = baseline.get('shoulder_width_px')
            is_shoulders_compressed = (shoulder_width_px is not None and baseline_shoulder is not None and
                                       shoulder_width_px < baseline_shoulder - self.SLEEP_BASELINE_SHOULDER_WIDTH_DELTA)
        else:
            is_nose_drooping = nose_below_px is not None and nose_below_px < nose_below_thresh
            is_head_down = avg_head_tilt < head_tilt_thresh
            torso_thresh = getattr(self.settings, 'sleep_reclined_torso_height_threshold', 175) if self.settings else 175
            is_torso_elongated = torso_height_px is not None and torso_height_px > torso_thresh
            sh_width_thresh = getattr(self.settings, 'sleep_reclined_shoulder_width_threshold', 60) if self.settings else 60
            is_shoulders_compressed = shoulder_width_px is not None and shoulder_width_px < sh_width_thresh

        nose_above_sh_thresh = getattr(self.settings, 'sleep_nose_above_shoulders_threshold', 0.08) if self.settings else 0.08
        is_nose_above_shoulders = len(tracking['nose_above_shoulders_history']) > 0 and avg_nose_above_shoulders > nose_above_sh_thresh

        nose_y_thresh = getattr(self.settings, 'sleep_nose_y_norm_threshold', 0.30) if self.settings else 0.30
        is_nose_high_in_frame = nose_y_normalized is not None and nose_y_normalized < nose_y_thresh

        is_minimal_movement = avg_movement < self.MINIMAL_MOVEMENT_THRESHOLD
        is_stable_posture = head_tilt_variance < self.STABLE_POSTURE_VARIANCE
        is_eyes_not_visible = avg_eye_vis is not None and avg_eye_vis < self.EYES_NOT_VISIBLE_THRESHOLD

        # --- State Machine ---
        current_sleep_state = 'ALERT'
        if self.SLEEP_STATE_MACHINE_ENABLED:
            current_sleep_state = self._update_sleep_state_machine(
                tracking, timestamp_sec, is_head_down,
                is_sustained_low_eyes, is_minimal_movement,
                head_bob_detected, avg_wrist_velocity
            )
            if current_sleep_state == 'LOOKING_DOWN_WORKING':
                return False, False, {
                    'head_tilt': head_tilt,
                    'movement': movement_score,
                    'nose_below_px': nose_below_px,
                    'sleep_state': current_sleep_state,
                    'avg_wrist_velocity': avg_wrist_velocity,
                    'status': 'looking_down_working_suppressed'
                }

        # --- HEAD DROP DETECTION ---
        head_drop_detected = False
        nose_y_drop = 0.0
        head_tilt_drop = 0.0

        nose_y_drop_thresh = getattr(self.settings, 'sleep_nose_y_drop_threshold', 0.15) if self.settings else 0.15
        head_tilt_drop_thresh = getattr(self.settings, 'sleep_head_tilt_drop_threshold', 30.0) if self.settings else 30.0

        if has_baseline:
            baseline_nose_y = baseline.get('nose_y_normalized')
            baseline_head_tilt = baseline.get('head_tilt')

            if nose_y_normalized is not None and baseline_nose_y is not None:
                nose_y_drop = nose_y_normalized - baseline_nose_y
                if nose_y_drop > nose_y_drop_thresh:
                    head_drop_detected = True

            if head_tilt is not None and baseline_head_tilt is not None:
                head_tilt_drop = head_tilt - baseline_head_tilt
                if head_tilt_drop > head_tilt_drop_thresh:
                    head_drop_detected = True

        head_drop_from_delta = False
        if len(tracking.get('head_tilt_deltas', [])) >= 1:
            last_delta = list(tracking['head_tilt_deltas'])[-1]
            if last_delta > head_tilt_drop_thresh:
                head_drop_from_delta = True
                head_drop_detected = True

        self.logger.debug(
            f"[HEAD DROP DEBUG] Person {person_idx}: "
            f"has_baseline={has_baseline}, nose_y={nose_y_normalized}, head_tilt={head_tilt}, "
            f"nose_y_drop={nose_y_drop:.4f}, head_tilt_drop={head_tilt_drop:.1f}, "
            f"head_drop={head_drop_detected}, delta_drop={head_drop_from_delta}"
        )

        # Reclined sleep detection: nose tilted back + torso elongated + shoulders compressed
        # A reclined sleeping person leans BACK (head doesn't drop forward).
        is_reclined_sleep = False
        if has_baseline:
            baseline_nose_y = baseline.get('nose_y_normalized')
            if (baseline_nose_y is not None and nose_y_normalized is not None
                    and nose_y_normalized < baseline_nose_y - 0.05
                    and is_torso_elongated and is_shoulders_compressed):
                is_reclined_sleep = True
        else:
            # Without baseline, use absolute thresholds for reclined posture
            if is_torso_elongated and is_shoulders_compressed and is_nose_high_in_frame:
                is_reclined_sleep = True

        # Score calculation
        sleep_score = 0
        is_hands_spread = wrist_dist is not None and wrist_dist > self.SLEEP_HANDS_SPREAD_THRESHOLD

        if head_drop_detected:
            sleep_score += 5
        elif is_reclined_sleep:
            sleep_score += 4  # Slightly less than head_drop, requires longer duration

        if is_hands_spread:
            sleep_score -= 2
        if is_wrists_active:
            sleep_score -= 1

        # Haar cascade eye closure boost
        haar_eye_closed = False
        if haar_result is None:
            haar_result = {}
        haar_eye_closed = haar_result.get('eyes_closed', False)
        if haar_eye_closed:
            haar_boost = getattr(self.settings, 'haar_eye_score_boost', 5) if self.settings else 5
            sleep_score += haar_boost

        score_thresh = getattr(self.settings, 'sleep_score_threshold', 3) if self.settings else 3
        sleep_indicators_met = sleep_score >= score_thresh

        debug_info = {
            'head_tilt': head_tilt,
            'nose_y_normalized': nose_y_normalized,
            'nose_below_px': nose_below_px,
            'movement': movement_score,
            'avg_eye_vis': avg_eye_vis,
            'shoulder_width_px': shoulder_width_px,
            'wrist_distance': wrist_dist,
            'baseline_established': has_baseline,
            'sleep_score': sleep_score,
            'sleep_score_threshold': score_thresh,
            'head_drop_detected': head_drop_detected,
            'head_drop_from_delta': head_drop_from_delta,
            'is_reclined_sleep': is_reclined_sleep,
            'nose_y_drop': nose_y_drop,
            'head_tilt_drop': head_tilt_drop,
            'nose_y_drop_thresh': nose_y_drop_thresh,
            'head_tilt_drop_thresh': head_tilt_drop_thresh,
            'is_hands_spread': is_hands_spread,
            'is_wrists_active': is_wrists_active,
            'avg_wrist_velocity': avg_wrist_velocity,
            'sleep_state': current_sleep_state,
            'haar_eye_closed': haar_eye_closed,
            'haar_eye_info': haar_result,
        }

        # Hard gate: head drop OR haar eye closure OR reclined posture must be detected
        if not head_drop_detected and not haar_eye_closed and not is_reclined_sleep:
            tracking['pose_sleep_start'] = None
            tracking['pose_sleep_duration'] = 0
            return False, False, debug_info

        # Detect sleep condition
        if sleep_indicators_met:
            if tracking['pose_sleep_start'] is None:
                tracking['pose_sleep_start'] = timestamp_sec
                self.logger.debug(
                    f"[Pose-Based Sleep] Person {person_idx} started tracking - "
                    f"score={sleep_score}, baseline={has_baseline}, "
                    f"head_drop={head_drop_detected}, nose_y_drop={nose_y_drop:.4f}"
                )

            tracking['pose_sleep_duration'] = timestamp_sec - tracking['pose_sleep_start']

            if sleep_score >= self.SLEEP_STRONG_SCORE:
                is_sleeping = tracking['pose_sleep_duration'] >= self.SLEEP_STRONG_DURATION
                is_microsleeping = not is_sleeping
            else:
                is_sleeping = tracking['pose_sleep_duration'] >= self.SLEEP_MODERATE_DURATION
                is_microsleeping = tracking['pose_sleep_duration'] >= self.SLEEP_MICROSLEEP_DURATION and not is_sleeping

            debug_info['pose_sleep_duration'] = tracking['pose_sleep_duration']

            return is_sleeping, is_microsleeping, debug_info
        else:
            if tracking['pose_sleep_start'] is not None:
                self.logger.debug(f"[Pose-Based Sleep] Person {person_idx} stopped - score={sleep_score}")
            tracking['pose_sleep_start'] = None
            tracking['pose_sleep_duration'] = 0

            return False, False, debug_info

    def detect_ir_forward_lean_sleep(
        self,
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

        if len(landmarks.landmark) < self.YOLO_MIN_KEYPOINTS:
            return False, False, {
                'ir_forward_lean': False,
                'reason': 'insufficient_landmarks',
                'landmark_count': len(landmarks.landmark)
            }

        # Safe settings access with defaults
        head_vis_thresh = getattr(self.settings, 'ir_forward_lean_head_vis_threshold', 0.15) if self.settings else 0.15
        body_vis_thresh = getattr(self.settings, 'ir_forward_lean_body_vis_threshold', 0.2) if self.settings else 0.2
        min_body_kps = getattr(self.settings, 'ir_forward_lean_min_body_keypoints', 3) if self.settings else 3
        score_threshold = getattr(self.settings, 'ir_forward_lean_score_threshold', 4) if self.settings else 4
        min_duration = getattr(self.settings, 'ir_forward_lean_min_duration', 5.0) if self.settings else 5.0
        sleep_duration = getattr(self.settings, 'ir_forward_lean_sleep_duration', 10.0) if self.settings else 10.0

        h, w = frame_shape[:2]

        head_indices = self.YOLO_HEAD_INDICES
        body_indices = self.YOLO_BODY_INDICES

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
            tracking = self._get_ir_forward_lean_tracking(person_idx)
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
            if shoulder_relative < self.IR_SHOULDER_RELATIVE_THRESHOLD:
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
        if aspect_ratio < self.IR_BBOX_ASPECT_RATIO_THRESHOLD:
            score += 1
            score_breakdown['squashed_bbox'] = 1
        else:
            score_breakdown['squashed_bbox'] = 0
        score_breakdown['aspect_ratio'] = round(aspect_ratio, 2)

        # Signal 4: Low body movement (+1)
        tracking = self._get_ir_forward_lean_tracking(person_idx)
        body_movement = self._calculate_body_movement(landmarks, tracking, body_indices)
        if body_movement is not None and body_movement < self.IR_LOW_MOVEMENT_THRESHOLD:
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
                self.logger.info(
                    f"[IR FORWARD LEAN] Person {person_idx}: SLEEP detected "
                    f"(score={score}/{score_threshold}, duration={duration:.1f}s)"
                )
                return True, False, debug_info
            elif duration >= min_duration:
                self.logger.info(
                    f"[IR FORWARD LEAN] Person {person_idx}: MICROSLEEP detected "
                    f"(score={score}/{score_threshold}, duration={duration:.1f}s)"
                )
                return False, True, debug_info
            else:
                self.logger.debug(
                    f"[IR FORWARD LEAN] Person {person_idx}: forward lean detected but duration too short "
                    f"(score={score}, duration={duration:.1f}s, need {min_duration:.0f}s)"
                )
        else:
            tracking['sub_threshold_streak'] = tracking.get('sub_threshold_streak', 0) + 1
            if tracking['sub_threshold_streak'] >= self.SUB_THRESHOLD_STREAK_LIMIT:
                tracking['start_time'] = None
                tracking['sub_threshold_streak'] = 0
            debug_info['duration'] = 0
            debug_info['sub_threshold_streak'] = tracking['sub_threshold_streak']

        return False, False, debug_info

    def detect_eye_closure_haar(
        self,
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

        if frame is None or landmarks is None or self.eye_cascade is None or self.settings is None:
            return result

        if not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
            return result

        tracking = self._get_per_person_sleep_tracking(person_idx)
        h, w = frame.shape[:2]
        settings = self.settings

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

        # Fix 10: If face ROI is very dark (IR/low-light), skip face detection
        # and fall through to posture-only detection. Haar cascades are unreliable in near-dark.
        roi_mean_intensity = float(np.mean(gray_roi))
        if roi_mean_intensity < 50:
            self.logger.debug(
                f"[HAAR] Face ROI too dark (mean={roi_mean_intensity:.0f}), skipping face detection"
            )
            result['low_light_skip'] = True
            return result

        gray_roi = cv2.equalizeHist(gray_roi)

        # --- Detect face within ROI ---
        scale_factor = getattr(settings, 'haar_eye_scale_factor', 1.1)
        min_neighbors = getattr(settings, 'haar_eye_min_neighbors', 3)

        faces = self.face_cascade.detectMultiScale(
            gray_roi, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=(30, 30)
        )

        # Fallback to profile face
        if len(faces) == 0:
            faces = self.profile_face_cascade.detectMultiScale(
                gray_roi, scaleFactor=scale_factor, minNeighbors=max(1, min_neighbors - 1),
                minSize=(30, 30)
            )
            if len(faces) == 0:
                flipped_roi = cv2.flip(gray_roi, 1)
                faces = self.profile_face_cascade.detectMultiScale(
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
        eyes = self.eye_cascade.detectMultiScale(
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

    def reset_tracking(self, person_idx: Optional[int] = None) -> None:
        """Reset tracking state for a person or all persons.

        Args:
            person_idx: Person index to reset. If None, resets all tracking.
        """
        if person_idx is not None:
            if person_idx in self.per_person_tracking:
                self.per_person_tracking[person_idx] = self._create_tracking_dict()
            if person_idx in self.ir_forward_lean_tracking:
                self.ir_forward_lean_tracking[person_idx] = {
                    'start_time': None,
                    'previous_body_keypoints': None,
                    'sub_threshold_streak': 0,
                }
        else:
            self.per_person_tracking.clear()
            self.ir_forward_lean_tracking.clear()

    def get_tracking_state(self, person_idx: int) -> Dict[str, Any]:
        """Get current tracking state for a person.

        Args:
            person_idx: Person index

        Returns:
            Dict containing sleep and IR forward lean tracking state
        """
        return {
            'sleep_tracking': dict(self._get_per_person_sleep_tracking(person_idx)),
            'ir_forward_lean_tracking': dict(self._get_ir_forward_lean_tracking(person_idx)),
        }
