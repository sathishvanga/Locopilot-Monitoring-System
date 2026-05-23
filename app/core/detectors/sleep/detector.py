"""SleepDetector class — main entry point for sleep/microsleep detection."""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
import logging
import numpy as np

from .pose_geometry import (
    get_keypoint as _pg_get_keypoint,
    validate_pose_landmarks as _pg_validate_pose_landmarks,
    calculate_head_tilt_angle as _pg_calculate_head_tilt_angle,
    calculate_movement_score as _pg_calculate_movement_score,
    calculate_wrist_distance as _pg_calculate_wrist_distance,
    _calculate_body_movement as _pg_calculate_body_movement,
)
from .state_machine import update_sleep_state_machine as _sm_update_sleep_state_machine
from .ir_fallback import detect_ir_forward_lean_sleep as _ir_detect_ir_forward_lean_sleep
from .haar_eye_closure import (
    _load_haar_cascades as _haar_load_haar_cascades,
    detect_eye_closure_haar as _haar_detect_eye_closure_haar,
)


class SleepDetector:
    """Detects sleep and microsleep using pose landmarks and eye analysis."""

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
        """Initialize the SleepDetector."""
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
        self.SLEEP_MICROSLEEP_DURATION = getattr(s, 'sleep_microsleep_duration', 5) if s else 5

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

        # Head drop consecutive check — require N consecutive head_drop=True before confirming
        self.SLEEP_HEAD_DROP_MIN_CONSECUTIVE = getattr(s, 'sleep_head_drop_min_consecutive', 2) if s else 2

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

        # Wrist velocity thresholds
        self.SLEEP_WRIST_VEL_STILL = getattr(s, 'sleep_wrist_velocity_still_threshold', 0.005) if s else 0.005
        self.SLEEP_WRIST_VEL_ACTIVE = getattr(s, 'sleep_wrist_velocity_active_threshold', 0.03) if s else 0.03
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
        return _haar_load_haar_cascades(self, eye_cascade_path)

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
            # Consecutive head drop counter for noise filtering
            'head_drop_consecutive_count': 0,
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
        """Get or initialize per-person sleep tracking state."""
        if person_idx not in self.per_person_tracking:
            self.per_person_tracking[person_idx] = self._create_tracking_dict()
        return self.per_person_tracking[person_idx]

    def _get_ir_forward_lean_tracking(self, person_idx: int) -> Dict[str, Any]:
        """Get or initialize per-person IR forward-lean sleep tracking state."""
        if person_idx not in self.ir_forward_lean_tracking:
            self.ir_forward_lean_tracking[person_idx] = {
                'start_time': None,
                'previous_body_keypoints': None,
                'sub_threshold_streak': 0,  # Consecutive frames below score threshold
            }
        return self.ir_forward_lean_tracking[person_idx]

    def get_keypoint(self, landmarks: Any, keypoint_name: str) -> Any:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_get_keypoint(self, landmarks, keypoint_name)

    def validate_pose_landmarks(
        self,
        pose_landmarks: Any,
        min_landmarks: Optional[int] = None,
        min_visibility: Optional[float] = None
    ) -> bool:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_validate_pose_landmarks(self, pose_landmarks, min_landmarks, min_visibility)

    def calculate_head_tilt_angle(self, landmarks: Any) -> Optional[float]:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_calculate_head_tilt_angle(self, landmarks)

    def calculate_movement_score(
        self,
        current_landmarks: Any,
        previous_landmarks: Any
    ) -> float:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_calculate_movement_score(self, current_landmarks, previous_landmarks)

    def calculate_wrist_distance(
        self,
        pose_landmarks: Any,
        frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[float], Optional[str]]:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_calculate_wrist_distance(self, pose_landmarks, frame_shape)

    def _calculate_body_movement(
        self,
        landmarks: Any,
        tracking: Dict[str, Any],
        body_indices: List[int]
    ) -> Optional[float]:
        """Thin shim — see :mod:`.pose_geometry`."""
        return _pg_calculate_body_movement(self, landmarks, tracking, body_indices)

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
        """Thin shim — see :mod:`.state_machine`."""
        return _sm_update_sleep_state_machine(
            self, tracking, timestamp_sec, is_head_down,
            is_sustained_low_eyes, is_minimal_movement,
            head_bob_detected, avg_wrist_velocity,
        )

    def detect_pose_based_sleep(
        self,
        landmarks: Any,
        timestamp_sec: float,
        person_idx: int,
        frame_shape: Tuple[int, ...],
        haar_result: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Detect sleep/microsleep from pose landmarks."""
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
                # Wrap delta to [-180, 180] — without this, a tilt going
                # +170 -> -170 across the +/-180 seam produces a -340 deg
                # delta and trips the head-drop threshold falsely.
                delta = (delta + 180) % 360 - 180
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

        if has_baseline:
            baseline_nose = baseline.get('nose_below_px')
            is_nose_drooping = (nose_below_px is not None and baseline_nose is not None and
                                nose_below_px < baseline_nose - self.SLEEP_BASELINE_NOSE_BELOW_DELTA)

            baseline_tilt = baseline.get('head_tilt')
            # Wrap (avg_head_tilt - baseline_tilt) into [-180, 180] before
            # comparing against the negative SLEEP_BASELINE_HEAD_TILT_DELTA
            # threshold.  Same wrap-around bug class as the per-frame delta
            # and the head_tilt_drop branch: a baseline near +180 with a
            # current avg near -180 (or vice-versa) would otherwise yield a
            # 300+ deg apparent down-tilt and trip ``is_head_down`` falsely.
            if baseline_tilt is not None:
                avg_tilt_delta = (avg_head_tilt - baseline_tilt + 180) % 360 - 180
                is_head_down = avg_tilt_delta < -self.SLEEP_BASELINE_HEAD_TILT_DELTA
            else:
                is_head_down = False

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
        head_drop_from_delta = False

        nose_y_drop_thresh = getattr(self.settings, 'sleep_nose_y_drop_threshold', 0.15) if self.settings else 0.15
        head_tilt_drop_thresh = getattr(self.settings, 'sleep_head_tilt_drop_threshold', 30.0) if self.settings else 30.0

        # Compute nose_y_drop up-front so we can use it as a single short-circuit
        # guard across ALL head-drop branches (nose-y, head-tilt-vs-baseline,
        # and per-frame delta).  Spec ``docs/specs/code-review-fixes/tasks/
        # 0001-restore-determinism-contract.md`` line 982 requires:
        #     if nose_y_drop is not None and nose_y_drop < 0:
        #         # skip head-drop branches
        # Without this, a noisy head-tilt or per-frame delta blip can fire a
        # head-drop while the nose actually moved UP — physically impossible
        # for a real sleep onset.  When ``nose_y_drop`` is None (landmark
        # unavailable) we keep the existing behavior and let the other
        # branches evaluate normally.
        nose_y_drop_value: Optional[float] = None
        if has_baseline:
            baseline_nose_y = baseline.get('nose_y_normalized')
            if nose_y_normalized is not None and baseline_nose_y is not None:
                nose_y_drop_value = nose_y_normalized - baseline_nose_y
                nose_y_drop = nose_y_drop_value

        skip_head_drop_branches = (
            nose_y_drop_value is not None and nose_y_drop_value < 0
        )

        if has_baseline and not skip_head_drop_branches:
            baseline_head_tilt = baseline.get('head_tilt')

            if nose_y_drop_value is not None:
                # Only count downward drops — if the nose moved UP relative
                # to baseline (smaller y), that's the opposite of sleeping.
                if nose_y_drop_value > nose_y_drop_thresh:
                    head_drop_detected = True

            if head_tilt is not None and baseline_head_tilt is not None:
                # Wrap head_tilt_drop to [-180, 180] so a baseline near +180
                # vs current near -180 doesn't fabricate a 300+ deg drop.
                head_tilt_drop = (head_tilt - baseline_head_tilt + 180) % 360 - 180
                if head_tilt_drop > head_tilt_drop_thresh:
                    head_drop_detected = True

        if not skip_head_drop_branches:
            if len(tracking.get('head_tilt_deltas', [])) >= 1:
                last_delta = list(tracking['head_tilt_deltas'])[-1]
                if last_delta > head_tilt_drop_thresh:
                    head_drop_from_delta = True
                    head_drop_detected = True

        # FP-FIX: Require consecutive head_drop=True frames to confirm
        # Single-frame pose estimation noise should not trigger sleep detection
        raw_head_drop = head_drop_detected
        if head_drop_detected:
            tracking['head_drop_consecutive_count'] = tracking.get('head_drop_consecutive_count', 0) + 1
            if tracking['head_drop_consecutive_count'] < self.SLEEP_HEAD_DROP_MIN_CONSECUTIVE:
                head_drop_detected = False  # Not yet confirmed — need more consecutive frames
        else:
            tracking['head_drop_consecutive_count'] = 0

        self.logger.debug(
            f"[HEAD DROP DEBUG] Person {person_idx}: "
            f"has_baseline={has_baseline}, nose_y={nose_y_normalized}, head_tilt={head_tilt}, "
            f"nose_y_drop={nose_y_drop:.4f}, head_tilt_drop={head_tilt_drop:.1f}, "
            f"head_drop={head_drop_detected}, delta_drop={head_drop_from_delta}, "
            f"raw_head_drop={raw_head_drop}, consecutive={tracking.get('head_drop_consecutive_count', 0)}"
        )

        # Score calculation
        sleep_score = 0
        is_hands_spread = wrist_dist is not None and wrist_dist > self.SLEEP_HANDS_SPREAD_THRESHOLD

        if head_drop_detected:
            sleep_score += 5

        if is_hands_spread:
            sleep_score -= 2
        if is_wrists_active:
            sleep_score -= 1

        # Haar cascade eye closure boost.
        # Gated on HAAR_EYE_DETECTION_ENABLED so the master switch actually
        # controls the score path, not just instantiation. CLAUDE.md notes
        # Haar is "non-functional from overhead" and the default is OFF;
        # without this gate a single-frame Haar positive could still cross
        # the sleep_score threshold on its own (boost=5, threshold=5).
        haar_eye_closed = False
        if haar_result is None:
            haar_result = {}
        _haar_enabled = bool(
            getattr(self.settings, 'haar_eye_detection_enabled', False)
        ) if self.settings else False
        if _haar_enabled:
            haar_eye_closed = haar_result.get('eyes_closed', False)
            if haar_eye_closed:
                haar_boost = getattr(self.settings, 'haar_eye_score_boost', 5) if self.settings else 5
                sleep_score += haar_boost

        score_thresh = getattr(self.settings, 'sleep_score_threshold', 5) if self.settings else 5
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

        # Hard gate: head drop OR haar eye closure must be detected
        if not head_drop_detected and not haar_eye_closed:
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
        """Thin shim — see :mod:`.ir_fallback`."""
        return _ir_detect_ir_forward_lean_sleep(
            self, landmarks, bbox, timestamp_sec, person_idx, frame_shape,
        )

    def detect_eye_closure_haar(
        self,
        frame: Any,
        landmarks: Any,
        person_idx: int,
        bbox: List[int],
        timestamp_sec: float
    ) -> Dict[str, Any]:
        """Thin shim — see :mod:`.haar_eye_closure`."""
        return _haar_detect_eye_closure_haar(
            self, frame, landmarks, person_idx, bbox, timestamp_sec,
        )

    def cleanup_stale_tracking(self, active_person_indices: set) -> None:
        """Remove tracking entries for person indices that are no longer active."""
        active = set(active_person_indices)

        stale_person_keys = [
            k for k in self.per_person_tracking if k not in active
        ]
        for k in stale_person_keys:
            del self.per_person_tracking[k]

        stale_ir_keys = [
            k for k in self.ir_forward_lean_tracking if k not in active
        ]
        for k in stale_ir_keys:
            del self.ir_forward_lean_tracking[k]

        if stale_person_keys or stale_ir_keys:
            self.logger.debug(
                f"[SleepDetector] Cleaned up stale tracking: "
                f"removed {len(stale_person_keys)} person entries, "
                f"{len(stale_ir_keys)} IR entries. "
                f"Active indices: {active}"
            )

    def reset_tracking(self, person_idx: Optional[int] = None) -> None:
        """Reset tracking state for a person or all persons."""
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

    def reset(self) -> None:
        """Detector-protocol alias for :meth:`reset_tracking`."""
        self.reset_tracking()

    def on_suppressed(self, person_idx: Optional[int], activity_name: str) -> None:
        """Hook invoked when sleep is suppressed by the train-stopped gate."""
        if person_idx is None or activity_name != 'sleep':
            return
        # Re-use the canonical per-person reset path.
        self.reset_tracking(person_idx=person_idx)

    def get_tracking_state(self, person_idx: int) -> Dict[str, Any]:
        """Get current tracking state for a person."""
        return {
            'sleep_tracking': dict(self._get_per_person_sleep_tracking(person_idx)),
            'ir_forward_lean_tracking': dict(self._get_ir_forward_lean_tracking(person_idx)),
        }
