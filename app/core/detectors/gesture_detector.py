"""Hand gesture detection for LP/ALP coordination.

This module provides gesture detection functionality for monitoring hand gesture
coordination between Loco Pilot (LP) and Assistant Loco Pilot (ALP) crew members.

Key features:
- Detects raised hand gestures for LP/ALP signaling
- Tracks gesture sessions with configurable timeout
- Validates gesture coordination (both must raise hands within time window)
- Context-aware filtering (suppresses during work activities like packing, writing)
- Object proximity detection (backpack/bag handling)
- Velocity and trajectory analysis for distinguishing signals from control operations
"""
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.utils.pose_utils import get_keypoint as _canonical_get_keypoint


class GestureDetector:
    """Detects hand gesture coordination between LP and ALP.

    LP (Loco Pilot) and ALP (Assistant Loco Pilot) must exchange hand gestures
    for coordination. A violation occurs if one person doesn't raise their hand
    when the other does within the coordination window.

    Attributes:
        settings: Optional settings object for configuration
        session_timeout: Time window for gesture coordination (seconds)
        gesture_sessions: Tracking data for gesture sessions per person
        temporal_suppression_window: Window to suppress gestures after work activity
        hand_position_history: Historical hand positions for velocity analysis
    """

    # Minimum required landmarks for gesture detection
    MIN_LANDMARKS = 8
    MIN_VISIBILITY = 0.25

    # Thresholds for raised hand detection (proportional to person bbox height)
    # Calibrated so that at 1080p with ~500px person bbox height, values match
    # the original absolute pixel thresholds (80, 30, 20, 150 respectively).
    WRIST_SHOULDER_VERTICAL_RATIO = 0.16  # orig 80px / 500px
    WRIST_ELBOW_DISTANCE_RATIO = -0.06  # orig -30px / 500px (negative = bent OK)
    ARM_EXTENSION_RATIO = 0.04  # orig 20px / 500px
    ELBOW_SHOULDER_TOLERANCE_RATIO = 0.30  # orig 150px / 500px

    # Thresholds for control zone filtering (proportional to person bbox height)
    # Calibrated at 1080p / 500px person bbox height.
    CONTROL_ZONE_WRIST_SHOULDER_MIN_RATIO = 0.06  # orig 30px / 500px
    CONTROL_ZONE_WRIST_SHOULDER_MAX_RATIO = 0.20  # orig 100px / 500px
    CONTROL_ZONE_WRIST_ELBOW_MAX_RATIO = 0.10  # orig 50px / 500px
    CONTROL_ZONE_ELBOW_SHOULDER_OFFSET_RATIO = 0.06  # orig 30px / 500px

    # Minimum absolute pixel floor for all scaled thresholds
    MIN_PIXEL_THRESHOLD = 20

    # Visibility thresholds
    WRIST_VISIBILITY_MIN = 0.3
    ELBOW_VISIBILITY_MIN = 0.3
    SHOULDER_VISIBILITY_MIN = 0.4

    # Bounding box expansion margins for arm extension tolerance
    BBOX_MARGIN_X_FACTOR = 1.0  # 100% of box width
    BBOX_MARGIN_Y_FACTOR = 1.5  # 150% of box height for raised hands

    # Object proximity threshold for backpack detection (proportional to bbox height)
    BACKPACK_PROXIMITY_RATIO = 0.50  # orig 250px / 500px

    # Velocity analysis thresholds (proportional to bbox height per second)
    RAPID_RAISE_VELOCITY_RATIO = 0.30  # orig 150px/s / 500px
    HAND_HISTORY_MAX_LENGTH = 10

    def __init__(
        self,
        settings: Optional[Any] = None,
        session_timeout: float = 10.0,
        coordination_window: float = 5.0,
        get_keypoint_func: Optional[Any] = None
    ):
        """Initialize the GestureDetector.

        Args:
            settings: Optional settings object for additional configuration
            session_timeout: Timeout for gesture session tracking (seconds)
            coordination_window: Time window for coordinated gesture detection (seconds)
            get_keypoint_func: Function to extract keypoints from landmarks by name.
                              Should accept (landmarks, keypoint_name) and return
                              an object with x, y, z, visibility attributes.
        """
        self.settings = settings
        self.session_timeout = session_timeout
        self.coordination_window = coordination_window

        # Session tracking
        self.gesture_sessions: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Recent person activities for temporal suppression
        # Format: {person_idx: {'writing': last_timestamp, 'packing': last_timestamp, ...}}
        self.recent_person_activities: Dict[int, Dict[str, float]] = {}
        self.temporal_suppression_window = 10.0  # seconds

        # Hand position history for velocity analysis
        self.hand_position_history: Dict[int, Dict[str, deque]] = {}

        # Keypoint extraction function
        if get_keypoint_func is None:
            self._get_keypoint = _canonical_get_keypoint
        else:
            self._get_keypoint = get_keypoint_func

    def get_keypoint(self, landmarks: Any, keypoint_name: str) -> Any:
        """Get a keypoint from landmarks by name.

        Delegates to the canonical ``get_keypoint`` in
        ``app.core.utils.pose_utils`` (or a user-supplied override).

        Args:
            landmarks: Pose landmarks object
            keypoint_name: Name of keypoint (e.g., 'nose', 'left_wrist')

        Returns:
            Landmark object with x, y, z, visibility attributes
        """
        return self._get_keypoint(landmarks, keypoint_name)

    def _scale_threshold(self, ratio: float, bbox_height: int) -> int:
        """Scale a proportional threshold by person bounding box height.

        Converts a resolution-independent ratio into an absolute pixel value
        scaled to the detected person's size. Applies a minimum floor to
        prevent thresholds from becoming too small at very low resolutions.

        Args:
            ratio: Proportional threshold relative to bbox height.
                   Positive ratios return positive results (with floor).
                   Negative ratios return negative results (floor applied
                   to magnitude then negated).
            bbox_height: Height of the person bounding box in pixels.

        Returns:
            Scaled pixel threshold, guaranteed to have magnitude >= MIN_PIXEL_THRESHOLD.
        """
        if ratio < 0:
            return -max(self.MIN_PIXEL_THRESHOLD, int(abs(ratio) * bbox_height))
        return max(self.MIN_PIXEL_THRESHOLD, int(ratio * bbox_height))

    def detect_raised_hand(
        self,
        landmarks: Any,
        person_bbox: List[int],
        frame_shape: Tuple[int, ...],
        person_activities: Optional[Dict[str, Any]] = None,
        backpack_detections: Optional[List[Any]] = None,
        person_idx: Optional[int] = None,
        current_timestamp: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Detect if person has raised hand for signaling.

        This method detects deliberate hand-raising gestures used for
        communication signals between crew members. It filters out:
        - Hands reaching toward control panels (forward reach)
        - Hands near backpacks (packing activity)
        - Gestures during work activities (writing, phone use, etc.)

        Args:
            landmarks: Pose landmarks for the person
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width, ...)
            person_activities: Current detected activities for context filtering
            backpack_detections: List of backpack bounding boxes for proximity check
            person_idx: Person index for temporal tracking
            current_timestamp: Current timestamp for temporal suppression

        Returns:
            Tuple of (hand_raised: bool, debug_info: dict)
        """
        if landmarks is None:
            return False, {}

        h, w = frame_shape[:2]

        # Extract key body landmarks
        try:
            right_wrist = self.get_keypoint(landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')
            right_shoulder = self.get_keypoint(landmarks, 'right_shoulder')
            left_shoulder = self.get_keypoint(landmarks, 'left_shoulder')
            right_elbow = self.get_keypoint(landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(landmarks, 'left_elbow')
            right_hip = self.get_keypoint(landmarks, 'right_hip')
            left_hip = self.get_keypoint(landmarks, 'left_hip')
            nose = self.get_keypoint(landmarks, 'nose')
        except (IndexError, AttributeError, ValueError):
            return False, {'error': 'landmark_extraction_failed'}

        # Convert to pixel coordinates
        right_wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_elbow_coords = (int(right_elbow.x * w), int(right_elbow.y * h))
        left_elbow_coords = (int(left_elbow.x * w), int(left_elbow.y * h))
        avg_shoulder_y = (right_shoulder_coords[1] + left_shoulder_coords[1]) / 2

        # Get bounding box coordinates
        mx1, my1, mx2, my2 = person_bbox
        box_width = mx2 - mx1
        box_height = my2 - my1

        # Compute resolution-normalized thresholds scaled by person bbox height
        wrist_shoulder_vertical_min = self._scale_threshold(self.WRIST_SHOULDER_VERTICAL_RATIO, box_height)
        wrist_elbow_distance_min = self._scale_threshold(self.WRIST_ELBOW_DISTANCE_RATIO, box_height)
        arm_extension_min = self._scale_threshold(self.ARM_EXTENSION_RATIO, box_height)
        elbow_shoulder_tolerance = self._scale_threshold(self.ELBOW_SHOULDER_TOLERANCE_RATIO, box_height)
        backpack_proximity_threshold = self._scale_threshold(self.BACKPACK_PROXIMITY_RATIO, box_height)

        # Expand bounding box for arm extension tolerance
        margin_x = box_width * self.BBOX_MARGIN_X_FACTOR
        margin_y = box_height * self.BBOX_MARGIN_Y_FACTOR

        expanded_x1 = mx1 - margin_x
        expanded_y1 = my1 - margin_y
        expanded_x2 = mx2 + margin_x
        expanded_y2 = my2 + margin_y

        # Check if wrists are within expanded box
        right_wrist_in_expanded = (
            expanded_x1 <= right_wrist_coords[0] <= expanded_x2 and
            right_wrist_coords[1] >= expanded_y1
        )
        left_wrist_in_expanded = (
            expanded_x1 <= left_wrist_coords[0] <= expanded_x2 and
            left_wrist_coords[1] >= expanded_y1
        )

        # ==========================================
        # TEMPORAL SUPPRESSION (work activity check)
        # ==========================================
        if person_idx is not None and current_timestamp is not None:
            suppression_result = self._check_temporal_suppression(
                person_idx, current_timestamp
            )
            if suppression_result is not None:
                return False, suppression_result

        # ==========================================
        # CONTEXT-AWARE FILTERING (current activities)
        # ==========================================
        if person_activities:
            context_result = self._check_activity_context(person_activities, person_idx)
            if context_result is not None:
                return False, context_result

        # ==========================================
        # OBJECT PROXIMITY DETECTION (backpack)
        # ==========================================
        if backpack_detections and len(backpack_detections) > 0:
            proximity_result = self._check_backpack_proximity(
                backpack_detections, right_wrist_coords, left_wrist_coords, person_idx,
                backpack_proximity_threshold
            )
            if proximity_result is not None:
                return False, proximity_result

        # ==========================================
        # RAISED HAND DETECTION LOGIC
        # ==========================================

        # Calculate arm measurements
        right_arm_extension = abs(right_wrist_coords[0] - right_shoulder_coords[0])
        left_arm_extension = abs(left_wrist_coords[0] - left_shoulder_coords[0])

        right_wrist_elbow_distance = right_elbow_coords[1] - right_wrist_coords[1]
        left_wrist_elbow_distance = left_elbow_coords[1] - left_wrist_coords[1]

        right_wrist_shoulder_vertical = right_shoulder_coords[1] - right_wrist_coords[1]
        left_wrist_shoulder_vertical = left_shoulder_coords[1] - left_wrist_coords[1]

        # Check if in control zone (forward reach pattern)
        right_in_control_zone = self._is_in_control_zone(
            right_wrist_coords, right_elbow_coords, right_shoulder_coords,
            right_wrist_shoulder_vertical, right_wrist_elbow_distance,
            my1, my2, box_height
        )

        left_in_control_zone = self._is_in_control_zone(
            left_wrist_coords, left_elbow_coords, left_shoulder_coords,
            left_wrist_shoulder_vertical, left_wrist_elbow_distance,
            my1, my2, box_height
        )

        # Right hand raised detection
        right_hand_raised = (
            right_wrist_in_expanded and
            not right_in_control_zone and
            right_wrist_shoulder_vertical > wrist_shoulder_vertical_min and
            right_wrist_elbow_distance > wrist_elbow_distance_min and
            right_arm_extension > arm_extension_min and
            (right_elbow_coords[1] < right_shoulder_coords[1] + elbow_shoulder_tolerance) and
            right_wrist.visibility > self.WRIST_VISIBILITY_MIN and
            right_elbow.visibility > self.ELBOW_VISIBILITY_MIN and
            right_shoulder.visibility > self.SHOULDER_VISIBILITY_MIN and
            0 < right_wrist_coords[0] < w and
            0 < right_wrist_coords[1] < h
        )

        # Left hand raised detection
        left_hand_raised = (
            left_wrist_in_expanded and
            not left_in_control_zone and
            left_wrist_shoulder_vertical > wrist_shoulder_vertical_min and
            left_wrist_elbow_distance > wrist_elbow_distance_min and
            left_arm_extension > arm_extension_min and
            (left_elbow_coords[1] < left_shoulder_coords[1] + elbow_shoulder_tolerance) and
            left_wrist.visibility > self.WRIST_VISIBILITY_MIN and
            left_elbow.visibility > self.ELBOW_VISIBILITY_MIN and
            left_shoulder.visibility > self.SHOULDER_VISIBILITY_MIN and
            0 < left_wrist_coords[0] < w and
            0 < left_wrist_coords[1] < h
        )

        hand_gesture_detected = right_hand_raised or left_hand_raised

        if not hand_gesture_detected:
            return False, {}

        # Analyze velocity if timestamp available
        velocity_analysis = {}
        if person_idx is not None and current_timestamp is not None:
            velocity_analysis = self.analyze_hand_velocity(
                person_idx, landmarks, frame_shape, current_timestamp, box_height
            )

        debug_info = {
            'hand_raised': 'right' if right_hand_raised else 'left',
            'shoulder_y': avg_shoulder_y,
            'wrist_y': right_wrist_coords[1] if right_hand_raised else left_wrist_coords[1],
            'right_wrist_shoulder_vertical': right_wrist_shoulder_vertical,
            'left_wrist_shoulder_vertical': left_wrist_shoulder_vertical,
            'right_wrist_elbow_distance': right_wrist_elbow_distance,
            'left_wrist_elbow_distance': left_wrist_elbow_distance,
            'velocity_analysis': velocity_analysis
        }

        return True, debug_info

    def _is_in_control_zone(
        self,
        wrist_coords: Tuple[int, int],
        elbow_coords: Tuple[int, int],
        shoulder_coords: Tuple[int, int],
        wrist_shoulder_vertical: float,
        wrist_elbow_distance: float,
        bbox_y1: int,
        bbox_y2: int,
        bbox_height: int = 0
    ) -> bool:
        """Check if hand is in control operation zone (forward reach pattern).

        This identifies forward reaches to operate controls vs upward signaling.
        All pixel thresholds are normalized by person bounding box height.

        Args:
            wrist_coords: Wrist position (x, y)
            elbow_coords: Elbow position (x, y)
            shoulder_coords: Shoulder position (x, y)
            wrist_shoulder_vertical: Vertical distance from shoulder to wrist
            wrist_elbow_distance: Vertical distance from elbow to wrist
            bbox_y1: Top of person bounding box
            bbox_y2: Bottom of person bounding box
            bbox_height: Person bounding box height for threshold scaling

        Returns:
            True if hand appears to be in control operation zone
        """
        if bbox_height <= 0:
            bbox_height = bbox_y2 - bbox_y1

        cz_wrist_shoulder_min = self._scale_threshold(self.CONTROL_ZONE_WRIST_SHOULDER_MIN_RATIO, bbox_height)
        cz_wrist_shoulder_max = self._scale_threshold(self.CONTROL_ZONE_WRIST_SHOULDER_MAX_RATIO, bbox_height)
        cz_wrist_elbow_max = self._scale_threshold(self.CONTROL_ZONE_WRIST_ELBOW_MAX_RATIO, bbox_height)
        cz_elbow_shoulder_offset = self._scale_threshold(self.CONTROL_ZONE_ELBOW_SHOULDER_OFFSET_RATIO, bbox_height)

        return (
            # Hand in reasonable vertical range
            wrist_coords[1] > (bbox_y1 + (bbox_y2 - bbox_y1) * 0.2) and
            wrist_coords[1] < (bbox_y1 + (bbox_y2 - bbox_y1) * 0.8) and
            # Wrist above shoulder but not too far (scaled range)
            cz_wrist_shoulder_min < wrist_shoulder_vertical < cz_wrist_shoulder_max and
            # Small wrist-elbow distance (forward reach pattern)
            wrist_elbow_distance < cz_wrist_elbow_max and
            # Elbow below shoulder (forward reach pattern)
            elbow_coords[1] > shoulder_coords[1] + cz_elbow_shoulder_offset
        )

    def _check_temporal_suppression(
        self,
        person_idx: int,
        current_timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """Check if gesture should be suppressed due to recent work activity.

        Args:
            person_idx: Person index
            current_timestamp: Current timestamp

        Returns:
            Suppression info dict if suppressed, None otherwise
        """
        if person_idx not in self.recent_person_activities:
            return None

        recent_acts = self.recent_person_activities[person_idx]

        for activity_type in ['writing', 'packing', 'cell_phone']:
            if activity_type in recent_acts:
                time_since_activity = current_timestamp - recent_acts[activity_type]

                if time_since_activity < self.temporal_suppression_window:
                    return {
                        'suppressed': True,
                        'reason': f'Recent {activity_type} activity ({time_since_activity:.1f}s ago)',
                        'matched_person_idx': person_idx,
                        'time_since_activity': time_since_activity
                    }

        return None

    def _check_activity_context(
        self,
        person_activities: Dict[str, Any],
        person_idx: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Check if gesture should be suppressed due to current activity.

        Args:
            person_activities: Current detected activities
            person_idx: Person index for logging

        Returns:
            Suppression info dict if suppressed, None otherwise
        """
        suppression_activities = [
            ('packing', 'Person engaged in packing bags activity'),
            ('writing', 'Person engaged in writing activity'),
            ('eating_drinking', 'Person engaged in eating/drinking activity'),
            ('cell_phone', 'Person using cell phone'),
        ]

        for activity_key, reason in suppression_activities:
            if person_activities.get(activity_key, False):
                return {
                    'suppressed': True,
                    'reason': reason,
                    'matched_person_idx': person_idx
                }

        return None

    def _check_backpack_proximity(
        self,
        backpack_detections: List[Any],
        right_wrist_coords: Tuple[int, int],
        left_wrist_coords: Tuple[int, int],
        person_idx: Optional[int],
        proximity_threshold: int = 250
    ) -> Optional[Dict[str, Any]]:
        """Check if hands are near backpack (packing activity).

        Args:
            backpack_detections: List of backpack bounding boxes
            right_wrist_coords: Right wrist position
            left_wrist_coords: Left wrist position
            person_idx: Person index for logging
            proximity_threshold: Scaled proximity threshold in pixels

        Returns:
            Suppression info dict if suppressed, None otherwise
        """
        for backpack_bbox in backpack_detections:
            bx1, by1, bx2, by2 = backpack_bbox[:4]
            backpack_center_x = (bx1 + bx2) / 2
            backpack_center_y = (by1 + by2) / 2

            right_dist = np.sqrt(
                (right_wrist_coords[0] - backpack_center_x) ** 2 +
                (right_wrist_coords[1] - backpack_center_y) ** 2
            )
            left_dist = np.sqrt(
                (left_wrist_coords[0] - backpack_center_x) ** 2 +
                (left_wrist_coords[1] - backpack_center_y) ** 2
            )

            if right_dist < proximity_threshold or left_dist < proximity_threshold:
                return {
                    'suppressed': True,
                    'reason': 'Hand near backpack object (likely packing, not signaling)',
                    'matched_person_idx': person_idx,
                    'right_dist_to_backpack': right_dist,
                    'left_dist_to_backpack': left_dist
                }

        return None

    def check_gesture_coordination(
        self,
        lp_gesture: bool,
        alp_gesture: bool,
        timestamp: float
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Check LP/ALP gesture coordination.

        Determines if there is a coordination failure where one person
        raised their hand but the other did not within the coordination window.

        Args:
            lp_gesture: LP hand gesture detected in current frame
            alp_gesture: ALP hand gesture detected in current frame
            timestamp: Current timestamp in seconds

        Returns:
            Tuple of (lp_violation, alp_violation, session_info):
            - lp_violation: True if ALP raised hand but LP failed to coordinate
            - alp_violation: True if LP raised hand but ALP failed to coordinate
            - session_info: Additional session tracking information
        """
        # Get last hand raise times from sessions
        lp_last_raise_time = self.gesture_sessions.get('LP', {}).get('last_raise_time')
        alp_last_raise_time = self.gesture_sessions.get('ALP', {}).get('last_raise_time')

        # Update session if gesture detected
        if lp_gesture:
            self.update_session('LP', True, timestamp)
        if alp_gesture:
            self.update_session('ALP', True, timestamp)

        # Check if both raised hands within coordination window
        def both_within_window(lp_time: Optional[float], alp_time: Optional[float]) -> bool:
            if lp_time is None or alp_time is None:
                return False
            lp_recent = (timestamp - lp_time) <= self.coordination_window
            alp_recent = (timestamp - alp_time) <= self.coordination_window
            return lp_recent and alp_recent

        lp_not_coordinating = False
        alp_not_coordinating = False

        # Guard: if both gestures are detected in the same frame, this is
        # successful coordination -- skip violation checks entirely to avoid
        # a race where pre-update timestamps would incorrectly flag a violation.
        if lp_gesture and alp_gesture:
            pass  # Both coordinated in the same frame -- no violation
        elif alp_gesture and not lp_gesture:
            # ALP raised hand, LP didn't in current frame
            if not both_within_window(lp_last_raise_time, alp_last_raise_time):
                lp_not_coordinating = True
        elif lp_gesture and not alp_gesture:
            # LP raised hand, ALP didn't in current frame
            if not both_within_window(lp_last_raise_time, alp_last_raise_time):
                alp_not_coordinating = True

        session_info = {
            'lp_last_raise': lp_last_raise_time,
            'alp_last_raise': alp_last_raise_time,
            'coordination_window': self.coordination_window,
            'both_coordinated': both_within_window(
                self.gesture_sessions.get('LP', {}).get('last_raise_time'),
                self.gesture_sessions.get('ALP', {}).get('last_raise_time')
            )
        }

        return lp_not_coordinating, alp_not_coordinating, session_info

    def update_session(self, person_role: str, gesture_detected: bool, timestamp: float):
        """Update gesture session tracking.

        Args:
            person_role: Role of the person ('LP' or 'ALP')
            gesture_detected: Whether gesture was detected
            timestamp: Current timestamp
        """
        if person_role not in self.gesture_sessions:
            self.gesture_sessions[person_role] = {}

        session = self.gesture_sessions[person_role]

        if gesture_detected:
            session['last_raise_time'] = timestamp
            session['gesture_count'] = session.get('gesture_count', 0) + 1

        session['last_update'] = timestamp

        # Clean up expired sessions
        self._cleanup_expired_sessions(timestamp)

    def _cleanup_expired_sessions(self, current_time: float):
        """Remove session data older than session_timeout.

        Args:
            current_time: Current timestamp
        """
        expired_keys = []
        for role, session in self.gesture_sessions.items():
            last_update = session.get('last_update', 0)
            if current_time - last_update > self.session_timeout:
                expired_keys.append(role)

        for key in expired_keys:
            del self.gesture_sessions[key]

    def analyze_hand_velocity(
        self,
        person_idx: int,
        landmarks: Any,
        frame_shape: Tuple[int, ...],
        timestamp: float,
        bbox_height: int = 0
    ) -> Dict[str, Any]:
        """Analyze hand velocity and trajectory patterns.

        Detects rapid hand raises (signaling) vs static positions (control operations).
        Velocity threshold is normalized by person bounding box height.

        Args:
            person_idx: Person identifier
            landmarks: Pose landmarks
            frame_shape: Frame dimensions (h, w, ...)
            timestamp: Current timestamp
            bbox_height: Person bounding box height for threshold scaling

        Returns:
            Velocity/trajectory analysis results
        """
        h, w = frame_shape[:2]

        # Initialize history for this person
        if person_idx not in self.hand_position_history:
            self.hand_position_history[person_idx] = {
                'right_wrist': deque(maxlen=self.HAND_HISTORY_MAX_LENGTH),
                'left_wrist': deque(maxlen=self.HAND_HISTORY_MAX_LENGTH),
                'timestamps': deque(maxlen=self.HAND_HISTORY_MAX_LENGTH)
            }

        history = self.hand_position_history[person_idx]

        # Get current wrist positions
        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        right_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

        # Append current positions
        history['right_wrist'].append(right_coords)
        history['left_wrist'].append(left_coords)
        history['timestamps'].append(timestamp)

        # Need at least 3 positions to analyze velocity
        if len(history['timestamps']) < 3:
            return {
                'right_velocity': 0.0,
                'left_velocity': 0.0,
                'right_trajectory': 'unknown',
                'left_trajectory': 'unknown',
                'rapid_raise_detected': False,
                'analysis_quality': 'insufficient_data'
            }

        def calculate_velocity(
            position_history: deque,
            timestamps: deque
        ) -> Tuple[float, str]:
            """Calculate velocity and trajectory direction."""
            if len(position_history) < 2:
                return 0.0, 'unknown'

            recent_positions = list(position_history)[-3:]
            recent_times = list(timestamps)[-3:]

            dx = recent_positions[-1][0] - recent_positions[0][0]
            dy = recent_positions[-1][1] - recent_positions[0][1]
            dt = recent_times[-1] - recent_times[0]

            if dt == 0:
                return 0.0, 'unknown'

            displacement = np.sqrt(dx**2 + dy**2)
            velocity = displacement / dt  # pixels/second

            # Determine trajectory
            if abs(dy) > abs(dx) * 1.5:
                trajectory = 'upward' if dy < 0 else 'downward'  # Y increases downward
            elif abs(dx) > abs(dy) * 1.5:
                trajectory = 'lateral'
            else:
                trajectory = 'diagonal'

            return velocity, trajectory

        right_vel, right_traj = calculate_velocity(history['right_wrist'], history['timestamps'])
        left_vel, left_traj = calculate_velocity(history['left_wrist'], history['timestamps'])

        # Detect rapid hand raise (velocity threshold scaled by bbox height)
        rapid_raise_velocity = self._scale_threshold(self.RAPID_RAISE_VELOCITY_RATIO, bbox_height) if bbox_height > 0 else self.MIN_PIXEL_THRESHOLD
        rapid_raise = (
            (right_vel > rapid_raise_velocity and right_traj == 'upward') or
            (left_vel > rapid_raise_velocity and left_traj == 'upward')
        )

        return {
            'right_velocity': right_vel,
            'left_velocity': left_vel,
            'right_trajectory': right_traj,
            'left_trajectory': left_traj,
            'rapid_raise_detected': rapid_raise,
            'analysis_quality': 'good' if len(history['timestamps']) >= 5 else 'limited'
        }

    def update_recent_activity(
        self,
        person_idx: int,
        activity_type: str,
        timestamp: float
    ):
        """Update recent activity record for temporal suppression.

        Args:
            person_idx: Person identifier
            activity_type: Type of activity ('writing', 'packing', 'cell_phone')
            timestamp: Timestamp when activity was detected
        """
        if person_idx not in self.recent_person_activities:
            self.recent_person_activities[person_idx] = {}

        self.recent_person_activities[person_idx][activity_type] = timestamp

    def assign_role_by_camera_angle(
        self,
        person_bbox: List[int],
        frame_width: int,
        camera_position: str = 'center'
    ) -> str:
        """Assign LP/ALP role based on person position and camera angle.

        In a locomotive cab, LP typically sits on the left side and ALP on the right
        (or vice versa depending on country/railroad). This method assigns roles
        based on horizontal position in the frame.

        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame_width: Width of the frame
            camera_position: Camera mounting position ('center', 'left', 'right')

        Returns:
            'LP' or 'ALP' based on position
        """
        # Calculate person center X
        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        frame_center_x = frame_width / 2

        # Default: LP on left side of frame, ALP on right
        if camera_position == 'center':
            return 'LP' if person_center_x < frame_center_x else 'ALP'
        elif camera_position == 'left':
            # Camera on left - person closer to camera is LP
            return 'LP' if person_center_x < frame_center_x else 'ALP'
        elif camera_position == 'right':
            # Camera on right - person farther from camera is LP
            return 'ALP' if person_center_x < frame_center_x else 'LP'
        else:
            return 'LP' if person_center_x < frame_center_x else 'ALP'

    def reset(self):
        """Reset all session tracking data."""
        self.gesture_sessions.clear()
        self.recent_person_activities.clear()
        self.hand_position_history.clear()
