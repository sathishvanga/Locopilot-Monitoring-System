"""Activity detection for writing, cell phone, and packing bags.

This module provides detection logic for various activities including:
- Writing detection (book interaction, wrist proximity, posture)
- Cell phone detection (phone proximity to person)
- Packing bags detection (wrist inside backpack, hand motion analysis)
"""
from typing import Dict, List, Any, Optional, Tuple
import math
import logging
from collections import deque, defaultdict

import numpy as np

from app.core.utils.geometry import bbox_overlap_with_margin, calculate_iou


class ActivityDetector:
    """Detects activities: writing, cell phone usage, packing bags.

    This class extracts activity detection logic from the main monitor,
    providing a cleaner interface for detecting specific activities based
    on pose landmarks and object detections.
    """

    # Default thresholds (can be overridden by settings)
    DEFAULT_WRITING_WRIST_DISTANCE = 300
    DEFAULT_RELAXED_WRIST_DISTANCE = 400
    DEFAULT_WRIST_VISIBILITY_THRESHOLD = 0.3
    DEFAULT_ELBOW_VISIBILITY_THRESHOLD = 0.25
    DEFAULT_HEAD_DOWN_THRESHOLD = 0.01

    # Default activity margins
    DEFAULT_CELL_PHONE_MARGIN = 180
    DEFAULT_WRITING_MARGIN = 180
    DEFAULT_PACKING_MARGIN = 100
    DEFAULT_PACKING_REGION_MARGIN = 150

    def __init__(self, settings: Optional[Any] = None, get_keypoint_func: Optional[Any] = None):
        """Initialize the activity detector.

        Args:
            settings: Configuration settings object with detection thresholds.
            get_keypoint_func: Function to get keypoints from landmarks by name.
                              Should have signature: (landmarks, keypoint_name: str) -> landmark
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Store keypoint accessor function
        self._get_keypoint_func = get_keypoint_func

        # Initialize thresholds from settings or use defaults
        self._init_thresholds()

        # Packing motion history for temporal analysis
        # Format: {person_idx: {'distances': deque, 'timestamps': deque, 'active_hand': deque}}
        self.packing_motion_history: Dict[int, Dict[str, deque]] = defaultdict(dict)

    def _init_thresholds(self) -> None:
        """Initialize detection thresholds from settings or defaults."""
        if self.settings:
            self.writing_wrist_distance = getattr(
                self.settings, 'writing_wrist_distance', self.DEFAULT_WRITING_WRIST_DISTANCE
            )
            self.relaxed_wrist_distance = getattr(
                self.settings, 'relaxed_wrist_distance', self.DEFAULT_RELAXED_WRIST_DISTANCE
            )
            self.wrist_visibility_threshold = getattr(
                self.settings, 'wrist_visibility_threshold', self.DEFAULT_WRIST_VISIBILITY_THRESHOLD
            )
            self.elbow_visibility_threshold = getattr(
                self.settings, 'elbow_visibility_threshold', self.DEFAULT_ELBOW_VISIBILITY_THRESHOLD
            )
            self.head_down_threshold = getattr(
                self.settings, 'head_down_threshold', self.DEFAULT_HEAD_DOWN_THRESHOLD
            )
            self.cell_phone_margin = getattr(
                self.settings, 'activity_cell_phone_margin', self.DEFAULT_CELL_PHONE_MARGIN
            )
            self.writing_margin = getattr(
                self.settings, 'activity_writing_margin', self.DEFAULT_WRITING_MARGIN
            )
            self.packing_margin = getattr(
                self.settings, 'activity_packing_margin', self.DEFAULT_PACKING_MARGIN
            )
            self.packing_region_margin = getattr(
                self.settings, 'activity_packing_region_margin', self.DEFAULT_PACKING_REGION_MARGIN
            )
        else:
            self.writing_wrist_distance = self.DEFAULT_WRITING_WRIST_DISTANCE
            self.relaxed_wrist_distance = self.DEFAULT_RELAXED_WRIST_DISTANCE
            self.wrist_visibility_threshold = self.DEFAULT_WRIST_VISIBILITY_THRESHOLD
            self.elbow_visibility_threshold = self.DEFAULT_ELBOW_VISIBILITY_THRESHOLD
            self.head_down_threshold = self.DEFAULT_HEAD_DOWN_THRESHOLD
            self.cell_phone_margin = self.DEFAULT_CELL_PHONE_MARGIN
            self.writing_margin = self.DEFAULT_WRITING_MARGIN
            self.packing_margin = self.DEFAULT_PACKING_MARGIN
            self.packing_region_margin = self.DEFAULT_PACKING_REGION_MARGIN

    def get_keypoint(self, landmarks: Any, keypoint_name: str) -> Any:
        """Get a keypoint from landmarks by name.

        Args:
            landmarks: Pose landmarks (YoloPoseLandmarks or similar)
            keypoint_name: Name of keypoint (e.g., 'left_wrist', 'nose')

        Returns:
            Landmark object with x, y, visibility attributes
        """
        if self._get_keypoint_func is not None:
            return self._get_keypoint_func(landmarks, keypoint_name)

        # Fallback: try to access by index using YOLO keypoint mapping
        from app.services.yolo_pose_adapter import get_keypoint_by_name
        return get_keypoint_by_name(landmarks, keypoint_name)

    def calculate_wrist_distance(
        self, pose_landmarks: Any, frame_shape: Tuple[int, ...]
    ) -> Tuple[Optional[float], Optional[str]]:
        """Calculate Euclidean distance between left and right wrists.

        Falls back to elbow distance if wrists are not visible.

        Args:
            pose_landmarks: Pose landmarks with visibility scores
            frame_shape: Tuple of (height, width) of the frame

        Returns:
            tuple: (distance in pixels, source) where source is 'wrist', 'elbow',
                   or 'single_wrist', or (None, None) if not detectable
        """
        if not pose_landmarks:
            return None, None

        try:
            landmarks = pose_landmarks.landmark if hasattr(pose_landmarks, 'landmark') else pose_landmarks
            h, w = frame_shape[:2]

            # Get wrist landmarks
            right_wrist = self.get_keypoint(landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')

            # Try wrists first (primary method)
            if (right_wrist.visibility >= self.wrist_visibility_threshold and
                left_wrist.visibility >= self.wrist_visibility_threshold):
                # Convert normalized coordinates to pixel coordinates
                right_wrist_px = (right_wrist.x * w, right_wrist.y * h)
                left_wrist_px = (left_wrist.x * w, left_wrist.y * h)

                # Calculate Euclidean distance
                distance = np.sqrt(
                    (right_wrist_px[0] - left_wrist_px[0])**2 +
                    (right_wrist_px[1] - left_wrist_px[1])**2
                )
                return distance, 'wrist'

            # FALLBACK: Use elbows if wrists not visible
            right_elbow = self.get_keypoint(landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(landmarks, 'left_elbow')

            if (right_elbow.visibility >= self.elbow_visibility_threshold and
                left_elbow.visibility >= self.elbow_visibility_threshold):
                right_elbow_px = (right_elbow.x * w, right_elbow.y * h)
                left_elbow_px = (left_elbow.x * w, left_elbow.y * h)
                distance = np.sqrt(
                    (right_elbow_px[0] - left_elbow_px[0])**2 +
                    (right_elbow_px[1] - left_elbow_px[1])**2
                )
                return distance, 'elbow'

            # FALLBACK: Single wrist to shoulder midpoint
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

    def detect_head_looking_down(self, pose_landmarks: Any) -> bool:
        """Check if head is tilted down (looking at lap area).

        Uses nose position relative to eyes to detect downward head tilt,
        which indicates reading/writing posture.

        Args:
            pose_landmarks: Pose landmarks

        Returns:
            bool: True if head is looking down, False otherwise
        """
        try:
            nose = self.get_keypoint(pose_landmarks, 'nose')
            left_eye = self.get_keypoint(pose_landmarks, 'left_eye')
            right_eye = self.get_keypoint(pose_landmarks, 'right_eye')

            if nose is None or left_eye is None or right_eye is None:
                return False

            # Calculate average eye Y position
            eye_y = (left_eye.y + right_eye.y) / 2

            # Head is looking down when nose is significantly below eye line
            return nose.y > eye_y + self.head_down_threshold

        except Exception as e:
            self.logger.debug(f"Exception in detect_head_looking_down: {e}")
            return False

    def detect_writing_posture(
        self, pose_landmarks: Any, frame_shape: Tuple[int, ...]
    ) -> bool:
        """Instantly detect writing posture based on hand position.

        Checks if hands are in typical writing position using multiple criteria:
        1. Hands below shoulders (relaxed check for camera angles)
        2. Hands in lap area (strict check)
        3. Head looking down (indicates reading/writing posture)

        Args:
            pose_landmarks: Pose landmarks
            frame_shape: Tuple of (height, width) of the frame

        Returns:
            bool: True if writing posture detected, False otherwise
        """
        if not pose_landmarks:
            return False

        try:
            h, w = frame_shape[:2]

            # Get key body points
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_hip = self.get_keypoint(pose_landmarks, 'left_hip')
            right_hip = self.get_keypoint(pose_landmarks, 'right_hip')
            left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
            right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')

            # Calculate vertical positions (normalized 0-1)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            left_wrist_y = left_wrist.y
            right_wrist_y = right_wrist.y

            # Option 1: Lap area (strict) - wrists below hips
            left_in_lap = left_wrist_y > hip_y
            right_in_lap = right_wrist_y > hip_y

            # Option 2: Below shoulders (relaxed)
            left_below_shoulders = left_wrist_y > shoulder_y
            right_below_shoulders = right_wrist_y > shoulder_y

            # Calculate wrist distance (in pixels)
            left_wrist_x = int(left_wrist.x * w)
            left_wrist_y_px = int(left_wrist.y * h)
            right_wrist_x = int(right_wrist.x * w)
            right_wrist_y_px = int(right_wrist.y * h)

            wrist_distance = ((left_wrist_x - right_wrist_x) ** 2 +
                            (left_wrist_y_px - right_wrist_y_px) ** 2) ** 0.5

            # Check if head is looking down
            head_looking_down = self.detect_head_looking_down(pose_landmarks)

            # Writing posture criteria:
            hands_in_lap = left_in_lap and right_in_lap
            hands_below_shoulders = left_below_shoulders and right_below_shoulders
            hands_in_writing_position = hands_in_lap or hands_below_shoulders

            # Detect if wrists are close enough
            wrists_close = wrist_distance <= self.writing_wrist_distance

            # Writing detected if:
            # - Hands below shoulders + wrists close, OR
            # - Head looking down + hands below shoulders (wider tolerance)
            if hands_in_writing_position and wrists_close:
                return True

            if head_looking_down and hands_below_shoulders and wrist_distance <= self.relaxed_wrist_distance:
                return True

            return False

        except Exception as e:
            self.logger.debug(f"Exception in detect_writing_posture: {e}")
            return False

    def detect_writing(
        self, landmarks: Any, book_detections: List,
        person_bbox: List[int], frame_shape: Tuple[int, ...]
    ) -> Tuple[bool, Dict]:
        """Detect writing activity.

        Combines book proximity detection with writing posture analysis.

        Args:
            landmarks: Pose landmarks for the person
            book_detections: List of book bounding boxes [x1, y1, x2, y2]
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width, ...)

        Returns:
            Tuple of (detected: bool, evidence: dict)
        """
        evidence = {
            'book_detected': False,
            'hand_near_book': False,
            'writing_posture': False,
            'method': 'none'
        }

        if landmarks is None:
            return False, evidence

        h, w = frame_shape[:2]

        # Check for writing posture
        posture_detected = self.detect_writing_posture(landmarks, frame_shape)
        evidence['writing_posture'] = posture_detected

        # Check book interaction if books are detected
        if book_detections:
            evidence['book_detected'] = True

            right_wrist = self.get_keypoint(landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')

            wrists_visible = (right_wrist.visibility >= 0.5 or left_wrist.visibility >= 0.5)

            if wrists_visible:
                right_hand_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
                left_hand_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

                # Use larger margin for book-to-person association
                person_book_margin = 250

                for book_bbox in book_detections:
                    # Check if book is in this person's region
                    book_in_region = bbox_overlap_with_margin(book_bbox, person_bbox, person_book_margin)

                    if book_in_region:
                        # Check hands for interaction with book
                        right_near = (right_wrist.visibility >= 0.5 and
                                     self.check_hand_object_interaction(right_hand_coords, book_bbox, self.writing_margin))
                        left_near = (left_wrist.visibility >= 0.5 and
                                    self.check_hand_object_interaction(left_hand_coords, book_bbox, self.writing_margin))

                        if right_near or left_near:
                            evidence['hand_near_book'] = True
                            evidence['method'] = 'book_hand'
                            return True, evidence

        # Fallback to posture-only detection
        if posture_detected:
            evidence['method'] = 'posture_only'
            return True, evidence

        return False, evidence

    def check_hand_object_interaction(
        self, hand_coords: Tuple[float, float],
        object_bbox: List[int],
        margin: int = 50
    ) -> bool:
        """Check if hand is interacting with an object.

        Args:
            hand_coords: (x, y) coordinates of hand
            object_bbox: [x1, y1, x2, y2] bounding box of object
            margin: proximity margin in pixels (default 50)

        Returns:
            True if hand is within margin of object bbox
        """
        if hand_coords is None or object_bbox is None:
            return False

        hx, hy = hand_coords
        x1, y1, x2, y2 = object_bbox[:4]
        return (x1 - margin <= hx <= x2 + margin and
                y1 - margin <= hy <= y2 + margin)

    def detect_cell_phone(
        self, phone_detections: List, landmarks: Any,
        person_bbox: List[int], frame_shape: Tuple[int, ...],
        margin: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """Detect cell phone usage near person.

        Checks if any detected cell phone is in proximity to the person's
        hands within their bounding box region.

        Args:
            phone_detections: List of phone bounding boxes [x1, y1, x2, y2]
            landmarks: Pose landmarks for the person
            person_bbox: Person bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width, ...)
            margin: Proximity margin in pixels (default from settings)

        Returns:
            Tuple of (detected: bool, evidence: dict)
        """
        evidence = {
            'phones_in_frame': len(phone_detections),
            'phone_in_region': False,
            'hand_near_phone': False,
            'which_hand': None
        }

        if not phone_detections or landmarks is None:
            return False, evidence

        if margin is None:
            margin = self.cell_phone_margin

        h, w = frame_shape[:2]

        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        right_hand_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_hand_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

        # Use stricter margin for phone detection
        check_margin = min(margin, 100)

        for phone_bbox in phone_detections:
            # Check if phone bbox overlaps with person bbox
            phone_in_region = bbox_overlap_with_margin(phone_bbox, person_bbox, check_margin)

            if phone_in_region:
                evidence['phone_in_region'] = True

                # Check if hand is near the phone
                right_near = self.check_hand_object_interaction(right_hand_coords, phone_bbox, check_margin)
                left_near = self.check_hand_object_interaction(left_hand_coords, phone_bbox, check_margin)

                if right_near or left_near:
                    evidence['hand_near_phone'] = True
                    evidence['which_hand'] = 'right' if right_near else 'left'
                    return True, evidence

        return False, evidence

    def is_wrist_inside_backpack(
        self, wrist_coords: Optional[Tuple[float, float]],
        backpack_bbox: List[int],
        margin: int = 30
    ) -> Tuple[bool, float]:
        """Check if wrist keypoint is inside or very close to backpack bounding box.

        SIMPLIFIED PACKING DETECTION: If wrist is inside/near backpack bbox, packing detected.

        Args:
            wrist_coords: (x, y) coordinates of wrist keypoint
            backpack_bbox: [x1, y1, x2, y2] bounding box of backpack/bag
            margin: additional margin around bbox (default 30px for tight detection)

        Returns:
            tuple: (is_inside, distance_to_center)
                - is_inside: True if wrist is inside/near backpack bbox
                - distance_to_center: Distance from wrist to backpack center (for confidence)
        """
        if wrist_coords is None or backpack_bbox is None:
            return False, float('inf')

        wx, wy = wrist_coords
        x1, y1, x2, y2 = backpack_bbox[:4]

        # Calculate backpack center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate distance to center
        distance = math.sqrt((wx - center_x) ** 2 + (wy - center_y) ** 2)

        # Check if wrist is inside bbox (with margin)
        is_inside = (x1 - margin <= wx <= x2 + margin and
                     y1 - margin <= wy <= y2 + margin)

        return is_inside, distance

    def analyze_packing_hand_motion(
        self, person_idx: int, landmarks: Any,
        frame_shape: Tuple[int, ...], timestamp_sec: float,
        backpack_bbox: List[int]
    ) -> Dict[str, Any]:
        """Analyze hand motion patterns to detect actual packing activity.

        Detects repeated back-and-forth movement between body and backpack:
        - Tracks hand distance from backpack center over time
        - Identifies direction changes (closer -> farther -> closer -> farther)
        - Validates moderate velocity (15-200 pixels per second)
        - Requires at least 1 direction change for packing pattern

        Args:
            person_idx: Person identifier
            landmarks: Pose landmarks
            frame_shape: (height, width, channels)
            timestamp_sec: Current timestamp
            backpack_bbox: (x1, y1, x2, y2) backpack bounding box

        Returns:
            dict with motion analysis results:
            - 'packing_motion_detected': bool
            - 'direction_changes': int
            - 'avg_velocity': float (pixels per second)
            - 'history_length': int
            - 'hand_consistency': bool
            - 'sustained_proximity': bool
            - 'reason': str
        """
        h, w = frame_shape[:2]

        # Calculate backpack center
        bp_x1, bp_y1, bp_x2, bp_y2 = backpack_bbox[:4]
        backpack_center_x = (bp_x1 + bp_x2) / 2
        backpack_center_y = (bp_y1 + bp_y2) / 2

        # Get current hand positions (use closer hand to backpack)
        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        right_x, right_y = int(right_wrist.x * w), int(right_wrist.y * h)
        left_x, left_y = int(left_wrist.x * w), int(left_wrist.y * h)

        # Calculate distances to backpack center
        right_dist = math.sqrt((right_x - backpack_center_x)**2 + (right_y - backpack_center_y)**2)
        left_dist = math.sqrt((left_x - backpack_center_x)**2 + (left_y - backpack_center_y)**2)

        # Use the hand closer to backpack
        current_distance = min(right_dist, left_dist)
        active_hand = 'right' if right_dist < left_dist else 'left'

        # Initialize packing motion history for this person
        if person_idx not in self.packing_motion_history:
            self.packing_motion_history[person_idx] = {
                'distances': deque(maxlen=6),  # Track last 6 frames
                'timestamps': deque(maxlen=6),
                'active_hand': deque(maxlen=6)
            }

        history = self.packing_motion_history[person_idx]

        # Add current position
        history['distances'].append(current_distance)
        history['timestamps'].append(timestamp_sec)
        history['active_hand'].append(active_hand)

        # Need at least 3 samples to detect pattern
        if len(history['distances']) < 3:
            return {
                'packing_motion_detected': False,
                'direction_changes': 0,
                'avg_velocity': 0.0,
                'history_length': len(history['distances']),
                'hand_consistency': True,
                'active_hand': active_hand,
                'sustained_proximity': False,
                'sustained_proximity_time': False,
                'time_span': 0,
                'reason': 'insufficient_history'
            }

        # Analyze motion pattern
        distances = list(history['distances'])

        # Calculate velocity (pixels per second)
        velocities = []
        for i in range(1, len(distances)):
            distance_change = abs(distances[i] - distances[i-1])
            time_diff = history['timestamps'][i] - history['timestamps'][i-1]
            if time_diff > 0:
                velocity = distance_change / time_diff
            else:
                velocity = distance_change / 2.0  # Fallback
            velocities.append(velocity)

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0

        # Detect direction changes
        direction_changes = 0
        prev_direction = None

        for i in range(1, len(distances)):
            current_direction = 'closer' if distances[i] < distances[i-1] else 'farther'
            if prev_direction and current_direction != prev_direction:
                direction_changes += 1
            prev_direction = current_direction

        # Check hand consistency (at least 2 of last 3 frames use same hand)
        recent_hands = list(history['active_hand'])[-3:]
        hand_consistency = recent_hands.count(active_hand) >= 2 if len(recent_hands) >= 2 else True

        # Check sustained proximity
        proximity_threshold = 100  # pixels
        sustained_proximity = all(d < proximity_threshold for d in distances[-3:]) if len(distances) >= 3 else False

        # Calculate time span
        time_span = history['timestamps'][-1] - history['timestamps'][0] if len(history['timestamps']) >= 2 else 0
        sustained_proximity_time = time_span >= 4.0  # 4+ seconds

        # Packing detected if:
        # - Motion pattern detected (direction changes + velocity + consistency), OR
        # - Sustained proximity for 4+ seconds
        packing_detected = (
            (direction_changes >= 1 and
             15 <= avg_velocity <= 200 and
             hand_consistency) or
            (sustained_proximity and sustained_proximity_time)
        )

        return {
            'packing_motion_detected': packing_detected,
            'direction_changes': direction_changes,
            'avg_velocity': avg_velocity,
            'history_length': len(history['distances']),
            'hand_consistency': hand_consistency,
            'active_hand': active_hand,
            'sustained_proximity': sustained_proximity,
            'sustained_proximity_time': sustained_proximity_time,
            'time_span': time_span,
            'reason': 'valid_pattern' if packing_detected else ('sustained_proximity' if (sustained_proximity and sustained_proximity_time) else 'no_packing_pattern')
        }

    def detect_packing_bags(
        self, landmarks: Any, bag_detections: List,
        person_bbox: List[int], person_idx: int,
        timestamp: float, frame_shape: Tuple[int, ...]
    ) -> Tuple[bool, Dict]:
        """Detect packing bags activity.

        Uses two methods:
        1. Simple detection: wrist inside backpack bounding box
        2. Motion analysis: hand motion patterns near backpack

        Args:
            landmarks: Pose landmarks for the person
            bag_detections: List of backpack bounding boxes [x1, y1, x2, y2]
            person_bbox: Person bounding box [x1, y1, x2, y2]
            person_idx: Person identifier for motion history tracking
            timestamp: Current timestamp in seconds
            frame_shape: Frame dimensions (height, width, ...)

        Returns:
            Tuple of (detected: bool, evidence: dict)
        """
        evidence = {
            'bags_in_frame': len(bag_detections),
            'bag_in_region': False,
            'wrist_inside': False,
            'motion_detected': False,
            'method': 'none',
            'wrist_check': {},
            'motion_analysis': {}
        }

        if not bag_detections or landmarks is None:
            return False, evidence

        h, w = frame_shape[:2]

        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        # Check wrist visibility
        right_visible = right_wrist.visibility > 0.3
        left_visible = left_wrist.visibility > 0.3

        # Get hand coordinates if visible
        right_hand_coords = None
        left_hand_coords = None

        if right_visible:
            right_hand_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        else:
            # Fallback to elbow
            right_elbow = self.get_keypoint(landmarks, 'right_elbow')
            if right_elbow.visibility > 0.3:
                right_hand_coords = (int(right_elbow.x * w), int(right_elbow.y * h))

        if left_visible:
            left_hand_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        else:
            # Fallback to elbow
            left_elbow = self.get_keypoint(landmarks, 'left_elbow')
            if left_elbow.visibility > 0.3:
                left_hand_coords = (int(left_elbow.x * w), int(left_elbow.y * h))

        for backpack_bbox in bag_detections:
            # Check if backpack is in this person's region
            bag_in_region = bbox_overlap_with_margin(
                backpack_bbox, person_bbox, self.packing_region_margin
            )

            if not bag_in_region:
                continue

            evidence['bag_in_region'] = True

            # Method 1: Wrist inside backpack bbox
            right_inside, right_dist = self.is_wrist_inside_backpack(
                right_hand_coords, backpack_bbox, margin=40
            )
            left_inside, left_dist = self.is_wrist_inside_backpack(
                left_hand_coords, backpack_bbox, margin=40
            )

            evidence['wrist_check'] = {
                'right_wrist_inside': right_inside,
                'left_wrist_inside': left_inside,
                'right_dist': right_dist,
                'left_dist': left_dist,
                'closest_distance': min(right_dist, left_dist),
                'backpack_bbox': list(backpack_bbox[:4])
            }

            if right_inside or left_inside:
                evidence['wrist_inside'] = True
                evidence['method'] = 'wrist_inside'
                return True, evidence

            # Method 2: Hand near backpack with motion analysis
            right_near = self.check_hand_object_interaction(
                right_hand_coords, backpack_bbox, self.packing_margin
            )
            left_near = self.check_hand_object_interaction(
                left_hand_coords, backpack_bbox, self.packing_margin
            )

            if right_near or left_near:
                motion_analysis = self.analyze_packing_hand_motion(
                    person_idx, landmarks, frame_shape, timestamp, backpack_bbox
                )
                evidence['motion_analysis'] = motion_analysis

                if motion_analysis['packing_motion_detected']:
                    evidence['motion_detected'] = True
                    evidence['method'] = 'motion_analysis'
                    return True, evidence

                # Check sustained proximity fallback
                if (motion_analysis.get('sustained_proximity', False) and
                    motion_analysis.get('sustained_proximity_time', False)):
                    evidence['motion_detected'] = True
                    evidence['method'] = 'sustained_proximity'
                    return True, evidence

        return False, evidence

    def clear_motion_history(self, person_idx: Optional[int] = None) -> None:
        """Clear packing motion history.

        Args:
            person_idx: If provided, clear only for this person.
                       If None, clear all history.
        """
        if person_idx is not None:
            if person_idx in self.packing_motion_history:
                del self.packing_motion_history[person_idx]
        else:
            self.packing_motion_history.clear()
