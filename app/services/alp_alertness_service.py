"""
ALP alertness service - Detects if ALP is standing using pose estimation

This service checks if the Assistant Loco Pilot (ALP) is standing during
the pre-arrival window (30-60 seconds before station arrival).
"""

import logging
import threading
from typing import Dict, Optional, Tuple
import numpy as np

from ..utils.config import get_settings

logger = logging.getLogger(__name__)


class ALPAlertnessService:
    """
    Service for detecting if ALP is standing using pose estimation

    Uses YOLO-Pose keypoints to determine if a person is in standing position:
    - Checks hip-to-ankle distance ratio
    - Analyzes shoulder position in frame
    - Returns confidence score for standing detection

    YOLO-Pose keypoint indices:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder
    7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist
    11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee
    15: left_ankle, 16: right_ankle
    """

    # Keypoint indices for pose analysis
    KEYPOINTS = {
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

    def __init__(self):
        """Initialize the ALP alertness service"""
        self.settings = get_settings()

        # Standing detection thresholds
        # Hip-to-ankle distance ratio threshold (higher = standing, lower = sitting)
        self.standing_ratio_threshold = 0.35  # Hip-ankle distance should be > 35% of frame height
        self.min_keypoint_confidence = 0.5  # Minimum confidence for keypoints

        # Shoulder position threshold (% of frame height from top)
        # When standing, shoulders should be in upper portion of frame
        self.shoulder_height_threshold = 0.6  # Shoulders above 60% mark from top

        logger.info(
            f"ALPAlertnessService initialized - "
            f"standing_ratio_threshold: {self.standing_ratio_threshold}, "
            f"shoulder_height_threshold: {self.shoulder_height_threshold}"
        )

    def is_alp_standing(
        self,
        pose_keypoints: np.ndarray,
        frame_shape: Tuple[int, int, int],
        person_role: Optional[str] = None
    ) -> Dict:
        """
        Detect if a person (ALP) is standing using pose estimation

        Args:
            pose_keypoints: YOLO-Pose keypoints array (17, 3) - (x, y, confidence)
            frame_shape: Frame shape (height, width, channels)
            person_role: Role of the person (for logging)

        Returns:
            Dictionary with:
                - is_standing: bool
                - confidence: float (0-1)
                - reason: str
                - hip_ankle_ratio: float
                - shoulder_position: float
        """
        frame_height, frame_width = frame_shape[:2]

        result = {
            'is_standing': False,
            'confidence': 0.0,
            'reason': 'Unable to determine',
            'hip_ankle_ratio': 0.0,
            'shoulder_position': 0.0,
            'keypoints_valid': False
        }

        try:
            # Validate keypoints shape
            if pose_keypoints is None or len(pose_keypoints) < 17:
                result['reason'] = 'Invalid or missing keypoints'
                return result

            # Extract relevant keypoints with confidence
            left_hip = self._get_keypoint(pose_keypoints, 'left_hip')
            right_hip = self._get_keypoint(pose_keypoints, 'right_hip')
            left_ankle = self._get_keypoint(pose_keypoints, 'left_ankle')
            right_ankle = self._get_keypoint(pose_keypoints, 'right_ankle')
            left_shoulder = self._get_keypoint(pose_keypoints, 'left_shoulder')
            right_shoulder = self._get_keypoint(pose_keypoints, 'right_shoulder')

            # Check if we have enough valid keypoints
            hips_valid = left_hip is not None or right_hip is not None
            ankles_valid = left_ankle is not None or right_ankle is not None
            shoulders_valid = left_shoulder is not None or right_shoulder is not None

            if not (hips_valid and ankles_valid and shoulders_valid):
                result['reason'] = 'Insufficient keypoint visibility'
                return result

            result['keypoints_valid'] = True

            # Calculate average hip position
            hip_y = self._average_y([left_hip, right_hip])

            # Calculate average ankle position
            ankle_y = self._average_y([left_ankle, right_ankle])

            # Calculate average shoulder position
            shoulder_y = self._average_y([left_shoulder, right_shoulder])

            # Metric 1: Hip-to-ankle distance ratio
            # Standing: large distance (legs extended)
            # Sitting: small distance (legs bent)
            hip_ankle_distance = abs(ankle_y - hip_y)
            hip_ankle_ratio = hip_ankle_distance / frame_height
            result['hip_ankle_ratio'] = hip_ankle_ratio

            # Metric 2: Shoulder position in frame
            # Standing: shoulders higher in frame (lower y value)
            # Sitting: shoulders lower in frame (higher y value)
            shoulder_position = shoulder_y / frame_height
            result['shoulder_position'] = shoulder_position

            # Combine metrics for standing detection
            standing_by_ratio = hip_ankle_ratio >= self.standing_ratio_threshold
            standing_by_shoulder = shoulder_position <= self.shoulder_height_threshold

            # Both metrics should indicate standing for high confidence
            if standing_by_ratio and standing_by_shoulder:
                result['is_standing'] = True
                result['confidence'] = 0.9
                result['reason'] = 'Both hip-ankle ratio and shoulder position indicate standing'
            elif standing_by_ratio:
                result['is_standing'] = True
                result['confidence'] = 0.6
                result['reason'] = 'Hip-ankle ratio indicates standing'
            elif standing_by_shoulder:
                result['is_standing'] = True
                result['confidence'] = 0.5
                result['reason'] = 'Shoulder position indicates standing'
            else:
                result['is_standing'] = False
                result['confidence'] = 0.8
                result['reason'] = (
                    f'Sitting detected - hip-ankle ratio: {hip_ankle_ratio:.2f} '
                    f'(threshold: {self.standing_ratio_threshold}), '
                    f'shoulder pos: {shoulder_position:.2f} '
                    f'(threshold: {self.shoulder_height_threshold})'
                )

            # Log detection for debugging
            role_str = f" ({person_role})" if person_role else ""
            logger.debug(
                f"Standing detection{role_str}: {result['is_standing']} "
                f"(confidence: {result['confidence']:.2f}) - {result['reason']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in standing detection: {e}")
            result['reason'] = f'Error: {str(e)}'
            return result

    def _get_keypoint(
        self,
        keypoints: np.ndarray,
        name: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get a keypoint by name with confidence check

        Args:
            keypoints: Keypoints array
            name: Keypoint name

        Returns:
            Tuple of (x, y, confidence) or None if not valid
        """
        idx = self.KEYPOINTS.get(name)
        if idx is None or idx >= len(keypoints):
            return None

        kp = keypoints[idx]
        x, y = kp[0], kp[1]
        confidence = kp[2] if len(kp) > 2 else 1.0

        if confidence < self.min_keypoint_confidence:
            return None

        return (x, y, confidence)

    def _average_y(self, keypoints: list) -> float:
        """
        Calculate average y position from valid keypoints

        Args:
            keypoints: List of keypoints (may contain None)

        Returns:
            Average y position
        """
        valid = [kp[1] for kp in keypoints if kp is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)


# Global service instance
_alp_alertness_service: Optional[ALPAlertnessService] = None
_alp_alertness_service_lock = threading.Lock()


def get_alp_alertness_service() -> ALPAlertnessService:
    """
    Get the global ALP alertness service instance.

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        ALPAlertnessService instance
    """
    global _alp_alertness_service
    if _alp_alertness_service is None:
        with _alp_alertness_service_lock:
            if _alp_alertness_service is None:
                _alp_alertness_service = ALPAlertnessService()
    return _alp_alertness_service
