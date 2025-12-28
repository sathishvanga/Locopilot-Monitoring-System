"""
Burst Verification Service - Native Frame Verification for Activity Detection.

When an activity is first detected on a sampled frame (0.5 FPS), this service
verifies the detection by analyzing 10 consecutive native video frames (30 FPS).
An activity is only confirmed if 6+ out of 10 native frames show the activity.

This approach provides:
1. Higher accuracy - verifies on dense native frame data (~0.33 seconds)
2. Reduced false positives - single lucky sampled frame won't trigger
3. Clear decision point - exactly 10 frames, exactly 6+ threshold
"""

import cv2
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VerificationStatus(Enum):
    """State machine states for burst verification."""
    IDLE = "idle"           # Waiting for first detection
    VERIFYING = "verifying" # Currently collecting native frames (shouldn't persist between calls)
    CONFIRMED = "confirmed" # Activity verified and active


@dataclass
class VerificationState:
    """State for a single activity verification."""
    status: VerificationStatus = VerificationStatus.IDLE
    confirmation_frame: Optional[int] = None  # Native frame index where confirmed
    last_detection_frame: Optional[int] = None  # Track for continuation


@dataclass
class BurstConfig:
    """Configuration for burst verification per activity type."""
    frames: int = 10        # Number of native frames to verify
    threshold: int = 6      # Minimum detections required (out of frames)


class BurstVerificationService:
    """
    Verifies activity detections using native video frames.

    When activity is first detected on a sampled frame, extracts 10 native frames
    and confirms only if 6+ frames show the activity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the burst verification service.

        Args:
            config: Optional configuration dictionary with:
                - 'verification_frames': Default number of frames (default: 10)
                - 'confirmation_threshold': Default threshold (default: 6)
                - 'activity_overrides': Dict of per-activity settings
        """
        # Default configuration
        self.default_frames = 10
        self.default_threshold = 6

        # Per-activity configuration overrides
        self.activity_config: Dict[str, BurstConfig] = {
            'microsleep': BurstConfig(frames=5, threshold=3),    # Faster for safety
            'sleep': BurstConfig(frames=8, threshold=5),         # Slightly faster for safety
            'cell_phone': BurstConfig(frames=10, threshold=6),   # Standard
            'writing': BurstConfig(frames=10, threshold=6),      # Standard
            'packing_bags': BurstConfig(frames=10, threshold=6), # Standard
            'mind_diversion': BurstConfig(frames=10, threshold=7),  # Stricter
            'group_detected': BurstConfig(frames=10, threshold=7),  # Stricter
            'lp_hand_gesture': BurstConfig(frames=5, threshold=3),  # Fast for gestures
            'alp_hand_gesture': BurstConfig(frames=5, threshold=3), # Fast for gestures
            'no_person_detected': BurstConfig(frames=10, threshold=7),  # Stricter
        }

        # Apply user config overrides
        if config:
            if 'verification_frames' in config:
                self.default_frames = config['verification_frames']
            if 'confirmation_threshold' in config:
                self.default_threshold = config['confirmation_threshold']
            if 'activity_overrides' in config:
                for activity_type, settings in config['activity_overrides'].items():
                    self.activity_config[activity_type] = BurstConfig(
                        frames=settings.get('frames', self.default_frames),
                        threshold=settings.get('threshold', self.default_threshold)
                    )

        # Track verification state per (activity_type, person_idx)
        # Format: {(activity_type, person_idx): VerificationState}
        self.states: Dict[Tuple[str, int], VerificationState] = {}

        # Cache for video capture to avoid repeated opens
        self._video_cache: Dict[str, cv2.VideoCapture] = {}

        logger.info(f"Burst verification service initialized with default {self.default_frames} frames, {self.default_threshold} threshold")
        logger.info(f"Activity configs: {[(k, v.frames, v.threshold) for k, v in self.activity_config.items()]}")

    def get_config(self, activity_type: str) -> BurstConfig:
        """Get configuration for an activity type."""
        return self.activity_config.get(
            activity_type,
            BurstConfig(frames=self.default_frames, threshold=self.default_threshold)
        )

    def get_state(self, activity_type: str, person_idx: int = 0) -> VerificationState:
        """Get or create verification state for an activity/person."""
        key = (activity_type, person_idx)
        if key not in self.states:
            self.states[key] = VerificationState()
        return self.states[key]

    def verify_activity(
        self,
        activity_type: str,
        video_path: str,
        detection_frame_idx: int,
        native_fps: float,
        detector_func: Callable[[Any, str], bool],
        person_idx: int = 0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify detection by checking consecutive native frames.

        This is the main entry point. When an activity is first detected on a
        sampled frame, this method:
        1. Extracts N native frames starting from the detection point
        2. Runs the detector function on each frame
        3. Returns True if threshold detections are met

        Args:
            activity_type: Type of activity ('cell_phone', 'packing_bags', etc.)
            video_path: Path to the video file
            detection_frame_idx: Native frame index where activity was detected
            native_fps: Native video FPS (e.g., 30.0)
            detector_func: Function(frame, activity_type) -> bool
            person_idx: Person index for per-person tracking

        Returns:
            Tuple of (confirmed: bool, details: dict)
            - confirmed: True if activity verified (threshold met)
            - details: Dict with verification details for logging
        """
        state = self.get_state(activity_type, person_idx)
        config = self.get_config(activity_type)

        details = {
            'activity_type': activity_type,
            'person_idx': person_idx,
            'detection_frame': detection_frame_idx,
            'config_frames': config.frames,
            'config_threshold': config.threshold,
        }

        # If already confirmed, extend the activity
        if state.status == VerificationStatus.CONFIRMED:
            state.last_detection_frame = detection_frame_idx
            details['action'] = 'extended'
            details['confirmed'] = True
            return True, details

        # First detection - trigger burst verification
        if state.status == VerificationStatus.IDLE:
            logger.info(f"BURST: Triggering verification for '{activity_type}' person={person_idx} at frame {detection_frame_idx}")

            # Extract native frames
            frames = self.extract_native_frames(
                video_path,
                detection_frame_idx,
                count=config.frames,
                native_fps=native_fps
            )

            if len(frames) < config.frames:
                logger.warning(f"BURST: Only extracted {len(frames)}/{config.frames} frames for '{activity_type}'")

            details['frames_extracted'] = len(frames)

            # Run detection on each frame
            detection_count = 0
            frame_results = []

            for i, frame in enumerate(frames):
                try:
                    detected = detector_func(frame, activity_type)
                    frame_results.append(detected)
                    if detected:
                        detection_count += 1
                except Exception as e:
                    logger.error(f"BURST: Detection error on frame {i}: {e}")
                    frame_results.append(False)

            details['detection_count'] = detection_count
            details['frame_results'] = frame_results
            details['detection_rate'] = f"{detection_count}/{len(frames)}"

            # Decision
            if detection_count >= config.threshold:
                state.status = VerificationStatus.CONFIRMED
                state.confirmation_frame = detection_frame_idx
                state.last_detection_frame = detection_frame_idx
                details['action'] = 'confirmed'
                details['confirmed'] = True
                logger.info(f"BURST: CONFIRMED '{activity_type}' ({detection_count}/{len(frames)} >= {config.threshold})")
                return True, details
            else:
                # Rejected - stay in IDLE for next detection attempt
                state.status = VerificationStatus.IDLE
                details['action'] = 'rejected'
                details['confirmed'] = False
                logger.info(f"BURST: REJECTED '{activity_type}' ({detection_count}/{len(frames)} < {config.threshold})")
                return False, details

        # Shouldn't reach here, but handle gracefully
        details['action'] = 'unknown_state'
        details['confirmed'] = False
        return False, details

    def extract_native_frames(
        self,
        video_path: str,
        start_frame: int,
        count: int,
        native_fps: float
    ) -> List[Any]:
        """
        Extract consecutive native frames from video.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            count: Number of frames to extract
            native_fps: Video FPS (for logging)

        Returns:
            List of frame images (numpy arrays)
        """
        frames = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"BURST: Failed to open video: {video_path}")
                return frames

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read consecutive frames
            for i in range(count):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    logger.warning(f"BURST: Failed to read frame {start_frame + i}")
                    break

            cap.release()

            duration_sec = count / native_fps if native_fps > 0 else 0
            logger.debug(f"BURST: Extracted {len(frames)} frames from {start_frame} (~{duration_sec:.3f}s)")

        except Exception as e:
            logger.error(f"BURST: Frame extraction error: {e}")

        return frames

    def is_activity_confirmed(self, activity_type: str, person_idx: int = 0) -> bool:
        """Check if an activity is currently confirmed."""
        state = self.get_state(activity_type, person_idx)
        return state.status == VerificationStatus.CONFIRMED

    def reset_activity(self, activity_type: str, person_idx: int = 0):
        """
        Reset verification state when activity ends.

        Call this when the activity ends (grace period exceeded) to allow
        new burst verification on next detection.
        """
        key = (activity_type, person_idx)
        if key in self.states:
            old_status = self.states[key].status
            self.states[key] = VerificationState()
            logger.debug(f"BURST: Reset '{activity_type}' person={person_idx} (was {old_status.value})")

    def reset_all_for_person(self, person_idx: int):
        """Reset all activity states for a specific person."""
        keys_to_reset = [k for k in self.states.keys() if k[1] == person_idx]
        for key in keys_to_reset:
            self.states[key] = VerificationState()
        if keys_to_reset:
            logger.debug(f"BURST: Reset all states for person {person_idx}")

    def reset_all(self):
        """Reset all verification states."""
        self.states.clear()
        logger.debug("BURST: Reset all states")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all verification states for debugging."""
        summary = {}
        for (activity_type, person_idx), state in self.states.items():
            key = f"{activity_type}_p{person_idx}"
            summary[key] = {
                'status': state.status.value,
                'confirmation_frame': state.confirmation_frame,
                'last_detection_frame': state.last_detection_frame
            }
        return summary


# Singleton instance
_burst_service: Optional[BurstVerificationService] = None


def get_burst_verification_service(config: Optional[Dict[str, Any]] = None) -> BurstVerificationService:
    """
    Get singleton instance of the burst verification service.

    Args:
        config: Optional configuration to apply on first initialization

    Returns:
        BurstVerificationService: Singleton instance
    """
    global _burst_service
    if _burst_service is None:
        _burst_service = BurstVerificationService(config)
    return _burst_service


def reset_burst_verification_service():
    """Reset the singleton instance (useful for testing)."""
    global _burst_service
    if _burst_service is not None:
        _burst_service.reset_all()
    _burst_service = None
