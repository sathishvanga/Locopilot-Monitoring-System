"""
Side Window Motion Service - Optical flow motion detection for train motion state

This service uses OpenCV Farneback optical flow to detect actual physical motion
by analyzing the oval side windows on the left side of the locomotive cabin.
This catches unscheduled stops (signals, crossings) that schedule data cannot detect.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

from ..utils.config import get_settings

logger = logging.getLogger(__name__)


class OpticalFlowMotionState(str, Enum):
    """Motion state as determined by optical flow analysis."""
    STOPPED = "stopped"
    RUNNING = "running"
    UNCERTAIN = "uncertain"


@dataclass
class MotionDetectionResult:
    """Result of motion detection analysis."""
    motion_state: OpticalFlowMotionState
    magnitude: float
    confidence: float
    frame_count: int = 1
    roi_used: Tuple[float, float, float, float] = (0.0, 0.0, 0.18, 0.40)
    debug_info: Optional[Dict] = None

    def is_stopped(self) -> bool:
        """Check if motion state indicates train is stopped."""
        return self.motion_state == OpticalFlowMotionState.STOPPED

    def is_running(self) -> bool:
        """Check if motion state indicates train is running."""
        return self.motion_state == OpticalFlowMotionState.RUNNING

    def __repr__(self):
        return (
            f"MotionDetectionResult(state={self.motion_state.value}, "
            f"magnitude={self.magnitude:.2f}, confidence={self.confidence:.2f})"
        )


class SideWindowMotionService:
    """
    Service for optical flow motion detection on side window region.

    Uses OpenCV Farneback dense optical flow to analyze motion in the
    locomotive cabin's side windows to determine if train is moving.

    Features:
    - Configurable ROI for side window region
    - Farneback dense optical flow analysis
    - Batch processing with voting for multi-frame analysis
    - Configurable thresholds for stopped/running classification

    ROI Configuration (Side Window 2 - Default):
        Left 18% of frame width, 30-70% of frame height
        x=0.0, width=0.18, y=0.30, height=0.40
    """

    def __init__(self):
        """Initialize the side window motion service."""
        self.settings = get_settings()

        # Check if optical flow is enabled
        self.enabled = getattr(self.settings, 'optical_flow_enabled', True)

        # ROI configuration (ratios of frame dimensions)
        self.roi_x_ratio = getattr(self.settings, 'motion_roi_x_ratio', 0.0)
        self.roi_width_ratio = getattr(self.settings, 'motion_roi_width_ratio', 0.18)
        self.roi_y_ratio = getattr(self.settings, 'motion_roi_y_ratio', 0.30)
        self.roi_height_ratio = getattr(self.settings, 'motion_roi_height_ratio', 0.40)

        # Thresholds for motion classification
        self.stopped_threshold = getattr(self.settings, 'motion_stopped_threshold', 2.0)
        self.running_threshold = getattr(self.settings, 'motion_running_threshold', 5.0)
        self.confidence_threshold = getattr(self.settings, 'motion_confidence_threshold', 0.7)

        # Farneback optical flow parameters
        self.flow_params = {
            'pyr_scale': 0.5,      # Image scale for pyramid
            'levels': 3,           # Number of pyramid levels
            'winsize': 15,         # Averaging window size
            'iterations': 3,       # Iterations at each level
            'poly_n': 5,           # Polynomial expansion neighborhood
            'poly_sigma': 1.2,     # Standard deviation for Gaussian
            'flags': 0             # Additional flags
        }

        # Store previous frame for motion detection
        self._prev_gray_frame: Optional[np.ndarray] = None
        self._prev_roi: Optional[np.ndarray] = None

        logger.info(
            f"SideWindowMotionService initialized - "
            f"enabled: {self.enabled}, "
            f"ROI: x={self.roi_x_ratio}, w={self.roi_width_ratio}, "
            f"y={self.roi_y_ratio}, h={self.roi_height_ratio}, "
            f"thresholds: stopped<{self.stopped_threshold}, running>{self.running_threshold}"
        )

    def detect_motion(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> MotionDetectionResult:
        """
        Detect motion in a single frame pair using optical flow.

        Args:
            frame: Current BGR frame
            prev_frame: Previous BGR frame (optional, uses internal state if None)

        Returns:
            MotionDetectionResult with motion state and magnitude
        """
        if not self.enabled:
            return MotionDetectionResult(
                motion_state=OpticalFlowMotionState.UNCERTAIN,
                magnitude=0.0,
                confidence=0.0,
                debug_info={'reason': 'service_disabled'}
            )

        try:
            # Extract ROI from frame
            roi = self._extract_roi(frame)
            if roi is None:
                return MotionDetectionResult(
                    motion_state=OpticalFlowMotionState.UNCERTAIN,
                    magnitude=0.0,
                    confidence=0.0,
                    debug_info={'reason': 'roi_extraction_failed'}
                )

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Get previous frame ROI
            if prev_frame is not None:
                prev_roi = self._extract_roi(prev_frame)
                if prev_roi is not None:
                    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
                else:
                    prev_gray = self._prev_roi
            else:
                prev_gray = self._prev_roi

            # Store current for next iteration
            self._prev_roi = gray.copy()

            # Need previous frame for optical flow
            if prev_gray is None:
                return MotionDetectionResult(
                    motion_state=OpticalFlowMotionState.UNCERTAIN,
                    magnitude=0.0,
                    confidence=0.0,
                    debug_info={'reason': 'no_previous_frame'}
                )

            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                self.flow_params['pyr_scale'],
                self.flow_params['levels'],
                self.flow_params['winsize'],
                self.flow_params['iterations'],
                self.flow_params['poly_n'],
                self.flow_params['poly_sigma'],
                self.flow_params['flags']
            )

            # Calculate magnitude and angle of flow vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Calculate mean magnitude (excluding outliers)
            # Use percentile to avoid noise spikes
            mean_magnitude = float(np.percentile(magnitude, 90))

            # Classify motion state
            motion_state, confidence = self._classify_motion(mean_magnitude)

            logger.debug(
                f"[OPTICAL-FLOW] Magnitude={mean_magnitude:.2f}, "
                f"State={motion_state.value}, Confidence={confidence:.2f}"
            )

            return MotionDetectionResult(
                motion_state=motion_state,
                magnitude=mean_magnitude,
                confidence=confidence,
                roi_used=(self.roi_x_ratio, self.roi_y_ratio,
                         self.roi_width_ratio, self.roi_height_ratio),
                debug_info={
                    'flow_shape': flow.shape,
                    'percentile_90': mean_magnitude,
                    'mean': float(np.mean(magnitude)),
                    'max': float(np.max(magnitude))
                }
            )

        except Exception as e:
            logger.warning(f"[OPTICAL-FLOW] Error in detect_motion: {e}")
            return MotionDetectionResult(
                motion_state=OpticalFlowMotionState.UNCERTAIN,
                magnitude=0.0,
                confidence=0.0,
                debug_info={'error': str(e)}
            )

    def detect_motion_batch(
        self,
        frames: List[np.ndarray]
    ) -> MotionDetectionResult:
        """
        Detect motion using multiple consecutive frames with voting.

        Analyzes optical flow between consecutive frame pairs and uses
        voting to determine final motion state. More robust than single
        frame analysis.

        Args:
            frames: List of consecutive BGR frames (at least 2 required)

        Returns:
            MotionDetectionResult with aggregated motion state
        """
        if not self.enabled:
            return MotionDetectionResult(
                motion_state=OpticalFlowMotionState.UNCERTAIN,
                magnitude=0.0,
                confidence=0.0,
                frame_count=len(frames),
                debug_info={'reason': 'service_disabled'}
            )

        if len(frames) < 2:
            return MotionDetectionResult(
                motion_state=OpticalFlowMotionState.UNCERTAIN,
                magnitude=0.0,
                confidence=0.0,
                frame_count=len(frames),
                debug_info={'reason': 'insufficient_frames'}
            )

        logger.info(
            f"[OPTICAL-FLOW] Analyzing {len(frames)} frames for motion detection"
        )

        # Analyze each consecutive frame pair
        magnitudes = []
        states = []

        for i in range(1, len(frames)):
            result = self.detect_motion(frames[i], frames[i - 1])
            magnitudes.append(result.magnitude)
            states.append(result.motion_state)

        # Calculate aggregate statistics
        mean_magnitude = np.mean(magnitudes) if magnitudes else 0.0
        median_magnitude = np.median(magnitudes) if magnitudes else 0.0

        # Vote on motion state
        stopped_votes = sum(1 for s in states if s == OpticalFlowMotionState.STOPPED)
        running_votes = sum(1 for s in states if s == OpticalFlowMotionState.RUNNING)
        uncertain_votes = sum(1 for s in states if s == OpticalFlowMotionState.UNCERTAIN)
        total_votes = len(states)

        # Determine final state based on voting
        if total_votes == 0:
            final_state = OpticalFlowMotionState.UNCERTAIN
            confidence = 0.0
        elif stopped_votes > total_votes * self.confidence_threshold:
            final_state = OpticalFlowMotionState.STOPPED
            confidence = stopped_votes / total_votes
        elif running_votes > total_votes * self.confidence_threshold:
            final_state = OpticalFlowMotionState.RUNNING
            confidence = running_votes / total_votes
        else:
            # No clear majority - use magnitude as tiebreaker
            final_state, confidence = self._classify_motion(median_magnitude)

        logger.info(
            f"[OPTICAL-FLOW] Batch result: {final_state.value} "
            f"(stopped={stopped_votes}, running={running_votes}, "
            f"uncertain={uncertain_votes}, magnitude={mean_magnitude:.2f})"
        )

        return MotionDetectionResult(
            motion_state=final_state,
            magnitude=float(mean_magnitude),
            confidence=confidence,
            frame_count=len(frames),
            roi_used=(self.roi_x_ratio, self.roi_y_ratio,
                     self.roi_width_ratio, self.roi_height_ratio),
            debug_info={
                'magnitudes': magnitudes,
                'states': [s.value for s in states],
                'stopped_votes': stopped_votes,
                'running_votes': running_votes,
                'uncertain_votes': uncertain_votes,
                'median_magnitude': float(median_magnitude)
            }
        )

    def _extract_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the side window ROI from frame.

        Args:
            frame: Full BGR frame

        Returns:
            ROI region as numpy array or None if extraction fails
        """
        try:
            h, w = frame.shape[:2]

            # Calculate pixel coordinates from ratios
            x1 = int(w * self.roi_x_ratio)
            x2 = int(w * (self.roi_x_ratio + self.roi_width_ratio))
            y1 = int(h * self.roi_y_ratio)
            y2 = int(h * (self.roi_y_ratio + self.roi_height_ratio))

            # Validate coordinates
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))

            roi = frame[y1:y2, x1:x2]

            # Ensure ROI is large enough for optical flow
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                logger.warning(
                    f"[OPTICAL-FLOW] ROI too small: {roi.shape}"
                )
                return None

            return roi

        except Exception as e:
            logger.warning(f"[OPTICAL-FLOW] ROI extraction error: {e}")
            return None

    def _classify_motion(
        self,
        magnitude: float
    ) -> Tuple[OpticalFlowMotionState, float]:
        """
        Classify motion state based on optical flow magnitude.

        Args:
            magnitude: Mean optical flow magnitude

        Returns:
            Tuple of (motion_state, confidence)
        """
        if magnitude < self.stopped_threshold:
            # Low motion - train is stopped
            # Higher confidence when magnitude is closer to 0
            confidence = 1.0 - (magnitude / self.stopped_threshold)
            return OpticalFlowMotionState.STOPPED, confidence

        elif magnitude > self.running_threshold:
            # High motion - train is running
            # Cap confidence at 1.0 for very high magnitudes
            confidence = min(1.0, magnitude / (self.running_threshold * 2))
            return OpticalFlowMotionState.RUNNING, confidence

        else:
            # Uncertain range between thresholds
            # Calculate position in the uncertain zone
            range_width = self.running_threshold - self.stopped_threshold
            position = (magnitude - self.stopped_threshold) / range_width

            # Assign state based on which threshold it's closer to
            if position < 0.5:
                state = OpticalFlowMotionState.STOPPED
                confidence = 0.5 - position
            else:
                state = OpticalFlowMotionState.RUNNING
                confidence = position - 0.5

            return state, confidence

    def reset(self):
        """Reset internal state (call when switching videos)."""
        self._prev_gray_frame = None
        self._prev_roi = None
        logger.debug("[OPTICAL-FLOW] Internal state reset")

    def get_roi_coordinates(
        self,
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Get ROI pixel coordinates for a given frame size.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Tuple of (x1, y1, x2, y2) pixel coordinates
        """
        x1 = int(frame_width * self.roi_x_ratio)
        x2 = int(frame_width * (self.roi_x_ratio + self.roi_width_ratio))
        y1 = int(frame_height * self.roi_y_ratio)
        y2 = int(frame_height * (self.roi_y_ratio + self.roi_height_ratio))
        return (x1, y1, x2, y2)


# Global service instance
_motion_service: Optional[SideWindowMotionService] = None


def get_side_window_motion_service() -> SideWindowMotionService:
    """
    Get the global side window motion service instance.

    Returns:
        SideWindowMotionService instance
    """
    global _motion_service
    if _motion_service is None:
        _motion_service = SideWindowMotionService()
    return _motion_service
