"""
Train State Detection Service

Detects train stopped/moving state using ROI-based optical flow analysis.

Key Insight: LP/ALP move in cabin regardless of train state.
Solution: Analyze ONLY window/edge regions (scenery), EXCLUDE cabin interior.

Detection Logic:
1. Extract window/edge ROI (left side of frame where window is visible)
2. Calculate optical flow ONLY within ROI
3. If average flow magnitude < threshold = STOPPED
4. Apply temporal filtering (require sustained state)

Dual ROI System (ALP Occlusion Handling):
- PRIMARY ROI: Side window behind ALP (37-52% x, 0-15% y)
- SECONDARY ROI: Left side window (0-12% x, 15-40% y) - below and left of primary
- When ALP stands near primary window and occludes it, falls back to secondary ROI
- Occlusion detected via person bounding box overlap or flow inconsistency

Camera Setup (Indian Locomotive Cab):
- Camera mounted in upper rear-right corner, looking forward-left
- LEFT edge of frame shows cab window with outside scenery
- Center/right shows cabin interior (controls, crew, seats)
"""

import cv2
import numpy as np
import os
import logging
from collections import deque
from typing import Any, Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass

# Import TrainState from activity_models to avoid duplicate enum definition
from app.models.activity_models import TrainStateEnum as TrainState


@dataclass
class StoppedPeriod:
    """Represents a period when the train was stopped"""
    start_time: float
    end_time: Optional[float] = None

    def is_complete(self) -> bool:
        return self.end_time is not None

    def duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class TrainStateDetector:
    """
    Detects train stopped/moving state using ROI-based optical flow.

    Key Insight: LP/ALP move in cabin regardless of train state.
    Solution: Analyze ONLY window/edge regions (scenery), EXCLUDE cabin interior.

    Detection Logic:
    1. Extract window/edge ROI (left edge of frame where window is visible)
    2. Calculate dense optical flow (Farneback) ONLY within ROI
    3. If average flow magnitude < threshold = STOPPED
    4. Apply temporal filtering (require sustained state before change)

    Dual ROI Approach (ALP Occlusion Handling):
    - PRIMARY ROI: Side window (behind ALP standing position)
    - SECONDARY ROI: Front window above control panel (in front of LP)
    - When ALP stands near primary window, their movement can cause false motion detection
    - System detects occlusion via:
      a) Person bounding box overlap with primary ROI
      b) Flow inconsistency (high primary flow + low secondary flow)
    - Falls back to secondary ROI when primary is occluded

    The left portion of frame shows the cab window with outside view.
    When train moves: scenery flows past window (high optical flow in left ROI)
    When train stops: scenery is static (low/zero optical flow in left ROI)
    Crew movements in center/right don't affect detection.
    """

    def __init__(self, config: Dict):
        """
        Initialize train state detector.

        Args:
            config: Configuration dictionary with keys:
                - motion_threshold: Optical flow magnitude threshold (default: 2.0)
                - min_stopped_duration: Seconds before state changes to STOPPED (default: 5.0)
                - roi_x_start/end, roi_y_start/end: Primary side window ROI coordinates
                - secondary_roi_enabled: Enable secondary ROI fallback (default: True)
                - secondary_roi_x_start/end, secondary_roi_y_start/end: Secondary window ROI
                  (above control panel, used when primary is occluded by ALP)
                - occlusion_overlap_threshold: Person overlap % to trigger fallback (default: 0.3)
                - occlusion_flow_diff_threshold: Flow difference to detect occlusion (default: 3.0)
                - adaptive_roi: Use color-based ROI detection (default: False)
                - debug_frames: Enable debug frame saving (default: False)
                - debug_dir: Directory for debug frames (default: 'train_state_debug')
                - debug_interval: Save every Nth frame (default: 1)
                - approaching_stop_flow_threshold: Flow threshold for approaching stop (default: 4.0)
                - approaching_stop_lookback_frames: Frames to analyze for trend (default: 15)
        """
        # Configuration with validation
        self.motion_threshold = config.get('motion_threshold', 2.0)
        self.min_stopped_duration = config.get('min_stopped_duration', 5.0)

        # APPROACHING_STOP detection settings
        self.approaching_stop_flow_threshold = config.get('approaching_stop_flow_threshold', 4.0)
        self.approaching_stop_lookback_frames = config.get('approaching_stop_lookback_frames', 15)

        # PRIMARY ROI: Side window/door - TOP ONLY (above person height to avoid crew movement)
        self.roi_x_start = max(0.0, min(1.0, config.get('roi_x_start', 0.40)))
        self.roi_x_end = max(0.0, min(1.0, config.get('roi_x_end', 0.49)))
        self.roi_y_start = max(0.0, min(1.0, config.get('roi_y_start', 0.0)))
        self.roi_y_end = max(0.0, min(1.0, config.get('roi_y_end', 0.12)))

        # SECONDARY ROI: Left side window (fallback when primary ROI is occluded by ALP)
        # This window is on the far left of the frame, showing outside scenery
        # Used when ALP stands near the primary side window
        self.secondary_roi_enabled = config.get('secondary_roi_enabled', True)
        self.secondary_roi_x_start = max(0.0, min(1.0, config.get('secondary_roi_x_start', 0.0)))
        self.secondary_roi_x_end = max(0.0, min(1.0, config.get('secondary_roi_x_end', 0.12)))
        self.secondary_roi_y_start = max(0.0, min(1.0, config.get('secondary_roi_y_start', 0.30)))
        self.secondary_roi_y_end = max(0.0, min(1.0, config.get('secondary_roi_y_end', 0.55)))

        # Occlusion detection settings
        # If a person bounding box overlaps with primary ROI by this percentage, consider it occluded
        self.occlusion_overlap_threshold = config.get('occlusion_overlap_threshold', 0.3)
        # Flow difference threshold - if primary shows high flow but secondary shows low, primary may be occluded
        self.occlusion_flow_diff_threshold = config.get('occlusion_flow_diff_threshold', 3.0)

        self.adaptive_roi = config.get('adaptive_roi', False)

        # Debug settings
        self.debug_enabled = config.get('debug_frames', False)
        self.debug_dir = config.get('debug_dir', 'train_state_debug')
        self.debug_interval = config.get('debug_interval', 1)
        self._debug_frame_count = 0

        # State tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.current_state = TrainState.UNKNOWN
        self.previous_state = TrainState.UNKNOWN
        self._state_changed = False

        # Temporal filtering
        self.state_start_time: Optional[float] = None
        self.potential_stopped_start: Optional[float] = None
        self.potential_moving_start: Optional[float] = None

        # Flow tracking for smoothing (using deque for O(1) operations)
        self.flow_history_size = 5  # Number of frames to average
        self.flow_history: Deque[float] = deque(maxlen=self.flow_history_size)

        # APPROACHING_STOP: Flow history for trend analysis (detecting deceleration)
        self.approaching_stop_flow_history: Deque[Tuple[float, float]] = deque(
            maxlen=self.approaching_stop_lookback_frames
        )  # Stores (timestamp, flow_magnitude) tuples

        # Stopped periods tracking
        self.stopped_periods: List[StoppedPeriod] = []
        self._current_stopped_period: Optional[StoppedPeriod] = None

        # ROI masks (lazily initialized on first frame)
        self.roi_mask: Optional[np.ndarray] = None
        self.secondary_roi_mask: Optional[np.ndarray] = None
        self.frame_shape: Optional[Tuple[int, int]] = None

        # Occlusion tracking
        self.primary_roi_occluded: bool = False
        self.using_secondary_roi: bool = False

        # ROI quality tracking (overexposure/underexposure detection)
        self.primary_roi_valid: bool = True
        self.secondary_roi_valid: bool = True
        self.primary_roi_quality_reason: str = "valid"
        self.secondary_roi_quality_reason: str = "valid"
        self.roi_quality_unknown: bool = False  # True when both ROIs are invalid

        # Last flow for debug visualization
        self.last_flow: Optional[np.ndarray] = None
        self.last_flow_magnitude: float = 0.0
        self.last_primary_flow: float = 0.0
        self.last_secondary_flow: float = 0.0

        # Logger
        self.logger = logging.getLogger('TrainStateDetector')

        # Periodic logging counter
        self._log_interval = 10  # Log every 10 frames at INFO level
        self._frame_count = 0

        # Log initialization config
        self.logger.info(
            f"TrainStateDetector initialized: "
            f"threshold={self.motion_threshold}, min_stopped={self.min_stopped_duration}s, "
            f"Primary ROI(x={self.roi_x_start:.0%}-{self.roi_x_end:.0%}, y={self.roi_y_start:.0%}-{self.roi_y_end:.0%})"
        )
        if self.secondary_roi_enabled:
            self.logger.info(
                f"Secondary ROI enabled: "
                f"(x={self.secondary_roi_x_start:.0%}-{self.secondary_roi_x_end:.0%}, "
                f"y={self.secondary_roi_y_start:.0%}-{self.secondary_roi_y_end:.0%})"
            )

    def _create_roi_mask(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create mask for primary window ROI where outside scenery is visible.

        The ROI should cover only the window area (not crew members)
        where scenery is visible - motion detected here indicates train moving.

        Args:
            frame_shape: Shape of frame (height, width, channels)

        Returns:
            Boolean mask where True indicates ROI pixels
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        # Primary side window/door ROI (top area above person height)
        x_start = int(width * self.roi_x_start)
        x_end = int(width * self.roi_x_end)
        y_start = int(height * self.roi_y_start)
        y_end = int(height * self.roi_y_end)
        if x_end > x_start and y_end > y_start:
            mask[y_start:y_end, x_start:x_end] = True

        return mask

    def _create_secondary_roi_mask(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create mask for secondary window ROI (front window in front of LP).

        This ROI is used as fallback when the primary ROI is occluded by ALP
        standing near the side window.

        Args:
            frame_shape: Shape of frame (height, width, channels)

        Returns:
            Boolean mask where True indicates secondary ROI pixels
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        # Secondary front window ROI
        x_start = int(width * self.secondary_roi_x_start)
        x_end = int(width * self.secondary_roi_x_end)
        y_start = int(height * self.secondary_roi_y_start)
        y_end = int(height * self.secondary_roi_y_end)
        if x_end > x_start and y_end > y_start:
            mask[y_start:y_end, x_start:x_end] = True

        return mask

    def _get_primary_roi_bounds(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Get pixel bounds for primary ROI (x_start, y_start, x_end, y_end)."""
        height, width = frame_shape[:2]
        return (
            int(width * self.roi_x_start),
            int(height * self.roi_y_start),
            int(width * self.roi_x_end),
            int(height * self.roi_y_end)
        )

    def _check_roi_quality(
        self,
        gray: np.ndarray,
        roi_mask: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if ROI has enough texture for reliable optical flow.

        Detects problematic conditions:
        1. Overexposure (bright sunlight through window - mostly white pixels)
        2. Underexposure (darkness/night - mostly black pixels)
        3. Low texture (uniform area - no features to track)

        Optical flow requires texture/features to track pixel movement.
        When window shows only bright sunlight, there's nothing to track,
        leading to false "stopped" detection even when train is moving.

        Args:
            gray: Grayscale frame
            roi_mask: Boolean mask for the ROI

        Returns:
            Tuple of (is_valid, reason) where:
            - is_valid: True if ROI has enough texture for reliable flow
            - reason: "valid", "overexposed", "underexposed", or "low_texture"
        """
        roi_pixels = gray[roi_mask]

        if roi_pixels.size == 0:
            return False, "empty_roi"

        mean_brightness = float(np.mean(roi_pixels))
        std_brightness = float(np.std(roi_pixels))  # Texture/variance indicator

        # Check overexposure (sunlight) - high brightness + low variance
        # Window showing only bright sunlight will be mostly white (>240) with little variation
        if mean_brightness > 240 and std_brightness < 15:
            return False, "overexposed"

        # Check underexposure (darkness/night)
        # Very dark window with no visible features
        if mean_brightness < 20 and std_brightness < 10:
            return False, "underexposed"

        # Check low texture (uniform area - nothing to track)
        # Even with moderate brightness, if variance is very low, optical flow won't work
        if std_brightness < 8:
            return False, "low_texture"

        return True, "valid"

    def _get_secondary_roi_bounds(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """Get pixel bounds for secondary ROI (x_start, y_start, x_end, y_end)."""
        height, width = frame_shape[:2]
        return (
            int(width * self.secondary_roi_x_start),
            int(height * self.secondary_roi_y_start),
            int(width * self.secondary_roi_x_end),
            int(height * self.secondary_roi_y_end)
        )

    def _check_roi_occlusion(
        self,
        person_boxes: Optional[List[List[float]]],
        frame_shape: Tuple[int, ...]
    ) -> bool:
        """
        Check if primary ROI is occluded by a person (ALP standing near window).

        Args:
            person_boxes: List of person bounding boxes [x1, y1, x2, y2] in pixels
            frame_shape: Shape of frame

        Returns:
            True if primary ROI is likely occluded by a person
        """
        if person_boxes is None or len(person_boxes) == 0:
            return False

        # Get primary ROI bounds in pixels
        roi_x1, roi_y1, roi_x2, roi_y2 = self._get_primary_roi_bounds(frame_shape)
        roi_area = max(1, (roi_x2 - roi_x1) * (roi_y2 - roi_y1))

        for box in person_boxes:
            if len(box) < 4:
                continue

            # Person bounding box
            px1, py1, px2, py2 = box[:4]

            # Calculate intersection
            inter_x1 = max(roi_x1, px1)
            inter_y1 = max(roi_y1, py1)
            inter_x2 = min(roi_x2, px2)
            inter_y2 = min(roi_y2, py2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                overlap_ratio = inter_area / roi_area

                if overlap_ratio >= self.occlusion_overlap_threshold:
                    self.logger.debug(
                        f"Primary ROI occluded: person box overlaps {overlap_ratio:.1%}"
                    )
                    return True

        return False

    def _create_adaptive_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Alternative: Automatically detect window region by analyzing
        brightness/color differences between outside (brighter) and
        interior (darker green walls).

        Args:
            frame: BGR frame

        Returns:
            Boolean mask where True indicates ROI (window) pixels
        """
        # Convert to HSV to detect green interior walls
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Interior walls are green (H: 35-85) - EXCLUDE these areas
        green_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))

        # Non-green areas are likely window/outside - INCLUDE these
        window_mask = cv2.bitwise_not(green_mask)

        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        window_mask = cv2.morphologyEx(window_mask, cv2.MORPH_OPEN, kernel)

        return window_mask.astype(bool)

    def _calculate_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Calculate dense optical flow for the entire frame.

        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame

        Returns:
            Flow array or None on error
        """
        try:
            # Validate frame dimensions match
            if prev_gray.shape != curr_gray.shape:
                self.logger.warning(
                    f"Frame dimension mismatch: prev={prev_gray.shape}, curr={curr_gray.shape}"
                )
                return None

            # Calculate dense optical flow (Farneback method)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5,      # Pyramid scale
                levels=3,           # Pyramid levels
                winsize=15,         # Window size
                iterations=3,       # Iterations per level
                poly_n=5,           # Polynomial expansion neighborhood
                poly_sigma=1.2,     # Gaussian standard deviation
                flags=0
            )

            return flow

        except cv2.error as e:
            self.logger.error(f"OpenCV error in optical flow calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in optical flow calculation: {e}")
            return None

    def _get_roi_flow_magnitude(
        self,
        flow: np.ndarray,
        roi_mask: Optional[np.ndarray]
    ) -> float:
        """
        Calculate average flow magnitude within a specific ROI.

        Args:
            flow: Optical flow array
            roi_mask: Boolean mask for the ROI

        Returns:
            Average flow magnitude in ROI (0.0 if empty)
        """
        # Calculate magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Apply ROI mask
        if roi_mask is not None:
            roi_magnitude = magnitude[roi_mask]
        else:
            roi_magnitude = magnitude.flatten()

        return float(np.mean(roi_magnitude)) if roi_magnitude.size > 0 else 0.0

    def _calculate_roi_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> float:
        """
        Calculate dense optical flow ONLY in window ROI regions.
        Excludes cabin interior to avoid crew movement noise.

        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame

        Returns:
            Average flow magnitude in ROI only (0.0 on error)
        """
        flow = self._calculate_optical_flow(prev_gray, curr_gray)
        if flow is None:
            return 0.0

        # Store for debug visualization
        self.last_flow = flow

        return self._get_roi_flow_magnitude(flow, self.roi_mask)

    def _calculate_dual_roi_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        person_boxes: Optional[List[List[float]]] = None
    ) -> Tuple[float, bool, bool]:
        """
        Calculate optical flow using dual ROI with fallback logic.

        When primary ROI is occluded (ALP standing near window) or has poor quality
        (overexposed due to sunlight), falls back to secondary ROI.

        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame
            person_boxes: Optional list of person bounding boxes

        Returns:
            Tuple of (flow_magnitude, using_secondary_roi, quality_unknown)
            - flow_magnitude: Average optical flow in the active ROI
            - using_secondary_roi: True if secondary ROI is being used
            - quality_unknown: True if both ROIs have poor quality (can't determine motion)
        """
        flow = self._calculate_optical_flow(prev_gray, curr_gray)
        if flow is None:
            return 0.0, False, False

        # Store for debug visualization
        self.last_flow = flow

        # Calculate flow for primary ROI
        primary_flow = self._get_roi_flow_magnitude(flow, self.roi_mask)
        self.last_primary_flow = primary_flow

        # Check primary ROI quality (overexposure/underexposure/low texture)
        self.primary_roi_valid, self.primary_roi_quality_reason = self._check_roi_quality(
            curr_gray, self.roi_mask
        )

        # If secondary ROI is not enabled, just use primary (even if quality is poor)
        if not self.secondary_roi_enabled or self.secondary_roi_mask is None:
            self.roi_quality_unknown = not self.primary_roi_valid
            if not self.primary_roi_valid:
                self.logger.debug(
                    f"Primary ROI quality issue: {self.primary_roi_quality_reason}, "
                    f"no secondary ROI available"
                )
            return primary_flow, False, not self.primary_roi_valid

        # Calculate flow for secondary ROI
        secondary_flow = self._get_roi_flow_magnitude(flow, self.secondary_roi_mask)
        self.last_secondary_flow = secondary_flow

        # Check secondary ROI quality
        self.secondary_roi_valid, self.secondary_roi_quality_reason = self._check_roi_quality(
            curr_gray, self.secondary_roi_mask
        )

        # Check if primary ROI is occluded by a person
        is_occluded_by_person = self._check_roi_occlusion(
            person_boxes, (prev_gray.shape[0], prev_gray.shape[1])
        )

        # Check for flow-based occlusion detection:
        # If primary shows high flow but secondary shows low flow, primary may be
        # detecting person movement rather than scenery
        is_flow_inconsistent = (
            primary_flow > self.motion_threshold and
            secondary_flow <= self.motion_threshold and
            (primary_flow - secondary_flow) > self.occlusion_flow_diff_threshold
        )

        # Determine if we should use secondary ROI
        # Use secondary if: primary is occluded by person, flow inconsistent, OR primary has quality issues
        primary_unusable = is_occluded_by_person or is_flow_inconsistent or not self.primary_roi_valid
        use_secondary = primary_unusable and self.secondary_roi_valid

        self.primary_roi_occluded = is_occluded_by_person or is_flow_inconsistent
        self.using_secondary_roi = use_secondary

        # Check if BOTH ROIs have quality issues (can't reliably determine motion)
        both_rois_invalid = not self.primary_roi_valid and not self.secondary_roi_valid
        self.roi_quality_unknown = both_rois_invalid

        if both_rois_invalid:
            self.logger.warning(
                f"Both ROIs have quality issues - cannot reliably determine motion. "
                f"Primary: {self.primary_roi_quality_reason}, Secondary: {self.secondary_roi_quality_reason}"
            )
            # Return primary flow but mark as unknown quality
            return primary_flow, False, True

        if use_secondary:
            reason_parts = []
            if is_occluded_by_person:
                reason_parts.append("person_occlusion")
            if is_flow_inconsistent:
                reason_parts.append("flow_inconsistent")
            if not self.primary_roi_valid:
                reason_parts.append(f"primary_{self.primary_roi_quality_reason}")

            self.logger.debug(
                f"Using secondary ROI: reasons=[{', '.join(reason_parts)}], "
                f"primary_flow={primary_flow:.2f}, secondary_flow={secondary_flow:.2f}"
            )
            return secondary_flow, True, False
        else:
            return primary_flow, False, False

    def _smooth_flow_value(self, flow_magnitude: float) -> float:
        """
        Smooth flow values using moving average to reduce noise.

        Args:
            flow_magnitude: Current flow magnitude

        Returns:
            Smoothed flow magnitude
        """
        # deque with maxlen automatically discards oldest items (O(1) operation)
        self.flow_history.append(flow_magnitude)

        # Return moving average
        return sum(self.flow_history) / len(self.flow_history)

    def _is_approaching_stop(self, smoothed_flow: float, timestamp: float) -> bool:
        """
        Detect if train is approaching a stop (decelerating).

        Uses linear regression on flow history to detect consistent deceleration.
        Train is "approaching stop" when:
        1. Flow is between motion_threshold and approaching_stop_flow_threshold
        2. Flow shows a consistent downward trend (negative slope)

        Args:
            smoothed_flow: Current smoothed flow magnitude
            timestamp: Current timestamp

        Returns:
            True if train appears to be approaching a stop
        """
        # Store flow in history for trend analysis
        self.approaching_stop_flow_history.append((timestamp, smoothed_flow))

        # Need enough history for trend analysis
        if len(self.approaching_stop_flow_history) < 5:
            return False

        # Check if flow is in the "approaching stop" range
        # (below high threshold but above stopped threshold)
        if smoothed_flow >= self.approaching_stop_flow_threshold:
            return False  # Still moving at normal speed

        if smoothed_flow <= self.motion_threshold:
            return False  # Already stopped

        # Perform linear regression to detect downward trend
        # Extract timestamps and flow values
        times = [t for t, _ in self.approaching_stop_flow_history]
        flows = [f for _, f in self.approaching_stop_flow_history]

        n = len(times)
        if n < 3:
            return False

        # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        sum_t = sum(times)
        sum_f = sum(flows)
        sum_tf = sum(t * f for t, f in zip(times, flows))
        sum_t2 = sum(t * t for t in times)

        denominator = n * sum_t2 - sum_t * sum_t
        if abs(denominator) < 1e-10:
            return False

        slope = (n * sum_tf - sum_t * sum_f) / denominator

        # Negative slope indicates deceleration (flow decreasing over time)
        # Threshold: slope should be significantly negative (e.g., < -0.1 per second)
        is_decelerating = slope < -0.1

        if is_decelerating:
            self.logger.debug(
                f"[{timestamp:.2f}s] Approaching stop detected: "
                f"flow={smoothed_flow:.2f}, slope={slope:.3f}/s"
            )

        return is_decelerating

    def _update_state(self, is_moving: bool, timestamp: float, is_approaching_stop: bool = False) -> None:
        """
        Update train state with temporal filtering.
        Requires sustained detection before changing state.

        Args:
            is_moving: Whether motion is detected in current frame
            timestamp: Current timestamp in seconds
            is_approaching_stop: Whether train is decelerating toward a stop
        """
        self._state_changed = False

        if is_moving:
            # Motion detected at normal speed
            self.potential_stopped_start = None  # Reset stopped timer

            # Check for APPROACHING_STOP transition
            if is_approaching_stop and self.current_state == TrainState.MOVING:
                # Train is decelerating - transition to APPROACHING_STOP
                self._transition_to_state(TrainState.APPROACHING_STOP, timestamp)
                return

            if self.current_state not in (TrainState.MOVING,):
                if self.potential_moving_start is None:
                    self.potential_moving_start = timestamp

                # Check if we've been moving long enough
                # Use shorter threshold for moving (more responsive)
                moving_duration = timestamp - self.potential_moving_start
                if moving_duration >= 1.0:  # 1 second threshold for moving
                    self._transition_to_state(TrainState.MOVING, timestamp)
        else:
            # No motion detected (stopped or nearly stopped)
            self.potential_moving_start = None  # Reset moving timer

            if self.current_state != TrainState.STOPPED:
                if self.potential_stopped_start is None:
                    self.potential_stopped_start = timestamp

                # Check if we've been stopped long enough
                stopped_duration = timestamp - self.potential_stopped_start
                if stopped_duration >= self.min_stopped_duration:
                    self._transition_to_state(TrainState.STOPPED, timestamp)

    def _transition_to_state(self, new_state: TrainState, timestamp: float) -> None:
        """
        Transition to a new state and update tracking.

        Args:
            new_state: The new state to transition to
            timestamp: Current timestamp
        """
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = timestamp
        self._state_changed = True

        # Track stopped periods
        if new_state == TrainState.STOPPED:
            # Start new stopped period
            self._current_stopped_period = StoppedPeriod(start_time=timestamp)
            self.logger.info(f"[{timestamp:.2f}s] Train state changed to STOPPED")
        elif new_state == TrainState.APPROACHING_STOP:
            # Train is decelerating toward a stop
            self.logger.info(f"[{timestamp:.2f}s] Train state changed to APPROACHING_STOP")
        elif new_state == TrainState.MOVING:
            # End current stopped period if active
            if self._current_stopped_period is not None:
                self._current_stopped_period.end_time = timestamp
                self.stopped_periods.append(self._current_stopped_period)
                self.logger.info(
                    f"[{timestamp:.2f}s] Train state changed to MOVING "
                    f"(was stopped for {self._current_stopped_period.duration():.1f}s)"
                )
                self._current_stopped_period = None
            else:
                self.logger.info(f"[{timestamp:.2f}s] Train state changed to MOVING")

    def _save_debug_frame(
        self,
        frame: np.ndarray,
        flow_magnitude: float,
        state: TrainState,
        timestamp: float
    ) -> None:
        """
        Save annotated debug frame for visual verification.

        Args:
            frame: Original BGR frame
            flow_magnitude: Current flow magnitude
            state: Current train state
            timestamp: Current timestamp
        """
        if not self.debug_enabled:
            return

        self._debug_frame_count += 1
        if self._debug_frame_count % self.debug_interval != 0:
            return

        debug_frame = frame.copy()
        h, w = frame.shape[:2]

        # Calculate primary ROI bounding box
        roi_x_start = int(w * self.roi_x_start)
        roi_x_end = int(w * self.roi_x_end)
        roi_y_start = int(h * self.roi_y_start)
        roi_y_end = int(h * self.roi_y_end)

        # 1. Draw primary ROI boundary
        # Color: Green if active and valid, Orange if quality issue, Gray if using secondary
        if not self.primary_roi_valid:
            primary_color = (0, 165, 255)  # Orange - quality issue
            primary_label = f"PRIMARY ({self.primary_roi_quality_reason.upper()})"
        elif self.using_secondary_roi:
            primary_color = (128, 128, 128)  # Gray - not in use
            primary_label = "PRIMARY ROI (OCCLUDED)"
        else:
            primary_color = (0, 255, 0)  # Green - active and valid
            primary_label = "WINDOW ROI"
        primary_thickness = 2 if self.using_secondary_roi else 3
        if roi_x_end > roi_x_start and roi_y_end > roi_y_start:
            cv2.rectangle(debug_frame, (roi_x_start, roi_y_start),
                          (roi_x_end, roi_y_end), primary_color, primary_thickness)
            cv2.putText(debug_frame, primary_label,
                        (roi_x_start + 5, roi_y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, primary_color, 2)

        # 2. Draw secondary ROI boundary if enabled
        if self.secondary_roi_enabled:
            sec_x_start = int(w * self.secondary_roi_x_start)
            sec_x_end = int(w * self.secondary_roi_x_end)
            sec_y_start = int(h * self.secondary_roi_y_start)
            sec_y_end = int(h * self.secondary_roi_y_end)

            # Color: Cyan if active and valid, Orange if quality issue, Gray if not used
            if not self.secondary_roi_valid:
                secondary_color = (0, 165, 255)  # Orange - quality issue
                secondary_label = f"SECONDARY ({self.secondary_roi_quality_reason.upper()})"
            elif self.using_secondary_roi:
                secondary_color = (255, 255, 0)  # Cyan - active
                secondary_label = "SECONDARY ROI (ACTIVE)"
            else:
                secondary_color = (128, 128, 128)  # Gray - not in use
                secondary_label = "SECONDARY ROI"
            secondary_thickness = 3 if self.using_secondary_roi else 2
            if sec_x_end > sec_x_start and sec_y_end > sec_y_start:
                cv2.rectangle(debug_frame, (sec_x_start, sec_y_start),
                              (sec_x_end, sec_y_end), secondary_color, secondary_thickness)
                cv2.putText(debug_frame, secondary_label,
                            (sec_x_start + 5, sec_y_start + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, secondary_color, 2)

        # 2b. Draw warning banner if both ROIs have quality issues
        if self.roi_quality_unknown:
            # Draw red warning banner at top
            cv2.rectangle(debug_frame, (0, 0), (w, 35), (0, 0, 180), -1)
            cv2.putText(
                debug_frame, "WARNING: BOTH ROIs INVALID - CANNOT DETERMINE MOTION",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # 3. Draw optical flow vectors in the active ROI
        if self.last_flow is not None:
            step = 16
            # Draw flow vectors in both ROIs
            active_roi = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
            if self.using_secondary_roi and self.secondary_roi_enabled:
                active_roi = (sec_x_start, sec_y_start, sec_x_end, sec_y_end)

            for y in range(active_roi[1], active_roi[3], step):
                for x in range(active_roi[0], active_roi[2], step):
                    if x < self.last_flow.shape[1] and y < self.last_flow.shape[0]:
                        fx, fy = self.last_flow[y, x]
                        cv2.arrowedLine(debug_frame, (x, y), (int(x + fx * 3), int(y + fy * 3)),
                                        (0, 255, 255), 1, tipLength=0.3)

        # 4. Draw state indicator (top-right)
        state_color = {
            TrainState.MOVING: (0, 0, 255),      # Red
            TrainState.STOPPED: (0, 255, 0),     # Green
            TrainState.UNKNOWN: (128, 128, 128)  # Gray
        }.get(state, (255, 255, 255))

        cv2.putText(
            debug_frame, f"STATE: {state.name}",
            (w - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2
        )
        cv2.putText(
            debug_frame, f"Flow: {flow_magnitude:.2f}",
            (w - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            debug_frame, f"Threshold: {self.motion_threshold:.2f}",
            (w - 280, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
        )

        # 5. Show ROI flow values and quality status
        if self.roi_quality_unknown:
            roi_label = "QUALITY UNKNOWN"
            roi_color = (0, 165, 255)  # Orange
        elif self.using_secondary_roi:
            roi_label = "Using: SECONDARY"
            roi_color = (255, 255, 255)
        else:
            roi_label = "Using: PRIMARY"
            roi_color = (255, 255, 255)
        cv2.putText(
            debug_frame, roi_label,
            (w - 280, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2
        )

        # Primary ROI info with quality
        primary_quality_color = (0, 255, 0) if self.primary_roi_valid else (0, 165, 255)
        primary_info = f"Primary: {self.last_primary_flow:.2f}"
        if not self.primary_roi_valid:
            primary_info += f" [{self.primary_roi_quality_reason}]"
        cv2.putText(
            debug_frame, primary_info,
            (w - 280, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, primary_quality_color, 1
        )

        # Secondary ROI info with quality
        if self.secondary_roi_enabled:
            secondary_quality_color = (0, 255, 0) if self.secondary_roi_valid else (0, 165, 255)
            secondary_info = f"Secondary: {self.last_secondary_flow:.2f}"
            if not self.secondary_roi_valid:
                secondary_info += f" [{self.secondary_roi_quality_reason}]"
            cv2.putText(
                debug_frame, secondary_info,
                (w - 280, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, secondary_quality_color, 1
            )

        # 6. Add timestamp
        cv2.putText(
            debug_frame, f"Time: {timestamp:.2f}s",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # 7. Save frame
        os.makedirs(self.debug_dir, exist_ok=True)
        filename = f"train_state_{timestamp:.2f}s_{state.name}.jpg"
        cv2.imwrite(os.path.join(self.debug_dir, filename), debug_frame)

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        person_boxes: Optional[List[List[float]]] = None
    ) -> TrainState:
        """
        Analyze single frame for motion in window ROI regions.

        Uses dual ROI approach: primary window ROI with fallback to secondary
        window ROI when primary is occluded by ALP standing near the window.

        Args:
            frame: BGR frame from video
            timestamp: Current timestamp in seconds
            person_boxes: Optional list of person bounding boxes for occlusion detection

        Returns:
            Current train state (MOVING, STOPPED, or UNKNOWN)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize ROI masks on first frame
        if self.roi_mask is None or self.frame_shape != frame.shape[:2]:
            self.frame_shape = frame.shape[:2]
            if self.adaptive_roi:
                self.roi_mask = self._create_adaptive_roi_mask(frame)
            else:
                self.roi_mask = self._create_roi_mask(frame.shape)

            # Initialize secondary ROI mask if enabled
            if self.secondary_roi_enabled:
                self.secondary_roi_mask = self._create_secondary_roi_mask(frame.shape)

        # Need previous frame for optical flow
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.current_state

        # Calculate optical flow using dual ROI with fallback logic
        flow_magnitude, using_secondary, quality_unknown = self._calculate_dual_roi_flow(
            self.prev_gray, gray, person_boxes
        )
        self.last_flow_magnitude = flow_magnitude

        # Handle case when both ROIs have quality issues (overexposed/underexposed)
        # In this case, we cannot reliably determine motion, so maintain previous state
        if quality_unknown:
            self._frame_count += 1
            if self._frame_count % self._log_interval == 0:
                self.logger.warning(
                    f"[{timestamp:.2f}s] ROI quality unknown (primary: {self.primary_roi_quality_reason}, "
                    f"secondary: {self.secondary_roi_quality_reason}) - maintaining state: {self.current_state.name}"
                )

            # Save debug frame showing quality issue
            self._save_debug_frame(frame, flow_magnitude, self.current_state, timestamp)

            # Store current frame for next iteration
            self.prev_gray = gray

            # Return current state without updating (can't determine motion reliably)
            return self.current_state

        # Smooth flow values
        smoothed_flow = self._smooth_flow_value(flow_magnitude)

        # Determine if train is moving based on threshold
        is_moving = smoothed_flow > self.motion_threshold

        # Check for approaching stop (decelerating toward station)
        is_approaching_stop = self._is_approaching_stop(smoothed_flow, timestamp)

        # Debug logging - log flow values every frame
        roi_indicator = "SECONDARY" if using_secondary else "PRIMARY"
        self.logger.debug(
            f"[{timestamp:.2f}s] Flow: {flow_magnitude:.2f} ({roi_indicator}) | "
            f"Smoothed: {smoothed_flow:.2f} | Threshold: {self.motion_threshold:.2f} | "
            f"Moving: {is_moving} | Approaching: {is_approaching_stop} | State: {self.current_state.name}"
        )

        # Periodic INFO logging (every N frames) for easier monitoring
        self._frame_count += 1
        if self._frame_count % self._log_interval == 0:
            self.logger.info(
                f"[{timestamp:.2f}s] Train State: {self.current_state.name} | "
                f"Flow: {smoothed_flow:.2f}/{self.motion_threshold:.2f} | ROI: {roi_indicator}"
            )

        # Update state with temporal filtering
        self._update_state(is_moving, timestamp, is_approaching_stop)

        # Save debug frame if enabled
        self._save_debug_frame(frame, smoothed_flow, self.current_state, timestamp)

        # Store current frame for next iteration
        self.prev_gray = gray

        return self.current_state

    def get_state(self) -> TrainState:
        """
        Get current train state.

        Returns:
            Current TrainState (MOVING, STOPPED, or UNKNOWN)
        """
        return self.current_state

    def is_stopped(self) -> bool:
        """
        Quick check if train is currently stopped.

        Returns:
            True if train is in STOPPED state
        """
        return self.current_state == TrainState.STOPPED

    def is_moving(self) -> bool:
        """
        Quick check if train is currently moving.

        Returns:
            True if train is in MOVING state
        """
        return self.current_state == TrainState.MOVING

    def is_approaching_stop(self) -> bool:
        """
        Quick check if train is approaching a stop (decelerating).

        Returns:
            True if train is in APPROACHING_STOP state
        """
        return self.current_state == TrainState.APPROACHING_STOP

    def is_quality_unknown(self) -> bool:
        """
        Check if motion detection quality is unknown due to ROI issues.

        This occurs when both primary and secondary ROIs have quality problems
        (overexposed from sunlight, underexposed, or low texture), making it
        impossible to reliably determine train motion.

        Returns:
            True if both ROIs have quality issues and motion cannot be determined
        """
        return self.roi_quality_unknown

    def get_roi_quality_status(self) -> Dict[str, Any]:
        """
        Get detailed ROI quality status for debugging/monitoring.

        Returns:
            Dictionary with quality information for both ROIs:
            - primary_valid: bool
            - primary_reason: str (valid, overexposed, underexposed, low_texture)
            - secondary_valid: bool
            - secondary_reason: str
            - quality_unknown: bool (True if both invalid)
        """
        return {
            'primary_valid': self.primary_roi_valid,
            'primary_reason': self.primary_roi_quality_reason,
            'secondary_valid': self.secondary_roi_valid,
            'secondary_reason': self.secondary_roi_quality_reason,
            'quality_unknown': self.roi_quality_unknown,
        }

    def state_changed(self) -> bool:
        """
        Check if state changed in the last analyze_frame call.

        Returns:
            True if state changed in the last frame analysis
        """
        return self._state_changed

    def get_stopped_periods(self) -> List[Tuple[float, float]]:
        """
        Get list of all completed stopped periods.

        Returns:
            List of (start_time, end_time) tuples for all stopped periods
        """
        periods = []
        for period in self.stopped_periods:
            if period.is_complete():
                periods.append((period.start_time, period.end_time))

        # Include current stopped period if active
        if self._current_stopped_period is not None and self.state_start_time is not None:
            periods.append((self._current_stopped_period.start_time, None))

        return periods

    def get_state_duration(self) -> float:
        """
        Get how long the train has been in the current state.

        Returns:
            Duration in seconds (0 if state_start_time not set)
        """
        if self.state_start_time is None:
            return 0.0
        # Note: This requires knowing current timestamp
        # For now, return 0 - caller should track duration themselves
        return 0.0

    def finalize(self, final_timestamp: float) -> None:
        """
        Finalize detection at end of video.
        Closes any open stopped period.

        Args:
            final_timestamp: Final timestamp of video
        """
        if self._current_stopped_period is not None:
            self._current_stopped_period.end_time = final_timestamp
            self.stopped_periods.append(self._current_stopped_period)
            self.logger.info(
                f"[{final_timestamp:.2f}s] Finalized stopped period "
                f"(duration: {self._current_stopped_period.duration():.1f}s)"
            )
            self._current_stopped_period = None

    def reset(self) -> None:
        """
        Reset detector state for processing a new video.
        """
        self.prev_gray = None
        self.current_state = TrainState.UNKNOWN
        self.previous_state = TrainState.UNKNOWN
        self._state_changed = False
        self.state_start_time = None
        self.potential_stopped_start = None
        self.potential_moving_start = None
        self.flow_history.clear()
        self.approaching_stop_flow_history.clear()
        self.stopped_periods.clear()
        self._current_stopped_period = None
        self.roi_mask = None
        self.secondary_roi_mask = None
        self.frame_shape = None
        self.last_flow = None
        self.last_flow_magnitude = 0.0
        self.last_primary_flow = 0.0
        self.last_secondary_flow = 0.0
        self.primary_roi_occluded = False
        self.using_secondary_roi = False
        # Reset ROI quality tracking
        self.primary_roi_valid = True
        self.secondary_roi_valid = True
        self.primary_roi_quality_reason = "valid"
        self.secondary_roi_quality_reason = "valid"
        self.roi_quality_unknown = False
        self._debug_frame_count = 0
        self._frame_count = 0
