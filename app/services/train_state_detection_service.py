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
from typing import Dict, List, Tuple, Optional, Deque
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
                - roi1_x_start/end, roi1_y_start/end: Front window ROI coordinates
                - roi2_x_start/end, roi2_y_start/end: Side window ROI coordinates
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

        # ROI: Side window/door - TOP ONLY (above person height to avoid crew movement)
        self.roi_x_start = max(0.0, min(1.0, config.get('roi_x_start', 0.37)))
        self.roi_x_end = max(0.0, min(1.0, config.get('roi_x_end', 0.52)))
        self.roi_y_start = max(0.0, min(1.0, config.get('roi_y_start', 0.0)))
        self.roi_y_end = max(0.0, min(1.0, config.get('roi_y_end', 0.15)))

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

        # ROI mask (lazily initialized on first frame)
        self.roi_mask: Optional[np.ndarray] = None
        self.frame_shape: Optional[Tuple[int, int]] = None

        # Last flow for debug visualization
        self.last_flow: Optional[np.ndarray] = None
        self.last_flow_magnitude: float = 0.0

        # Logger
        self.logger = logging.getLogger('TrainStateDetector')

        # Periodic logging counter
        self._log_interval = 10  # Log every 10 frames at INFO level
        self._frame_count = 0

        # Log initialization config
        self.logger.info(
            f"TrainStateDetector initialized: "
            f"threshold={self.motion_threshold}, min_stopped={self.min_stopped_duration}s, "
            f"ROI(x={self.roi_x_start:.0%}-{self.roi_x_end:.0%}, y={self.roi_y_start:.0%}-{self.roi_y_end:.0%})"
        )

    def _create_roi_mask(self, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Create mask for window ROI where outside scenery is visible.

        The ROI should cover only the window area (not crew members)
        where scenery is visible - motion detected here indicates train moving.

        Args:
            frame_shape: Shape of frame (height, width, channels)

        Returns:
            Boolean mask where True indicates ROI pixels
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        # Side window/door ROI (top area above person height)
        x_start = int(width * self.roi_x_start)
        x_end = int(width * self.roi_x_end)
        y_start = int(height * self.roi_y_start)
        y_end = int(height * self.roi_y_end)
        if x_end > x_start and y_end > y_start:
            mask[y_start:y_end, x_start:x_end] = True

        return mask

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

    def _calculate_roi_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        Calculate dense optical flow ONLY in window ROI regions.
        Excludes cabin interior to avoid crew movement noise.

        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame

        Returns:
            Average flow magnitude in ROI only (0.0 on error)
        """
        try:
            # Validate frame dimensions match
            if prev_gray.shape != curr_gray.shape:
                self.logger.warning(
                    f"Frame dimension mismatch: prev={prev_gray.shape}, curr={curr_gray.shape}"
                )
                return 0.0

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

            # Store for debug visualization
            self.last_flow = flow

            # Calculate magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # CRITICAL: Apply ROI mask to exclude cabin interior
            # Only analyze edge regions (windows) where scenery is visible
            if self.roi_mask is not None:
                roi_magnitude = magnitude[self.roi_mask]
            else:
                # Fallback to full frame if mask not ready
                roi_magnitude = magnitude.flatten()

            # Return average magnitude in ROI only
            return float(np.mean(roi_magnitude)) if roi_magnitude.size > 0 else 0.0

        except cv2.error as e:
            self.logger.error(f"OpenCV error in optical flow calculation: {e}")
            return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error in optical flow calculation: {e}")
            return 0.0

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

        # Calculate ROI bounding box
        x_start = int(w * self.roi_x_start)
        x_end = int(w * self.roi_x_end)
        y_start = int(h * self.roi_y_start)
        y_end = int(h * self.roi_y_end)

        # 1. Draw ROI boundary (green rectangle) - Side Window
        if x_end > x_start and y_end > y_start:
            cv2.rectangle(debug_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
            cv2.putText(debug_frame, "WINDOW ROI", (x_start + 5, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2. Draw optical flow vectors (if available) within ROI
        if self.last_flow is not None:
            step = 16
            for y in range(y_start, y_end, step):
                for x in range(x_start, x_end, step):
                    if x < self.last_flow.shape[1] and y < self.last_flow.shape[0]:
                        fx, fy = self.last_flow[y, x]
                        cv2.arrowedLine(debug_frame, (x, y), (int(x + fx * 3), int(y + fy * 3)),
                                        (0, 255, 255), 1, tipLength=0.3)

        # 3. Draw state indicator (top-right)
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

        # 4. Add timestamp
        cv2.putText(
            debug_frame, f"Time: {timestamp:.2f}s",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # 5. Save frame
        os.makedirs(self.debug_dir, exist_ok=True)
        filename = f"train_state_{timestamp:.2f}s_{state.name}.jpg"
        cv2.imwrite(os.path.join(self.debug_dir, filename), debug_frame)

    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> TrainState:
        """
        Analyze single frame for motion in window ROI regions.

        Args:
            frame: BGR frame from video
            timestamp: Current timestamp in seconds

        Returns:
            Current train state (MOVING, STOPPED, or UNKNOWN)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize ROI mask on first frame
        if self.roi_mask is None or self.frame_shape != frame.shape[:2]:
            self.frame_shape = frame.shape[:2]
            if self.adaptive_roi:
                self.roi_mask = self._create_adaptive_roi_mask(frame)
            else:
                self.roi_mask = self._create_roi_mask(frame.shape)

        # Need previous frame for optical flow
        if self.prev_gray is None:
            self.prev_gray = gray
            return self.current_state

        # Calculate optical flow in ROI
        flow_magnitude = self._calculate_roi_optical_flow(self.prev_gray, gray)
        self.last_flow_magnitude = flow_magnitude

        # Smooth flow values
        smoothed_flow = self._smooth_flow_value(flow_magnitude)

        # Determine if train is moving based on threshold
        is_moving = smoothed_flow > self.motion_threshold

        # Check for approaching stop (decelerating toward station)
        is_approaching_stop = self._is_approaching_stop(smoothed_flow, timestamp)

        # Debug logging - log flow values every frame
        self.logger.debug(
            f"[{timestamp:.2f}s] Flow: {flow_magnitude:.2f} | "
            f"Smoothed: {smoothed_flow:.2f} | Threshold: {self.motion_threshold:.2f} | "
            f"Moving: {is_moving} | Approaching: {is_approaching_stop} | State: {self.current_state.name}"
        )

        # Periodic INFO logging (every N frames) for easier monitoring
        self._frame_count += 1
        if self._frame_count % self._log_interval == 0:
            self.logger.info(
                f"[{timestamp:.2f}s] Train State: {self.current_state.name} | "
                f"Flow: {smoothed_flow:.2f}/{self.motion_threshold:.2f}"
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
        self.frame_shape = None
        self.last_flow = None
        self.last_flow_magnitude = 0.0
        self._debug_frame_count = 0
        self._frame_count = 0
