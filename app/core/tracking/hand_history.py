"""Hand history tracking: smoothing buffers, position history, velocity/trajectory analysis,
and the packing wrist-motion gate.

Extracted from locopilot_monitor.py (T3 in the refactor plan):
- _get_smoothed_hand_position (lines 1592-1627)
- analyze_hand_velocity_and_trajectory (lines 2406-2501)
- _check_wrist_motion_for_packing (lines 2631-2670)

State contract: ``smoothing_buffers`` and ``position_history`` are regular
``dict`` objects (NOT ``defaultdict``). The monitor aliases them by reference
(``self.hand_smoothing_buffers = self._hand_history.smoothing_buffers``) so
the per-person cleanup code continues to see the same object.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np


class HandHistoryTracker:
    """Per-person hand-position state plus the helpers that read it.

    Attributes:
        smoothing_buffers: dict keyed by ``(person_idx, hand_side)`` ->
            ``{'positions': deque(maxlen=smoothing_window),
              'timestamps': deque(maxlen=smoothing_window)}``.
            This is the same object the monitor exposes as
            ``self.hand_smoothing_buffers``.
        position_history: dict keyed by ``person_idx`` ->
            ``{'right_wrist': deque(maxlen=history_max_length),
              'left_wrist': deque(maxlen=history_max_length),
              'timestamps': deque(maxlen=history_max_length)}``.
            This is the same object the monitor exposes as
            ``self.hand_position_history``.
    """

    def __init__(
        self,
        *,
        history_max_length: int = 10,
        smoothing_window: int = 3,
        packing_wrist_motion_min_velocity: float = 0.008,
        packing_wrist_motion_gate_enabled: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.history_max_length = history_max_length
        self.smoothing_window = smoothing_window
        self.packing_wrist_motion_min_velocity = packing_wrist_motion_min_velocity
        self.packing_wrist_motion_gate_enabled = packing_wrist_motion_gate_enabled
        self.logger = logger or logging.getLogger(__name__)

        # Plain dicts (NOT defaultdict) so the monitor can alias by reference
        # and so cleanup code can call .pop()/del without surprising the
        # collection's default-factory behavior.
        self.smoothing_buffers: dict = dict()
        self.position_history: dict = dict()

    # ------------------------------------------------------------------
    # Smoothed hand position (was _get_smoothed_hand_position)
    # ------------------------------------------------------------------
    def get_smoothed_hand_position(
        self,
        person_idx: int,
        hand_side: str,
        landmark: Any,
        w: int,
        h: int,
        timestamp_sec: float,
    ) -> Tuple[int, int]:
        """Get temporally smoothed hand position to reduce pose estimation noise.

        Uses simple average over last 3 positions (6 seconds @ 0.5fps).

        Args:
            person_idx: Person identifier
            hand_side: 'right' or 'left'
            landmark: Pose landmark with .x, .y attributes (normalised coords)
            w, h: Frame dimensions in pixels
            timestamp_sec: Current timestamp

        Returns:
            tuple: (smoothed_x, smoothed_y) pixel coordinates
        """
        key = (person_idx, hand_side)
        if key not in self.smoothing_buffers:
            self.smoothing_buffers[key] = {
                'positions': deque(maxlen=self.smoothing_window),
                'timestamps': deque(maxlen=self.smoothing_window),
            }

        buffer = self.smoothing_buffers[key]

        current_x = int(landmark.x * w)
        current_y = int(landmark.y * h)
        buffer['positions'].append((current_x, current_y))
        buffer['timestamps'].append(timestamp_sec)

        if len(buffer['positions']) > 0:
            avg_x = sum(pos[0] for pos in buffer['positions']) / len(buffer['positions'])
            avg_y = sum(pos[1] for pos in buffer['positions']) / len(buffer['positions'])
            return (int(avg_x), int(avg_y))

        return (current_x, current_y)

    # ------------------------------------------------------------------
    # Velocity + trajectory (was analyze_hand_velocity_and_trajectory)
    # ------------------------------------------------------------------
    def analyze_velocity_and_trajectory(
        self,
        person_idx: int,
        landmarks: Any,
        frame_shape: Tuple[int, ...],
        timestamp_sec: float,
        *,
        get_keypoint,
    ) -> Dict[str, Any]:
        """
        Analyze hand velocity and trajectory patterns to enhance gesture detection.

        Detects rapid hand raises (signaling) vs static positions (control operations).

        Args:
            person_idx: Person index
            landmarks: MediaPipe pose landmarks
            frame_shape: Frame dimensions (h, w, c)
            timestamp_sec: Current timestamp
            get_keypoint: Callable that returns a keypoint by name (was
                ``self.get_keypoint`` on the monitor).

        Returns:
            dict: Velocity/trajectory analysis results
        """
        h, w = frame_shape[:2]

        # Initialize history for this person
        if person_idx not in self.position_history:
            self.position_history[person_idx] = {
                'right_wrist': deque(maxlen=self.history_max_length),
                'left_wrist': deque(maxlen=self.history_max_length),
                'timestamps': deque(maxlen=self.history_max_length),
            }

        history = self.position_history[person_idx]

        # Get current wrist positions
        right_wrist = get_keypoint(landmarks, 'right_wrist')
        left_wrist = get_keypoint(landmarks, 'left_wrist')

        right_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

        # Append current positions
        history['right_wrist'].append(right_coords)
        history['left_wrist'].append(left_coords)
        history['timestamps'].append(timestamp_sec)

        # Need at least 3 positions to analyze velocity
        if len(history['timestamps']) < 3:
            return {
                'right_velocity': 0.0,
                'left_velocity': 0.0,
                'right_trajectory': 'unknown',
                'left_trajectory': 'unknown',
                'rapid_raise_detected': False,
                'analysis_quality': 'insufficient_data',
            }

        # Calculate velocities (pixels per second)
        def calculate_velocity(position_history, timestamps):
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

        # Detect rapid hand raise: velocity > 150 px/s AND upward trajectory
        rapid_raise = (
            (right_vel > 150 and right_traj == 'upward') or
            (left_vel > 150 and left_traj == 'upward')
        )

        return {
            'right_velocity': right_vel,
            'left_velocity': left_vel,
            'right_trajectory': right_traj,
            'left_trajectory': left_traj,
            'rapid_raise_detected': rapid_raise,
            'analysis_quality': 'good' if len(history['timestamps']) >= 5 else 'limited',
        }

    # ------------------------------------------------------------------
    # Packing wrist-motion gate (was _check_wrist_motion_for_packing)
    # ------------------------------------------------------------------
    def check_wrist_motion_for_packing(self, person_idx: int, timestamp_sec: float) -> bool:
        """Check if wrists show sufficient motion to indicate actual packing activity.

        Returns True if wrist motion is above the threshold, False if hands are stationary.
        """
        if not self.packing_wrist_motion_gate_enabled:
            return True  # Gate disabled, always pass

        min_velocity = self.packing_wrist_motion_min_velocity

        history = self.position_history.get(person_idx)
        if not history or len(history.get('timestamps', [])) < 2:
            return True  # Not enough data, allow detection

        timestamps = list(history['timestamps'])
        right_positions = list(history.get('right_wrist', []))
        left_positions = list(history.get('left_wrist', []))

        # Calculate recent wrist velocities (last 3 frames)
        max_velocity = 0.0
        n = min(3, len(timestamps) - 1)
        for i in range(-n, 0):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue

            for positions in [right_positions, left_positions]:
                if len(positions) >= abs(i - 1):
                    try:
                        p1 = positions[i - 1]
                        p2 = positions[i]
                        if p1 is not None and p2 is not None:
                            disp = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                            # Normalize by frame diagonal (~1280 for 720p)
                            vel = disp / (1280.0 * dt)
                            max_velocity = max(max_velocity, vel)
                    except (IndexError, TypeError):
                        continue

        return max_velocity >= min_velocity
