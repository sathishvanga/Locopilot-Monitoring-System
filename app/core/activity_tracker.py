"""Activity state management and tracking.

This module provides activity tracking functionality for managing detection states,
grace periods, consecutive detections, and activity lifecycle (start/end).

Extracted from locopilot_monitor.py for modularity and reusability.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

from app.utils.logger import get_logger


@dataclass
class ActivityConfig:
    """Configuration for an activity type.

    Defines thresholds and parameters for activity detection and tracking.

    Attributes:
        min_duration: Minimum duration in seconds for an activity to be recorded.
        required_consecutive: Number of consecutive frames required to start an activity.
        margin: Optional pixel margin for proximity-based detection.
        grace_frames: Number of frames to allow gaps in detection before ending activity.
        region_margin: Optional margin for region-based detection (e.g., packing bags).
        wrist_inside_margin: Optional margin for wrist proximity detection.
        sustained_proximity_seconds: Optional duration for sustained proximity detection.
    """

    min_duration: float = 0.0
    required_consecutive: int = 1
    margin: Optional[int] = None
    grace_frames: int = 5
    region_margin: Optional[int] = None
    wrist_inside_margin: Optional[int] = None
    sustained_proximity_seconds: Optional[float] = None


@dataclass
class ActivityState:
    """Represents the current state of an activity being tracked.

    Attributes:
        active: Whether the activity is currently active.
        start_time: Timestamp string when activity started (HH:MM:SS.microseconds).
        ocr_start_time: OCR-extracted timestamp from frame (HH:MM:SS format).
        ocr_end_time: OCR-extracted end timestamp (HH:MM:SS format).
        start_frame_count: Frame count when activity started.
        last_frame_count: Last frame count where activity was detected.
        frames: List of frame indices during the activity.
        duration: Duration of the activity in seconds.
        person_roles: Dictionary of person roles involved in the activity.
        first_detection_time: Timestamp of first detection.
        last_detection_time: Timestamp of last detection.
    """

    active: bool = False
    start_time: Optional[str] = None
    ocr_start_time: Optional[str] = None
    ocr_end_time: Optional[str] = None
    start_frame_count: int = 0
    last_frame_count: int = 0
    frames: List[int] = field(default_factory=list)
    duration: float = 0.0
    person_roles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    first_detection_time: Optional[str] = None
    last_detection_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for backward compatibility."""
        return {
            'active': self.active,
            'start_time': self.start_time,
            'ocr_start_time': self.ocr_start_time,
            'ocr_end_time': self.ocr_end_time,
            'start_frame_count': self.start_frame_count,
            'last_frame_count': self.last_frame_count,
            'frames': self.frames,
            'duration': self.duration,
            'person_roles': self.person_roles,
            'first_detection_time': self.first_detection_time,
            'last_detection_time': self.last_detection_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivityState':
        """Create ActivityState from dictionary."""
        return cls(
            active=data.get('active', False),
            start_time=data.get('start_time'),
            ocr_start_time=data.get('ocr_start_time'),
            ocr_end_time=data.get('ocr_end_time'),
            start_frame_count=data.get('start_frame_count', 0),
            last_frame_count=data.get('last_frame_count', 0),
            frames=data.get('frames', []),
            duration=data.get('duration', 0.0),
            person_roles=data.get('person_roles', {}),
            first_detection_time=data.get('first_detection_time'),
            last_detection_time=data.get('last_detection_time'),
        )


class ActivityTracker:
    """Manages activity tracking state per person.

    This class handles the lifecycle of activity detection including:
    - Starting and ending activities
    - Tracking consecutive detections
    - Managing grace periods for detection gaps
    - Generating activity records

    Attributes:
        activity_configs: Dictionary mapping activity names to their configurations.
        fps: Frames per second for the video being processed.
        activities: Dictionary tracking state for each activity type.
        consecutive_detections: Counter for consecutive detections per activity.
        grace_counters: Grace period counters per activity.
        activity_thresholds: Threshold configuration per activity (for backward compatibility).
        per_person_consecutive_detections: Per-person consecutive detection tracking.
        per_person_grace_counters: Per-person grace period tracking.
        all_activities: List of finalized activity records.
    """

    def __init__(
        self,
        activity_configs: Dict[str, ActivityConfig],
        fps: float = 1.0
    ) -> None:
        """Initialize the ActivityTracker.

        Args:
            activity_configs: Dictionary mapping activity names to ActivityConfig objects.
            fps: Frames per second for time calculations. Defaults to 1.0.
        """
        self.logger = get_logger('ActivityTracker')
        self.activity_configs = activity_configs
        self.fps = fps

        # Initialize activity state dictionaries
        self.activities: Dict[str, Dict[str, Any]] = {
            name: {
                'active': False,
                'start_time': None,
                'ocr_start_time': None,
                'frames': [],
                'duration': 0
            }
            for name in activity_configs
        }

        # Build activity_thresholds from configs (for backward compatibility)
        self.activity_thresholds: Dict[str, Dict[str, Any]] = {}
        for name, cfg in activity_configs.items():
            entry: Dict[str, Any] = {
                'min_duration': cfg.min_duration,
                'required_consecutive': cfg.required_consecutive,
                'margin': cfg.margin,
                'grace_frames': cfg.grace_frames,
            }
            # Include optional fields only when set
            if cfg.region_margin is not None:
                entry['region_margin'] = cfg.region_margin
            if cfg.wrist_inside_margin is not None:
                entry['wrist_inside_margin'] = cfg.wrist_inside_margin
            if cfg.sustained_proximity_seconds is not None:
                entry['sustained_proximity_seconds'] = cfg.sustained_proximity_seconds
            self.activity_thresholds[name] = entry

        # Consecutive detection counters for temporal filtering
        self.consecutive_detections: Dict[str, int] = {name: 0 for name in activity_configs}

        # Grace period counters - allows brief interruptions without resetting
        self.grace_counters: Dict[str, int] = {name: 0 for name in activity_configs}

        # Per-person tracking (for multi-person scenarios)
        self.per_person_consecutive_detections: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.per_person_grace_counters: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # All finalized activity records
        self.all_activities: List[Dict[str, Any]] = []

        self.logger.info(
            f"ActivityTracker initialized with {len(activity_configs)} activity types, fps={fps}"
        )

    def start_activity(
        self,
        activity_name: str,
        timestamp: str,
        fps: float,
        frame_count: int,
        person_roles: Optional[Dict[int, Dict[str, Any]]] = None,
        ocr_timestamp: Optional[str] = None,
        frame_idx_buffer: Optional[List[int]] = None
    ) -> None:
        """Start tracking a new activity.

        Args:
            activity_name: Name of the activity to start.
            timestamp: Timestamp when activity started (video playback time, HH:MM:SS format).
            fps: Frames per second of the video.
            frame_count: Frame count when activity started.
            person_roles: Optional dictionary of person roles.
            ocr_timestamp: Optional OCR-extracted timestamp from frame (HH:MM:SS format).
            frame_idx_buffer: Optional list of frame indices to store as activity frames.
        """
        if activity_name not in self.activities:
            self.logger.warning(f"Unknown activity: {activity_name}")
            return

        activity = self.activities[activity_name]

        if not activity['active']:
            activity['active'] = True
            activity['start_time'] = timestamp

            # OCR timestamp handling
            ocr_ts = ocr_timestamp if ocr_timestamp else None
            activity['ocr_start_time'] = ocr_ts
            activity['start_frame_count'] = frame_count
            activity['last_frame_count'] = frame_count

            # Store frame indices if provided
            if frame_idx_buffer is not None:
                activity['frames'] = list(frame_idx_buffer)
            else:
                activity['frames'] = []

            activity['duration'] = 0
            activity['person_roles'] = person_roles if person_roles else {}

            # Track actual detection timestamps for precise clip duration
            activity['first_detection_time'] = timestamp
            activity['last_detection_time'] = timestamp

            # Log with OCR timestamp if available
            if ocr_ts:
                self.logger.info(
                    f"[{timestamp}] Activity started: {activity_name} (Frame timestamp: {ocr_ts})"
                )
            else:
                self.logger.info(f"[{timestamp}] Activity started: {activity_name}")

    def end_activity(
        self,
        activity_name: str,
        timestamp: str,
        fps: float,
        frame_count: int,
        people_count: int = 1,
        ocr_timestamp: Optional[str] = None,
        sample_fps: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """End tracking an activity and create a record if it meets minimum duration.

        This method ends the activity, calculates duration, and returns an activity
        record if the activity meets the minimum duration threshold.

        Args:
            activity_name: Name of the activity to end.
            timestamp: Timestamp when activity ended (video playback time).
            fps: Frames per second of the video.
            frame_count: Frame count when activity ended.
            people_count: Number of people involved. Defaults to 1.
            ocr_timestamp: Optional OCR-extracted end timestamp.
            sample_fps: Sample FPS used for frame capture. Defaults to self.fps.

        Returns:
            Activity record dictionary if activity meets minimum duration, None otherwise.
        """
        if activity_name not in self.activities:
            self.logger.warning(f"Unknown activity: {activity_name}")
            return None

        activity = self.activities[activity_name]

        if not activity['active']:
            return None

        activity['active'] = False

        # Handle OCR end timestamp
        ocr_ts = None
        if ocr_timestamp:
            ocr_ts = ocr_timestamp
            self.logger.info(f"[OCR END] Using provided timestamp: {ocr_ts}")
        else:
            self.logger.info("[OCR END] Will calculate from start + duration (more reliable)")

        activity['ocr_end_time'] = ocr_ts

        # Calculate duration based on captured frames
        effective_fps = sample_fps if sample_fps is not None else self.fps
        total_clip_frames = len(activity['frames'])
        actual_clip_duration = total_clip_frames / effective_fps if effective_fps > 0 else 0

        # Check minimum duration threshold
        min_duration = self.activity_thresholds[activity_name]['min_duration']

        if actual_clip_duration < min_duration:
            self.logger.debug(
                f"[{timestamp}] Activity '{activity_name}' too short "
                f"({actual_clip_duration:.2f}s < {min_duration}s) - discarded"
            )
            activity['frames'] = []
            activity['duration'] = 0
            self.consecutive_detections[activity_name] = 0
            self.grace_counters[activity_name] = 0
            return None

        # Calculate precise activity duration from detection window
        first_detection = activity.get('first_detection_time', activity['start_time'])
        last_detection = activity.get('last_detection_time', activity['start_time'])

        first_detection_seconds = self._time_to_seconds(first_detection)
        last_detection_seconds = self._time_to_seconds(last_detection)
        actual_activity_duration = last_detection_seconds - first_detection_seconds

        # Calculate OCR end time if we have start but not end
        ocr_start_time_str = activity.get('ocr_start_time')
        ocr_end_time_str = activity.get('ocr_end_time')

        if ocr_start_time_str and not ocr_end_time_str:
            ocr_start_seconds = self._time_to_seconds(ocr_start_time_str)
            ocr_end_seconds = ocr_start_seconds + actual_clip_duration
            hours = int(ocr_end_seconds // 3600)
            minutes = int((ocr_end_seconds % 3600) // 60)
            seconds = int(ocr_end_seconds % 60)
            ocr_end_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            activity['ocr_end_time'] = ocr_end_time_str

        # Create activity record
        start_frame = activity.get('start_frame_count', frame_count)

        activity_record = {
            'activity_name': activity_name,
            'start_time': activity['start_time'],
            'end_time': timestamp,
            'ocr_start_time': activity.get('ocr_start_time'),
            'ocr_end_time': activity.get('ocr_end_time'),
            'start_frame': start_frame,
            'end_frame': frame_count,
            'duration': actual_clip_duration,
            'actual_activity_duration': actual_activity_duration,
            'first_detection_time': first_detection,
            'last_detection_time': last_detection,
            'first_detection_seconds': first_detection_seconds,
            'last_detection_seconds': last_detection_seconds,
            'frame_count': total_clip_frames,
            'frames': activity['frames'].copy(),
            'person_roles': activity.get('person_roles', {}),
            'people_count': len(activity.get('person_roles', {})) if activity.get('person_roles') else people_count,
        }

        # Calculate end time string for logging
        end_time_str = str(timedelta(seconds=last_detection_seconds))

        self.logger.info(f"[{end_time_str}] Activity ended: {activity_name}")
        self.logger.info(
            f"  Clip Duration: {actual_clip_duration:.2f}s ({total_clip_frames} frames @ {effective_fps} FPS)"
        )
        self.logger.debug(
            f"  Min Duration Threshold: {min_duration}s | "
            f"Required Consecutive: {self.activity_thresholds[activity_name]['required_consecutive']} frames"
        )

        # Reset activity state
        activity['frames'] = []
        activity['duration'] = 0
        self.consecutive_detections[activity_name] = 0
        self.grace_counters[activity_name] = 0

        return activity_record

    def update_detection(
        self,
        activity_name: str,
        detected: bool,
        timestamp: str,
        frame_idx: int,
        person_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update detection state for an activity.

        This method manages consecutive detection counting and grace periods.
        It returns information about whether the activity should be started or ended.

        Args:
            activity_name: Name of the activity.
            detected: Whether the activity was detected in this frame.
            timestamp: Current timestamp string.
            frame_idx: Current frame index.
            person_idx: Optional person index for per-person tracking.

        Returns:
            Dictionary with keys:
                - 'should_start': True if activity should be started
                - 'should_end': True if activity should be ended
                - 'consecutive_count': Current consecutive detection count
                - 'grace_remaining': Remaining grace frames
        """
        if activity_name not in self.activities:
            return {
                'should_start': False,
                'should_end': False,
                'consecutive_count': 0,
                'grace_remaining': 0
            }

        config = self.activity_thresholds[activity_name]
        required_consecutive = config['required_consecutive']
        grace_frames = config['grace_frames']

        # Use per-person tracking if person_idx provided
        if person_idx is not None:
            consecutive = self.per_person_consecutive_detections[person_idx]
            grace = self.per_person_grace_counters[person_idx]
        else:
            consecutive = self.consecutive_detections
            grace = self.grace_counters

        activity = self.activities[activity_name]
        should_start = False
        should_end = False

        if detected:
            # Increment consecutive detection count
            if person_idx is not None:
                consecutive[activity_name] += 1
                grace[activity_name] = 0
            else:
                consecutive[activity_name] += 1
                grace[activity_name] = 0

            # Update last detection time if activity is active
            if activity['active']:
                activity['last_detection_time'] = timestamp
                activity['last_frame_count'] = frame_idx

            # Check if we should start the activity
            current_count = consecutive[activity_name] if person_idx is None else consecutive[activity_name]
            if not activity['active'] and current_count >= required_consecutive:
                should_start = True

        else:
            # Activity not detected - use grace period
            if activity['active']:
                if person_idx is not None:
                    grace[activity_name] += 1
                    if grace[activity_name] >= grace_frames:
                        should_end = True
                        consecutive[activity_name] = 0
                else:
                    grace[activity_name] += 1
                    if grace[activity_name] >= grace_frames:
                        should_end = True
                        consecutive[activity_name] = 0
            else:
                # Reset consecutive count if not active and not detected
                if person_idx is not None:
                    consecutive[activity_name] = 0
                else:
                    consecutive[activity_name] = 0

        current_consecutive = consecutive[activity_name] if person_idx is None else consecutive[activity_name]
        current_grace = grace[activity_name] if person_idx is None else grace[activity_name]

        return {
            'should_start': should_start,
            'should_end': should_end,
            'consecutive_count': current_consecutive,
            'grace_remaining': grace_frames - current_grace if activity['active'] else grace_frames
        }

    def add_frame_to_activity(self, activity_name: str, frame_idx: int) -> None:
        """Add a frame index to an active activity.

        Args:
            activity_name: Name of the activity.
            frame_idx: Frame index to add.
        """
        if activity_name in self.activities and self.activities[activity_name]['active']:
            self.activities[activity_name]['frames'].append(frame_idx)

    def get_active_activities(self, person_idx: Optional[int] = None) -> List[str]:
        """Get list of currently active activities.

        Args:
            person_idx: Optional person index (not used in current implementation,
                        reserved for future per-person activity tracking).

        Returns:
            List of activity names that are currently active.
        """
        return [name for name, state in self.activities.items() if state['active']]

    def get_activity_state(self, activity_name: str) -> Optional[Dict[str, Any]]:
        """Get the current state of an activity.

        Args:
            activity_name: Name of the activity.

        Returns:
            Dictionary with activity state, or None if activity doesn't exist.
        """
        if activity_name in self.activities:
            return self.activities[activity_name].copy()
        return None

    def is_activity_active(self, activity_name: str) -> bool:
        """Check if an activity is currently active.

        Args:
            activity_name: Name of the activity.

        Returns:
            True if the activity is active, False otherwise.
        """
        if activity_name in self.activities:
            return self.activities[activity_name]['active']
        return False

    def get_threshold(self, activity_name: str, key: str) -> Any:
        """Get a threshold value for an activity.

        Args:
            activity_name: Name of the activity.
            key: Threshold key (e.g., 'min_duration', 'required_consecutive').

        Returns:
            Threshold value, or None if not found.
        """
        if activity_name in self.activity_thresholds:
            return self.activity_thresholds[activity_name].get(key)
        return None

    def reset_activity(self, activity_name: str) -> None:
        """Reset an activity to its initial state.

        Args:
            activity_name: Name of the activity to reset.
        """
        if activity_name in self.activities:
            self.activities[activity_name] = {
                'active': False,
                'start_time': None,
                'ocr_start_time': None,
                'frames': [],
                'duration': 0
            }
            self.consecutive_detections[activity_name] = 0
            self.grace_counters[activity_name] = 0
            self.logger.debug(f"Activity '{activity_name}' reset to initial state")

    def reset_all_activities(self) -> None:
        """Reset all activities to their initial state."""
        for activity_name in self.activities:
            self.reset_activity(activity_name)
        self.logger.info("All activities reset to initial state")

    def cleanup_stale_person_tracking(self, active_person_indices: List[int]) -> int:
        """Remove entries from per-person tracking dicts for persons no longer detected.

        This prevents unbounded memory growth when person indices change over time.

        Args:
            active_person_indices: List of person indices currently detected in the frame.

        Returns:
            Number of stale entries removed.
        """
        active_set = set(active_person_indices)
        total_removed = 0

        # Clean up per-person consecutive detections
        stale_keys = set(self.per_person_consecutive_detections.keys()) - active_set
        for stale_key in stale_keys:
            del self.per_person_consecutive_detections[stale_key]
            total_removed += 1

        # Clean up per-person grace counters
        stale_keys = set(self.per_person_grace_counters.keys()) - active_set
        for stale_key in stale_keys:
            del self.per_person_grace_counters[stale_key]
            total_removed += 1

        if total_removed > 0:
            self.logger.debug(
                f"Cleaned up {total_removed} stale person tracking entries "
                f"(active persons: {sorted(active_set)})"
            )

        return total_removed

    def finalize_all_activities(self, timestamp: str, fps: float, frame_count: int) -> List[Dict[str, Any]]:
        """End all active activities and return their records.

        This should be called at the end of processing to finalize any
        activities that are still active.

        Args:
            timestamp: Final timestamp.
            fps: Frames per second.
            frame_count: Final frame count.

        Returns:
            List of activity records for all finalized activities.
        """
        finalized = []
        active_activities = self.get_active_activities()

        for activity_name in active_activities:
            record = self.end_activity(
                activity_name=activity_name,
                timestamp=timestamp,
                fps=fps,
                frame_count=frame_count
            )
            if record is not None:
                finalized.append(record)

        self.logger.info(f"Finalized {len(finalized)} active activities")
        return finalized

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS.microseconds to seconds.

        Args:
            time_str: Time string in HH:MM:SS or HH:MM:SS.microseconds format.

        Returns:
            Time in seconds as float.
        """
        if time_str is None:
            return 0.0

        parts = time_str.split(':')
        if len(parts) != 3:
            return 0.0

        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
