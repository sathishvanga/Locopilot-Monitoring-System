"""Per-person tracking state bag (T7).

Bundles the cluster of per-person bookkeeping dicts that previously lived as
separate attributes on ``LocopilotActivityMonitor`` plus the small
consecutive-detection counter helper and the stale-entry cleanup routine.

The dicts on this class use the EXACT same Python types as the monolith's
``__init__`` so that the rewire (Section 3) can alias the live attributes
without changing semantics.
"""

import logging
from collections import defaultdict
from typing import Optional


class PerPersonState:
    """Owns the per-person tracking dicts that previously lived as separate
    attributes on LocopilotActivityMonitor.

    The monitor (post-rewire) keeps direct attribute aliases to these dicts
    for backward compatibility — the dicts are the same objects.
    """

    def __init__(self) -> None:
        # Plain dicts keyed by activity name (NOT by person_idx).
        # Initialised in the monolith via ``{name: 0 for name in ACTIVITY_REGISTRY}``;
        # we keep them as plain ``dict`` so callers populate them per
        # activity name as they do today.
        self.consecutive_detections: dict = {}
        self.grace_counters: dict = {}

        # Per-person consecutive detection tracking for temporal filtering.
        # Format: {person_idx: defaultdict(int)} — uses defaultdict so any
        # activity_type key auto-initialises.
        self.per_person_consecutive_detections: defaultdict = defaultdict(lambda: defaultdict(int))
        self.per_person_grace_counters: defaultdict = defaultdict(lambda: defaultdict(int))

        # Hand position history for velocity/trajectory analysis
        # Format: {person_idx: {'right_wrist': deque, 'left_wrist': deque, 'timestamps': deque}}
        self.hand_position_history: dict = {}

        # Landmark stability tracking
        # Format: {person_idx: {'right_shoulder': deque, 'left_shoulder': deque}}
        self.landmark_stability_history: dict = {}

        # Wrist proximity tracking for writing detection (per person)
        # Format: {person_idx: {...}} or {f"book_posture_{person_idx}": {...}}
        self.wrist_proximity_tracking: dict = {}

        # No-pose sleep detection tracking (for IR mode where YOLO pose fails)
        # Format: {person_idx: {'first_seen': timestamp, ...}}
        self.no_pose_sleep_tracking: dict = {}

        # Recent activities per person, used for temporal suppression / coordination.
        # Format: {person_idx: {'writing': last_timestamp, ...}}
        self.recent_person_activities: dict = {}

        # Hand smoothing buffers — tuple-keyed by (person_idx, hand_side).
        # Format: {(person_idx, hand_side): {'positions': deque, 'timestamps': deque}}
        self.hand_smoothing_buffers: dict = {}

    # ------------------------------------------------------------------
    # Consecutive-detection counter
    # ------------------------------------------------------------------
    def update_consecutive(
        self,
        person_idx: int,
        activity_type: str,
        detected: bool,
        timestamp_sec: float,
        *,
        required_consecutive: int,
        grace_frames: int,
    ) -> bool:
        """Update per-person consecutive detection counters with temporal filtering.

        Mirrors ``LocopilotActivityMonitor.update_per_person_detection`` byte-for-byte.

        Args:
            person_idx: Person index (0, 1, 2, ...)
            activity_type: Activity name ('cell_phone', 'writing', 'packing_bags')
            detected: Boolean - was activity detected in current frame?
            timestamp_sec: Current timestamp
            required_consecutive: Threshold for triggering the activity.
            grace_frames: Number of grace frames allowed before resetting.

        Returns:
            bool: True if activity should trigger alert (threshold met)
        """
        # Access tracking for this person (defaultdict auto-initializes for any activity type)
        person_counters = self.per_person_consecutive_detections[person_idx]
        person_grace = self.per_person_grace_counters[person_idx]

        if detected:
            person_counters[activity_type] += 1
            person_grace[activity_type] = 0

            if person_counters[activity_type] >= required_consecutive:
                return True  # Trigger activity
        else:
            if person_counters[activity_type] > 0:
                person_grace[activity_type] += 1

                if person_grace[activity_type] > grace_frames:
                    person_counters[activity_type] = 0
                    person_grace[activity_type] = 0

        return False

    # ------------------------------------------------------------------
    # Stale-entry cleanup
    # ------------------------------------------------------------------
    def cleanup_stale(
        self,
        active_person_indices,
        *,
        sleep_detector,
        logger: Optional[logging.Logger] = None,
    ) -> int:
        """CR-012: Remove entries from per-person tracking dicts for persons no longer detected.

        Mirrors ``LocopilotActivityMonitor._cleanup_stale_person_tracking``.

        Args:
            active_person_indices: Set of person indices currently detected in the frame.
            sleep_detector: SleepDetector instance with cleanup_stale_tracking(active_set).
            logger: Optional logger; if provided, emits the same debug message as the monolith.

        Returns:
            int: total entries removed across all dicts.
        """
        active_set = set(active_person_indices)

        # Delegate sleep-detector cleanup to its own method (C-11 fix)
        sleep_detector.cleanup_stale_tracking(active_set)

        # All per-person tracking dictionaries to clean up
        tracking_dicts = [
            ('per_person_consecutive_detections', self.per_person_consecutive_detections),
            ('per_person_grace_counters', self.per_person_grace_counters),
            ('hand_position_history', self.hand_position_history),
            ('landmark_stability_history', self.landmark_stability_history),
            ('wrist_proximity_tracking', self.wrist_proximity_tracking),
            ('no_pose_sleep_tracking', self.no_pose_sleep_tracking),
            ('recent_person_activities', self.recent_person_activities),
        ]

        total_removed = 0
        for dict_name, tracking_dict in tracking_dicts:
            stale_keys = set(tracking_dict.keys()) - active_set
            for stale_key in stale_keys:
                del tracking_dict[stale_key]
                total_removed += 1

        # M-03: Clean up tuple-keyed dicts where key is (person_idx, hand_side).
        # These cannot be cleaned by simple set subtraction against active_set
        # because the keys are tuples, not plain integers.
        tuple_keyed_dicts = [
            ('hand_smoothing_buffers', self.hand_smoothing_buffers),
        ]

        for dict_name, tracking_dict in tuple_keyed_dicts:
            stale_keys = [
                key for key in tracking_dict
                if key[0] not in active_set
            ]
            for stale_key in stale_keys:
                del tracking_dict[stale_key]
                total_removed += 1

        if total_removed > 0 and logger is not None:
            logger.debug(f"[CR-012] Cleaned up {total_removed} stale person tracking entries "
                         f"(active persons: {sorted(active_set)})")

        return total_removed
