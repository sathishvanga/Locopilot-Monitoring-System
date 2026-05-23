"""Unit tests for T3: HandHistoryTracker + coordination helper."""

from __future__ import annotations

from app.core.tracking.coordination import check_hand_gesture_coordination
from app.core.tracking.hand_history import HandHistoryTracker


class _FakeLandmark:
    __slots__ = ('x', 'y', 'visibility')

    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class _FakeLandmarks:
    """Stand-in for a MediaPipe-/YOLO-pose landmark container.

    The HandHistoryTracker reads wrist positions via ``get_keypoint(landmarks,
    'right_wrist')`` and ``get_keypoint(landmarks, 'left_wrist')``. We expose
    those two slots directly and provide a matching ``get_keypoint`` callable.
    """

    def __init__(self, right_wrist: _FakeLandmark, left_wrist: _FakeLandmark):
        self.right_wrist = right_wrist
        self.left_wrist = left_wrist


def _get_keypoint(landmarks, name: str) -> _FakeLandmark:
    return getattr(landmarks, name)


def test_smoothing_buffers_and_position_history_are_plain_dicts():
    tracker = HandHistoryTracker()
    assert type(tracker.smoothing_buffers) is dict
    assert type(tracker.position_history) is dict


def test_velocity_trajectory_quality_progresses_to_good_on_5th_call():
    """Push 5 fake landmark frames; on the 5th call the analyzer should
    report ``analysis_quality='good'`` (history length >= 5)."""
    tracker = HandHistoryTracker()
    frame_shape = (480, 640, 3)

    # Generate 5 frames of upward-moving wrists (rapid raise).
    # x in normalized [0,1], y decreasing (moving up).
    results = []
    for i in range(5):
        right = _FakeLandmark(x=0.5, y=0.9 - i * 0.05)
        left = _FakeLandmark(x=0.4, y=0.85 - i * 0.05)
        landmarks = _FakeLandmarks(right_wrist=right, left_wrist=left)
        result = tracker.analyze_velocity_and_trajectory(
            person_idx=0,
            landmarks=landmarks,
            frame_shape=frame_shape,
            timestamp_sec=float(i) * 0.5,
            get_keypoint=_get_keypoint,
        )
        results.append(result)

    # Calls 1 and 2 have insufficient_data (< 3 timestamps).
    assert results[0]['analysis_quality'] == 'insufficient_data'
    assert results[1]['analysis_quality'] == 'insufficient_data'
    # Call 3 has 'limited' (>=3 but <5 timestamps).
    assert results[2]['analysis_quality'] == 'limited'
    # Call 4 has 'limited' (4 timestamps).
    assert results[3]['analysis_quality'] == 'limited'
    # Call 5 should report 'good' (>=5 timestamps).
    assert results[4]['analysis_quality'] == 'good'


def test_check_hand_gesture_coordination_alp_only_flags_lp_not_coordinating():
    """When only ALP raised a hand and the LP has never raised, LP is flagged
    as not coordinating."""
    lp_not_coord, alp_not_coord = check_hand_gesture_coordination(
        lp_detected=False,
        alp_detected=True,
        current_time=10.0,
        recent_person_activities={},
        hand_gesture_coordination_window=5.0,
    )
    assert lp_not_coord is True
    assert alp_not_coord is False


def test_check_hand_gesture_coordination_lp_only_flags_alp_not_coordinating():
    lp_not_coord, alp_not_coord = check_hand_gesture_coordination(
        lp_detected=True,
        alp_detected=False,
        current_time=10.0,
        recent_person_activities={},
        hand_gesture_coordination_window=5.0,
    )
    assert lp_not_coord is False
    assert alp_not_coord is True


def test_check_hand_gesture_coordination_both_within_window_clears_flag():
    """If both raised hands within the coordination window, no failure is flagged."""
    recent = {
        0: {'lp_hand_raise': 9.5},
        1: {'alp_hand_raise': 9.0},
    }
    lp_not_coord, alp_not_coord = check_hand_gesture_coordination(
        lp_detected=True,
        alp_detected=False,
        current_time=10.0,
        recent_person_activities=recent,
        hand_gesture_coordination_window=5.0,
    )
    assert lp_not_coord is False
    assert alp_not_coord is False


def test_packing_wrist_motion_gate_disabled_returns_true():
    tracker = HandHistoryTracker(packing_wrist_motion_gate_enabled=False)
    assert tracker.check_wrist_motion_for_packing(person_idx=0, timestamp_sec=1.0) is True


def test_packing_wrist_motion_no_history_allows_detection():
    tracker = HandHistoryTracker()
    # No history seeded yet -> allow detection.
    assert tracker.check_wrist_motion_for_packing(person_idx=42, timestamp_sec=1.0) is True
