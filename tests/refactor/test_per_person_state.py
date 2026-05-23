"""Unit tests for the extracted PerPersonState (T7)."""

from collections import defaultdict

from app.core.tracking.per_person_state import PerPersonState


class DummySleepDetector:
    def __init__(self):
        self.calls = []

    def cleanup_stale_tracking(self, active_set):
        self.calls.append(set(active_set))


def test_default_dict_types_match_monolith():
    pps = PerPersonState()
    # Plain dicts keyed by activity name in the monolith.
    assert isinstance(pps.consecutive_detections, dict)
    assert not isinstance(pps.consecutive_detections, defaultdict)
    assert isinstance(pps.grace_counters, dict)
    assert not isinstance(pps.grace_counters, defaultdict)

    # Per-person dicts are defaultdict(lambda: defaultdict(int)).
    assert isinstance(pps.per_person_consecutive_detections, defaultdict)
    assert isinstance(pps.per_person_grace_counters, defaultdict)
    inner = pps.per_person_consecutive_detections[0]
    assert isinstance(inner, defaultdict)
    assert inner['unseen_activity'] == 0  # defaultdict(int) auto-init

    # Plain dicts.
    for d in (
        pps.hand_position_history,
        pps.landmark_stability_history,
        pps.wrist_proximity_tracking,
        pps.no_pose_sleep_tracking,
        pps.recent_person_activities,
        pps.hand_smoothing_buffers,
    ):
        assert isinstance(d, dict)
        assert not isinstance(d, defaultdict)


def test_update_consecutive_triggers_when_threshold_met():
    pps = PerPersonState()
    triggered = pps.update_consecutive(
        0, 'cell_phone', True, 1.0,
        required_consecutive=1, grace_frames=2,
    )
    assert triggered is True
    # Counter incremented and grace reset.
    assert pps.per_person_consecutive_detections[0]['cell_phone'] == 1
    assert pps.per_person_grace_counters[0]['cell_phone'] == 0


def test_update_consecutive_below_threshold_returns_false():
    pps = PerPersonState()
    triggered = pps.update_consecutive(
        0, 'writing', True, 1.0,
        required_consecutive=3, grace_frames=2,
    )
    assert triggered is False
    assert pps.per_person_consecutive_detections[0]['writing'] == 1


def test_update_consecutive_grace_period_resets():
    pps = PerPersonState()
    # Build up to 2 detections (below threshold of 5).
    for _ in range(2):
        pps.update_consecutive(0, 'writing', True, 1.0,
                               required_consecutive=5, grace_frames=2)
    assert pps.per_person_consecutive_detections[0]['writing'] == 2

    # 1st miss: grace +=1 (still 1, not > 2)
    pps.update_consecutive(0, 'writing', False, 2.0,
                           required_consecutive=5, grace_frames=2)
    assert pps.per_person_consecutive_detections[0]['writing'] == 2
    assert pps.per_person_grace_counters[0]['writing'] == 1

    # 2nd miss: grace becomes 2; not > 2, so still no reset.
    pps.update_consecutive(0, 'writing', False, 3.0,
                           required_consecutive=5, grace_frames=2)
    assert pps.per_person_consecutive_detections[0]['writing'] == 2
    assert pps.per_person_grace_counters[0]['writing'] == 2

    # 3rd miss: grace becomes 3 (> 2) -> resets both counters.
    pps.update_consecutive(0, 'writing', False, 4.0,
                           required_consecutive=5, grace_frames=2)
    assert pps.per_person_consecutive_detections[0]['writing'] == 0
    assert pps.per_person_grace_counters[0]['writing'] == 0


def test_cleanup_stale_removes_inactive_indices_from_every_dict():
    pps = PerPersonState()

    # Seed all dicts for indices 0, 1, 2.
    for i in (0, 1, 2):
        pps.per_person_consecutive_detections[i]['cell_phone'] = i + 1
        pps.per_person_grace_counters[i]['cell_phone'] = i
        pps.hand_position_history[i] = {'right_wrist': 'r', 'left_wrist': 'l'}
        pps.landmark_stability_history[i] = {'right_shoulder': 's'}
        pps.wrist_proximity_tracking[i] = {'consecutive_frames': 1}
        pps.no_pose_sleep_tracking[i] = {'first_seen': i}
        pps.recent_person_activities[i] = {'writing': float(i)}
        pps.hand_smoothing_buffers[(i, 'right')] = {'positions': []}
        pps.hand_smoothing_buffers[(i, 'left')] = {'positions': []}

    sleep_detector = DummySleepDetector()
    removed = pps.cleanup_stale({0}, sleep_detector=sleep_detector)

    # Sleep detector got called with the active set.
    assert sleep_detector.calls == [{0}]

    # Indices 1 and 2 are gone from every dict; index 0 remains.
    for d in (
        pps.per_person_consecutive_detections,
        pps.per_person_grace_counters,
        pps.hand_position_history,
        pps.landmark_stability_history,
        pps.wrist_proximity_tracking,
        pps.no_pose_sleep_tracking,
        pps.recent_person_activities,
    ):
        assert 0 in d
        assert 1 not in d
        assert 2 not in d

    # Tuple-keyed dict: (1, *) and (2, *) gone, (0, *) keys remain.
    remaining_keys = set(pps.hand_smoothing_buffers.keys())
    assert remaining_keys == {(0, 'right'), (0, 'left')}

    # Total removed: 7 plain dicts × 2 (indices 1, 2) + 4 tuple keys = 18.
    assert removed == 18


def test_cleanup_stale_no_op_when_all_active():
    pps = PerPersonState()
    pps.recent_person_activities[0] = {'writing': 1.0}
    pps.recent_person_activities[1] = {'writing': 1.0}
    sleep_detector = DummySleepDetector()
    removed = pps.cleanup_stale({0, 1}, sleep_detector=sleep_detector)
    assert removed == 0
    assert sleep_detector.calls == [{0, 1}]


def test_cleanup_stale_logs_when_entries_removed():
    pps = PerPersonState()
    pps.recent_person_activities[5] = {}

    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def debug(self, msg):
            self.messages.append(msg)

    logger = CapturingLogger()
    sleep_detector = DummySleepDetector()
    removed = pps.cleanup_stale({0}, sleep_detector=sleep_detector, logger=logger)
    assert removed == 1
    assert any('[CR-012]' in m for m in logger.messages)
