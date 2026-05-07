"""Tests for the extracted mind-diversion suppression rules (T8)."""

from types import SimpleNamespace

from app.core.detectors.mind_diversion_suppression import (
    should_suppress_mind_diversion,
)


def _stub_settings():
    return SimpleNamespace(
        mind_diversion_suppress_with_writing=True,
        mind_diversion_writing_grace_seconds=10.0,
        mind_diversion_wrist_distance_threshold=80.0,
    )


def _no_kp(_pose, _name):  # pragma: no cover - not invoked in this test
    raise AssertionError("get_keypoint should not be called when writing is active")


def test_writing_active_returns_suppressed_writing_active():
    """person_activities={'writing': True} -> (True, 'suppressed_writing_active')."""
    suppressed, reason = should_suppress_mind_diversion(
        person_idx=0,
        person_activities={'writing': True},
        pose_landmarks=None,
        detections={},
        frame_shape=(720, 1280),
        current_time=None,
        settings=_stub_settings(),
        recent_person_activities={},
        get_keypoint=_no_kp,
    )
    assert suppressed is True
    assert reason == "suppressed_writing_active"


def test_no_signals_returns_false_none():
    """No writing, no book, no pose -> (False, None)."""
    suppressed, reason = should_suppress_mind_diversion(
        person_idx=0,
        person_activities={},
        pose_landmarks=None,
        detections={},
        frame_shape=(720, 1280),
        current_time=None,
        settings=_stub_settings(),
        recent_person_activities={},
        get_keypoint=_no_kp,
    )
    assert suppressed is False
    assert reason is None


def test_book_detection_suppresses():
    suppressed, reason = should_suppress_mind_diversion(
        person_idx=1,
        person_activities={},
        pose_landmarks=None,
        detections={'book': [[0, 0, 10, 10]]},
        frame_shape=(720, 1280),
        current_time=None,
        settings=_stub_settings(),
        recent_person_activities={},
        get_keypoint=_no_kp,
    )
    assert suppressed is True
    assert reason == "suppressed_book_detected"


def test_recent_writing_within_grace_window():
    suppressed, reason = should_suppress_mind_diversion(
        person_idx=2,
        person_activities={},
        pose_landmarks=None,
        detections={},
        frame_shape=(720, 1280),
        current_time=12.0,
        settings=_stub_settings(),
        recent_person_activities={2: {'writing': 8.0}},  # 4s ago, within 10s grace
        get_keypoint=_no_kp,
    )
    assert suppressed is True
    assert reason == "suppressed_recent_writing"
