"""T1 unit tests for writing fallback extraction.

Verifies:
1. ``check_hand_object_interaction`` returns True for a hand within margin.
2. ``detect_writing_by_wrist_proximity`` accumulates ``consecutive_frames``
   correctly across 3 successive calls (with a stub ``activity_detector``
   and a stub landmark object).
"""

from __future__ import annotations

from app.core.detectors.writing_fallbacks import (
    WritingFallbackThresholds,
    detect_writing_by_wrist_proximity,
)
from app.core.utils.pose_checks import check_hand_object_interaction


class _StubActivityDetector:
    """Minimal stand-in for ``app.core.detectors.activity_detector``.

    The real detector returns ``(distance, source)`` from
    ``calculate_wrist_distance`` and a bool from ``detect_head_looking_down``.
    """

    def __init__(self, distance: float, source: str = 'wrist', head_down: bool = True):
        self._distance = distance
        self._source = source
        self._head_down = head_down

    def calculate_wrist_distance(self, pose_landmarks, frame_shape):
        return self._distance, self._source

    def detect_head_looking_down(self, pose_landmarks):
        return self._head_down


def test_check_hand_object_interaction_within_margin():
    """Hand at (10, 10), object [0,0,5,5], margin=10 -> within expanded box."""
    assert check_hand_object_interaction((10, 10), [0, 0, 5, 5], margin=10) is True


def test_check_hand_object_interaction_outside_margin():
    """Sanity: hand far outside the expanded margin returns False."""
    assert check_hand_object_interaction((100, 100), [0, 0, 5, 5], margin=10) is False


def test_check_hand_object_interaction_none_inputs():
    """None inputs short-circuit to False."""
    assert check_hand_object_interaction(None, [0, 0, 5, 5], margin=10) is False
    assert check_hand_object_interaction((10, 10), None, margin=10) is False


def test_writing_wrist_proximity_accumulates_consecutive_frames():
    """Three successive in-threshold + head-down calls must:

    - Increment ``consecutive_frames`` from 1 -> 2 -> 3 across calls.
    - Track ``duration`` from the first call's timestamp.
    """
    thresholds = WritingFallbackThresholds(
        max_wrist_distance=300,
        max_single_wrist_distance=200,
        max_elbow_distance=450,
        # Set high so we never short-circuit by the duration check during the
        # 3 frames of accumulation; we are asserting the counter, not the
        # CONFIRMED outcome.
        writing_required_consecutive=10,
        writing_min_duration=10.0,
        person_book_overlap_margin=20,
        book_posture_required_consecutive=2,
        book_posture_min_duration=1.0,
    )
    tracking: dict = {}
    detector = _StubActivityDetector(distance=100.0, source='wrist', head_down=True)

    # Frame 1
    out = detect_writing_by_wrist_proximity(
        pose_landmarks=object(),
        frame_shape=(720, 1280),
        person_idx=0,
        timestamp_sec=1.0,
        activity_detector=detector,
        wrist_proximity_tracking=tracking,
        thresholds=thresholds,
    )
    assert out is False  # not yet at required_consecutive=10
    assert tracking[0]['consecutive_frames'] == 1
    assert tracking[0]['start_time'] == 1.0

    # Frame 2
    detect_writing_by_wrist_proximity(
        pose_landmarks=object(),
        frame_shape=(720, 1280),
        person_idx=0,
        timestamp_sec=1.5,
        activity_detector=detector,
        wrist_proximity_tracking=tracking,
        thresholds=thresholds,
    )
    assert tracking[0]['consecutive_frames'] == 2
    assert abs(tracking[0]['duration'] - 0.5) < 1e-9

    # Frame 3
    detect_writing_by_wrist_proximity(
        pose_landmarks=object(),
        frame_shape=(720, 1280),
        person_idx=0,
        timestamp_sec=2.0,
        activity_detector=detector,
        wrist_proximity_tracking=tracking,
        thresholds=thresholds,
    )
    assert tracking[0]['consecutive_frames'] == 3
    assert abs(tracking[0]['duration'] - 1.0) < 1e-9
