"""Unit tests for the lifted pose validators (T4).

These tests pin the public surface of ``app.core.utils.pose_validators`` so a
behavior-changing edit during the rewire (Section 3 / TR) is caught
immediately. They use plain stub objects -- no MediaPipe / cv2 / YOLO --
to keep the test pure.
"""
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.utils.pose_validators import (
    check_landmark_stability,
    validate_anatomical_consistency,
    validate_pose_landmarks,
)


def _stub_landmark(x: float, y: float, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, visibility=visibility)


# ---------------------------------------------------------------------------
# validate_pose_landmarks
# ---------------------------------------------------------------------------

def test_validate_pose_landmarks_returns_false_when_below_min_count() -> None:
    stub = [_stub_landmark(0.5, 0.5) for _ in range(9)]
    assert validate_pose_landmarks(stub, min_landmarks=10) is False


def test_validate_pose_landmarks_returns_false_when_landmarks_none() -> None:
    assert validate_pose_landmarks(None, min_landmarks=10) is False


def test_validate_pose_landmarks_returns_true_when_sufficient_visible() -> None:
    stub = [_stub_landmark(0.5, 0.5, visibility=0.9) for _ in range(12)]
    assert validate_pose_landmarks(stub, min_landmarks=10, min_visibility=0.3) is True


def test_validate_pose_landmarks_rejects_low_visibility() -> None:
    stub = [_stub_landmark(0.5, 0.5, visibility=0.1) for _ in range(12)]
    assert validate_pose_landmarks(stub, min_landmarks=10, min_visibility=0.3) is False


# ---------------------------------------------------------------------------
# validate_anatomical_consistency (smoke-only: ensure callable surface
# matches and obviously-wrong inputs return False)
# ---------------------------------------------------------------------------

def test_validate_anatomical_consistency_inverted_person() -> None:
    """Nose well below the shoulders triggers the 'inverted detection' rule."""
    landmarks_by_name = {
        'right_shoulder': _stub_landmark(0.4, 0.30, visibility=0.9),
        'left_shoulder': _stub_landmark(0.6, 0.30, visibility=0.9),
        'right_elbow':    _stub_landmark(0.35, 0.45),
        'left_elbow':     _stub_landmark(0.65, 0.45),
        'right_wrist':    _stub_landmark(0.30, 0.60),
        'left_wrist':     _stub_landmark(0.70, 0.60),
        'right_hip':      _stub_landmark(0.42, 0.70),
        'left_hip':       _stub_landmark(0.58, 0.70),
        # Nose far below shoulders => inverted -> False
        'nose':           _stub_landmark(0.5, 0.90, visibility=0.9),
    }

    def get_keypoint(_pose: Any, name: str) -> SimpleNamespace:
        return landmarks_by_name[name]

    is_valid, reason = validate_anatomical_consistency(
        pose_landmarks=object(),
        frame_shape=(480, 640),
        get_keypoint=get_keypoint,
    )
    assert is_valid is False
    assert reason == "Nose below shoulders (inverted detection)"


# ---------------------------------------------------------------------------
# check_landmark_stability (history dict mutation contract)
# ---------------------------------------------------------------------------

def test_check_landmark_stability_mutates_history_in_place() -> None:
    """First call seeds history; second call computes a jump."""
    history: dict = {}

    landmarks_by_name = {
        'right_shoulder': _stub_landmark(0.5, 0.5),
        'left_shoulder':  _stub_landmark(0.6, 0.5),
    }

    def get_keypoint(_pose: Any, name: str) -> SimpleNamespace:
        return landmarks_by_name[name]

    is_stable_first, max_jump_first = check_landmark_stability(
        person_idx=0,
        pose_landmarks=object(),
        frame_shape=(480, 640),
        history=history,
        max_jump_threshold=100.0,
        get_keypoint=get_keypoint,
    )
    # First sample: not enough data -> assumed stable, jump 0.
    assert is_stable_first is True
    assert max_jump_first == 0
    # The validator must have populated `history` IN PLACE.
    assert 0 in history
    assert 'right_shoulder' in history[0]
    assert 'left_shoulder' in history[0]
    assert len(history[0]['right_shoulder']) == 1

    # Second call with identical landmarks -> still stable, jump 0.
    is_stable_second, max_jump_second = check_landmark_stability(
        person_idx=0,
        pose_landmarks=object(),
        frame_shape=(480, 640),
        history=history,
        max_jump_threshold=100.0,
        get_keypoint=get_keypoint,
    )
    assert is_stable_second is True
    assert max_jump_second == 0
    assert len(history[0]['right_shoulder']) == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
