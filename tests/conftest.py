"""Shared pytest fixtures for the Locopilot Monitoring System test suite.

Fixtures defined here are available to every test under ``tests/``.

Design goals:
- Zero network / GPU / model-weight requirements.
- Detectors are already constructible in isolation (they accept ``settings``
  and ``logger``) so fixtures focus on providing pure-Python stand-ins for
  pose landmarks and detection dicts.
- Callers can parametrize pose fakes to simulate alert / drowsy / head-up
  postures without loading MediaPipe or YOLO.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pytest


# ---------------------------------------------------------------------------
# Settings / logger fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_settings():
    """Return a ``Settings`` instance with defaults and no .env file.

    Tests should prefer these pristine defaults over ``get_settings()`` so
    that env-var leakage between tests is impossible.  The ``_env_file=None``
    argument is the documented pydantic-settings escape hatch for disabling
    .env auto-loading.
    """
    from app.utils.config import Settings
    return Settings(_env_file=None)


@pytest.fixture
def stub_logger():
    """Return a plain ``logging.Logger`` for passing to detector constructors.

    Kept minimal — no handlers attached.  Detectors only use ``.debug`` /
    ``.warning`` which are safe no-ops with the default logging config.
    """
    return logging.getLogger("locopilot.tests")


# ---------------------------------------------------------------------------
# Fake pose landmark helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeLandmark:
    """Minimal stand-in for a YOLO/MediaPipe keypoint.

    Exposes ``x``, ``y``, ``z``, ``visibility`` — the four attributes that
    every detector in ``app/core/detectors`` accesses.  Coordinates are
    normalized to the ``[0.0, 1.0]`` frame-relative range, matching the YOLO
    pose adapter contract.
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.95


class FakePoseLandmarks:
    """Object-style wrapper exposing ``.landmark`` like ``YoloPoseLandmarks``.

    Accepts an indexed list of 17 :class:`FakeLandmark` instances in the
    standard COCO keypoint order used by the YOLO pose adapter.  Detectors
    call ``landmarks.landmark[i]`` or pass the wrapper to the shared
    ``get_keypoint`` utility, both of which work against this shape.
    """

    def __init__(self, landmark_list: List[FakeLandmark]):
        self.landmark = landmark_list


def build_alert_pose(
    *,
    nose_y: float = 0.30,
    shoulder_y: float = 0.45,
    hip_y: float = 0.75,
    wrist_y: float = 0.60,
    visibility: float = 0.95,
) -> FakePoseLandmarks:
    """Construct a well-formed upright-pilot pose used as the baseline.

    Landmark layout (COCO17):
        0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
        9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles.

    All keypoints get identical visibility.  X coordinates are chosen so
    left/right pairs straddle the frame midline — detectors that compute
    shoulder midpoints, torso height, etc. then work normally.
    """
    eye_y = nose_y + 0.005
    ear_y = nose_y + 0.01
    elbow_y = (shoulder_y + wrist_y) / 2.0
    knee_y = (hip_y + 0.92) / 2.0
    return FakePoseLandmarks([
        FakeLandmark(x=0.50, y=nose_y, visibility=visibility),       # 0 nose
        FakeLandmark(x=0.47, y=eye_y, visibility=visibility),        # 1 left_eye
        FakeLandmark(x=0.53, y=eye_y, visibility=visibility),        # 2 right_eye
        FakeLandmark(x=0.45, y=ear_y, visibility=visibility),        # 3 left_ear
        FakeLandmark(x=0.55, y=ear_y, visibility=visibility),        # 4 right_ear
        FakeLandmark(x=0.40, y=shoulder_y, visibility=visibility),   # 5 left_shoulder
        FakeLandmark(x=0.60, y=shoulder_y, visibility=visibility),   # 6 right_shoulder
        FakeLandmark(x=0.38, y=elbow_y, visibility=visibility),      # 7 left_elbow
        FakeLandmark(x=0.62, y=elbow_y, visibility=visibility),      # 8 right_elbow
        FakeLandmark(x=0.36, y=wrist_y, visibility=visibility),      # 9 left_wrist
        FakeLandmark(x=0.64, y=wrist_y, visibility=visibility),      # 10 right_wrist
        FakeLandmark(x=0.44, y=hip_y, visibility=visibility),        # 11 left_hip
        FakeLandmark(x=0.56, y=hip_y, visibility=visibility),        # 12 right_hip
        FakeLandmark(x=0.43, y=knee_y, visibility=visibility),       # 13 left_knee
        FakeLandmark(x=0.57, y=knee_y, visibility=visibility),       # 14 right_knee
        FakeLandmark(x=0.42, y=0.95, visibility=visibility),         # 15 left_ankle
        FakeLandmark(x=0.58, y=0.95, visibility=visibility),         # 16 right_ankle
    ])


@pytest.fixture
def stub_yolo_keypoints():
    """Return a factory that builds alert/drowsy/head-up poses on demand.

    Example usage::

        def test_something(stub_yolo_keypoints):
            alert = stub_yolo_keypoints()               # baseline
            drowsy = stub_yolo_keypoints(nose_y=0.55)   # head dropped

    Every invocation returns a fresh :class:`FakePoseLandmarks` instance so
    detectors cannot accidentally mutate shared state.
    """
    def _make(**overrides) -> FakePoseLandmarks:
        return build_alert_pose(**overrides)

    return _make


@pytest.fixture
def alert_pose() -> FakePoseLandmarks:
    """Return a single pristine alert-posture pose — convenience wrapper."""
    return build_alert_pose()
