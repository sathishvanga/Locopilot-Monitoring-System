"""Shared pytest fixtures and helpers for the Locopilot test suite.

The 0001 and 0006 review-fix agents wrote tests that import three names
from ``tests.conftest`` — ``minimal_settings``, ``stub_logger``, and the
helper ``build_alert_pose``. This file supplies them so those tests
can run without a live config / log dir / pose model.

Path-checking and disk side-effects are stubbed out so collection is
hermetic on a developer machine: ``LOCOPILOT_SKIP_PATH_CHECKS=1`` and
``LOG_DIR=/tmp/locopilot_test_logs`` are set BEFORE any ``app.*`` import.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Hermetic environment — must run BEFORE importing anything from ``app``.
# ---------------------------------------------------------------------------
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_DIR", "/tmp/locopilot_test_logs")
Path(os.environ["LOG_DIR"]).mkdir(parents=True, exist_ok=True)

# Ensure the repo root is on sys.path when pytest is invoked from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Settings & logger fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_settings():
    """A default-constructed Settings instance, safe for unit tests.

    Detectors call ``getattr(settings, '...', default)`` for every
    threshold, so the project's normal Settings object is sufficient
    — no extra stubbing required.
    """
    from app.utils.config import get_settings
    return get_settings()


@pytest.fixture
def stub_logger():
    """A no-op logger that swallows every record.

    Used by detectors that take ``logger=`` in their constructor. The
    project logger writes to disk; we don't want test runs creating
    rotation-handle files in /tmp on every invocation.
    """
    logger = logging.getLogger("locopilot.tests.stub")
    logger.setLevel(logging.CRITICAL + 1)  # silence everything
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Synthetic pose builder
# ---------------------------------------------------------------------------

# YOLO 17-keypoint COCO layout — index → name (canonical):
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
#   13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle


class _Landmark:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __repr__(self) -> str:  # for test diagnostics
        return f"_Landmark(x={self.x:.3f}, y={self.y:.3f}, vis={self.visibility:.2f})"


class _PoseLandmarks:
    """MediaPipe-compatible container — exposes ``.landmark[i]``."""

    __slots__ = ("landmark",)

    def __init__(self, landmarks: List[_Landmark]):
        self.landmark = landmarks


def build_alert_pose(
    nose_y: float = 0.30,
    shoulder_y: float = 0.45,
    visibility: float = 1.0,
) -> _PoseLandmarks:
    """Synthetic 17-keypoint alert pose for sleep / gesture / writing tests.

    All x coordinates fall in [0, 1] (normalized YOLO/MediaPipe space).
    Defaults place the nose above the shoulder line so detectors see an
    "upright, alert" pose. Tests override ``nose_y`` and ``shoulder_y``
    to drive specific head-tilt / posture math (see e.g.
    ``tests/detectors/test_sleep_detector.py`` for the wrap-around
    regression test).
    """
    # Heuristic geometry — close to a real overhead-cabin alert pose,
    # without being so symmetric that detectors short-circuit on
    # degenerate landmarks.
    layout = [
        (0.50, nose_y),         # 0  nose
        (0.48, nose_y - 0.02),  # 1  left_eye
        (0.52, nose_y - 0.02),  # 2  right_eye
        (0.46, nose_y - 0.01),  # 3  left_ear
        (0.54, nose_y - 0.01),  # 4  right_ear
        (0.42, shoulder_y),     # 5  left_shoulder
        (0.58, shoulder_y),     # 6  right_shoulder
        (0.40, shoulder_y + 0.10),  # 7  left_elbow
        (0.60, shoulder_y + 0.10),  # 8  right_elbow
        (0.42, shoulder_y + 0.18),  # 9  left_wrist
        (0.58, shoulder_y + 0.18),  # 10 right_wrist
        (0.45, shoulder_y + 0.25),  # 11 left_hip
        (0.55, shoulder_y + 0.25),  # 12 right_hip
        (0.45, shoulder_y + 0.40),  # 13 left_knee
        (0.55, shoulder_y + 0.40),  # 14 right_knee
        (0.45, shoulder_y + 0.55),  # 15 left_ankle
        (0.55, shoulder_y + 0.55),  # 16 right_ankle
    ]
    return _PoseLandmarks([
        _Landmark(x=x, y=y, visibility=visibility) for (x, y) in layout
    ])


# ---------------------------------------------------------------------------
# stub_yolo_keypoints — factory fixture for post-reset frame tests
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_yolo_keypoints():
    """Factory that returns a synthetic 17-keypoint pose for detector tests.

    Same underlying geometry as ``build_alert_pose`` but exposed as a
    callable so tests can vary the head/shoulder positions per call:

        pose_a = stub_yolo_keypoints()           # default alert pose
        pose_b = stub_yolo_keypoints(nose_y=0.55)  # head-down pose
    """
    def _build(nose_y: float = 0.30, shoulder_y: float = 0.45,
               visibility: float = 1.0) -> _PoseLandmarks:
        return build_alert_pose(
            nose_y=nose_y, shoulder_y=shoulder_y, visibility=visibility
        )
    return _build


# Re-export under the names the tests import.
__all__ = ["minimal_settings", "stub_logger", "build_alert_pose",
           "stub_yolo_keypoints"]
