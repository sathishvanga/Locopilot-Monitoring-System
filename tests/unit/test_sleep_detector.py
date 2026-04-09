"""Unit tests for ``app.core.detectors.sleep_detector.SleepDetector``.

These tests exercise the detector in total isolation from the rest of the
monitoring pipeline.  They do not load YOLO weights, do not call MediaPipe,
and do not touch the filesystem beyond OpenCV's bundled Haar cascades (which
ship with the ``opencv-python`` wheel).

Regression coverage:
- Isolated construction still works without a parent monitor.
- Baseline calibration advances to ``baseline_calibrating=False`` once the
  sample and duration thresholds are met.
- ``cleanup_stale_tracking`` removes entries for persons no longer active.
- Feeding a pose where the nose clearly moved **up** (not down) must not
  produce a sleep/microsleep classification — regression for the 2026
  head-tilt wrap-around fix described in CLAUDE.md.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cv2", reason="SleepDetector requires OpenCV (cv2) for Haar cascade loading")
pytest.importorskip("numpy", reason="SleepDetector requires numpy")

from app.core.detectors.sleep_detector import SleepDetector  # noqa: E402
from tests.conftest import build_alert_pose  # noqa: E402


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_sleep_detector_constructs_without_monitor(minimal_settings, stub_logger):
    """SleepDetector must be constructible from (settings, logger) alone.

    Guards against accidental coupling to ``LocopilotActivityMonitor``.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)

    assert detector.settings is minimal_settings
    assert detector.logger is stub_logger
    # Tracking dicts start empty and lazily populate per person.
    assert detector.per_person_tracking == {}
    assert detector.ir_forward_lean_tracking == {}
    # Threshold init pulled values from settings (or defaults).
    assert isinstance(detector.SLEEP_BASELINE_MIN_SAMPLES, int)
    assert detector.SLEEP_BASELINE_MIN_SAMPLES >= 1
    assert detector.SLEEP_BASELINE_ENABLED is True


def test_sleep_detector_validate_pose_landmarks_rejects_invalid(
    minimal_settings, stub_logger
):
    """``validate_pose_landmarks`` should reject None and out-of-range coords.

    Sanity guard for downstream code that assumes landmarks are either
    ``None`` or a valid normalized pose.  Catches regressions where we
    forget to filter garbage pose outputs before running expensive
    per-frame analysis.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)

    # None is rejected.
    assert detector.validate_pose_landmarks(None) is False

    # A well-formed pose passes.
    good = build_alert_pose()
    assert detector.validate_pose_landmarks(good) is True

    # A pose with too few landmarks (below MIN_POSE_LANDMARKS) is rejected.
    from tests.conftest import FakeLandmark, FakePoseLandmarks
    stubby = FakePoseLandmarks([
        FakeLandmark(x=0.5, y=0.5, visibility=0.9) for _ in range(5)
    ])
    assert detector.validate_pose_landmarks(stubby) is False


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------

def test_sleep_detector_baseline_calibration_reaches_calibrated(
    minimal_settings, stub_logger
):
    """Feed enough well-formed pose frames to complete baseline calibration.

    The detector starts with ``baseline_calibrating=True`` and flips to
    ``False`` once both ``SLEEP_BASELINE_MIN_SAMPLES`` samples have been
    accumulated AND ``SLEEP_BASELINE_CALIBRATION_WINDOW`` seconds have
    elapsed since the first sample.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)
    person_idx = 0
    frame_shape = (720, 1280, 3)

    # We need to span at least the calibration window so the "elapsed" gate
    # passes.  Provide several extra samples for safety margin.
    num_samples = max(detector.SLEEP_BASELINE_MIN_SAMPLES + 2, 5)
    window = detector.SLEEP_BASELINE_CALIBRATION_WINDOW
    # Spread timestamps so the last sample is strictly past the window.
    dt = (window + 1.0) / max(num_samples - 1, 1)

    for i in range(num_samples):
        landmarks = build_alert_pose()  # fresh, no shared state
        timestamp_sec = i * dt
        detector.detect_pose_based_sleep(
            landmarks=landmarks,
            timestamp_sec=timestamp_sec,
            person_idx=person_idx,
            frame_shape=frame_shape,
        )

    tracking = detector.per_person_tracking[person_idx]
    assert tracking["baseline_calibrating"] is False, (
        "Baseline calibration should have completed after "
        f"{num_samples} samples spanning {window + 1.0:.1f}s"
    )
    assert tracking["baseline"] is not None, (
        "Calibration completion must populate the baseline dict"
    )
    # Baseline medians must exist for the core signals used downstream.
    for key in ("nose_below_px", "head_tilt", "torso_height_px"):
        assert key in tracking["baseline"]


# ---------------------------------------------------------------------------
# Stale person tracking cleanup
# ---------------------------------------------------------------------------

def test_sleep_detector_cleanup_stale_tracking_removes_inactive(
    minimal_settings, stub_logger
):
    """Stale person entries should be removed by ``cleanup_stale_tracking``.

    Simulates three persons {0, 1, 2}, then declares only person 0 as
    active.  Persons 1 and 2 must be purged from both tracking dicts.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)

    # Seed three persons in per_person_tracking.
    for pidx in (0, 1, 2):
        detector._get_per_person_sleep_tracking(pidx)

    # Seed IR forward-lean tracking as well to exercise both branches.
    for pidx in (0, 1, 2):
        detector._get_ir_forward_lean_tracking(pidx)

    assert set(detector.per_person_tracking.keys()) == {0, 1, 2}
    assert set(detector.ir_forward_lean_tracking.keys()) == {0, 1, 2}

    detector.cleanup_stale_tracking({0})

    assert set(detector.per_person_tracking.keys()) == {0}, (
        "cleanup_stale_tracking must remove per_person entries "
        "whose indices are not in the active set"
    )
    assert set(detector.ir_forward_lean_tracking.keys()) == {0}, (
        "cleanup_stale_tracking must also purge ir_forward_lean entries"
    )


# ---------------------------------------------------------------------------
# Head-tilt wrap-around regression
# ---------------------------------------------------------------------------

def test_sleep_detector_nose_y_drop_guard_blocks_false_head_tilt(
    minimal_settings, stub_logger
):
    """Nose moving *up* must never trigger sleep, even with noisy head_tilt.

    Regression for the 2026 ``atan2`` wrap-around fix documented in
    ``CLAUDE.md``: a distant LP's pose estimation can produce a head tilt
    that jumps 300+ degrees between frames.  If we only look at head tilt
    deltas we get a false head-drop signal.  The fix is to gate on the
    actual vertical nose motion (``nose_y_drop``) — if the nose moved **up**
    (``nose_y`` went DOWN — smaller y means higher in frame), no sleep
    should be reported.

    The test establishes a baseline with the nose at y=0.45, then feeds a
    frame where the nose has moved UP to y=0.30.  A correctly-guarded
    detector must return ``(is_sleeping=False, is_microsleeping=False)``
    regardless of any head-tilt artifacts.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)
    person_idx = 7
    frame_shape = (720, 1280, 3)

    # --- Phase 1: calibrate with nose relatively low ---
    num_samples = max(detector.SLEEP_BASELINE_MIN_SAMPLES + 3, 6)
    window = detector.SLEEP_BASELINE_CALIBRATION_WINDOW
    dt = (window + 1.0) / max(num_samples - 1, 1)

    baseline_nose_y = 0.45
    for i in range(num_samples):
        landmarks = build_alert_pose(nose_y=baseline_nose_y)
        detector.detect_pose_based_sleep(
            landmarks=landmarks,
            timestamp_sec=i * dt,
            person_idx=person_idx,
            frame_shape=frame_shape,
        )

    assert detector.per_person_tracking[person_idx]["baseline_calibrating"] is False

    # --- Phase 2: nose moves UP (smaller y == higher in frame) ---
    post_baseline_t = num_samples * dt
    risen_pose = build_alert_pose(nose_y=0.30)  # clearly above baseline

    is_sleeping, is_microsleeping, debug_info = detector.detect_pose_based_sleep(
        landmarks=risen_pose,
        timestamp_sec=post_baseline_t + 1.0,
        person_idx=person_idx,
        frame_shape=frame_shape,
    )

    assert is_sleeping is False, (
        "Nose moved UP relative to baseline — must not classify as sleeping. "
        f"debug_info={debug_info!r}"
    )
    assert is_microsleeping is False, (
        "Nose moved UP relative to baseline — must not classify as microsleep. "
        f"debug_info={debug_info!r}"
    )
    # Debug info should reflect that the nose y-drop is non-positive.
    if "nose_y_drop" in debug_info:
        assert debug_info["nose_y_drop"] <= 0.0, (
            "nose_y_drop must be <= 0 when the nose has risen above baseline "
            f"(got {debug_info['nose_y_drop']!r})"
        )
