"""Detector-level regression tests for ``SleepDetector`` head-tilt math.

Spec source: ``docs/specs/code-review-fixes/tasks/0001-restore-determinism-contract.md``.

The bug:
    ``calculate_head_tilt_angle`` returns the result of
    ``arctan2(dy, dx) * 180/pi - 90``.  Without normalizing back to
    [-180, 180], a head whose tilt sweeps across the +/-180 seam (which
    happens routinely on small / distant LP poses where pose-estimation
    noise dominates) produces deltas like -340 deg in one frame —
    far above the 30 deg ``head_tilt_drop_threshold`` and tripping a
    false sleep "head drop".  The same wrap is needed on the per-frame
    delta accumulator and on the baseline-relative ``head_tilt_drop``
    in ``detect_pose_based_sleep``.

These tests live under ``tests/detectors/`` per the spec's acceptance
criteria.  They monkeypatch ``calculate_head_tilt_angle`` so the test
controls the angle sequence directly — building landmarks that produce
a specific sub-degree tilt is brittle and unrelated to the math being
verified.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "cv2", reason="SleepDetector requires OpenCV for Haar cascade loading"
)
pytest.importorskip("numpy", reason="SleepDetector requires numpy")

from app.core.detectors.sleep_detector import SleepDetector  # noqa: E402
from tests.conftest import build_alert_pose  # noqa: E402


def _seed_baseline(
    detector: SleepDetector,
    person_idx: int,
    *,
    baseline_head_tilt: float,
    baseline_nose_y: float = 0.45,
) -> None:
    """Pre-populate a calibrated baseline so detection runs the post-calibration path.

    The detector's normal calibration takes ``SLEEP_BASELINE_MIN_SAMPLES``
    poses spread across ``SLEEP_BASELINE_CALIBRATION_WINDOW`` seconds.  For
    a focused unit test it's clearer to install a synthetic baseline and
    drive the head-drop branch directly.
    """
    tracking = detector._get_per_person_sleep_tracking(person_idx)
    tracking["baseline_calibrating"] = False
    tracking["baseline"] = {
        "head_tilt": baseline_head_tilt,
        "nose_y_normalized": baseline_nose_y,
        "nose_below_px": 0.0,
        "torso_height_px": 200.0,
        "shoulder_width_px": 120.0,
        "nose_above_shoulders_norm": 0.05,
        "movement": 0.0,
        "eye_vis": 0.9,
    }


def test_head_tilt_wrap_does_not_fire_false_head_drop(
    minimal_settings, stub_logger, monkeypatch
):
    """A tilt walk from +170 to -170 across 5 frames must not signal a head drop.

    Why this matters
    ----------------
    Without the ``(angle + 180) % 360 - 180`` wrap on both the
    delta-since-last-frame and the delta-vs-baseline, the +170 -> -170
    transition produces a -340 degree apparent drop — well past the 30 deg
    ``sleep_head_tilt_drop_threshold``.  CLAUDE.md ("Head tilt angle
    wrapping") catalogs the FP this caused on small LP bboxes.

    Test design
    -----------
    1. Construct ``SleepDetector`` with default settings.
    2. Seed a calibrated baseline at +170 deg head tilt — i.e. the
       baseline lives right next to the +/-180 wrap seam.
    3. Monkey-patch ``calculate_head_tilt_angle`` to return scripted
       values [+170, +175, +180, -175, -170], one per frame.  Each
       adjacent pair has a real motion of just 5 deg (the short way
       around through the seam) but a naive subtraction yields
       -355 / +5 / -355 / +5 deltas.  Likewise the
       baseline-relative drop unwrapped is 0 / +5 / +10 / -345 / -340 —
       the last two would trip the 30 deg threshold without the wrap.
    4. Pass a non-drowsy alert pose so all *other* sleep signals stay
       quiet.  The test isolates the head-tilt math.
    5. Run the 5 frames through ``detect_pose_based_sleep``.  On every
       frame, ``head_drop_detected`` and ``head_drop_from_delta`` must
       both be False, and the wrapped ``head_tilt_drop`` must stay
       within (-180, 180] and below the 30 deg threshold.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)
    person_idx = 3
    frame_shape = (720, 1280, 3)

    # Baseline sits at +170 deg — right next to the wrap seam.
    _seed_baseline(detector, person_idx, baseline_head_tilt=170.0)

    # Scripted tilt walk: [+170, +175, +180, -175, -170].
    # Real motion: 5 deg per step the SHORT way through the +/-180 seam.
    # Naive baseline-relative drops: 0 / +5 / +10 / -345 / -340.
    # The last two would falsely trip the +30 deg head-tilt-drop threshold
    # without the wrap. After wrap they collapse to +10 / +15 / +20 — all
    # below threshold, all consistent with the real 20 deg total motion
    # the LP underwent.
    scripted_tilts = [170.0, 175.0, 180.0, -175.0, -170.0]
    scripted_iter = iter(scripted_tilts)

    def _scripted_head_tilt(_landmark_list):
        return next(scripted_iter)

    monkeypatch.setattr(
        detector, "calculate_head_tilt_angle", _scripted_head_tilt
    )

    # The pose itself is alert and unchanged frame-to-frame so non-tilt
    # signals (eye visibility, movement, torso height) stay neutral.
    base_pose = build_alert_pose()

    # Timestamps past the baseline calibration window so the post-calibration
    # branch (which is where the head-tilt-drop math lives) is exercised.
    t0 = detector.SLEEP_BASELINE_CALIBRATION_WINDOW + 10.0

    for frame_idx, expected_tilt in enumerate(scripted_tilts):
        timestamp_sec = t0 + frame_idx * 1.0
        is_sleeping, is_microsleeping, debug = detector.detect_pose_based_sleep(
            landmarks=base_pose,
            timestamp_sec=timestamp_sec,
            person_idx=person_idx,
            frame_shape=frame_shape,
        )

        # Early frames may bail out at the "building_history" gate — that
        # is itself a guarantee head_drop_detected stayed False, so the
        # rest of the assertions only apply once the deeper code path
        # actually computed those signals.
        if debug.get("status") == "building_history":
            assert is_sleeping is False
            assert is_microsleeping is False
            continue

        assert debug.get("head_tilt") == pytest.approx(expected_tilt), (
            f"Frame {frame_idx}: scripted head_tilt was {expected_tilt} "
            f"but debug returned {debug.get('head_tilt')!r}"
        )

        # Wrapped delta-vs-baseline must stay in (-180, 180].
        if "head_tilt_drop" in debug:
            head_tilt_drop = debug["head_tilt_drop"]
            assert -180.0 < head_tilt_drop <= 180.0, (
                f"Frame {frame_idx}: head_tilt_drop {head_tilt_drop!r} "
                "escaped the [-180, 180] range — wrap not applied. The "
                "spec-0001 fix normalizes "
                "(head_tilt - baseline + 180) % 360 - 180."
            )

        # The headline assertion: no false head-drop signal anywhere along
        # the wrap walk.  If this fires on a late frame (e.g. -175 vs the
        # +170 baseline) the regression has reappeared.
        assert debug.get("head_drop_detected", False) is False, (
            f"Frame {frame_idx} (tilt={expected_tilt}): head_drop_detected "
            f"flipped True. Wrap-around fix lost? debug={debug!r}"
        )
        assert debug.get("head_drop_from_delta", False) is False, (
            f"Frame {frame_idx} (tilt={expected_tilt}): "
            "head_drop_from_delta flipped True. The per-frame delta wrap "
            f"is missing. debug={debug!r}"
        )

        # Belt-and-suspenders: confirm sleep classification stays clean.
        assert is_sleeping is False, (
            f"Frame {frame_idx}: is_sleeping fired off the wrap-around "
            "delta alone, with no other supporting signals."
        )
        assert is_microsleeping is False, (
            f"Frame {frame_idx}: is_microsleeping fired off the wrap-around "
            "delta alone, with no other supporting signals."
        )


def test_calculate_head_tilt_angle_normalizes_to_pm_180(
    minimal_settings, stub_logger
):
    """``calculate_head_tilt_angle`` must always return a value in [-180, 180].

    The post-fix function applies ``(angle + 180) % 360 - 180`` so any
    geometry — including the singular case where the nose sits directly
    above the neck (``delta_x ~= 0, delta_y < 0``) and would otherwise
    return a value just past the +/-180 seam — falls inside the canonical
    range.  Downstream delta math depends on this invariant.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)

    # build_alert_pose places nose at (0.50, 0.30) and shoulders at
    # (0.40, 0.45) / (0.60, 0.45) — neck midpoint (0.50, 0.45).  delta_x
    # is 0, delta_y is -0.15: arctan2 = -90 deg; pre-90-shift the angle
    # is -180 (or +180) — exactly on the seam.  Wrap must canonicalize.
    pose = build_alert_pose()
    angle = detector.calculate_head_tilt_angle(pose.landmark)

    assert angle is not None, "Well-formed pose must yield an angle"
    assert -180.0 <= angle <= 180.0, (
        f"calculate_head_tilt_angle returned {angle!r} — outside the "
        "[-180, 180] canonical range. Wrap-around fix missing."
    )


def test_nose_y_drop_negative_does_not_trigger_head_drop(
    minimal_settings, stub_logger, monkeypatch
):
    """Negative ``nose_y_drop`` (nose moved UP) must never count as a head drop.

    Without the ``nose_y_drop >= 0`` guard, if the head-tilt drop branch
    happens to misfire (e.g. because of a brief wrap glitch that the
    delta-history doesn't catch), the detector could still arrive at
    ``head_drop_detected = True`` even though the nose physically rose
    above baseline.  The guard short-circuits that branch.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)
    person_idx = 11
    frame_shape = (720, 1280, 3)

    # Baseline nose_y at 0.55, baseline tilt at 0.  Then we'll feed a
    # frame where the nose moved UP (smaller y == higher in frame).
    _seed_baseline(
        detector, person_idx,
        baseline_head_tilt=0.0,
        baseline_nose_y=0.55,
    )

    # Force ``calculate_head_tilt_angle`` to be a no-op so only the
    # nose-y branch can possibly trip head_drop.
    monkeypatch.setattr(
        detector, "calculate_head_tilt_angle", lambda _ll: 0.0
    )

    risen_pose = build_alert_pose(nose_y=0.30)  # 0.25 above baseline

    is_sleeping, is_microsleeping, debug = detector.detect_pose_based_sleep(
        landmarks=risen_pose,
        timestamp_sec=detector.SLEEP_BASELINE_CALIBRATION_WINDOW + 5.0,
        person_idx=person_idx,
        frame_shape=frame_shape,
    )

    assert debug.get("nose_y_drop", 0.0) <= 0.0, (
        "Nose moved UP — nose_y_drop must be non-positive, got "
        f"{debug.get('nose_y_drop')!r}"
    )
    assert debug.get("head_drop_detected") is False, (
        "Negative nose_y_drop must NOT trigger head_drop_detected. "
        f"debug={debug!r}"
    )
    assert is_sleeping is False
    assert is_microsleeping is False


def test_negative_nose_y_drop_short_circuits_head_tilt_branch(
    minimal_settings, stub_logger
):
    """The ``nose_y_drop < 0`` guard must skip the ``head_tilt_drop`` branch too.

    Why this test is shaped this way
    --------------------------------
    The first guard test (``test_nose_y_drop_negative_does_not_trigger_head_drop``)
    monkey-patches ``calculate_head_tilt_angle`` to return a constant 0,
    which keeps the head-tilt branch quiescent — and thereby HIDES whether
    the guard actually short-circuits that branch when the tilt math would
    naturally have fired it.  This test removes the patch and instead
    constructs a real pose whose tilt is naturally far enough above the
    seeded baseline_head_tilt to make ``head_tilt_drop > 30`` deg.  If
    head_drop_detected still flips True, the guard is only covering the
    nose-y branch and the head-tilt branch can still fabricate sleep
    signals when the nose physically moved upward — exactly NIT-1's gap.

    Construction
    ------------
    With ``build_alert_pose(nose_y=0.30, shoulder_y=0.45)``:
        nose at (0.50, 0.30), neck midpoint at (0.50, 0.45)
        delta_y = -0.15, delta_x = 0  ->  arctan2 = -90 deg
        angle = -90 - 90 = -180 -> wraps to -180

    With ``build_alert_pose(nose_y=0.30)`` and shifted x landmarks:
    we need a pose whose head_tilt is above ``baseline_head_tilt + 30``
    after wrapping.  We choose baseline_head_tilt = -180 (right at the
    seam) and feed the standard alert pose for a head_tilt of -180.
    That alone is not enough — head_tilt_drop = 0 — so we instead
    use a slightly-rotated pose: nose pulled to x=0.65 keeps nose_y at
    0.30 (still above baseline_nose_y = 0.45) but yields:
        delta_x = 0.15, delta_y = -0.15
        arctan2(-0.15, 0.15) = -45 deg
        angle = -45 - 90 = -135 deg  (in range, no wrap)
    With baseline_head_tilt = -180:
        head_tilt_drop = (-135 - (-180) + 180) % 360 - 180
                       = (225) % 360 - 180  =  225 - 180  =  45 deg
    That's > 30 deg head_tilt_drop_thresh so the head-tilt branch WOULD
    fire head_drop_detected = True absent the nose_y_drop guard.

    Because nose_y_drop = 0.30 - 0.45 = -0.15 < 0, the new guard must
    short-circuit ALL head-drop branches and head_drop_detected stays
    False.

    We do NOT monkey-patch ``calculate_head_tilt_angle`` — the angle
    math runs for real.
    """
    detector = SleepDetector(settings=minimal_settings, logger=stub_logger)
    person_idx = 17
    frame_shape = (720, 1280, 3)

    # Baseline: head_tilt at -180 (right at the +/-180 seam, where the
    # wrap matters), nose_y at 0.45 — the "neck level" for the alert pose.
    _seed_baseline(
        detector, person_idx,
        baseline_head_tilt=-180.0,
        baseline_nose_y=0.45,
    )

    # Build a pose whose nose is BOTH (a) higher in the frame than baseline
    # (smaller y) -> nose_y_drop < 0, and (b) shifted laterally so the
    # computed head_tilt sits ~45 deg above the -180 baseline -> would
    # trigger the head-tilt branch absent the guard.
    pose = build_alert_pose(nose_y=0.30)
    # Pull the nose laterally to get a non-vertical neck->nose vector.
    # Layout index 0 is the nose (see conftest.build_alert_pose).
    pose.landmark[0].x = 0.65  # delta_x = +0.15 vs neck midpoint 0.50

    # Sanity: the standalone tilt math itself produces ~ -135 deg here,
    # i.e. ~45 deg above the seeded baseline.  If the helper or the wrap
    # rule changes shape this assertion will catch it before the headline
    # check runs.
    natural_tilt = detector.calculate_head_tilt_angle(pose.landmark)
    assert natural_tilt is not None, "Pose well-formed, calculate must succeed"
    head_tilt_drop_naive = (natural_tilt - (-180.0) + 180.0) % 360.0 - 180.0
    assert head_tilt_drop_naive > 30.0, (
        f"Test setup invariant broken: expected head_tilt_drop > 30 deg "
        f"to actually exercise the head-tilt branch, got "
        f"{head_tilt_drop_naive:.2f} deg (natural_tilt={natural_tilt:.2f})."
    )

    # Seed real tilt history so the per-frame delta branch and avg-tilt
    # paths work against believable values rather than starting from
    # empty.  The values mirror an alert pilot whose head has been near
    # the -180 seam — i.e. matches the seeded baseline.  We do NOT seed
    # in a way that would trigger the head_tilt_deltas branch (no large
    # jumps); we want the head_tilt-vs-baseline branch to be the one
    # that the guard suppresses.
    tracking = detector._get_per_person_sleep_tracking(person_idx)
    for prior_tilt in (-180.0, -179.0, -180.0, -178.0):
        tracking["head_tilt_history"].append(prior_tilt)
    # Backfill movement_history so the building_history early-exit
    # doesn't prevent the head-drop computation from running.
    for _ in range(int(2 * detector.sample_fps) + 2):
        tracking["movement_history"].append(0.0)

    is_sleeping, is_microsleeping, debug = detector.detect_pose_based_sleep(
        landmarks=pose,
        timestamp_sec=detector.SLEEP_BASELINE_CALIBRATION_WINDOW + 5.0,
        person_idx=person_idx,
        frame_shape=frame_shape,
    )

    # The pose was constructed so the head-tilt branch would fire absent
    # the guard.  The guard must short-circuit it.
    assert debug.get("nose_y_drop", 0.0) < 0.0, (
        "Pose was constructed with nose ABOVE baseline; nose_y_drop must "
        f"be negative.  Got {debug.get('nose_y_drop')!r} — test setup "
        "broken or wrap rule changed."
    )
    assert debug.get("head_drop_detected") is False, (
        "Negative nose_y_drop must short-circuit ALL head-drop branches, "
        "including the head_tilt_drop branch.  This test exercises that "
        "branch with a real pose; if head_drop_detected flipped True the "
        "NIT-1 guard is only covering the nose-y branch.  "
        f"debug={debug!r}"
    )
    assert is_sleeping is False
    assert is_microsleeping is False
