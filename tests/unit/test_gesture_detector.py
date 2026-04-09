"""Unit tests for ``app.core.detectors.gesture_detector.GestureDetector``.

These tests focus on the pieces of :class:`GestureDetector` that run without
any YOLO / MediaPipe dependency:

- Isolated construction.
- LP/ALP coordination window bookkeeping in
  :meth:`GestureDetector.check_gesture_coordination`.
- Full state reset via :meth:`GestureDetector.reset`.

The full ``detect_raised_hand`` path is intentionally out-of-scope here — it
needs realistic pose landmarks and is covered indirectly by the integration
snapshot stub.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy", reason="GestureDetector requires numpy")

from app.core.detectors.gesture_detector import GestureDetector  # noqa: E402


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_gesture_detector_constructs_without_monitor(minimal_settings):
    """GestureDetector must be constructible from settings alone.

    Also verifies that the default session / coordination windows are
    plausible positive floats (the rule engine breaks if either is zero).
    """
    detector = GestureDetector(settings=minimal_settings)

    assert detector.settings is minimal_settings
    assert detector.session_timeout > 0
    assert detector.coordination_window > 0
    assert detector.gesture_sessions == {}
    assert detector.recent_person_activities == {}
    assert detector.hand_position_history == {}


# ---------------------------------------------------------------------------
# Coordination window bookkeeping
# ---------------------------------------------------------------------------

def test_check_gesture_coordination_within_window(minimal_settings):
    """LP + ALP previous raises within the coordination window → no violation.

    Seed BOTH LP (t=0) and ALP (t=1) via explicit ``update_session`` calls,
    then trigger a coordination check at t=3.0 where only ALP raises.
    Both prior raises are still within the 5 s coordination window (3s and
    2s ago respectively), so neither role should be flagged.
    """
    detector = GestureDetector(
        settings=minimal_settings,
        session_timeout=10.0,
        coordination_window=5.0,
    )

    # Seed both previous raises so the pre-update snapshot inside
    # check_gesture_coordination has a valid ALP last_raise_time.
    detector.update_session("LP", gesture_detected=True, timestamp=0.0)
    detector.update_session("ALP", gesture_detected=True, timestamp=1.0)
    assert detector.gesture_sessions["LP"]["last_raise_time"] == 0.0
    assert detector.gesture_sessions["ALP"]["last_raise_time"] == 1.0

    # ALP raises again at t=3.0 — well within the 5s window for both roles.
    lp_violation, alp_violation, session_info = detector.check_gesture_coordination(
        lp_gesture=False,
        alp_gesture=True,
        timestamp=3.0,
    )

    assert lp_violation is False, "LP's prior raise at t=0 is within the 5s window"
    assert alp_violation is False, "Only ALP raised in this call — no ALP violation"
    assert session_info["both_coordinated"] is True
    assert session_info["coordination_window"] == 5.0
    # ALP's last_raise_time should have advanced to the new call.
    assert detector.gesture_sessions["ALP"]["last_raise_time"] == 3.0


def test_check_gesture_coordination_outside_window(minimal_settings):
    """ALP raises after LP's raise has expired → LP violation.

    Seed LP at t=0, then ALP raises at t=6.0 with a 5 s coordination
    window.  LP is no longer recent, so the detector must flag
    ``lp_violation=True`` while leaving ``alp_violation=False``.  The 6s
    gap is also below the 10 s session_timeout, so LP's session row is
    still present (we're strictly testing the *coordination* window, not
    session expiry).
    """
    detector = GestureDetector(
        settings=minimal_settings,
        session_timeout=10.0,
        coordination_window=5.0,
    )

    detector.update_session("LP", gesture_detected=True, timestamp=0.0)

    lp_violation, alp_violation, session_info = detector.check_gesture_coordination(
        lp_gesture=False,
        alp_gesture=True,
        timestamp=6.0,
    )

    assert lp_violation is True, (
        "LP's last raise (t=0) is 6s ago, outside the 5s coordination window"
    )
    assert alp_violation is False
    assert session_info["both_coordinated"] is False
    # ALP's raise should have been recorded even though coordination failed.
    assert detector.gesture_sessions["ALP"]["last_raise_time"] == 6.0


def test_check_gesture_coordination_simultaneous_raises_no_violation(
    minimal_settings,
):
    """LP and ALP raising hands in the same frame → no violation.

    Explicit guard in the source code handles the race where pre-update
    timestamps would otherwise falsely flag a violation.
    """
    detector = GestureDetector(
        settings=minimal_settings,
        session_timeout=10.0,
        coordination_window=5.0,
    )

    lp_violation, alp_violation, session_info = detector.check_gesture_coordination(
        lp_gesture=True,
        alp_gesture=True,
        timestamp=1.0,
    )

    assert lp_violation is False
    assert alp_violation is False
    assert detector.gesture_sessions["LP"]["last_raise_time"] == 1.0
    assert detector.gesture_sessions["ALP"]["last_raise_time"] == 1.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_gesture_detector_reset(minimal_settings):
    """``reset()`` must clear every tracked dict managed by the detector."""
    detector = GestureDetector(settings=minimal_settings)

    detector.update_session("LP", gesture_detected=True, timestamp=0.0)
    detector.update_session("ALP", gesture_detected=True, timestamp=1.0)
    detector.recent_person_activities[0] = {"writing": 0.5}
    detector.hand_position_history[0] = {"placeholder": True}

    assert detector.gesture_sessions  # non-empty precondition
    assert detector.recent_person_activities
    assert detector.hand_position_history

    detector.reset()

    assert detector.gesture_sessions == {}
    assert detector.recent_person_activities == {}
    assert detector.hand_position_history == {}
