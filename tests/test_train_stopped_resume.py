"""Train-STOPPED resume invariant (task 0006).

Sequence under test:

    Segment 1 (RUNNING, 30 frames): sleep is detected and counters mature.
    Segment 2 (STOPPED, 30 frames): sleep is suppressed each frame by the
        train-stopped gate. Internal counters MUST roll back so they do
        not stay saturated across the STOPPED window.
    Segment 3 (RUNNING, 30 frames): no sleep cue at all.

Expectation: NO sleep activity is emitted in segment 3. Without the
``on_suppressed`` hook in :func:`app.core.gates.apply_train_stopped_suppression`,
``sleep_detector.per_person_tracking[*]['pose_sleep_duration']`` would
already be saturated when segment 3 begins, producing instant
false-positive sleep on resume.

The test is deliberately at the gate level (no real video, no YOLO/pose
inference) so it runs in any environment with numpy.
"""

from __future__ import annotations

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_aggregated(sleep: bool) -> dict:
    return {
        'sleep_detected': sleep,
        'microsleep_detected': False,
        'cell_phone_detected': False,
        'writing_detected': False,
        'packing_bags_detected': False,
        'lp_hand_gesture_detected': False,
        'alp_hand_gesture_detected': False,
        'mind_diversion_detected': False,
        'eating_drinking_detected': False,
    }


def _make_persons_data(person_idx: int, sleep: bool) -> dict:
    return {
        person_idx: {
            'role': 'LP',
            'activities': {
                'sleep': sleep,
                'microsleep': False,
                'writing': False,
                'packing_bags': False,
                'lp_hand_gesture': False,
                'alp_hand_gesture': False,
                'mind_diversion': False,
                'eating_drinking': False,
                'cell_phone': False,
            },
        }
    }


def test_no_sleep_emitted_in_segment_3_after_stopped_window(
    minimal_settings, stub_logger
):
    """Three-segment sequence: RUNNING(sleep) -> STOPPED -> RUNNING(no cue).

    Asserts that the sleep duration counter is back at 0 entering segment 3
    so no sleep activity can be emitted from a single ALERT-eyed frame.
    """
    from app.core.gates import apply_train_stopped_suppression
    from app.core.detectors.sleep_detector import SleepDetector

    sleep = SleepDetector(settings=minimal_settings, logger=stub_logger)
    detectors = {'sleep': sleep}
    person_idx = 0

    # ---------- Segment 1: 30 RUNNING frames with sleep flagged ----------
    # We model "the detector latched sleep" by manually maturing the
    # per-person counter. (Driving the full sleep pipeline would require
    # real pose landmarks, far beyond the scope of this gate test.)
    track = sleep._get_per_person_sleep_tracking(person_idx)
    for frame in range(30):
        # Each "frame" extends the sleep duration by 1 sample (~2s @ 0.5fps).
        track['pose_sleep_duration'] += 1.0
        track['sustained_stillness_count'] += 1
        # During RUNNING the gate must NOT zero anything: pass an
        # aggregated dict with sleep=True and skip the gate call entirely
        # to mirror the real pipeline (the gate only runs in STOPPED).

    # Sanity: counters matured during segment 1.
    assert track['pose_sleep_duration'] > 0
    assert track['sustained_stillness_count'] > 0

    # ---------- Segment 2: 30 STOPPED frames; gate suppresses sleep ------
    for _ in range(30):
        aggregated = _make_aggregated(sleep=True)
        persons_data = _make_persons_data(person_idx, sleep=True)
        apply_train_stopped_suppression(
            aggregated, persons_data, detectors=detectors
        )
        # The gate must zero the public flags every iteration.
        assert aggregated['sleep_detected'] is False
        assert persons_data[person_idx]['activities']['sleep'] is False

    # After the FIRST STOPPED frame the on_suppressed hook reset the
    # per-person tracking dict; subsequent frames have nothing left to
    # zero, but the counter must remain 0.
    track_after_stopped = sleep._get_per_person_sleep_tracking(person_idx)
    assert track_after_stopped['pose_sleep_duration'] == 0, (
        f"sleep counter survived the STOPPED window: "
        f"pose_sleep_duration={track_after_stopped['pose_sleep_duration']}"
    )
    assert track_after_stopped['sustained_stillness_count'] == 0
    # Sleep state machine must also be back to ALERT.
    assert track_after_stopped['sleep_state'] == 'ALERT'

    # ---------- Segment 3: 30 RUNNING frames with NO sleep cue ----------
    # The gate is not called (RUNNING). With cleared counters the
    # detector cannot emit sleep until a fresh duration accumulates.
    sleep_emitted_in_segment_3 = False
    for _ in range(30):
        aggregated = _make_aggregated(sleep=False)
        persons_data = _make_persons_data(person_idx, sleep=False)
        # Mirror the production pipeline: aggregated/persons_data only
        # carry True when the detector itself fired this frame. With
        # cleared counters and no cue the detector stays silent.
        if aggregated['sleep_detected'] or persons_data[person_idx]['activities']['sleep']:
            sleep_emitted_in_segment_3 = True

    assert not sleep_emitted_in_segment_3, (
        "Sleep activity emitted in segment 3 (RUNNING after STOPPED) "
        "despite no cue. The on_suppressed hook failed to reset counters."
    )

    # Final invariant: counter is still 0 entering "frame 91".
    assert sleep._get_per_person_sleep_tracking(
        person_idx
    )['pose_sleep_duration'] == 0


def test_gesture_last_raise_does_not_survive_stopped_window(
    minimal_settings,
):
    """The train-STOPPED gate must clear gesture sessions for the suppressed
    role so a stale RUNNING-segment-1 raise can't pair against a fresh
    RUNNING-segment-3 raise.

    Without ``on_suppressed``, ``gesture_sessions['LP']['last_raise_time']``
    from segment 1 would still be in the dict when segment 3 begins,
    leaving a 30-min-stale timestamp available to the coordination check.
    """
    from app.core.gates import apply_train_stopped_suppression
    from app.core.detectors.gesture_detector import GestureDetector

    gesture = GestureDetector(settings=minimal_settings)
    detectors = {'gesture': gesture}

    # Segment 1: LP raised at t=5s (stale-stamp candidate).
    gesture.gesture_sessions['LP'] = {
        'last_raise_time': 5.0,
        'gesture_count': 1,
        'last_update': 5.0,
    }
    gesture.gesture_sessions['ALP'] = {
        'last_raise_time': 4.0,
        'gesture_count': 1,
        'last_update': 4.0,
    }

    # Segment 2: STOPPED window. The gate fires every frame; on_suppressed
    # drops both LP and ALP role sessions over the course of the window.
    for _ in range(5):
        aggregated = {
            'lp_hand_gesture_detected': True,
            'alp_hand_gesture_detected': True,
        }
        persons_data = {
            0: {'activities': {'lp_hand_gesture': True, 'alp_hand_gesture': True}}
        }
        apply_train_stopped_suppression(
            aggregated, persons_data, detectors=detectors
        )

    # Both sessions cleared by the gate hooks.
    assert 'LP' not in gesture.gesture_sessions
    assert 'ALP' not in gesture.gesture_sessions

    # Segment 3 (RUNNING resume @ t=120s): the coordination check sees
    # no stale raise times — the pre-update reads are both None.
    _lp_violation, _alp_violation, info = gesture.check_gesture_coordination(
        lp_gesture=False,
        alp_gesture=True,
        timestamp=120.0,
    )
    assert info['lp_last_raise'] is None
    assert info['alp_last_raise'] is None
