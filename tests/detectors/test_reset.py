"""Per-detector reset() contract tests (task 0006).

Each detector must expose a ``reset()`` method that wipes every per-video
state machine to the same shape as a freshly constructed instance. This
prevents state from one video bleeding into the next when detectors are
reused across video boundaries (worker pool, single-process loop, etc.).

These tests deliberately avoid running real frames through the detectors
— they manipulate the documented state attributes directly to confirm
``reset()`` clears them. Heavier integration coverage (frame-driven
behavior) lives in :mod:`tests.detectors.test_train_motion_first_frame`
and :mod:`tests.test_train_stopped_resume`.
"""

from __future__ import annotations

import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# TrainMotionDetector
# ---------------------------------------------------------------------------

def test_train_motion_detector_reset_matches_fresh_instance():
    from app.core.detectors.train_motion_detector import TrainMotionDetector

    fresh = TrainMotionDetector()
    used = TrainMotionDetector()

    # Drive used through a few synthetic frames so every state field is
    # populated. We bypass the full process_frame path and poke the
    # documented attributes directly to keep the test independent of the
    # vibration scoring logic.
    used.prev_gray = np.full((10, 10), 128, dtype=np.uint8)
    used.prev_gray_window = np.full((5, 5), 64, dtype=np.uint8)
    used.state_history.append(("RUNNING", 0.9))
    used.state_history.append(("STOPPED", 0.4))
    used._prev_block_vars = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    used.person_bbox_history_buf.append([[1, 2, 3, 4]])
    used._vib_history.append(2.5)
    used._vib_history.append(3.5)
    used._frames_seen = 7

    used.reset()

    # Both fresh and post-reset must hold None for prev_gray.
    assert fresh.prev_gray is None
    assert used.prev_gray is None
    assert used.prev_gray_window is None
    assert len(used.state_history) == len(fresh.state_history) == 0
    assert used._prev_block_vars is None
    assert len(used.person_bbox_history_buf) == 0
    assert len(used._vib_history) == 0
    assert used._frames_seen == fresh._frames_seen == 0


def test_train_motion_detector_reset_preserves_thresholds():
    """Reset must NOT touch configuration / thresholds."""
    from app.core.detectors.train_motion_detector import TrainMotionDetector

    det = TrainMotionDetector()
    det.prev_gray = np.zeros((4, 4), dtype=np.uint8)
    snapshot = {
        'vibration_threshold': det.vibration_threshold,
        'running_threshold': det.running_threshold,
        'temporal_window': det.temporal_window,
        'window_flow_threshold': det.window_flow_threshold,
        'weight_vibration': det.weight_vibration,
        'weight_window': det.weight_window,
        'weight_stability': det.weight_stability,
        'window_roi_x1': det.window_roi_x1,
        'window_roi_y2': det.window_roi_y2,
    }
    det.reset()
    for k, v in snapshot.items():
        assert getattr(det, k) == v, f"reset() must not change {k}"


# ---------------------------------------------------------------------------
# SleepDetector
# ---------------------------------------------------------------------------

def test_sleep_detector_reset_matches_fresh_instance(minimal_settings, stub_logger):
    from app.core.detectors.sleep_detector import SleepDetector

    fresh = SleepDetector(settings=minimal_settings, logger=stub_logger)
    used = SleepDetector(settings=minimal_settings, logger=stub_logger)

    # Trigger lazy creation of per-person tracking dicts.
    _ = used._get_per_person_sleep_tracking(0)
    _ = used._get_per_person_sleep_tracking(1)
    _ = used._get_ir_forward_lean_tracking(0)
    used.per_person_tracking[0]['pose_sleep_duration'] = 5.0
    used.ir_forward_lean_tracking[0]['sub_threshold_streak'] = 3

    assert len(used.per_person_tracking) >= 2
    assert len(used.ir_forward_lean_tracking) >= 1

    used.reset()

    assert len(used.per_person_tracking) == len(fresh.per_person_tracking) == 0
    assert len(used.ir_forward_lean_tracking) == len(fresh.ir_forward_lean_tracking) == 0


def test_sleep_detector_reset_is_alias_for_reset_tracking(minimal_settings, stub_logger):
    """reset() must wipe the same dicts as reset_tracking()."""
    from app.core.detectors.sleep_detector import SleepDetector

    a = SleepDetector(settings=minimal_settings, logger=stub_logger)
    b = SleepDetector(settings=minimal_settings, logger=stub_logger)

    for det in (a, b):
        _ = det._get_per_person_sleep_tracking(0)
        det.per_person_tracking[0]['pose_sleep_duration'] = 4.0

    a.reset()
    b.reset_tracking()

    assert dict(a.per_person_tracking) == dict(b.per_person_tracking)
    assert dict(a.ir_forward_lean_tracking) == dict(b.ir_forward_lean_tracking)


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

def test_gesture_detector_reset_matches_fresh_instance(minimal_settings):
    from collections import deque
    from app.core.detectors.gesture_detector import GestureDetector

    fresh = GestureDetector(settings=minimal_settings)
    used = GestureDetector(settings=minimal_settings)

    used.gesture_sessions['LP'] = {
        'last_raise_time': 12.5, 'gesture_count': 3, 'last_update': 13.0,
    }
    used.gesture_sessions['ALP'] = {'last_raise_time': 11.0}
    used.recent_person_activities[0] = {'writing': 5.0}
    used.hand_position_history[1] = {
        'right_wrist': deque([(100, 200, 1.0)], maxlen=10)
    }

    used.reset()

    assert dict(used.gesture_sessions) == dict(fresh.gesture_sessions) == {}
    assert dict(used.recent_person_activities) == dict(fresh.recent_person_activities) == {}
    assert dict(used.hand_position_history) == dict(fresh.hand_position_history) == {}


# ---------------------------------------------------------------------------
# MindDiversionDetector
# ---------------------------------------------------------------------------

def test_mind_diversion_detector_reset_matches_fresh_instance(minimal_settings, stub_logger):
    from app.core.detectors.mind_diversion_detector import MindDiversionDetector

    fresh = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)
    used = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)

    used._recent_person_activities[0] = {'writing': 3.5}
    used._recent_person_activities[1] = {'writing': 4.2}

    assert len(used._recent_person_activities) == 2

    used.reset()

    assert dict(used._recent_person_activities) == dict(fresh._recent_person_activities) == {}


# ---------------------------------------------------------------------------
# ActivityDetector
# ---------------------------------------------------------------------------

def test_activity_detector_reset_matches_fresh_instance(minimal_settings):
    from collections import deque
    from app.core.detectors.activity_detector import ActivityDetector

    fresh = ActivityDetector(settings=minimal_settings)
    used = ActivityDetector(settings=minimal_settings)

    used.packing_motion_history[0] = {
        'distances': deque([10.0, 12.0], maxlen=20),
        'timestamps': deque([0.5, 1.0], maxlen=20),
        'active_hand': deque(['left', 'left'], maxlen=20),
    }

    assert len(used.packing_motion_history) == 1

    used.reset()

    assert dict(used.packing_motion_history) == dict(fresh.packing_motion_history) == {}


# ---------------------------------------------------------------------------
# on_suppressed hook contract (per spec: train-stopped gate fans out)
# ---------------------------------------------------------------------------

def test_sleep_detector_on_suppressed_clears_only_target_person(minimal_settings, stub_logger):
    from app.core.detectors.sleep_detector import SleepDetector

    det = SleepDetector(settings=minimal_settings, logger=stub_logger)
    _ = det._get_per_person_sleep_tracking(0)
    _ = det._get_per_person_sleep_tracking(1)
    det.per_person_tracking[0]['pose_sleep_duration'] = 9.0
    det.per_person_tracking[1]['pose_sleep_duration'] = 9.0

    det.on_suppressed(0, 'sleep')

    assert det.per_person_tracking[0]['pose_sleep_duration'] == 0
    # Other person untouched.
    assert det.per_person_tracking[1]['pose_sleep_duration'] == 9.0


def test_sleep_detector_on_suppressed_ignores_other_activities(minimal_settings, stub_logger):
    from app.core.detectors.sleep_detector import SleepDetector

    det = SleepDetector(settings=minimal_settings, logger=stub_logger)
    _ = det._get_per_person_sleep_tracking(0)
    det.per_person_tracking[0]['pose_sleep_duration'] = 7.0

    det.on_suppressed(0, 'writing')

    # Sleep counter must remain — only 'sleep' activity triggers the reset.
    assert det.per_person_tracking[0]['pose_sleep_duration'] == 7.0


def test_gesture_detector_on_suppressed_drops_role_session(minimal_settings):
    from app.core.detectors.gesture_detector import GestureDetector

    det = GestureDetector(settings=minimal_settings)
    det.gesture_sessions['LP'] = {'last_raise_time': 100.0}
    det.gesture_sessions['ALP'] = {'last_raise_time': 100.0}

    det.on_suppressed(0, 'lp_hand_gesture')

    assert 'LP' not in det.gesture_sessions
    assert 'ALP' in det.gesture_sessions  # ALP untouched

    det.on_suppressed(1, 'alp_hand_gesture')
    assert 'ALP' not in det.gesture_sessions


def test_mind_diversion_on_suppressed_clears_writing_grace(minimal_settings, stub_logger):
    from app.core.detectors.mind_diversion_detector import MindDiversionDetector

    det = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)
    det._recent_person_activities[0] = {'writing': 3.0}
    det._recent_person_activities[1] = {'writing': 3.0}

    det.on_suppressed(0, 'writing')

    assert 0 not in det._recent_person_activities
    assert 1 in det._recent_person_activities  # other person unaffected


def test_activity_detector_on_suppressed_drops_packing_history(minimal_settings):
    from collections import deque
    from app.core.detectors.activity_detector import ActivityDetector

    det = ActivityDetector(settings=minimal_settings)
    det.packing_motion_history[0] = {
        'distances': deque([5.0], maxlen=20),
        'timestamps': deque([0.5], maxlen=20),
        'active_hand': deque(['left'], maxlen=20),
    }
    det.packing_motion_history[1] = {
        'distances': deque([5.0], maxlen=20),
        'timestamps': deque([0.5], maxlen=20),
        'active_hand': deque(['left'], maxlen=20),
    }

    det.on_suppressed(0, 'packing_bags')

    assert 0 not in det.packing_motion_history
    assert 1 in det.packing_motion_history


# ---------------------------------------------------------------------------
# Gate <-> detector wiring smoke test
# ---------------------------------------------------------------------------

def test_apply_train_stopped_suppression_invokes_detector_hooks(
    minimal_settings, stub_logger
):
    """End-to-end: gate fans out on_suppressed for every (person, activity)
    pair whose flag was actually flipped from True to False."""
    from app.core.gates import apply_train_stopped_suppression
    from app.core.detectors.sleep_detector import SleepDetector
    from app.core.detectors.gesture_detector import GestureDetector
    from app.core.detectors.mind_diversion_detector import MindDiversionDetector
    from app.core.detectors.activity_detector import ActivityDetector
    from collections import deque

    sleep = SleepDetector(settings=minimal_settings, logger=stub_logger)
    gesture = GestureDetector(settings=minimal_settings)
    mind = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)
    activity = ActivityDetector(settings=minimal_settings)

    # Seed each detector with state we expect the gate to clear.
    _ = sleep._get_per_person_sleep_tracking(0)
    sleep.per_person_tracking[0]['pose_sleep_duration'] = 12.0
    gesture.gesture_sessions['LP'] = {'last_raise_time': 5.0}
    mind._recent_person_activities[0] = {'writing': 1.0}
    activity.packing_motion_history[0] = {
        'distances': deque([1.0], maxlen=10),
        'timestamps': deque([1.0], maxlen=10),
        'active_hand': deque(['left'], maxlen=10),
    }

    aggregated = {
        'sleep_detected': True,
        'lp_hand_gesture_detected': True,
        'mind_diversion_detected': True,
        'packing_bags_detected': True,
        'writing_detected': True,
    }
    persons_data = {
        0: {
            'activities': {
                'sleep': True,
                'lp_hand_gesture': True,
                'mind_diversion': True,
                'packing_bags': True,
                'writing': True,
            }
        }
    }

    apply_train_stopped_suppression(
        aggregated,
        persons_data,
        detectors={
            'sleep': sleep,
            'gesture': gesture,
            'mind_diversion': mind,
            'activity': activity,
        },
    )

    # Aggregated/per-person flags zeroed (existing ARCH-08b contract).
    for k in aggregated:
        assert aggregated[k] is False
    for v in persons_data[0]['activities'].values():
        assert v is False

    # Detector counters cleared (new contract).
    assert sleep.per_person_tracking[0]['pose_sleep_duration'] == 0
    assert 'LP' not in gesture.gesture_sessions
    assert 0 not in mind._recent_person_activities
    assert 0 not in activity.packing_motion_history


def test_apply_train_stopped_suppression_skips_hooks_when_flag_already_false(
    minimal_settings, stub_logger
):
    """Hooks must only fire when the flag was actually flipped — if the
    activity was already False there's no counter to roll back."""
    from app.core.gates import apply_train_stopped_suppression
    from app.core.detectors.sleep_detector import SleepDetector

    sleep = SleepDetector(settings=minimal_settings, logger=stub_logger)
    _ = sleep._get_per_person_sleep_tracking(0)
    sleep.per_person_tracking[0]['pose_sleep_duration'] = 8.0

    aggregated = {'sleep_detected': False}
    persons_data = {0: {'activities': {'sleep': False}}}

    apply_train_stopped_suppression(
        aggregated, persons_data, detectors={'sleep': sleep}
    )

    # Counter untouched because no flag actually flipped.
    assert sleep.per_person_tracking[0]['pose_sleep_duration'] == 8.0


def test_apply_train_stopped_suppression_fires_hook_once_per_window(
    minimal_settings, stub_logger
):
    """When ``previously_suppressed`` is supplied, a (person, activity) pair
    must trigger ``on_suppressed`` only on the FIRST flag flip in the
    current STOPPED window — not on every frame the activity stays
    positive while the gate keeps zeroing it."""
    from app.core.gates import apply_train_stopped_suppression
    from app.core.detectors.sleep_detector import SleepDetector

    sleep = SleepDetector(settings=minimal_settings, logger=stub_logger)

    call_log: list = []

    original_hook = sleep.on_suppressed

    def spy_hook(pidx, act):
        call_log.append((pidx, act))
        original_hook(pidx, act)

    sleep.on_suppressed = spy_hook  # type: ignore[assignment]

    tracker: dict = {}

    # Frame 1: activity True -> gate flips it False, hook fires once.
    aggregated = {'sleep_detected': True}
    persons_data = {0: {'activities': {'sleep': True}}}
    apply_train_stopped_suppression(
        aggregated, persons_data,
        detectors={'sleep': sleep},
        previously_suppressed=tracker,
    )
    assert call_log == [(0, 'sleep')]

    # Frame 2 (still STOPPED): detector has recomputed sleep as True. Gate
    # zeros it again, but the hook MUST NOT re-fire — tracker says we
    # already cleared the counter for this (person, activity) pair this
    # window.
    aggregated = {'sleep_detected': True}
    persons_data = {0: {'activities': {'sleep': True}}}
    apply_train_stopped_suppression(
        aggregated, persons_data,
        detectors={'sleep': sleep},
        previously_suppressed=tracker,
    )
    assert call_log == [(0, 'sleep')], (
        "on_suppressed re-fired for the same (person, activity) within a "
        f"single STOPPED window: {call_log}"
    )

    # Train resumes -> caller clears the tracker.
    tracker.clear()

    # Frame 3 (new STOPPED window): hook must fire again, exactly once.
    aggregated = {'sleep_detected': True}
    persons_data = {0: {'activities': {'sleep': True}}}
    apply_train_stopped_suppression(
        aggregated, persons_data,
        detectors={'sleep': sleep},
        previously_suppressed=tracker,
    )
    assert call_log == [(0, 'sleep'), (0, 'sleep')]


# ---------------------------------------------------------------------------
# One-frame post-reset behavioral parity (per spec §Acceptance 1)
# ---------------------------------------------------------------------------
# These tests confirm that running a single frame through a freshly-``reset()``
# detector produces output identical to running that same frame through a
# brand-new detector. They use only synthetic landmarks (FakePoseLandmarks via
# the ``stub_yolo_keypoints`` fixture), so the suite stays weight-free.

def test_sleep_detector_post_reset_matches_fresh_one_frame(
    minimal_settings, stub_logger, stub_yolo_keypoints
):
    """Running one frame through a reset SleepDetector matches a fresh one."""
    from app.core.detectors.sleep_detector import SleepDetector

    fresh = SleepDetector(settings=minimal_settings, logger=stub_logger)
    used = SleepDetector(settings=minimal_settings, logger=stub_logger)

    # Drive ``used`` into a non-trivial state.
    _ = used._get_per_person_sleep_tracking(0)
    used.per_person_tracking[0]['pose_sleep_duration'] = 7.5
    used.per_person_tracking[0]['previous_landmarks'] = stub_yolo_keypoints(
        nose_y=0.55  # different head pose so movement_score wouldn't be zero
    )

    used.reset()

    pose = stub_yolo_keypoints()  # alert posture
    frame_shape = (480, 640, 3)
    timestamp = 12.0

    fresh_sleep, fresh_micro, fresh_debug = fresh.detect_pose_based_sleep(
        landmarks=pose,
        timestamp_sec=timestamp,
        person_idx=0,
        frame_shape=frame_shape,
    )
    used_sleep, used_micro, used_debug = used.detect_pose_based_sleep(
        landmarks=pose,
        timestamp_sec=timestamp,
        person_idx=0,
        frame_shape=frame_shape,
    )

    assert fresh_sleep == used_sleep
    assert fresh_micro == used_micro
    # Public per-person tracking must match. ``previous_landmarks`` is a
    # FakePoseLandmarks reference identity-different across instances, so
    # compare structural counter fields only.
    fresh_tracking = fresh.per_person_tracking[0]
    used_tracking = used.per_person_tracking[0]
    for key in (
        'pose_sleep_duration',
        'sleep_state',
        'first_drop_time',
        'last_alert_time',
    ):
        if key in fresh_tracking or key in used_tracking:
            assert fresh_tracking.get(key) == used_tracking.get(key), key


def test_gesture_detector_post_reset_matches_fresh_one_frame(
    minimal_settings, stub_yolo_keypoints
):
    """Running one frame through a reset GestureDetector matches a fresh one."""
    from collections import deque
    from app.core.detectors.gesture_detector import GestureDetector

    fresh = GestureDetector(settings=minimal_settings)
    used = GestureDetector(settings=minimal_settings)

    used.gesture_sessions['LP'] = {
        'last_raise_time': 5.0,
        'gesture_count': 2,
        'last_update': 6.0,
    }
    used.recent_person_activities[0] = {'writing': 4.0}
    used.hand_position_history[0] = {
        'right_wrist': deque([(120, 240, 1.0)], maxlen=10)
    }

    used.reset()

    pose = stub_yolo_keypoints()  # arms-down baseline
    person_bbox = [200, 100, 440, 460]
    frame_shape = (480, 640, 3)

    fresh_raised, fresh_debug = fresh.detect_raised_hand(
        landmarks=pose,
        person_bbox=person_bbox,
        frame_shape=frame_shape,
        person_activities={},
        backpack_detections=None,
        person_idx=0,
        current_timestamp=10.0,
    )
    used_raised, used_debug = used.detect_raised_hand(
        landmarks=pose,
        person_bbox=person_bbox,
        frame_shape=frame_shape,
        person_activities={},
        backpack_detections=None,
        person_idx=0,
        current_timestamp=10.0,
    )

    assert fresh_raised == used_raised
    # Session/activity dicts must remain empty post-reset for both
    # instances after a single arms-down frame.
    assert dict(fresh.gesture_sessions) == dict(used.gesture_sessions)
    assert dict(fresh.recent_person_activities) == dict(used.recent_person_activities)


def test_mind_diversion_post_reset_matches_fresh_one_frame(
    minimal_settings, stub_logger, stub_yolo_keypoints
):
    """Running one frame through a reset MindDiversionDetector matches fresh."""
    from app.core.detectors.mind_diversion_detector import MindDiversionDetector

    fresh = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)
    used = MindDiversionDetector(settings=minimal_settings, logger=stub_logger)

    used._recent_person_activities[0] = {'writing': 1.0}
    used._recent_person_activities[1] = {'writing': 2.0}

    used.reset()

    pose = stub_yolo_keypoints()
    frame_shape = (480, 640, 3)

    fresh_det, fresh_sub, fresh_debug = fresh.detect_mind_diversion(
        pose_landmarks=pose,
        face_landmarks=None,
        frame_shape=frame_shape,
        writing_active=False,
        person_idx=0,
        current_time=15.0,
    )
    used_det, used_sub, used_debug = used.detect_mind_diversion(
        pose_landmarks=pose,
        face_landmarks=None,
        frame_shape=frame_shape,
        writing_active=False,
        person_idx=0,
        current_time=15.0,
    )

    assert fresh_det == used_det
    assert fresh_sub == used_sub
    # Activity-history dict must remain empty after reset + one
    # writing_active=False frame.
    assert (
        dict(fresh._recent_person_activities)
        == dict(used._recent_person_activities)
    )


def test_activity_detector_post_reset_matches_fresh_one_frame(
    minimal_settings, stub_yolo_keypoints
):
    """Running one frame through a reset ActivityDetector matches a fresh one."""
    from collections import deque
    from app.core.detectors.activity_detector import ActivityDetector

    fresh = ActivityDetector(settings=minimal_settings)
    used = ActivityDetector(settings=minimal_settings)

    used.packing_motion_history[0] = {
        'distances': deque([8.0, 9.0], maxlen=20),
        'timestamps': deque([0.5, 1.0], maxlen=20),
        'active_hand': deque(['left', 'left'], maxlen=20),
    }

    used.reset()

    pose = stub_yolo_keypoints()
    person_bbox = [200, 100, 440, 460]
    frame_shape = (480, 640, 3)

    fresh_writing, fresh_evidence = fresh.detect_writing(
        landmarks=pose,
        book_detections=[],
        person_bbox=person_bbox,
        frame_shape=frame_shape,
    )
    used_writing, used_evidence = used.detect_writing(
        landmarks=pose,
        book_detections=[],
        person_bbox=person_bbox,
        frame_shape=frame_shape,
    )

    assert fresh_writing == used_writing
    assert fresh_evidence == used_evidence
    # No book in scene + no packing call -> packing_motion_history must
    # stay empty for both instances.
    assert (
        dict(fresh.packing_motion_history)
        == dict(used.packing_motion_history)
        == {}
    )
