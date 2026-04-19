"""Unit tests for task 0007 — detector instances reused from ``_worker_models``.

These tests exercise three contracts:

1. ``worker_initializer`` stores constructed ``SleepDetector``/
   ``GestureDetector``/``MindDiversionDetector``/``ActivityDetector``
   instances on the module-level ``_worker_models`` dict exactly once per
   worker.
2. ``LocopilotActivityMonitor.__init__`` reuses those instances when they
   are passed in via ``preloaded_models``.
3. When pre-loaded detectors are reused, the monitor resets per-chunk
   state (``SleepDetector.per_person_tracking`` and
   ``GestureDetector.gesture_sessions``) so no data bleeds across chunks.

The tests avoid loading any GPU/heavy models by stubbing ``ultralytics``
and ``mediapipe`` imports through ``MagicMock`` instances in the
``preloaded_models`` dict.  The ``worker_initializer`` test patches
``ultralytics.YOLO`` and ``mediapipe.solutions.face_mesh.FaceMesh`` so the
real weights are never fetched.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the project root importable when tests are run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# mediapipe >= 0.10.32 drops the legacy ``mp.solutions`` namespace that
# ``locopilot_monitor`` still references for pose/face_mesh/drawing
# constants.  For unit tests we never actually run inference, so inject a
# lightweight stub ``solutions`` module with the attributes the monitor
# touches during ``__init__``.  This keeps tests runnable on the pinned
# local venv without installing an older mediapipe.
import types as _types  # noqa: E402
import mediapipe as _mp  # noqa: E402

if not hasattr(_mp, 'solutions'):
    _stub_solutions = _types.SimpleNamespace()
    _stub_face_mesh_mod = _types.SimpleNamespace(FaceMesh=MagicMock(name='FaceMesh'))
    _stub_pose_mod = _types.SimpleNamespace()
    _stub_drawing_utils_mod = _types.SimpleNamespace()
    _stub_drawing_styles_mod = _types.SimpleNamespace()
    _stub_solutions.face_mesh = _stub_face_mesh_mod
    _stub_solutions.pose = _stub_pose_mod
    _stub_solutions.drawing_utils = _stub_drawing_utils_mod
    _stub_solutions.drawing_styles = _stub_drawing_styles_mod
    _mp.solutions = _stub_solutions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_preloaded_models(*, extra: dict | None = None) -> dict:
    """Build the minimum ``preloaded_models`` dict the monitor needs.

    ``LocopilotActivityMonitor.__init__`` validates that ``'yolo'`` and
    ``'yolo_pose'`` are non-None and then calls ``.model.fuse()`` on the
    YOLO object only when fresh-loading (the fast path skips fuse on
    preloaded models, so a simple MagicMock is sufficient).
    """
    yolo = MagicMock(name='yolo_model')
    yolo.model = MagicMock()
    yolo_pose = MagicMock(name='yolo_pose')
    preloaded = {
        'yolo': yolo,
        'yolo_pose': yolo_pose,
        'face_mesh': MagicMock(name='face_mesh'),
        'mp_face_mesh': MagicMock(name='mp_face_mesh'),
        'preprocessing_service': None,
    }
    if extra:
        preloaded.update(extra)
    return preloaded


# ---------------------------------------------------------------------------
# worker_initializer — detector construction
# ---------------------------------------------------------------------------


def test_worker_initializer_stores_detectors_on_worker_models():
    """``worker_initializer`` should populate ``_worker_models`` with all
    four detector instances exactly once per worker."""
    from app.utils import video_multiprocessing as vmp
    from app.utils.multiprocessing_config import MultiprocessingConfig

    # Reset global worker state between tests.
    vmp._worker_models = None

    # Build a config that preloads models but uses CPU and nonsense paths
    # (we'll patch YOLO + FaceMesh to avoid hitting the network).
    cfg = MultiprocessingConfig(
        max_workers=1,
        yolo_model_path='yolo26n.pt',
        yolo_pose_model_path='yolo26n-pose.pt',
        yolo_device='cpu',
        preload_models=True,
    )

    fake_yolo = MagicMock(name='yolo_weights')
    fake_yolo.model = MagicMock()
    fake_face_mesh = MagicMock(name='face_mesh_instance')

    with patch('ultralytics.YOLO', return_value=fake_yolo), \
         patch('mediapipe.solutions.face_mesh.FaceMesh', return_value=fake_face_mesh), \
         patch('app.services.yolo_pose_adapter.YoloPoseAdapter') as pose_adapter_cls:
        pose_adapter_cls.return_value = MagicMock(name='pose_adapter')
        vmp.worker_initializer(cfg)

    assert vmp._worker_models is not None, \
        'worker_initializer must populate _worker_models'

    for key in (
        'sleep_detector',
        'gesture_detector',
        'mind_diversion_detector',
        'activity_detector',
    ):
        assert key in vmp._worker_models, f'missing key: {key}'
        assert vmp._worker_models[key] is not None, f'{key} is None'

    # Sanity-check class names so future refactors (or swap-ins of stubs)
    # don't silently drop the real types.
    assert type(vmp._worker_models['sleep_detector']).__name__ == 'SleepDetector'
    assert type(vmp._worker_models['gesture_detector']).__name__ == 'GestureDetector'
    assert (
        type(vmp._worker_models['mind_diversion_detector']).__name__
        == 'MindDiversionDetector'
    )
    assert type(vmp._worker_models['activity_detector']).__name__ == 'ActivityDetector'

    # Reset back so we don't leak state to other tests.
    vmp._worker_models = None


# ---------------------------------------------------------------------------
# LocopilotActivityMonitor — reuse vs fresh construction
# ---------------------------------------------------------------------------


def test_monitor_creates_fresh_detectors_when_preloaded_is_none():
    """When ``preloaded_models`` is ``None`` (single-process path), the
    monitor must construct its own detectors."""
    from locopilot_monitor import LocopilotActivityMonitor
    from app.core.detectors import (
        ActivityDetector,
        GestureDetector,
        MindDiversionDetector,
        SleepDetector,
    )

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch('locopilot_monitor.YOLO') as mock_yolo_cls, \
         patch('app.services.yolo_pose_adapter.YoloPoseAdapter') as mock_pose_adapter_cls:
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_pose_adapter_cls.return_value = MagicMock()

        monitor = LocopilotActivityMonitor(
            video_path='/tmp/nonexistent.mp4',
            output_dir=tmpdir,
            save_annotated_frames=False,
            frame_save_interval=1,
            sample_fps=0.5,
            run_dir=None,
            create_run_dir=False,
            preloaded_models=None,
        )

    assert isinstance(monitor.sleep_detector, SleepDetector)
    assert isinstance(monitor.gesture_detector, GestureDetector)
    assert isinstance(monitor.mind_diversion_detector, MindDiversionDetector)
    assert isinstance(monitor.activity_detector, ActivityDetector)


def test_monitor_reuses_preloaded_detectors_when_present():
    """When ``preloaded_models`` contains detector stubs, the monitor must
    use those exact instances instead of constructing new ones."""
    from locopilot_monitor import LocopilotActivityMonitor

    # Stubs with the minimum surface the monitor touches during __init__:
    # reset_tracking() on sleep, reset() on gesture.
    sleep_stub = MagicMock(name='sleep_stub')
    sleep_stub.per_person_tracking = {'person_0': {'sleep_score': 3}}
    gesture_stub = MagicMock(name='gesture_stub')
    mind_stub = MagicMock(name='mind_stub')
    activity_stub = MagicMock(name='activity_stub')

    preloaded = _make_preloaded_models(
        extra={
            'sleep_detector': sleep_stub,
            'gesture_detector': gesture_stub,
            'mind_diversion_detector': mind_stub,
            'activity_detector': activity_stub,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = LocopilotActivityMonitor(
            video_path='/tmp/nonexistent.mp4',
            output_dir=tmpdir,
            save_annotated_frames=False,
            frame_save_interval=1,
            sample_fps=0.5,
            run_dir=None,
            create_run_dir=False,
            preloaded_models=preloaded,
        )

    assert monitor.sleep_detector is sleep_stub, \
        'monitor must reuse preloaded sleep_detector instance'
    assert monitor.gesture_detector is gesture_stub, \
        'monitor must reuse preloaded gesture_detector instance'
    assert monitor.mind_diversion_detector is mind_stub, \
        'monitor must reuse preloaded mind_diversion_detector instance'
    assert monitor.activity_detector is activity_stub, \
        'monitor must reuse preloaded activity_detector instance'

    # reset_tracking() must have been called during __init__ (state reset
    # on reuse — the per-chunk contract documented in the monitor docstring).
    sleep_stub.reset_tracking.assert_called_once()
    gesture_stub.reset.assert_called_once()
    # clear_motion_history() on the activity detector must also fire — the
    # monitor must zero ActivityDetector.packing_motion_history so stale
    # deques from a prior chunk cannot trigger packing_bags FPs in the
    # first frames of a new chunk.
    activity_stub.clear_motion_history.assert_called_once()


def test_real_sleep_detector_state_cleared_after_chunk_reset():
    """End-to-end check that a real SleepDetector's ``per_person_tracking``
    dict is empty after the monitor reuses it across chunks."""
    from locopilot_monitor import LocopilotActivityMonitor
    from app.core.detectors import SleepDetector

    # Build a real SleepDetector (no heavy deps), pollute its state, then
    # pass it through the monitor and verify the state was cleared.
    real_sleep = SleepDetector(settings=None, sample_fps=0.5)
    # Seed some tracking state so the reset is observable.
    real_sleep.per_person_tracking[0] = {'some': 'state'}
    real_sleep.ir_forward_lean_tracking[0] = {'start_time': 1.0}
    assert real_sleep.per_person_tracking, 'precondition: state should be seeded'

    preloaded = _make_preloaded_models(
        extra={
            'sleep_detector': real_sleep,
            # keep the others fresh — only sleep reset is load-bearing here
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = LocopilotActivityMonitor(
            video_path='/tmp/nonexistent.mp4',
            output_dir=tmpdir,
            save_annotated_frames=False,
            frame_save_interval=1,
            sample_fps=0.5,
            run_dir=None,
            create_run_dir=False,
            preloaded_models=preloaded,
        )

    assert monitor.sleep_detector is real_sleep
    assert monitor.sleep_detector.per_person_tracking == {}, \
        'per_person_tracking must be cleared on chunk reset'
    assert monitor.sleep_detector.ir_forward_lean_tracking == {}, \
        'ir_forward_lean_tracking must be cleared on chunk reset'


def test_real_activity_detector_motion_history_cleared_after_chunk_reset():
    """Regression for the packing_bags FP at chunk boundaries.

    ``ActivityDetector.packing_motion_history`` holds per-person deques of
    (distance, timestamp, active_hand) used by
    ``analyze_packing_hand_motion`` to compute
    ``sustained_proximity_time = timestamps[-1] - timestamps[0] >= 4.0``.
    If stale timestamps from a prior chunk survive into the new chunk,
    the first close-proximity sample will flip ``sustained_proximity_time``
    to ``True`` and trigger a packing_bags false positive. The monitor
    must call ``clear_motion_history()`` on every preloaded reuse.
    """
    from collections import deque

    from locopilot_monitor import LocopilotActivityMonitor
    from app.core.detectors import ActivityDetector

    real_activity = ActivityDetector(settings=None)
    # Seed stale motion history so the reset is observable.
    real_activity.packing_motion_history[0] = {
        'distances': deque([120.0, 118.0, 115.0], maxlen=6),
        'timestamps': deque([10.0, 12.0, 14.0], maxlen=6),
        'active_hand': deque(['left', 'left', 'left'], maxlen=6),
    }
    real_activity.packing_motion_history[1] = {
        'distances': deque([200.0], maxlen=6),
        'timestamps': deque([18.0], maxlen=6),
        'active_hand': deque(['right'], maxlen=6),
    }
    assert real_activity.packing_motion_history, \
        'precondition: motion history should be seeded'

    preloaded = _make_preloaded_models(
        extra={
            'activity_detector': real_activity,
            # keep the others fresh
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = LocopilotActivityMonitor(
            video_path='/tmp/nonexistent.mp4',
            output_dir=tmpdir,
            save_annotated_frames=False,
            frame_save_interval=1,
            sample_fps=0.5,
            run_dir=None,
            create_run_dir=False,
            preloaded_models=preloaded,
        )

    assert monitor.activity_detector is real_activity
    assert monitor.activity_detector.packing_motion_history == {}, \
        'packing_motion_history must be cleared on chunk reset'
