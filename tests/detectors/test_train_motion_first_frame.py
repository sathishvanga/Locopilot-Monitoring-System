"""First-frame-after-reset semantics for ``TrainMotionDetector``.

Before task 0006, ``TrainMotionDetector`` had no ``reset()``. When a single
detector instance was reused across two videos, the first frame of video B
was diffed against the last frame of video A — a guaranteed phantom
RUNNING signal at the start of every video after the first.

This test feeds the detector a contrived "video A" (a moving sequence
that drives ``prev_gray`` to a known non-static state), calls ``reset()``,
and then feeds a single static frame and confirms the detector does NOT
report ``RUNNING``. The actual observed result for a static
first-frame-after-reset is ``STOPPED``: ``compute_vibration`` populates
``prev_gray`` and returns the all-zero vibration block, so by the time
``process_frame`` evaluates its UNKNOWN early-return guard at the bottom
``prev_gray is None`` is already False; the combined score is ~0 and the
raw decision falls into the ``STOPPED`` branch (which is the safe outcome
for the safety-critical contract — anything except ``RUNNING`` is fine).
The assertion below stays at ``state != "RUNNING"`` to reflect that
contract regardless of which non-running label the detector emits.
"""

from __future__ import annotations

import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_textured_frame(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    """Return a deterministic BGR frame with high-frequency texture so the
    detector's vibration analysis has something to bite into."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _make_static_frame(value: int = 128, h: int = 240, w: int = 320) -> np.ndarray:
    """Return a uniform BGR frame — no inter-frame motion possible."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_train_motion_first_frame_after_reset_is_not_running():
    """A static first frame after reset must not report RUNNING."""
    from app.core.detectors.train_motion_detector import TrainMotionDetector

    det = TrainMotionDetector()

    # Video A: feed several high-vibration frames to guarantee prev_gray
    # holds a non-uniform pattern and the temporal smoother has data.
    for i in range(6):
        frame = _make_textured_frame(seed=i)
        det.process_frame(frame, person_bboxes=[])

    # Sanity: prev_gray is populated (state from video A is real).
    assert det.prev_gray is not None
    # state_history should have entries from the smoother.
    assert len(det.state_history) > 0

    # Cross the video boundary.
    det.reset()

    # Sanity: reset cleared the diff-against state.
    assert det.prev_gray is None
    assert det.prev_gray_window is None
    assert len(det.state_history) == 0

    # Video B: a single static frame. With reset, this must NOT report
    # RUNNING (no prior frame to diff against -> UNKNOWN/STOPPED-equivalent).
    static_frame = _make_static_frame(value=128)
    state, conf, diag = det.process_frame(static_frame, person_bboxes=[])

    assert state != "RUNNING", (
        f"first frame after reset reported RUNNING (state={state}, "
        f"conf={conf}, vib_mean={diag.get('vib_mean')}); reset() did not "
        "clear prev_gray correctly"
    )


def test_train_motion_without_reset_can_report_running_on_video_b_first_frame():
    """Regression-witness: WITHOUT reset, the first frame of video B is
    diffed against the last frame of video A and the detector can fire.

    This locks in the fact that ``reset()`` is the correct fix — if this
    test ever passes after a code change to the detector internals, the
    setup above no longer drives prev_gray to a "different" state and the
    primary test loses signal.
    """
    from app.core.detectors.train_motion_detector import TrainMotionDetector

    det = TrainMotionDetector()

    # Video A: a sequence of high-noise frames so prev_gray ends up holding
    # a dense random pattern.
    for i in range(6):
        det.process_frame(_make_textured_frame(seed=i), person_bboxes=[])
    last_gray_a = det.prev_gray
    assert last_gray_a is not None

    # Video B's first frame is uniform 128. Without reset, the diff against
    # last_gray_a (random noise around 128) will be large.
    static_first_frame_b = _make_static_frame(value=128)

    # Don't reset. Just call process_frame.
    state_b, conf_b, diag_b = det.process_frame(static_first_frame_b, person_bboxes=[])

    # The point isn't that we see exactly RUNNING — it's that the diff
    # produced is non-trivial (i.e. the vibration measurement is not zero).
    # If reset() were not the fix, we'd see vib_mean > 0 here, in contrast
    # to the reset path where it's exactly 0.
    assert diag_b['vib_mean_raw'] > 0.0, (
        "Without reset, the static first frame of video B should diff "
        "against video A's last frame and produce a non-zero vibration "
        "signal. If this assertion fails, the witness setup may need to "
        "be tuned (e.g. larger texture amplitude)."
    )


def test_train_motion_reset_then_frame_records_no_diff():
    """After reset, the very first ``process_frame`` should populate
    ``prev_gray`` but compute zero vibration (no prior frame to diff)."""
    from app.core.detectors.train_motion_detector import TrainMotionDetector

    det = TrainMotionDetector()

    # Run a real video A.
    for i in range(4):
        det.process_frame(_make_textured_frame(seed=i), person_bboxes=[])

    det.reset()

    static_frame = _make_static_frame(value=200)
    _state, _conf, diag = det.process_frame(static_frame, person_bboxes=[])

    # First frame after reset takes the early-return path:
    # prev_gray is None -> compute_vibration returns the all-zero result
    # block and stores prev_gray.
    assert diag['vib_mean'] == 0.0
    assert diag['vib_mean_raw'] == 0.0
    assert diag['vib_score'] == 0.0
    # And prev_gray is populated for the next call.
    assert det.prev_gray is not None
