"""Smoke tests for the T6 refactor extractions.

T6 lifts three byte-identical helpers out of ``locopilot_monitor.py``:

* ``app.core.utils.video_io.video_capture_context``
* ``app.core.pipeline.frame_sampling.sample_video_frames``
* ``app.core.pipeline.pose_batch.detect_poses_batch``

These tests don't exercise OpenCV / ultralytics — they verify that the new
modules import cleanly, expose the expected public symbols, and that the
generator / batch helpers behave correctly with no work to do (empty input
or an immediately-failing capture). The full behavioral coverage continues
to come from the existing ground-truth regression on a real video; that
runs after the rewire (TR) lands.
"""
from __future__ import annotations

import inspect
import logging

import pytest


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------

def test_video_io_import():
    from app.core.utils.video_io import video_capture_context

    assert callable(video_capture_context)


def test_frame_sampling_import():
    from app.core.pipeline.frame_sampling import sample_video_frames

    assert callable(sample_video_frames)


def test_pose_batch_import():
    from app.core.pipeline.pose_batch import detect_poses_batch

    assert callable(detect_poses_batch)


# ---------------------------------------------------------------------------
# Behavioral micro-tests (no real video / model required)
# ---------------------------------------------------------------------------

def test_video_capture_context_releases_on_failure(tmp_path):
    """Opening a non-existent path returns an unopened capture. The context
    manager must still exit cleanly without raising."""
    from app.core.utils.video_io import video_capture_context

    bogus = str(tmp_path / "does_not_exist.mp4")
    with video_capture_context(bogus) as cap:
        assert cap.isOpened() is False


def test_sample_video_frames_raises_on_unopenable_video(tmp_path):
    """The sampler should raise ``RuntimeError`` matching the original
    behavior preserved from ``locopilot_monitor.py``."""
    from app.core.pipeline.frame_sampling import sample_video_frames

    bogus = str(tmp_path / "missing.mp4")
    gen = sample_video_frames(bogus, sample_fps=1.0, logger=logging.getLogger("test"))
    with pytest.raises(RuntimeError):
        next(gen)


def test_detect_poses_batch_empty_returns_empty_list(monkeypatch):
    """Empty frame list must short-circuit to an empty list. The lifted
    body imports ``YoloPoseLandmarks`` / ``PersonKeypoints`` from
    ``app.services.yolo_pose_adapter`` first (matching today's behavior),
    so we monkeypatch ``sys.modules`` to stub that import out — the
    detector code path is what matters here, not the heavy adapter
    package initialisation."""
    import sys
    import types

    fake_pkg = types.ModuleType("app.services.yolo_pose_adapter")
    fake_pkg.YoloPoseLandmarks = object  # type: ignore[attr-defined]
    fake_pkg.PersonKeypoints = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.yolo_pose_adapter", fake_pkg)

    from app.core.pipeline.pose_batch import detect_poses_batch

    class _SentinelAdapter:
        # If the helper touches .model or .conf_threshold on an empty list,
        # this attribute access would raise AttributeError.
        pass

    result = detect_poses_batch(_SentinelAdapter(), [], logger=logging.getLogger("test"))
    assert result == []


def test_sample_video_frames_signature_keyword_only():
    """The new public signature must keep ``sample_fps`` keyword-only so
    callers don't accidentally pass it positionally — this matches the
    rewire plan in Section 3 (TR-3)."""
    from app.core.pipeline.frame_sampling import sample_video_frames

    sig = inspect.signature(sample_video_frames)
    params = sig.parameters
    assert params["video_path"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["sample_fps"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["start_frame"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["end_frame"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["logger"].kind == inspect.Parameter.KEYWORD_ONLY


def test_detect_poses_batch_signature_keyword_only():
    from app.core.pipeline.pose_batch import detect_poses_batch

    sig = inspect.signature(detect_poses_batch)
    params = sig.parameters
    # First two are positional (adapter, frames), rest are keyword-only.
    assert params["yolo_pose_adapter"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["frames"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["batch_size"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["conf_threshold"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["device"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["logger"].kind == inspect.Parameter.KEYWORD_ONLY
