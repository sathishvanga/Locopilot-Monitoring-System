"""Unit tests for ``app/core/frame_pipeline.py``.

These tests only exercise the scaffolding introduced by task 0002:

* ``FrameState`` construction and default-field behavior.
* ``FramePipeline`` runs stages in the order they were registered.
* A no-op stage can be inserted mid-pipeline without affecting what
  downstream stages observe.
* Stages remain idempotent on a stub ``FrameState`` (re-running a stage
  does not clobber earlier state keys it was not supposed to touch).

They intentionally use pure-Python stubs for the monitor and stages so
they can run on a workstation without YOLO / cv2 / OpenCV models. The
actual per-frame YOLO + MediaPipe pipeline is validated separately by
the (forthcoming) snapshot test referenced in task 0006.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest

from app.core.frame_pipeline import FramePipeline, FrameState, Stage


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class StubMonitor:
    """Minimal monitor stub. Stages mutate ``call_log`` so tests can
    assert the call order. No detection models are loaded."""

    call_log: List[str] = field(default_factory=list)
    frame_buffer: List[Any] = field(default_factory=list)
    frame_idx_buffer: List[int] = field(default_factory=list)


class RecordingStage:
    """Appends its name onto ``state.activity_flags['calls']`` every run."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        calls = state.activity_flags.setdefault("calls", [])
        # ``state.activity_flags['calls']`` is a list; append preserves order.
        calls.append(self.name)
        monitor.call_log.append(self.name)
        return state


class NoOpStage:
    """Stage that explicitly returns ``None`` (the pipeline should keep
    using the previous state)."""

    name = "noop"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        return None  # type: ignore[return-value]


class IncrementStage:
    """Stage that increments a counter on the state. Used to verify that
    inserting a no-op stage mid-pipeline does not affect downstream stages
    — the counter should still end up at the expected value regardless of
    where the no-op was inserted."""

    name = "increment"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        counter = state.activity_flags.get("counter", 0)
        state.activity_flags["counter"] = counter + 1
        return state


def make_state(**overrides: Any) -> FrameState:
    """Build a minimal ``FrameState`` with a tiny numpy frame."""
    defaults: Dict[str, Any] = dict(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_idx=0,
        timestamp_sec=0.0,
        sample_idx=0,
        total_frames=100,
        fps=1.0,
    )
    defaults.update(overrides)
    return FrameState(**defaults)


# --------------------------------------------------------------------------- #
# FrameState construction                                                     #
# --------------------------------------------------------------------------- #


def test_framestate_defaults() -> None:
    state = make_state()
    assert state.frame_idx == 0
    assert state.fps == 1.0
    assert state.face_results is None
    assert state.detections == {}
    assert state.person_roles == {}
    assert state.persons_data == {}
    assert state.aggregated == {}
    assert state.motion_state == "UNKNOWN"
    assert state.activities_map == {}
    assert state.activity_flags == {}
    assert state.sleep_state_overrides_writing is False
    assert state.lp_not_coordinating is False
    assert state.alp_not_coordinating is False


# --------------------------------------------------------------------------- #
# Ordering + no-op + idempotency                                              #
# --------------------------------------------------------------------------- #


def test_pipeline_runs_stages_in_order() -> None:
    monitor = StubMonitor()
    pipeline = FramePipeline(
        [
            RecordingStage("alpha"),
            RecordingStage("beta"),
            RecordingStage("gamma"),
        ]
    )
    state = pipeline.run(make_state(), monitor)

    assert state.activity_flags["calls"] == ["alpha", "beta", "gamma"]
    assert monitor.call_log == ["alpha", "beta", "gamma"]


def test_pipeline_reordering_changes_call_order() -> None:
    monitor = StubMonitor()
    pipeline = FramePipeline(
        [
            RecordingStage("gamma"),
            RecordingStage("alpha"),
            RecordingStage("beta"),
        ]
    )
    state = pipeline.run(make_state(), monitor)
    assert state.activity_flags["calls"] == ["gamma", "alpha", "beta"]


def test_noop_stage_is_safe_mid_pipeline() -> None:
    """A stage returning ``None`` must not clobber state and must not
    affect what the downstream stages observe."""
    monitor = StubMonitor()
    pipeline_without_noop = FramePipeline(
        [IncrementStage(), IncrementStage(), IncrementStage()]
    )
    state_without = pipeline_without_noop.run(make_state(), monitor)

    monitor_with = StubMonitor()
    pipeline_with_noop = FramePipeline(
        [IncrementStage(), NoOpStage(), IncrementStage(), NoOpStage(), IncrementStage()]
    )
    state_with = pipeline_with_noop.run(make_state(), monitor_with)

    assert state_without.activity_flags["counter"] == 3
    assert state_with.activity_flags["counter"] == 3


def test_stage_is_idempotent_on_stub_state() -> None:
    """Running ``IncrementStage`` twice should cleanly double the counter
    — i.e., each invocation is a pure function of the state it receives
    and does not stash hidden side effects on the stage instance."""
    monitor = StubMonitor()
    stage = IncrementStage()
    state = make_state()
    stage.run(state, monitor)
    stage.run(state, monitor)
    stage.run(state, monitor)
    assert state.activity_flags["counter"] == 3


def test_pipeline_len_and_iter() -> None:
    stages = [RecordingStage("a"), RecordingStage("b")]
    pipeline = FramePipeline(stages)
    assert len(pipeline) == 2
    assert [s.name for s in pipeline] == ["a", "b"]


def test_pipeline_append_and_insert() -> None:
    monitor = StubMonitor()
    pipeline = FramePipeline([RecordingStage("a")])
    pipeline.append(RecordingStage("c"))
    pipeline.insert(1, RecordingStage("b"))
    state = pipeline.run(make_state(), monitor)
    assert state.activity_flags["calls"] == ["a", "b", "c"]


def test_stage_protocol_runtime_checkable() -> None:
    """Anything with a ``name`` attribute and a ``run(state, monitor)``
    method satisfies the ``Stage`` protocol."""
    stage = RecordingStage("x")
    assert isinstance(stage, Stage)
    assert isinstance(NoOpStage(), Stage)
    assert isinstance(IncrementStage(), Stage)


# --------------------------------------------------------------------------- #
# Contract: stages can be imported + constructed without the monitor          #
# --------------------------------------------------------------------------- #


def test_all_14_stages_importable_and_constructible() -> None:
    """Acceptance criterion: 'Each stage can be imported and called
    without constructing a full LocopilotActivityMonitor'."""
    from app.core.pipeline.stages import (
        EvidenceStage,
        FaceMeshStage,
        GestureCoordinationStage,
        GroupDetectStage,
        NoPersonScheduleSuppressStage,
        ObjectDetectStage,
        PerPersonActivitiesStage,
        PersonDedupStage,
        RoleIdentifyStage,
        SleepWritingOverrideStage,
        StateMachineGateStage,
        TemporalFilterStage,
        TrainMotionDetectStage,
        TrainMotionSuppressStage,
    )

    stages = [
        FaceMeshStage(),
        ObjectDetectStage(),
        PersonDedupStage(),
        RoleIdentifyStage(),
        GroupDetectStage(),
        TrainMotionDetectStage(),
        PerPersonActivitiesStage(),
        StateMachineGateStage(),
        SleepWritingOverrideStage(),
        GestureCoordinationStage(),
        TrainMotionSuppressStage(),
        NoPersonScheduleSuppressStage(),
        TemporalFilterStage(),
        EvidenceStage(),
    ]
    # All 14 stages expose a non-empty ``name`` attribute.
    names = [s.name for s in stages]
    assert len(names) == 14
    assert len(set(names)) == 14, f"duplicate stage names: {names}"
    for s in stages:
        assert isinstance(s, Stage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
