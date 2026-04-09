"""Frame processing pipeline scaffolding.

Introduced as part of architecture review task 0002 to decompose the
``LocopilotActivityMonitor._process_frames_core`` method into an ordered
list of typed stages operating on a shared ``FrameState`` dataclass.

This module intentionally contains no detection logic of its own. The
individual stage classes live under ``app/core/pipeline/stages/`` and each
extract a contiguous section of the original ``_process_frames_core`` body
verbatim so that behavior (and output) remains byte-identical to the
pre-refactor implementation.

CUTOVER STATUS (2026-04-09 merge into feature/cropping-image-applying-yolo)
--------------------------------------------------------------------------
The scaffolding (``FrameState`` + ``FramePipeline`` + 14 stage files) is
landed, but ``_process_frames_core`` in ``locopilot_monitor.py`` is NOT yet
calling the pipeline. This is intentional: the stages were extracted from
commit ``929272b`` and four subsequent commits have modified the original
``_process_frames_core`` body with phone/sleep/microsleep/train-motion
fixes that are NOT yet ported into the stage files. Cutting over now would
silently regress those fixes.

Follow-up task: re-extract the stages from the current
``_process_frames_core`` body, verify against a snapshot test (task 0006),
and switch ``_process_frames_core`` to iterate ``self.frame_pipeline.stages``.
The scaffolding is in place so that follow-up is a pure edit to the stage
files plus a 50-line orchestrator change in the monitor.

Design notes
------------
* ``FrameState`` is a plain dataclass that is *threaded through* every stage.
  Stages read from and write to the state in place and return it, so callers
  can treat the pipeline as a simple ``reduce`` over stages.
* Each ``Stage`` receives the monitor instance as a run-time parameter
  (rather than capturing it in ``__init__``), so that stages remain trivially
  importable and testable with a stub monitor object.
* ``FramePipeline`` is just an ordered list of ``Stage`` instances with a
  ``run`` method. Reordering or inserting a stage is a pure edit to the list
  at construction time — ``_process_frames_core`` itself never needs to
  change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class FrameState:
    """Mutable state threaded through every stage of the frame pipeline.

    Fields at construction time (populated by ``_process_frames_core``):
        frame: The BGR input frame (numpy array).
        frame_idx: Source-video frame index.
        timestamp_sec: Wall-clock-ish seconds from the start of the video.
        sample_idx: Sequential sample index (for progress logging + periodic saves).
        total_frames: Total frames in the source video (used for progress %).
        fps: Native FPS of the source video.
        batch_object_detections / batch_pose_results / batch_sleep_pose_results:
            Optional pre-computed batch inference results.
        batch_idx: Index into the batch arrays for this frame.
        save_clips / log_per_person_detections / enable_stale_cleanup:
            Passthrough flags controlling stage behavior.

    Fields populated as stages run (default empty):
        timestamp: Pre-computed ``HH:MM:SS`` string (populated by FaceMeshStage).
        rgb_frame: BGR->RGB conversion of ``frame`` (populated by FaceMeshStage).
        face_results: MediaPipe face-mesh output.
        detections: Dict of YOLO detections keyed by class.
        people_count: Deduplicated person count (at least 1).
        person_roles: Per-person role dict (LP / ALP / etc.).
        group_detected_flag: Whether a multi-person group was voted through.
        persons_data: Per-person detection + activity dict.
        aggregated: Aggregated cross-person activity flags.
        motion_state: Current train motion state string (``"UNKNOWN"`` when
            no motion detector is wired in this build).
        annotated_frame_for_activity: Annotated BGR frame for clips + evidence.
        activity_flags: Flat dict of bool flags populated by the activity
            stages (sleep, microsleep, cell_phone, etc.) and mutated by
            override / suppression stages.
        activities_map: Final activity map passed into temporal filtering.
        ocr_timestamp: Parsed OCR timestamp (optional).
        no_person_detected_flag: True when frame has no deduplicated persons.
        sleep_state_overrides_writing: Flag set by SleepWritingOverrideStage.
    """

    frame: np.ndarray
    frame_idx: int
    timestamp_sec: float
    sample_idx: int
    total_frames: int
    fps: float

    # Batch inference passthroughs
    batch_object_detections: Any = None
    batch_pose_results: Any = None
    batch_sleep_pose_results: Any = None
    batch_idx: int = 0

    # Flags controlling stage behavior
    save_clips: bool = True
    log_per_person_detections: bool = True
    enable_stale_cleanup: bool = True

    # Populated during pipeline execution
    timestamp: str = ""
    rgb_frame: Optional[np.ndarray] = None
    face_results: Any = None
    detections: Dict[str, Any] = field(default_factory=dict)
    people_count: int = 1
    person_roles: Dict[int, Any] = field(default_factory=dict)
    group_detected_flag: bool = False
    persons_data: Dict[int, Any] = field(default_factory=dict)
    aggregated: Dict[str, Any] = field(default_factory=dict)
    motion_state: str = "UNKNOWN"
    annotated_frame_for_activity: Any = None

    # Flat activity flags mutated by override / suppression stages
    activity_flags: Dict[str, bool] = field(default_factory=dict)
    activities_map: Dict[str, bool] = field(default_factory=dict)

    ocr_timestamp: Any = None
    no_person_detected_flag: bool = False
    sleep_state_overrides_writing: bool = False

    # Coordination results from GestureCoordinationStage
    lp_not_coordinating: bool = False
    alp_not_coordinating: bool = False


@runtime_checkable
class Stage(Protocol):
    """Protocol every pipeline stage must implement.

    Stages are constructed once (at monitor init time) and ``run`` is called
    once per sampled frame. ``run`` must return the (possibly mutated)
    ``FrameState`` so the pipeline can treat the chain as a pure reduction.
    """

    name: str

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        ...


class FramePipeline:
    """Ordered collection of ``Stage`` instances.

    A new stage can be added, removed, or reordered simply by editing the
    list passed to the constructor — no edits inside ``_process_frames_core``
    are required.
    """

    def __init__(self, stages: Iterable[Stage]) -> None:
        self.stages: List[Stage] = list(stages)

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        """Run every stage in order, threading ``state`` through each.

        Each stage is expected to return the same (mutated) ``FrameState``
        instance. If a stage returns ``None`` the previous state is kept so
        no-op stages can be written as minimal shims.
        """
        for stage in self.stages:
            result = stage.run(state, monitor)
            if result is not None:
                state = result
        return state

    def insert(self, index: int, stage: Stage) -> None:
        """Insert ``stage`` at ``index`` (convenience for tests + follow-ups)."""
        self.stages.insert(index, stage)

    def append(self, stage: Stage) -> None:
        """Append ``stage`` to the end of the pipeline."""
        self.stages.append(stage)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)
