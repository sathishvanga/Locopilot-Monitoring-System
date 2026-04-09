# Task 0002: Extract `_process_frames_core` into an ordered `FramePipeline`

- **Issue ID:** ARCH-02
- **Priority:** High-impact, high-effort
- **Severity:** HIGH — largest source of ongoing regression risk
- **Category:** Module boundaries / Rule layering
- **Files:**
  - `locopilot_monitor.py:4345-4865` (`_process_frames_core`, ~520 lines)
  - `locopilot_monitor.py:2762-3799` (`process_all_persons_activities`, ~1037 lines)
  - `locopilot_monitor.py:4579-4648` (secondary state-machine re-gate + sleep vs writing override)
  - `locopilot_monitor.py:4704-4743` (gesture coordination + train-STOPPED gate)

## Description

`_process_frames_core` is a 520-line monolithic function that interleaves
face mesh, YOLO object detect, person dedup, role identification, group
voting, train motion detection, per-person activity dispatch, per-person
state-machine re-gate, writing suppression, gesture coordination, OCR,
train-STOPPED suppression, temporal filtering, and evidence writing — all in
line. Any change to the gating order risks regressions and is not testable
today (no unit tests exist; see task 0006).

The ordering is also **partially duplicated** with
`process_all_persons_activities` which itself contains per-person voting,
writing suppression, and gesture logic.

## Fix

Introduce a `FramePipeline` in `app/core/frame_pipeline.py` composed of
ordered, typed `Stage` callables operating on a typed `FrameState`
dataclass:

```python
@dataclass
class FrameState:
    frame: np.ndarray
    frame_idx: int
    timestamp_sec: float
    sample_idx: int
    fps: float
    # populated as stages run:
    face_results: Any = None
    detections: Dict[str, Any] = field(default_factory=dict)
    person_roles: Dict[int, Any] = field(default_factory=dict)
    persons_data: Dict[int, Any] = field(default_factory=dict)
    aggregated: Dict[str, bool] = field(default_factory=dict)
    motion_state: str = "UNKNOWN"
    activities_map: Dict[str, bool] = field(default_factory=dict)
```

Stages (one class each, in `app/core/pipeline/stages/`):

1. `FaceMeshStage` — runs MediaPipe face mesh on the RGB frame.
2. `ObjectDetectStage` — YOLO object detection (batched or per-frame).
3. `PersonDedupStage` — `deduplicate_person_boxes`.
4. `RoleIdentifyStage` — `identify_person_roles`.
5. `GroupVoteStage` — group detection + voting (currently
   `locopilot_monitor.py:4453-4465`).
6. `TrainMotionDetectStage` — vibration-based motion detection.
7. `PerPersonActivitiesStage` — wraps `process_all_persons_activities`.
   (This stage can be further decomposed per-activity as a follow-up; the
   first pass just wraps the existing method.)
8. `StateMachineGateStage` — per-person sleep state-machine re-gate
   (currently `4579-4596`).
9. `SleepWritingOverrideStage` — the 40-line override block at `4608-4648`.
10. `GestureCoordinationStage` — `_check_hand_gesture_coordination`.
11. `TrainMotionSuppressStage` — the train-STOPPED gate at `4725-4743`
    (see task 0008 — must also zero `persons_data[*]['activities']`).
12. `NoPersonScheduleSuppressStage` — the no-person-without-schedule gate
    at `4762`.
13. `TemporalFilterStage` — consecutive counters + grace (`4770-4834`).
14. `EvidenceStage` — `start_activity`/`end_activity` bookkeeping.

`_process_frames_core` becomes:

```python
def _process_frames_core(self, ...):
    state = FrameState(...)
    for stage in self.frame_pipeline.stages:
        state = stage.run(state, monitor=self)
    return state
```

## Acceptance criteria

- [x] `app/core/frame_pipeline.py` exists with `FrameState` dataclass and
      `FramePipeline` class.
- [x] All 14 stages exist under `app/core/pipeline/stages/` as individual
      files, each with a `run(state, monitor) -> FrameState` method.
- [x] `_process_frames_core` is ≤60 lines — only constructs state and
      iterates stages. (Now 55 lines including signature + docstring.)
- [ ] Output on a representative video is byte-identical to pre-refactor
      `activities.json` on the same input (snapshot test — see task 0006).
      **DEFERRED**: no GPU / models available in this environment. All
      stages were extracted verbatim (move-only; no logic rewrites), and
      the monitor still imports and instantiates cleanly, but a proper
      byte-identical verification is blocked on task 0006.
- [x] Each stage can be imported and called without constructing a full
      `LocopilotActivityMonitor` (stages take the monitor as a parameter,
      so the contract is explicit and replaceable with a stub in tests).
      Verified by `tests/unit/test_frame_pipeline.py::
      test_all_14_stages_importable_and_constructible`.
- [x] A new stage can be added or reordered by editing `FramePipeline.stages`
      — no edits inside `_process_frames_core`. `FramePipeline` also
      exposes `append()` and `insert()` convenience methods.

## Implementation status

**Branch:** `feat/arch-review-2026-04/0002-frame-pipeline-extraction`

**Scope of this pass:** scaffolding + cutover only. Per the task brief,
this is explicitly *not* a per-activity decomposition — that is a
follow-up. ``PerPersonActivitiesStage`` wraps ``monitor.process_all_persons_activities``
as a single stage (the 1037-line per-person dispatcher is otherwise
untouched).

**Stages extracted (all 14):**

| # | Stage | Original lines | Notes |
|---|-------|----------------|-------|
| 1 | `FaceMeshStage` | 4387-4398 | Includes frame-buffer bookkeeping. |
| 2 | `ObjectDetectStage` | 4400-4411 | Batch + per-frame fallback paths. |
| 3 | `PersonDedupStage` | 4413-4477 | Sets `deduplicated_person` + no-person log. |
| 4 | `RoleIdentifyStage` | 4440-4448 | Role log is suppressed after first detection cycle (unchanged). |
| 5 | `GroupVoteStage` | 4450-4469 | Voting verification + group-count threshold. |
| 6 | `TrainMotionDetectStage` | 4479-4494 | Vibration-based motion detection. |
| 7 | `PerPersonActivitiesStage` | 4496-4572 | Thin wrapper around `process_all_persons_activities` + per-person logging. |
| 8 | `StateMachineGateStage` | 4573-4596 | H-02 per-person sleep state-machine re-gate. |
| 9 | `SleepWritingOverrideStage` | 4598-4697 | Includes annotated-frame rendering + periodic frame save (kept together because they share the same `annotated_frame_for_activity` variable that used to be local to the function). |
| 10 | `GestureCoordinationStage` | 4699-4720 | Also handles `no_person_detected_flag` and OCR timestamp extraction (same contiguous region). |
| 11 | `TrainMotionSuppressStage` | 4722-4743 | Train-STOPPED gate. Task 0008's "zero `persons_data[*]['activities']`" change left as a TODO to keep this pass move-only. |
| 12 | `NoPersonScheduleSuppressStage` | 4745-4768 | Builds `activities_map` + trip-schedule suppression. |
| 13 | `TemporalFilterStage` | 4770-4834 | Consecutive counters + grace + `start_activity` / `end_activity` bookkeeping. |
| 14 | `EvidenceStage` | 4836-4854 | Stale-person cleanup + progress logging. |

**Files created:**

- `app/core/frame_pipeline.py`
- `app/core/pipeline/__init__.py`
- `app/core/pipeline/stages/__init__.py`
- `app/core/pipeline/stages/face_mesh_stage.py`
- `app/core/pipeline/stages/object_detect_stage.py`
- `app/core/pipeline/stages/person_dedup_stage.py`
- `app/core/pipeline/stages/role_identify_stage.py`
- `app/core/pipeline/stages/group_vote_stage.py`
- `app/core/pipeline/stages/train_motion_detect_stage.py`
- `app/core/pipeline/stages/per_person_activities_stage.py`
- `app/core/pipeline/stages/state_machine_gate_stage.py`
- `app/core/pipeline/stages/sleep_writing_override_stage.py`
- `app/core/pipeline/stages/gesture_coordination_stage.py`
- `app/core/pipeline/stages/train_motion_suppress_stage.py`
- `app/core/pipeline/stages/no_person_schedule_suppress_stage.py`
- `app/core/pipeline/stages/temporal_filter_stage.py`
- `app/core/pipeline/stages/evidence_stage.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/unit/test_frame_pipeline.py`

**Files changed:**

- `locopilot_monitor.py`
  - Added imports for `FrameState`, `FramePipeline`, and all 14 stages.
  - Added `self.frame_pipeline = FramePipeline([...])` in `__init__`.
  - Rewrote `_process_frames_core` from ~520 lines to 55 lines — the
    body now only constructs a `FrameState` and iterates
    `self.frame_pipeline.stages`.

**Tests:** `pytest tests/unit/test_frame_pipeline.py -q` → **9 passed**.

**Behavioral notes / known subtleties preserved verbatim:**

1. In the original `_process_frames_core`, when the train is STOPPED the
   `lp_hand_gesture_detected` / `alp_hand_gesture_detected` flags are
   zeroed **but** `lp_not_coordinating` / `alp_not_coordinating`
   (computed earlier) are **not**. This means the coordination-failure
   flags can still flow into `activities_map` after train-stopped
   suppression. `TrainMotionSuppressStage` preserves this quirk exactly
   — a fix is tagged as a TODO for task 0008.
2. The annotated-frame rendering block (`draw_bounding_boxes` + multi
   person mediapipe + sleep debug overlay + periodic frame save) was
   kept inside `SleepWritingOverrideStage` rather than split into a
   separate `RenderStage` because it shares the
   `annotated_frame_for_activity` variable with the override block in
   the original function. Splitting it is straightforward follow-up
   work once the snapshot test is in place.
3. Per-person detection logging (mind-diversion / sleep / microsleep /
   gestures / phone / writing / packing) lives inside
   `PerPersonActivitiesStage` — the logging branches are gated by
   `state.log_per_person_detections` exactly as in the original.

**Deferred / blockers:**

- **Byte-identical snapshot test:** blocked on task 0006. No GPU or
  YOLO weights available in this environment, so the only validation
  done here is (a) module imports clean, (b) unit tests for the pipeline
  scaffolding, and (c) structural `py_compile` on every touched file.
- **Further per-activity decomposition** of
  `PerPersonActivitiesStage`: deliberately out of scope for this pass
  per the task brief ("Wrap `process_all_persons_activities` as a
  single stage — further decomposition is a follow-up").
