# Task 0006 — Detector reset contract + train-stopped invariant

**Severity:** CRITICAL
**Source:** `docs/code-review-2026-05-08.md` cross-cutting themes #3 + #4, top-fix #6.
**Estimated effort:** Half day.

---

## Problem

### 1. Detector state leaks across video boundaries

None of the extracted detectors (`sleep`, `gesture`, `mind_diversion`, `activity`) have a `reset()` contract enforced by `video_processing_service`. **`train_motion_detector` has no reset method at all** — its `prev_gray` is diffed against the last frame of the previous video, producing a phantom RUNNING signal at the start of every video after the first.

State that leaks (per review):
- `sleep_detector.per_person_tracking`, `ir_forward_lean_tracking`
- `gesture_detector.gesture_sessions`, `recent_person_activities`, `hand_position_history`
- `mind_diversion_detector._recent_person_activities`
- `activity_detector.packing_motion_history`
- `train_motion_detector.prev_gray`, `state_history`, `_prev_block_vars`, `person_bbox_history_buf`, `_vib_history`, `_frames_seen`, `_prev_block_vars`, `prev_gray_window`

### 2. Train-stopped gate is an honor system

`gates.apply_train_stopped_suppression` is the only enforcement point, but every detector keeps its own state machines advancing while the activity is suppressed. When the train resumes, internal counters are already mature — instant FP violations on resume:

- Sleep: `pose_sleep_duration` keeps growing while suppressed (`sleep_detector.py:1080`).
- Gesture: `lp_last_raise_time` persists across STOPPED windows; a single ALP raise can fire `lp_not_coordinating` against a 30-min-stale LP raise (`gesture_detector.py:530-572`).
- Packing: `packing_motion_history` direction-changes accumulate (`activity_detector.py:537-541`).
- Writing: grace-period `recent_person_activities[idx]['writing']` is set even when the writing event itself was suppressed (`mind_diversion_detector.py:514`).

Additionally, `app/core/pipeline/stages/train_motion_suppress_stage.py:40-77` does NOT call `apply_train_stopped_suppression` despite gates.py existing for exactly this purpose. There is even an explicit TODO at `:74-76`.

---

## Files to change

- `app/core/detectors/train_motion_detector.py` — add `reset()` method
- `app/core/detectors/sleep_detector.py` — verify `reset_tracking()` clears all six dicts
- `app/core/detectors/gesture_detector.py` — add `reset()`
- `app/core/detectors/mind_diversion_detector.py` — add `reset()`
- `app/core/detectors/activity_detector.py` — add `reset()`
- `app/services/video_processing_service.py:150-477` — call all detector resets at video start
- `app/core/gates.py` — extend `apply_train_stopped_suppression` to also reset per-detector state
- `app/core/pipeline/stages/train_motion_suppress_stage.py:40-77` — replace manual flag-zeroing with `apply_train_stopped_suppression` call; remove TODO

---

## Fix

### TrainMotionDetector.reset

```python
def reset(self) -> None:
    """Clear all per-video state. MUST be called between videos."""
    self.prev_gray = None
    self.prev_gray_window = None
    self.state_history.clear()
    self._prev_block_vars = None
    self.person_bbox_history_buf.clear()
    self._vib_history.clear()
    self._frames_seen = 0
```

### Other detectors

Each detector that holds per-video state grows a `reset()` method clearing the documented dicts/deques. Sleep already has `reset_tracking()` — confirm coverage and rename for symmetry if needed.

### `video_processing_service.process_video` start-of-video block

```python
self._detectors["train_motion"].reset()
self._detectors["sleep"].reset()
self._detectors["gesture"].reset()
self._detectors["mind_diversion"].reset()
self._detectors["activity"].reset()
```

### Gate extension

`gates.apply_train_stopped_suppression` currently zeros aggregated flags. Extend it to also call a per-detector `on_suppressed(person_idx, activity_name)` hook that resets the relevant state machine counters when an activity is suppressed by the train-stopped gate. This keeps detectors pure (no `train_state` parameter) while preventing counter maturation.

### Replace `train_motion_suppress_stage` body

```python
from app.core.gates import apply_train_stopped_suppression

def process(self, state: FrameState) -> FrameState:
    if state.train_motion_state == "STOPPED":
        apply_train_stopped_suppression(state.activity_flags, state.persons_data)
    return state
```

Remove the TODO at lines 74-76.

---

## Acceptance criteria

1. `tests/detectors/test_reset.py`:
   - For each detector, instantiate, run 100 frames of video A, call `reset()`, run 1 frame of video B, assert internal state is identical to a fresh instance running that same frame of video B.
2. `tests/detectors/test_train_motion_first_frame.py`:
   - First frame of video B (after reset) does NOT report RUNNING when the actual frame is static.
3. `tests/test_train_stopped_resume.py`:
   - Sequence: 30 frames RUNNING with sleep flagged, 30 frames STOPPED, 30 frames RUNNING with no sleep cue. Assert NO sleep activity is emitted in the third segment.
4. `grep -n "TODO" app/core/pipeline/stages/train_motion_suppress_stage.py` returns no hits.
5. Existing ground-truth regression suite does not degrade.

---

## Out of scope

- Refactoring detectors to be fully stateless (would be a much larger task).
- Wiring `train_state` into every detector's main entry method.
