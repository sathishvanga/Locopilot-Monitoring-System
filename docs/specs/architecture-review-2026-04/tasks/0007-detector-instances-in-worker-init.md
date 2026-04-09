# Task 0007: Move detector construction into `worker_initializer`

- **Issue ID:** ARCH-07
- **Priority:** Medium-impact, medium-effort
- **Severity:** MEDIUM — per-chunk re-initialization overhead
- **Category:** Performance
- **Files:**
  - `app/utils/video_multiprocessing.py:119-295` (`worker_initializer`,
    `_worker_models` dict)
  - `app/utils/video_multiprocessing.py:598-607` (per-chunk
    `LocopilotActivityMonitor` construction)
  - `locopilot_monitor.py:283-864` (`LocopilotActivityMonitor.__init__`)
  - `locopilot_monitor.py:811-849` (current detector instantiation block)

## Description

`worker_initializer` correctly pre-loads YOLO and YOLO-Pose weights once
per worker process (video_multiprocessing.py:159-289), which is good.

But `LocopilotActivityMonitor.__init__` then re-creates the following
**every chunk** (every 15s of video):

- `SleepDetector` (with fresh `per_person_tracking` dict + Haar cascade
  loads)
- `GestureDetector`
- `MindDiversionDetector`
- `ActivityDetector`
- `VotingVerificationService`
- `OCRTimestampService`
- `EvidenceManager`
- `PersonTracker`
- `TrainMotionDetector` (when enabled)
- `FrameAnnotator`
- `ImagePreprocessingService` (subset — the worker preloads one but the
  monitor can re-wrap it)

None of these hold heavy model weights, but all do some work on
construction (Haar XML loads, dict allocations, service config reads). At
one monitor rebuild per 15s chunk over a 30-minute video that's ~120
rebuilds per worker.

There's also a cleaner architectural benefit: once detectors are on
`_worker_models`, unit tests (task 0006) can exercise them without any
monitor machinery.

## Fix

1. Extend `worker_initializer` in
   `app/utils/video_multiprocessing.py:236-290` to also construct:

   ```python
   from app.core.detectors import (
       SleepDetector, GestureDetector, MindDiversionDetector,
       ActivityDetector, ObjectDetector,
   )
   from app.services.voting_verification_service import VotingVerificationService

   worker_settings = get_settings()

   _worker_models['sleep_detector'] = SleepDetector(
       settings=worker_settings,
       sample_fps=worker_settings.sample_fps,
       logger=logger,
   )
   _worker_models['gesture_detector'] = GestureDetector(
       settings=worker_settings,
       ...
   )
   _worker_models['mind_diversion_detector'] = MindDiversionDetector(
       settings=worker_settings, logger=logger,
   )
   _worker_models['voting_service'] = VotingVerificationService(
       yolo_model=_worker_models['yolo_voting'],
       yolo_pose_model=_worker_models['yolo_pose_voting'],
       yolo_roi_model=_worker_models['yolo_roi'],
   )
   ```

2. Modify `LocopilotActivityMonitor.__init__` (`locopilot_monitor.py:283`)
   to accept a `preloaded_models` dict **that already contains detectors**
   and reuse them when present:

   ```python
   self.sleep_detector = (
       preloaded_models.get('sleep_detector')
       if preloaded_models else None
   ) or SleepDetector(settings=self.settings, ...)
   ```

3. Critical: **per-chunk state reset.** `SleepDetector.per_person_tracking`,
   `consecutive_detections`, `grace_counters`, and sleep baseline state
   are currently implicitly reset by constructor allocation. Now that the
   detector is long-lived, add explicit reset calls:
   - Either reset at the start of each chunk (loses the warm-up benefit).
   - Or — far better, combined with task 0003 — keep the state **across**
     chunks for the same worker so temporal continuity actually improves.

4. Document the decision in the constructor docstring: "state persists
   across chunks within the same worker."

## Acceptance criteria

- [x] `worker_initializer` constructs `SleepDetector`, `GestureDetector`,
      `MindDiversionDetector`, `ActivityDetector` exactly once per
      worker process.  **`VotingVerificationService` intentionally
      deferred** — it depends on the VideoReader changes planned in
      task 0004 and moving it now would produce churn that has to be
      re-done.  A `TODO(task-0004)` marker is left in
      `app/utils/video_multiprocessing.py` at the detector-preload
      block.
- [x] `LocopilotActivityMonitor.__init__` reuses preloaded detector
      instances when present instead of constructing new ones
      (`locopilot_monitor.py` ~lines 781-835).  Uses the
      ``preloaded = (preloaded_models.get('sleep_detector') if preloaded_models else None) or SleepDetector(...)``
      pattern so the single-process path still constructs fresh
      detectors when ``preloaded_models`` is ``None``.
- [x] A single log line on every chunk reports reuse and which
      detectors came from the worker pool
      (``Using pre-loaded detectors from worker pool ...``).  The
      original spec called for different wording between first and
      subsequent chunks — the monitor doesn't know which chunk it is,
      so we emit one consistent message per reuse instead.  The
      `worker_initializer` itself logs a separate one-shot
      ``Worker <pid> pre-loaded detectors: ...`` line exactly once per
      worker.
- [x] Detector state behavior is explicitly documented in the
      `LocopilotActivityMonitor.__init__` docstring: **state RESETS per
      chunk when reusing pre-loaded detectors.**  `SleepDetector
      .reset_tracking()` and `GestureDetector.reset()` are invoked
      during `__init__` whenever their respective preloaded instance is
      reused.  Task 0003 widens the chunk overlap so baseline
      calibration still warms up inside each chunk.
- [ ] Activity counts on a representative 30-min video are within ±5%
      of the single-process path (regression guard).  **Not validated
      in this pass** — the worktree has no GPU access and the task
      constraints forbid running the full pipeline.  This needs a
      manual GPU-server regression run after the branch lands.

## Implementation status

**Status:** IMPLEMENTED (pending regression guard on GPU server)

**Branch:** `feat/arch-review-2026-04/0007-detector-instances-in-worker-init`

**Files changed:**
- `app/utils/video_multiprocessing.py` — extends `worker_initializer`
  to construct the four detector instances after YOLO/face-mesh
  preload and store them on `_worker_models`.  Wraps the detector
  preload in its own try/except so a detector-construction failure
  falls back to per-chunk construction instead of killing the worker.
- `locopilot_monitor.py` — `LocopilotActivityMonitor.__init__`
  reuses preloaded detector instances when present and resets their
  per-chunk state via `SleepDetector.reset_tracking()` and
  `GestureDetector.reset()`.  Docstring updated to document the
  state-reset contract.
- `tests/unit/test_preloaded_detectors.py` — four unit tests:
  1. `test_worker_initializer_stores_detectors_on_worker_models` —
     runs `worker_initializer` with mocked YOLO/FaceMesh/adapter and
     asserts the four detector keys are populated on `_worker_models`
     with the right types.
  2. `test_monitor_creates_fresh_detectors_when_preloaded_is_none` —
     constructs the monitor with `preloaded_models=None` and asserts
     each detector is an instance of the real class.
  3. `test_monitor_reuses_preloaded_detectors_when_present` — passes
     MagicMock stubs through `preloaded_models` and asserts the
     monitor stores those exact instances, plus that
     `sleep.reset_tracking()` and `gesture.reset()` are called during
     `__init__`.
  4. `test_real_sleep_detector_state_cleared_after_chunk_reset` —
     passes a real `SleepDetector` with seeded
     `per_person_tracking`/`ir_forward_lean_tracking` state through
     the monitor and asserts both dicts are empty after the reuse
     path runs.
- `docs/specs/architecture-review-2026-04/tasks/0007-detector-instances-in-worker-init.md`
  — this file (implementation status).

**Test run:** `python -m pytest tests/unit/test_preloaded_detectors.py -q`
→ 4 passed.

**Follow-ups:**
- Task 0004: move `VotingVerificationService` into `worker_initializer`
  once its VideoReader contract is stable.
- Manual regression run on a 30-minute video (multiprocessing vs
  single-process) to confirm activity-count parity within ±5%.
