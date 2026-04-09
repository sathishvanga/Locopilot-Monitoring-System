# Task 0003: Widen `overlap_seconds` to cover baseline + coordination windows

- **Issue ID:** ARCH-03
- **Priority:** High-impact, low-effort (do first)
- **Severity:** HIGH — actively suppresses sleep detection at every chunk seam
- **Category:** Determinism / Chunk-boundary state
- **Files:**
  - `app/utils/video_multiprocessing.py:364` (`overlap_seconds: float = 2.0`)
  - `app/utils/video_multiprocessing.py:422-450` (overlap computation)
  - `app/utils/config.py` (add validator; `mp_chunk_duration`,
    `sleep_baseline_calibration_window`, `hand_gesture_coordination_window`)
  - `locopilot_monitor.py:~4384` (baseline maturity guard — `timestamp_sec < 10.0`)

## Description

The multiprocessing path relies on a 2-second overlap warm-up (C-02 fix) to
prime per-worker temporal state. But key windows are much larger:

- `sleep_baseline_calibration_window` = 10.0s (SleepDetector)
- `hand_gesture_coordination_window` ≈ 10.0s
- `sleep_head_drop_min_consecutive` / `sleep_state_machine` require multi-
  second state accumulation
- `_process_frames_core` has an explicit baseline maturity guard:
  head_drop is ignored when `timestamp_sec < 10.0` per memory note.

With `mp_chunk_duration=15.0s` and `overlap_seconds=2.0s`, each 15s chunk
spends ~10s in pre-baseline state. Pose-based sleep is effectively
suppressed for the first 10 of every 15 seconds when running in
multiprocessing mode — about 66% of chunk wall time.

Additionally, `consecutive_detections` and the sleep state machine
(`ALERT → DROWSY → MICROSLEEP → SLEEPING`) reset at every chunk boundary
because each chunk builds a fresh `LocopilotActivityMonitor`
(`video_multiprocessing.py:598`).

## Fix

Pick **option A** (simple) or **option B** (complete):

### Option A — Widen the warm-up window (preferred for first pass)

1. Change the default in `video_multiprocessing.py:364`:
   ```python
   overlap_seconds: float = 12.0  # covers baseline + coordination windows
   ```
2. Expose `mp_overlap_seconds` in `Settings` (`app/utils/config.py`) with a
   default of 12.0.
3. Add a `model_validator(mode='after')` to `Settings` that asserts:
   ```python
   assert self.mp_overlap_seconds >= max(
       self.sleep_baseline_calibration_window,
       self.hand_gesture_coordination_window,
   ), "mp_overlap_seconds must cover sleep baseline and gesture coordination"
   ```
4. Verify that the canonical-region activity filter in
   `video_multiprocessing.py:648-664` correctly discards warm-up activities —
   no additional change expected, but cover it with a test.

### Option B — Serialize `WorkerStateSnapshot` across chunks

Preferred if the performance cost of a 12s overlap is too high.

1. At the end of each worker chunk, capture:
   - `SleepDetector.per_person_tracking` (per-person history buffers)
   - `SleepDetector` baseline statistics
   - `consecutive_detections`, `grace_counters`, `activities` active state
   - Hand gesture coordination recent-raise timestamps
2. Pass the snapshot as input to the next chunk's worker via the orchestrator.
3. The orchestrator submits chunks in order (breaks parallelism within a
   video but preserves it across videos) **or** adds a post-pass stitching
   step that re-runs temporal filtering over the joined raw detections.

## Acceptance criteria

- [x] `overlap_seconds` default is ≥12.0 (option A) or `WorkerStateSnapshot`
      is serialized across chunks (option B).
- [x] `Settings` has a validator that fails startup if
      `mp_overlap_seconds < max(sleep_baseline_calibration_window, hand_gesture_coordination_window)`.
- [ ] On a representative 30-min video, the number of detected microsleeps
      is not lower than the single-process path by more than 5%.
- [x] A unit test exercises `calculate_frame_ranges` with the default
      overlap and asserts every non-first chunk has
      `overlap_start_frame < start_frame` by at least `12 * native_fps` frames.

## Implementation status

**Status:** Implemented on branch `feat/arch-review-2026-04/0003-widen-overlap-seconds`
(option A — widen warm-up window).  Not yet merged.

### Changes

- `app/utils/video_multiprocessing.py:calculate_frame_ranges`: default
  `overlap_seconds` raised from `2.0` to `12.0`; docstring updated to
  reference the sleep baseline + gesture coordination windows.  Function
  signature preserved so call-site overrides still work.
- `app/utils/multiprocessing_config.py`: `MultiprocessingConfig.overlap_seconds`
  is now `Optional[float] = None` and resolved in `__post_init__` from
  `Settings.mp_overlap_seconds` (matching the pattern already used for
  `chunk_duration_seconds`, `gpu_batch_size`, `gpu_batch_enabled`).  This
  removes the stale hard-coded `2.0` fallback and routes the `MP_OVERLAP_SECONDS`
  environment variable through Settings.
- `app/utils/config.py`:
  - New field `mp_overlap_seconds: float = 12.0` (reads `MP_OVERLAP_SECONDS`
    env var).
  - New `@model_validator(mode='after') _validate_overlap_window` that raises
    `ValueError` when `mp_overlap_seconds <
    max(sleep_baseline_calibration_window, hand_gesture_coordination_window)`.
  - Added `model_validator` to the pydantic import list.
- `.env.example`: documented `MP_OVERLAP_SECONDS=12.0` with the invariant note.
- `tests/unit/test_overlap_window.py` (new): 6 pytest cases covering
  (a) the 12-second overlap on non-first chunks, (b) the default-arg value,
  (c) validator rejection when overlap is below sleep window, (d) valid
  matching configuration, (e) boundary case (equality), (f) rejection when
  only the gesture window is violated.  All 6 pass locally under
  `.venv/bin/python -m pytest tests/unit/test_overlap_window.py -q`.

### Not performed here

- The 30-min representative-video microsleep comparison (third acceptance
  criterion) is a runtime/video-based benchmark outside the scope of a
  unit-test task and should be validated on the GPU server during rollout.
- Option B (per-chunk `WorkerStateSnapshot` serialization) is intentionally
  not implemented; the task spec selects option A.
- `mp_chunk_duration` was deliberately left at 15.0s per task constraints.
