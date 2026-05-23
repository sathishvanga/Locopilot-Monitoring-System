# Task 0001 — Restore the two-pass determinism contract

**Severity:** CRITICAL
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #1, top-fix #1.
**Estimated effort:** 1 day.

---

## Problem

CLAUDE.md documents a "Pass-1 parallel chunked workers, then Pass-2 sequential temporal filter (deterministic two-pass)" guarantee. The review found **two distinct ways the multiprocess (production) path produces different activities than the single-frame path for the same video**:

1. **`YOLOHandler.detect_objects_batch` is missing the `cup_bottle` handler entirely** (`app/core/models/yolo_handler.py:862-919` vs single-frame `:316-319`). When `useMultiprocessing=true`, `eating_drinking` activity loses its primary signal and falls back to ROI detection only.
2. **`detect_objects_batch` book-near-person uses hard-coded `margin=200`** (`app/core/models/yolo_handler.py:911-917`), but the single-frame path uses the configurable `self.book_person_margin` (default 150). Multiprocess runs surface more `writing` activities than serial runs.

Separately, in the extracted sleep detector:

3. **The `atan2` wrap-around fix documented in CLAUDE.md is lost** (`app/core/detectors/sleep_detector.py:344-350`). Original monolith had `(delta + 180) % 360 - 180` plus a `nose_y_drop >= 0` guard; neither is present in the extracted detector. A wrap from `+170°` to `-170°` produces `delta = -340°` for one frame — easily exceeding the 30° drop threshold and firing a false sleep "head drop". The same bug exists in the delta computation at line 730 and in `head_tilt_drop` at line 986.

---

## Files to change

- `app/core/models/yolo_handler.py:862-919` — `detect_objects_batch`
- `app/core/detectors/sleep_detector.py:344-350, 730, 986` — head-tilt math

---

## Fix

### YOLOHandler.detect_objects_batch

1. Initialize `detections['cup_bottle'] = []` alongside the other class keys.
2. Mirror the cup/bottle handler from the single-frame `detect_objects` (lines 316-319) inside the batch loop.
3. Replace the hard-coded `margin=200` at line 911 with `margin=self.book_person_margin`.

### sleep_detector.calculate_head_tilt_angle

```python
angle = np.arctan2(delta_y, delta_x) * 180 / np.pi - 90
angle = (angle + 180) % 360 - 180  # normalize to [-180, 180]
```

### sleep_detector — delta normalization (line ~730)

```python
delta = tilt_list[-1] - tilt_list[-2]
delta = (delta + 180) % 360 - 180
```

### sleep_detector — head_tilt_drop (line ~986)

```python
head_tilt_drop = (head_tilt - baseline_head_tilt + 180) % 360 - 180
```

Add `if nose_y_drop < 0: return  # only count downward drops` guard at line 982.

---

## Acceptance criteria

1. Add a regression test `tests/test_determinism.py` that runs the same video through serial (`useMultiprocessing=false`) and parallel paths, then asserts `sha256(serial_activities_json) == sha256(parallel_activities_json)` after stripping wall-clock fields.
2. Add a unit test in `tests/detectors/test_sleep_detector.py` that synthesizes a frame sequence with head tilt walking from `+170°` to `-170°` over 5 frames and asserts `head_drop_detected` stays False.
3. Existing ground-truth precision/recall regression suite (`tests/ground_truth/`) does not degrade.

---

## Out of scope

- Refactoring `detect_objects_batch` into smaller methods.
- Splitting `sleep_detector.py` (separate task).
