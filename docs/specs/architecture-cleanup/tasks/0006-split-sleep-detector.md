# Task 0006 — Split `sleep_detector.py` into a package

**Severity:** MEDIUM (god-file)
**Source:** Architecture review 2026-05-09, finding #5.
**Estimated effort:** 1 day.

---

## Problem

`app/core/detectors/sleep_detector.py` is **1,614 lines** in a single class. It contains five orthogonal concerns:

- Pose geometry helpers (head tilt, wrist distance, movement score, body movement, landmark validation).
- DROWSY state machine (`_update_sleep_state_machine`).
- Pose-based sleep detection (`detect_pose_based_sleep` — the main entry point).
- IR forward-lean fallback (`detect_ir_forward_lean_sleep`).
- Haar-cascade eye-closure (`detect_eye_closure_haar`).

This is the kind of file new contributors are afraid to touch. The detector has been the source of a wrap-around bug already (see `docs/specs/code-review-fixes/tasks/0001-restore-determinism-contract.md`), and it has the highest test-fixture footprint in the project.

---

## Files to change

**Replace `app/core/detectors/sleep_detector.py` with a package** (mirroring the pattern Wave1-B used for the VLM service):

```
app/core/detectors/sleep/
  __init__.py             # re-exports SleepDetector
  detector.py             # SleepDetector class: __init__, state, detect_pose_based_sleep,
                          # cleanup_stale_tracking, reset, reset_tracking, on_suppressed,
                          # get_tracking_state (~700 LOC)
  pose_geometry.py        # get_keypoint, validate_pose_landmarks, calculate_head_tilt_angle,
                          # calculate_movement_score, calculate_wrist_distance,
                          # _calculate_body_movement (~225 LOC)
  state_machine.py        # _update_sleep_state_machine + state constants (~100 LOC)
  ir_fallback.py          # detect_ir_forward_lean_sleep (~200 LOC)
  haar_eye_closure.py     # detect_eye_closure_haar + _load_haar_cascades (~200 LOC)
```

**Keep** `app/core/detectors/sleep_detector.py` as a 3-line shim using the **same `sys.modules` swap pattern as Wave1-B's `vlm_verification_service.py`**:

```python
import sys
from app.core.detectors.sleep import detector as _impl
sys.modules[__name__] = _impl
```

This preserves every existing import (`from app.core.detectors.sleep_detector import SleepDetector`) and any `monkeypatch.setattr(sleep_detector, "_private_helper", ...)` that tests may use.

**Update:**
- `app/core/detectors/__init__.py` — its `from app.core.detectors.sleep_detector import SleepDetector` keeps working unchanged through the shim. Verify only.

**DO NOT touch:**
- `locopilot_monitor.py` — it imports `SleepDetector` from `app.core.detectors`; the import resolves the same way.
- Any other detector file. Sleep is the only detector being split in this task.
- `tests/detectors/test_reset.py` and any other sleep-related tests — they should keep passing unmodified.

---

## Splitting rules (read carefully)

This is a **MOVE**, not a refactor. Function bodies copy verbatim into their new homes. Karpathy guidelines apply: no renaming, no logic changes, no "small improvements," no docstring rewrites, no behavior tweaks. If you find a bug while moving, leave it; flag it in your report.

**Cross-file references inside the package:**
- `pose_geometry.py` exports pure functions. `detector.py` imports them: `from .pose_geometry import calculate_head_tilt_angle, calculate_movement_score, ...`.
- `state_machine.py` exports `update_sleep_state_machine(tracking_dict, ...)` as a module-level function. `detector.py` calls it from inside its method.
- `ir_fallback.py` and `haar_eye_closure.py` similarly export module-level functions that take a `SleepDetector` instance (or its tracking dict) as the first argument. The `SleepDetector` methods become thin shims that call these functions.
- The `_init_thresholds` and `_create_tracking_dict` private methods stay inside `SleepDetector` — they're tied to `self`.

**Constants:**
- DROWSY state strings live in `state_machine.py`.
- Threshold default values (yaw, pitch, head-tilt, etc.) stay in `_init_thresholds` (i.e., on the instance), as today.

---

## Acceptance criteria

1. `pytest tests/detectors/` is green.
2. `pytest tests/detectors/test_reset.py` is green.
3. `pytest tests/regression/` is green — behavior on real-video fixtures preserved.
4. `pytest tests/` is green overall.
5. `from app.core.detectors.sleep_detector import SleepDetector` still works.
6. `from app.core.detectors import SleepDetector` still works.
7. `python -c "from locopilot_monitor import LocopilotActivityMonitor"` still works.
8. Each file under `app/core/detectors/sleep/` is < 800 LOC. (`detector.py` may approach 800 because `detect_pose_based_sleep` is ~575 LOC and stays whole.)
9. `wc -l app/core/detectors/sleep/*.py` totals close to 1,614 (the original size). This is a structural split, not a code reduction.

---

## Out of scope

- Refactoring `detect_pose_based_sleep` (~575 LOC). It stays whole inside `detector.py`.
- Splitting any other detector (`activity_detector.py`, `gesture_detector.py`, `mind_diversion_detector.py`).
- Changing any threshold value, any pose computation, any state-machine transition.
- Fixing any bug discovered during the move — flag it in the completion report instead.
- Writing new tests. Existing test surface is the contract.
