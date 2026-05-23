# Task 0004 — Make `ActivityConfig` a view over `ACTIVITY_REGISTRY`

**Severity:** LOW (drift risk)
**Source:** Architecture review 2026-05-09, finding #8.
**Estimated effort:** 0.25 day.

---

## Problem

The system has two parallel definitions of per-activity tunables:

- **Canonical:** `ACTIVITY_REGISTRY` in `app/core/activity_registry.py:100–241` defines `min_duration`, `required_consecutive`, `grace_frames`, margins, etc. for each of the 13 activities.
- **Duplicate:** `ActivityConfig` dataclass in `app/core/activity_tracker.py:17–40` re-declares the same fields independently.

Two sources of truth → drift. A change in the registry that misses the dataclass causes silent behavior divergence.

`activity_detection_service.py:38–47` already re-derives metadata from the registry to keep things in sync — that indirection becomes unnecessary once the dataclass is a view.

---

## Files to change

- `app/core/activity_tracker.py` — replace the standalone dataclass with a view derived from the registry.
- `app/core/activity_registry.py` — no schema change; ensure all fields needed by `ActivityConfig` are present (audit).
- `app/services/activity_detection_service.py:38–47` — remove the re-derivation indirection if it becomes redundant.

---

## Fix

1. Audit every field used on `ActivityConfig` instances throughout the codebase (grep for `.min_duration`, `.required_consecutive`, `.grace_frames`, `.margin`, etc. on tracker objects). Make sure the registry has every one.
2. Replace `ActivityConfig` with a builder:
   ```python
   def build_activity_config(name: str) -> ActivityConfig:
       entry = ACTIVITY_REGISTRY[name]
       return ActivityConfig(
           name=name,
           min_duration=entry["min_duration"],
           required_consecutive=entry["required_consecutive"],
           grace_frames=entry["grace_frames"],
           # ... etc, every field
       )
   ```
   Or, better: drop the dataclass entirely and have `ActivityTracker` read from `ACTIVITY_REGISTRY` directly via a small accessor.
3. Anywhere a hard-coded `ActivityConfig(...)` appears, replace with `build_activity_config(name)`.
4. `activity_detection_service.py:38–47` — if its job was to re-derive registry data into config dicts, replace with a single `ACTIVITY_REGISTRY` import + lookup.
5. The choice between "dataclass view" and "delete dataclass" depends on how many call sites take an `ActivityConfig` instance. If <5 call sites and the dataclass is just bag-of-fields, delete it and read from the registry directly. If more, keep the dataclass as a view.

---

## Acceptance criteria

1. Editing a value in `ACTIVITY_REGISTRY` propagates to behavior with **no** edit to `activity_tracker.py`. Add a unit test that flips `min_duration` for one activity and asserts the tracker observes the new value.
2. `ActivityConfig` is no longer hand-defined with hard-coded thresholds.
3. `pytest tests/` is green.
4. `pytest tests/regression/` is green.

---

## Out of scope

- Changing any threshold values.
- Adding new activities to the registry.
- Changing `ActivityTracker`'s public interface.
