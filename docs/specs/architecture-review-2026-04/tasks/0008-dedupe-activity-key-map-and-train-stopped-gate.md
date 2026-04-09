# Task 0008: Dedupe `activity_key_map` and apply train-STOPPED gate to `persons_data`

- **Issue ID:** ARCH-08
- **Priority:** Low-impact, low-effort
- **Severity:** LOW — latent bug + minor duplication
- **Category:** Rule layering / Duplication
- **Files:**
  - `locopilot_monitor.py:3637-3645` (first copy of `activity_key_map`)
  - `locopilot_monitor.py:3659-3667` (second copy, same function)
  - `locopilot_monitor.py:4725-4743` (train-STOPPED gate)

## Description

Two independent issues, grouped because the fixes are tiny.

### 8a — Copy-pasted `activity_key_map`

Inside `process_all_persons_activities`, the same `activity_key_map` dict
is defined in two adjacent blocks 14 lines apart:

```python
# locopilot_monitor.py:3637
activity_key_map = {
    'mind_diversion': 'mind_diversion',
    'cell_phone': 'cell_phone',
    'writing': 'writing',
    'packing_bags': 'packing_bags',
    'lp_hand_gesture': 'lp_hand_gesture',
    'alp_hand_gesture': 'alp_hand_gesture',
    'eating_drinking': 'eating_drinking',
}
```

and again at `3659-3667`. Adding a new activity (or renaming one) requires
updating both.

### 8b — Train-STOPPED gate skips `persons_data`

The train-STOPPED suppression block at
`locopilot_monitor.py:4725-4743` zeroes the **aggregated** booleans:

```python
sleep_detected = False
writing_detected = False
packing_detected = False
lp_hand_gesture_detected = False
alp_hand_gesture_detected = False
mind_diversion_detected = False
eating_drinking_detected = False
```

but it does **not** iterate `persons_data[pidx]['activities']` to zero the
per-person flags. Any downstream consumer that reads `persons_data`
(annotation, debug overlays, voting re-verification) sees stale positive
per-person flags — a latent bug that is only dormant because the current
downstream consumers happen to read only the aggregated values.

## Fix

### 8a

If task 0001 is already merged, delete both copies — per-activity voting
keys live in `ACTIVITY_REGISTRY`.

Otherwise, lift the map to a class constant:

```python
class LocopilotActivityMonitor:
    # ...
    VOTING_ACTIVITY_KEY_MAP = {
        'mind_diversion': 'mind_diversion',
        'cell_phone': 'cell_phone',
        'writing': 'writing',
        'packing_bags': 'packing_bags',
        'lp_hand_gesture': 'lp_hand_gesture',
        'alp_hand_gesture': 'alp_hand_gesture',
        'eating_drinking': 'eating_drinking',
    }
```

Replace both inline blocks with `self.VOTING_ACTIVITY_KEY_MAP`.

### 8b

After the train-STOPPED suppression block
(`locopilot_monitor.py:4725-4743`), also iterate `persons_data`:

```python
if self.train_motion_detector is not None and self.current_motion_state == "STOPPED":
    # ... existing aggregated zeroing ...
    SUPPRESSED_WHEN_STOPPED = {
        'sleep', 'writing', 'packing_bags',
        'lp_hand_gesture', 'alp_hand_gesture',
        'mind_diversion', 'eating_drinking',
    }
    for pidx, pdata in persons_data.items():
        for act in SUPPRESSED_WHEN_STOPPED:
            if act in pdata.get('activities', {}):
                pdata['activities'][act] = False
```

Keep `microsleep` and `cell_phone` active per spec (they are safety-critical).

## Acceptance criteria

- [x] `grep -n "activity_key_map" locopilot_monitor.py` finds at most one
      definition (or zero if task 0001 is merged).
- [x] After the train-STOPPED gate runs, every non-microsleep/non-cell-phone
      activity flag in both the aggregated booleans AND
      `persons_data[*]['activities']` is `False`.
- [x] A unit test constructs a fake `persons_data` + aggregated pair,
      runs the suppression block, and asserts both structures are
      zeroed consistently.

## Implementation status

**Branch:** `feat/arch-review-2026-04/0008-dedupe-activity-key-map`
**Date:** 2026-04-09

### 8a — DONE

Lifted `activity_key_map` to class constant `LocopilotActivityMonitor.VOTING_ACTIVITY_KEY_MAP`
(near the other class constants, line 281). Both inline copies inside
`process_all_persons_activities` (formerly at lines 3389 and 3411) now read
`self.VOTING_ACTIVITY_KEY_MAP`. `grep -n "activity_key_map"` now returns zero
matches in `locopilot_monitor.py`.

### 8b — DONE (helper + constant), NOT APPLICABLE inline

Commit `929272b refactor: Remove train motion state and rule engine services`
removed the entire train-motion subsystem from the monolith, so the
`locopilot_monitor.py:4725-4743` block this task referenced no longer
exists in this branch. There is no live "train-STOPPED gate" to patch
in-place — the latent bug it referenced is therefore dormant-by-absence.

To preserve the spirit of the fix and give any future reintroduction a
single safe entry point, implemented:

- `LocopilotActivityMonitor.SUPPRESSED_WHEN_STOPPED` class constant
  (frozenset; `microsleep` and `cell_phone` intentionally omitted as
  safety-critical).
- `app/core/gates.py` module with
  `DEFAULT_SUPPRESSED_WHEN_STOPPED` + `apply_train_stopped_suppression(aggregated, persons_data, suppressed=None)`.
  Pure function that zeroes both the aggregated flat booleans
  (`{name}_detected` or bare `{name}`) AND each
  `persons_data[pidx]['activities'][name]` consistently.

When the train-motion subsystem is reintroduced, callers should invoke
`apply_train_stopped_suppression(...)` instead of hand-rolling another
block that zeroes only the aggregated flags.

### Tests

`tests/unit/test_train_stopped_gate.py` — 7 tests, 6 passed, 1 skipped:

- `test_aggregated_flags_zeroed_except_safety_critical` PASSED
- `test_per_person_activity_flags_zeroed_except_safety_critical` PASSED
- `test_aggregated_and_per_person_are_consistent_after_gate` PASSED
- `test_handles_empty_persons_data` PASSED
- `test_handles_person_without_activities_dict` PASSED
- `test_custom_suppressed_set_override` PASSED
- `test_monitor_class_constants_match_helper_default` SKIPPED (mediapipe
  not installed in test env; test is a drift guard that auto-skips when
  `locopilot_monitor.py` cannot be fully imported — see `pytest.skip`
  fallback in the test body).

Run: `/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -m pytest tests/unit/test_train_stopped_gate.py -v`
