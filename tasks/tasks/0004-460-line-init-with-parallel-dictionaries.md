# CR-004: 460-line `__init__` with 4 parallel dictionaries requiring synchronized updates

- **Severity:** High
- **Category:** Code Organization / Maintainability
- **Lines:** 150-611

## Description

The constructor is 461 lines long and maintains 4 parallel dictionaries (`activities`, `consecutive_detections`, `grace_counters`, `per_person_consecutive_detections`) that must be kept in sync for each activity type. Adding a new activity requires updating all dictionaries.

## Affected Code

```python
self.consecutive_detections = {
    'microsleep': 0, 'sleep': 0, 'cell_phone': 0, 'writing': 0,
    'packing_bags': 0, ...
}
self.grace_counters = {
    'microsleep': 0, 'sleep': 0, 'cell_phone': 0, 'writing': 0,
    'packing_bags': 0, ...
}
```

## Suggested Fix

Use a single `ActivityConfig` dataclass or registry pattern where each activity type auto-generates its tracking entries. A factory method can initialize all parallel dicts from a single source of truth.
