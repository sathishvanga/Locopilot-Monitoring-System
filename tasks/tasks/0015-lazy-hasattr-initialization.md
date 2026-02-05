# CR-015: Lazy `hasattr` initialization instead of declaring attributes in `__init__`

- **Severity:** Medium
- **Category:** Code Quality / Anti-Pattern
- **Lines:** 4001, 4780

## Description

Some attributes are lazily initialized using `hasattr` checks inside methods rather than being declared in `__init__`. This makes the class interface unpredictable and hides state.

## Affected Code

```python
if not hasattr(self, 'packing_motion_history'):   # Line 4001
    self.packing_motion_history = {}
if not hasattr(self, 'hand_smoothing_buffers'):    # Line 4780
    self.hand_smoothing_buffers = {}
```

## Suggested Fix

Declare all instance attributes in `__init__`, even if initialized to empty values. This makes the full state of the object visible in one place.
