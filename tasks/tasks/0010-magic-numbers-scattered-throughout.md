# CR-010: 50+ magic numbers scattered throughout the codebase

- **Severity:** High
- **Category:** Code Quality / Maintainability
- **Lines:** 2084, 2091, 2109, 1486, 1503, 3507, 1229-1264, 1381-1398, 1618-1622, and many more

## Description

Confidence thresholds, distance thresholds, pixel margins, duration windows, and other numeric constants are hardcoded directly in the logic without named constants or configuration.

## Affected Code

```python
MAX_WRIST_DISTANCE = 300       # Line 1618
MAX_ELBOW_DISTANCE = 450       # Line 1619
WRITING_WRIST_DISTANCE = 300   # Line 1486
HEAD_DOWN_THRESHOLD = 0.01     # Line 1538
```

## Suggested Fix

Extract all magic numbers into named constants at the class level or into a dedicated configuration file/dataclass. Group them by detection type (sleep, writing, packing, etc.).
