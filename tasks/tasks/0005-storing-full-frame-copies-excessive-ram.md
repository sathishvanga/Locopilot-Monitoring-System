# CR-005: Storing full `frame.copy()` for every detected activity frame consumes excessive RAM

- **Severity:** High
- **Category:** Performance / Memory
- **Lines:** 6612, 7078

## Description

Full frame copies are stored in memory for every detected activity frame. On long videos, this can consume gigabytes of RAM.

## Affected Code

```python
self.activities[activity]['frames'].append(frame.copy())
```

## Suggested Fix

Store frame indices instead of frame copies and extract frames from the video on demand when saving evidence.
