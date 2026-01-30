# CR-016: Repeated `import` statements inside methods

- **Severity:** Medium
- **Category:** Code Quality / Performance
- **Lines:** 1864, 1964, 2168, 3071

## Description

Several methods contain `import logging` and `import time` statements inside their bodies rather than at module level. While Python caches imports, this is an anti-pattern that adds overhead and clutters method bodies.

## Affected Code

```python
def detect_objects_in_roi(self, ...):
    import logging    # Line 1864
def detect_objects_in_rois_batch(self, ...):
    import logging    # Line 1964
def detect_objects_batch(self, ...):
    import logging    # Line 2168
    import time
```

## Suggested Fix

Move all imports to the top of the file at module level.
