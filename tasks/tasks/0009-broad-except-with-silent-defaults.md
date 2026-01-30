# CR-009: Broad `except Exception` with silent defaults in 10+ methods

- **Severity:** High
- **Category:** Error Handling / Debugging
- **Lines:** 949, 979, 1015, 1509 (and 30+ more locations)

## Description

Over 30 methods catch `except Exception` and silently return default values (`None`, `False`, `0.0`) without logging, making debugging extremely difficult. Failures are invisible.

## Affected Code

```python
except Exception as e:
    return None   # Line 949 - calculate_eye_aspect_ratio
except Exception as e:
    return None   # Line 979 - calculate_head_tilt_angle
except Exception as e:
    return 0.0    # Line 1015 - calculate_movement_score
except Exception as e:
    return False   # Line 1509 - detect_writing_posture
```

## Suggested Fix

Add `self.logger.debug()` or `self.logger.warning()` calls inside all exception handlers. Consider narrowing exception types to expected failures (e.g., `IndexError`, `ValueError`).
