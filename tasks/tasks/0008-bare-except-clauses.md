# CR-008: Bare `except:` clauses catch `KeyboardInterrupt` and `SystemExit`

- **Severity:** High
- **Category:** Error Handling / Best Practice
- **Lines:** 2834, 2843, 2872, 2897

## Description

Bare `except:` without specifying an exception type catches all exceptions including `KeyboardInterrupt` and `SystemExit`, preventing the application from being interrupted or shut down gracefully.

## Affected Code

```python
try:
    cv2.line(annotated_frame, start_pt, end_pt, (0, 255, 255), 3)
except:
    continue
```

## Suggested Fix

Replace all bare `except:` with `except Exception:` at minimum. In drawing code, use `except (cv2.error, ValueError, TypeError):` for more specific handling.
