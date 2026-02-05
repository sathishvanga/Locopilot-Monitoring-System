# CR-021: `locals()` check for variable deletion is fragile

- **Severity:** Low
- **Category:** Code Quality / Anti-Pattern
- **Lines:** 6664

## Description

Using `locals()` to check for variable existence is a fragile pattern that can break with Python compiler optimizations and is difficult to understand.

## Affected Code

```python
if 'variable_name' in locals():
    del variable_name
```

## Suggested Fix

Use explicit `None` initialization and check against `None`, or use a try/except `NameError` if variable existence is truly uncertain.
