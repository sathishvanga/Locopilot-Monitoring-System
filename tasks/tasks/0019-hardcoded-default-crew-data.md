# CR-019: Hardcoded default crew data could leak into production evidence

- **Severity:** Low
- **Category:** Security / Data Integrity
- **Lines:** 525-528

## Description

Default crew member data is hardcoded (e.g., `"John Doe"`, `"C-001"`, `"TRIP-123"`). If the API override fails to provide real crew data, these dummy values will appear in production evidence records.

## Affected Code

```python
self.trip_id = "TRIP-123"
self.crew_name = "John Doe"
self.crew_id = "C-001"
self.crew_role = 1
```

## Suggested Fix

Use `None` or sentinel values as defaults and validate that real crew data is provided before generating evidence. Raise an error or log a warning if defaults are still in place when evidence is created.
