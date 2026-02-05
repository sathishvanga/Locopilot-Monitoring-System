# CR-014: `per_person_consecutive_detections` only tracks 3 of 10+ activity types

- **Severity:** Medium
- **Category:** Bug / KeyError Risk
- **Lines:** 841

## Description

The per-person consecutive detection dictionary only initializes keys for `cell_phone`, `writing`, and `packing_bags`. Accessing any other activity type (e.g., `microsleep`, `sleep`, `mind_diversion`) would raise a `KeyError`.

## Affected Code

```python
self.per_person_consecutive_detections[person_idx] = {
    'cell_phone': 0, 'writing': 0, 'packing_bags': 0
}
```

## Suggested Fix

Either initialize all 10 activity types consistently, or use `defaultdict(int)` to avoid `KeyError` on missing keys.
