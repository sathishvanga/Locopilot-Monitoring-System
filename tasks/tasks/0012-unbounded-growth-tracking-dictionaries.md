# CR-012: Unbounded growth of 10+ per-person tracking dictionaries

- **Severity:** Medium
- **Category:** Memory / Resource Leak
- **Lines:** 451-473

## Description

Per-person tracking dictionaries (`per_person_sleep_tracking`, `per_person_consecutive_detections`, `per_person_grace_counters`, `hand_position_history`, `landmark_stability_history`, `wrist_proximity_tracking`, `no_pose_sleep_tracking`, etc.) are never pruned. Over long videos, persons who leave the frame still occupy memory indefinitely.

## Affected Code

```python
self.per_person_sleep_tracking = {}      # never pruned
self.per_person_consecutive_detections = {}  # never pruned
self.per_person_grace_counters = {}      # never pruned
self.hand_position_history = {}          # never pruned
self.landmark_stability_history = {}     # never pruned
```

## Suggested Fix

Implement a periodic cleanup that removes entries for person indices not seen in the last N frames. Use an LRU-style eviction or timestamp-based expiry.
