# Task 0020: Unbounded `per_person_tracking` defaultdict in SleepDetector

- **Issue ID:** C-11
- **Priority:** Phase 4 - Memory & Performance (Item 20)
- **Severity:** CRITICAL
- **Category:** Memory Leak
- **File:** `app/core/detectors/sleep_detector.py:92`

## Description

`defaultdict(self._create_tracking_dict)` grows without bounds across long videos. No cleanup method exists. In a 24/7 monitoring scenario with shifting person indices, this will grow indefinitely.

## Fix

Add `cleanup_stale_tracking(active_person_indices)` method and call it from the monitor's existing cleanup cycle.
