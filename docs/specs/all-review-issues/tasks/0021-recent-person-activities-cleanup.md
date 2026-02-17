# Task 0021: `recent_person_activities` dict grows without bounds

- **Issue ID:** H-04
- **Priority:** Phase 4 - Memory & Performance (Item 21)
- **Severity:** HIGH
- **Category:** Memory Leak
- **File:** `locopilot_monitor.py:580`

## Description

Entries added at lines 3476-3478, 3515-3517, etc. but NOT in `_cleanup_stale_person_tracking`'s `tracking_dicts` list.

## Fix

Add `('recent_person_activities', self.recent_person_activities)` to `tracking_dicts` list in `_cleanup_stale_person_tracking`.
