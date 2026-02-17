# Task 0022: `hand_smoothing_buffers` grows without bounds

- **Issue ID:** M-03
- **Priority:** Phase 4 - Memory & Performance (Item 22)
- **Severity:** MEDIUM
- **Category:** Memory Leak
- **File:** `locopilot_monitor.py:651`

## Description

Keys are `(person_idx, hand_side)` tuples. Cleanup uses integer keys so can't clean these.

## Fix

Add special cleanup for tuple-keyed dicts, or restructure as `{person_idx: {'right': ..., 'left': ...}}`.
