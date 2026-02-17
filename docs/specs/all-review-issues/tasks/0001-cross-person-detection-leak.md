# Task 0001: Cross-person detection leak via shared `detections` dict

- **Issue ID:** C-01
- **Priority:** Phase 1 - Critical Correctness (Item 1)
- **Severity:** CRITICAL
- **Category:** Detection Correctness
- **File:** `locopilot_monitor.py:3050-3053`

## Description

`detections['cell_phone'].extend(person_detections['cell_phone'])` mutates the shared `detections` dict. When processing person 0, ROI detections are appended. When processing person 1, person 0's detections are still in the list, causing false positives for person 1.

## Fix

Use per-person scoped detection lists: `person_cell_phones = detections['cell_phone'] + person_detections['cell_phone']` and use that for the person's activity checks instead of mutating the shared dict.
