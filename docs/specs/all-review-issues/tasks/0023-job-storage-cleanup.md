# Task 0023: In-memory job storage unbounded growth

- **Issue ID:** M-23
- **Priority:** Phase 4 - Memory & Performance (Item 23)
- **Severity:** MEDIUM
- **Category:** Memory
- **File:** `app/services/job_manager.py:85`

## Description

`self._jobs` dict grows indefinitely. `cleanup_completed_jobs` exists but never called automatically.

## Fix

Add periodic cleanup task or `max_retained_jobs` limit.
