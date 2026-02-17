# Task 0010: No timeout on `future.result()` -- hangs possible

- **Issue ID:** H-08
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 10)
- **Severity:** HIGH
- **Category:** Reliability
- **File:** `app/utils/video_multiprocessing.py:748`

## Description

If a worker hangs (deadlock in OpenCV, stuck GPU), main process waits indefinitely.

## Fix

Add per-chunk timeout: `future.result(timeout=chunk_duration * 10)`.
