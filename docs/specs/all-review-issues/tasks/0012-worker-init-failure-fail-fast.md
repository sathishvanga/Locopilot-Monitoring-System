# Task 0012: Worker init failure causes silent model re-loading

- **Issue ID:** H-11
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 12)
- **Severity:** HIGH
- **Category:** Multiprocessing
- **File:** `app/utils/video_multiprocessing.py:420`

## Description

When `_worker_models` is `None` (failed init), monitor attempts to reload all models from scratch in the worker, causing severe performance degradation instead of a clear failure.

## Fix

Check `_worker_models` at start of `process_frame_range` and fail fast with `RuntimeError`.
