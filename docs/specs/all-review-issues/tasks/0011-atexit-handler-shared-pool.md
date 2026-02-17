# Task 0011: Shared pool has no `atexit` shutdown handler

- **Issue ID:** H-09
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 11)
- **Severity:** HIGH
- **Category:** Resource Management
- **File:** `app/utils/video_multiprocessing.py:231-264`

## Description

`get_shared_pool()` creates a global `ProcessPoolExecutor` with no `atexit` handler. Worker processes may become orphaned on application exit.

## Fix

Register `atexit.register(shutdown_shared_pool)` when creating the shared pool.
