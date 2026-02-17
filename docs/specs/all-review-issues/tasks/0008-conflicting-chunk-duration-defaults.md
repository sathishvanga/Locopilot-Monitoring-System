# Task 0008: Conflicting chunk duration defaults (15s vs 6s)

- **Issue ID:** C-03
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 8)
- **Severity:** CRITICAL
- **Category:** Configuration
- **File:** `app/utils/config.py:75`, `app/utils/multiprocessing_config.py:30`

## Description

`Settings.mp_chunk_duration=15.0` vs `MultiprocessingConfig.chunk_duration_seconds=6.0`. The actual runtime value depends on the call path. `ActivityDetectionService` uses 15s, direct `MultiprocessingConfig()` construction gets 6s. The comment says "15s chunks ensure hand gesture coordination detection works correctly."

## Fix

Eliminate the duplicate. `MultiprocessingConfig.chunk_duration_seconds` should always be supplied from `Settings.mp_chunk_duration`. Remove the hardcoded default in `MultiprocessingConfig`.
