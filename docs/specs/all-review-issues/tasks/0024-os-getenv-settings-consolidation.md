# Task 0024: `os.getenv` calls bypass pydantic-settings -- dual config sources

- **Issue ID:** C-09
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 24)
- **Severity:** CRITICAL
- **Category:** Code Quality
- **File:** `locopilot_monitor.py:539, 547-548, 585`

## Description

`CELL_PHONE_CONFIDENCE`, `GPU_BATCH_SIZE`, `GPU_BATCH_ENABLED`, `HAND_GESTURE_COORDINATION_WINDOW` read directly from `os.getenv` instead of `self.settings`. Creates dual sources of truth.

## Fix

Add these fields to `Settings` class in `config.py` and access exclusively through `self.settings`.
