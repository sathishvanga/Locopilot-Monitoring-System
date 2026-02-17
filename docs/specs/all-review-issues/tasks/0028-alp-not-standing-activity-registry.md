# Task 0028: `alp_not_standing` bypasses ACTIVITY_REGISTRY

- **Issue ID:** H-05
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 28)
- **Severity:** HIGH
- **Category:** Code Quality
- **File:** `locopilot_monitor.py:803-812`

## Description

Manually added to tracking dicts after registry-based init, defeating the single-source-of-truth pattern.

## Fix

Add `'alp_not_standing': ActivityConfig(required_consecutive=2, grace_frames=3)` to `_build_activity_registry()` and remove manual init.
