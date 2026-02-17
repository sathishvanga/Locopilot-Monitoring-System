# Task 0003: Sleep vs writing suppression chicken-and-egg

- **Issue ID:** H-01
- **Priority:** Phase 1 - Critical Correctness (Item 3)
- **Severity:** HIGH
- **Category:** Detection Correctness
- **File:** `locopilot_monitor.py:4444-4471`

## Description

Writing suppresses sleep, but `sleep_state_overrides_writing` only activates when state machine is already DROWSY+. On the first frame of sleep onset, state machine is still ALERT, so writing suppresses sleep, preventing the state machine from ever advancing.

## Fix

Add override condition checking drowsiness indicators directly from `pose_sleep_info` (e.g., `is_reclined_sleep`, `nose_y_drop < -0.05`) at lines 4449-4456.
