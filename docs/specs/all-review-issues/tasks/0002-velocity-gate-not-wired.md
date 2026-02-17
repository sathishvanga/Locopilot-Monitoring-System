# Task 0002: Hand gesture velocity gate computed but result discarded

- **Issue ID:** H-03
- **Priority:** Phase 1 - Critical Correctness (Item 2)
- **Severity:** HIGH
- **Category:** False Positives
- **File:** `locopilot_monitor.py:2488-2499`

## Description

Velocity analysis (`rapid_raise_detected`) is computed at significant cost but its result is only logged, not used to gate the gesture detection return value.

## Fix

Wire `rapid_raise_detected` as a required condition, or increase `required_consecutive` when velocity is low.
