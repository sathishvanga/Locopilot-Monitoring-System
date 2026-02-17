# Task 0004: Sleep state machine gate checks only one person (global)

- **Issue ID:** H-02
- **Priority:** Phase 1 - Critical Correctness (Item 4)
- **Severity:** HIGH
- **Category:** State Machine
- **File:** `locopilot_monitor.py:4428-4442`

## Description

Gate iterates over all `persons_data` and sets `state_machine_ready = True` if ANY person is DROWSY+. Person 0's SLEEPING state lets person 1's microsleep bypass the gate.

## Fix

Apply state machine gate per-person in `process_all_persons_activities` before aggregation.
