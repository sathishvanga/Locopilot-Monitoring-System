# Task 0006: `eating_drinking` missing from ACTIVITY_REGISTRY and all maps

- **Issue ID:** C-06
- **Priority:** Phase 1 - Critical Correctness (Item 6)
- **Severity:** CRITICAL
- **Category:** Detection Correctness
- **File:** `locopilot_monitor.py:72-136, 670-682, 685-697, 700-712`

## Description

Per memory Fix 12, `eating_drinking` should be independent activity type 13. But it's NOT in `ACTIVITY_REGISTRY`, `activity_type_map`, `activity_descriptions`, or `evidence_rules`. Currently piggybacked on `mind_diversion` as a sub-type. Rule engine also missing it.

## Fix

Add `eating_drinking` to `ACTIVITY_REGISTRY`, all 4 maps, `activities_map`, and the rule engine service's `ACTIVITY_NAMES` + `ALLOWED_WHEN_STOPPED`.
