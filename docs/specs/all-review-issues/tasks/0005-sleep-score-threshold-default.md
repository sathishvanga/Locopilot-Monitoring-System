# Task 0005: Sleep score threshold default still 3 (should be 5)

- **Issue ID:** H-18
- **Priority:** Phase 1 - Critical Correctness (Item 5)
- **Severity:** HIGH
- **Category:** False Positives
- **File:** `app/core/detectors/sleep_detector.py:1108`

## Description

`getattr(self.settings, 'sleep_score_threshold', 3)` -- fallback is `3` but Feb 14 Fix 1 tightened it to `5`.

## Fix

Change default to `5`: `getattr(self.settings, 'sleep_score_threshold', 5)`.
