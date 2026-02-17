# Task 0027: `calculate_wrist_distance` duplicated in SleepDetector and ActivityDetector

- **Issue ID:** H-15
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 27)
- **Severity:** HIGH
- **Category:** Code Duplication
- **File:** `sleep_detector.py:433-519`, `activity_detector.py:118-201`

## Description

Nearly identical implementations with slightly different hardcoded vs configurable thresholds.

## Fix

Extract into shared utility in `app/core/utils/pose_utils.py`.
