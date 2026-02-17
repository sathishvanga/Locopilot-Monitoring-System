# Task 0026: `get_keypoint` function duplicated across 4 detectors

- **Issue ID:** H-14
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 26)
- **Severity:** HIGH
- **Category:** Code Duplication
- **File:** `sleep_detector.py:269`, `gesture_detector.py:106`, `activity_detector.py:101`, `mind_diversion_detector.py:104`

## Description

4 different implementations with different error handling and fallback maps. SleepDetector has the most complete version with `fallback_map` for MediaPipe keypoints.

## Fix

Create canonical `get_keypoint` in `app/core/utils/pose_utils.py`, have all detectors import it.
