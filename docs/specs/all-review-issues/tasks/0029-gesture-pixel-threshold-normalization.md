# Task 0029: Gesture detection pixel thresholds not resolution-normalized

- **Issue ID:** H-16
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 29)
- **Severity:** HIGH
- **Category:** False Positives / False Negatives
- **File:** `app/core/detectors/gesture_detector.py:40-48`

## Description

`WRIST_SHOULDER_VERTICAL_MIN=80`, `ARM_EXTENSION_MIN=20`, etc. are absolute pixel values. At 4K, 80px is tiny; at 480p, 80px is enormous. `_scale_margin` exists in monitor but not in this detector.

## Fix

Normalize by person bbox height: `threshold = max(20, int(bbox_height * 0.12))`.
