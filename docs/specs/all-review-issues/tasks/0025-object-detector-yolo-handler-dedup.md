# Task 0025: ObjectDetector / YOLOHandler ~80% code duplication

- **Issue ID:** C-10
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 25)
- **Severity:** CRITICAL
- **Category:** Code Quality / Maintenance Risk
- **File:** `app/core/detectors/object_detector.py` (821 lines), `app/core/models/yolo_handler.py` (1055 lines)

## Description

Near-identical implementations of `detect_objects()`, `detect_objects_in_rois_batch()`, `detect_objects_batch()`, `validate_object_aspect_ratio()`, `_boxes_overlap_or_near()`, `get_roi_around_keypoint()`, `preprocess_frames_for_detection()`. Bug fixes in one may not propagate to the other.

## Fix

Consolidate into a single class. Either `YOLOHandler` contains all logic and `ObjectDetector` wraps it, or extract shared logic into a common base class.
