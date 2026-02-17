# Task 0030: All remaining Medium and Low issues

- **Issue ID:** Multiple (see below)
- **Priority:** Phase 5 - Code Quality & Deduplication (Item 30)
- **Severity:** MEDIUM / LOW
- **Category:** Various

## Description

Remaining issues not covered in tasks 0001-0029. These should be addressed as part of ongoing code quality improvements.

## Remaining Medium Issues

| ID | Title | File | Category |
|----|-------|------|----------|
| M-01 | Packing bags detection breaks out of backpack loop too early | `locopilot_monitor.py:3479, 3524` | Detection Correctness |
| M-02 | Legacy global sleep state reset is dead code | `locopilot_monitor.py:4470-4471` | Code Quality |
| M-04 | Control zone pixel thresholds not resolution-scaled | `locopilot_monitor.py:2363-2405` | False Positives |
| M-05 | `packing` vs `packing_bags` naming inconsistency | `locopilot_monitor.py:3018, 4608, 3600` | Code Quality |
| M-06 | Magic numbers in gesture detection | `locopilot_monitor.py` (multiple lines) | Code Quality |
| M-07 | Broad `except Exception` swallows stack traces | `locopilot_monitor.py:3653-3655` | Code Quality |
| M-08 | VideoCapture not using context manager in `end_activity` | `locopilot_monitor.py:4004-4007` | Resource Management |
| M-09 | `_worker_config` is dead code | `app/utils/video_multiprocessing.py:48, 224` | Code Quality |
| M-10 | `config_dict` always empty -- dead parameter | `app/utils/video_multiprocessing.py:718, 725` | Code Quality |
| M-11 | Thread over-subscription risk | `app/utils/multiprocessing_config.py:37-38, 63-84, 92-100` | Performance |
| M-12 | `mp.set_start_method(force=True)` is global and unnecessary | `app/utils/video_multiprocessing.py:571` | Code Quality |
| M-13 | Sleep state machine has no SLEEPING -> MICROSLEEP transition | `app/core/detectors/sleep_detector.py:616-628` | State Machine |
| M-14 | Cell phone visibility not checked before pixel conversion | `app/core/detectors/activity_detector.py:443-447` | False Positives |
| M-15 | MindDiversionDetector shoulder/ear visibility not checked | `app/core/detectors/mind_diversion_detector.py:186-196` | False Positives |
| M-16 | PersonTracker loses person indices on role update | `app/core/tracking/person_tracker.py:267` | Detection Correctness |
| M-17 | Face mesh indices accessed without bounds checking | `app/core/detectors/mind_diversion_detector.py:349-356` | Robustness |
| M-18 | Gesture coordination timing race condition | `app/core/detectors/gesture_detector.py:482-506` | Detection Correctness |
| M-19 | Full-frame YOLO call without confidence filter | `app/core/detectors/object_detector.py:390-395` | Performance |
| M-20 | Hardcoded 1920x1080 frame dimensions in person_tracker | `app/core/tracking/person_tracker.py:345-346` | Detection Correctness |
| M-21 | Writing wrist distance 300px not scale-normalized | `app/core/detectors/activity_detector.py:283-296` | False Positives / False Negatives |
| M-24 | Side window motion service is stateful and not thread-safe | `app/services/side_window_motion_service.py:101` | Concurrency |
| M-25 | 11 singleton getters lack thread-safe locking | Multiple service files | Concurrency |

## Remaining Low Issues

| ID | Title | File |
|----|-------|------|
| L-01 | `temporal_suppression_window` hardcoded at 10s | `locopilot_monitor.py:581` |
| L-02 | Full frame copy for optical flow | `locopilot_monitor.py:4578` |
| L-03 | Inconsistent indentation in return statement | `locopilot_monitor.py:2502-2529` |
| L-04 | Duplicate `import json` in video_multiprocessing | `app/utils/video_multiprocessing.py:9, 831` |
| L-05 | `NumpyEncoder` missing `np.bool_` handling | `app/utils/video_multiprocessing.py:34-43` |
| L-06 | Nested `_get_smoothed_hand_position` function | `locopilot_monitor.py:3342-3378` |
| L-07 | `chin` variable extracted but never used | `app/core/detectors/mind_diversion_detector.py:353` |
| L-08 | `bag_max_aspect_ratio` of 1.2 may reject legitimate bags | `app/core/detectors/object_detector.py:96` |
| L-09 | DEBUG messages logged at INFO level in ObjectDetector | `app/core/detectors/object_detector.py:249-254`, `app/core/models/yolo_handler.py:602-618` |
| L-10 | No-op string operation in YoloPoseAdapter | `app/services/yolo_pose_adapter.py:301` |
| L-11 | Gamma LUT recomputed on every frame | `app/services/image_preprocessing_service.py:243` |
| L-12 | `_find_overlapping_groups` is dead code | `app/services/concurrent_activity_grouping_service.py:153` |

## Fix

Address each issue individually following the fix guidance in the original review document. Prioritize Medium issues before Low issues.
