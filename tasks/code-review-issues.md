# Code Review Issues: `locopilot_monitor.py`

**File:** `locopilot_monitor.py` (7,273 lines)
**Total Findings:** 21
**Date:** 2026-01-31

## Summary Table

| ID     | Severity | Category                        | Title                                                        | Lines            |
|--------|----------|---------------------------------|--------------------------------------------------------------|------------------|
| CR-001 | Critical | Architecture / Code Organization | God Class violates Single Responsibility Principle            | 150-7195         |
| CR-002 | Critical | Code Quality / Maintainability   | No type hints on any public API                              | 151+             |
| CR-003 | High     | Maintainability / Duplication    | Near-identical duplication between `process_video` and `process_video_range` | ~6200, ~6721 |
| CR-004 | High     | Code Organization / Maintainability | 460-line `__init__` with 4 parallel dictionaries           | 150-611          |
| CR-005 | High     | Performance / Memory             | Storing full `frame.copy()` for every detected activity frame | 6612, 7078      |
| CR-006 | High     | Performance / Bottleneck         | Per-person YOLO inference runs full-frame detection N times   | 4502             |
| CR-007 | High     | Bug / Logic Error                | Non-deterministic LP/ALP role assignment based on bbox area   | 5754             |
| CR-008 | High     | Error Handling / Best Practice   | Bare `except:` clauses catch `KeyboardInterrupt` and `SystemExit` | 2834, 2843, 2872, 2897 |
| CR-009 | High     | Error Handling / Debugging       | Broad `except Exception` with silent defaults in 10+ methods | 949, 979, 1015, 1509+ |
| CR-010 | High     | Code Quality / Maintainability   | 50+ magic numbers scattered throughout the codebase          | 2084, 2091, 1486, 1618+ |
| CR-011 | High     | Performance / Resource Management | VideoCapture opened per `end_activity` call just for metadata | 5997            |
| CR-012 | Medium   | Memory / Resource Leak           | Unbounded growth of 10+ per-person tracking dictionaries     | 451-473          |
| CR-013 | Medium   | Bug / Multiprocessing            | Optical flow breaks at chunk boundaries in multiprocessing    | 7030             |
| CR-014 | Medium   | Bug / KeyError Risk              | `per_person_consecutive_detections` only tracks 3 of 10+ types | 841            |
| CR-015 | Medium   | Code Quality / Anti-Pattern      | Lazy `hasattr` initialization instead of declaring in `__init__` | 4001, 4780    |
| CR-016 | Medium   | Code Quality / Performance       | Repeated `import` statements inside methods                  | 1864, 1964, 2168, 3071 |
| CR-017 | Medium   | Code Quality / Logging           | Inconsistent logging with multiple logger instances           | 131-133, 170    |
| CR-018 | Medium   | Code Quality / Dead Code         | Commented-out debug code remaining in production              | 3449-3528        |
| CR-019 | Low      | Security / Data Integrity        | Hardcoded default crew data could leak into production        | 525-528          |
| CR-020 | Low      | Portability / Encoding           | Unicode emoji in log messages can cause encoding issues       | 1871, 2749       |
| CR-021 | Low      | Code Quality / Anti-Pattern      | `locals()` check for variable deletion is fragile             | 6664             |

---

## Critical Severity

### CR-001: God Class violates Single Responsibility Principle

- **Category:** Architecture / Code Organization
- **Lines:** 150-7195 (entire class)

**Description:**
The `LocopilotActivityMonitor` class handles everything: model loading, frame sampling, object detection, pose estimation, sleep/writing/packing detection, hand gestures, person identification, video I/O, and evidence management. This is untestable in isolation and violates the single responsibility principle.

**Affected Code:**
The entire `LocopilotActivityMonitor` class spanning ~7,000 lines with 50+ methods.

**Suggested Fix:**
Decompose into smaller classes: `ActivityDetector`, `FrameSampler`, `ModelManager`, `ActivityStateTracker`, and `EvidenceCollector`. Each class should own one concern and communicate via well-defined interfaces.

---

### CR-002: No type hints on any public API

- **Category:** Code Quality / Maintainability
- **Lines:** 151 (constructor and all public methods)

**Description:**
The constructor has 8 untyped parameters. No public methods have type annotations, making the API difficult to understand, use, and validate.

**Affected Code:**
```python
def __init__(self, video_path, output_dir, ...):  # 8 untyped params
```

**Suggested Fix:**
Add type hints to all public methods and the constructor. Use `typing` module for complex types (e.g., `Optional[str]`, `List[dict]`).

---

## High Severity

### CR-003: Near-identical duplication between `process_video` and `process_video_range`

- **Category:** Maintainability / Duplication
- **Lines:** ~6200 (`process_video`, 482 lines) and ~6721 (`process_video_range`, 435 lines)

**Description:**
These two methods share extensive logic (frame sampling, YOLO detection, person deduplication, multi-person activity processing, activity lifecycle management) but are duplicated. Bug fixes must be manually replicated in both.

**Affected Code:**
Both methods contain duplicated: frame sampling via `sample_video_frames()`, YOLO object detection, person box deduplication with IOU=0.5, multi-person activity processing, and activity lifecycle management.

**Suggested Fix:**
Extract shared logic into a common `_process_frames_core()` method that both `process_video` and `process_video_range` call with their specific parameters.

---

### CR-004: 460-line `__init__` with 4 parallel dictionaries requiring synchronized updates

- **Category:** Code Organization / Maintainability
- **Lines:** 150-611

**Description:**
The constructor is 461 lines long and maintains 4 parallel dictionaries (`activities`, `consecutive_detections`, `grace_counters`, `per_person_consecutive_detections`) that must be kept in sync for each activity type. Adding a new activity requires updating all dictionaries.

**Affected Code:**
```python
self.consecutive_detections = {
    'microsleep': 0, 'sleep': 0, 'cell_phone': 0, 'writing': 0,
    'packing_bags': 0, ...
}
self.grace_counters = {
    'microsleep': 0, 'sleep': 0, 'cell_phone': 0, 'writing': 0,
    'packing_bags': 0, ...
}
```

**Suggested Fix:**
Use a single `ActivityConfig` dataclass or registry pattern where each activity type auto-generates its tracking entries. A factory method can initialize all parallel dicts from a single source of truth.

---

### CR-005: Storing full `frame.copy()` for every detected activity frame consumes excessive RAM

- **Category:** Performance / Memory
- **Lines:** 6612, 7078

**Description:**
Full frame copies are stored in memory for every detected activity frame. On long videos, this can consume gigabytes of RAM.

**Affected Code:**
```python
self.activities[activity]['frames'].append(frame.copy())
```

**Suggested Fix:**
Store frame indices instead of frame copies and extract frames from the video on demand when saving evidence.

---

### CR-006: Per-person YOLO inference runs full-frame detection N times for N persons

- **Category:** Performance / Bottleneck
- **Lines:** 4502

**Description:**
In `process_all_persons_activities`, full-frame YOLO inference is run once per detected person per frame, causing O(N) GPU inference calls where N is the number of persons.

**Suggested Fix:**
Run YOLO inference once per frame and distribute the results to each person based on bounding box overlap.

---

### CR-007: Non-deterministic LP/ALP role assignment based on bounding box area

- **Category:** Bug / Logic Error
- **Lines:** 5754

**Description:**
Person roles (LP vs ALP) are assigned based on bounding box area. When persons have similar sizes, roles can flip between frames, breaking hand gesture coordination logic that depends on stable role assignments.

**Suggested Fix:**
Implement temporal role tracking using position continuity (IoU-based tracking across frames) or a simple tracker like SORT to maintain consistent person identities.

---

### CR-008: Bare `except:` clauses catch `KeyboardInterrupt` and `SystemExit`

- **Category:** Error Handling / Best Practice
- **Lines:** 2834, 2843, 2872, 2897

**Description:**
Bare `except:` without specifying an exception type catches all exceptions including `KeyboardInterrupt` and `SystemExit`, preventing the application from being interrupted or shut down gracefully.

**Affected Code:**
```python
try:
    cv2.line(annotated_frame, start_pt, end_pt, (0, 255, 255), 3)
except:
    continue
```

**Suggested Fix:**
Replace all bare `except:` with `except Exception:` at minimum. In drawing code, use `except (cv2.error, ValueError, TypeError):` for more specific handling.

---

### CR-009: Broad `except Exception` with silent defaults in 10+ methods

- **Category:** Error Handling / Debugging
- **Lines:** 949, 979, 1015, 1509 (and 30+ more locations)

**Description:**
Over 30 methods catch `except Exception` and silently return default values (`None`, `False`, `0.0`) without logging, making debugging extremely difficult. Failures are invisible.

**Affected Code:**
```python
except Exception as e:
    return None   # Line 949 - calculate_eye_aspect_ratio
except Exception as e:
    return None   # Line 979 - calculate_head_tilt_angle
except Exception as e:
    return 0.0    # Line 1015 - calculate_movement_score
except Exception as e:
    return False   # Line 1509 - detect_writing_posture
```

**Suggested Fix:**
Add `self.logger.debug()` or `self.logger.warning()` calls inside all exception handlers. Consider narrowing exception types to expected failures (e.g., `IndexError`, `ValueError`).

---

### CR-010: 50+ magic numbers scattered throughout the codebase

- **Category:** Code Quality / Maintainability
- **Lines:** 2084, 2091, 2109, 1486, 1503, 3507, 1229-1264, 1381-1398, 1618-1622, and many more

**Description:**
Confidence thresholds, distance thresholds, pixel margins, duration windows, and other numeric constants are hardcoded directly in the logic without named constants or configuration.

**Affected Code:**
```python
MAX_WRIST_DISTANCE = 300       # Line 1618
MAX_ELBOW_DISTANCE = 450       # Line 1619
WRITING_WRIST_DISTANCE = 300   # Line 1486
HEAD_DOWN_THRESHOLD = 0.01     # Line 1538
```

**Suggested Fix:**
Extract all magic numbers into named constants at the class level or into a dedicated configuration file/dataclass. Group them by detection type (sleep, writing, packing, etc.).

---

### CR-011: VideoCapture opened per `end_activity` call just for metadata

- **Category:** Performance / Resource Management
- **Lines:** 5997

**Description:**
Each time an activity ends, a `VideoCapture` is opened solely to read total frame count and FPS, then immediately closed. This is wasteful I/O for metadata that never changes.

**Affected Code:**
```python
with video_capture_context(self.video_path) as cap:
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = video_total_frames / fps
```

**Suggested Fix:**
Cache video metadata (total frames, FPS, duration) once during `__init__` or on first use, and reuse it in all subsequent `end_activity` calls.

---

## Medium Severity

### CR-012: Unbounded growth of 10+ per-person tracking dictionaries

- **Category:** Memory / Resource Leak
- **Lines:** 451-473

**Description:**
Per-person tracking dictionaries (`per_person_sleep_tracking`, `per_person_consecutive_detections`, `per_person_grace_counters`, `hand_position_history`, `landmark_stability_history`, `wrist_proximity_tracking`, `no_pose_sleep_tracking`, etc.) are never pruned. Over long videos, persons who leave the frame still occupy memory indefinitely.

**Affected Code:**
```python
self.per_person_sleep_tracking = {}      # never pruned
self.per_person_consecutive_detections = {}  # never pruned
self.per_person_grace_counters = {}      # never pruned
self.hand_position_history = {}          # never pruned
self.landmark_stability_history = {}     # never pruned
```

**Suggested Fix:**
Implement a periodic cleanup that removes entries for person indices not seen in the last N frames. Use an LRU-style eviction or timestamp-based expiry.

---

### CR-013: Optical flow breaks at chunk boundaries in multiprocessing mode

- **Category:** Bug / Multiprocessing
- **Lines:** 7030

**Description:**
In `process_video_range`, `_prev_motion_frame` is `None` at each chunk start because each worker has fresh state. This means optical flow-based motion detection produces no results for the first frame(s) of every chunk.

**Suggested Fix:**
Pass the last frame of the previous chunk as initialization data to the next chunk's worker, or add a small overlap between chunks (e.g., 1-2 frames).

---

### CR-014: `per_person_consecutive_detections` only tracks 3 of 10+ activity types

- **Category:** Bug / KeyError Risk
- **Lines:** 841

**Description:**
The per-person consecutive detection dictionary only initializes keys for `cell_phone`, `writing`, and `packing_bags`. Accessing any other activity type (e.g., `microsleep`, `sleep`, `mind_diversion`) would raise a `KeyError`.

**Affected Code:**
```python
self.per_person_consecutive_detections[person_idx] = {
    'cell_phone': 0, 'writing': 0, 'packing_bags': 0
}
```

**Suggested Fix:**
Either initialize all 10 activity types consistently, or use `defaultdict(int)` to avoid `KeyError` on missing keys.

---

### CR-015: Lazy `hasattr` initialization instead of declaring attributes in `__init__`

- **Category:** Code Quality / Anti-Pattern
- **Lines:** 4001, 4780

**Description:**
Some attributes are lazily initialized using `hasattr` checks inside methods rather than being declared in `__init__`. This makes the class interface unpredictable and hides state.

**Affected Code:**
```python
if not hasattr(self, 'packing_motion_history'):   # Line 4001
    self.packing_motion_history = {}
if not hasattr(self, 'hand_smoothing_buffers'):    # Line 4780
    self.hand_smoothing_buffers = {}
```

**Suggested Fix:**
Declare all instance attributes in `__init__`, even if initialized to empty values. This makes the full state of the object visible in one place.

---

### CR-016: Repeated `import` statements inside methods

- **Category:** Code Quality / Performance
- **Lines:** 1864, 1964, 2168, 3071

**Description:**
Several methods contain `import logging` and `import time` statements inside their bodies rather than at module level. While Python caches imports, this is an anti-pattern that adds overhead and clutters method bodies.

**Affected Code:**
```python
def detect_objects_in_roi(self, ...):
    import logging    # Line 1864
def detect_objects_in_rois_batch(self, ...):
    import logging    # Line 1964
def detect_objects_batch(self, ...):
    import logging    # Line 2168
    import time
```

**Suggested Fix:**
Move all imports to the top of the file at module level.

---

### CR-017: Inconsistent logging with multiple logger instances

- **Category:** Code Quality / Logging
- **Lines:** 131-133, 170, and various methods

**Description:**
The module uses three different logging approaches: `self.logger` (instance-level), module-level `gesture_logger` and `monitor_logger`, and fresh loggers created inside methods. This creates inconsistent log output and makes filtering difficult.

**Affected Code:**
```python
gesture_logger = _setup_module_logger('HandGestureDetection')   # Line 132
monitor_logger = _setup_module_logger('LocopilotMonitor')       # Line 133
self.logger = _setup_module_logger(...)                         # Line 170
```

**Suggested Fix:**
Consolidate to a single logger hierarchy. Use `self.logger` consistently throughout the class and child loggers (e.g., `self.logger.getChild('gesture')`) for subsystem-specific logging.

---

### CR-018: Commented-out debug code remaining in production

- **Category:** Code Quality / Dead Code
- **Lines:** 3449-3528

**Description:**
A large block of commented-out debug code (~80 lines) remains in the source, adding clutter and confusion about intended behavior.

**Suggested Fix:**
Remove all commented-out code. Use version control (git) to preserve history if the code is needed later.

---

## Low Severity

### CR-019: Hardcoded default crew data could leak into production evidence

- **Category:** Security / Data Integrity
- **Lines:** 525-528

**Description:**
Default crew member data is hardcoded (e.g., `"John Doe"`, `"C-001"`, `"TRIP-123"`). If the API override fails to provide real crew data, these dummy values will appear in production evidence records.

**Affected Code:**
```python
self.trip_id = "TRIP-123"
self.crew_name = "John Doe"
self.crew_id = "C-001"
self.crew_role = 1
```

**Suggested Fix:**
Use `None` or sentinel values as defaults and validate that real crew data is provided before generating evidence. Raise an error or log a warning if defaults are still in place when evidence is created.

---

### CR-020: Unicode emoji in log messages can cause encoding issues

- **Category:** Portability / Encoding
- **Lines:** 1871, 2749

**Description:**
Log messages contain Unicode emoji characters that can cause encoding errors on systems with limited Unicode support or when piping log output.

**Affected Code:**
```python
self.logger.info("... [checkmark emoji] ...")    # Line 1871
self.logger.debug("... [search emoji] ...")      # Line 2749
```

**Suggested Fix:**
Replace emoji characters with ASCII equivalents (e.g., `[OK]`, `[SEARCH]`) in log messages.

---

### CR-021: `locals()` check for variable deletion is fragile

- **Category:** Code Quality / Anti-Pattern
- **Lines:** 6664

**Description:**
Using `locals()` to check for variable existence is a fragile pattern that can break with Python compiler optimizations and is difficult to understand.

**Affected Code:**
```python
if 'variable_name' in locals():
    del variable_name
```

**Suggested Fix:**
Use explicit `None` initialization and check against `None`, or use a try/except `NameError` if variable existence is truly uncertain.

---

## Statistics

| Severity | Count | Issue IDs           |
|----------|-------|---------------------|
| Critical | 2     | CR-001, CR-002      |
| High     | 9     | CR-003 - CR-011     |
| Medium   | 7     | CR-012 - CR-018     |
| Low      | 3     | CR-019 - CR-021     |
| **Total**| **21**|                     |
