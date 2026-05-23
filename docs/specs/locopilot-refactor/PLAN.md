# Locopilot Monitor Refactor Plan

Behavior-preserving extraction plan for `locopilot_monitor.py` (5439 lines, 1
class + 2 module-level helpers, ~43 methods). Goal: shrink the monolith to
<1500 lines (target <1000) by lifting cohesive blocks into new modules under
`app/core/...`, following the existing extraction idiom (`utils/geometry`,
`models/yolo_handler`, `detectors/*`, `tracking/person_tracker`,
`visualization/frame_annotator`, `evidence_manager`).

This run is **planning only**. No edits are made to `locopilot_monitor.py` or
any production file. Each task in Section 2 is independent and creates only
new files. The single sequential rewire (Section 3) is the only step that
edits `locopilot_monitor.py`, and it must run after all extractions in
Section 2 land.

---

## Section 1 — Inventory

Format: `[lines]  name  →  what it does · self.* read/written`

### Module-level (outside the class)

| Lines | Symbol | Purpose · Notes |
|---|---|---|
| 109-140 | `_setup_module_logger(name, level)` | File-only logger factory. No `self`. Pure helper. **Already used by other modules indirectly via passed-in logger; keep in monolith but exposable.** |
| 145-156 | `video_capture_context(video_path)` | `@contextmanager` ensuring `cv2.VideoCapture.release()`. Pure. Several other code paths re-implement this. |

### `LocopilotActivityMonitor` methods

| Lines | Method | Purpose · self.* it touches |
|---|---|---|
| 179-762 | `__init__` | Wires up models, detectors, tracking dicts, settings. Touches everything. **Stays in monolith** (mostly thin assignments). |
| 764-781 | `set_trip_schedule` | Stores trip schedule; logs. `self.trip_schedule`, `self.logger`. Trivial; stays. |
| 783-803 | `_extract_ocr_timestamp` | Calls `self.ocr_service.extract_timestamp`. `self.ocr_service`, `self.logger`. Trivial; stays. |
| 805-823 | `get_keypoint` | Single-line delegation to `self._get_keypoint_by_name`. Stays as a helper alias. |
| 825-859 | `update_per_person_detection` | Per-person consecutive-detection counter. Reads/writes `self.per_person_consecutive_detections`, `self.per_person_grace_counters`, `self.activity_thresholds`. **EXTRACT (T7).** |
| 861-930 | `sample_video_frames` | Frame sampler generator; uses `video_capture_context`. Reads `self.sample_fps`, `self.logger`. **EXTRACT (T6).** |
| 936-961 | `check_hands_below_shoulders` | Pure pose check. `self.get_keypoint`, `self.logger`. **EXTRACT (T1).** |
| 963-1081 | `detect_writing_by_wrist_proximity` | Writing fallback: wrist-proximity + head-down. Reads/writes `self.wrist_proximity_tracking`, `self.activity_detector`, `self.logger`, `self.MAX_WRIST_DISTANCE`, `self.MAX_SINGLE_WRIST_DISTANCE`, `self.MAX_ELBOW_DISTANCE`, `self.WRITING_REQUIRED_CONSECUTIVE`, `self.WRITING_MIN_DURATION`. **EXTRACT (T1).** |
| 1083-1168 | `detect_writing_by_book_and_posture` | Writing fallback via book + head-down. Reads/writes `self.wrist_proximity_tracking`, `self.activity_detector`, `self.logger`, `self.PERSON_BOOK_OVERLAP_MARGIN`, `self.BOOK_POSTURE_REQUIRED_CONSECUTIVE`, `self.BOOK_POSTURE_MIN_DURATION`. **EXTRACT (T1).** |
| 1172-1234 | `detect_poses_batch` | YOLO pose batch inference loop. Reads `self.yolo_pose`, `self.yolo_device`, `self.logger`. **EXTRACT (T6).** |
| 1238-1349 | `draw_mediapipe_outputs` | Single-person MediaPipe overlay. Touches `self.mp_drawing`, `self.mp_pose`, `self.mp_face_mesh`, `self.mp_drawing_styles`, `self.SLEEP_STRONG_DURATION`, `self.SLEEP_MICROSLEEP_DURATION`. **EXTRACT (T2).** |
| 1352-1590 | `draw_multi_person_mediapipe_outputs` | Per-person skeleton + activity-warning overlay. Touches `self.mp_drawing`, `self.mp_face_mesh`, `self.mp_drawing_styles`, `self.get_keypoint`. **EXTRACT (T2).** |
| 1592-1627 | `_get_smoothed_hand_position` | 3-frame avg of hand pixels. Reads/writes `self.hand_smoothing_buffers`. **EXTRACT (T3).** |
| 1629-1643 | `check_hand_object_interaction` | Pure: bbox-with-margin around point. **EXTRACT (T1).** |
| 1648-1691 | `validate_pose_landmarks` | Counts valid landmarks + visibility avg. Reads `self.MIN_POSE_LANDMARKS`, `self.MIN_POSE_VISIBILITY`. **EXTRACT (T4).** |
| 1693-1763 | `validate_anatomical_consistency` | Pure pose-anatomy sanity check. `self.get_keypoint`. **EXTRACT (T4).** |
| 1765-1809 | `check_landmark_stability` | 3-frame shoulder-jump check. Reads/writes `self.landmark_stability_history`, `self.max_landmark_jump_threshold`. **EXTRACT (T4).** |
| 1811-2347 | `detect_hand_gesture` | The 540-line gesture engine: temporal suppression + control-zone filter + raise detection + velocity gate. Reads `self.recent_person_activities`, `self.temporal_suppression_window`, `self.logger`, `self.max_landmark_jump_threshold`, `self.hand_position_history`, `self.hand_history_max_length`. Calls `validate_pose_landmarks`, `validate_anatomical_consistency`, `check_landmark_stability`, `analyze_hand_velocity_and_trajectory`, `get_keypoint`. **LEAVE IN MONOLITH** (touches >5 self attrs, internally calls 4 other methods that are themselves being extracted; risky to lift wholesale this round). See Section 4. |
| 2349-2404 | `_check_hand_gesture_coordination` | Coordination-window logic. Reads `self.recent_person_activities`, `self.hand_gesture_coordination_window`. **EXTRACT (T3).** |
| 2406-2501 | `analyze_hand_velocity_and_trajectory` | Velocity + trajectory from wrist history. Reads/writes `self.hand_position_history`, `self.hand_history_max_length`, `self.get_keypoint`. **EXTRACT (T3).** |
| 2503-2507 | `analyze_packing_hand_motion` | One-line delegate to `self.activity_detector`. Stays (trivial wrapper). |
| 2509-2570 | `_update_static_backpack_tracking` | NMS-style bookkeeping. Reads/writes `self.static_backpack_candidates`, `self.static_backpack_iou_threshold`, `self.static_backpack_min_frames`, `self.static_backpack_suppression_enabled`. **EXTRACT (T5).** |
| 2572-2629 | `_update_static_phone_tracking` | Same pattern, phones. Reads/writes `self.static_phone_*`. **EXTRACT (T5).** |
| 2631-2670 | `_check_wrist_motion_for_packing` | Velocity gate. Reads `self.settings`, `self.hand_position_history`. **EXTRACT (T3).** |
| 2674-2680 | `_match_pose_to_roles` | One-line delegate to `self.person_tracker`. Stays. |
| 2682-3864 | `process_all_persons_activities` | The 1183-line per-frame multi-person orchestrator. Reads/writes nearly everything. **STAYS IN MONOLITH.** See Section 4. |
| 3869-3873 | `calculate_head_pose_angles` | One-line delegate. Stays. |
| 3875-3937 | `should_suppress_mind_diversion` | Mind-diversion suppression rules. Reads `self.settings`, `self.recent_person_activities`, `self.get_keypoint`. **EXTRACT (T8).** |
| 3942-3950 | `identify_person_roles` | One-line delegate. Stays. |
| 3952-4049 | `_annotate_evidence_frame` | Re-runs object + pose inference and overlays them on a single evidence frame. Reads `self.object_detector`, `self.yolo_pose`, `self._get_keypoint_by_name`, `self.logger`. **EXTRACT (T2).** |
| 4051-4090 | `start_activity` | Initializes activity record. Reads/writes `self.activities`, `self.frame_idx_buffer`, `self.current_motion_state`, `self.logger`. **STAYS** (mutates >5 keys on `self.activities`; tightly coupled to `end_activity` and frame-buffer state). |
| 4092-4141 | `_cleanup_stale_person_tracking` | Deletes stale per-person entries across 7 dicts. Reads/writes all per-person tracking dicts. **EXTRACT (T7).** |
| 4143-4157 | `_get_video_metadata` | Lazy-load + cache via `video_capture_context`. Reads/writes `self._video_total_frames`, `self._video_fps`, `self._video_duration_seconds`. Stays (small; tied to instance cache). |
| 4159-4424 | `end_activity` | 266-line evidence-record finalizer (clip + image + JSON record). Reads/writes `self.activities`, `self.evidence_counter`, `self.all_activities`, `self.crew_*`, `self.trip_id`, `self.evidence_clips_dir`, `self.evidence_manager`, `self.sample_fps`, `self.video_path`, `self.settings`, `self.consecutive_detections`, `self.grace_counters`. **STAYS** (touches >>5 self attrs and is tightly coupled to evidence_manager which is already extracted). |
| 4426-4468 | `_reencode_to_h264` | ffmpeg reencode. `self.logger`. **EXTRACT (T2).** |
| 4470-4496 | `save_video_clip` | Writes mp4v then re-encodes. **EXTRACT (T2).** |
| 4498-4517 | `extract_video_segment` | Delegate to `self.evidence_manager`. Stays. |
| 4519-5089 | `_process_frames_core` | The 570-line per-sample pipeline (face mesh, YOLO, dedupe, multi-person, gates, activities-map, OCR, lifecycle). Touches >30 self attrs, calls 12 other monitor methods. **STAYS IN MONOLITH.** See Section 4. |
| 5091-5172 | `process_video` | Outer loop calling `_process_frames_core`. Stays. |
| 5174-5325 | `process_video_range` | Multiprocessing two-pass batch entry. Stays. |
| 5327-5371 | `cleanup` | Frees models. Stays. |
| 5373-5385 | `__del__` | Calls `cleanup`. Stays. |
| 5387-5397 | `generate_summary_report` | Delegate to `evidence_manager`. Stays. |

---

## Section 2 — Independent Extraction Tasks

All paths are absolute. Each task creates ONLY new files. No task edits
`locopilot_monitor.py`. No two tasks touch the same target file.

Common ground rules for every executor:
- The Python test env is `/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11`.
- Verification is import-smoke + a focused unit/property test using
  the test env. The deploy box has no git, so behavior must remain byte-identical
  after the rewire (Section 3) — the extracted code should be a direct lift
  with the only allowed delta being: the `self.*` reads it needed are now
  function arguments or attributes on a small dataclass/context.
- Logging: pass an injected `logger: logging.Logger` argument with default
  `logging.getLogger(__name__)`. Preserve every log message string verbatim
  (operators grep production logs for these strings).
- No new dependencies. Only stdlib + numpy + cv2 + the existing
  `app.core.utils.geometry` helpers.

---

### T1 — Writing fallback detectors + small geometry helpers

**Title:** Lift writing-detection fallbacks and the trivial pose/object
proximity helpers into `app/core/detectors/writing_fallbacks.py` and
`app/core/utils/pose_checks.py`.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/detectors/writing_fallbacks.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/utils/pose_checks.py`

**Source line ranges to copy from `locopilot_monitor.py`:**
- 936-961  `check_hands_below_shoulders` → `pose_checks.py`
- 1629-1643 `check_hand_object_interaction` → `pose_checks.py`
- 963-1081 `detect_writing_by_wrist_proximity` → `writing_fallbacks.py`
- 1083-1168 `detect_writing_by_book_and_posture` → `writing_fallbacks.py`

**Public API (new modules):**

```python
# app/core/utils/pose_checks.py
def check_hands_below_shoulders(pose_landmarks, get_keypoint, logger=None) -> bool: ...
def check_hand_object_interaction(
    hand_coords: tuple[float, float] | None,
    object_bbox: list[int] | None,
    margin: int = 50,
) -> bool: ...
```

```python
# app/core/detectors/writing_fallbacks.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class WritingFallbackThresholds:
    max_wrist_distance: int
    max_single_wrist_distance: int
    max_elbow_distance: int
    writing_required_consecutive: int
    writing_min_duration: float
    person_book_overlap_margin: int
    book_posture_required_consecutive: int
    book_posture_min_duration: float

def detect_writing_by_wrist_proximity(
    *,
    pose_landmarks,
    frame_shape,
    person_idx: int,
    timestamp_sec: float,
    activity_detector,            # exposes calculate_wrist_distance + detect_head_looking_down
    wrist_proximity_tracking: dict,  # mutated in place
    thresholds: WritingFallbackThresholds,
    logger,
) -> bool: ...

def detect_writing_by_book_and_posture(
    *,
    pose_landmarks,
    person_bbox: list[int],
    book_bboxes: list[list[int]],
    person_idx: int,
    timestamp_sec: float,
    activity_detector,
    wrist_proximity_tracking: dict,  # mutated in place; reuses tracking_key=f"book_posture_{person_idx}"
    thresholds: WritingFallbackThresholds,
    bbox_overlap_with_margin_fn: Callable = ...,  # default: app.core.utils.geometry.bbox_overlap_with_margin
    logger,
) -> bool: ...
```

**State contract:**
- The two writing functions mutate `wrist_proximity_tracking` in-place
  (the very same dict the monitor stores on `self`). Both today read keys
  `person_idx` and `f"book_posture_{person_idx}"`; preserve those keys
  exactly.
- `activity_detector` is the existing `app.core.detectors.activity_detector`
  instance; we call `.calculate_wrist_distance()` and
  `.detect_head_looking_down()` on it just like the monitor does.
- All numeric thresholds come from the `WritingFallbackThresholds` dataclass
  built once by the caller from `self.MAX_WRIST_DISTANCE` etc.

**Dependencies on already-extracted modules:** `app.core.utils.geometry`
(uses `bbox_overlap_with_margin`).

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.utils.pose_checks import check_hands_below_shoulders, check_hand_object_interaction
from app.core.detectors.writing_fallbacks import (
    detect_writing_by_wrist_proximity,
    detect_writing_by_book_and_posture,
    WritingFallbackThresholds,
)
print('ok')
"
```
Plus a unit test `tests/refactor/test_writing_fallbacks.py` that:
1. Calls `check_hand_object_interaction((10,10), [0,0,5,5], margin=10)` → True.
2. Builds a fake landmark object + a `WritingFallbackThresholds` and asserts that the wrist-proximity tracker accumulates `consecutive_frames` correctly across 3 successive calls.

---

### T2 — Visualization & video I/O lift

**Title:** Move per-frame drawing helpers and the H.264 reencode/write into
the existing `app.core.visualization` and a new `app.core.media` module so
the monolith no longer carries cv2 putText / VideoWriter code.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/visualization/mediapipe_overlay.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/visualization/evidence_frame_annotator.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/media/__init__.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/media/clip_writer.py`

**Source line ranges to copy:**
- 1238-1349 `draw_mediapipe_outputs` → `mediapipe_overlay.py`
- 1352-1590 `draw_multi_person_mediapipe_outputs` → `mediapipe_overlay.py`
- 3952-4049 `_annotate_evidence_frame` → `evidence_frame_annotator.py`
- 4426-4468 `_reencode_to_h264` → `clip_writer.py`
- 4470-4496 `save_video_clip` → `clip_writer.py`

**Public API:**

```python
# app/core/visualization/mediapipe_overlay.py
def draw_mediapipe_outputs(
    frame, pose_results, face_results, *,
    mp_drawing, mp_pose, mp_face_mesh, mp_drawing_styles,
    pose_sleep_info=None, head_pose_info=None,
    sleep_strong_duration_sec: float = 2.0,
    sleep_microsleep_duration_sec: float = 2.0,
): ...

def draw_multi_person_mediapipe_outputs(
    frame, persons_data, face_results, *,
    mp_drawing, mp_face_mesh, mp_drawing_styles,
    get_keypoint,  # the same self._get_keypoint_by_name used today
): ...
```

```python
# app/core/visualization/evidence_frame_annotator.py
def annotate_evidence_frame(
    frame, *, activity_name: str, frame_number: int,
    object_detector,           # has detect_objects(...)
    yolo_pose,                 # has process(frame); may be None
    get_keypoint,              # function reference
    logger=None,
): ...
```

```python
# app/core/media/clip_writer.py
def reencode_to_h264(input_path: str, *, ffmpeg_path: str = '/usr/bin/ffmpeg', logger=None) -> bool: ...
def save_video_clip(frames: list, output_path: str, fps: float, *, logger=None) -> None: ...
```

**State contract:**
- All four functions are pure / stateless. They take the cv2 frame and the
  MediaPipe / YOLO modules they need by reference. The monolith currently
  reads `self.mp_drawing`, `self.mp_face_mesh` etc. directly; keep the same
  attribute names on the passed-in objects.
- `save_video_clip` calls `reencode_to_h264` after `cv2.VideoWriter.release()`
  exactly as today.

**Dependencies:** none beyond stdlib + cv2 + numpy.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.visualization.mediapipe_overlay import (
    draw_mediapipe_outputs, draw_multi_person_mediapipe_outputs,
)
from app.core.visualization.evidence_frame_annotator import annotate_evidence_frame
from app.core.media.clip_writer import save_video_clip, reencode_to_h264
print('ok')
"
```
Plus `tests/refactor/test_clip_writer.py` that builds a 5-frame 32x32 BGR
clip in memory, calls `save_video_clip(frames, '/tmp/_t.mp4', 1.0)` and
asserts the file exists and ffprobe reports >=5 frames.

---

### T3 — Hand-history bundle: smoothing, velocity, packing motion gate, gesture-coordination window

**Title:** Consolidate per-person hand-position state and the helpers that
read it into a single cohesive `HandHistoryTracker` plus a tiny
`coordination_window` pure helper.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/tracking/hand_history.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/tracking/coordination.py`

**Source line ranges to copy:**
- 1592-1627 `_get_smoothed_hand_position` → `hand_history.py`
- 2406-2501 `analyze_hand_velocity_and_trajectory` → `hand_history.py`
- 2631-2670 `_check_wrist_motion_for_packing` → `hand_history.py`
- 2349-2404 `_check_hand_gesture_coordination` → `coordination.py`

**Public API:**

```python
# app/core/tracking/hand_history.py
class HandHistoryTracker:
    def __init__(
        self,
        *,
        history_max_length: int = 10,
        smoothing_window: int = 3,
        packing_wrist_motion_min_velocity: float = 0.008,
        packing_wrist_motion_gate_enabled: bool = True,
    ): ...

    # tuple-keyed by (person_idx, hand_side)
    smoothing_buffers: dict           # exposed for _cleanup_stale_person_tracking compatibility
    # int-keyed by person_idx
    position_history: dict             # exposed similarly

    def get_smoothed_hand_position(
        self, person_idx: int, hand_side: str, landmark, w: int, h: int, timestamp_sec: float
    ) -> tuple[int, int]: ...

    def analyze_velocity_and_trajectory(
        self, person_idx: int, landmarks, frame_shape, timestamp_sec: float, *, get_keypoint,
    ) -> dict: ...

    def check_wrist_motion_for_packing(self, person_idx: int, timestamp_sec: float) -> bool: ...
```

```python
# app/core/tracking/coordination.py
def check_hand_gesture_coordination(
    *,
    lp_detected: bool,
    alp_detected: bool,
    current_time: float,
    recent_person_activities: dict,        # the live dict from the monitor
    hand_gesture_coordination_window: float,
) -> tuple[bool, bool]:
    """Returns (lp_not_coordinating, alp_not_coordinating)."""
```

**State contract:**
- `HandHistoryTracker.smoothing_buffers` is the exact same dict the monitor
  references as `self.hand_smoothing_buffers`. To preserve the
  `_cleanup_stale_person_tracking` semantics (T7), the rewire in Section 3
  will reassign `self.hand_smoothing_buffers = self._hand_history.smoothing_buffers`
  (alias same dict object).
- `HandHistoryTracker.position_history` is the same dict the monitor
  references as `self.hand_position_history`. Same aliasing strategy.
- `check_hand_gesture_coordination` is pure: it reads `recent_person_activities`
  but never mutates it.

**Dependencies:** none beyond stdlib + numpy.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.tracking.hand_history import HandHistoryTracker
from app.core.tracking.coordination import check_hand_gesture_coordination
t = HandHistoryTracker()
ok = check_hand_gesture_coordination(lp_detected=True, alp_detected=False, current_time=10.0, recent_person_activities={}, hand_gesture_coordination_window=5.0)
assert ok == (False, True)
print('ok')
"
```
Plus `tests/refactor/test_hand_history.py` that pushes 5 fake landmarks and
asserts `analyze_velocity_and_trajectory` reports `analysis_quality='good'`
on the 5th call.

---

### T4 — Pose validators

**Title:** Lift the three pure validators that gate gesture detection into
`app/core/utils/pose_validators.py`.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/utils/pose_validators.py`

**Source line ranges to copy:**
- 1648-1691 `validate_pose_landmarks`
- 1693-1763 `validate_anatomical_consistency`
- 1765-1809 `check_landmark_stability`

**Public API:**

```python
# app/core/utils/pose_validators.py
def validate_pose_landmarks(
    pose_landmarks, *, min_landmarks: int = 10, min_visibility: float = 0.3,
) -> bool: ...

def validate_anatomical_consistency(
    pose_landmarks, frame_shape, *, get_keypoint,
) -> tuple[bool, str]: ...

def check_landmark_stability(
    person_idx: int, pose_landmarks, frame_shape, *,
    history: dict,                  # mutated in place; same shape as today
    max_jump_threshold: float = 100.0,
    get_keypoint,
) -> tuple[bool, float]: ...
```

**State contract:**
- `check_landmark_stability` mutates `history` (the dict the monitor calls
  `self.landmark_stability_history`); pass the live ref so cleanup logic in
  T7 still sees the same object.
- The other two are pure.

**Dependencies:** stdlib only.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.utils.pose_validators import (
    validate_pose_landmarks, validate_anatomical_consistency, check_landmark_stability,
)
print('ok')
"
```
Plus a unit test that builds a stub landmark list of length 9 and asserts
`validate_pose_landmarks(stub, min_landmarks=10) is False`.

---

### T5 — Static-fixture suppression (backpacks + phones)

**Title:** Lift the two parallel static-object trackers into a single class
`StaticObjectFilter` since they share the IoU-NMS-by-frame-count algorithm.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/tracking/static_object_filter.py`

**Source line ranges to copy:**
- 2509-2570 `_update_static_backpack_tracking`
- 2572-2629 `_update_static_phone_tracking`

**Public API:**

```python
# app/core/tracking/static_object_filter.py
class StaticObjectFilter:
    """Tracks bbox stability across frames; objects that remain in the same
    location (IoU >= threshold) for >= min_frames are flagged static and
    filtered out of detection lists."""

    def __init__(
        self,
        *,
        label: str,                 # 'backpack' or 'phone' (used in log lines verbatim)
        iou_threshold: float,
        min_frames: int,
        enabled: bool,
        log_level: str = 'debug',   # 'debug' for backpacks, 'info' for phones (preserves current behavior)
        logger=None,
    ): ...

    candidates: list                # exposed for cleanup/debug

    def filter(self, detections: list) -> list: ...
```

**State contract:**
- The class owns its `candidates` list (replaces `self.static_backpack_candidates`
  and `self.static_phone_candidates`). The rewire constructs two instances
  with the corresponding settings.
- Log lines must remain identical to the source — the executor must keep
  `[STATIC BACKPACK] Suppressed static backpack at ...` and
  `[STATIC PHONE] Suppressed static phone at ... — likely panel instrument`
  byte-identical (operators grep these).
- Uses `calculate_iou` from `app.core.utils.geometry`.

**Dependencies:** `app.core.utils.geometry.calculate_iou`.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.tracking.static_object_filter import StaticObjectFilter
f = StaticObjectFilter(label='backpack', iou_threshold=0.8, min_frames=2, enabled=True)
b = [10,10,30,30]
for _ in range(3):
    out = f.filter([b])
assert out == []
print('ok')
"
```

---

### T6 — Frame sampling + pose batch inference

**Title:** Lift the frame sampler generator and the YOLO pose batch helper
into `app/core/pipeline/frame_sampling.py` and
`app/core/pipeline/pose_batch.py`. Also publish `video_capture_context` from
a tiny shared module so the writing-fallback / evidence-annotator / pose-batch
extracts can share it without re-defining.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/pipeline/frame_sampling.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/pipeline/pose_batch.py`
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/utils/video_io.py`

**Source line ranges to copy:**
- 145-156 `video_capture_context` → `video_io.py`
- 861-930 `sample_video_frames` → `frame_sampling.py`
- 1172-1234 `detect_poses_batch` → `pose_batch.py`

**Public API:**

```python
# app/core/utils/video_io.py
@contextmanager
def video_capture_context(video_path: str): ...
```

```python
# app/core/pipeline/frame_sampling.py
def sample_video_frames(
    video_path: str, *,
    sample_fps: float,
    start_frame: int | None = None,
    end_frame: int | None = None,
    logger=None,
) -> Iterator[tuple[int, float, Any, int]]:
    """Yields (sample_index, timestamp_sec, frame_bgr, frame_idx)."""
```

```python
# app/core/pipeline/pose_batch.py
def detect_poses_batch(
    yolo_pose_adapter,                # exposes .model and .conf_threshold
    frames: list, *,
    batch_size: int = 8,
    conf_threshold: float | None = None,
    device: str = 'cpu',
    logger=None,
) -> list[dict]: ...
```

**State contract:** All three are stateless functions. The pose-batch
function imports `YoloPoseLandmarks` and `PersonKeypoints` from
`app.services.yolo_pose_adapter` lazily (matching today's behavior).

**Dependencies:** cv2, ultralytics (already in env), `app.services.yolo_pose_adapter`.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.utils.video_io import video_capture_context
from app.core.pipeline.frame_sampling import sample_video_frames
from app.core.pipeline.pose_batch import detect_poses_batch
print('ok')
"
```
No video file is required for the smoke test (the generator is not iterated).

---

### T7 — Per-person tracking dict bag + consecutive-detection counter

**Title:** Bundle the cluster of per-person bookkeeping dicts and the small
counter helper into `app/core/tracking/per_person_state.py`. This is the
single owner of the dicts that `_cleanup_stale_person_tracking` walks.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/tracking/per_person_state.py`

**Source line ranges to copy:**
- 825-859  `update_per_person_detection`
- 4092-4141 `_cleanup_stale_person_tracking` (sleep-detector cleanup call stays — it's a method on the SleepDetector instance — see state contract)

**Public API:**

```python
# app/core/tracking/per_person_state.py
from collections import defaultdict

class PerPersonState:
    """Owns the per-person tracking dicts that previously lived as separate
    attributes on LocopilotActivityMonitor. The monitor keeps direct
    attribute aliases for backward compatibility — the dicts are the same
    objects."""

    def __init__(self): ...

    consecutive_detections: defaultdict   # {person_idx: defaultdict(int)} keyed by activity_name
    grace_counters: defaultdict
    hand_position_history: dict
    landmark_stability_history: dict
    wrist_proximity_tracking: dict
    no_pose_sleep_tracking: dict
    recent_person_activities: dict
    hand_smoothing_buffers: dict          # tuple-keyed (person_idx, hand_side)

    def update_consecutive(
        self, person_idx: int, activity_type: str, detected: bool,
        timestamp_sec: float, *,
        required_consecutive: int, grace_frames: int,
    ) -> bool: ...

    def cleanup_stale(
        self, active_person_indices: set[int], *,
        sleep_detector,                   # has cleanup_stale_tracking(active_set)
        logger=None,
    ) -> int:
        """Returns total entries removed."""
```

**State contract:**
- The monitor will, after rewire, store `self._pps = PerPersonState()` and
  re-export the dicts as attributes (`self.recent_person_activities = self._pps.recent_person_activities`,
  etc.) so the gigantic `process_all_persons_activities` (which is staying
  in the monolith) continues to read/mutate the same objects unchanged.
- T3 (`HandHistoryTracker`) and T7 share `hand_position_history` and
  `hand_smoothing_buffers` — the rewire wires the same dict object into both
  by passing `pps.hand_position_history` to the `HandHistoryTracker`
  constructor.
- `cleanup_stale` calls `sleep_detector.cleanup_stale_tracking(active_set)`
  exactly as today; the SleepDetector instance is passed in.

**Dependencies:** none.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.tracking.per_person_state import PerPersonState
pps = PerPersonState()
ok = pps.update_consecutive(0, 'cell_phone', True, 1.0, required_consecutive=1, grace_frames=2)
assert ok is True
print('ok')
"
```
Plus a unit test that builds 3 person indices, then calls
`pps.cleanup_stale({0}, sleep_detector=DummySleepDetector())` and asserts
indices 1 and 2 are gone from every dict.

---

### T8 — Mind-diversion suppression rules

**Title:** Lift the standalone "should we suppress mind diversion?"
function into `app/core/detectors/mind_diversion_suppression.py`. Pure
function over settings + landmarks + recent-activities map.

**Target new files:**
- `/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/app/core/detectors/mind_diversion_suppression.py`

**Source line ranges to copy:**
- 3875-3937 `should_suppress_mind_diversion`

**Public API:**

```python
# app/core/detectors/mind_diversion_suppression.py
def should_suppress_mind_diversion(
    *,
    person_idx: int,
    person_activities: dict,
    pose_landmarks,
    detections: dict,
    frame_shape,
    current_time: float | None,
    settings,                        # exposes mind_diversion_suppress_with_writing,
                                     # mind_diversion_writing_grace_seconds,
                                     # mind_diversion_wrist_distance_threshold
    recent_person_activities: dict,
    get_keypoint,
) -> tuple[bool, str | None]: ...
```

**State contract:** Pure; reads `recent_person_activities` but never
mutates it.

**Dependencies:** numpy.

**Verification:**
```bash
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from app.core.detectors.mind_diversion_suppression import should_suppress_mind_diversion
print('ok')
"
```
Plus a unit test that calls it with `person_activities={'writing': True}`
and asserts `(True, 'suppressed_writing_active')`.

---

## Section 3 — Final Sequential Rewire Task (`TR`)

This is the only step that edits `locopilot_monitor.py`. Run it after every
T-task in Section 2 has landed and verified. Do NOT parallelize.

### TR-1 Imports to add at top of `locopilot_monitor.py`

After the existing block at lines 22-30, add:

```python
from app.core.utils.pose_checks import (
    check_hands_below_shoulders as _check_hands_below_shoulders_fn,
    check_hand_object_interaction as _check_hand_object_interaction_fn,
)
from app.core.utils.pose_validators import (
    validate_pose_landmarks as _validate_pose_landmarks_fn,
    validate_anatomical_consistency as _validate_anatomical_consistency_fn,
    check_landmark_stability as _check_landmark_stability_fn,
)
from app.core.utils.video_io import video_capture_context  # supersedes the local def
from app.core.detectors.writing_fallbacks import (
    detect_writing_by_wrist_proximity as _detect_writing_by_wrist_proximity_fn,
    detect_writing_by_book_and_posture as _detect_writing_by_book_and_posture_fn,
    WritingFallbackThresholds,
)
from app.core.detectors.mind_diversion_suppression import (
    should_suppress_mind_diversion as _should_suppress_mind_diversion_fn,
)
from app.core.tracking.hand_history import HandHistoryTracker
from app.core.tracking.coordination import check_hand_gesture_coordination as _check_hand_gesture_coordination_fn
from app.core.tracking.static_object_filter import StaticObjectFilter
from app.core.tracking.per_person_state import PerPersonState
from app.core.pipeline.frame_sampling import sample_video_frames as _sample_video_frames_fn
from app.core.pipeline.pose_batch import detect_poses_batch as _detect_poses_batch_fn
from app.core.visualization.mediapipe_overlay import (
    draw_mediapipe_outputs as _draw_mediapipe_outputs_fn,
    draw_multi_person_mediapipe_outputs as _draw_multi_person_mediapipe_outputs_fn,
)
from app.core.visualization.evidence_frame_annotator import (
    annotate_evidence_frame as _annotate_evidence_frame_fn,
)
from app.core.media.clip_writer import (
    save_video_clip as _save_video_clip_fn,
    reencode_to_h264 as _reencode_to_h264_fn,
)
```

### TR-2 `__init__` additions

Replace the block that initializes `self.static_backpack_candidates`,
`self.static_backpack_iou_threshold`, etc. (lines 575-590) with two
`StaticObjectFilter` instances:

```python
self._backpack_filter = StaticObjectFilter(
    label='backpack',
    iou_threshold=getattr(settings, 'packing_static_iou_threshold', 0.80),
    min_frames=getattr(settings, 'packing_static_min_frames', 10),
    enabled=getattr(settings, 'packing_static_suppression_enabled', True),
    log_level='debug',
    logger=self.logger,
)
self._phone_filter = StaticObjectFilter(
    label='phone',
    iou_threshold=getattr(settings, 'phone_static_iou_threshold', 0.70),
    min_frames=getattr(settings, 'phone_static_min_frames', 5),
    enabled=getattr(settings, 'phone_static_suppression_enabled', True),
    log_level='info',
    logger=self.logger,
)
```

After the existing `self.hand_smoothing_buffers = {}` line, replace the
loose tracking-dict initialisations with:

```python
self._pps = PerPersonState()
# Backward-compat aliases (same dict objects)
self.consecutive_detections = self._pps.consecutive_detections
self.grace_counters = self._pps.grace_counters
self.per_person_consecutive_detections = self._pps.per_person_consecutive_detections
self.per_person_grace_counters = self._pps.per_person_grace_counters
self.hand_position_history = self._pps.hand_position_history
self.landmark_stability_history = self._pps.landmark_stability_history
self.wrist_proximity_tracking = self._pps.wrist_proximity_tracking
self.no_pose_sleep_tracking = self._pps.no_pose_sleep_tracking
self.recent_person_activities = self._pps.recent_person_activities
self.hand_smoothing_buffers = self._pps.hand_smoothing_buffers
```

Add `self._hand_history = HandHistoryTracker(history_max_length=self.hand_history_max_length, ...)`
sharing the same `hand_position_history` and `hand_smoothing_buffers` dicts.

Build the writing thresholds bundle once:

```python
self._writing_thresholds = WritingFallbackThresholds(
    max_wrist_distance=self.MAX_WRIST_DISTANCE,
    max_single_wrist_distance=self.MAX_SINGLE_WRIST_DISTANCE,
    max_elbow_distance=self.MAX_ELBOW_DISTANCE,
    writing_required_consecutive=self.WRITING_REQUIRED_CONSECUTIVE,
    writing_min_duration=self.WRITING_MIN_DURATION,
    person_book_overlap_margin=self.PERSON_BOOK_OVERLAP_MARGIN,
    book_posture_required_consecutive=self.BOOK_POSTURE_REQUIRED_CONSECUTIVE,
    book_posture_min_duration=self.BOOK_POSTURE_MIN_DURATION,
)
```

### TR-3 Method bodies to delete and replace with thin wrappers

Replace each method body listed below with a one-/two-line wrapper that
forwards to the extracted function. Keep the public method signature and
return type identical (so `process_all_persons_activities` and
`_process_frames_core` keep working unchanged).

| Method | New body |
|---|---|
| `check_hands_below_shoulders` | `return _check_hands_below_shoulders_fn(pose_landmarks, get_keypoint=self.get_keypoint, logger=self.logger)` |
| `check_hand_object_interaction` | `return _check_hand_object_interaction_fn(hand_coords, object_bbox, margin)` |
| `detect_writing_by_wrist_proximity` | one-line forward to `_detect_writing_by_wrist_proximity_fn(...)` passing `self.activity_detector`, `self.wrist_proximity_tracking`, `self._writing_thresholds`, `self.logger`. |
| `detect_writing_by_book_and_posture` | same pattern, forward all kwargs. |
| `validate_pose_landmarks` | forward to `_validate_pose_landmarks_fn` with class defaults (`self.MIN_POSE_LANDMARKS`, `self.MIN_POSE_VISIBILITY`). |
| `validate_anatomical_consistency` | forward to `_validate_anatomical_consistency_fn(... get_keypoint=self.get_keypoint)`. |
| `check_landmark_stability` | forward to `_check_landmark_stability_fn(... history=self.landmark_stability_history, max_jump_threshold=self.max_landmark_jump_threshold, get_keypoint=self.get_keypoint)`. |
| `_get_smoothed_hand_position` | `return self._hand_history.get_smoothed_hand_position(...)`. |
| `analyze_hand_velocity_and_trajectory` | `return self._hand_history.analyze_velocity_and_trajectory(... get_keypoint=self.get_keypoint)`. |
| `_check_wrist_motion_for_packing` | `return self._hand_history.check_wrist_motion_for_packing(person_idx, timestamp_sec)`. |
| `_check_hand_gesture_coordination` | forward to `_check_hand_gesture_coordination_fn(...)` passing `self.recent_person_activities`, `self.hand_gesture_coordination_window`. |
| `_update_static_backpack_tracking` | `return self._backpack_filter.filter(backpack_detections)`. |
| `_update_static_phone_tracking` | `return self._phone_filter.filter(phone_detections)`. |
| `update_per_person_detection` | forward to `self._pps.update_consecutive(...)` building required_consecutive/grace_frames from `self.activity_thresholds[activity_type]`. |
| `_cleanup_stale_person_tracking` | `self._pps.cleanup_stale(active_person_indices, sleep_detector=self.sleep_detector, logger=self.logger)`. |
| `sample_video_frames` | forward to `_sample_video_frames_fn(video_path, sample_fps=self.sample_fps, start_frame=start_frame, end_frame=end_frame, logger=self.logger)`. |
| `detect_poses_batch` | forward to `_detect_poses_batch_fn(self.yolo_pose, frames, batch_size=batch_size, conf_threshold=conf_threshold, device=self.yolo_device, logger=self.logger)`. |
| `draw_mediapipe_outputs` | forward to `_draw_mediapipe_outputs_fn` passing `self.mp_drawing`, `self.mp_pose`, `self.mp_face_mesh`, `self.mp_drawing_styles`, `self.SLEEP_STRONG_DURATION`, `self.SLEEP_MICROSLEEP_DURATION`. |
| `draw_multi_person_mediapipe_outputs` | forward to `_draw_multi_person_mediapipe_outputs_fn` passing the mp* refs and `get_keypoint=self.get_keypoint`. |
| `_annotate_evidence_frame` | forward to `_annotate_evidence_frame_fn` passing `self.object_detector`, `self.yolo_pose`, `self._get_keypoint_by_name`, `self.logger`. |
| `_reencode_to_h264` | `return _reencode_to_h264_fn(input_path, logger=self.logger)`. |
| `save_video_clip` | `return _save_video_clip_fn(frames, output_path, fps, logger=self.logger)`. |
| `should_suppress_mind_diversion` | forward to `_should_suppress_mind_diversion_fn(... settings=self.settings, recent_person_activities=self.recent_person_activities, get_keypoint=self.get_keypoint)`. |

Also delete the local `def video_capture_context(video_path):` at lines
145-156 — the import from `app.core.utils.video_io` supersedes it.

### TR-4 Smoke-test command

After TR-1..TR-3 land, run from repo root with the test env:

```bash
cd /Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System
/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11 -c "
from locopilot_monitor import LocopilotActivityMonitor, video_capture_context
print('imports ok')
# Construction smoke-test against a missing video path is enough — __init__
# does NOT open the video (CR-011 lazy-load), it only loads models.
m = LocopilotActivityMonitor(
    'example_data/latest.mp4',
    output_dir='/tmp/_refactor_smoke',
    save_annotated_frames=False,
    sample_fps=0.5,
    create_run_dir=False,
    preloaded_models=None,
)
# Touch the rewired call sites with stub inputs:
assert m.check_hand_object_interaction((10,10), [0,0,5,5], margin=10) is True
assert m.check_hand_object_interaction((100,100), [0,0,5,5], margin=10) is False
print('basic methods ok')
print('lines:', sum(1 for _ in open('locopilot_monitor.py')))
"
```

Then run the existing ground-truth precision/recall regression on a single
short video to confirm the activity counts are unchanged from a pre-rewire
baseline (operator-supplied; document the baseline as part of TR's success
criteria).

---

## Section 4 — Risks and methods left in the monolith

These are intentionally NOT extracted this round. Lifting them would force
either an opaque "context object" with 30+ fields (defeating the purpose) or
a circular dependency between extracted modules. They are flagged for a
future, larger refactor.

| Method | Lines | Why it stays |
|---|---|---|
| `__init__` | 179-762 | Wires up the singleton-like model + detector graph; needs to stay where the class declaration is. |
| `process_all_persons_activities` | 2682-3865 | 1183 lines of orchestration. Reads/writes >25 self attrs (settings, all detector instances, all per-person dicts, `recent_person_activities`, `static_*`, `wrist_proximity_tracking`, `_writing_last_book_seen`, `consecutive_detections`, `no_pose_sleep_tracking`). After T1-T8 land, the method body itself doesn't change but it now calls into the new modules through the thin wrappers. A future task could split this into `pipeline/per_person_pipeline.py` with a `PerPersonContext` dataclass. |
| `_process_frames_core` | 4519-5089 | 570 lines of frame-level orchestration: face mesh, object detection, dedupe, person-roles, train motion, multi-person dispatch, gates (train-stopped suppression, sleep-overrides), activity-map lifecycle. Touches 30+ self attrs and references 12+ self methods. Same future-task candidate as above. |
| `start_activity` | 4051-4090 | Mutates 9 keys on `self.activities[name]` and reads `self.frame_idx_buffer`, `self.current_motion_state`. Tightly coupled to `end_activity`; lift them together later. |
| `end_activity` | 4159-4424 | 266 lines, touches 14+ self attrs (activities, evidence_counter, all_activities, crew_*, trip_id, evidence_clips_dir, sample_fps, video_path, settings, consecutive_detections, grace_counters, evidence_manager, ACTIVITY_REGISTRY). Future "ActivityRecorder" extraction. |
| `process_video` / `process_video_range` | 5091-5172, 5174-5325 | The two outer loops. Already small; not worth the extraction churn now. |
| `cleanup` / `__del__` | 5327-5385 | Tied to instance attributes loaded in `__init__`. |
| `_get_video_metadata` | 4143-4157 | 15 lines, owns instance-level cache. Trivial; not worth a module. |
| `set_trip_schedule`, `_extract_ocr_timestamp`, `get_keypoint`, `extract_video_segment`, `_match_pose_to_roles`, `calculate_head_pose_angles`, `identify_person_roles`, `analyze_packing_hand_motion`, `generate_summary_report` | Various | Already either trivial setters or one-line delegates to existing extracted modules. |

### Expected outcome

Once T1-T8 + TR land, `locopilot_monitor.py` should drop from 5439 lines to
roughly **2100-2400 lines** (we lift ~3000 lines of pure helpers, drawing,
clip I/O, and per-person bookkeeping). Reaching <1500 lines requires the
follow-up split of `process_all_persons_activities` and
`_process_frames_core`, which is a separate planning round.
