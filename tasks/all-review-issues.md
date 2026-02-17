# Locopilot Monitor - Code Review Issues

> Generated: 2026-02-16 | Reviewed: locopilot_monitor.py, video_multiprocessing.py, all detectors, all services
> Total: 11 Critical | 24 High | 25 Medium | 12 Low

---

## CRITICAL (11)

### C-01: Cross-person detection leak via shared `detections` dict
- **File:** `locopilot_monitor.py:3050-3053`
- **Category:** Detection Correctness
- **Description:** `detections['cell_phone'].extend(person_detections['cell_phone'])` mutates the shared `detections` dict. When processing person 0, ROI detections are appended. When processing person 1, person 0's detections are still in the list, causing false positives for person 1.
- **Fix:** Use per-person scoped detection lists: `person_cell_phones = detections['cell_phone'] + person_detections['cell_phone']` and use that for the person's activity checks instead of mutating the shared dict.

### C-02: Temporal state discontinuity at chunk boundaries
- **File:** `app/utils/video_multiprocessing.py:468-473`
- **Category:** Multiprocessing Correctness
- **Description:** Each worker creates a fresh `LocopilotActivityMonitor` with zeroed temporal state (consecutive_detections=0, sleep state=AWAKE, no baseline calibration). Activities spanning chunk boundaries are split or lost entirely. A 20s sleep episode split across two 15s chunks may not meet `min_duration` or `required_consecutive` in either chunk.
- **Fix:** Implement the two-pass pipeline: Pass 1 workers emit raw per-frame detection booleans, Pass 2 runs sequential temporal filtering in the main process. Alternative: add overlap regions where each chunk processes N extra seconds from the previous chunk to warm up state.

### C-03: Conflicting chunk duration defaults (15s vs 6s)
- **File:** `app/utils/config.py:75`, `app/utils/multiprocessing_config.py:30`
- **Category:** Configuration
- **Description:** `Settings.mp_chunk_duration=15.0` vs `MultiprocessingConfig.chunk_duration_seconds=6.0`. The actual runtime value depends on the call path. `ActivityDetectionService` uses 15s, direct `MultiprocessingConfig()` construction gets 6s. The comment says "15s chunks ensure hand gesture coordination detection works correctly."
- **Fix:** Eliminate the duplicate. `MultiprocessingConfig.chunk_duration_seconds` should always be supplied from `Settings.mp_chunk_duration`. Remove the hardcoded default in `MultiprocessingConfig`.

### C-04: All GPU workers load duplicate models -- OOM risk
- **File:** `app/utils/video_multiprocessing.py:146-167`
- **Category:** Resource Management
- **Description:** When `YOLO_DEVICE=0` (GPU), every worker process loads both YOLO11m and YOLO11m-pose onto the same GPU. With `max_workers_cap=10`, this means up to 10 copies (500MB-1GB each). On RTX 4000 Ada (20GB) with VLM using ~9.4GB, only ~10.6GB remains -- 10 workers will OOM.
- **Fix:** When using GPU, cap `max_workers` to 4-6 (GPU inference is already parallelized via CUDA). Add validation in `get_num_workers()` that caps workers when `yolo_device != 'cpu'`.


### C-06: `eating_drinking` missing from ACTIVITY_REGISTRY and all maps
- **File:** `locopilot_monitor.py:72-136, 670-682, 685-697, 700-712`
- **Category:** Detection Correctness
- **Description:** Per memory Fix 12, `eating_drinking` should be independent activity type 13. But it's NOT in `ACTIVITY_REGISTRY`, `activity_type_map`, `activity_descriptions`, or `evidence_rules`. Currently piggybacked on `mind_diversion` as a sub-type. Rule engine also missing it.
- **Fix:** Add `eating_drinking` to `ACTIVITY_REGISTRY`, all 4 maps, `activities_map`, and the rule engine service's `ACTIVITY_NAMES` + `ALLOWED_WHEN_STOPPED`.

### C-07: Hardcoded S3 upload API URL
- **File:** `app/services/s3_upload_service.py:28`
- **Category:** Configuration / Security
- **Description:** `self.api_url = "https://api.mindcoinapps.com/ai_demo_api/amazonUpload/uploadWithFolder"` is hardcoded. Cannot be changed per environment without code modification.
- **Fix:** Move to `config.py` settings as `s3_upload_api_url` and reference via `self.settings.s3_upload_api_url`.

### C-08: Voting cache stores raw video frames -- up to 1.9GB RAM
- **File:** `app/services/voting_verification_service.py:48, 98-119`
- **Category:** Memory
- **Description:** LRU cache (max 32 entries) stores raw numpy frames. With 10 frames of 1080p per entry (~60MB each), max capacity consumes ~1.9GB.
- **Fix:** Reduce `max_size` to 4-8, add memory budget check, or cache only inference results (not raw frames).

### C-09: `os.getenv` calls bypass pydantic-settings -- dual config sources
- **File:** `locopilot_monitor.py:539, 547-548, 585`
- **Category:** Code Quality
- **Description:** `CELL_PHONE_CONFIDENCE`, `GPU_BATCH_SIZE`, `GPU_BATCH_ENABLED`, `HAND_GESTURE_COORDINATION_WINDOW` read directly from `os.getenv` instead of `self.settings`. Creates dual sources of truth.
- **Fix:** Add these fields to `Settings` class in `config.py` and access exclusively through `self.settings`.

### C-10: ObjectDetector / YOLOHandler ~80% code duplication
- **File:** `app/core/detectors/object_detector.py` (821 lines), `app/core/models/yolo_handler.py` (1055 lines)
- **Category:** Code Quality / Maintenance Risk
- **Description:** Near-identical implementations of `detect_objects()`, `detect_objects_in_rois_batch()`, `detect_objects_batch()`, `validate_object_aspect_ratio()`, `_boxes_overlap_or_near()`, `get_roi_around_keypoint()`, `preprocess_frames_for_detection()`. Bug fixes in one may not propagate to the other.
- **Fix:** Consolidate into a single class. Either `YOLOHandler` contains all logic and `ObjectDetector` wraps it, or extract shared logic into a common base class.

### C-11: Unbounded `per_person_tracking` defaultdict in SleepDetector
- **File:** `app/core/detectors/sleep_detector.py:92`
- **Category:** Memory Leak
- **Description:** `defaultdict(self._create_tracking_dict)` grows without bounds across long videos. No cleanup method exists. In a 24/7 monitoring scenario with shifting person indices, this will grow indefinitely.
- **Fix:** Add `cleanup_stale_tracking(active_person_indices)` method and call it from the monitor's existing cleanup cycle.

---

## HIGH (24)

### H-01: Sleep vs writing suppression chicken-and-egg
- **File:** `locopilot_monitor.py:4444-4471`
- **Category:** Detection Correctness
- **Description:** Writing suppresses sleep, but `sleep_state_overrides_writing` only activates when state machine is already DROWSY+. On the first frame of sleep onset, state machine is still ALERT, so writing suppresses sleep, preventing the state machine from ever advancing.
- **Fix:** Add override condition checking drowsiness indicators directly from `pose_sleep_info` (e.g., `is_reclined_sleep`, `nose_y_drop < -0.05`) at lines 4449-4456.

### H-02: Sleep state machine gate checks only one person (global)
- **File:** `locopilot_monitor.py:4428-4442`
- **Category:** State Machine
- **Description:** Gate iterates over all `persons_data` and sets `state_machine_ready = True` if ANY person is DROWSY+. Person 0's SLEEPING state lets person 1's microsleep bypass the gate.
- **Fix:** Apply state machine gate per-person in `process_all_persons_activities` before aggregation.

### H-03: Hand gesture velocity gate computed but result discarded
- **File:** `locopilot_monitor.py:2488-2499`
- **Category:** False Positives
- **Description:** Velocity analysis (`rapid_raise_detected`) is computed at significant cost but its result is only logged, not used to gate the gesture detection return value.
- **Fix:** Wire `rapid_raise_detected` as a required condition, or increase `required_consecutive` when velocity is low.

### H-04: `recent_person_activities` dict grows without bounds
- **File:** `locopilot_monitor.py:580`
- **Category:** Memory Leak
- **Description:** Entries added at lines 3476-3478, 3515-3517, etc. but NOT in `_cleanup_stale_person_tracking`'s `tracking_dicts` list.
- **Fix:** Add `('recent_person_activities', self.recent_person_activities)` to `tracking_dicts` list in `_cleanup_stale_person_tracking`.

### H-05: `alp_not_standing` bypasses ACTIVITY_REGISTRY
- **File:** `locopilot_monitor.py:803-812`
- **Category:** Code Quality
- **Description:** Manually added to tracking dicts after registry-based init, defeating the single-source-of-truth pattern.
- **Fix:** Add `'alp_not_standing': ActivityConfig(required_consecutive=2, grace_frames=3)` to `_build_activity_registry()` and remove manual init.

### H-06: Two-pass pipeline documented but not implemented
- **File:** `app/utils/video_multiprocessing.py`
- **Category:** Architecture
- **Description:** `process_video_range_raw()` does not exist. `temporal_filtering_service.py` exists only as stale `.pyc`. The deterministic results goal is NOT achieved.
- **Fix:** Implement the two-pass pipeline, or update project memory and document current single-pass limitations.

### H-07: Silent failure on partial multiprocessing results
- **File:** `app/utils/video_multiprocessing.py:741-774`
- **Category:** Safety / Reliability
- **Description:** Failed worker chunks are logged but processing continues with partial results. A safety-critical system silently dropping entire video segments is a significant risk.
- **Fix:** Retry failed ranges with single-process fallback, or propagate failure metadata to API response.

### H-08: No timeout on `future.result()` -- hangs possible
- **File:** `app/utils/video_multiprocessing.py:748`
- **Category:** Reliability
- **Description:** If a worker hangs (deadlock in OpenCV, stuck GPU), main process waits indefinitely.
- **Fix:** Add per-chunk timeout: `future.result(timeout=chunk_duration * 10)`.

### H-09: Shared pool has no `atexit` shutdown handler
- **File:** `app/utils/video_multiprocessing.py:231-264`
- **Category:** Resource Management
- **Description:** `get_shared_pool()` creates a global `ProcessPoolExecutor` with no `atexit` handler. Worker processes may become orphaned on application exit.
- **Fix:** Register `atexit.register(shutdown_shared_pool)` when creating the shared pool.

### H-10: Config defaults evaluated at import time via `os.getenv`
- **File:** `app/utils/multiprocessing_config.py:23-52`
- **Category:** Configuration
- **Description:** All `MultiprocessingConfig` fields use `os.getenv()` in `@dataclass` defaults, evaluated at module import time. `.env` values loaded later by pydantic-settings won't be reflected.
- **Fix:** Move `os.getenv` calls into `__post_init__` or convert to pydantic-settings.

### H-11: Worker init failure causes silent model re-loading
- **File:** `app/utils/video_multiprocessing.py:420`
- **Category:** Multiprocessing
- **Description:** When `_worker_models` is `None` (failed init), monitor attempts to reload all models from scratch in the worker, causing severe performance degradation instead of a clear failure.
- **Fix:** Check `_worker_models` at start of `process_frame_range` and fail fast with `RuntimeError`.

### H-12: `lru_cache` on `get_settings()` + relative `.env` path in spawn workers
- **File:** `app/utils/config.py:490-500`
- **Category:** Configuration
- **Description:** `Settings` reads from `.env` via relative path. Spawned workers inherit parent's cwd, but if cwd changes before spawning, `.env` silently falls back to defaults.
- **Fix:** Set `env_file` to an absolute path in `SettingsConfigDict`.

### H-13: Double cleanup of monitor in workers
- **File:** `app/utils/video_multiprocessing.py:480, 504`
- **Category:** Code Quality
- **Description:** `monitor.cleanup()` called at line 480 (normal exit) and again in the `finally` block (line 504).
- **Fix:** Remove explicit cleanup at line 480, rely solely on `finally` block.

### H-14: `get_keypoint` function duplicated across 4 detectors
- **File:** `sleep_detector.py:269`, `gesture_detector.py:106`, `activity_detector.py:101`, `mind_diversion_detector.py:104`
- **Category:** Code Duplication
- **Description:** 4 different implementations with different error handling and fallback maps. SleepDetector has the most complete version with `fallback_map` for MediaPipe keypoints.
- **Fix:** Create canonical `get_keypoint` in `app/core/utils/pose_utils.py`, have all detectors import it.

### H-15: `calculate_wrist_distance` duplicated in SleepDetector and ActivityDetector
- **File:** `sleep_detector.py:433-519`, `activity_detector.py:118-201`
- **Category:** Code Duplication
- **Description:** Nearly identical implementations with slightly different hardcoded vs configurable thresholds.
- **Fix:** Extract into shared utility in `app/core/utils/pose_utils.py`.

### H-16: Gesture detection pixel thresholds not resolution-normalized
- **File:** `app/core/detectors/gesture_detector.py:40-48`
- **Category:** False Positives / False Negatives
- **Description:** `WRIST_SHOULDER_VERTICAL_MIN=80`, `ARM_EXTENSION_MIN=20`, etc. are absolute pixel values. At 4K, 80px is tiny; at 480p, 80px is enormous. `_scale_margin` exists in monitor but not in this detector.
- **Fix:** Normalize by person bbox height: `threshold = max(20, int(bbox_height * 0.12))`.

### H-17: Person tracker role assignment by bbox area is fragile
- **File:** `app/core/tracking/person_tracker.py:111-112`
- **Category:** Detection Correctness
- **Description:** "Largest bbox = closest to camera" breaks when one person is standing and one sitting, camera is overhead, or person is partially out of frame.
- **Fix:** Add secondary signal: bbox bottom-edge y-coordinate, or make the role assignment metric configurable.

### H-18: Sleep score threshold default still 3 (should be 5)
- **File:** `app/core/detectors/sleep_detector.py:1108`
- **Category:** False Positives
- **Description:** `getattr(self.settings, 'sleep_score_threshold', 3)` -- fallback is `3` but Feb 14 Fix 1 tightened it to `5`.
- **Fix:** Change default to `5`: `getattr(self.settings, 'sleep_score_threshold', 5)`.

### H-19: No retry logic for S3 uploads -- processing results lost
- **File:** `app/services/s3_upload_service.py`
- **Category:** Reliability
- **Description:** Single HTTP request with no retry. Network hiccups or 503 responses permanently lose the expensive processing output.
- **Fix:** Add exponential backoff retry (3 retries) for retriable status codes (429, 500, 502, 503).

### H-20: No retry logic for external API calls
- **File:** `app/services/external_api_service.py:103, 189`
- **Category:** Reliability
- **Description:** Both `_post_no_events` and `_post_violations` make single POST requests. If CVVR API is temporarily unavailable, all violation data for the trip is permanently lost.
- **Fix:** Add retry with exponential backoff. Consider dead-letter queue for persistent failures.

### H-21: Synchronous video processing blocks async event loop
- **File:** `app/controllers/video_controller.py:288`
- **Category:** Performance / Reliability
- **Description:** `video_processing_service.process_video(...)` called synchronously from `async def` endpoint. Blocks the entire event loop for minutes during ML inference.
- **Fix:** Wrap in `asyncio.get_running_loop().run_in_executor(None, ...)`.

### H-22: `preload_app=True` with gunicorn + CUDA contexts
- **File:** `gunicorn_config.py:73`
- **Category:** Reliability
- **Description:** GPU model initialization runs in master process, then workers are forked. CUDA contexts cannot be forked safely -- tensors and streams become invalid in children.
- **Fix:** Set `preload_app = False` or ensure no CUDA operations happen at preload time.

### H-23: Rule engine missing `eating_drinking` and `alp_not_standing`
- **File:** `app/services/rule_engine_service.py:39, 53, 63, 370`
- **Category:** Detection Correctness
- **Description:** `ACTIVITY_NAMES`, `ALLOWED_WHEN_STOPPED`, `ALWAYS_VIOLATION`, `_get_activity_type` don't include these activities. Returns `UNKNOWN` for them.
- **Fix:** Add `eating_drinking` to maps + `ALLOWED_WHEN_STOPPED`. Add `alp_not_standing` to `ALWAYS_VIOLATION`.

### H-24: GPU models not thread-safe for concurrent inference
- **File:** `app/services/gpu_resource_manager.py`
- **Category:** Concurrency
- **Description:** `get_models()` returns shared model references. YOLO models have internal state. Two concurrent `model.predict()` calls on the same instance can corrupt results.
- **Fix:** Ensure semaphore serializes all inference, or maintain a pool of model instances.

---

## MEDIUM (25)

### M-01: Packing bags detection breaks out of backpack loop too early
- **File:** `locopilot_monitor.py:3479, 3524`
- **Category:** Detection Correctness
- **Description:** `break` exits the backpack loop after the first match attempt. If first backpack doesn't match but second would, it's never checked.
- **Fix:** Only break on positive detection, continue to next backpack on failed motion analysis.

### M-02: Legacy global sleep state reset is dead code
- **File:** `locopilot_monitor.py:4470-4471`
- **Category:** Code Quality
- **Description:** `self.pose_sleep_start = None` and `self.pose_sleep_duration = 0` reset global (legacy) tracking but don't affect `SleepDetector.per_person_tracking`.
- **Fix:** Remove these global resets or also reset per-person tracking in SleepDetector.

### M-03: `hand_smoothing_buffers` grows without bounds
- **File:** `locopilot_monitor.py:651`
- **Category:** Memory Leak
- **Description:** Keys are `(person_idx, hand_side)` tuples. Cleanup uses integer keys so can't clean these.
- **Fix:** Add special cleanup for tuple-keyed dicts, or restructure as `{person_idx: {'right': ..., 'left': ...}}`.

### M-04: Control zone pixel thresholds not resolution-scaled
- **File:** `locopilot_monitor.py:2363-2405`
- **Category:** False Positives
- **Description:** `30 < wrist_shoulder_vertical < 100`, `wrist_elbow_distance < 50` are pixel-based. At 4K these are tiny; at 480p they're proportionally large.
- **Fix:** Scale relative to person bbox height using `_scale_margin`.

### M-05: `packing` vs `packing_bags` naming inconsistency
- **File:** `locopilot_monitor.py:3018, 4608, 3600`
- **Category:** Code Quality
- **Description:** `person_activities` uses `'packing'` but `activities_map` uses `'packing_bags'`. An `activity_key_map` bridges this.
- **Fix:** Standardize on one name everywhere.

### M-06: Magic numbers in gesture detection
- **File:** `locopilot_monitor.py` (multiple lines)
- **Category:** Code Quality
- **Description:** 15+ hardcoded pixel thresholds (250, 30, 100, 50, 80, -30, 20, 150, etc.) scattered through detection methods.
- **Fix:** Extract into `Settings` class or class-level named constants.

### M-07: Broad `except Exception` swallows stack traces
- **File:** `locopilot_monitor.py:3653-3655`
- **Category:** Code Quality
- **Description:** Catches ANY exception during per-person processing with `continue`. Subtle bugs (KeyError, etc.) are silently swallowed.
- **Fix:** Log with `exc_info=True`: `self.logger.error(f"Error: {e}", exc_info=True)`.

### M-08: VideoCapture not using context manager in `end_activity`
- **File:** `locopilot_monitor.py:4004-4007`
- **Category:** Resource Management
- **Description:** If `cap.read()` raises, `cap.release()` is never called. The `video_capture_context` exists for this.
- **Fix:** Use `with video_capture_context(self.video_path) as cap:`.

### M-09: `_worker_config` is dead code
- **File:** `app/utils/video_multiprocessing.py:48, 224`
- **Category:** Code Quality
- **Description:** Set in `worker_initializer` but never read anywhere.
- **Fix:** Remove.

### M-10: `config_dict` always empty -- dead parameter
- **File:** `app/utils/video_multiprocessing.py:718, 725`
- **Category:** Code Quality
- **Description:** Empty dict passed to every worker, accepted but never used.
- **Fix:** Remove from both `process_frame_range` and `process_video_parallel`.

### M-11: Thread over-subscription risk
- **File:** `app/utils/multiprocessing_config.py:37-38, 63-84, 92-100`
- **Category:** Performance
- **Description:** 10 workers x 3 torch threads x 3 opencv threads = massive over-subscription on 12-core machine.
- **Fix:** Synchronize all thread counts through `set_worker_env_vars` calculation.

### M-12: `mp.set_start_method(force=True)` is global and unnecessary
- **File:** `app/utils/video_multiprocessing.py:571`
- **Category:** Code Quality
- **Description:** Globally sets start method, but `get_shared_pool` already uses `mp.get_context()`. Could break other multiprocessing code.
- **Fix:** Remove. The context-based approach at line 252 is correct.

### M-13: Sleep state machine has no SLEEPING -> MICROSLEEP transition
- **File:** `app/core/detectors/sleep_detector.py:616-628`
- **Category:** State Machine
- **Description:** Once in SLEEPING, only exit is `has_hand_activity -> ALERT`. A person who briefly lifts head but doesn't move hands stays in SLEEPING.
- **Fix:** Add partial recovery: `elif not is_sustained_low_eyes and not is_head_down: new_state = 'MICROSLEEP'`.

### M-14: Cell phone visibility not checked before pixel conversion
- **File:** `app/core/detectors/activity_detector.py:443-447`
- **Category:** False Positives
- **Description:** Wrist coordinates converted to pixels without visibility check. Low-visibility keypoints may have nonsensical coordinates.
- **Fix:** Add `if right_wrist.visibility < threshold: right_hand_coords = None`.

### M-15: MindDiversionDetector shoulder/ear visibility not checked
- **File:** `app/core/detectors/mind_diversion_detector.py:186-196`
- **Category:** False Positives
- **Description:** Only nose visibility checked. Low-visibility shoulders/ears produce unreliable yaw/pitch angles.
- **Fix:** Check visibility of shoulders and ears (>= 0.3) before computing angles.

### M-16: PersonTracker loses person indices on role update
- **File:** `app/core/tracking/person_tracker.py:267`
- **Category:** Detection Correctness
- **Description:** `_prev_person_boxes` stored as list. Non-contiguous indices (e.g., {0, 2}) cause enumerate mismatch.
- **Fix:** Store as dict keyed by person index: `{idx: info['bbox'] for idx, info in person_roles.items()}`.

### M-17: Face mesh indices accessed without bounds checking
- **File:** `app/core/detectors/mind_diversion_detector.py:349-356`
- **Category:** Robustness
- **Description:** Assumes 455+ landmarks. If `refine_landmarks=False` or model changes, indices could be out of bounds.
- **Fix:** Add `if len(face_lm) < 468: return`.

### M-18: Gesture coordination timing race condition
- **File:** `app/core/detectors/gesture_detector.py:482-506`
- **Category:** Detection Correctness
- **Description:** Sessions updated before coordination check uses old timing values, but session_info uses new values. Inconsistent state.
- **Fix:** Capture all timing values either before or after session updates consistently.

### M-19: Full-frame YOLO call without confidence filter
- **File:** `app/core/detectors/object_detector.py:390-395`
- **Category:** Performance
- **Description:** No `conf` parameter passed. YOLO uses default 0.25, producing many low-confidence detections filtered later. Wastes GPU memory and CPU time.
- **Fix:** Pass `conf=0.20` floor confidence.

### M-20: Hardcoded 1920x1080 frame dimensions in person_tracker
- **File:** `app/core/tracking/person_tracker.py:345-346`
- **Category:** Detection Correctness
- **Description:** `frame_w = max(bbox[2], 1920)` as "estimate". Wrong for 720p or 4K.
- **Fix:** Add `frame_shape` parameter to `match_pose_to_roles`.

### M-21: Writing wrist distance 300px not scale-normalized
- **File:** `app/core/detectors/activity_detector.py:283-296`
- **Category:** False Positives / False Negatives
- **Description:** 300px absolute threshold. At 4K close wrists appear >300px; at 480p distant wrists appear <300px.
- **Fix:** Normalize by person bbox height.

### M-22: Error messages expose internal details in production
- **File:** `app/main.py:305`
- **Category:** Security
- **Description:** Exception messages (file paths, database details) returned to clients in 500 responses.
- **Fix:** Return generic messages in production; log full errors internally.

### M-23: In-memory job storage unbounded growth
- **File:** `app/services/job_manager.py:85`
- **Category:** Memory
- **Description:** `self._jobs` dict grows indefinitely. `cleanup_completed_jobs` exists but never called automatically.
- **Fix:** Add periodic cleanup task or `max_retained_jobs` limit.

### M-24: Side window motion service is stateful and not thread-safe
- **File:** `app/services/side_window_motion_service.py:101`
- **Category:** Concurrency
- **Description:** `self._prev_roi` stores state for frame-to-frame optical flow. Concurrent video jobs corrupt each other's state.
- **Fix:** Make stateless (require `prev_frame` param) or use per-session state objects.

### M-25: 11 singleton getters lack thread-safe locking
- **File:** Multiple service files
- **Category:** Concurrency
- **Description:** Simple global variable singletons with no `threading.Lock()` across s3_upload, rule_engine, train_motion_resolver, side_window_motion, ocr_timestamp, trip_data, alp_alertness, etrain_delay, external_api, concurrent_grouping, minio services.
- **Fix:** Add `threading.Lock()` to each singleton getter.

---

## LOW (12)

### L-01: `temporal_suppression_window` hardcoded at 10s
- **File:** `locopilot_monitor.py:581`
- **Description:** Not configurable via settings. Memory notes it should be 5s (Fix 7) but code shows 10.0.
- **Fix:** Move to `self.settings.temporal_suppression_window`.

### L-02: Full frame copy for optical flow
- **File:** `locopilot_monitor.py:4578`
- **Description:** `self._prev_motion_frame = frame.copy()` stores full BGR frame (~6MB at 1080p). Optical flow only needs grayscale.
- **Fix:** Store `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` -- 66% memory reduction.

### L-03: Inconsistent indentation in return statement
- **File:** `locopilot_monitor.py:2502-2529`
- **Description:** LP/ALP gesture result dicts have inconsistent indentation.
- **Fix:** Fix to consistent 4-space indentation.

### L-04: Duplicate `import json` in video_multiprocessing
- **File:** `app/utils/video_multiprocessing.py:9, 831`
- **Description:** Module-level import at line 9, redundant import inside function at line 831.
- **Fix:** Remove line 831.

### L-05: `NumpyEncoder` missing `np.bool_` handling
- **File:** `app/utils/video_multiprocessing.py:34-43`
- **Description:** Handles `np.integer`, `np.floating`, `np.ndarray` but not `np.bool_`. YOLO outputs may contain numpy booleans.
- **Fix:** Add `if isinstance(obj, np.bool_): return bool(obj)`.

### L-06: Nested `_get_smoothed_hand_position` function
- **File:** `locopilot_monitor.py:3342-3378`
- **Description:** Defined as nested function inside long method. Captures `self` from closure, harder to test independently.
- **Fix:** Extract to proper class method.

### L-07: `chin` variable extracted but never used
- **File:** `app/core/detectors/mind_diversion_detector.py:353`
- **Description:** `chin = face_lm[152]` assigned but never referenced.
- **Fix:** Remove.

### L-08: `bag_max_aspect_ratio` of 1.2 may reject legitimate bags
- **File:** `app/core/detectors/object_detector.py:96`
- **Description:** Requires bags to be almost square. Duffel bags (ratio 1.5-2.0) would be filtered.
- **Fix:** Increase to 2.0 or make configurable.

### L-09: DEBUG messages logged at INFO level in ObjectDetector
- **File:** `app/core/detectors/object_detector.py:249-254`, `app/core/models/yolo_handler.py:602-618`
- **Description:** `[DEBUG ROI]` messages at INFO level. At 0.5 FPS with 8 ROIs x 2 persons = 8 INFO messages per frame.
- **Fix:** Change to `self.logger.debug()`.

### L-10: No-op string operation in YoloPoseAdapter
- **File:** `app/services/yolo_pose_adapter.py:301`
- **Description:** `name_lower = name_lower.replace('_', '_')` replaces underscores with underscores.
- **Fix:** Remove.

### L-11: Gamma LUT recomputed on every frame
- **File:** `app/services/image_preprocessing_service.py:243`
- **Description:** Lookup table rebuilt for every frame even when gamma value hasn't changed.
- **Fix:** Cache LUT by gamma value using `lru_cache` or dict.

### L-12: `_find_overlapping_groups` is dead code
- **File:** `app/services/concurrent_activity_grouping_service.py:153`
- **Description:** Union-Find implementation never called. Service uses `_find_minute_groups` instead.
- **Fix:** Remove or mark as legacy.

---

## RECOMMENDED PRIORITY ORDER

### Phase 1: Critical Correctness (Week 1)
1. **C-01**: Cross-person detection leak (active FP source)
2. **H-03**: Velocity gate not wired (active FP source)
3. **H-01**: Sleep vs writing suppression (active FN source)
4. **H-02**: Per-person state machine gate (active FP source)
5. **H-18**: Sleep score threshold default 3->5 (active FP source)
6. **C-06**: eating_drinking missing from registries

### Phase 2: Multiprocessing Fixes (Week 2)
7. **C-02**: Chunk boundary state loss (two-pass or overlap)
8. **C-03**: Conflicting chunk duration defaults
9. **C-04**: GPU OOM risk -- cap workers for GPU mode
10. **H-08**: Future result timeout
11. **H-09**: atexit handler for shared pool
12. **H-11**: Worker init failure -- fail fast

### Phase 3: Security & Reliability (Week 3)
13. **C-05**: CORS wildcard + credentials
14. **C-07**: Hardcoded S3 URL
15. **H-19**: S3 upload retry logic
16. **H-20**: External API retry logic
17. **H-21**: Sync event loop blocking
18. **M-22**: Error detail leakage

### Phase 4: Memory & Performance (Week 4)
19. **C-08**: Voting cache memory (1.9GB)
20. **C-11**: Unbounded per_person_tracking
21. **H-04**: recent_person_activities cleanup
22. **M-03**: hand_smoothing_buffers cleanup
23. **M-23**: Job storage cleanup

### Phase 5: Code Quality & Deduplication (Ongoing)
24. **C-09**: os.getenv -> settings consolidation
25. **C-10**: ObjectDetector / YOLOHandler dedup
26. **H-14**: get_keypoint dedup (4 implementations)
27. **H-15**: calculate_wrist_distance dedup
28. **H-05**: alp_not_standing -> ACTIVITY_REGISTRY
29. **H-16**: Gesture pixel threshold normalization
30. All remaining Medium and Low issues
