# Locopilot Monitoring System

CCTV-based activity monitoring for Indian railway locomotive pilots (LP) and
assistant loco pilots (ALP). Reads overhead cabin video, detects
safety-critical activities (sleeping, microsleep, cell-phone use,
eating/drinking, writing in logbook, packing bags, hand gestures,
mind-diversion, etc.), and posts confirmed violations to the customer's
mindcoinapps API.

---

## High-level architecture

Two pipelines, in series:

```
Pipeline 1 — classical CV (the original detector)
   Frames → YOLO v8 (person + objects) → Pose (yolo / rtmpose / rtmw)
        → Per-detector rule engine (sleep, writing, gesture, …)
        → Train-motion gate (suppresses non-safety-critical when STOPPED)
        → Temporal filtering (consecutive frames + grace periods)
        → activities.json + evidence clips/frames

Pipeline 2 — VLM verification layer (added 2026-04-26)
   For each confirmed activity in {writing, eating_drinking, packing_bags}
   send the keyframe + activity-specific prompt to Qwen2.5-VL-7B-AWQ
   served by vLLM on :8001. Verdict (TP/FP/UNCERTAIN) + reasoning attached
   to the activity as a `vlm_review` block.

Then: S3 upload → external mindcoinapps API → response
```

Pipeline 1 remains the source of truth for what counts as a violation.
Pipeline 2 only filters (drops FALSE_POSITIVE @ confidence>=VLM_DROP_THRESHOLD
and annotates every verified activity with `vlm_review`); it never adds
violations. Verifier is fail-open — if vLLM is down, Pipeline-1 verdicts
pass through unchanged.

---

## Repository layout

```
locopilot_monitor.py            Core ~3200-line frame processor (LocopilotActivityMonitor).
                                The 1,200-line process_all_persons_activities was extracted
                                to app/core/multi_person_runner.py in the 2026-05-09 cleanup;
                                the monolith now delegates to it via a one-line shim.
gunicorn_config.py              Workers pinned to 1 (per-process GPU singleton, see C-1)
deploy-gpu.sh                   rsync + restart on the GPU box
start_server.sh                 Local dev launcher
requirements.txt
.env / .env.production / .env.example   pydantic-settings config sources

app/
  main.py                       FastAPI app (Gunicorn entrypoint)
  controllers/
    video_controller.py         POST /api/v1/video/process-and-upload, jobs queue, media routes
  services/
    video_processing_service.py Orchestrates a single video run
    activity_detection_service.py
    job_manager.py              Queue (3 workers) for /api/video/jobs
    gpu_resource_manager.py     Per-process MAX_CONCURRENT_VIDEOS gate (lazy init)
    image_preprocessing_service.py  Bilateral filter + CLAHE + (optional) regional gamma
    yolo_pose_adapter.py        Adapter pattern: yolo / rtmpose / rtmw all share interface
    s3_upload_service.py        Evidence uploads (clips + jpegs)
    minio_service.py
    external_api_service.py     POST cvvr/cvvrTripViolations/addUpdateBulk
    ocr_timestamp_service.py    Reads time overlay from CCTV frames
    vlm_verification_service.py 12-line back-compat shim. Real code lives in vlm/ below.
    vlm/                        Pipeline-2 verifier package (split out 2026-05-09):
      service.py                  VlmVerificationService orchestrator + telemetry
      vlm_client.py               HTTP client + circuit breaker
      keyframe_processor.py       Keyframe resolve / supplement / stitch / bbox count
      image_encoder.py            ROI detect, crop, base64 encode
      verdict_parser.py           JSON parse, calibration, motion-state logic
      motion_classifier.py        2026-05-10: OCR-based camera detection +
                                  per-camera window-ROI frame-diff motion check.
                                  Runs BEFORE the VLM call to drop train-stationary
                                  writing/eating FPs that vibration-detector and
                                  VLM-text both miss. See "Window-region motion
                                  classifier" section below.
    concurrent_activity_grouping_service.py
  core/
    activity_registry.py        Single source of truth for activity types,
                                consecutive-frame requirements, margins, evidence rules
    activity_tracker.py         Re-exports ActivityConfig from activity_registry.py.
    multi_person_runner.py      MultiPersonActivityRunner — extracted 2026-05-09 from
                                LocopilotActivityMonitor.process_all_persons_activities.
                                Per-frame multi-person detector dispatch.
    gates.py                    apply_train_stopped_suppression (single place that enforces
                                "writing/sleep/etc. only count while train is RUNNING")
    detectors/
      sleep_detector.py         12-line back-compat shim. Real code in sleep/ below.
      sleep/                    Sleep detector package (split out 2026-05-09):
        detector.py               SleepDetector class + detect_pose_based_sleep
        pose_geometry.py          Head tilt / wrist distance / movement score helpers
        state_machine.py          DROWSY state machine
        ir_fallback.py            IR forward-lean fallback
        haar_eye_closure.py       Haar-cascade eye-closure fallback
      gesture_detector.py       Raise-hold-lower trajectory + RTMW hand-shape (optional)
      object_detector.py        YOLO wrapper, zone suppression, SAHI (opt-in)
      train_motion_detector.py  Vibration-based RUNNING/STOPPED/UNCERTAIN
      mind_diversion_detector.py  Head yaw/pitch toward window vs forward
      activity_detector.py      Writing posture + hand-near-book + cell-phone proximity
  utils/
    config.py                   pydantic Settings — every flag below lives here
    video_multiprocessing.py    Pass-1 parallel chunked workers, then Pass-2 sequential
                                temporal filter (deterministic two-pass, see notes below)
    logger.py
  models/                       pydantic request/response models
  repositories/
    activity_repository.py      Reads/writes activities.json
deploy/
  locopilot-vlm.service         Systemd unit for the vLLM server (Pipeline-2)
  README-vlm.md                 VLM rollout/rollback runbook
tests/
  ground_truth/                 Hand-labelled violations per video (precision/recall regression)
.claude/
  vlm_spike/                    Phase-0 50-frame benchmark fixture + verdicts (don't delete)
```

---

## GPU server

| | |
|--|--|
| IP / port | `103.116.80.162` / `3781` |
| User | `admin1` (password in `server details.txt`, never commit) |
| App path | `/opt/poc2` |
| Venv | `/opt/poc2/venv/bin/python3` (3.12) |
| GPU | NVIDIA RTX 4000 Ada Generation, 20 475 MiB |
| CUDA | 12.8 driver / 12.0 toolchain |
| Disk | 730 GB on `/`, ~520 GB free at `/opt/poc2` |
| Logs | `/opt/poc2/logs/LocopilotMonitoring.log` |
| Evidence | `/opt/poc2/locopilot_evidence/run_<timestamp>/` |
| Uploads | `/tmp/locopilot_uploads/<tripId>_<ts>.mp4` |
| HF cache | `~admin1/.cache/huggingface/hub/` (Qwen2.5-VL-7B + AWQ already present) |

### Live services

| Unit | Purpose | Listening | Memory |
|---|---|---|---|
| `locopilot.service` | Gunicorn + FastAPI + the detector | :8000 | ~7-8 GB GPU peak |
| `locopilot-vlm.service` | vLLM serving Qwen2.5-VL-7B-AWQ | :8001 | ~10 GB GPU resident |

GPU budget: ~10 GB vLLM + ~8 GB detector peak + ~3 GB margin = fits in 20 GB.

### One-worker rule (C-1)
`gunicorn_config.py` pins `workers = 1`. `GPUResourceManager` is a per-process
singleton enforcing `MAX_CONCURRENT_VIDEOS`. N workers would N-multiply the cap
and risk GPU OOM. Don't change without redesigning the resource manager.

---

## Deployment flow

```bash
# Standard deploy
./deploy-gpu.sh                     # rsync + copy .env.production → .env + restart

# Manual (when you only changed one file)
sshpass -p "$PASS" scp -P 3781 path/to/file.py admin1@103.116.80.162:/opt/poc2/path/to/
ssh ... "cat /tmp/.sp | sudo -S systemctl restart locopilot.service"
```

`.env.production` is what runs on the server; `.env` is local dev. `.env.example`
is the template — keep it in sync with new settings or new contributors will
miss flags.

---

## Configuration (env vars)

Every flag is in `app/utils/config.py` (pydantic Settings). Below are the
groups that matter most. See `.env.example` for the full list with comments.

### Detection / pipeline
| Flag | Default | Purpose |
|---|---|---|
| `POSE_MODEL` | `yolo` | `yolo` / `rtmpose` / `rtmw` (whole-body w/ hand keypoints) |
| `YOLO_ALWAYS_PREPROCESS` | `1` | Bilateral + CLAHE for every frame (not just dark) |
| `ZONE_SUPPRESSION_ENABLED` | `1` | Suppress static objects (chairs, suitcases in fixed zones) |
| `SAHI_ENABLED` | `0` | Sliced inference for small-object recall (needs `pip install sahi`) |
| `GESTURE_TRAJECTORY_ENABLED` | `1` | Raise-hold-lower state machine for hand-gesture FP cut |
| `HAAR_EYE_DETECTION_ENABLED` | `0` | Off by default; non-functional from overhead. Reclined posture covers it |
| `WRITING_VISUAL_DETECTION_ENABLED` | `0` | HSV paper segmentation, opt-in |

### Train-motion gate
| Flag | Default | Purpose |
|---|---|---|
| `TRAIN_MOTION_DETECTION_ENABLED` | `1` (prod) | Vibration + window-flow detector that emits RUNNING/STOPPED/UNCERTAIN |
| `TRAIN_MOTION_RUNNING_GROUP_THRESHOLD` | `5` | `>5` people in cab → `group_detected` (3-person supervisor visits OK) |

When STOPPED: sleep, writing, packing_bags, lp/alp_hand_gesture, mind_diversion,
eating_drinking are all suppressed. microsleep + cell_phone remain active
(safety-critical even at stations).

The schedule-aware "rules engine" that fetched live train schedules from
RailRadar + etrain.info to distinguish scheduled-halt vs unscheduled-stop
windows was deleted in the 2026-05-09 architecture cleanup (it was confirmed
not customer-facing — `no_person_detected` is internal passthrough state, not
a posted violation). Suppression now applies on every STOPPED period from the
vibration detector. Stale env vars `TRAIN_MOTION_RULES_ENABLED`, `ETRAIN_*`,
`TRIP_API_*` in `.env.production` are silently ignored (`extra="ignore"` in
pydantic Settings).

### VLM verifier (Pipeline-2)
| Flag | Default | Purpose |
|---|---|---|
| `VLM_VERIFICATION_ENABLED` | `0` (1 in prod after 2026-04-26) | Master switch |
| `VLM_BASE_URL` | `http://localhost:8001/v1` | vLLM endpoint |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | |
| `VLM_VERIFY_ACTIVITIES` | `writing,eating_drinking` | Allowlist; non-listed activities pass through |
| `VLM_DROP_THRESHOLD` | `0.80` | Min confidence to drop a FALSE_POSITIVE. Set >1.0 to record verdicts without dropping |
| `VLM_TIMEOUT_SECONDS` | `8.0` (15 in prod) | Per-call HTTP timeout; on timeout the activity passes through |
| `VLM_MAX_ACTIVITIES_PER_RUN` | `0` | Cap (0=no cap) |

### Other
- `MEDIA_API_KEY` — gates `/api/jobs/{run_id}/media` and `/api/status`. Currently unset → "rollout mode" warnings in log.
- `CVVR_API_ENABLED` — toggles posting to mindcoinapps.

---

## Activity registry

Single source of truth: `app/core/activity_registry.py`.

| Code | Key | Description | Evidence rule |
|--|--|--|--|
| 1 | `sleep` | Sleeping | EAR + head-drop + reclined-posture |
| 2 | `microsleep` | Micro-sleep | Same gate, shorter duration |
| 3 | `cell_phone` | Cell phone use | `phone_in_hand` |
| 4 | `lp_hand_gesture` | LP hand gesture | Trajectory + coordination |
| 5 | `writing` | WRITING LOG BOOK WHILE RUNNING | `hand_near_book_or_wrist_proximity` |
| 6 | `packing_bags` | Packing bags | `wrist_inside_backpack_bbox_or_hand_near_backpack` |
| 7 | `mind_diversion` | Looking out window | yaw/pitch toward window |
| 8 | `alp_hand_gesture` | ALP hand gesture | Same as #4, ALP role |
| 11 | `no_person_detected` | (passthrough activity, not a violation) |
| 13 | `eating_drinking` | Eating/drinking | Trajectory classifier OR cup/bottle proximity |

`required_consecutive` and `min_duration` per activity are tuned in
`activity_registry.py` and have caused most of the recent precision wins —
e.g. cell_phone consecutive bumped 1→3 to drop momentary face-touches.

---

## Two-pass deterministic pipeline (for multiprocessing)

To get reproducible results when `useMultiprocessing=true`:

- **Pass 1** (parallel chunk workers): each worker calls
  `process_video_range_raw()` → returns raw per-frame detection dicts (no
  temporal smoothing). Avoids per-worker temporal state divergence.
- **Pass 2** (sequential, main process):
  `TemporalFilteringService.apply_temporal_filtering()` consumes the merged
  raw stream and produces the final activities.json.

Single-process path (`process_video`, `process_video_range`) is unchanged —
only the multi-process path goes through the two-pass split.

Known limitation: hand-gesture coordination + sleep state machine are still
per-worker, so raw flags at chunk boundaries may differ from single-process.
Temporal filtering is fully deterministic.

`mp_overlap_seconds` validator: must cover both
`sleep_baseline_calibration_window` and `hand_gesture_coordination_window` or
the app fails fast at startup. See `app/utils/config.py:_validate_overlap_window`.

---

## Train-motion gate (`app/core/gates.py`)

`apply_train_stopped_suppression(aggregated, persons_data, suppressed=…)` is
the **single place** that zeroes out activities when the train is STOPPED. It
mutates BOTH the aggregated booleans AND each per-person `activities` dict so
downstream consumers (overlays, debug) stay consistent (ARCH-08b).

Default suppressed set:
```
sleep, writing, packing_bags, lp_hand_gesture, alp_hand_gesture,
mind_diversion, eating_drinking
```
microsleep + cell_phone are never suppressed.

---

## VLM verifier (Pipeline-2) — full reference

### Why it exists
Pipeline-1 over-fires on writing & eating: hand-near-book proximity is
indistinguishable from hand-on-brake-handle in pixel space, and a radio
handset to the face looks like a bottle. A 7B VLM can read scene context
("LP is operating the brake handle, not writing") and drop those FPs.

Phase-0 spike (2026-04-26, 50 hand-curated frames, `.claude/vlm_spike/`):

| Activity | TP kept | FPs flagged | Notes |
|---|---|---|---|
| Writing (35) | 1 | 34 | Brake-handle and idle-lap nailed |
| Eating (10) | 1 | 9 | Radio-handset confounder caught |
| Packing (5) | 5 | 0 | All initially passed; e2e later found a controls-reach FP |

End-to-end live (`run_20260426_085645`, 5-activity TV22_10 trip):

```
[vlm] verification stats: verified=5 kept=5 dropped=0 uncertain=0
      skipped_unavailable=0 parse_errors=0
```

Latencies: cold first call ~13 s (Marlin kernel JIT), steady-state 1.6-2.0 s
per activity.

### Architecture
- vLLM runs as `locopilot-vlm.service` on :8001, OpenAI-compatible API.
- `app/services/vlm_verification_service.py` is a stateless HTTP client.
- Hook is in `app/controllers/video_controller.py` between Pipeline-1 finishing
  and S3 upload (lines ~752-810). Runs while local clip/image paths are still
  valid; rewrites `activities.json` with `vlm_review` blocks; in enforcement
  mode also filters `result['clip_files']` so dropped activities' clips
  aren't uploaded.
- Singleton service init (lazy on first call).
- Fail-open: timeout / connection refused / non-2xx → `status:
  "SKIPPED_VLM_UNAVAILABLE"` is recorded and the activity passes through.
- objectType normalization: `"packing bags"` (with space, what writers emit)
  matches `"packing_bags"` (registry key) via `.replace(" ", "_")`.

### Activity-specific prompts
Per-activity prompt in `vlm_verification_service.py` enumerates the
confounders observed in the codebase. Output is forced JSON and parsed with
tolerance for ```code-fence``` wrappers. Today: writing, eating_drinking,
packing_bags, cell_phone are wired. To add a new activity:
1. Add prompt constant
2. Add to `_PROMPTS_BY_OBJECT_TYPE`
3. Add to `VLM_VERIFY_ACTIVITIES` env var

### vlm_review block shape (what lands in activities.json)
```json
"vlm_review": {
  "status": "OK",                   // or SKIPPED_*, PARSE_ERROR
  "verdict": {
    "verdict": "FALSE_POSITIVE",    // TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN
    "confidence": 0.8,
    "primary_object_in_hand": "brake_or_throttle",
    "book_visible_on_desk": true,
    "hand_actually_on_book": false,
    "reasoning": "LP is holding the brake handle, not interacting with the log book."
  },
  "latency_sec": 1.97,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
}
```

### Rollout state (current production)
Enforcement enabled (`VLM_DROP_THRESHOLD=0.80`) for writing, eating_drinking,
and packing_bags. To roll back to observe-only without disabling the
verifier, raise `VLM_DROP_THRESHOLD` above 1.0. See `deploy/README-vlm.md`.

### Operations
```bash
# Service health
sudo systemctl status locopilot-vlm.service
curl -sf http://localhost:8001/v1/models | jq .

# Live verifier output
grep -E '\[vlm\]|\[VLM\]' /opt/poc2/logs/LocopilotMonitoring.log | tail -30

# Disable verifier without stopping vLLM
sed -i 's/^VLM_VERIFICATION_ENABLED=1/VLM_VERIFICATION_ENABLED=0/' /opt/poc2/.env
sudo systemctl restart locopilot.service

# Stop vLLM (verifier auto fail-opens)
sudo systemctl stop locopilot-vlm.service
```

---

## Window-region motion classifier (Pipeline-2 pre-VLM)

`app/services/vlm/motion_classifier.py` — added 2026-05-10 after manual
review of a 12-video batch (`run_20260510_103222`..`104537`) showed 5 of
6 posted writing violations were train-stationary FPs.

### Why it exists

Two upstream gates are unreliable for stop-state detection on this
trainset:

1. **Pipeline-1 vibration motion** is fooled by diesel idle. The user's
   own deep-research note documents this: RUNNING vib median 4.77,
   STOPPED vib median 2.01 — bimodal but the diesel idle keeps
   STOPPED-state vibration above the running threshold. So
   `motionState=RUNNING` ships even when the train is at a station.

2. **VLM motion verdict** is unreliable for ROI-cropped activities
   (writing/eating/packing/cell_phone). The keyframe stitcher crops to
   the person+object bbox before sending — the cabin window (the only
   motion cue) is REMOVED from the VLM input. Qwen2.5-VL-7B-AWQ then
   confabulates a stock phrase ("FRAME 3: motion blur in right window")
   for nearly every activity regardless of state — observed identical
   verbatim string across 6 different scenes.

The motion classifier runs BEFORE the VLM call, on the *uncropped*
keyframes the verifier already loaded. If the cabin's window region is
static across the keyframe burst, the activity is dropped as a
FALSE_POSITIVE without spending a VLM call.

### How it works

1. **Camera detection (OCR, once per source video, cached):** EasyOCR
   on the bottom-right text overlay. The CCTV stamps `CAB 1 ALP camera 3`
   or `CAB 1 LP camera 2` at ~y=560-590. EasyOCR introduces character
   substitutions (`CAB`→`CAU`, `ALP`→`ALR`, `camera`→`amera` or `era`),
   so the regex matches loosely on the trailing digit:
   - "amera 3" / "ALR" / "ALP" → `ALP_CAM3`
   - "amera 2" / bare "LP" (not part of "ALP") → `LP_CAM2`

   First call OCRs every keyframe in the burst until one matches (LP
   body sometimes occludes the overlay on a single frame). Subsequent
   activities for the same source video are O(1) cache lookup.

2. **Window ROI per camera:**
   ```python
   CAMERA_WINDOW_ROIS = {
       "ALP_CAM3": (850, 0, 960, 400),  # upper-right grille
       "LP_CAM2":  (0,   0, 200, 400),  # upper-left window
   }
   ```

3. **Motion score:** mean abs pixel diff in the ROI between consecutive
   keyframes, taken as the median across all consecutive pairs (5
   keyframes → 4 pairs → 1 score).

4. **Decision:** `score < MOTION_DIFF_STOPPED_THRESHOLD` (10.0) →
   synthetic FALSE_POSITIVE verdict with `confidence=0.85`,
   `train_appears_to_be="stopped"`, log line:
   ```
   [vlm] WINDOW-MOTION DROP activity type=5 at t=665.00
   (camera=ALP_CAM3 window_diff=6.60 below threshold=10.00, 4 keyframe pairs)
   ```

### Threshold tuning provenance

Tuned on a 7-clip batch (1 TP `vid11_527s` + 6 stationary FPs across
`run_20260510_084423` through `run_20260510_085654`):

| Activity (window-ROI median diff) | GT | score |
|---|---|---|
| vid01 @ 415s | FP_stopped | 3.24 |
| vid04 @ 965s | FP_stopped | 7.33 |
| vid06a @ 653s | FP_stopped | 25.15 (people-on-platform outlier) |
| vid06b @ 665s | FP_stopped | 6.60 |
| vid06c @ 725s | FP_stopped | 7.55 |
| vid07 @ 1257s | FP_stopped | 0.86 |
| **vid11 @ 527s** | **TP_running** | **13.15** |

Threshold of 10.0 catches 5 of 6 FPs cleanly; vid06a's 25.15 isn't caught
but Pipeline-1 already flagged it `motionState=STOPPED`, so the existing
API motion filter at the boundary excludes it. Net: all 6 FPs suppressed
end-to-end while preserving the only TP.

### Scope and fall-open behaviour

- Only runs for `object_type not in _FULL_FRAME_OBJECT_TYPES`, i.e.
  writing/eating/packing/cell_phone/sleep/mind_diversion. For full-frame
  types (solo_person/no_person/group_detected) the VLM already sees the
  window in the un-cropped strip and can read motion textually — no
  override needed.
- `classify_motion()` returns `None` (and verifier proceeds with the
  normal VLM call) when:
  - OCR can't identify the camera (pattern not matched on any keyframe)
  - Fewer than 2 keyframes are available
  - cv2 fails to read a keyframe
  This makes the gate fail-open: a broken classifier never adds FPs,
  it only fails to subtract them.

### Operations

```bash
# See classifier decisions
grep 'WINDOW-MOTION' /opt/poc2/logs/LocopilotMonitoring.log | tail -20

# Inspect a specific activity's classifier output
jq '.[] | select(.activityStartTime=="665.00") | .vlm_review' \
  /opt/poc2/locopilot_evidence/run_<id>/activities.json

# Disable temporarily by removing the classify_motion call in
# app/services/vlm/service.py around line ~1245 — there is no env flag
# for this yet (intentionally; a future ARCH task should add
# VLM_WINDOW_MOTION_ENABLED if rollback discipline becomes important).
```

### Limitations and what's NOT solved

- **Door-open + people-on-platform scenes** (vid06a archetype): people
  motion through the door inflates the window-ROI diff above threshold,
  so the classifier doesn't fire. Caught only by Pipeline-1's separate
  vibration verdict + the API-boundary motion filter.
- **New camera install:** if a different cabin uses cameras other than
  ALP cam 3 / LP cam 2, the camera detection returns `None` and no
  override happens. Add the camera + ROI to `CAMERA_WINDOW_ROIS` and
  the regex to `_ALP_TOKENS` / `_LP_TOKENS`.
- **Threshold tuned on a 7-clip sample.** A larger labelled corpus
  could justify either raising or lowering it; the current value sits
  in the middle of a genuine but tight gap (TP=13.15 vs FP_max=7.55)
  and may need adjustment as more labelled data arrives.

---

## API surface

```
POST /api/v1/video/process-and-upload   multipart, runs full pipeline + S3 + external API
POST /api/video/jobs                    enqueue job (3 worker queue)
GET  /api/video/jobs/{id}               status
GET  /api/video/jobs/{id}/result        result
POST /api/video/jobs/{id}/cancel
GET  /api/video/queue/status
GET  /api/jobs/{run_id}/media/{file}    serve clip/image (gated by MEDIA_API_KEY when set)
GET  /health
```

Form params for `process-and-upload`: `video_file` (required), `tripId`
(required), `division`, `subFolderName`, `authToken`, crew names/ids,
`useMultiprocessing`, `useMockDetection`, `saveClips`, `trainNumber`,
`tripDate`, `videoStartTime`.

`result` dict from `process_video` returns: `runDirectory` (camelCase!),
`activitiesJsonPath`, `activities`, `activitiesCount`, `processingTime`,
`summary`, `clipsGenerated`. Use `runDirectory` not `run_dir` when reading
this dict — earlier code paths look for `run_dir` and silently fall through.

---

## Common debugging

```bash
# Latest run
ls -t /opt/poc2/locopilot_evidence | head -3
cat /opt/poc2/locopilot_evidence/run_<latest>/activities.json | jq '.[].des' | sort | uniq -c

# Live processing
tail -f /opt/poc2/logs/LocopilotMonitoring.log | grep -E 'Progress:|Activity|ERROR|\\[vlm\\]'

# Sleep FP investigation
grep -E 'HEAD DROP DEBUG|Pose-Based Sleep|Sleep State Machine' /opt/poc2/logs/LocopilotMonitoring.log

# Train motion / suppression
grep 'Train STOPPED' /opt/poc2/logs/LocopilotMonitoring.log | tail -20

# What the verifier did on a run
jq '.[] | {t: .activityStartTime, type: .activityType, vlm: .vlm_review.verdict}' \
  /opt/poc2/locopilot_evidence/run_<id>/activities.json
```

---

## Known FP patterns (catalogued in commits + memory)

- **Head tilt angle wrapping**: `atan2` discontinuity caused 300+ deg drops
  on small (distant) persons. Fixed with `(delta + 180) % 360 - 180` and a
  `nose_y_drop >= 0` guard in `sleep_detector.py`.
- **Small LP bbox**: noisy pose at distance triggered phantom sleep — same
  guard handles it.
- **Hand-near-book = brake-handle**: Pipeline-1 can't distinguish these; the
  VLM verifier exists specifically to catch this archetype.
- **Radio-handset = phone**: `cell_phone` confidence raised 0.30→0.40 + radio
  zone suppression + VLM verifier as the final filter.
- **Writing posture = idle hands in lap**: pose-only fallback is permissive;
  VLM verifier rejects "no book interaction".
- **YOLO suitcase FP on control panel**: confidence raised 0.45→0.65.
- **Hand-gesture FP from single-person frames**: requires 2 persons visible
  before the coordination check fires.
- **Baseline maturity**: `head_drop` is ignored when `timestamp_sec < 10.0`
  (calibration window).

---

## Testing

- **Ground truth**: `tests/ground_truth/<video>.json` — hand-labelled
  violations with OCR + video-second timestamps. Used to measure
  precision/recall regression.
- **Spike fixture**: `.claude/vlm_spike/` — 50 frame benchmark + verdicts
  from Phase-0. Re-runnable via `vlm_spike/run_benchmark.py` against a vLLM
  endpoint.
- Local-test python env (Apple Silicon Mac, has cv2 + ultralytics):
  `/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11`. Requires
  `numpy<2`, `opencv-python-headless==4.10.0.84`.

---

## Conventions / gotchas

- **`runDirectory` vs `run_dir`**: process_video returns `runDirectory`
  (camelCase). Several controller paths look for `run_dir` first and fall
  through silently — use `result.get('runDirectory') or result.get('run_dir')
  or result.get('runDir') or ''` defensively.
- **numpy in JSON**: Pipeline-1 outputs occasionally contain numpy float32
  values that `json.dump` can't serialize. Use `default=str` on any
  `json.dump` of activities (the VLM hook does this).
- **objectType has spaces**: written as `"packing bags"` not `"packing_bags"`.
  Match either.
- **Singleton services**: external_api, vlm_verification, gpu_resource_manager,
  job_manager all use thread-safe double-checked-locking singleton init.
- **Env var validators**: pydantic `@model_validator(mode='after')` catches
  incoherent flag combinations at startup (e.g. POSE_MODEL=rtmpose needs
  rtmlib importable; absolute YOLO weight paths must exist).
- **No emojis in commits or code** unless explicitly asked; logs use
  bracketed prefixes (`[vlm]`, `[OK]`, `[ERROR]`) for grep-ability.
- **Don't bump gunicorn workers** above 1 without redesigning
  `GPUResourceManager` — it's a per-process singleton.

---

## Quick rollback recipes

| Scenario | Action |
|---|---|
| VLM verifier misbehaving | `VLM_VERIFICATION_ENABLED=0` in `.env` + `systemctl restart locopilot` |
| vLLM eating GPU | `systemctl stop locopilot-vlm.service` (verifier fail-opens) |
| Bad code deploy | `git checkout HEAD~1 -- <files>` + `systemctl restart locopilot` |
| External API spam | `CVVR_API_ENABLED=0` |
| Disable a single activity | Remove its key from `VLM_VERIFY_ACTIVITIES` (verifier) or its required_consecutive in `activity_registry.py` (Pipeline-1) |

---

## Pointers to deeper docs

- `deploy/README-vlm.md` — VLM rollout / GPU-sharing / ops cheatsheet
- `tests/ground_truth/README.md` — GT format + scoring conventions
- `.env.example` — every settable flag with comments
- `tasks/code-review-critical-fixes.md` — historical critical-fix log (C-1, C-9, etc.)
- `docs/specs/architecture-cleanup/PLAN.md` — 2026-05-09 three-wave cleanup
  (god-class splits, dormant-pipeline deletion, MultiPersonRunner extraction);
  task specs in `docs/specs/architecture-cleanup/tasks/0001..0008-*.md`.
