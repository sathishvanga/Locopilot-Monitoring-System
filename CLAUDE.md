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
locopilot_monitor.py            Core ~5200-line frame processor (LocopilotActivityMonitor)
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
    trip_data_service.py
    etrain_delay_service.py     etrain.info live status (for arrival window)
    vlm_verification_service.py NEW: Pipeline-2 verifier (Qwen2.5-VL via vLLM)
    concurrent_activity_grouping_service.py
  core/
    activity_registry.py        Single source of truth for activity types,
                                consecutive-frame requirements, margins, evidence rules
    gates.py                    apply_train_stopped_suppression (single place that enforces
                                "writing/sleep/etc. only count while train is RUNNING")
    detectors/
      sleep_detector.py         EAR + reclined posture + head-tilt + state machine
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
| `TRAIN_MOTION_RULES_ENABLED` | `1` (prod) | Engine that suppresses non-safety-critical activities while STOPPED |
| `TRAIN_MOTION_DETECTION_ENABLED` | `1` (prod) | Required by the rules engine; vibration + window-flow detector |
| `TRAIN_MOTION_RUNNING_GROUP_THRESHOLD` | `5` | `>5` people in cab → `group_detected` (3-person supervisor visits OK) |

When STOPPED: sleep, writing, packing_bags, lp/alp_hand_gesture, mind_diversion,
eating_drinking are all suppressed. microsleep + cell_phone remain active
(safety-critical even at stations).

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
- `ETRAIN_ENABLED` — fetch live train status from etrain.info.

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
  incoherent flag combinations at startup (e.g. TRAIN_MOTION_RULES_ENABLED=1
  needs TRAIN_MOTION_DETECTION_ENABLED=1).
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
