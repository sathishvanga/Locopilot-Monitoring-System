# Locopilot Monitoring System — Project Notes for Claude

Detects risky driver/crew activities (sleep, cell phone, writing, packing,
gestures, mind-diversion) from cab video and posts verified verdicts to the
external CVVR API. Designed for ~0.5 FPS sampling on an RTX 4000 Ada.

---

## Two-pipeline architecture

| Pipeline | Role | Where | Output |
|---|---|---|---|
| **Pipeline-1** | On-device CV detection (YOLO + pose + per-person rules) | `locopilot_monitor.py`, `app/core/detectors/`, `app/core/multi_person_runner.py` | `activities.json` |
| **Pipeline-2** | VLM (vision-language model) verification — false-positive filter on Pipeline-1's verdicts | `app/services/vlm/` | Updates `activities.json` with VLM verdicts, posts to CVVR |

Pipeline-2 is *fail-open*: if VLM is unavailable, Pipeline-1's verdicts pass
through unchanged.

### Pipeline-1 stages (per-frame loop)

```
Frame Sampling (~0.5 FPS)         app/core/pipeline/frame_sampling.py
        ↓
YOLO Object Detection             locopilot_monitor.py:400 (ObjectDetector)
        ↓
YOLO-Pose (17 keypoints)          app/services/yolo_pose_adapter.py
        ↓
PersonTracker (identity)          app/core/tracking/
        ↓
STEP 3.5: TrainMotionDetector     app/core/detectors/train_motion_detector.py
  (vibration + side-window flow)  → self.current_motion_state RUNNING|STOPPED
        ↓
MultiPersonActivityRunner         app/core/multi_person_runner.py
  parallel per-person detectors:
    • SleepDetector               app/core/detectors/sleep_detector.py
    • ActivityDetector            app/core/detectors/activity_detector.py
    • GestureDetector             app/core/detectors/gesture_detector.py
    • MindDiversionDetector       app/core/detectors/mind_diversion_detector.py
        ↓
Temporal gating (consecutive frames + grace)
        ↓
ActivityRepository (atomic write) app/repositories/activity_repository.py
        ↓
activities.json   ─────────────▶  Pipeline-2 (VLM verify) ─▶ CVVR API
```

### Pipeline-2 motion override

Pipeline-1's vibration-based motion detector is fooled by diesel-idle (engine
on, train stopped). Pipeline-2 re-checks motion using a window-region
frame-diff classifier (`app/services/vlm/motion_classifier.py`) and can flip
`motionState` from RUNNING → STOPPED in the aggregated verdict
(`app/services/vlm/service.py`, see `motion_overrides` counter).

---

## YOLO models

Stock Ultralytics weights, configured via env (overridable):

```
YOLO_WEIGHTS_PRELOAD = yolo11l.pt        # object detection
YOLO_POSE_WEIGHTS    = yolo11l-pose.pt   # multi-person pose
```

Defaults are set in `app/utils/config.py:100-101` and propagated through:
- `locopilot_monitor.py:395`
- `app/core/models/model_loader.py:82, 112`
- `app/core/models/yolo_handler.py:69-70`
- `gunicorn_config.py:97`
- `test_train_motion.py:53, 640`
- `.env.example`, `.env.production`

### Class-name compatibility (IMPORTANT)

The codebase matches detections by **COCO class names** (string match against
`model.names[cls]`). The current model code expects:

| Used in code | Source | In stock COCO? |
|---|---|---|
| `person` | YOLO | ✓ |
| `cell phone` | YOLO | ✓ |
| `book` | YOLO | ✓ |
| `backpack`, `handbag`, `suitcase` | YOLO (bag aliases) | ✓ |
| `cup`, `bottle` | YOLO | ✓ |
| `pen`, `pencil`, `paper` | YOLO | ✗ **NOT in COCO** |

`pen`/`pencil`/`paper` appear in target-class lists at:
- `app/core/models/yolo_handler.py:385, 438, 679`
- `app/core/detectors/object_detector.py:402`

These were detected by the **prior custom-trained model**
(`yolo26s_locopilot_v8.pt`, 9 domain classes, retained at
`/opt/poc2/yolo26m_locopilot_v9.pt` for reference). With stock YOLO11l
(COCO-pretrained, 80 classes) these queries return nothing, so the **writing**
activity (which relies on hand-near-pen/paper proximity) will degrade.

If full activity coverage is required, either:
1. Retrain a custom YOLO that includes `pen`/`pencil`/`paper`, or
2. Strip those class names from `target_classes` lists and accept reduced
   writing-detection sensitivity, or
3. Roll back `YOLO_WEIGHTS_PRELOAD` to `yolo26s_locopilot_v8.pt`.

---

## Entrypoints

- **API:** `POST /v1/video/process` → `app/controllers/video_controller.py`
- **Job runner:** `process_video_job()` in `app/main.py:95`
- **Orchestrator:** `VideoProcessingService.process_video()` in
  `app/services/video_processing_service.py:144`
- **Core detector:** `LocopilotActivityMonitor` in `locopilot_monitor.py`
- **CLI (motion tuning):** `python test_train_motion.py --video <path>`

## Outputs

1. `run_dir/activities.json` — primary, atomic-written
2. `run_dir/clips/*.mp4`, `run_dir/frames/*.jpg` — evidence media
   (`EvidenceManager`)
3. CVVR API via `ExternalAPIService.post_cvvr_results()` (skipped when
   `motionState=STOPPED`)
4. S3/MinIO upload from `video_controller.py`

## Deployment

- `start_server.sh` — local launch (gunicorn + Uvicorn workers)
- `deploy-gpu.sh` — GPU server deploy (read script before running; touches
  the production GPU host noted in `server details.txt`)
- `gunicorn_config.py` — worker count, preloaded models, env passthrough

## Specs and docs

- `docs/specs/architecture-cleanup/` — recent refactor notes
- `docs/specs/locopilot-refactor/` — high-level refactor plan
- `docs/specs/post-verifier-merge/` — VLM verification merge
- `docs/code-review-2026-05-08.md` — recent review findings

---

## Conventions for this repo

- Frame sampling default: **0.5 FPS** (one frame every 2 s).
- Custom YOLO weights formerly lived at `/opt/poc2/` on the GPU host —
  current default is stock COCO-pretrained YOLO11l.
- Atomic JSON writes everywhere — use `ActivityRepository`, not raw
  `json.dump`.
- Temporal gating uses `consecutive_detections` + `grace` counters per
  activity per person; finalised activities have `start_time`, `end_time`,
  `evidence`, `confidence`, `crew role` (LP/ALP).
- Multiprocessing mode chunks video with overlap (ARCH-03) — be careful
  with detector state (e.g. `train_motion_detector.prev_gray`) that must not
  leak across chunks; see `locopilot_monitor.py:2862, 2900, 3006`.
