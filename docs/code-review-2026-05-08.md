# Locopilot Monitoring System — Consolidated Code Review

**Date:** 2026-05-08
**Method:** Six parallel `code-reviewer` agents, each scoped to a non-overlapping module group.
**Scope:** Every Python file under `app/`, the legacy monolith `locopilot_monitor.py` (~5200 lines), config, deploy scripts, and `requirements.txt`.
**Total findings:** ~276 (~50 Critical, ~76 High, ~88 Medium, ~62 Low).

---

## Cross-cutting themes

### 1. The "single source of truth" claim is false in three places

- **`activities.json` has THREE independent writers**, each with a different `NumpyEncoder`, none atomic, none locked: `ActivityRepository.save_activities`, `video_multiprocessing.process_video_parallel:1077`, `locopilot_monitor.py:4314`. The repository's encoder is missing `np.bool_` → guaranteed `TypeError` on real payloads.
- **`activity_registry.py` is supposed to be canonical**, but `activity_tracker.ActivityConfig` is a parallel duck-typed dataclass; `cell_phone` `min_duration_seconds` is in seconds while `required_consecutive` is in frames — they disagree if `SAMPLE_FPS` ever changes; and the `train_motion_suppress_stage` has a TODO that bypasses `gates.apply_train_stopped_suppression`.
- **`detect_objects_batch` is missing `cup_bottle` entirely** and uses a hard-coded `200`-px `book_near_person` margin where the single-frame path uses the configurable `book_person_margin`. Multiprocess (production) and single-frame paths produce **different activities for the same video** — breaking the documented "two-pass deterministic" guarantee at the detection layer, not the temporal layer.

### 2. GPU concurrency model is broken on the production endpoint

The 1-worker Gunicorn + `MAX_CONCURRENT_VIDEOS` semaphore is the entire safety story for OOM avoidance, and **`/api/v1/video/process-and-upload` (the production endpoint) bypasses it**. So does the queue's `process_video_job`. `gpu_resource_manager._semaphore` is also constructed in sync `initialize()` outside any event loop — fragile under test runners. Net: 3 queue workers + concurrent direct POSTs can run N+3 videos with no admission control on a 20 GB RTX 4000 Ada.

### 3. Detector state leaks across video boundaries

None of the extracted detectors (`sleep`, `gesture`, `mind_diversion`, `activity`, `train_motion`) have a `reset()` contract enforced by `video_processing_service`. `train_motion_detector` has **no reset method at all** — its `prev_gray` is diffed against the last frame of the previous video, producing a phantom RUNNING signal at the start of every video after the first. The CLAUDE.md-documented `atan2` head-tilt wrap-around fix `(delta + 180) % 360 - 180` was **lost in the extraction** of `sleep_detector.py:344-350`.

### 4. The train-stopped gate is an honor system; detectors don't respect it internally

`gates.apply_train_stopped_suppression` is the only enforcement point, but every detector keeps its own state machines advancing while suppressed. When the train resumes from a stop, `pose_sleep_duration`, `gesture_sessions[*].last_raise_time`, `packing_motion_history[idx]`, and writing-grace counters are already mature → instant FP violations on resume.

### 5. External-system integrations are not production-ready

- **No DLQ on external API failure**: `external_api_service` retries 3× then drops the entire trip's violations. The log even says "Consider dead-letter queue recovery" — but nothing persists the payload.
- **No idempotency**: `addUpdateBulk` retries on accepted-but-timed-out responses → duplicate violations.
- **All HTTP is sync** (`requests.post`/`urllib.request.urlopen`) inside async handlers → event-loop pinning per video.
- **VLM HTTP**: per-activity blocking call, no connection pooling, no retry/backoff, no circuit breaker; cold-start tax is `N × timeout_seconds` per video.
- **VLM fail-open is actually fail-destructive**: a single unhandled exception in `_verify_one` (`json.JSONDecodeError`, OpenCV failure, etc.) silently truncates the kept-violations list mid-loop.
- **OCR has no date awareness** — only `HH:MM:SS` regex; every overnight loco trip (most of them) wraps incorrectly.
- **`videoUrl` is an open SSRF**: no host allowlist, no IP-range filter (RFC1918, `169.254.169.254`).

### 6. Secrets in source control

- **`MINIO_ACCESS_KEY=admin`, `MINIO_SECRET_KEY=login123`** as defaults in `app/utils/config.py:166-167` AND in committed `.env.example`.
- **Hardcoded server password** in `deploy-gpu.sh:12` (used with `sshpass`/`sudo -S`).
- **Bearer tokens captured into request context** by `logging_middleware.py:47, 58` — every log line interpolating the context could write the token.

### 7. Drift between extracted modules and legacy monolith

- `gesture_detector.py` exists but `detect_hand_gesture` is **still 538 lines on the monolith** at `locopilot_monitor.py:1178-1716`.
- `mind_diversion_detector.should_suppress_mind_diversion` and `mind_diversion_suppression.should_suppress_mind_diversion` have **drifted out of byte-identity** (the latter raises `AttributeError` on `None` landmarks; caught silently).
- `__init__` copies ~150 settings into `self.*` ALL_CAPS attributes with **divergent fallback defaults** — `HEAD_DOWN_THRESHOLD` is `0.01` here, `0.05` in Settings.
- `ffmpeg` path is `/usr/bin/ffmpeg` hard-coded in `evidence_manager.reencode_to_h264:163` and `clip_writer.reencode_to_h264:26`, but `extract_video_segment` honors `FFMPEG_PATH` — fails silently on macOS dev boxes.

### 8. `requirements.txt` is unsafe for a safety-critical system

Unpinned `ultralytics>=8.0.0`, `opencv-contrib-python>=4.11.0.86`, `fastapi>=0.104.0`, `pydantic>=2.5.0`. **No torch pin at all** despite needing CUDA 12.x wheels. `pillow>=8.0.0` allows known-vulnerable Pillow ≤9.x.

---

## Top fixes — prioritized for impact-per-day

1. **Restore the determinism contract.** Mirror the missing `cup_bottle` handler and replace the hard-coded `200` margin with `self.book_person_margin` in `detect_objects_batch`. Re-apply the `atan2` wrap-around fix in `sleep_detector.calculate_head_tilt_angle` and in delta computation. Add a regression test that hashes `activities.json` after serial vs parallel runs of the same video. **High customer-facing impact.**

2. **Atomic, locked, single-writer `activities.json`.** Lift one canonical `NumpyEncoder` (with `np.bool_`) into `app/utils/json_utils.py`. Make `ActivityRepository` the only writer; have `video_multiprocessing` and `locopilot_monitor` delegate. `tmp + os.replace` + `portalocker.LOCK_EX`.

3. **Wrap the production endpoint in `acquire_gpu_slot`.** Add the same admission gate that the analyze/queue endpoints use to `/api/v1/video/process-and-upload`. Wire `process_video_job` through the same path. One change closes the OOM door.

4. **Add a DLQ + idempotency key on the external API call.** Persist failed payloads to `<run_dir>/_failed_external_api/` with a startup re-drain task. Add `Idempotency-Key: sha256(payload)+trip_id` header reused across retries. Removes silent data loss + duplicate violations on retry.

5. **Rotate MinIO creds + scrub `.env.example` + scrub `deploy-gpu.sh` password.** Default secrets to empty in `config.py`; raise on use in production. Confirm `.env.production` was never committed (`git log --all --full-history -- .env.production`); if it was, rotate everything in it.

6. **Add detector reset contract + restore train-stopped invariant.** Add `reset()` to `train_motion_detector` (currently absent). Have `video_processing_service` call `reset()`/`reset_tracking()` on every detector at video start. Wire `train_state` into the existing `gates.apply_train_stopped_suppression` so it also resets per-detector state machines (sleep `pose_sleep_*`, gesture `last_raise_time`, packing motion history, writing grace) — otherwise counters silently mature during stops and produce instant violations on resume.

7. **VLM verifier: per-activity try/except + async client.** Wrap each `_verify_one` body in `try/except Exception` recording `SKIPPED_VLM_UNAVAILABLE` and **still appending to `kept`** — Pipeline-2 must never lose Pipeline-1 results. Replace `urllib.urlopen` with a singleton `httpx.AsyncClient`; add a 1-retry + 30s circuit breaker.

8. **Lock the SSRF surface.** Allowlist of MinIO hosts, IP-range filter (RFC1918 + 169.254/16 + loopback), max download size on `/api/v1/video/process-and-upload`'s `videoUrl`.

9. **Pin `requirements.txt`.** Generate `requirements.lock` with hashes via `pip-compile`. Pin `torch==2.x.y+cu121` matching prod CUDA. Bump `pillow>=10.3.0`. Add `pip-audit` to CI.

10. **Standardize logging + redact `Authorization`.** Drop the in-monolith `_setup_module_logger` and the per-module roll-your-own variants; use `app.utils.logger.get_logger`. Strip `Authorization` and similar fields in `logging_middleware`. Replace `HTTPException(500, str(e))` with `internal_error` everywhere.

---

# Detailed module reports

Each section below is the verbatim output of one specialized review agent.

---

## 1. API / HTTP layer

**Files:** `app/main.py`, `app/controllers/video_controller.py`, `app/middleware/logging_middleware.py`, `app/models/*.py`, `gunicorn_config.py`, `start_server.sh`, `deploy-gpu.sh`.
**Findings:** 8 Critical / 12 High / 17 Medium / 15 Low.

### CRITICAL

**C1. `process-and-upload` does NOT acquire a GPU slot — breaks the 1-worker concurrency model**
`app/controllers/video_controller.py:635-746`. The async-job endpoint and `/video/analyze` both go through `gpu_resource_manager.try_enqueue()` + `acquire_gpu_slot()`, but `/api/v1/video/process-and-upload` (the production endpoint per CLAUDE.md) skips the admission gate entirely. It calls `video_processing_service.process_video` directly inside a thread executor. Two simultaneous POSTs to this endpoint will both grab the GPU at the same time, ignoring `MAX_CONCURRENT_VIDEOS`, and will OOM the 20 GB RTX 4000 (Pipeline-1 ~8 GB peak + vLLM ~10 GB resident leaves ~3 GB margin).
**Fix:** wrap the `run_in_executor` call in `async with gpu_resource_manager.acquire_gpu_slot():` and front it with the `try_enqueue/release_enqueue_on_error` finally pattern that `process_video` already uses (lines 259-281, 372-374, 451-460).

**C2. `video_file.file.read()` is a synchronous blocking call inside an async handler**
`app/controllers/video_controller.py:710`. `process-and-upload` calls `video_file.file.read()` (sync) on the SpooledTemporaryFile while in an async handler, blocking the entire event loop for the duration of the upload (5 GB max per `settings.max_upload_size`). Other concurrent requests (health checks, status polls, etc.) hang. The same handler also reads the entire file into RAM before writing.
**Fix:** stream to disk in chunks: `while chunk := await video_file.read(8 * 1024 * 1024): out.write(chunk)`. Or use `aiofiles` + iter chunks. Also enforce `Content-Length` ceiling before reading any bytes.

**C3. `process-and-upload` performs zero file-size / extension/mime validation before reading**
`app/controllers/video_controller.py:692-713`. Only the filename extension is checked at line 702. There is no size check (the analyze endpoint calls `validate_video_file(filename, file_size)` at line 339 before saving — this endpoint does not). DoS by upload.
**Fix:** call `video_processing_service.validate_video_file(filename, file_size)` after streaming or short-circuit on `Content-Length`. Reject `Content-Length > settings.max_upload_size` with 413 before reading.

**C4. `videoUrl` parameter is an open SSRF / arbitrary-URL fetch**
`app/controllers/video_controller.py:239-329`. The `videoUrl` Form field is validated only as "starts with http:// or https://" then passed straight into `minio_service.parse_minio_url` + `download_video`. No allowlist of MinIO hosts, no IP-range filter (RFC1918, link-local, metadata `169.254.169.254`), no path/bucket allowlist.
**Fix:** add `settings.minio_allowed_hosts`; reject any URL whose `urlparse(...).netloc` isn't in the set. Resolve the host and reject if the IP is private/loopback/link-local. Cap the downloaded byte size.

**C5. Hardcoded server password in committed `deploy-gpu.sh`**
`deploy-gpu.sh:12`. CLAUDE.md (`server details.txt`, never commit) explicitly says this. Whoever has this file has root on the GPU box (line 104 / 149 use `sudo -S`).
**Fix:** read from env (`SERVER_PASS="${LOCOPILOT_DEPLOY_PASS:?set this env var}"`) or move to SSH key auth and delete the password from disk + git history.

**C6. `MEDIA_API_KEY` "rollout mode" leaves PII routes wide open in production**
`app/controllers/video_controller.py:117-128`. The dependency intentionally allows requests through when `settings.media_api_key` is unset, and only logs once. `/api/jobs/{run_id}/media/{filename:path}` serves video clips of LP/ALP faces — direct PII. `run_id`s are sequentially time-shaped, low entropy.
**Fix:** flip the default — fail closed. If `MEDIA_API_KEY` is unset, refuse to start (pydantic validator) or at minimum return 401 from these routes.

**C7. `Authorization` header is captured into request context and may be persisted in logs**
`app/middleware/logging_middleware.py:47, 58`. The full `Authorization` header value is stored under `"authorization"` in the request context dict that the logger formatter pulls fields from. Any `logger.info`/`logger.error` later in the request's lifecycle that interpolates context fields will write the bearer to disk.
**Fix:** strip or redact: `auth_header = "***" if auth_header != "N/A" else "None"`.

**C8. CORS allows any origin with `*` methods/headers — combined with no auth on most endpoints**
`app/main.py:263-269`. `allow_methods=["*"]`, `allow_headers=["*"]`. With C6 unfixed, a malicious page can drive uploads/downloads cross-origin.
**Fix:** explicit allowlist of trusted UI origins; `allow_methods=["GET","POST"]`; explicit `allow_headers`. Reject `*`.

### HIGH

- **H1.** `finally` rollback condition uses `locals()` — fragile and may double-decrement (`video_controller.py:451-460`).
- **H2.** `process-and-upload` swallows S3 / activities-update errors without surfacing them (`video_controller.py:865-925`).
- **H3.** No idempotency on `process-and-upload`; retries duplicate violations.
- **H4.** Generic 500s leak `tripId`, filename, and `str(e)` in messages (`video_controller.py:1021-1032`).
- **H5.** `tripId` validation regex too restrictive AND inconsistent with model (`JobSubmitRequest.video_path` accepts any string).
- **H6.** `submit_video_job` accepts arbitrary `video_path` from the client → file-existence oracle / partial path traversal (`video_controller.py:1070-1099`).
- **H7.** `get_run_media` reads entire file into RAM, even for non-Range responses (`video_controller.py:611-617`).
- **H8.** `videoUrl` JSON body bypasses Form validators silently (`video_controller.py:201-223`).
- **H9.** `_RUN_ID_RE` enforced but `output_dir` join is still raw — symlink-escape gap (`video_controller.py:484, 564`).
- **H10.** Job worker count can exceed admission cap → races against gunicorn 1-worker contract; `process_video_job` does NOT call `acquire_gpu_slot` (`main.py:85-135, 180-184`).
- **H11.** Pydantic v2 migration debt: `@validator`, `class Config` everywhere across `video_models.py`, `activity_models.py`, `trip_models.py`.
- **H12.** `process_video_job` discards exception details from worker results; service may be instantiated per-job rather than as singleton (`main.py:115-128`).

### MEDIUM

- **M1.** `process_video` reuses `lpCrewName` for `crew_name` even when ALP-only crew was provided.
- **M2.** `VideoProcessingResponse` schema does not describe what `process_video` actually returns — required fields likely missing.
- **M3.** `VideoProcessingResponse.violations` example still shows singular `type` (legacy).
- **M4.** `process-and-upload` missing `response_model=...` declaration.
- **M5.** `process_video` calls `external_api_service._transform_events_to_violations` (private).
- **M6.** Hardcoded `crew_role=1` (LP) for both endpoints regardless of `cameraAngle`.
- **M7.** `OCRTimestampResult.roi_coords: Optional[tuple]` — pydantic v1 idiom.
- **M8.** `LoggingMiddleware` uses `datetime.now()` not `datetime.now(UTC)`; mix of naive/aware times.
- **M9.** Middleware uses emojis (`📥`/`📤`/`💥`) — violates CLAUDE.md "no emojis" convention.
- **M10.** `request_path = request.url.path.rstrip("/")` collapses `/health/` and `/`.
- **M11.** Module-level `VideoProcessingService()` instantiation at import; not a documented singleton.
- **M12.** `external_api_service` fetched twice per request.
- **M13.** Two health endpoints with different shapes (`/health` vs `/api/health`).
- **M14.** `start_server.sh` runs `pip install` on every start; uses emojis.
- **M15.** `gunicorn_config.py` emoji `✅` in comments.
- **M16.** `JobStatusResponse.config: Dict[str, Any]` is opaque and echoed back — token oracle risk.
- **M17.** `process-and-upload` doesn't clean up uploaded video on success (cleanup commented out).

### LOW

- **L1–L15.** Various: `import os` shadowed locally, `import json as _json` aliasing, dead `VideoUploadRequest` model, silent `cameraAngle` fallback, mixed log prefix vocabulary, `0.0.0.0` binding in dev, `max_requests=100` worker recycle drops in-flight jobs, magic-number `max_length` constants, raw `exc.errors()` logged with input values, unvalidated `subFolderName`, per-process `_media_api_key_missing_warned` flag, hardcoded `gunicorn_config` values.

### Top 5 fixes

1. Wrap `/api/v1/video/process-and-upload` and `process_video_job` in `acquire_gpu_slot()`.
2. Fail-closed `MEDIA_API_KEY` + remove hardcoded password from `deploy-gpu.sh`.
3. Stream uploads to disk with size cap, drop sync `file.read()` in async handler.
4. Lock down `videoUrl` SSRF surface (host allowlist, IP-range filter, max bytes).
5. Redact `Authorization` from request context, swap `HTTPException(500, str(e))` for opaque `internal_error`.

---

## 2. Core orchestration services

**Files:** `app/services/video_processing_service.py`, `activity_detection_service.py`, `job_manager.py`, `gpu_resource_manager.py`, `vlm_verification_service.py`.
**Findings:** 9 Critical / 13 High / 13 Medium / 10 Low.

### CRITICAL

**C1. `job_manager.start_workers` does not configure GPU semaphore and triple-counts active slots**
`gpu_resource_manager.py:209` defines `asyncio.Semaphore(max_concurrent_videos)` but `job_manager.py` independently uses `num_workers=settings.job_queue_num_workers` (3) without ever calling `acquire_gpu_slot()`. If `MAX_CONCURRENT_VIDEOS < num_workers`, three jobs run concurrently with no GPU slot enforcement, defeating the per-process gate. Also `acquire_gpu_slot()` itself increments `_active_count`, but callers using `try_enqueue/mark_enqueued_started` paths can also call `increment_active()` — `try_enqueue` (line 537) reads `_active_count + _pending_count` while pending is incremented and active will *also* be incremented later, so position calculation and `try_enqueue` cap double-count one slot.
**Fix:** in `_process_job` wrap `await self._process_func(job)` inside `async with gpu_resource_manager.acquire_gpu_slot():`. Decide ONE counter authority — semaphore or active/pending pair, never both.

**C2. `gpu_resource_manager._semaphore = asyncio.Semaphore(...)` constructed during sync `initialize()` outside any event loop**
`gpu_resource_manager.py:209, 228`. `asyncio.Semaphore()` ties to the current running loop on first `await`. If `initialize()` is called from sync code, the semaphore is created with no loop bound. If `acquire_gpu_slot()` is later awaited from a *different* loop, it raises `RuntimeError: <Semaphore [...]> is bound to a different event loop`.
**Fix:** lazy-construct the semaphore inside `acquire_gpu_slot` on first `await` instead of in `initialize()`.

**C3. `VideoProcessingService.process_video` is sync but called from async worker — blocks the event loop for the entire video**
`video_processing_service.py:150-477` is a fully synchronous method that can run for tens of seconds to minutes. `job_manager._process_job` calls `await self._process_func(job)` (line 452). If `process_func` is wrapped naively (not visible in scope but the call site is `await`), the entire event loop stalls.
**Fix:** in `job_manager._process_job` invoke via `await asyncio.to_thread(self._process_func, job)` if `_process_func` is sync.

**C4. VLM HTTP call uses blocking `urllib.request.urlopen` from inside async-served pipeline**
`vlm_verification_service.py:1042`. Each VLM verification blocks for up to `vlm_timeout_seconds` (15s prod). Every activity in a video serializes 15s of head-of-line blocking. No connection pooling — fresh TCP+TLS per activity.
**Fix:** switch to `httpx.AsyncClient` with a singleton client and per-call `timeout`.

**C5. Fail-open for VLM is broken when an individual `_verify_one` raises an unhandled exception**
`vlm_verification_service.py:991-1085` catches `URLError/HTTPError/TimeoutError/OSError` (line 1046) but not `json.JSONDecodeError` from `json.loads(resp.read())` at line 1045, nor `KeyError`/`AttributeError` from `act.get(...)` patterns elsewhere, nor `cv2`/`numpy` failures inside `_stitch_keyframes`. **This silently truncates the violation list** mid-loop.
**Fix:** in `verify_activities`, wrap the per-activity body in `try/except Exception` that records `SKIPPED_VLM_UNAVAILABLE` and `kept.append(act)`. Pipeline-2 must NEVER lose Pipeline-1 results.

**C6. Singleton `__new__` pattern races with `__init__` reset; `_initialized` flag is per-instance not class-level for some, class-level for others**
`gpu_resource_manager.py:68` uses *instance-level* `_initialized` set inside `__new__`. `job_manager.py:62` sets `cls._instance._initialized = False` inside the lock — brief window where another thread is mid-`__init__`.
**Fix:** standardize: `_initialized` should be class-level, set at the *end* of `__init__` after the lock is held.

**C7. `_evict_oldest_completed_jobs` lock invariant enforced by comment only**
`job_manager.py:376-419`. Any future contributor adding `await` inside this method silently breaks the lock invariant.
**Fix:** rename to `_evict_oldest_completed_jobs_locked` and add `assert self._jobs_lock.locked()`.

**C8. Job state mutation outside of `_jobs_lock` — TOCTOU race in `_worker`**
`job_manager.py:356`: `if job.status == JobStatus.CANCELLED:` is read after the lock was released. Cancellation can be silently ignored until the next async checkpoint.
**Fix:** re-check `job.status` inside `_process_job` under the lock at line 444.

**C9. `cv2.imread` failure in `_stitch_keyframes` silently drops frames; no logging**
`vlm_verification_service.py:671-672`. Diagnosing "VLM never verified this video" requires you to know to look at OS-level perms or disk corruption.
**Fix:** `if img is None: logger.warning("[vlm] cv2.imread failed for %s", p); continue`.

### HIGH

- **H1.** Lost-job on shutdown: `stop_workers` cancels mid-flight tasks, leaving the queue with un-popped `job_id`s. SIGTERM mid-batch loses up to `max_queue_size=10` jobs.
- **H2.** `_process_job` exception handler may overwrite `CANCELLED` with `FAILED` (line 474-485).
- **H3.** Multiprocess pipeline shuts down the shared pool after EVERY job (`shutdown_shared_pool(wait=True)` at `activity_detection_service.py:415-416`), defeating "shared global pool". Move shutdown to FastAPI lifespan.
- **H4.** `_stitch_keyframes` holds ~60MB of decoded frames per call until function return.
- **H5.** `process_video` writes activities.json without `tmp+os.replace` — crash mid-write corrupts source-of-truth artifact.
- **H6.** TOCTOU in `cleanup_uploaded_video` (`os.path.exists` + `os.remove`).
- **H7.** VLM motion override regex permits prompt-injection by VLM output — a maliciously crafted `motion_evidence: "platform"` flips `motionState` to STOPPED and silently drops a real violation.
- **H8.** `_parse_verdict` strips backticks naively; brittle on nested `{`.
- **H9.** `validate_video_file` checks `file_size` from caller-supplied int, not from disk.
- **H10.** Singleton `VlmVerificationService` freezes `_verify_set` at `__init__`; inconsistent live-reread across settings.
- **H11.** `gpu_resource_manager` `_streams` discarded without `synchronize()` on unload — pending kernels race with `empty_cache`.
- **H12.** `handle_oom_error` mutates `_current_batch_size` without lock.
- **H13.** No VLM retry with backoff on transient failures; cold-start tax `N × timeout_seconds` per video.

### MEDIUM

- **M1.** `process_video` violates SRP — 327 lines mixing schedule fetch, run-dir creation, dispatch, VLM hook, motion-filter, external-API post.
- **M2.** `vlm_verification_service.py` is 1100 LOC of mixed responsibility — split into prompts, image utils, http client, orchestrator.
- **M3.** `JobStatus.QUEUED` set AFTER `queue.put_nowait` — invariant violated.
- **M4.** `acquire_gpu_slot` doesn't `torch.cuda.empty_cache()` on slot release.
- **M5.** `mock` activity detection imports `cv2` inside the method.
- **M6.** `os.environ['PYTORCH_CUDA_ALLOC_CONF']` set inside `initialize()` — too late on most code paths.
- **M7.** Logging f-string interpolation in tight loops.
- **M8.** `verify_activities` discards activities silently when cap hit; wrong counter incremented.
- **M9.** `_has_hard_stopped_cue` regex doesn't split on em-dash; inconsistent behavior.
- **M10.** `_parse_verdict` doesn't validate `verdict` field is in expected enum.
- **M11.** `try_enqueue` lock-ordering inconsistent across the file.
- **M12.** `Job` retention purely time-based via `cleanup_completed_jobs`, but no scheduler invokes it.
- **M13.** Missing entry/exit logging in many service methods.

### LOW

- **L1–L10.** Magic strings for `motionState`, literal `f"run"` default, unseeded random, oversized `gpu_resource_manager.py` (1129 LOC), wrong `max_queue_size` log, deprecated `crew_name` legacy params, Unicode `≥`/`≤` in prompts, emojis in `activity_detection_service.py:300` and `video_processing_service.py:244`, unnecessary `import json as _json` alias.

### Top 5 fixes

1. Wrap VLM HTTP in `try/except Exception` per-activity (C5).
2. Migrate VLM HTTP to async with connection pooling (C4).
3. Reconcile job-queue concurrency with GPU semaphore (C1).
4. Atomic write of `activities.json` (H5).
5. Stop tearing down the multiprocess pool every job (H3).

---

## 3. I/O & integration services

**Files:** `s3_upload_service.py`, `minio_service.py`, `external_api_service.py`, `etrain_delay_service.py`, `ocr_timestamp_service.py`, `trip_data_service.py`, `image_preprocessing_service.py`, `concurrent_activity_grouping_service.py`, `yolo_pose_adapter.py`.
**Findings:** 9 Critical / 18 High / 18 Medium / 10 Low.

### CRITICAL

**C1. Secrets logged in plaintext (token length is not "safe")**
`s3_upload_service.py:75` logs `f"S3 upload with Bearer token (length: {len(auth_token)})"` at DEBUG. Length leaks rotation/format info. `external_api_service.py:140`'s catch-all `exc_info=True` on a `requests` exception will dump the full request including `Authorization` header.
**Fix:** never log token length; install a `logging.Filter` that strips `Authorization`/`secret_key` keys; on exception, log only `e.response.status_code` + `type(e).__name__`.

**C2. MinIO secret leaked via `__init__` failures**
`minio_service.py:36-44`. `Minio(...)` raises with the full kwargs in some versions. `cert_check=False` (line 41) silently disabling TLS verification, combined with `secure=settings.minio_secure` defaulting to `True`, gives the worst case: HTTPS that doesn't verify.
**Fix:** wrap in try/except; gate `cert_check=False` behind explicit `MINIO_INSECURE_TLS=1`; default to verification.

**C3. External API silently drops violations on partial server failure**
`external_api_service.py:351-400`. If 5xx persists past 3 retries, the call returns `{"success": False, ...}` and the caller has no DLQ. A whole trip of violations is lost on a 5-minute upstream hiccup.
**Fix:** write the full unposted payload to `<run_dir>/_failed_external_api/<trip_id>_<ts>.json` before returning failure, and add a startup re-drain task. **Single highest-impact data-correctness gap.**

**C4. `addUpdateBulk` POST is not idempotent across retries**
`external_api_service.py:91-103`. Retries on `429/500/502/503/504`. If the server actually accepted+committed but the response was lost (timeout, 502), retry creates duplicates.
**Fix:** generate one `Idempotency-Key: <trip_id>:<sha256(payload)>` header per logical request, reused across retries.

**C5. `requests.post(..., files=..., stream=False)` reads entire upload into memory**
`s3_upload_service.py:91-97`. For 100s-of-MB videos, `requests` materializes the whole file in memory. With Gunicorn `workers=1` this can OOM the GPU box.
**Fix:** use `requests-toolbelt MultipartEncoder` with a streaming generator, or switch to `boto3` direct multipart upload.

**C6. No body size limit, no preflight check on `upload_file`**
`s3_upload_service.py:39-97`. Accepts any local path of any size.
**Fix:** `if file_size > MAX_UPLOAD_BYTES: return False, None, "exceeds max"`.

**C7. OCR timestamp parsing has no date/timezone awareness**
`ocr_timestamp_service.py:56-58, 367-387`. Only parses `HH:MM:SS`, not the date. The `DATE_TIME_PATTERN` (line 57) is defined but never used. A video that crosses midnight (overnight loco trips, which is most of them) produces timestamps that go `23:59:58` → `00:00:01` and any downstream "timestamp_sec" arithmetic breaks.
**Fix:** parse `DATE_TIME_PATTERN` first, fall back to `TIME_PATTERN`; carry a monotonic seconds counter that detects wrap; attach IST `ZoneInfo("Asia/Kolkata")`.

**C8. OCR `_last_timestamp` cache is corrupted under threading**
`ocr_timestamp_service.py:74-80, 172-174`. Singleton shared across worker threads, but `_last_timestamp`/`_last_extraction_time` are unguarded mutable instance state. Two videos racing through OCR can return Video A's timestamp on Video B's frame.
**Fix:** key by job id under a `threading.Lock`, or drop the cache.

**C9. Train number injected unsanitized into URL**
`trip_data_service.py:156-158`. Although line 137 validates `train_num_clean.isdigit()`, the path uses unstripped `train_number` at line 158: `url = f"{self.api_url}/{train_number}"`.
**Fix:** use `train_num_clean` in the URL.

### HIGH

- **H1.** No connection pooling on any `requests` call site — all 4 services use bare `requests.post`/`requests.get`.
- **H2.** Sync I/O blocks the FastAPI event loop — all five HTTP services use blocking `requests.*`.
- **H3.** Retry sleep inside synchronous loop blocks the event loop (`time.sleep(2 ** n)`).
- **H4.** Singleton uses `lru_cache` AND DCL — one is dead code (`minio_service.py:169-184`).
- **H5.** Etrain HTML parsing one-`<td>`-rename away from total breakage; no fallback to BeautifulSoup, no metric on parse failure.
- **H6.** Etrain user-agent unrealistic and brittle — Cloudflare-flagged; risk of IP ban.
- **H7.** `_event_to_violation` swallows every exception per event silently — caller sees "success" with N events lost.
- **H8.** `_calculate_clip_duration` accepts seconds-as-string and HH:MM:SS interchangeably — silent 60× corruption (`"6.00"` vs `"6:00"`).
- **H9.** CLAHE service documented as RGB but `cv2.COLOR_RGB2LAB`/`COLOR_LAB2RGB` constants subtly wrong if any caller passes BGR.
- **H10.** `apply_unsharp_masking` math is incorrect — strength double-counted (`image_preprocessing_service.py:295`).
- **H11.** Adapter-pattern claim unfulfilled — no `rtmpose_adapter.py` or `rtmw_adapter.py`; no shared `Protocol`/`ABC`.
- **H12.** `YoloPoseLandmarks.__init__` does CPU sync per landmark — slow at high FPS (10× the cost it should be).
- **H13.** `PersonKeypoints` slicing keeps GPU tensors alive longer than needed.
- **H14.** `concurrent_activity_grouping_service.py:218-242` — `subprocess.run(ffmpeg)` with `capture_output=True` buffers stderr unboundedly.
- **H15.** `YOLO_POSE_WEIGHTS` read via `os.getenv` directly instead of pydantic-settings.
- **H16.** Etrain delay regex matches "delay" in the wrong column (ads, headers).
- **H17.** `_deduplicate_violations` key uses `startTime` string verbatim — `"6.00"`, `"6.0"`, `"00:00:06"` all dedupe differently.
- **H18.** Logger inconsistency — three different logger acquisition patterns; secrets-stripping won't apply uniformly.

### MEDIUM

- **M1.** Emojis in log messages defeat grep (`external_api_service.py`).
- **M2.** `s3_upload_service.upload_multiple_files` is sequential — long pole of the pipeline.
- **M3.** `requests` lacks explicit `verify=` and AWS region pinning.
- **M4.** Presigned URL TTL invisible — implies long-lived URLs or public-read.
- **M5.** OCR confidence threshold `> 0` for top region but `> 0.3` for bottom — inconsistent.
- **M6.** OCR tesseract preprocessing uses INTER_CUBIC + 3× scaling unconditionally.
- **M7.** `_parse_timestamp` OCR error correction is destructive (global `O→0`, `l→1`, `I→1`, `.→:`).
- **M8.** `image_preprocessing_service.py` has no NaN/inf guard; uniform-frame edge case.
- **M9.** Defensive `processed_frame = frame.copy()` is unnecessary if all sub-ops return new arrays.
- **M10.** `etrain_delay_service.get_adjusted_schedule` mutates input `base_schedule` in place — corrupts cache.
- **M11.** Etrain cache key omits division/route variation.
- **M12.** `_schedule_cache = TTLCache(...)` allocated at import; per-worker cache hit rate.
- **M13.** Service entry/exit logging missing on most public methods.
- **M14.** `concurrent_activity_grouping_service._merge_video_clips` re-encodes with `libx264` even if all input clips are already h.264.
- **M15.** `_request_with_retry` final-attempt-on-retriable-status returns the response without raising.
- **M16.** `OCRTimestampService` engine fallback only at init; no runtime EasyOCR→Tesseract fallback.
- **M17.** `MEDIAPIPE_TO_YOLO_MAP` defined but unused (`yolo_pose_adapter.py:73-101`).
- **M18.** `YoloPoseLandmarks` clamps coordinates to [0, 1]; a wrist held just outside FOV gets fake "wrist at edge".

### LOW

- **L1–L10.** 5-min timeout regardless of file size; `get_settings()` called twice; naive `datetime.now()`; identity-then-equality on dicts; shallow copy of nested evidence; missing `Retry`-friendly `pool_pre_ping`; `exc_info=True` flooding logs on NaN frames; hardcoded log path in adapter; eager `import easyocr`; class-level vs module-level `MAX_RETRIES = 3` constants.

### Top 5 fixes

1. DLQ for `external_api_service` failures (C3).
2. Stop logging tokens & disable `exc_info=True` on `requests` exceptions; scrub `Authorization` via filter (C1, C2).
3. Add `Idempotency-Key` to `external_api_service._request_with_retry` (C4).
4. Replace OCR `HH:MM:SS`-only parser with `DATE_TIME_PATTERN` + midnight-wrap detector + IST tz; key cache by job id (C7, C8).
5. Replace four `requests.post/get` direct callers with `requests.Session` + `await asyncio.to_thread(...)` at controller layer (H1, H2, H3).

---

## 4. CV detectors

**Files:** `app/core/detectors/sleep_detector.py` (~70KB), `gesture_detector.py` (~33KB), `mind_diversion_detector.py` (~29KB), `mind_diversion_suppression.py`, `object_detector.py`, `train_motion_detector.py`, `activity_detector.py`, `writing_fallbacks.py`.
**Findings:** 9 Critical / 12 High / 11 Medium / 9 Low.

### CRITICAL

**C1. Head-tilt angle has the *unfixed* `atan2` wrap-around bug**
`sleep_detector.py:344-350`:
```python
delta_y = nose.y - neck_y
delta_x = nose.x - neck_x
angle = np.arctan2(delta_y, delta_x) * 180 / np.pi - 90
```
CLAUDE.md states the fix for the 300+ degree drop FP was `(delta + 180) % 360 - 180` plus a `nose_y_drop >= 0` guard. **Neither guard is present in this extracted detector.** A wrap from `+170°` to `-170°` produces `delta = -340°` for one frame and `+340°` for the next — easily exceeding the 30° drop threshold. The "consecutive count = 2" doesn't help because two consecutive wraps both produce `>30°` deltas.
**Fix:**
```python
angle = (np.arctan2(delta_y, delta_x) * 180 / np.pi - 90 + 180) % 360 - 180
delta = (tilt_list[-1] - tilt_list[-2] + 180) % 360 - 180
```
Add the `nose_y_drop >= 0` guard at line 982.

**C2. Class-level mutable state — stale per-person tracking persists across video boundaries**
`sleep_detector.py:75-76, 178-225`. `per_person_tracking: Dict[int, Dict[str, Any]] = defaultdict(self._create_tracking_dict)` combined with the singleton lifetime across multiple videos: stale baseline calibration, head-bob counters, state machine all carry over. Same issue in `ir_forward_lean_tracking`, `gesture_sessions`, `recent_person_activities`, `hand_position_history`, `packing_motion_history`, `_recent_person_activities`, plus all train-motion buffers.
**Fix:** Add a documented contract that `video_processing_service` calls `reset()`/`reset_tracking()` between videos.

**C3. `train_motion_detector.py` has no `reset()` — `prev_gray`/state buffers leak across videos**
`train_motion_detector.py:119-129`. The first frame of video B is diffed against the last frame of video A — yielding a massive bogus `vibration_mean` spike classified as `RUNNING` regardless of actual state.
**Fix:** Add a `reset()` method clearing all temporal buffers and call from `video_processing_service` start-of-video.

**C4. Detectors never re-check train-stopped gate internally**
None of `gesture_detector`, `activity_detector`, `mind_diversion_detector`, `sleep_detector`, `writing_fallbacks` accept a `train_state` parameter. The detector's own state machines (gesture sessions, sleep `pose_sleep_start`, packing motion history, mind-diversion grace timer, writing tracking buffers) keep accruing duration/consecutive frames *while the gate is suppressing the activity*. When the train resumes, those internal counters are already mature — producing instant violations with no real "evidence" period.
**Fix:** Either pass `train_state` into each detector and short-circuit-with-reset when STOPPED, or have `gates.py` *also* reset the per-person state in each detector.

**C5. `mind_diversion_suppression.py` and `mind_diversion_detector.py` are duplicate (and divergent) implementations**
- `mind_diversion_detector.py:591-592` adds null-checks: `if (left_wrist and right_wrist and nose ...)`.
- `mind_diversion_suppression.py:81` does NOT null-check — raises `AttributeError` on `None`, caught by silent `except (AttributeError, IndexError): pass`.
- Detector method *mutates* `self._recent_person_activities`; suppression module is read-only.
**Fix:** Pick one. The module-level pure form is preferable.

**C6. Hand-velocity bug: time-window mismatched with position-window**
`gesture_detector.py:679-700`. `recent_positions = list(position_history)[-3:]` and `recent_times = list(timestamps)[-3:]` happen to align with `maxlen=10`, but no assertion enforces it. When `dt` is very small (sub-sample-period FPS jitter) you get explosive false `rapid_raise_detected`.
**Fix:** Add a `MIN_DT = 1.0 / max_fps` guard, and compute velocity as `displacement * sample_fps`.

**C7. `validate_pose_landmarks` allows NaN visibility through**
`sleep_detector.py:308`. If visibility is `np.float32(NaN)`, `total_visibility += NaN` → validation passes, NaN propagates through every downstream calculation. NaN comparisons quietly evaluate False — `is_head_down` and `is_nose_drooping` go False, but the state machine relies on these and silently never advances.
**Fix:** Add `if not np.isfinite(visibility): return False`.

**C8. Zero-division & near-zero division in head pose math**
`mind_diversion_detector.py:209, 220, 387, 397`. `face_width > 0` at line 387 — when face mesh is slightly degenerate (profile view), `face_width` is small but nonzero, and the result OVERWRITES the pose-based yaw with garbage.
**Fix:** Use `if shoulder_width > MIN_SHOULDER_PX (e.g., 20px):` and `face_width > MIN_FACE_PX`.

**C9. Object detector creates `YOLOHandler` via `object.__new__` bypassing `__init__`**
`object_detector.py:137-182`. Any new attribute added to `YOLOHandler.__init__` is silently absent on this synthetic instance.
**Fix:** Make `YOLOHandler.__init__` accept `pose_model=None` officially, or factor out object-detection-only attributes into a base class.

### HIGH

- **H1.** Sleep state machine: `LOOKING_DOWN_WORKING` blocks all sleep transitions; pilot dozing off with hands on throttle is invisible (`sleep_detector.py:524-526, 958-966`).
- **H2.** Hard-coded magic numbers buried in code that should live in `activity_registry.py` / `Settings`: `* 720` fallback frame_height, `0.2`/`0.3`/`0.08` visibility thresholds, `person_book_margin = 250`, `proximity_threshold = 100`, `4.0` second sustained_proximity, `15 <= avg_velocity <= 200`, `deque(maxlen=6)`, `check_margin = min(margin, 100)`, ear-asymmetry `60`/`-60`, train-motion `interior_count < 100`, block variance thresholds `1200`/`800`/`400`, top 5% / bottom 8% mask, `confidence_threshold = 0.6`.
- **H3.** `cleanup_stale_tracking` defined but never invoked externally — dicts grow without bound in 24/7 service.
- **H4.** `_get_keypoint` failure modes inconsistent: some return `None`, some raise. Brittle.
- **H5.** Sleep state machine bypasses score gate inside `LOOKING_DOWN_WORKING` (early return at lines 958-966).
- **H6.** `is_wrist_inside_backpack` short-circuits without temporal guard — pilot resting one hand on bag fires single-frame `packing_bags`.
- **H7.** `_check_writing_suppression` records writing timestamp only when mind-diversion-detection is also triggered — grace period silently never engages otherwise.
- **H8.** Performance: per-frame `np.array(list(...))` allocations in tight per-person path.
- **H9.** `compute_stability` overwrites `_prev_block_vars` even when `interior_block_mask` shrinks/grows — phantom variance spikes when persons move.
- **H10.** `train_motion_detector` first-frame logic depends on `compute_vibration` side effect; `vibration_mean == 0` floating-point comparison.
- **H11.** Mind-diversion `face_mesh` refinement re-evaluates without preserving original — pose-positive can be silently flipped to false.
- **H12.** `_estimate_yaw_from_ear_asymmetry` always uses `60`/`-60`, below the `78` threshold — entire fallback path is dead code.

### MEDIUM

- **M1.** `gesture_detector._check_temporal_suppression` only suppresses, never clears.
- **M2.** `update_session` cleanup uses `last_update` only for the role being updated; not called from `check_gesture_coordination`.
- **M3.** Backpack proximity threshold scaled in gesture suppression but absolute pixels in packing detection.
- **M4.** `detect_packing_bags` calls `analyze_packing_hand_motion` even when bbox check fails — `packing_motion_history` polluted by distant frames.
- **M5.** `is_shoulder_slumping` computation done every frame but result never used in score logic — dead computation.
- **M6.** `train_motion_detector.create_interior_mask` allocates 2 MB uint8 mask every frame.
- **M7.** `_apply_detection_logic` and `_refine_with_face_mesh` duplicate the threshold ladder.
- **M8.** Object detector's `_cached_frame_objects` is single-cell, no synchronization.
- **M9.** `analyze_hand_velocity` returns `MIN_PIXEL_THRESHOLD = 20` when `bbox_height == 0` → any velocity >20 px/s triggers `rapid_raise`.
- **M10.** `mind_diversion_detector.py` writes to project-relative `logs/` path; pollutes repo in tests.
- **M11.** `head_tilt_drop` math same wrap-around bug as C1.

### LOW

- **L1–L9.** Sparse `__init__.py`; oversized `sleep_detector.py` (suggested splits: baseline, state machine, head bob, shoulder slump, Haar eye closure, IR forward lean); identical branches in `assign_role_by_camera_angle`; broad `try/except Exception`; magic landmark indices; `vibration_mean == 0` float compare; unused trackers; orphaned `check_forward_looking_exemption`; ROI string parsing in constructor; per-call object alloc.

### Top 5 fixes

1. Re-apply the head-tilt wrap-around fix (C1, M11).
2. Add `reset()` to TrainMotionDetector and call all detector resets between videos (C2, C3).
3. Dedupe `mind_diversion_suppression` and detector method (C5).
4. Add NaN/finite guards to `validate_pose_landmarks` (C7) — defends every downstream calculation.
5. Wire `train_state` into the state-machine resets (C4).

---

## 5. Pipeline / tracking / utils / visualization

**Files:** `app/core/activity_registry.py`, `activity_tracker.py`, `evidence_manager.py`, `frame_pipeline.py`, `gates.py`, `pipeline/{frame_sampling,pose_batch,stages/*}`, `tracking/{coordination,hand_history,per_person_state,person_tracker,static_object_filter}.py`, `utils/{geometry,pose_checks,pose_utils,pose_validators,video_io}.py`, `visualization/*`, `media/clip_writer.py`, `models/{model_loader,yolo_handler}.py`.
**Findings:** 9 Critical / 17 High / 17 Medium / 8 Low.

### CRITICAL

**C1. Frame pipeline stages read mutable monitor state — Pass-2 is NOT deterministic**
`temporal_filter_stage.py:32-100`, `per_person_activities_stage.py`. `TemporalFilterStage` mutates `monitor.consecutive_detections`, `monitor.grace_counters`, `monitor.activities[*]['frames']` — global, monitor-bound mutable state. CLAUDE.md claims "Pass-2 sequential temporal filter (deterministic)", but determinism only holds if `monitor` is constructed clean per run AND stages are called in the documented frame order.
**Fix:** Make Pass-1 output pure dicts of `{frame_idx → {activity → bool}}` only. Pass-2 should construct its own `ActivityTracker` instance.

**C2. `EvidenceManager.reencode_to_h264` hardcodes `/usr/bin/ffmpeg`**
`evidence_manager.py:163` vs `:108`. Two ffmpeg paths in the same class: `extract_video_segment` reads `os.environ.get('FFMPEG_PATH', 'ffmpeg')`, but `reencode_to_h264` is hard-coded. On macOS dev boxes re-encoding silently fails; the resulting clip is in mp4v that the browser cannot play. Same bug at `media/clip_writer.py:26`.
**Fix:** Use `os.environ.get('FFMPEG_PATH', 'ffmpeg')` in both places.

**C3. `EvidenceManager.extract_video_segment` doesn't `check=True` on subprocess**
`evidence_manager.py:122-146`. Missing clip becomes missing S3 upload; external API never gets evidence; no upload-side reconciliation.
**Fix:** Add a clip-vs-activity reconciliation pass before S3 upload that asserts `len(clip_files) == len(activities_to_upload)`.

**C4. `reencode_to_h264` race: temp path replaces input mid-read**
`evidence_manager.py:172`, `clip_writer.py:52`. `os.replace` is atomic, but no `fsync` between writing temp and renaming — power loss can land rename before bytes (0-byte clip).
**Fix:** `out.flush()` + `os.fsync(out.fileno())` before `os.replace`. Check `os.path.getsize > 0`.

**C5. `PersonTracker` greedy IoU matching can assign two current persons to the same prev person**
`person_tracker.py:212-222`. `_apply_temporal_tracking` builds `current_to_prev_match` via per-current-person argmax, with no exclusion of already-claimed prev indices. Bogus prev roles → LP/ALP role flips.
**Fix:** Use Hungarian assignment (`scipy.optimize.linear_sum_assignment`).

**C6. `match_pose_to_roles` discards keypoints once a YOLO idx is consumed — order-dependent**
`person_tracker.py:300-385`. Iterates `person_roles.items()` in insertion order; LP role is processed first and may "steal" a YOLO pose better suited to ALP whenever IoU > 0.2 but is not the actual best match for LP.
**Fix:** Compute the full `roles × yolo` IoU matrix, then global Hungarian.

**C7. `activity_registry.py` is "single source of truth" but `activity_tracker.ActivityConfig` is a parallel dataclass**
Two `ActivityConfig` classes. The registry's superset works only via duck-typing. Drift waiting to happen.
**Fix:** Delete `activity_tracker.ActivityConfig` entirely. Import from `activity_registry`.

**C8. Registry drift: `cell_phone` `min_duration=0.1` vs README claim 6s @ 0.5fps**
`activity_registry.py:127-141`. Registry expresses requirement in frames; the gate expresses in seconds. If `sample_fps` is dropped to 0.25 fps in production, the threshold doubles silently.
**Fix:** Add a `min_duration_seconds` property derived as `max(min_duration, required_consecutive / sample_fps)`. Or fail-fast at startup if mismatch.

**C9. `start_activity` ignores `frame_idx_buffer` parameter at the only known callsite**
`temporal_filter_stage.py:43-50`. `monitor.start_activity(...)` is called without `frame_idx_buffer`. Activity starts with `frames=[]` then gets only the frames AFTER `required_consecutive` is hit — the build-up frames are LOST. `cell_phone` clip starts mid-call.
**Fix:** Pass `frame_idx_buffer=list(monitor.frame_idx_buffer)[-required_consecutive:]`.

### HIGH

- **H1.** `apply_train_stopped_suppression` defined in `gates.py` but `train_motion_suppress_stage.py` does NOT call it; intentional TODO at `:74-76`.
- **H2.** `gates.apply_train_stopped_suppression` rebuilds frozenset on every call.
- **H3.** `frame_sampling.sample_video_frames` step/grab logic is correct but fragile; needs a regression test.
- **H4.** `frame_sampling`'s `step` calculation can produce massive over-stride at low sample_fps + variable bitrate; use `cap.get(CAP_PROP_POS_MSEC)` instead.
- **H5.** `pose_batch.detect_poses_batch` does not preprocess dark frames; object batch path does, pose batch path does not.
- **H6.** `EvidenceManager.save_video_clip` uses `cv2.VideoWriter_fourcc(*'mp4v')` with no `out.isOpened()` check.
- **H7.** `EvidenceManager.save_activities_json` JSON encoder swallows numpy bool.
- **H8.** Hand-gesture coordination uses `recent_person_activities` keys (`lp_hand_raise`/`alp_hand_raise`) but no writer in scope; if writer uses different keys, the entire gate is a no-op.
- **H9.** `static_object_filter.filter` matches each detection against ALL candidates again — O(N²) when O(N) suffices.
- **H10.** `StaticObjectFilter` mutates `self.candidates` to a fresh list; static object briefly occluded loses entire history.
- **H11.** `geometry.calculate_iou` accepts flipped boxes (`x2 < x1`) without rejection.
- **H12.** `deduplicate_person_boxes` lossy on mixed types — both branches identical.
- **H13.** `pose_validators.check_landmark_stability` mutates caller-owned dict but never cleans up.
- **H14.** `pose_utils.get_keypoint` raises `ValueError`; callers don't always catch.
- **H15.** `evidence_frame_annotator` re-runs YOLO on already-saved frame — duplicate inference cost.
- **H16.** `evidence_frame_annotator` swallows ALL exceptions → silent failures hide model corruption.
- **H17.** `mediapipe_overlay` does double `frame.copy()` per call; ~12 MB churn per sampled frame.

### MEDIUM

- **M1.** `frame_pipeline.FramePipeline.run` allows stages to return `None` — silent typo bug.
- **M2.** Two `_setup_module_logger` definitions, identical bodies.
- **M3.** `pose_checks` vs `pose_utils` vs `pose_validators` — overlapping concerns; `YOLO_KEYPOINT_INDICES` defined three times.
- **M4.** `coordination.check_hand_gesture_coordination` doesn't bound how old `recent_person_activities` entries get.
- **M5.** `hand_history.check_wrist_motion_for_packing` normalizes velocity by hard-coded 1280 (resolution-dependent).
- **M6.** `hand_history.analyze_velocity_and_trajectory` velocity in raw px/sec — resolution-dependent.
- **M7.** `pose_validators.validate_anatomical_consistency` rule 1 says "slope > 0.6 ≈ 30°" — actually 31°; anisotropic h≠w scaling means the gate rejects almost no actual leans.
- **M8.** `pose_validators.check_landmark_stability` "shouldn't jump more than 100px between frames" is sample-rate-dependent.
- **M9.** `EvidenceManager.save_activity_image` double-copy on every clip.
- **M10.** Annotator chain stacks 4 `frame.copy()` calls (~24 MB garbage per sampled frame).
- **M11.** `YOLOHandler._configure_thresholds` falls back to `self.object_model` for ROI; `yolo_roi_weights` env var is dead.
- **M12.** `YOLOHandler.detect_objects` `_cached_frame_objects` set but never read — dead state.
- **M13.** `YOLOHandler.detect_objects` uses class-name string comparison — fragile to model retraining.
- **M14.** **`YOLOHandler.detect_objects_batch` does NOT include `cup_bottle` outputs** — `eating_drinking` activity loses its primary signal in the multiprocess path.
- **M15.** **`detect_objects_batch` book-near-person uses hard-coded `margin=200`** — single-frame uses configurable `book_person_margin` (default 150). Multiprocess produces more `writing` activities than serial → "deterministic two-pass" promise broken at the detection layer.
- **M16.** `evidence_manager.save_activity_image_from_video` opens `VideoCapture` without context manager.
- **M17.** `clip_writer.save_video_clip` returns `None`; `EvidenceManager.save_video_clip` returns the path. Drift.

### LOW

- **L1–L8.** `evidence_manager.generate_evidence_filename` doesn't increment counter; two color tables for the same classes; `FrameState` lacks `__slots__`; `activity_registry.py` import-time silent fallback to defaults; `static_object_filter.filter` log strings inconsistent; `coordination.both_within_window` redefined per frame; `train_motion_detect_stage` hard key access; `gates.py` redundant docstring.

### Top 5 fixes

1. Fix the determinism contract (C1, M14, M15).
2. Wire the train-stopped gate through one helper (H1) — replace the TODO.
3. Kill duplicate `ActivityConfig` (C7) + dead `cup_bottle` registry (M14).
4. Fix evidence pipeline ffmpeg path + clip-vs-activity reconciliation (C2, C3, H6).
5. Hungarian person-role + pose-keypoint matching (C5, C6).

---

## 6. Legacy monolith + config + infra

**Files:** `locopilot_monitor.py` (~5200 lines), `app/utils/config.py`, `logger.py`, `multiprocessing_config.py`, `request_context.py`, `video_multiprocessing.py`, `app/repositories/activity_repository.py`, `.env.example`, `.gitignore`, `requirements.txt`.
**Findings:** 6 Critical / 11 High / 12 Medium / 10 Low.

### CRITICAL

**C-1. Real MinIO credentials baked into source defaults and committed in `.env.example`**
`app/utils/config.py:166-167` and `.env.example:169-172`:
```python
minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin")
minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "login123")
```
`.gitignore` whitelists `.env.example` (`!.env.example`), so the secret is in version-control and on every dev machine that clones. The default also means the service will silently come up with these creds if env vars are unset.
**Fix:** Default both to `None`/`""` and raise at startup if missing in production. **Rotate the MinIO password immediately** and scrub git history (`git filter-repo`).

**C-2. `ActivityRepository.save_activities` has zero concurrency or atomicity guarantees**
`activity_repository.py:91-128`. No file lock, no `os.replace` write-then-rename pattern. Three independent writers (`activity_repository.py:117-118`, `video_multiprocessing.py:1080-1084`, `locopilot_monitor.py:4313-4316`) with three different encoders.
**Fix:** Atomic write helper using `tempfile.mkstemp` + `os.fdopen` + `f.flush()` + `os.fsync()` + `os.replace()`. Wrap with `portalocker.Lock`. One canonical writer.

**C-3. Repository `NumpyEncoder` is missing `np.bool_` — will crash on real activity payloads**
`activity_repository.py:20-29` vs `video_multiprocessing.py:35-46`. The MP encoder handles `np.bool_`; the repo's does not. Pipeline-1 routinely produces `np.bool_` flags. Calling `repository.save_activities(...)` on raw activities will raise `TypeError: Object of type bool_ is not JSON serializable`.
**Fix:** Lift `NumpyEncoder` (with `np.bool_` and ideally also `np.datetime64`) into `app/utils/json_utils.py`.

**C-4. Process-pool worker timeouts cannot actually kill stuck workers (GPU lockup)**
`video_multiprocessing.py:984-1024`. `Future.cancel()` on a `ProcessPoolExecutor` future returns `False` and does **nothing** if the task has started. The hung worker keeps the GPU model loaded; the next chunk submitted to that worker may OOM.
**Fix:** Use `pebble.ProcessPool` (supports per-task cancellation by sending SIGTERM/SIGKILL). Or implement watchdog logic in `worker_initializer`. Or rebuild the pool when more than N timeouts in a window.

**C-5. `pose_model` validator is dead code — POSE_MODEL=rtmpose silently no-ops**
`config.py:810-819`. There is **no `pose_model` field** declared on `Settings`. With `extra="ignore"`, `POSE_MODEL=rtmpose` from `.env` is dropped silently and `getattr(self, 'pose_model', None)` always yields `None`.
**Fix:** Add `pose_model: str = os.getenv("POSE_MODEL", "yolo")` with a `field_validator`. Audit all "documented but not declared" flags by diffing `.env.example` keys against `Settings.model_fields`.

**C-6. `train_motion_detection_enabled` coherence-validator branch is misleading**
`config.py:788-807`. Validator re-reads env var manually instead of using `self`; comment "The latter is not a typed Settings field in this branch" is stale.
**Fix:** Drop `getattr(...)`/`os.getenv(...)` fallback; validate directly on `self.train_motion_detection_enabled`.

### HIGH

- **H-1.** No SIGTERM/SIGINT handling — workers leak GPU on shutdown. `atexit` does not run on `SIGTERM`.
- **H-2.** `import torch` at module top of `video_multiprocessing.py` runs in parent process — inflates RSS by ~700MB.
- **H-3.** `.env` and `.env.production` both exist on disk; verify with `git ls-files | grep -E '^\.env'`. If `.env.production` was ever committed, rotate every secret. `deploy-gpu.sh` similarly contains the SSH password.
- **H-4.** `_validate_flag_combinations` model_validator is order-sensitive; `os.path.isabs` check at line 777 silently allows missing weight files for the typical relative-path case.
- **H-5.** `__init__` is a 636-line god method with hardcoded fallback defaults that **diverge** from Settings defaults (`HEAD_DOWN_THRESHOLD` fallback `0.01` vs settings `0.05`; `SLEEP_STRONG_SCORE` fallback `4` vs settings `6`).
- **H-6.** `process_all_persons_activities` is **1187 lines** (1805→2992) — single-responsibility violation; biggest blocker to retiring the monolith.
- **H-7.** JSON write inside `process_video_parallel` bypasses the repository (`video_multiprocessing.py:1077-1084` and `locopilot_monitor.py:4314`).
- **H-8.** `requirements.txt` dangerously unpinned: `opencv-contrib-python>=4.11.0.86`, `ultralytics>=8.0.0`, `fastapi>=0.104.0`, `pydantic>=2.5.0`. **No torch / torchvision pin at all.** `pillow>=8.0.0` allows known-vulnerable Pillow ≤9.x.
- **H-9.** `os.environ['OMP_NUM_THREADS'] = '1'` set in `setup_logging` — order-of-execution determines effective thread count.
- **H-10.** `pytorch_cuda_alloc_conf` is a Settings field but never exported back to env — silent no-op on PyTorch.
- **H-11.** `enable_console_logs` and `LOG_DIR` read directly via `os.getenv` rather than via Settings — inconsistent.

### MEDIUM

- **M-1.** `_setup_module_logger` in `locopilot_monitor.py:160` duplicates `app/utils/logger.py` — two file handlers writing to the same path.
- **M-2.** Per-chunk timeout (`chunk_duration * 10`) misuses the time unit.
- **M-3.** `mp_max_workers_cap` defined twice with different defaults (12 vs 10).
- **M-4.** Repository's `validate_activities` runs Pydantic per-record, no batching.
- **M-5.** `repository.create_run_directory` uses naive `datetime.now()` with 1-second granularity — collisions in multiprocessing.
- **M-6.** `parse_time_to_seconds` returns `0.0` on parse failure — silently merges errors with valid 0.
- **M-7.** `request_context.py` returns `_request_context.get()` without copying — mutated in place.
- **M-8.** `RequestFormatter.format` mutates `record`; worker processes (spawn) inherit no contextvars.
- **M-9.** `consecutive_detections['group_detected']` indexed without registry key — would crash if registry init changes.
- **M-10.** `OPENCV_THREADS` defaulted to 4 in one place, 3 in another.
- **M-11.** `cleanup()` on pre-loaded models nulls shared references in current worker but global still holds them — confirm no follow-up code path re-uses.
- **M-12.** Logger filters `TimedRotatingFileHandler` for dedup; legacy `_setup_module_logger` adds vanilla `FileHandler` and slips through.

### LOW

- **L-1–L-10.** Stale comments; `parse_tile_grid_size` validator inconsistent fallback; `__all__` includes re-exported `video_capture_context`; redundant `os.environ.setdefault('QT_QPA_PLATFORM')`; `pillow>=8.0.0` floor; pin upper bound on `requests`; ultralytics ERROR-level filter suppresses model load diagnostics; `__del__` antipattern; `indent=2` in production (3× file size); `mp_overlap_seconds` default = 12.0 vs window defaults 5.0 (overlap ~80% larger than required).

### Legacy monolith — structural overview

`locopilot_monitor.py` is a 4359-line single class (`LocopilotActivityMonitor`). The refactor visible from imports (lines 22-73) and CLAUDE.md has *extracted* helpers into `app/core/{detectors,tracking,visualization,media,...}` — but the **orchestration** still lives in this class. Methods on the monolith mostly delegate (one-liners), but four methods remain enormous and contain the actual business logic:

- `__init__` — 636 lines (221→857) — model wiring + ~150 self-attribute mirrors of Settings
- `detect_hand_gesture` — 538 lines (1178→1716) — kept on monolith despite sibling `app/core/detectors/gesture_detector.py`
- `process_all_persons_activities` — 1187 lines (1805→2992) — per-frame rule engine; prime extraction candidate
- `_process_frames_core` — 572 lines (3439→4011) — single-frame pipeline

`process_video` and `process_video_range` are clean orchestrators on top of `_process_frames_core`. Public API surface that the new `app/` layer depends on:
- `LocopilotActivityMonitor(...)` constructor with `preloaded_models` kwarg
- `.process_video_range(start_frame, end_frame, save_clips)` returning `List[Dict]`
- `.cleanup()`
- Setters: `.trip_id`, `.crew_name`, `.crew_id`, `.crew_role`, `.crew_members`, `.camera_angle`, `.set_trip_schedule(...)`, `.set_video_start_time(...)`
- `.all_activities` (list, mutated by the run)

### Top 5 fixes

1. Atomic, locked, single-writer `activities.json` (C-2 + C-3 + H-7).
2. Rotate MinIO creds + scrub `.env.example` (C-1, H-3).
3. Worker timeout that actually kills (C-4 + H-1).
4. Settings as source of truth, end-to-end (C-5 + H-9 + H-10 + H-11 + M-3 + M-10).
5. Pin `requirements.txt` (H-8).

---

# Appendix — Files reviewed

**API/HTTP layer:** `app/main.py`, `app/controllers/video_controller.py`, `app/middleware/logging_middleware.py`, `app/models/{video,job,activity,trip}_models.py`, `gunicorn_config.py`, `start_server.sh`, `deploy-gpu.sh`.

**Core orchestration services:** `app/services/{video_processing,activity_detection,job_manager,gpu_resource_manager,vlm_verification}_service.py`.

**I/O & integration services:** `app/services/{s3_upload,minio,external_api,etrain_delay,ocr_timestamp,trip_data,image_preprocessing,concurrent_activity_grouping,yolo_pose_adapter}.py`.

**CV detectors:** `app/core/detectors/{sleep,gesture,mind_diversion,mind_diversion_suppression,object,train_motion,activity,writing_fallbacks}.py`.

**Pipeline / tracking / utils / visualization:** `app/core/{activity_registry,activity_tracker,evidence_manager,frame_pipeline,gates}.py`, `app/core/pipeline/**`, `app/core/tracking/**`, `app/core/utils/**`, `app/core/visualization/**`, `app/core/media/clip_writer.py`, `app/core/models/{model_loader,yolo_handler}.py`.

**Legacy monolith + config + infra:** `locopilot_monitor.py`, `app/utils/{config,logger,multiprocessing_config,request_context,video_multiprocessing}.py`, `app/repositories/activity_repository.py`, `.env.example`, `.gitignore`, `requirements.txt`.
