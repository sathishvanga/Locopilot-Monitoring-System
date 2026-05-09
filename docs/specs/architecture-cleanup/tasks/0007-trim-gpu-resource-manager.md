# Task 0007 — Trim `gpu_resource_manager.py`: remove per-worker CUDA streams

**Severity:** LOW (overengineered abstraction)
**Source:** Architecture review 2026-05-09, finding #4.
**Estimated effort:** 0.5 day.

---

## Problem

`app/services/gpu_resource_manager.py` (1,208 LOC) implements per-worker CUDA streams (`get_stream`, `assign_stream`, `release_stream`, `_initialize_cuda_streams`) for the case where multiple workers run inference on the same GPU concurrently. In default config (`max_concurrent_videos = 1`), this code is dead — there is at most one worker, and the semaphore alone enforces that.

Architecture review estimate: ~150 LOC of pure dead code, plus another ~50 LOC of related stream-bookkeeping inside `initialize()`. The rest of the file (singleton, model loading, OOM handling, batch-size backoff, semaphore, memory stats) is genuinely load-bearing.

**Initial review estimate of "1208 → 400 LOC" was too aggressive — most of this file is needed.** Realistic target: **~150–200 LOC removed** (the streams machinery only).

---

## Files to change

**Modify:**
- `app/services/gpu_resource_manager.py` — delete the methods listed below and any references to them.

**Audit (read-only, expect zero hits):**
- After deletion, `grep -rn "get_stream\|assign_stream\|release_stream\|_initialize_cuda_streams\|cuda_streams\|cuda_stream" --include="*.py" /Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/` must return zero hits.

**DO NOT touch:**
- Anything else in `gpu_resource_manager.py`. The singleton, model loading, OOM backoff, semaphore, memory stats, and properties all stay.
- `app/main.py`, `app/services/video_processing_service.py`, `app/controllers/video_controller.py`, `locopilot_monitor.py` — they call the load-bearing methods only. Verify your deletions don't break their call sites.

---

## What to delete

Inside `app/services/gpu_resource_manager.py`:

1. **Methods (verbatim):**
   - `_initialize_cuda_streams` (around line 239–256)
   - `get_stream` (around line 976–994)
   - `assign_stream` (around line 995–1016)
   - `release_stream` (around line 1017–1033)
2. **Call site:** any call to `_initialize_cuda_streams` from inside `initialize()` (around line 137+). Delete the call.
3. **Instance state:** any `self.cuda_streams = ...`, `self._stream_assignments = ...`, `self._stream_lock = ...` etc. initialized in `__init__` or `initialize`. Delete the attribute declarations and any reset of them in `unload_models` / `shutdown`.
4. **Imports:** `torch.cuda.Stream` may be the only consumer of certain torch imports; if an import becomes unused, delete it. `grep` to verify.

---

## What to KEEP (do not touch)

Every one of these is load-bearing:

- `__new__`, `__init__`, `initialize`, singleton + thread-lock pattern.
- `load_models`, `get_models`, `get_models_dict`, `get_yolo_model`, `get_pose_model`.
- `try_enqueue`, `mark_enqueued_started`, `release_enqueue_on_error`.
- `increment_active`, `can_accept_job`, `acquire_gpu_slot` (the semaphore — used by main.py and video_controller).
- `handle_oom_error`, `reduce_batch_size`, `reset_batch_size`, `current_batch_size`, `get_current_batch_size`.
- `clear_gpu_memory`, `clear_memory_cache`, `_log_gpu_memory_stats`, `get_memory_stats`, `get_status`, `log_startup_stats`, `reset_peak_stats`, `check_memory_health`.
- All `@property` accessors (`active_count`, `is_initialized`, `models_loaded`, `device_name`, `device`, `is_gpu_available`, `total_memory_mb`, `max_concurrent`).
- `unload_models`, `shutdown`.
- `get_gpu_resource_manager()` factory + the lazy `__getattr__` shim at the bottom.

---

## Acceptance criteria

1. `wc -l app/services/gpu_resource_manager.py` shows ≥ 100 LOC reduction (target: 150–200).
2. `grep -rn "get_stream\|assign_stream\|release_stream\|_initialize_cuda_streams\|cuda_streams" --include="*.py" /Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/` returns zero hits.
3. `pytest tests/test_gpu_admission.py -x` is green.
4. `pytest tests/` is green.
5. `python -c "from app.services.gpu_resource_manager import get_gpu_resource_manager; print('ok')"` succeeds.
6. App startup is unaffected: `python -c "from app.main import app; print('app ok')"` succeeds (or AST-only check if heavy deps unavailable).

---

## Out of scope

- Refactoring or simplifying any method that stays.
- Changing the singleton pattern.
- Touching `multiprocessing_config.py` or `video_multiprocessing.py` — they don't use streams.
- Adding new functionality.
