# Task 0003 — Wrap the production endpoint and queue worker in `acquire_gpu_slot`

**Severity:** CRITICAL
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #2, top-fix #3.
**Estimated effort:** Half day.

---

## Problem

The 1-worker Gunicorn + `MAX_CONCURRENT_VIDEOS` semaphore is the entire safety story for OOM avoidance on the 20 GB RTX 4000 Ada (Pipeline-1 ~8 GB peak + vLLM ~10 GB resident leaves ~3 GB margin).

Two endpoints **bypass this admission gate entirely**:

1. **`/api/v1/video/process-and-upload`** — the production endpoint per CLAUDE.md (`app/controllers/video_controller.py:635-746`). Calls `video_processing_service.process_video` directly inside a thread executor. Two simultaneous POSTs both grab the GPU at the same time.
2. **`process_video_job`** — the queue worker (`app/main.py:85-135`). Three queue workers can run three videos concurrently with no GPU gate.

The `gpu_resource_manager._semaphore` is itself fragile — constructed in sync `initialize()` outside any event loop, ties to whichever loop runs the first `await`. If awaited from a different loop later (test runner, background `asyncio.run`), it raises `RuntimeError: <Semaphore [...]> is bound to a different event loop`.

The `try_enqueue/release_enqueue_on_error/mark_enqueued_started` triplet has a **double-counting bug**: `acquire_gpu_slot()` itself increments `_active_count`, but `try_enqueue` also increments `_pending_count` and reads `_active_count + _pending_count` — so position calculation and the `try_enqueue` cap double-count one slot.

---

## Files to change

- `app/controllers/video_controller.py:635-746` — `process_video_and_upload`
- `app/main.py:85-135` — `process_video_job`
- `app/services/gpu_resource_manager.py:209, 228, 448, 537` — semaphore lifecycle + counter authority

---

## Fix

### Lazy semaphore construction

In `gpu_resource_manager.py`, remove `self._semaphore = asyncio.Semaphore(...)` from `initialize()`. Make `acquire_gpu_slot()` lazy-create on first `await`:

```python
async def acquire_gpu_slot(self):
    if self._semaphore is None:
        self._semaphore = asyncio.Semaphore(self._max_concurrent_videos)
    async with self._semaphore:
        ...
```

### Single counter authority

Decide one source of truth: either the semaphore OR the active/pending pair, not both. Recommended: keep the semaphore for blocking + drop `_active_count` increment from `acquire_gpu_slot` (rely on `_pending → active` transition in `mark_enqueued_started`). Update the `try_enqueue` cap to read `_pending_count` only.

### Wrap the two bypassing call sites

Both endpoints must run their heavy body inside:

```python
async with gpu_resource_manager.acquire_gpu_slot():
    await asyncio.to_thread(video_processing_service.process_video, ...)
```

For `process_video_and_upload`, also adopt the existing `try_enqueue / release_enqueue_on_error` finally pattern that `process_video` (analyze endpoint) uses at `video_controller.py:259-281, 372-374, 451-460`.

For `process_video_job`, instantiate `VideoProcessingService` via the singleton accessor (Task 0010 will introduce it; for now reuse `get_video_processing_service` if it exists, else inline-construct once at module scope).

### Initialize counters outside `try`

Move `admitted = False; slot_acquired = False` to *before* the `try` block at `video_controller.py:259`. Drop the `locals()` introspection in the `finally` rollback condition at line 451-460 — replace with explicit names.

---

## Acceptance criteria

1. Add `tests/test_gpu_admission.py` that:
   - Posts 5 concurrent requests to `/api/v1/video/process-and-upload` with `MAX_CONCURRENT_VIDEOS=2`. Asserts at most 2 are in `process_video` simultaneously (use a mock that records concurrency).
   - Submits 5 jobs via `/api/video/jobs` with `MAX_CONCURRENT_VIDEOS=2`. Same assertion.
2. `grep -rn "acquire_gpu_slot" app/` shows the new wrapping in both endpoints.
3. Existing semaphore tests pass. New test: construct `GPUResourceManager()` synchronously, then in two separate `asyncio.run()` calls await `acquire_gpu_slot()` — must not raise "bound to a different event loop".
4. The `try_enqueue` queue position no longer double-counts: when `_active_count=1` and queue is empty, a new request reports `position=2` not `position=3`.

---

## Out of scope

- Increasing `MAX_CONCURRENT_VIDEOS` past current production value.
- Migrating from `asyncio.Semaphore` to a different primitive.
