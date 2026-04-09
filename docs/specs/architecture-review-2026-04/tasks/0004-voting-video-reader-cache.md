# Task 0004: Per-worker long-lived `VideoReader` for voting verification

- **Issue ID:** ARCH-04
- **Priority:** Medium-impact, low-effort
- **Severity:** MEDIUM — O(frames sampled) re-opens in the hot loop
- **Category:** Performance
- **Files:**
  - `app/services/voting_verification_service.py:820-870`
    (`_extract_native_frames_near_timestamp`)
  - `app/utils/video_multiprocessing.py:236-290` (`_worker_models` dict)
  - `locopilot_monitor.py:3623, 4454` (voting invocation sites)

## Description

`_extract_native_frames_near_timestamp` opens a new `cv2.VideoCapture` and
seeks per call:

```python
# app/services/voting_verification_service.py:831
cap = cv2.VideoCapture(video_path)
...
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
```

`verify_batch` amortizes across activities at the same timestamp, but every
(video, timestamp) still re-opens and re-seeks the container. At 0.5 fps
sampling over a 30-minute video, that's hundreds of `VideoCapture` opens per
worker. H.264 seeks are not free.

## Fix

Introduce a `VideoReader` wrapper that keeps a single `cv2.VideoCapture`
open per `(worker_process, video_path)` and exposes a
`read_frames_near(timestamp_sec, num_frames)` method.

1. Create `app/services/video_reader.py`:
   ```python
   class VideoReader:
       def __init__(self, video_path: str):
           self.path = video_path
           self.cap = cv2.VideoCapture(video_path)
           self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
           self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

       def read_frames_near(self, timestamp_sec: float, num_frames: int) -> List[np.ndarray]:
           center = int(timestamp_sec * self.fps)
           start = max(0, center - num_frames // 2)
           if start + num_frames > self.total_frames:
               start = max(0, self.total_frames - num_frames)
           self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
           frames = []
           for _ in range(num_frames):
               ret, frame = self.cap.read()
               if not ret:
                   break
               frames.append(frame)
           return frames

       def close(self):
           if self.cap.isOpened():
               self.cap.release()
   ```

2. Cache it per worker in `_worker_models` as a small LRU
   (`{video_path: VideoReader}`) — typically size 1 in production.

3. In `VotingVerificationService`, accept an optional `video_reader` in the
   constructor (or lazily get it from the worker state) and replace the
   body of `_extract_native_frames_near_timestamp` with a single call to
   `video_reader.read_frames_near(timestamp_sec, num_frames)`.

4. Ensure `VideoReader.close()` is called from `monitor.cleanup()` and from
   the worker shutdown path so releases happen deterministically.

## Acceptance criteria

- [ ] `app/services/video_reader.py` exists with a tested `VideoReader`
      class that reuses a single `cv2.VideoCapture` per instance.
- [ ] `_extract_native_frames_near_timestamp` no longer instantiates
      `cv2.VideoCapture`; it calls into the cached reader.
- [ ] `_worker_models['video_readers']` holds a dict keyed by video path.
- [ ] `cleanup()` / `shutdown_shared_pool()` releases all `VideoReader`
      instances (grep for `cap.release()` should find one site inside
      `VideoReader.close`).
- [ ] On a fixture video, `verify_batch` completes at least 2× faster on
      the second voting call than the first (warm cap vs cold cap).

## Implementation status

Branch: `feat/arch-review-2026-04/0004-voting-video-reader-cache`
Date: 2026-04-09

### Files created
- `app/services/video_reader.py` - `VideoReader` wrapper (single open, cached
  fps/total_frames, context manager) + `VideoReaderLRU` (tiny per-worker LRU,
  default max_size=2).
- `tests/unit/test_video_reader.py` - 16 unit tests (all passing) covering
  single-open reuse, seek windowing, EOF/clamping, close idempotency, context
  manager, LRU eviction, stale-entry reopening, and
  `VotingVerificationService` wiring (both fast path and fallback path).
- `tests/__init__.py`, `tests/unit/__init__.py` - package markers.

### Files changed
- `app/services/voting_verification_service.py`
  - `__init__` accepts optional `video_reader_getter` callable.
  - `_extract_native_frames` now prefers the cached reader when the getter
    is set and falls back to the original `cv2.VideoCapture` open/seek/release
    path when it is not (preserves single-process semantics).
- `app/utils/video_multiprocessing.py`
  - `worker_initializer` populates `_worker_models['video_readers']` with a
    fresh `VideoReaderLRU(max_size=2)`.
  - New helper `get_or_create_video_reader(video_path)` exposed at module
    level for workers that want the cached reader directly.
  - New helper `_close_worker_video_readers()` releases every cached reader;
    registered as an atexit hook inside workers and called from
    `shutdown_shared_pool` so the main process releases any readers it owns.
- `locopilot_monitor.py`
  - Monitor pulls `video_readers` out of `preloaded_models` and passes
    `lru.get_or_create` into `VotingVerificationService` as the
    `video_reader_getter`.
  - `cleanup()` releases readers when the monitor owns them (single-process
    mode) and leaves the shared LRU alone in worker pool mode where the
    worker atexit hook is responsible for teardown.

### Acceptance criteria

- [x] `app/services/video_reader.py` exists with a tested `VideoReader`
      class that reuses a single `cv2.VideoCapture` per instance. Verified
      by `test_multiple_read_frames_near_calls_reuse_single_capture`.
- [x] `_extract_native_frames` no longer instantiates `cv2.VideoCapture`
      on the hot path when a getter is wired (worker pool mode). It calls
      into the cached reader. Verified by
      `test_voting_service_accepts_getter_and_delegates_extraction`.
- [x] `_worker_models['video_readers']` holds a dict (LRU) keyed by video
      path. Populated in `worker_initializer`, backfilled by the helper.
- [x] Releases are wired: `VideoReader.close` -> `VideoReaderLRU.close_all`
      -> `_close_worker_video_readers` is called from worker atexit,
      `shutdown_shared_pool`, and single-process `monitor.cleanup()`.
      Note: the single-process fallback inside `_extract_native_frames`
      still contains a `cap.release()` to preserve the
      "single-process fallback" constraint from the task brief; this is
      intentional and only runs when `video_reader_getter is None`.
- [ ] Wall-clock speedup on a fixture video: NOT verified. Running the
      full pipeline is outside the scope of this task (no GPU, no models);
      covered indirectly by the unit test proving the capture is opened
      exactly once across N extraction calls.

### Tests
`python -m pytest tests/unit/test_video_reader.py -q` -> 16 passed.
