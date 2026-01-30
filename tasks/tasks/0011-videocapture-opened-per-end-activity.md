# CR-011: VideoCapture opened per `end_activity` call just for metadata

- **Severity:** High
- **Category:** Performance / Resource Management
- **Lines:** 5997

## Description

Each time an activity ends, a `VideoCapture` is opened solely to read total frame count and FPS, then immediately closed. This is wasteful I/O for metadata that never changes.

## Affected Code

```python
with video_capture_context(self.video_path) as cap:
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = video_total_frames / fps
```

## Suggested Fix

Cache video metadata (total frames, FPS, duration) once during `__init__` or on first use, and reuse it in all subsequent `end_activity` calls.
