# CR-003: Near-identical duplication between `process_video` and `process_video_range`

- **Severity:** High
- **Category:** Maintainability / Duplication
- **Lines:** ~6200 (`process_video`, 482 lines) and ~6721 (`process_video_range`, 435 lines)

## Description

These two methods share extensive logic (frame sampling, YOLO detection, person deduplication, multi-person activity processing, activity lifecycle management) but are duplicated. Bug fixes must be manually replicated in both.

## Affected Code

Both methods contain duplicated: frame sampling via `sample_video_frames()`, YOLO object detection, person box deduplication with IOU=0.5, multi-person activity processing, and activity lifecycle management.

## Suggested Fix

Extract shared logic into a common `_process_frames_core()` method that both `process_video` and `process_video_range` call with their specific parameters.
