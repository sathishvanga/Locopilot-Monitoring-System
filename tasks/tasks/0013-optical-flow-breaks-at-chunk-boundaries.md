# CR-013: Optical flow breaks at chunk boundaries in multiprocessing mode

- **Severity:** Medium
- **Category:** Bug / Multiprocessing
- **Lines:** 7030

## Description

In `process_video_range`, `_prev_motion_frame` is `None` at each chunk start because each worker has fresh state. This means optical flow-based motion detection produces no results for the first frame(s) of every chunk.

## Suggested Fix

Pass the last frame of the previous chunk as initialization data to the next chunk's worker, or add a small overlap between chunks (e.g., 1-2 frames).
