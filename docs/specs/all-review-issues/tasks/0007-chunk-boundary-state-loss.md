# Task 0007: Temporal state discontinuity at chunk boundaries

- **Issue ID:** C-02
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 7)
- **Severity:** CRITICAL
- **Category:** Multiprocessing Correctness
- **File:** `app/utils/video_multiprocessing.py:468-473`

## Description

Each worker creates a fresh `LocopilotActivityMonitor` with zeroed temporal state (consecutive_detections=0, sleep state=AWAKE, no baseline calibration). Activities spanning chunk boundaries are split or lost entirely. A 20s sleep episode split across two 15s chunks may not meet `min_duration` or `required_consecutive` in either chunk.

## Fix

Implement the two-pass pipeline: Pass 1 workers emit raw per-frame detection booleans, Pass 2 runs sequential temporal filtering in the main process. Alternative: add overlap regions where each chunk processes N extra seconds from the previous chunk to warm up state.
