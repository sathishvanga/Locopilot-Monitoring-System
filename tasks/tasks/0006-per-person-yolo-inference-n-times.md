# CR-006: Per-person YOLO inference runs full-frame detection N times for N persons

- **Severity:** High
- **Category:** Performance / Bottleneck
- **Lines:** 4502

## Description

In `process_all_persons_activities`, full-frame YOLO inference is run once per detected person per frame, causing O(N) GPU inference calls where N is the number of persons.

## Suggested Fix

Run YOLO inference once per frame and distribute the results to each person based on bounding box overlap.
