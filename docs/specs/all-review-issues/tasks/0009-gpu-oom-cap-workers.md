# Task 0009: All GPU workers load duplicate models -- OOM risk

- **Issue ID:** C-04
- **Priority:** Phase 2 - Multiprocessing Fixes (Item 9)
- **Severity:** CRITICAL
- **Category:** Resource Management
- **File:** `app/utils/video_multiprocessing.py:146-167`

## Description

When `YOLO_DEVICE=0` (GPU), every worker process loads both YOLO26 and YOLO26-pose onto the same GPU. With `max_workers_cap=10`, this means up to 10 copies (500MB-1GB each). On RTX 4000 Ada (20GB), workers will OOM.

## Fix

When using GPU, cap `max_workers` to 4-6 (GPU inference is already parallelized via CUDA). Add validation in `get_num_workers()` that caps workers when `yolo_device != 'cpu'`.
