# Task 0017: Synchronous video processing blocks async event loop

- **Issue ID:** H-21
- **Priority:** Phase 3 - Security & Reliability (Item 17)
- **Severity:** HIGH
- **Category:** Performance / Reliability
- **File:** `app/controllers/video_controller.py:288`

## Description

`video_processing_service.process_video(...)` called synchronously from `async def` endpoint. Blocks the entire event loop for minutes during ML inference.

## Fix

Wrap in `asyncio.get_running_loop().run_in_executor(None, ...)`.
