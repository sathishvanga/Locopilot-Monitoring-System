# Task 0003 — Collapse `/api/video/analyze` and `/api/v1/video/process-and-upload`

**Severity:** LOW (duplication)
**Source:** Architecture review 2026-05-09, finding #10.
**Estimated effort:** 0.5 day.

---

## Problem

`app/controllers/video_controller.py` (1,524 LOC) exposes two near-identical endpoints:

- `POST /api/video/analyze` (lines 166–530) — synchronous video processing.
- `POST /api/v1/video/process-and-upload` (lines 682–1133) — same processing **plus** a final POST to the external CVVR API.

The handler bodies are ~99% duplicated: validation, crew parsing, GPU admission, executor dispatch, response shaping. The only divergence is the final `external_api_service.post_violations()` call in the second endpoint.

---

## Files to change

- `app/controllers/video_controller.py` — extract a shared private helper.

External callers may rely on **both URLs** existing; do not delete either route. The fix is to deduplicate the handler **bodies**, not the routes.

---

## Fix

1. Extract a private async helper, e.g.:
   ```python
   async def _run_video_pipeline(
       request_payload: VideoProcessingRequest,
       *,
       upload_to_external_api: bool,
       request: Request,
   ) -> VideoProcessingResponse:
       ...
   ```
   This function owns: input validation, GPU admission, executor dispatch, response build.
2. The `/api/video/analyze` route handler becomes a thin wrapper that calls `_run_video_pipeline(..., upload_to_external_api=False)`.
3. The `/api/v1/video/process-and-upload` route handler calls `_run_video_pipeline(..., upload_to_external_api=True)`. The external-API POST happens inside the helper, gated on the flag.
4. Both routes keep their original FastAPI decorators, paths, status codes, and `response_model`.
5. If a code path differs between the two endpoints (e.g., different success status codes, different DLQ behavior), preserve that difference behind a parameter — do not silently change either endpoint's contract.

---

## Acceptance criteria

1. Both URLs (`/api/video/analyze` and `/api/v1/video/process-and-upload`) still respond. Verified via `pytest tests/controllers/`.
2. Response payloads for the same input are byte-identical to pre-refactor for the non-upload endpoint, and identical-modulo-the-extra-upload-call for the upload endpoint.
3. The external API POST happens only on `/api/v1/video/process-and-upload` (verified by mock).
4. `video_controller.py` LOC drops by **at least 300** (target ~400).
5. No new public functions exposed; the helper is private (`_` prefix).

---

## Out of scope

- Deleting either route.
- Changing the request/response models.
- Touching `external_api_service.py` or `video_processing_service.py`.
- Async-job endpoints (`/api/video/jobs/*`) — leave them alone.
