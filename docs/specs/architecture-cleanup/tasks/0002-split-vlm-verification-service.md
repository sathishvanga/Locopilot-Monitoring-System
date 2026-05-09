# Task 0002 — Split `vlm_verification_service.py` into a package

**Severity:** MEDIUM (god-class)
**Source:** Architecture review 2026-05-09, finding #3.
**Estimated effort:** 1–2 days.

---

## Problem

`app/services/vlm_verification_service.py` is **2,486 lines** in a single class with ~14 internal helper functions covering:

- HTTP client + retry + circuit breaker
- Confidence calibration (isotonic + temperature mapping)
- JSONL telemetry writers with thread locks
- Multi-keyframe stitching
- ROI detection + cropping + base64 encoding
- Multi-frame consistency checks
- Bounding-box counting
- Verdict JSON parsing

These are five orthogonal CV/API concerns living in one file. New contributors cannot reason about correctness, and the test surface is unfocused. Behavior must be preserved exactly — `tests/regression/vlm_fixture/` is the source of truth.

---

## Files to change

**Replace `app/services/vlm_verification_service.py` with a package:**

```
app/services/vlm/
  __init__.py               # re-exports VlmVerificationService for back-compat
  service.py                # VlmVerificationService orchestrator (~400 LOC)
  vlm_client.py             # HTTP client, retry, circuit breaker (~400 LOC)
  keyframe_processor.py     # resolve / supplement / stitch / bbox count (~400 LOC)
  image_encoder.py          # ROI detect, crop, base64 encode (~400 LOC)
  verdict_parser.py         # JSON parse, calibration, motion-state logic (~400 LOC)
```

**Keep `app/services/vlm_verification_service.py`** as a 3-line shim:
```python
from app.services.vlm.service import VlmVerificationService

__all__ = ["VlmVerificationService"]
```

This keeps every existing import (`from app.services.vlm_verification_service import VlmVerificationService`) working unchanged.

---

## Fix — module boundaries

### `vlm_client.py`
Owns: HTTP POST to vLLM endpoint, timeout, retry, circuit breaker (the 220–260 region of the original file). Public surface: `class VlmClient` with one async method `infer(messages, *, timeout) -> VlmResponse`.

### `keyframe_processor.py`
Owns: keyframe resolution (which sampled frames to send to the VLM for an activity), keyframe stitching, supplementary-frame selection, bbox counting against detector outputs. Public surface: `KeyframeProcessor.resolve_keyframes(activity, frames) -> list[Keyframe]` and friends.

### `image_encoder.py`
Owns: ROI detection (cropping around the relevant person/object), letterboxing, base64 encoding for the OpenAI-compatible payload. Public surface: `ImageEncoder.encode(frame, roi) -> str`.

### `verdict_parser.py`
Owns: parsing the model's JSON, applying confidence calibration (isotonic / temperature), motion-state-aware verdict logic. Public surface: `VerdictParser.parse(raw_response, activity) -> Verdict` and the `ConfidenceCalibrator`.

### `service.py`
Owns: the public `VlmVerificationService` class (`verify_activities`, `is_enabled`). Composes a `VlmClient`, `KeyframeProcessor`, `ImageEncoder`, `VerdictParser` instance. Owns the JSONL telemetry writer (or that lives in its own tiny `telemetry.py` if it grows).

### Behavior invariants
- `VlmVerificationService(__init__)` signature, `verify_activities()` signature, `is_enabled()` semantics — all unchanged.
- Settings reads (`vlm_*` flags from `app/utils/config.py`) happen in `service.py`; sub-modules receive plain config dicts or kwargs, not `Settings`.
- Telemetry JSONL output paths and schemas unchanged.
- Circuit-breaker state must remain instance-scoped (one breaker per service, not one per request).

---

## Acceptance criteria

1. `pytest tests/regression/vlm_fixture/` passes for all five fixtures (`writing_fp`, `writing_tp`, `idle_person`, `empty_cab`, `no_object_writing`) — these are the behavior contract.
2. `pytest tests/` is fully green.
3. `from app.services.vlm_verification_service import VlmVerificationService` still works.
4. No public method signature on `VlmVerificationService` changes.
5. Each module under `app/services/vlm/` is < 600 LOC.
6. Run a smoke regression: process the same video before and after, compare `activities.json` (excluding wall-clock fields). Must be byte-identical.
7. Telemetry JSONL files written to the same paths as before.

---

## Out of scope

- Removing telemetry / calibration features (separate Wave 2 cull task may decide whether to delete).
- Changing the VLM model or prompt template.
- Editing the consumers (`app/services/video_processing_service.py` should not change).
- Touching `app/utils/config.py` `vlm_*` flags.
