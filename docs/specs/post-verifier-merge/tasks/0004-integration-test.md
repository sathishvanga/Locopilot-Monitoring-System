# Task 0004 — End-to-end integration test for the new ordering

**Severity:** HIGH (locks the contract before any prod rollout)
**Depends on:** Tasks 0001, 0002, 0003 all merged.
**Estimated effort:** 45 min.

---

## Goal

Pin the post-verifier-merge ordering with an integration test that proves three contracts at once:

1. **Verifier sees raw single-type activities** under flag=1 — never `_isCombined`.
2. **Grouping runs after verification** under flag=1, on the survivor set.
3. **The original FP we set out to fix is correctly handled** — given a writing TP and a co-occurring cell_phone FP that would have been minute-merged before, the verifier drops the cell_phone, then grouping emits a single-type writing record (or two single-type records if grouping decides not to bucket — both are acceptable).

The test must run on a clean dev laptop with no GPU, no vLLM, no MinIO.

## File to add

- `tests/services/test_post_verifier_merge.py` (new).

## Test design

Use the same monkey-patching strategy as `tests/services/test_vlm_motion_override.py`:

- Patch `_verify_one_async` on a fresh `VlmVerificationService` instance to return canned verdicts.
- Build a small list of raw activities representing what Pipeline-1 would emit before grouping (writing@t=305, cell_phone@t=319, writing@t=321 — three separate single-type detections in the same minute).
- Drive the full `process_video` or a thin equivalent that exercises both `verify_activities` and `group_concurrent_activities` under the flag.
- Assert:
  - The patched verifier received single-type activities (no `_isCombined`, no parallel `objectTypes`).
  - The patched verifier was called once per raw activity (3 times for the example above).
  - The post-grouping output has the cell_phone FP dropped and at most one merged writing record (depending on bucket logic).
  - With flag=0 (legacy ordering), the same input produces the legacy output (verifier sees a combined record, sub-type fanout fires).

## Suggested helpers

```python
def _raw_activity(idx: int, object_type: str, activity_type: int,
                  start_sec: float, end_sec: float, clip_path: str) -> dict:
    """Build a minimal raw activity (no _isCombined) like Pipeline-1 emits
    before the grouping service ever runs."""
    return {
        "id": f"raw-{idx}",
        "objectType": object_type,
        "activityType": activity_type,
        "des": f"raw {object_type}",
        "motionState": "RUNNING",
        "activityStartTime": f"{start_sec:.2f}",
        "activityEndTime": f"{end_sec:.2f}",
        "performingRole": "LP",
        "activityClip": clip_path,
        "activityImage": clip_path.replace("_clip.mp4", "_activity.jpg"),
    }
```

## Acceptance criteria

1. Test passes with flag=0 (legacy ordering still works).
2. Test passes with flag=1 (new ordering works).
3. Test fails if anyone reverts Task 0002 or Task 0003 (i.e. the assertions cover the regression surface).
4. `pytest tests/services/test_post_verifier_merge.py -q` runs in <5 seconds on a dev laptop.
5. Test does not require network access, GPU, vLLM, or MinIO.

## Definition of done

- Test file added with at least 3 tests:
  - `test_flag_off_preserves_legacy_ordering`
  - `test_flag_on_runs_verify_before_group`
  - `test_flag_on_drops_co_merged_fp_keeps_tp`
- All 26+ existing VLM tests still pass.
- Code committed with message `test(post-verifier-merge): pin verify-then-group ordering`.

## Out of scope

- E2E test that hits a real vLLM endpoint (those live elsewhere).
- Test with real video frames / OCR / S3 — keep it pure-Python with mocks.
