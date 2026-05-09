# Task 0001 — Add `concurrent_grouping_after_vlm` config flag

**Severity:** FOUNDATIONAL (gates tasks 0002, 0003, 0004)
**Estimated effort:** 15 min.

---

## Goal

Add a new pydantic-settings field that gates the post-verifier-merge ordering. All other tasks read this flag.

## Files to change

- `app/utils/config.py` — add the new setting near the other VLM flags.

## Concrete change

Add the field to the `Settings` class in `app/utils/config.py`, in the same group as the other VLM flags (around lines 720–815 where `vlm_*` settings live):

```python
# When 1, concurrent_activity_grouping_service.group_concurrent_activities
# runs AFTER VLM verification rather than before. The verifier therefore
# sees raw single-type activities and never has to un-merge a combined
# record (the per-sub-type fanout in vlm/service.py becomes a no-op).
# Grouping then runs on the post-VLM survivor set, so merged clips only
# include sub-clips that survived verification. Default 0 = legacy
# pre-VLM grouping. Production opts in via .env.production after smoke
# tests pass on the GPU box. Phase B (separate task) deletes the now-dead
# fanout code.
concurrent_grouping_after_vlm: bool = bool(int(os.getenv("CONCURRENT_GROUPING_AFTER_VLM", "0")))
```

## Acceptance criteria

1. `from app.utils.config import get_settings; get_settings().concurrent_grouping_after_vlm` returns `False` when env var is unset.
2. `CONCURRENT_GROUPING_AFTER_VLM=1` in env → `get_settings().concurrent_grouping_after_vlm` is `True`.
3. `pytest tests/unit/test_settings_validator.py -q` still passes.
4. `pytest tests/ -q --ignore=tests/regression` passes (full suite, untouched behaviour).
5. No other code is modified by this task. Tasks 0002/0003 will read the flag.

## Test additions

Add a small unit test if `tests/unit/test_settings_validator.py` doesn't already cover the on/off shape, otherwise no new tests required for this task — the integration test in Task 0004 exercises the flag end-to-end.

## Definition of done

- Field exists, default `False`.
- Env var `CONCURRENT_GROUPING_AFTER_VLM=1` flips it to `True`.
- Full test suite green.
- Code committed with message `config(grouping): add CONCURRENT_GROUPING_AFTER_VLM flag`.

## Out of scope

- Reading the flag from any caller — that is Tasks 0002 and 0003.
- Updating `.env.example` (handled in a Phase B doc-sync task).
