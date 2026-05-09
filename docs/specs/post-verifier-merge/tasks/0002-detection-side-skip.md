# Task 0002 — Skip detection-side grouping when flag enabled

**Severity:** HIGH (one half of the ordering swap)
**Depends on:** Task 0001 (`concurrent_grouping_after_vlm` flag exists).
**Estimated effort:** 30 min.

---

## Goal

When `settings.concurrent_grouping_after_vlm` is `True`, the three call sites that group activities at detection time must skip grouping and return raw activities. The grouping moves to a later stage (Task 0003).

When the flag is `False`, behaviour is byte-identical to today.

## Files to change

- `app/utils/video_multiprocessing.py` — line 1029-1031 (multi-process detection orchestrator).
- `app/services/activity_detection_service.py` — lines 304-306 (single-process path) and lines 401-403 (multi-process path post-call).

## Concrete change

Wrap each `group_concurrent_activities(...)` call in a flag check. The pattern is the same in all three sites: read settings, branch on the flag.

### Site 1 — `app/utils/video_multiprocessing.py:1029-1031`

Before:
```python
from ..services.concurrent_activity_grouping_service import get_concurrent_grouping_service
concurrent_grouping_service = get_concurrent_grouping_service()
all_activities = concurrent_grouping_service.group_concurrent_activities(all_activities, run_dir)
```

After:
```python
from ..utils.config import get_settings as _get_settings
if not _get_settings().concurrent_grouping_after_vlm:
    from ..services.concurrent_activity_grouping_service import get_concurrent_grouping_service
    concurrent_grouping_service = get_concurrent_grouping_service()
    all_activities = concurrent_grouping_service.group_concurrent_activities(all_activities, run_dir)
# else: grouping deferred to post-VLM in video_processing_service / video_controller
```

### Site 2 — `app/services/activity_detection_service.py:304-306`

Before:
```python
from .concurrent_activity_grouping_service import get_concurrent_grouping_service
concurrent_grouping_service = get_concurrent_grouping_service()
activities = concurrent_grouping_service.group_concurrent_activities(activities, actual_run_dir)
```

After:
```python
if not get_settings().concurrent_grouping_after_vlm:
    from .concurrent_activity_grouping_service import get_concurrent_grouping_service
    concurrent_grouping_service = get_concurrent_grouping_service()
    activities = concurrent_grouping_service.group_concurrent_activities(activities, actual_run_dir)
# else: grouping deferred to post-VLM
```

(`get_settings` is already imported at module level — see top of the file.)

### Site 3 — `app/services/activity_detection_service.py:401-403`

Same pattern as Site 2, using `run_dir` (not `actual_run_dir`) per the existing call.

## Acceptance criteria

1. With `CONCURRENT_GROUPING_AFTER_VLM=0` (default), the output of every call site is byte-identical to the current behaviour. No tests should regress.
2. With `CONCURRENT_GROUPING_AFTER_VLM=1`, the three call sites return raw activities (no `_isCombined`, no parallel `objectTypes` arrays, no merged `activityClip`).
3. `pytest tests/ -q --ignore=tests/regression` passes with the flag both unset and set to `1`.

## Tests

- Run the full existing suite with `CONCURRENT_GROUPING_AFTER_VLM=0` (default) — must pass unchanged.
- Run the full existing suite with `CONCURRENT_GROUPING_AFTER_VLM=1` — must still pass (post-VLM grouping isn't wired yet, but the existing tests don't depend on grouping happening at detection time).
- Detection-pipeline tests in `tests/refactor/` and `tests/test_determinism.py` are the most likely to surface regressions; pay extra attention to those.

## Definition of done

- All three call sites gated behind the flag.
- Full test suite green at flag=0 and flag=1.
- Code committed with message `refactor(grouping): gate detection-side grouping behind CONCURRENT_GROUPING_AFTER_VLM`.

## Out of scope

- Adding the post-VLM grouping call (Task 0003).
- Touching `vlm/service.py` (Phase B).
- End-to-end integration test (Task 0004).
