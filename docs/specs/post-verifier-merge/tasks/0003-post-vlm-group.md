# Task 0003 — Run grouping after VLM verify when flag enabled

**Severity:** HIGH (the other half of the ordering swap)
**Depends on:** Task 0001 (flag exists). Independent of Task 0002 (touches disjoint files; both can run in parallel after 0001).
**Estimated effort:** 45 min.

---

## Goal

When `settings.concurrent_grouping_after_vlm` is `True`, run `group_concurrent_activities` on the post-VLM survivor set in both production call paths. Re-save `activities.json` after grouping. Update any downstream filtering (e.g. `clip_files`) that depended on the pre-merged shape.

When flag is `False`, behaviour is byte-identical to today.

## Files to change

- `app/services/video_processing_service.py` — extend the VLM hook around lines 266-293.
- `app/controllers/video_controller.py` — extend the `/process-and-upload` VLM hook around lines 555-611, including the `clip_files` filtering at lines 586-595.

## Concrete change

### Site 1 — `app/services/video_processing_service.py`

Today (lines 266-293) the flow is:
```
save raw activities → run VLM verify → re-save post-VLM activities
```

Under flag=1 it becomes:
```
save raw activities → run VLM verify → re-save post-VLM activities → run grouping → re-save grouped activities
```

After the `self.activity_repository.save_activities(activities=activities, run_dir=run_dir)` call at line 279-282 (the VLM re-save), insert:

```python
if get_settings().concurrent_grouping_after_vlm:
    from .concurrent_activity_grouping_service import get_concurrent_grouping_service
    pre_group_count = len(activities)
    activities = get_concurrent_grouping_service().group_concurrent_activities(
        activities, run_dir
    )
    self.activity_repository.save_activities(
        activities=activities,
        run_dir=run_dir,
    )
    logger.info(
        f"[GROUP] post-VLM grouping: {pre_group_count} -> {len(activities)} "
        f"activities (run after verifier under "
        f"CONCURRENT_GROUPING_AFTER_VLM=1)"
    )
```

### Site 2 — `app/controllers/video_controller.py`

The `/process-and-upload` flow at lines 555-611 currently:
1. Reads `pre_vlm_activities` from disk
2. Runs `verify_activities`
3. Re-saves via `ActivityRepository`
4. Updates `result['activities']`, `result['activitiesCount']`
5. Filters `result['clip_files']` against the post-VLM `activityClip` paths

Under flag=1, after step 3 (the VLM re-save) and before step 4, run grouping:

```python
if get_settings().concurrent_grouping_after_vlm:
    from app.services.concurrent_activity_grouping_service import (
        get_concurrent_grouping_service,
    )
    post_vlm_activities = get_concurrent_grouping_service().group_concurrent_activities(
        post_vlm_activities, run_dir_for_vlm,
    )
    _ActivityRepository().save_activities(
        post_vlm_activities, os.path.dirname(activities_json_for_vlm),
    )
```

The `result['activities']`, `result['activitiesCount']` updates and the `clip_files` filtering at lines 586-595 must use `post_vlm_activities` AFTER grouping (since after grouping, `activityClip` points at the merged minute-NNN clip rather than per-source clips). The existing filter logic (set membership against `activityClip`) keeps working — it just operates on a smaller, merged-clip-aware set.

`get_settings` is already used in this file via `from app.utils.config import get_settings` at the top — confirm before adding the import.

## Acceptance criteria

1. With `CONCURRENT_GROUPING_AFTER_VLM=0` (default), both call paths produce byte-identical output to today. No regressions.
2. With `CONCURRENT_GROUPING_AFTER_VLM=1`:
   - `activities.json` after each path has `_isCombined: True` records (where applicable) — proving grouping ran.
   - `verify_activities` was called with single-type activities (no `_isCombined`).
   - `clip_files` filter only references merged clip paths.
3. `pytest tests/ -q --ignore=tests/regression` passes with the flag both unset and set to `1`.

## Tests

- Existing `tests/services/test_external_api.py` and any test that exercises the full `process_video` path must pass.
- The integration test added by Task 0004 will exercise the full flag=1 ordering — DO NOT add an end-to-end integration test in this task; keep the scope to the call-site changes.

## Definition of done

- Both call sites gated behind the flag.
- Full test suite green at flag=0 and flag=1.
- Code committed with message `refactor(grouping): run grouping after VLM verify when CONCURRENT_GROUPING_AFTER_VLM=1`.

## Out of scope

- Removing the per-sub-type fanout in `vlm/service.py` (Phase B).
- End-to-end integration test (Task 0004).
- Pre-gate semantics changes.
