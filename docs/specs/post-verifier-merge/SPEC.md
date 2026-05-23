# Post-verifier merge refactor

**Severity:** MEDIUM (architectural debt)
**Source:** Production debugging 2026-05-09 (run_20260509_072704 → run_20260509_092003 series).
**Estimated effort:** 1 day.

---

## Problem

The classical detection pipeline emits raw activities, then `concurrent_activity_grouping_service.group_concurrent_activities` merges all activities sharing a minute bucket and `performingRole` into a single combined record (with `_isCombined: True`, parallel `objectTypes` / `activityTypes` / `descriptions` arrays, and a re-encoded merged `activityClip`). This merged record is what the VLM verifier sees.

That ordering forces the verifier to *un-merge* a combined record at verify time so each rule-fire (e.g. `writing` and `cell_phone` glued together by the minute-grouper) can be judged independently. The current code does that with a per-sub-type fanout in `app/services/vlm/service.py` (`_extract_subtype_views`, `_aggregate_subtype_results`, `_strip_subtypes_from_parent`) plus an aggregator that reconstructs a pseudo-review for the parent record — about 350 lines of bespoke machinery exists solely to compensate for grouping happening before verification.

The fix is to swap the order: run the VLM verifier on raw single-type activities, then group the survivors. The verifier collapses to a 1:1 single-type-per-call shape (the pre-multi-sub form), and the grouping service runs on a smaller, already-validated set.

---

## Pipelines, before and after

### Current

```
Pipeline-1 detects N raw activities
  └─ group_concurrent_activities       ← emits M ≤ N records, some _isCombined
  └─ save activities.json
  └─ vlm_service.verify_activities      ← per-sub-type fanout for combined
                                            records; aggregator re-stitches
  └─ save activities.json (rewrite)
  └─ external_api / S3 upload
```

### Target (gated by `CONCURRENT_GROUPING_AFTER_VLM=1`)

```
Pipeline-1 detects N raw activities
  └─ save activities.json (raw)
  └─ vlm_service.verify_activities      ← N single-type calls, no fanout
  └─ save activities.json (post-VLM)
  └─ group_concurrent_activities        ← emits M ≤ N' surviving combined records
  └─ save activities.json (final)
  └─ external_api / S3 upload
```

The default of the new flag is `0` (legacy behavior preserved); production opts in via `.env.production`.

---

## Why a flag

1. **Atomic rollback**: production currently runs the pre-merge form. A regression in any of the four call sites that move grouping must be rollback-able by flipping a single env var, not by reverting code.
2. **Behavior parity for tests**: existing test suite was written against the pre-merge ordering. We keep both code paths live while we build confidence in the new one.
3. **Clean Phase B**: once the flag has been stable in prod for a week, Phase B (separate task spec) deletes the now-dead per-sub-type fanout code (~350 lines).

---

## Call sites that move

| Current call site | What it does | Action under flag=1 |
|---|---|---|
| `app/utils/video_multiprocessing.py:1029-1031` | groups raw multi-process detections before save | skip |
| `app/services/activity_detection_service.py:304-306` | groups raw single-process detections | skip |
| `app/services/activity_detection_service.py:401-403` | groups multi-process fallback path | skip |
| `app/services/video_processing_service.py:266-293` | runs VLM verify on already-grouped activities | after VLM, run grouping; re-save |
| `app/controllers/video_controller.py:555-611` | runs VLM verify in the `/process-and-upload` flow on already-grouped activities; filters `clip_files` | after VLM, run grouping; re-save; filter `clip_files` against post-grouped clips |

---

## Risks and out-of-scope

### Risks

1. **VLM call volume rises** by roughly the average bucket size (typically ~2× for trips with co-occurring detections). Mitigated by `VLM_MAX_ACTIVITIES_PER_RUN` cap.
2. **`vlm_review` field shape changes**: combined records previously carried `vlm_review.subtype_reviews`. Under flag=1 they carry per-source-activity reviews aggregated by the grouping step (or none, depending on grouping policy on `vlm_review`). Downstream consumers (UI, audit) tolerate missing fields.
3. **The `t=191` pre-gate-on-flickery-bbox case is NOT directly fixed by this refactor** — that's a separate fix on `_count_bboxes_in_keyframes` semantics. This refactor SIMPLIFIES the codebase enough that the pre-gate fix becomes a one-config-flag change.

### Out of scope (deliberately)

- Pre-gate behavior tuning (separate task).
- Deletion of `_extract_subtype_views`, `_aggregate_subtype_results`, `_strip_subtypes_from_parent` (Phase B, separate task once flag has stabilised).
- Re-encoding merged MP4s on partial drops.
- Verifier prompt changes.

---

## Tasks

| # | Title | Files | Dependencies |
|---|---|---|---|
| 0001 | Add `concurrent_grouping_after_vlm` config flag | `app/utils/config.py` | none |
| 0002 | Skip detection-side grouping when flag enabled | `app/utils/video_multiprocessing.py`, `app/services/activity_detection_service.py` | 0001 |
| 0003 | Run grouping after VLM verify when flag enabled | `app/services/video_processing_service.py`, `app/controllers/video_controller.py` | 0001 |
| 0004 | End-to-end integration test for the new ordering | `tests/services/test_post_verifier_merge.py` | 0001, 0002, 0003 |

Tasks 0002 and 0003 touch disjoint file sets and may run in parallel after 0001 lands.
