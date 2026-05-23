"""Task 0004 — End-to-end pin for the post-verifier-merge ordering.

This test pins the contract introduced by the post-verifier-merge refactor
(spec at ``docs/specs/post-verifier-merge/SPEC.md``) without standing up a
real video pipeline / vLLM / MinIO. Instead it drives the same verify+group
sequence the production controller and ``video_processing_service`` execute,
mirroring the pseudocode added in Task 0003:

    if get_settings().concurrent_grouping_after_vlm:
        kept, _ = vlm_service.verify_activities(activities)
        kept = grouping_service.group_concurrent_activities(kept, run_dir)
    else:
        grouped = grouping_service.group_concurrent_activities(activities, run_dir)
        kept, _ = vlm_service.verify_activities(grouped)

The three required tests are:

1. ``test_flag_off_preserves_legacy_ordering`` — flag=0 means grouping runs
   FIRST, so the verifier sees a ``_isCombined`` parent record (legacy
   per-sub-type fanout territory).
2. ``test_flag_on_runs_verify_before_group`` — flag=1 means the verifier
   sees raw single-type activities (no ``_isCombined``, no parallel
   ``objectTypes``), then grouping runs on the survivors.
3. ``test_flag_on_drops_co_merged_fp_keeps_tp`` — the production case that
   motivated the refactor: writing TP @ t=305 + cell_phone FP @ t=319 in the
   same minute. Under flag=1, the verifier drops the cell_phone (FP) and
   grouping receives ONLY writing single-type records, so the post-grouping
   output never carries a parallel ``cell phone`` ``objectTypes`` slot.

Mocking strategy: same as ``tests/services/test_vlm_motion_override.py`` and
``tests/services/test_vlm_merged_subtype.py`` — patch ``_verify_one_async``
on a fresh ``VlmVerificationService`` instance with canned per-activity
verdicts. No real vLLM, no GPU, no MinIO, no ffmpeg. ``run_dir=None`` is
passed to the grouping service so the merge_clips branch (which would shell
out to ffmpeg) is skipped — see ``_merge_video_clips`` length-1 path.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Environment defaults — set BEFORE importing app modules so pydantic Settings
# picks them up. Mirrors the pattern in test_vlm_motion_override.py.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("VLM_VERIFICATION_ENABLED", "1")
os.environ.setdefault("VLM_VERIFY_ACTIVITIES", "writing,cell_phone,eating_drinking")
os.environ.setdefault("VLM_SHADOW_MODE", "0")
os.environ.setdefault("VLM_DROP_THRESHOLD", "0.80")

import pytest

from app.utils.config import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_activity(
    idx: int,
    object_type: str,
    activity_type: int,
    start_sec: float,
    end_sec: float,
    clip_path: str,
) -> dict:
    """Build a minimal raw single-type activity (no ``_isCombined``) like
    Pipeline-1 emits BEFORE the grouping service ever runs.

    Shape is the contract the verifier expects under flag=1: no
    ``_isCombined``, no parallel ``objectTypes`` / ``activityTypes`` /
    ``descriptions``, no ``_sourceActivities``. One detection, one type.
    """
    return {
        "id": f"raw-{idx}",
        "objectType": object_type,
        "activityType": activity_type,
        "des": f"raw {object_type}",
        "motionState": "RUNNING",
        "performingRole": "LP",
        "activityStartTime": f"{start_sec:.2f}",
        "activityEndTime": f"{end_sec:.2f}",
        "activityClip": clip_path,
        # Use a definitely-missing path; the patched ``_verify_one_async``
        # never decodes a frame so this never has to exist on disk.
        "activityImage": clip_path.replace("_clip.mp4", "_activity.jpg"),
    }


def _fresh_service():
    """Return a ``VlmVerificationService`` with cleanly reset settings.

    Mirrors the pattern in ``test_vlm_motion_override.py`` /
    ``test_vlm_merged_subtype.py``. Ensures the circuit breaker is closed
    and the verify-set / drop threshold match the test's expectations.
    """
    from app.services.vlm_verification_service import VlmVerificationService
    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = False
    svc.settings.vlm_drop_threshold = 0.80
    svc.settings.vlm_motion_override_enabled = False
    svc._verify_set = frozenset({"writing", "cell_phone", "eating_drinking"})
    return svc


def _make_verdict(verdict: str, conf: float, **extra) -> dict:
    """Build the review dict shape ``_verify_one_async`` returns on the OK
    path. The verifier reads ``status``, ``verdict.verdict``, and
    ``verdict.confidence`` from this; everything else is opaque."""
    body = {"verdict": verdict, "confidence": conf, "reasoning": f"test {verdict}"}
    body.update(extra)
    return {
        "status": "OK",
        "verdict": body,
        "latency_sec": 0.05,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "frames_sent": 1,
    }


def _set_flag(value: bool) -> None:
    """Toggle ``CONCURRENT_GROUPING_AFTER_VLM`` and bust the
    ``lru_cache`` on ``get_settings`` so subsequent reads see the new value.
    Without ``cache_clear()`` the first ``get_settings()`` call's value is
    sticky for the rest of the test session — Tasks 0001/0002 agent notes
    flagged this gotcha explicitly.
    """
    os.environ["CONCURRENT_GROUPING_AFTER_VLM"] = "1" if value else "0"
    get_settings.cache_clear()


def _drive_pipeline(activities: list, svc, *, flag_on: bool) -> tuple:
    """Drive the verify+group sequence the way ``video_controller`` and
    ``video_processing_service`` do under each flag value.

    Under flag=0 (legacy):
        group(activities) -> verify(grouped) -> kept

    Under flag=1 (new):
        verify(activities) -> group(verified_kept) -> kept

    ``run_dir=None`` is fine for the grouping service: the only branch that
    actually touches the filesystem is ``_merge_video_clips`` for groups of
    >1 source-clips, and that branch short-circuits to the first clip when
    ``run_dir`` is missing — keeping the test pure-Python.

    Returns (kept, vlm_stats).
    """
    from app.services.concurrent_activity_grouping_service import (
        get_concurrent_grouping_service,
    )

    grouping = get_concurrent_grouping_service()

    if flag_on:
        kept, stats = svc.verify_activities(activities)
        kept = grouping.group_concurrent_activities(kept, None)
    else:
        grouped = grouping.group_concurrent_activities(activities, None)
        kept, stats = svc.verify_activities(grouped)

    return kept, stats


# ---------------------------------------------------------------------------
# Test 1 — Flag OFF preserves legacy ordering: group runs BEFORE verify, so
# the verifier sees a `_isCombined` parent. This protects the rollback path.
# ---------------------------------------------------------------------------

def test_flag_off_preserves_legacy_ordering():
    """When ``CONCURRENT_GROUPING_AFTER_VLM=0`` the grouping service must
    run FIRST and the verifier must see the (potentially combined) result.

    Three same-minute single-type activities collapse into one combined
    record (writing+cell_phone in minute bucket 5), so the verifier
    receives ONE activity with ``_isCombined=True`` and a parallel
    ``objectTypes`` array — the exact shape the per-sub-type fanout
    machinery in ``vlm/service.py`` was built to handle.
    """
    _set_flag(False)
    assert get_settings().concurrent_grouping_after_vlm is False, (
        "flag must be OFF for this test; check env wiring"
    )

    svc = _fresh_service()
    seen_calls: list[dict] = []

    async def fake_verify(activity, prompt, object_type):
        # Snapshot what the verifier saw so we can assert on shape AFTER
        # the call. We deepcopy the relevant fields rather than the whole
        # dict to avoid pulling in the (mutating) ``_sourceActivities`` ref.
        seen_calls.append({
            "object_type": object_type,
            "is_combined": bool(activity.get("_isCombined")),
            "object_types": list(activity.get("objectTypes", [])),
            "activity_types": list(activity.get("activityTypes", [])),
            "primary_object_type": activity.get("objectType"),
        })
        # Return TP for both sub-types so nothing gets dropped — we're
        # testing ORDER, not drop semantics.
        return _make_verdict("TRUE_POSITIVE", 0.90)

    svc._verify_one_async = fake_verify  # type: ignore[assignment]

    activities = [
        _raw_activity(1, "writing",    5, 305.0, 315.0, "/tmp/raw_writing_a_clip.mp4"),
        _raw_activity(2, "cell phone", 2, 319.0, 329.0, "/tmp/raw_cell_phone_a_clip.mp4"),
        _raw_activity(3, "writing",    5, 321.0, 325.0, "/tmp/raw_writing_b_clip.mp4"),
    ]

    kept, _stats = _drive_pipeline(activities, svc, flag_on=False)

    # Under legacy ordering, grouping ran FIRST and merged the 3 raw
    # detections into 1 combined minute-bucket-5 record before the verifier
    # ever saw them. The verifier should therefore have observed the
    # COMBINED parent shape (per-sub-type fanout: one call for writing, one
    # for cell_phone — the verifier's own machinery generates 2 calls from
    # the single combined input).
    assert any(c["is_combined"] for c in seen_calls), (
        "flag=0: legacy ordering must surface a _isCombined parent to the "
        "verifier — if no call saw _isCombined the grouping step never ran "
        "before verify and Task 0002 has been reverted"
    )
    # The fanout produces one call per distinct sub-type from the combined
    # parent. With the (writing + cell_phone) merge that's 2 sub-type calls.
    sub_types_seen = {c["object_type"] for c in seen_calls}
    assert "writing" in sub_types_seen
    assert "cell_phone" in sub_types_seen

    # And the kept output preserves the combined parent (TP for both
    # sub-types means no drop / no strip).
    assert len(kept) == 1
    assert kept[0].get("_isCombined") is True


# ---------------------------------------------------------------------------
# Test 2 — Flag ON: verify sees RAW single-type activities first.
# ---------------------------------------------------------------------------

def test_flag_on_runs_verify_before_group():
    """When ``CONCURRENT_GROUPING_AFTER_VLM=1`` the verifier must run on
    the raw single-type activities BEFORE grouping.

    Acceptance shape:
      - ``_verify_one_async`` is called once per raw activity (3 times here).
      - Each call sees a single-type input: no ``_isCombined``, no parallel
        ``objectTypes``, no ``_sourceActivities``.
      - The post-verify grouping step runs AFTER the verifier and then
        merges the survivors into a combined minute-bucket record.
    """
    _set_flag(True)
    assert get_settings().concurrent_grouping_after_vlm is True, (
        "flag must be ON for this test; check env wiring + cache_clear()"
    )

    svc = _fresh_service()
    seen_calls: list[dict] = []

    async def fake_verify(activity, prompt, object_type):
        seen_calls.append({
            "object_type": object_type,
            "is_combined": bool(activity.get("_isCombined")),
            "object_types": list(activity.get("objectTypes", [])),
            "primary_object_type": activity.get("objectType"),
            "id": activity.get("id"),
        })
        return _make_verdict("TRUE_POSITIVE", 0.90)

    svc._verify_one_async = fake_verify  # type: ignore[assignment]

    activities = [
        _raw_activity(1, "writing",    5, 305.0, 315.0, "/tmp/raw_writing_a_clip.mp4"),
        _raw_activity(2, "cell phone", 2, 319.0, 329.0, "/tmp/raw_cell_phone_a_clip.mp4"),
        _raw_activity(3, "writing",    5, 321.0, 325.0, "/tmp/raw_writing_b_clip.mp4"),
    ]

    kept, _stats = _drive_pipeline(activities, svc, flag_on=True)

    # Acceptance #1: the verifier was called once per RAW activity (3x).
    assert len(seen_calls) == 3, (
        f"flag=1: expected 3 verifier calls (one per raw single-type "
        f"activity), saw {len(seen_calls)}: {seen_calls!r}"
    )

    # Acceptance #2: every call saw a single-type shape — NO `_isCombined`,
    # NO parallel `objectTypes`, NO sub-type fanout from a parent.
    for call in seen_calls:
        assert call["is_combined"] is False, (
            f"flag=1: verifier saw a combined record but should only see "
            f"raw single-type activities. call={call!r}"
        )
        assert call["object_types"] == [], (
            f"flag=1: verifier saw parallel objectTypes={call['object_types']!r} "
            f"on what should be a single-type input"
        )

    # Acceptance #3: each raw activity was verified by id — no dedup or
    # parent collapsing happened pre-verify.
    seen_ids = {c["id"] for c in seen_calls}
    assert seen_ids == {"raw-1", "raw-2", "raw-3"}

    # And POST-verify, grouping kicked in: the 3 same-minute survivors
    # collapse to 1 combined minute-bucket record.
    assert len(kept) == 1, (
        "flag=1: 3 same-minute survivors should group into 1 combined "
        f"record; got {len(kept)} records"
    )
    assert kept[0].get("_isCombined") is True


# ---------------------------------------------------------------------------
# Test 3 — Flag ON: the production case that motivated the refactor.
# Writing TP @ t=305 + cell_phone FP @ t=319 → drop the FP, keep writing.
# Post-grouping output must NOT carry the cell_phone in `objectTypes`.
# ---------------------------------------------------------------------------

def test_flag_on_drops_co_merged_fp_keeps_tp():
    """The exact production failure mode from run_20260509_072704: a
    ``writing`` TP and a co-occurring ``cell_phone`` FP in the same minute
    bucket. Under legacy ordering the grouper merged them BEFORE verify, and
    the per-sub-type fanout had to un-merge them at verify time. Under
    flag=1, the verifier sees them as raw single-type inputs and drops the
    cell_phone FP cleanly — so the grouping step that runs after only ever
    sees writing survivors.

    Result contract: post-grouping output contains writing only — no
    ``cell phone`` in any ``objectTypes`` slot, no ``activityType=2`` in
    any ``activityTypes`` slot.
    """
    _set_flag(True)
    assert get_settings().concurrent_grouping_after_vlm is True

    svc = _fresh_service()

    async def fake_verify(activity, prompt, object_type):
        if object_type == "writing":
            return _make_verdict("TRUE_POSITIVE", 0.90)
        if object_type == "cell_phone":
            # The FP we want to drop. Confidence above the 0.80 threshold
            # so the verifier removes it from the kept set.
            return _make_verdict(
                "FALSE_POSITIVE", 0.95,
                object_in_hand="paper_only",
                reasoning="LP holding white papers, not a phone.",
            )
        pytest.fail(f"unexpected object_type {object_type!r}")

    svc._verify_one_async = fake_verify  # type: ignore[assignment]

    activities = [
        _raw_activity(1, "writing",    5, 305.0, 315.0, "/tmp/raw_writing_a_clip.mp4"),
        _raw_activity(2, "cell phone", 2, 319.0, 329.0, "/tmp/raw_cell_phone_a_clip.mp4"),
        _raw_activity(3, "writing",    5, 321.0, 325.0, "/tmp/raw_writing_b_clip.mp4"),
    ]

    kept, stats = _drive_pipeline(activities, svc, flag_on=True)

    # Verifier dropped exactly the cell_phone FP.
    assert stats["dropped"] == 1, (
        f"expected 1 drop (the cell_phone FP), got {stats['dropped']}; "
        f"full stats={stats!r}"
    )

    # Post-grouping: should be either 1 combined writing record (both
    # writing detections in the same minute bucket merged) OR 2 separate
    # writing records — both are acceptable per the spec ("at most one
    # merged writing record (depending on bucket logic)"). Crucially:
    # no cell_phone anywhere.
    assert 1 <= len(kept) <= 2, (
        f"expected 1 or 2 writing records post-grouping, got {len(kept)}"
    )

    for rec in kept:
        # Singular fields must not reference the dropped FP.
        assert rec.get("objectType") != "cell phone"
        assert rec.get("objectType") != "cell_phone"
        assert rec.get("activityType") != 2

        # Parallel arrays (whether _isCombined or _ensure_array_format'd)
        # must not contain cell_phone either.
        obj_types = rec.get("objectTypes", [])
        act_types = rec.get("activityTypes", [])
        assert "cell phone" not in obj_types
        assert "cell_phone" not in obj_types
        assert 2 not in act_types

        # And the writing TP must be present.
        assert rec.get("objectType") == "writing"
        assert 5 in act_types or rec.get("activityType") == 5

    # If grouping merged the two writing records into one combined record,
    # that record must NOT carry `_isCombined` from the cell_phone — its
    # `_sourceActivities` (when present) must reference writing clips only.
    if len(kept) == 1 and kept[0].get("_isCombined"):
        sources = kept[0].get("_sourceActivities", [])
        if sources:
            src_types = {s.get("activityType") for s in sources}
            assert src_types == {5}, (
                f"merged writing record must reference writing sources only; "
                f"got activityTypes={src_types!r}"
            )
            src_clips = kept[0].get("_sourceClips", [])
            assert all("cell_phone" not in c for c in src_clips), (
                f"merged record must not retain a cell_phone clip in "
                f"_sourceClips: {src_clips!r}"
            )


# ---------------------------------------------------------------------------
# Cleanup: reset the flag back to its pre-test default after THIS module
# finishes so we don't poison sibling tests that rely on the legacy default.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _restore_flag_after_module():
    yield
    # Restore whatever the env had at module-import time. If the env var was
    # set externally (CI, dev shell), keep that; otherwise clear it.
    pre_existing = os.environ.get("CONCURRENT_GROUPING_AFTER_VLM")
    if pre_existing is None:
        # _set_flag will have set it to "0"/"1"; pop to restore default.
        os.environ.pop("CONCURRENT_GROUPING_AFTER_VLM", None)
    get_settings.cache_clear()
