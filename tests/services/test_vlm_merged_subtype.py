"""Per-sub-type VLM verification of concurrent-merged activities.

Pipeline-1 occasionally merges co-occurring detections (e.g. ``writing`` +
``cell_phone`` in the same minute bucket) into one posted violation under a
single primary ``objectType``. The verifier historically only checked the
primary type, so a real ``writing`` event would protect a co-merged
``cell_phone`` FP from being filtered.

These tests pin the new contract:

  - When a merged activity has multiple distinct verifiable sub-types,
    the verifier dispatches one VLM call PER sub-type.
  - If ANY sub-type returns FALSE_POSITIVE @ confidence ≥ drop_threshold:
      * If at least one sibling sub-type survives, the FP sub-type is
        STRIPPED from the parent (objectTypes / activityTypes /
        descriptions / _sourceActivities / _sourceClips all updated).
      * If ALL sub-types come back FP, the parent is dropped entirely
        (existing single-result drop path).
  - When all sub-types come back TP, the activity passes through
    unchanged with a per-sub-type audit blob attached to ``vlm_review``.

Mocking: we monkey-patch ``_verify_one_async`` on the service instance.
Each call inspects ``activity['objectType']`` to know which sub-view it's
serving and returns the verdict the test wants.
"""
from __future__ import annotations

import os

os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("VLM_VERIFICATION_ENABLED", "1")
os.environ.setdefault("VLM_VERIFY_ACTIVITIES", "writing,cell_phone,eating_drinking")
os.environ.setdefault("VLM_SHADOW_MODE", "0")
os.environ.setdefault("VLM_DROP_THRESHOLD", "0.80")

import pytest


def _merged_writing_phone_activity() -> dict:
    """A minimal merged-activity dict matching what
    ``concurrent_activity_grouping_service`` emits for a co-occurring
    writing + cell_phone burst (the production case that motivated this
    fix — see run_20260509_072704)."""
    return {
        "id": "act-merged-1",
        # Singular fields point at the dominant primary sub-type.
        "objectType": "writing",
        "activityType": 5,
        "des": "WRITING LOG BOOK WHILE RUNNING; Using mobile phone",
        "motionState": "RUNNING",
        "activityStartTime": "305.00",
        "activityEndTime": "343.00",
        # Merged-marker + parallel arrays. Sub-type i's metadata is at
        # index i across all three arrays.
        "_isCombined": True,
        "objectTypes": ["writing", "cell phone"],
        "activityTypes": [5, 2],
        "descriptions": ["WRITING LOG BOOK WHILE RUNNING", "Using mobile phone"],
        "_sourceActivities": [
            {"activityType": 5, "clip": "/tmp/fake_writing_a_clip.mp4",
             "startTime": "305.00", "endTime": "315.00"},
            {"activityType": 2, "clip": "/tmp/fake_cell_phone_a_clip.mp4",
             "startTime": "319.00", "endTime": "329.00"},
            {"activityType": 5, "clip": "/tmp/fake_writing_b_clip.mp4",
             "startTime": "321.00", "endTime": "325.00"},
            {"activityType": 2, "clip": "/tmp/fake_cell_phone_b_clip.mp4",
             "startTime": "333.00", "endTime": "339.00"},
        ],
        "_sourceClips": [
            "/tmp/fake_writing_a_clip.mp4",
            "/tmp/fake_cell_phone_a_clip.mp4",
            "/tmp/fake_writing_b_clip.mp4",
            "/tmp/fake_cell_phone_b_clip.mp4",
        ],
        # `_resolve_keyframes` falls back to this when no `_sourceActivities`
        # entry has a JPG on disk — sufficient for tests that mock the
        # verifier and never actually decode a frame.
        "activityImage": "/tmp/no-such-file.jpg",
    }


def _fresh_service():
    from app.services.vlm_verification_service import VlmVerificationService
    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = False
    svc.settings.vlm_drop_threshold = 0.80
    svc._verify_set = frozenset({"writing", "cell_phone", "eating_drinking"})
    return svc


def _make_verdict(verdict: str, conf: float, **extra) -> dict:
    """Build the (review, verdict_dict) shape `_verify_one_async` returns
    on the OK path."""
    v = {"verdict": verdict, "confidence": conf, "reasoning": f"test {verdict}"}
    v.update(extra)
    return {
        "status": "OK",
        "verdict": v,
        "latency_sec": 0.1,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "frames_sent": 1,
    }


# ---------------------------------------------------------------------------
# Case A: writing TP, cell_phone FP @ 0.95 → strip phone, keep writing.
# This is the exact production case from run_20260509_072704.
# ---------------------------------------------------------------------------

def test_partial_drop_strips_fp_subtype_keeps_parent():
    svc = _fresh_service()

    async def fake_verify(activity, prompt, object_type):
        if object_type == "writing":
            return _make_verdict("TRUE_POSITIVE", 0.85)
        if object_type == "cell_phone":
            # The FP we want to strip: VLM is confident there's no phone.
            return _make_verdict(
                "FALSE_POSITIVE", 0.95,
                object_in_hand="paper_only",
                reasoning="LP holding white papers, not a smartphone.",
            )
        pytest.fail(f"unexpected object_type {object_type!r}")

    svc._verify_one_async = fake_verify  # type: ignore[assignment]
    act = _merged_writing_phone_activity()
    kept, stats = svc.verify_activities([act])

    # Parent activity must survive (writing was TP).
    assert len(kept) == 1, "writing TP must survive — partial-strip path"
    survivor = kept[0]

    # The FP sub-type must be stripped from the parallel arrays.
    assert "cell phone" not in survivor["objectTypes"]
    assert 2 not in survivor["activityTypes"]
    assert "Using mobile phone" not in survivor["descriptions"]

    # Source clips/activities must be filtered to writing only.
    sub_types_left = {s["activityType"] for s in survivor["_sourceActivities"]}
    assert sub_types_left == {5}
    assert all("cell_phone" not in c for c in survivor["_sourceClips"])

    # Singular fields must reflect the dominant survivor.
    assert survivor["objectType"] == "writing"
    assert survivor["activityType"] == 5
    assert "Using mobile phone" not in survivor["des"]

    # Audit blob: per-sub-type reviews retained for ops review.
    review = survivor["vlm_review"]
    assert review["status"] == "OK"
    assert "cell_phone" in review["subtype_reviews"]
    assert "writing" in review["subtype_reviews"]
    assert "cell_phone" in review["subtypes_dropped"]
    assert "writing" in review["subtypes_kept"]


# ---------------------------------------------------------------------------
# Case B: BOTH sub-types come back FP → drop the whole merged activity.
# ---------------------------------------------------------------------------

def test_all_subtypes_fp_drops_merged_activity():
    svc = _fresh_service()

    async def fake_verify(activity, prompt, object_type):
        return _make_verdict("FALSE_POSITIVE", 0.92)

    svc._verify_one_async = fake_verify  # type: ignore[assignment]
    act = _merged_writing_phone_activity()
    kept, stats = svc.verify_activities([act])

    # Whole merged activity must be filtered out.
    assert len(kept) == 0, "all-sub-types-FP must drop the parent"
    assert stats["dropped"] == 1


# ---------------------------------------------------------------------------
# Case C: BOTH sub-types come back TP → activity passes through unchanged,
# with a per-sub-type audit blob attached.
# ---------------------------------------------------------------------------

def test_all_subtypes_tp_keeps_activity_unchanged():
    svc = _fresh_service()

    async def fake_verify(activity, prompt, object_type):
        return _make_verdict("TRUE_POSITIVE", 0.90)

    svc._verify_one_async = fake_verify  # type: ignore[assignment]
    act = _merged_writing_phone_activity()
    kept, stats = svc.verify_activities([act])

    assert len(kept) == 1
    survivor = kept[0]

    # Both sub-types must survive intact.
    assert "writing" in survivor["objectTypes"]
    assert "cell phone" in survivor["objectTypes"]
    assert set(survivor["activityTypes"]) == {5, 2}
    assert len(survivor["_sourceActivities"]) == 4

    # Audit blob still records per-sub-type verdicts.
    review = survivor["vlm_review"]
    assert review["status"] == "OK"
    assert review["subtype_reviews"]["writing"]["verdict"]["verdict"] == "TRUE_POSITIVE"
    assert review["subtype_reviews"]["cell_phone"]["verdict"]["verdict"] == "TRUE_POSITIVE"
    assert review["subtypes_dropped"] == []


# ---------------------------------------------------------------------------
# Case D: cell_phone FP but at confidence < drop_threshold → no strip.
# Confidence-gating must be enforced for sub-types just like the primary path.
# ---------------------------------------------------------------------------

def test_subtype_fp_below_threshold_does_not_strip():
    svc = _fresh_service()
    svc.settings.vlm_drop_threshold = 0.80

    async def fake_verify(activity, prompt, object_type):
        if object_type == "writing":
            return _make_verdict("TRUE_POSITIVE", 0.85)
        if object_type == "cell_phone":
            # Borderline FP — verifier must NOT strip it because conf<threshold.
            return _make_verdict("FALSE_POSITIVE", 0.50)
        pytest.fail(f"unexpected object_type {object_type!r}")

    svc._verify_one_async = fake_verify  # type: ignore[assignment]
    act = _merged_writing_phone_activity()
    kept, stats = svc.verify_activities([act])

    assert len(kept) == 1
    survivor = kept[0]
    # Both sub-types remain because the FP didn't clear the threshold.
    assert "cell phone" in survivor["objectTypes"]
    assert 2 in survivor["activityTypes"]


# ---------------------------------------------------------------------------
# Case E: shadow mode → record per-sub-type verdicts but never strip / drop.
# Pipeline-1's verdict must remain authoritative under shadow rollout.
# ---------------------------------------------------------------------------

def test_shadow_mode_never_strips_or_drops():
    svc = _fresh_service()
    svc.settings.vlm_shadow_mode = True

    async def fake_verify(activity, prompt, object_type):
        # Even with cell_phone FP at high conf, shadow mode must not strip.
        if object_type == "cell_phone":
            return _make_verdict("FALSE_POSITIVE", 0.95)
        return _make_verdict("TRUE_POSITIVE", 0.90)

    svc._verify_one_async = fake_verify  # type: ignore[assignment]
    act = _merged_writing_phone_activity()
    kept, stats = svc.verify_activities([act])

    assert len(kept) == 1
    survivor = kept[0]
    # Sub-types remain intact in shadow mode.
    assert "cell phone" in survivor["objectTypes"]
    assert 2 in survivor["activityTypes"]
    # But the per-sub-type audit is still recorded.
    assert survivor["vlm_review"]["subtype_reviews"]["cell_phone"]["verdict"]["verdict"] == "FALSE_POSITIVE"
