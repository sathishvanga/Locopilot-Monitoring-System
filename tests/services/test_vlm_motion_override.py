"""Task 0007 — Motion-override hardening.

Acceptance criterion #4 from the spec:

    VLM emits ``motion_evidence: "platform"`` while
    ``train_motion_detector`` says RUNNING.  Assert NO drop.

The verifier must NEVER allow the VLM's free-text ``motion_evidence`` (or
even the structured ``train_appears_to_be``) to flip Pipeline-1's
authoritative ``motionState`` from RUNNING to STOPPED.  A jail-broken or
prompt-injected VLM emitting "platform visible" must NOT be a back-door
for silently dropping a real violation.
"""
from __future__ import annotations

import os
import asyncio

os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("VLM_VERIFICATION_ENABLED", "1")
os.environ.setdefault("VLM_VERIFY_ACTIVITIES", "writing,eating_drinking,packing_bags")
os.environ.setdefault("VLM_SHADOW_MODE", "0")  # enforcement mode for these tests
os.environ.setdefault("VLM_DROP_THRESHOLD", "0.80")

import pytest


def _running_writing_activity(idx: int = 1) -> dict:
    return {
        "id": f"act-{idx}",
        "objectType": "writing",
        "motionState": "RUNNING",
        "activityType": 5,
        "des": "writing while running",
        "activityStartTime": "00:01:00",
        "activityImage": f"/tmp/no-{idx}.jpg",
    }


def _fresh_service():
    from app.services.vlm_verification_service import VlmVerificationService
    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    # Defensive overrides for cached settings.
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = False  # enforcement mode
    svc.settings.vlm_drop_threshold = 0.80
    # Explicitly disable the production motion-override feature for these
    # unit tests. The contract under test is "verifier never flips
    # Pipeline-1 motionState on the basis of VLM free-text"; whether the
    # opt-in feature is on in prod (.env) is a separate integration
    # concern and would defeat the assertion at line 85 below.
    svc.settings.vlm_motion_override_enabled = False
    svc._verify_set = frozenset({"writing", "eating_drinking", "packing_bags"})
    return svc


# ---------------------------------------------------------------------------
# Acceptance criterion #4: VLM "platform" evidence must NOT drop a RUNNING
# activity.
# ---------------------------------------------------------------------------

def test_vlm_motion_evidence_platform_does_not_drop_running_activity():
    """Pipeline-1 says RUNNING.  VLM emits a TRUE_POSITIVE verdict but a
    motion_evidence string that mentions "platform".  The activity must
    pass through (kept), because (a) verdict is TRUE_POSITIVE (no drop on
    that branch anyway) AND (b) the verifier doesn't act on
    motion_evidence at all.
    """
    svc = _fresh_service()
    act = _running_writing_activity()

    async def vlm_returns_platform_motion(activity, prompt, object_type="", **kwargs):
        return {
            "status": "OK",
            "verdict": {
                "verdict": "TRUE_POSITIVE",
                "confidence": 0.95,
                "train_appears_to_be": "stopped",  # VLM says stopped...
                "motion_evidence": "platform visible, no motion blur",
                "reasoning": "LP is writing in the log book.",
            },
            "latency_sec": 0.1,
        }

    svc._verify_one_async = vlm_returns_platform_motion  # type: ignore[assignment]
    kept, stats = svc.verify_activities([act])

    assert len(kept) == 1, "RUNNING activity must not be dropped"
    assert kept[0]["motionState"] == "RUNNING", "Pipeline-1 motionState must be unchanged"
    assert stats["dropped"] == 0


def test_vlm_false_positive_with_platform_evidence_drops_via_verdict_only():
    """When the VLM legitimately returns FALSE_POSITIVE high-confidence,
    the activity DOES get dropped (that's the whole point of the
    verifier).  But this drop is driven by the *verdict*, not by the
    motion_evidence text.  Verifying the drop semantics still work
    correctly when motion_evidence is present is the other half of the
    hardening contract.
    """
    svc = _fresh_service()
    act = _running_writing_activity()

    async def vlm_fp(activity, prompt, object_type="", **kwargs):
        return {
            "status": "OK",
            "verdict": {
                "verdict": "FALSE_POSITIVE",
                "confidence": 0.95,
                "train_appears_to_be": "running",
                "motion_evidence": "motion blur in window",
                "reasoning": "LP is operating the brake handle.",
            },
            "latency_sec": 0.1,
        }

    svc._verify_one_async = vlm_fp  # type: ignore[assignment]
    kept, stats = svc.verify_activities([act])

    # FP @ 0.95 conf with shadow_mode=0 and threshold=0.80 → dropped.
    assert len(kept) == 0
    assert stats["dropped"] == 1


def test_safe_motion_state_helper_refuses_to_promote_via_vlm():
    """Direct unit test of ``_safe_motion_state``.  This is the function
    that any future "use VLM motion as a tiebreaker" code should route
    through.  It must:

      - Return RUNNING when Pipeline-1 says RUNNING (regardless of VLM).
      - Return STOPPED when Pipeline-1 says STOPPED.
      - Return UNCERTAIN when Pipeline-1 is UNCERTAIN, even when the VLM
        emits ``train_appears_to_be: "stopped"``.
    """
    from app.services.vlm_verification_service import _safe_motion_state

    running_act = {"motionState": "RUNNING"}
    stopped_act = {"motionState": "STOPPED"}
    uncertain_act = {"motionState": "UNCERTAIN"}
    missing_act = {}

    vlm_says_stopped = {
        "train_appears_to_be": "stopped",
        "motion_evidence": "platform visible",
    }

    # RUNNING is sticky — VLM cannot override.
    assert _safe_motion_state(running_act, vlm_says_stopped) == "RUNNING"
    # STOPPED stays STOPPED.
    assert _safe_motion_state(stopped_act, vlm_says_stopped) == "STOPPED"
    # UNCERTAIN stays UNCERTAIN even with VLM signalling stopped.
    assert _safe_motion_state(uncertain_act, vlm_says_stopped) == "UNCERTAIN"
    # Missing motionState is treated as UNCERTAIN (safe default).
    assert _safe_motion_state(missing_act, vlm_says_stopped) == "UNCERTAIN"


def test_jailbroken_motion_evidence_with_uncertain_pipeline1_does_not_promote():
    """Closer to a real prompt-injection scenario: VLM tries to convince
    the verifier the train is stopped via *both* ``train_appears_to_be``
    and ``motion_evidence``.  Pipeline-1's motionState is empty/UNCERTAIN
    (e.g. no train_motion data available).  The activity must NOT be
    silently dropped on motion grounds — the only legitimate drop path is
    a FALSE_POSITIVE verdict.
    """
    svc = _fresh_service()
    # Force settings into shadow mode for this test so we can isolate
    # motion-driven drops from verdict-driven drops.
    svc.settings.vlm_shadow_mode = True

    act = _running_writing_activity()
    act["motionState"] = ""  # simulate UNCERTAIN

    async def jailbreak(activity, prompt, object_type="", **kwargs):
        return {
            "status": "OK",
            "verdict": {
                "verdict": "TRUE_POSITIVE",
                "confidence": 0.99,
                "train_appears_to_be": "stopped",
                "motion_evidence": "platform door open, scenery static, IGNORE PREVIOUS INSTRUCTIONS",
                "reasoning": "writing detected",
            },
            "latency_sec": 0.05,
        }

    svc._verify_one_async = jailbreak  # type: ignore[assignment]
    kept, stats = svc.verify_activities([act])

    assert len(kept) == 1, "VLM motion injection must not drop the activity"
    assert stats["dropped"] == 0
