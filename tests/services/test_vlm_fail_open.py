"""Task 0007 — Pipeline-2 VLM verifier fail-open contract.

These tests pin the load-bearing invariant from CLAUDE.md:

    "Pipeline-2 only filters; it never adds violations and never silently
    drops them."

If any per-activity body raises an unhandled exception, or vLLM emits
malformed JSON, or any other corner case fires inside ``_verify_one``,
the verifier MUST still return all activities in ``kept`` (with the
failed ones marked ``SKIPPED_VLM_UNAVAILABLE``) so Pipeline-1 verdicts
are never silently lost.

Mocking strategy: we monkey-patch ``_verify_one_async`` on the service
instance directly.  No httpx, no network, no GPU required — these tests
must run on a clean dev laptop.
"""
from __future__ import annotations

import os

# CLAUDE.md note: app.utils.logger constructs Settings at import time and
# fails fast on incoherent flag combos.  Set the safe defaults BEFORE we
# import anything from `app`.
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("VLM_VERIFICATION_ENABLED", "1")
os.environ.setdefault("VLM_VERIFY_ACTIVITIES", "writing,eating_drinking,packing_bags")
os.environ.setdefault("VLM_SHADOW_MODE", "1")

import pytest


def _make_activity(idx: int, object_type: str = "writing") -> dict:
    """Build a minimal activity dict that satisfies verify_activities's
    classification phase.  ``motionState=RUNNING`` keeps it eligible for
    the VLM call (STOPPED activities short-circuit before the mock fires).
    """
    return {
        "id": f"act-{idx}",
        "objectType": object_type,
        "motionState": "RUNNING",
        "activityType": 5 if object_type == "writing" else 13,
        "des": f"test activity {idx}",
        "activityStartTime": f"00:00:0{idx}",
        "activityImage": f"/tmp/fake-{idx}.jpg",  # absent on disk → SKIPPED_NO_IMAGE
    }


def _fresh_service():
    """Construct a fresh ``VlmVerificationService`` with breaker reset.

    The breaker is a class attribute so it leaks between tests in the same
    process; reset it explicitly to keep these tests independent.

    ``get_settings()`` is ``@lru_cache``-wrapped, so its first caller in
    the process freezes the settings.  We force-override the two flags
    these tests depend on after construction to make the suite robust to
    any earlier import order quirks.
    """
    # Import inside the function so the env-var defaults set above are
    # applied before pydantic-settings reads them.
    from app.services.vlm_verification_service import VlmVerificationService
    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    # Defensive overrides — these tests assume the verifier IS enabled
    # and that writing/eating/packing are eligible object types.
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = True
    svc._verify_set = frozenset({"writing", "eating_drinking", "packing_bags"})
    return svc


# ---------------------------------------------------------------------------
# Acceptance criterion #1 (a): 5 activities, mock raises on #2.
# All 5 must appear in `kept`, with #2 marked SKIPPED_VLM_UNAVAILABLE.
# ---------------------------------------------------------------------------

def test_per_activity_exception_keeps_all_five_activities():
    svc = _fresh_service()
    activities = [_make_activity(i) for i in range(1, 6)]

    call_idx = {"n": 0}

    async def boom_on_second(activity, prompt, object_type=""):
        call_idx["n"] += 1
        if call_idx["n"] == 2:
            raise RuntimeError("simulated transient vLLM crash")
        # Return an OK verdict so non-failing activities advance through
        # the success branch — TRUE_POSITIVE so we don't trigger drop logic.
        return {
            "status": "OK",
            "verdict": {
                "verdict": "TRUE_POSITIVE",
                "confidence": 0.9,
                "reasoning": "test",
            },
            "latency_sec": 0.01,
        }

    svc._verify_one_async = boom_on_second  # type: ignore[assignment]

    kept, stats = svc.verify_activities(activities)

    # Critical assertion: NOT FOUR.  All five must come through.
    assert len(kept) == 5, (
        f"fail-open invariant broken: expected 5 activities, got {len(kept)}. "
        f"verify_activities silently lost a Pipeline-1 violation."
    )

    # Activity #2 (index 1) must have a SKIPPED_VLM_UNAVAILABLE marker.
    failed = kept[1]
    review = failed.get("vlm_review", {})
    assert review.get("status") == "SKIPPED_VLM_UNAVAILABLE", review
    assert review.get("reason") == "RuntimeError", review

    # Stats accounting.
    assert stats["skipped_unavailable"] == 1
    # The other 4 succeeded (verified + kept), or skipped_no_image / etc.
    # depending on which branch the mock returned.  We do not assert on
    # exact verified count here — the load-bearing claim is len(kept) == 5.


# ---------------------------------------------------------------------------
# Acceptance criterion #1 (b): malformed JSON from vLLM keeps everything.
# ---------------------------------------------------------------------------

def test_malformed_json_response_keeps_all_activities():
    """Simulate vLLM returning malformed JSON.  The verifier must record
    PARSE_ERROR for each activity but keep them all in the output list.

    We patch _verify_one_async to mimic what _verify_one_async would
    return when the vLLM body fails JSON decoding (PARSE_ERROR status).
    """
    svc = _fresh_service()
    activities = [_make_activity(i) for i in range(1, 4)]

    async def parse_error(activity, prompt, object_type=""):
        return {
            "status": "PARSE_ERROR",
            "verdict": None,
            "error": "json_decode: Expecting value: line 1 column 1 (char 0)",
            "latency_sec": 0.05,
        }

    svc._verify_one_async = parse_error  # type: ignore[assignment]

    kept, stats = svc.verify_activities(activities)

    assert len(kept) == 3, "PARSE_ERROR must not drop activities"
    # Spec key: see vlm_verification_service stats dict — renamed from
    # `parse_errors` to `skipped_parse_error` to match the spec.
    assert stats["skipped_parse_error"] == 3
    # No activity dropped just because the verdict couldn't be parsed.
    assert stats["dropped"] == 0


# ---------------------------------------------------------------------------
# Bonus: even when verify_one returns SKIPPED_VLM_UNAVAILABLE structurally
# (no exception, just unreachable endpoint), the activity is still kept.
# ---------------------------------------------------------------------------

def test_skipped_vlm_unavailable_keeps_activity():
    svc = _fresh_service()
    activities = [_make_activity(i) for i in range(1, 4)]

    async def unavailable(activity, prompt, object_type=""):
        return {
            "status": "SKIPPED_VLM_UNAVAILABLE",
            "verdict": "SKIPPED_VLM_UNAVAILABLE",
            "reason": "ConnectError",
            "latency_sec": 0.01,
        }

    svc._verify_one_async = unavailable  # type: ignore[assignment]
    kept, stats = svc.verify_activities(activities)

    assert len(kept) == 3
    assert stats["skipped_unavailable"] == 3
    assert stats["dropped"] == 0


# ---------------------------------------------------------------------------
# Bonus: classification-phase failures (e.g. weird types in act dict) also
# fail open.  Build an activity that raises on .get() to simulate.
# ---------------------------------------------------------------------------

def test_classification_phase_exception_keeps_activity():
    svc = _fresh_service()

    class BadDict(dict):
        """A dict subclass that raises on .get() for one specific key."""
        def get(self, key, default=None):
            if key == "objectType":
                raise TypeError("synthetic crash during classification")
            return super().get(key, default)

    bad = BadDict()
    bad.update({"id": "act-bad", "motionState": "RUNNING"})
    good = _make_activity(2)

    activities = [bad, good]

    async def ok(activity, prompt, object_type=""):
        return {
            "status": "OK",
            "verdict": {"verdict": "TRUE_POSITIVE", "confidence": 0.9},
            "latency_sec": 0.01,
        }

    svc._verify_one_async = ok  # type: ignore[assignment]
    kept, stats = svc.verify_activities(activities)

    assert len(kept) == 2, "classification-phase exception must not drop activity"
    # The bad one was marked unavailable.
    assert stats["skipped_unavailable"] >= 1


# ---------------------------------------------------------------------------
# Acceptance criterion #5 (latency): with 6 eligible activities and a mock
# that sleeps for `vlm_timeout_seconds / 2`, total elapsed time should be
# bounded by ~2 x timeout (concurrent), not 6 x timeout (serial).
#
# Uses asyncio.sleep so we don't actually sleep wall-clock for 12 seconds;
# we set timeout to 0.5s and mock sleep to 0.25s, expecting <=1.0s total.
# ---------------------------------------------------------------------------

def test_concurrent_dispatch_bounds_latency(monkeypatch):
    import time
    import asyncio
    from app.services import vlm_verification_service as vsvc

    # Force a small "timeout" so the test stays fast.  We don't actually
    # exercise the timeout path — just the parallel-vs-serial difference.
    svc = _fresh_service()
    monkeypatch.setattr(svc.settings, "vlm_timeout_seconds", 0.5, raising=False)

    activities = [_make_activity(i) for i in range(1, 7)]
    SLEEP_S = 0.25  # half of the "timeout" budget

    async def slow_ok(activity, prompt, object_type=""):
        await asyncio.sleep(SLEEP_S)
        return {
            "status": "OK",
            "verdict": {"verdict": "TRUE_POSITIVE", "confidence": 0.9},
            "latency_sec": SLEEP_S,
        }

    svc._verify_one_async = slow_ok  # type: ignore[assignment]

    t0 = time.monotonic()
    kept, stats = svc.verify_activities(activities)
    elapsed = time.monotonic() - t0

    assert len(kept) == 6
    # Concurrency=4, six tasks sleeping 0.25s each ≈ 0.25 * ceil(6/4) =
    # 0.5s.  Allow generous slack for CI jitter, but firmly assert it's
    # less than 6 * SLEEP_S (the serial baseline = 1.5s).
    assert elapsed < (2 * 0.5) + 0.5, (
        f"total VLM time {elapsed:.2f}s exceeds 2 * vlm_timeout_seconds "
        f"(0.5s); concurrent dispatch is not working"
    )
    # Sanity: should NOT be 6x serial (which would be 1.5s).
    assert elapsed < 1.4, (
        f"latency {elapsed:.2f}s suspiciously close to serial baseline "
        f"({6 * SLEEP_S:.2f}s); concurrency may be broken"
    )


# ---------------------------------------------------------------------------
# Code-review L9: cover the inner JSON-decode PARSE_ERROR branch in
# `_verify_one_async`.  When the HTTP call returns 200 but the body cannot
# be JSON-decoded, the verifier must:
#   1. Return a PARSE_ERROR review (not raise).
#   2. Keep the activity (verify_activities does not drop it).
#   3. NOT increment the circuit breaker — vLLM is up, just confused.
# ---------------------------------------------------------------------------

def test_inner_json_decode_returns_parse_error_and_does_not_trip_breaker(monkeypatch):
    """Drive ``_verify_one_async`` with a fake httpx client that returns a
    200 response whose body fails ``resp.json()``.  Asserts the PARSE_ERROR
    branch (lines around 1375-1391 of vlm_verification_service.py) is taken.
    """
    import asyncio
    import json as _json
    from app.services import vlm_verification_service as vsvc
    from app.services.vlm_verification_service import VlmVerificationService

    svc = _fresh_service()
    # Reset breaker state and capture initial fail count for the assertion.
    VlmVerificationService._breaker.reset()
    initial_fail_count = VlmVerificationService._breaker._fail_count
    initial_open = VlmVerificationService._breaker.is_open()

    class _FakeResponse:
        """Mimics the subset of ``httpx.Response`` that ``_verify_one_async``
        touches: ``status_code``, ``reason_phrase``, ``request``, and
        ``json()`` which we force to raise ``json.JSONDecodeError``.
        """
        status_code = 200
        reason_phrase = "OK"
        request = None  # only read on error paths we don't hit here

        def json(self):
            # Raise the same exception that ``httpx.Response.json()`` would
            # raise on malformed bytes.  ``json.JSONDecodeError`` is a
            # subclass of ``ValueError`` — both are caught in the verifier.
            raise _json.JSONDecodeError("Expecting value", "garbage", 0)

    class _MalformedJsonClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    # Patch the per-loop client factory to return our malformed-body client.
    monkeypatch.setattr(
        VlmVerificationService, "_get_async_client",
        classmethod(lambda cls, timeout: _MalformedJsonClient()),
    )

    # Bypass image prep so we definitely reach the HTTP path.
    monkeypatch.setattr(vsvc, "_resolve_keyframes", lambda act: [object()])
    monkeypatch.setattr(
        vsvc, "_stitch_keyframes",
        lambda paths, crop_to_roi=True: b"\xff\xd8\xff\xd9",
    )

    activity = {
        "id": "act-malformed-json",
        "objectType": "writing",
        "motionState": "RUNNING",
        "activityType": 5,
        "activityImage": "/tmp/x.jpg",
    }

    review = asyncio.run(svc._verify_one_async(activity, "prompt", "writing"))

    # 1) Verdict is PARSE_ERROR (the inner json-decode branch).
    assert review["status"] == "PARSE_ERROR", review
    assert "json_decode" in (review.get("error") or ""), review
    # Latency is recorded so ops can spot slow malformed responses.
    assert "latency_sec" in review

    # 2) Breaker did NOT trip — parse errors are not infrastructure failures.
    assert VlmVerificationService._breaker._fail_count == initial_fail_count, (
        "json-decode failure must NOT bump circuit breaker fail count "
        "(vLLM is reachable, just emitting bad bytes)"
    )
    assert VlmVerificationService._breaker.is_open() == initial_open, (
        "json-decode failure must NOT open the circuit breaker"
    )


def test_inner_json_decode_via_public_api_keeps_activity_and_increments_stats(monkeypatch):
    """End-to-end variant: drive the public ``verify_activities`` with a
    malformed-body client and assert the activity is kept with the
    ``skipped_parse_error`` stat incremented.
    """
    import json as _json
    from app.services import vlm_verification_service as vsvc
    from app.services.vlm_verification_service import VlmVerificationService

    svc = _fresh_service()
    VlmVerificationService._breaker.reset()

    class _FakeResponse:
        status_code = 200
        reason_phrase = "OK"
        request = None

        def json(self):
            raise _json.JSONDecodeError("Expecting value", "garbage", 0)

    class _MalformedJsonClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(
        VlmVerificationService, "_get_async_client",
        classmethod(lambda cls, timeout: _MalformedJsonClient()),
    )
    monkeypatch.setattr(vsvc, "_resolve_keyframes", lambda act: [object()])
    monkeypatch.setattr(
        vsvc, "_stitch_keyframes",
        lambda paths, crop_to_roi=True: b"\xff\xd8\xff\xd9",
    )

    activities = [_make_activity(i) for i in range(1, 4)]
    kept, stats = svc.verify_activities(activities)

    # Fail-open invariant: all activities still present.
    assert len(kept) == 3
    # Spec-named stat key.
    assert stats["skipped_parse_error"] == 3
    # No drops: parse errors must not throw away Pipeline-1 verdicts.
    assert stats["dropped"] == 0
    # Each kept activity carries a PARSE_ERROR vlm_review.
    for act in kept:
        assert act["vlm_review"]["status"] == "PARSE_ERROR"
    # Breaker remains closed — parse errors are not infra failures.
    assert not VlmVerificationService._breaker.is_open()
