"""Task 0007 — Circuit-breaker behaviour for the VLM verifier.

Acceptance criterion #3 from the spec:

    3 consecutive timeouts open the breaker; subsequent calls return
    SKIPPED_VLM_UNAVAILABLE immediately (no HTTP attempt).  After 30s,
    breaker closes.

We do NOT spin up a real httpx server.  Instead we mock the AsyncClient
class itself (specifically the ``post`` coroutine) and observe the
breaker state plus the count of HTTP attempts.
"""
from __future__ import annotations

import os
import time
import asyncio

os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("VLM_VERIFICATION_ENABLED", "1")
os.environ.setdefault("VLM_VERIFY_ACTIVITIES", "writing,eating_drinking,packing_bags")
os.environ.setdefault("VLM_SHADOW_MODE", "1")

import pytest
import httpx


def test_circuit_breaker_opens_after_three_failures_and_closes_after_cooldown():
    """Direct unit test of ``_CircuitBreaker``.

    Faster than wiring the whole service; covers the exact state machine
    the spec calls out.
    """
    from app.services.vlm_verification_service import _CircuitBreaker

    # Use a short cooldown so the test can verify the close-after-cooldown
    # branch without a real 30s sleep.
    cb = _CircuitBreaker(threshold=3, cooldown_s=0.2)

    assert not cb.is_open()
    cb.record_failure()
    assert not cb.is_open(), "1 failure should not open the breaker"
    cb.record_failure()
    assert not cb.is_open(), "2 failures should not open the breaker"
    cb.record_failure()
    assert cb.is_open(), "3 consecutive failures must open the breaker"

    # Subsequent calls while open: still open.
    assert cb.is_open()

    # After cooldown the breaker closes itself on next is_open() call.
    time.sleep(0.25)
    assert not cb.is_open(), "breaker should close after cooldown_s elapsed"

    # A success at any point also closes it.
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    cb.record_success()
    assert not cb.is_open()


def test_breaker_short_circuits_verify_one_no_http_attempt(monkeypatch):
    """When the breaker is open, ``_verify_one_async`` must return
    SKIPPED_VLM_UNAVAILABLE WITHOUT issuing any HTTP request.
    """
    from app.services import vlm_verification_service as vsvc
    from app.services.vlm_verification_service import VlmVerificationService

    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = True
    svc._verify_set = frozenset({"writing", "eating_drinking", "packing_bags"})

    # Force the breaker open.
    for _ in range(3):
        VlmVerificationService._breaker.record_failure()
    assert VlmVerificationService._breaker.is_open()

    # Spy: replace _get_async_client with a tracker that records calls.
    http_calls = {"n": 0}

    class TrackingClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            http_calls["n"] += 1
            raise AssertionError(
                "HTTP must NOT fire while breaker is open"
            )

    monkeypatch.setattr(
        VlmVerificationService, "_get_async_client",
        classmethod(lambda cls, timeout: TrackingClient()),
    )

    # Bypass image prep so we definitely reach the breaker check.
    monkeypatch.setattr(vsvc, "_resolve_keyframes", lambda act: [object()])
    monkeypatch.setattr(
        vsvc, "_stitch_keyframes",
        lambda paths, crop_to_roi=True: b"\xff\xd8\xff\xd9",
    )

    activity = {
        "id": "act-1",
        "objectType": "writing",
        "motionState": "RUNNING",
        "activityType": 5,
    }

    # Call _verify_one_async directly.  We expect a fast SKIPPED return.
    review = asyncio.run(svc._verify_one_async(activity, "prompt", "writing"))

    assert review["status"] == "SKIPPED_VLM_UNAVAILABLE"
    assert review.get("reason") == "circuit_breaker_open"
    assert http_calls["n"] == 0, (
        "open breaker must skip HTTP — observed n=%d" % http_calls["n"]
    )

    VlmVerificationService._breaker.reset()


def test_three_consecutive_timeouts_open_breaker_via_full_pipeline(monkeypatch):
    """End-to-end: wire a fake AsyncClient that always raises ReadTimeout,
    send 3 activities, assert all are SKIPPED and breaker opens by activity
    #3 (or earlier if dispatched in parallel — assert post-conditions).
    """
    from app.services.vlm_verification_service import VlmVerificationService

    svc = VlmVerificationService()
    VlmVerificationService._breaker.reset()
    svc.settings.vlm_verification_enabled = True
    svc.settings.vlm_shadow_mode = True
    svc._verify_set = frozenset({"writing", "eating_drinking", "packing_bags"})

    post_calls = {"n": 0}

    class TimeoutClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            post_calls["n"] += 1
            raise httpx.ReadTimeout("simulated")

    monkeypatch.setattr(
        VlmVerificationService, "_get_async_client",
        classmethod(lambda cls, timeout: TimeoutClient()),
    )

    # Also bypass image prep — return fake bytes so we get to the HTTP
    # path.  Easiest: monkeypatch _resolve_keyframes / _stitch_keyframes
    # at module level.
    from app.services import vlm_verification_service as vsvc
    monkeypatch.setattr(vsvc, "_resolve_keyframes", lambda act: [object()])
    monkeypatch.setattr(vsvc, "_stitch_keyframes", lambda paths, crop_to_roi=True: b"\xff\xd8\xff\xd9")

    activities = [
        {
            "id": f"act-{i}",
            "objectType": "writing",
            "motionState": "RUNNING",
            "activityType": 5,
            "activityImage": f"/tmp/x-{i}.jpg",
        }
        for i in range(5)
    ]

    kept, stats = svc.verify_activities(activities)

    # All 5 kept (fail-open).
    assert len(kept) == 5
    assert stats["skipped_unavailable"] == 5

    # Breaker should have opened.  (With concurrency=4, all 4 in-flight
    # requests fail before the breaker check on the 5th, so by the end
    # the failure count is at least 3 and the breaker is open.)
    assert VlmVerificationService._breaker.is_open(), (
        "after 3+ timeouts the breaker should be OPEN"
    )

    # Each timed-out request retries once, so post_calls is between 5
    # (no retries hit due to early breaker) and 10 (every call retried).
    assert post_calls["n"] >= 3
    assert post_calls["n"] <= 10

    VlmVerificationService._breaker.reset()


def test_breaker_resets_after_cooldown_in_full_pipeline(monkeypatch):
    """Open the breaker, advance fake time past cooldown, and confirm the
    next ``_verify_one_async`` call attempts HTTP again (does not short
    circuit).
    """
    from app.services.vlm_verification_service import (
        VlmVerificationService, _CircuitBreaker,
    )

    # Replace the class-level breaker with one that has a 0.05s cooldown
    # so we don't have to monkeypatch time.monotonic.
    short_breaker = _CircuitBreaker(threshold=3, cooldown_s=0.05)
    monkeypatch.setattr(VlmVerificationService, "_breaker", short_breaker)

    svc = VlmVerificationService()

    for _ in range(3):
        short_breaker.record_failure()
    assert short_breaker.is_open()

    # Wait past cooldown.
    time.sleep(0.1)
    assert not short_breaker.is_open(), "breaker should auto-close after cooldown"

    # Now a fresh _verify_one_async call SHOULD attempt HTTP (not short-circuit).
    http_attempts = {"n": 0}

    class CountingClient:
        is_closed = False

        async def post(self, *args, **kwargs):
            http_attempts["n"] += 1
            raise httpx.ReadTimeout("still down")

    monkeypatch.setattr(
        VlmVerificationService, "_get_async_client",
        classmethod(lambda cls, timeout: CountingClient()),
    )
    from app.services import vlm_verification_service as vsvc
    monkeypatch.setattr(vsvc, "_resolve_keyframes", lambda act: [object()])
    monkeypatch.setattr(vsvc, "_stitch_keyframes", lambda paths, crop_to_roi=True: b"\xff\xd8\xff\xd9")

    activity = {
        "id": "act-after-cooldown",
        "objectType": "writing",
        "motionState": "RUNNING",
        "activityType": 5,
    }
    review = asyncio.run(svc._verify_one_async(activity, "prompt", "writing"))

    assert review["status"] == "SKIPPED_VLM_UNAVAILABLE"
    # Should have attempted HTTP at least once (1 initial + 1 retry).
    assert http_attempts["n"] >= 1
