"""Tests for external_api_service idempotency, DLQ writes, and dedup.

Maps to acceptance criteria from
``docs/specs/code-review-fixes/tasks/0004-external-api-dlq-and-idempotency.md``:

    1. 3 x 503 then 200 — same Idempotency-Key on every attempt (4 calls total
       under ``MAX_RETRIES = 4``).
    2. 4 x 503 — DLQ file written under ``<run_dir>/_failed_external_api/``,
       no Authorization captured in record.
    3. _deduplicate_violations collapses "6.00" and "00:00:06" into one entry.

Acceptance criterion (4) lives in ``test_dlq.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# Make the repo root importable from any pytest cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Skip pose-model file existence checks so Settings construction works in CI.
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Tiny stand-in for ``requests.Response`` covering only what we touch."""

    def __init__(self, status_code: int, body: str = '{"ok": true}'):
        self.status_code = status_code
        self.text = body

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}


class _SequencedPoster:
    """Records every requests.post call and returns a scripted sequence."""

    def __init__(self, responses: List[_FakeResponse]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url, json=None, headers=None, timeout=None, **kw):
        # Record a defensive copy of the headers — the SUT mutates a private
        # copy but we still want to assert per-call headers, not the final state.
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": dict(headers) if headers else {},
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("Unexpected extra POST")
        return self._responses.pop(0)


@pytest.fixture
def fast_retry(monkeypatch):
    """Drop the inter-attempt sleep so the test suite stays under a second."""
    monkeypatch.setattr("app.services.external_api_service.time.sleep", lambda *_: None)


@pytest.fixture
def run_dir(tmp_path) -> str:
    """Create a per-test run directory matching the production layout.

    Spec 0004 places DLQ files under ``<run_dir>/_failed_external_api/``,
    so the tests need a real per-run directory to pass through to
    ``post_cvvr_results`` instead of the prior global drop dir.
    """
    target = tmp_path / "run_20260508_120000"
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


@pytest.fixture
def service(monkeypatch):
    """Build a fresh ExternalAPIService bound to a deterministic URL + token.

    Builds an isolated ``Settings`` instance per test (``_env_file=None`` skips
    .env auto-load) so test mutations cannot leak through the cached
    ``get_settings()`` singleton into sibling tests.
    """
    from app.services.external_api_service import ExternalAPIService
    from app.utils.config import Settings

    isolated = Settings(_env_file=None)
    isolated.cvvr_api_enabled = True
    isolated.cvvr_api_url = (
        "https://api.example.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulk"
    )
    isolated.cvvr_api_url_no_events = (
        "https://api.example.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"
    )
    isolated.cvvr_api_default_division = "demo"
    isolated.cvvr_api_token = "secret-bearer-token"
    isolated.cvvr_api_timeout = 5

    svc = ExternalAPIService()
    svc.settings = isolated
    return svc


def _sample_event() -> Dict[str, Any]:
    return {
        "activityType": 5,
        "des": "Writing logbook",
        "objectType": "logbook",
        "activityStartTime": "00:00:06",
        "activityEndTime": "00:00:12",
        "videoStartTime": "00:00:06",
        "videoEndTime": "00:00:12",
        "filename": "trip.mp4",
        "fileDuration": "00:30:00",
        "crewName": "LP1",
        "activityClip": "https://s3.example.com/clip.mp4",
        "crewRole": 1,
        "motionState": "RUNNING",
    }


# ---------------------------------------------------------------------------
# Acceptance 1 — same Idempotency-Key across retries
# ---------------------------------------------------------------------------


def test_idempotency_key_constant_across_retries(service, fast_retry, run_dir):
    """3 x 503 then 200: every attempt carries the *same* Idempotency-Key.

    With ``MAX_RETRIES = 4`` (spec 0004), the SUT fires four POSTs total
    before exhausting its retry budget; this scenario succeeds on the
    fourth attempt.
    """
    poster = _SequencedPoster(
        [
            _FakeResponse(503, body="busy"),
            _FakeResponse(503, body="busy"),
            _FakeResponse(503, body="busy"),
            _FakeResponse(200, body='{"ok": true}'),
        ]
    )

    with patch("app.services.external_api_service.requests.post", side_effect=poster):
        result = service.post_cvvr_results(
            trip_id="TRIP-A",
            events=[_sample_event()],
            video_s3_url="https://s3.example.com/video.mp4",
            division="demo",
            run_dir=run_dir,
        )

    assert result["success"] is True
    assert len(poster.calls) == 4
    keys = [c["headers"].get("Idempotency-Key") for c in poster.calls]
    assert all(keys), f"Every attempt must carry the header: got {keys}"
    assert len(set(keys)) == 1, f"Same key on every retry, got: {keys}"
    # Spec 0004 format: f"{trip_id}:{sha256-hex}". The trip-id prefix is
    # what makes a DLQ grep ergonomic for an operator.
    assert keys[0].startswith("TRIP-A:"), (
        f"Idempotency key must follow spec format <trip>:<sha256>, got {keys[0]!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance 2 — DLQ on retries-exhausted, no Authorization captured
# ---------------------------------------------------------------------------


def test_dlq_written_after_retries_exhausted_without_authorization(
    service, fast_retry, run_dir
):
    """4 x 503: retries-exhausted writes a DLQ record with the bearer stripped.

    With ``MAX_RETRIES = 4``, four 503 responses exhaust the budget exactly.
    The DLQ file must land under ``<run_dir>/_failed_external_api/`` per
    spec 0004.
    """
    poster = _SequencedPoster(
        [
            _FakeResponse(503, body="busy"),
            _FakeResponse(503, body="busy"),
            _FakeResponse(503, body="busy"),
            _FakeResponse(503, body="busy"),
        ]
    )

    with patch("app.services.external_api_service.requests.post", side_effect=poster):
        result = service.post_cvvr_results(
            trip_id="TRIP-B",
            events=[_sample_event()],
            video_s3_url="https://s3.example.com/video.mp4",
            division="demo",
            run_dir=run_dir,
        )

    assert result["success"] is False
    assert result.get("dlq_path"), "Retries-exhausted must write a DLQ record"
    assert len(poster.calls) == 4, (
        f"Expected exactly four POSTs under MAX_RETRIES=4, got {len(poster.calls)}"
    )

    # Spec 0004 layout: <run_dir>/_failed_external_api/<trip>_<ts>_<uuid8>.json
    dlq_dir = Path(run_dir) / "_failed_external_api"
    assert dlq_dir.is_dir(), f"DLQ directory missing under run_dir: {dlq_dir}"
    files = sorted(dlq_dir.iterdir())
    assert len(files) == 1, f"Expected exactly one DLQ file, got: {files}"

    # Filename must follow <trip>_<ts>_<uuid8>.json with an 8-hex uuid suffix.
    name = files[0].name
    assert name.startswith("TRIP-B_"), f"DLQ filename must lead with trip_id, got {name}"
    assert name.endswith(".json"), name
    parts = name[: -len(".json")].split("_")
    assert len(parts) == 3, f"Expected <trip>_<ts>_<uuid8> filename, got {name!r}"
    assert len(parts[-1]) == 8, f"Expected 8-char uuid suffix, got {parts[-1]!r}"
    int(parts[-1], 16)  # raises if not hex

    payload = json.loads(files[0].read_text())
    # The bearer token must NEVER survive a DLQ write — that file may end up
    # in logs / on a developer laptop.
    captured_headers = payload.get("headers", {})
    assert all(
        k.lower() != "authorization" for k in captured_headers
    ), f"Authorization must be stripped, got headers: {captured_headers}"
    # The idempotency key DOES survive so a drain replay still dedupes.
    assert payload.get("idempotency_key"), "Idempotency-Key must survive DLQ write"
    assert payload.get("idempotency_key", "").startswith("TRIP-B:"), (
        "Idempotency key in DLQ must follow spec format <trip>:<sha256>"
    )
    assert payload.get("trip_id") == "TRIP-B"
    assert payload.get("attempts") == service.MAX_RETRIES


# ---------------------------------------------------------------------------
# Acceptance 3 — dedup collapses "6.00" and "00:00:06"
# ---------------------------------------------------------------------------


def test_dedup_normalizes_start_time_across_notations(service):
    """``6.00`` and ``00:00:06`` describe the same instant — collapse them.

    Spec 0004 uses 2-decimal precision so the test fixtures use ``"6.00"``
    and ``"00:00:06"`` which both round to ``6.0`` at that precision. A
    genuinely-different ``"00:00:07"`` confirms we don't over-collapse.
    """
    from app.services.external_api_service import _time_to_seconds

    violations = [
        {"tripId": "T1", "types": [5], "startTime": "6.00"},
        {"tripId": "T1", "types": [5], "startTime": "00:00:06"},
        # Keep a genuinely-different one to confirm we don't over-collapse.
        {"tripId": "T1", "types": [5], "startTime": "00:00:07"},
    ]

    unique = service._deduplicate_violations(violations)

    assert len(unique) == 2, f"Expected 2 unique entries, got {len(unique)}: {unique}"
    seconds = sorted(round(_time_to_seconds(v["startTime"]), 2) for v in unique)
    assert seconds == [6.0, 7.0]
