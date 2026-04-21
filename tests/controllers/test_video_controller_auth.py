"""C-9: auth + run_id validation tests for the media / status endpoints.

Covers, for both ``GET /api/status/{run_id}`` and ``GET /api/jobs/{run_id}/media/{filename}``:
  * 401 when no ``X-API-Key`` header is sent.
  * 401 when the header value does not match the configured key.
  * 200 (or legitimate 2xx) when the correct key is sent against an
    existing fixture file.
  * 400 for malformed ``run_id`` values (traversal / non-canonical format),
    even when the correct auth header is present.

Fixtures use ``tmp_path`` for fake run directories — no network, no real
S3, no model weights required.
"""
from __future__ import annotations

# NOTE: ``app.utils.logger`` constructs Settings at import time and will
# fail if the developer's .env has TRAIN_MOTION_RULES_ENABLED=1 without
# TRAIN_MOTION_DETECTION_ENABLED=1. That coupling is pre-existing and
# unrelated to C-9. We flip the motion-detection flag on before any
# ``app`` import so this test module can be collected on dev machines
# that have the former set in .env.
import os

os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Shared API key used across the tests. Random-looking but deterministic so
# assertions are readable; this is only ever set on the in-test Settings
# object via monkeypatch — never written to disk.
_TEST_API_KEY = "test-media-key-c9-unit"
_WRONG_KEY = "definitely-not-the-right-key"
_VALID_RUN_ID = "run_20260420_120000"


@pytest.fixture
def app_with_fake_run(tmp_path, monkeypatch):
    """Build a FastAPI app whose video router points at a temp output_dir.

    Creates ``<tmp_path>/run_20260420_120000/clips/demo_clip.mp4`` with a
    few bytes so the media endpoint's ``isfile`` + read path succeed.
    Returns a tuple ``(app, run_id, filename, clip_bytes)``.
    """
    # Point the output dir (used by both endpoints) at tmp_path.
    run_id = _VALID_RUN_ID
    run_dir = tmp_path / run_id
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True)
    filename = "demo_clip.mp4"
    clip_bytes = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64
    (clips_dir / filename).write_bytes(clip_bytes)

    from app.controllers import video_controller as vc

    # Monkeypatch the module-level singletons used by the handlers.
    monkeypatch.setattr(vc.settings, "output_dir", str(tmp_path), raising=True)
    monkeypatch.setattr(vc.settings, "media_api_key", _TEST_API_KEY, raising=True)

    # Reset the one-shot "key missing" warning flag so each test starts clean.
    monkeypatch.setattr(vc, "_media_api_key_missing_warned", False, raising=True)

    # Stub the status service so the status endpoint returns 200 without
    # needing any real processing artifacts on disk.
    class _StubStatusService:
        def get_processing_status(self, run_dir_arg):
            return {"run_dir": run_dir_arg, "status": "ok"}

    monkeypatch.setattr(
        vc, "video_processing_service", _StubStatusService(), raising=True
    )

    # Mount only the video router — no startup event, no model loading.
    app = FastAPI()
    app.include_router(vc.router)
    return app, run_id, filename, clip_bytes


@pytest.fixture
def client(app_with_fake_run):
    app, _run_id, _filename, _clip_bytes = app_with_fake_run
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Media endpoint auth
# ---------------------------------------------------------------------------

def test_media_endpoint_401_without_key(client, app_with_fake_run):
    _, run_id, filename, _ = app_with_fake_run
    resp = client.get(f"/api/jobs/{run_id}/media/{filename}")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid_or_missing_api_key"


def test_media_endpoint_401_with_wrong_key(client, app_with_fake_run):
    _, run_id, filename, _ = app_with_fake_run
    resp = client.get(
        f"/api/jobs/{run_id}/media/{filename}",
        headers={"X-API-Key": _WRONG_KEY},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid_or_missing_api_key"


def test_media_endpoint_200_with_correct_key(client, app_with_fake_run):
    _, run_id, filename, clip_bytes = app_with_fake_run
    resp = client.get(
        f"/api/jobs/{run_id}/media/{filename}",
        headers={"X-API-Key": _TEST_API_KEY},
    )
    assert resp.status_code == 200
    assert resp.content == clip_bytes
    assert resp.headers.get("content-type", "").startswith("video/mp4")


# ---------------------------------------------------------------------------
# Status endpoint auth
# ---------------------------------------------------------------------------

def test_status_endpoint_401_without_key(client, app_with_fake_run):
    _, run_id, _, _ = app_with_fake_run
    resp = client.get(f"/api/status/{run_id}")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid_or_missing_api_key"


def test_status_endpoint_401_with_wrong_key(client, app_with_fake_run):
    _, run_id, _, _ = app_with_fake_run
    resp = client.get(
        f"/api/status/{run_id}",
        headers={"X-API-Key": _WRONG_KEY},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid_or_missing_api_key"


def test_status_endpoint_200_with_correct_key(client, app_with_fake_run):
    _, run_id, _, _ = app_with_fake_run
    resp = client.get(
        f"/api/status/{run_id}",
        headers={"X-API-Key": _TEST_API_KEY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# run_id validation — each case returns 400 even with a valid auth header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_run_id",
    [
        "run_x",
        "run_2026_0420_120000",  # underscores in wrong spots
        "run_20260420",          # missing time segment
        "RUN_20260420_120000",   # uppercase rejected
        "run_20260420_120000x",  # trailing garbage
        "run20260420_120000",    # missing underscore
        "run_00000000_000000aa", # trailing alpha chars
    ],
)
def test_status_endpoint_rejects_invalid_run_id(client, bad_run_id):
    # Note: raw traversal segments like ``..`` / ``../etc`` never reach the
    # handler — httpx (and production proxies) normalize them before routing,
    # which yields a 404 for a non-matching path. The regex still protects
    # against any traversal-style value that does survive normalization
    # (e.g. URL-encoded payloads, trailing-slash variants), which we cover
    # implicitly via the non-canonical-format cases above.
    resp = client.get(
        f"/api/status/{bad_run_id}",
        headers={"X-API-Key": _TEST_API_KEY},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "invalid_run_id"


@pytest.mark.parametrize(
    "bad_run_id",
    [
        "run_x",
        "run_20260420",
        "RUN_20260420_120000",
        "run_20260420_120000x",
    ],
)
def test_media_endpoint_rejects_invalid_run_id(client, bad_run_id):
    resp = client.get(
        f"/api/jobs/{bad_run_id}/media/demo_clip.mp4",
        headers={"X-API-Key": _TEST_API_KEY},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "invalid_run_id"


# ---------------------------------------------------------------------------
# Sanity: health endpoint stays unauthenticated (no dependency attached)
# ---------------------------------------------------------------------------

def test_health_endpoint_unauthenticated(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
