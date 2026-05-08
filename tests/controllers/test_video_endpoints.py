"""Endpoint tests for the SSRF + upload-size guards added in Task 0008.

Acceptance criteria from
``docs/specs/code-review-fixes/tasks/0008-lock-down-ssrf-surface.md``:

  * POST with ``Content-Length: 6000000000`` (above
    ``max_upload_size = 5 GB``) → 413, no bytes consumed.
  * POST with ``videoUrl=http://169.254.169.254/...`` → 400.

The test mounts the router into a fresh FastAPI app and mocks out all
heavyweight processing collaborators so the SSRF / size-cap branches can be
exercised hermetically.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# Same convention used by ``test_video_controller_auth.py``: pre-set env
# vars before any ``app`` import so Settings doesn't trip on a dev-machine
# .env that has TRAIN_MOTION_RULES_ENABLED=1 without
# TRAIN_MOTION_DETECTION_ENABLED, and skip absolute-path existence checks
# on YOLO weights.
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")


class _AsyncContextManagerStub:
    """Awaitable async-context-manager stand-in used to bypass the
    ``async with gpu_resource_manager.acquire_gpu_slot()`` block in the
    analyze endpoint."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """A TestClient that hits the real ``video_router`` with heavy services
    stubbed.

    We patch the controller-module-level singletons so importing the router
    never tries to load YOLO / VLM / GPU resources, then mount the router on
    a minimal FastAPI app.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.controllers import video_controller as vc

    # Stub the controller's module-level service objects.
    monkeypatch.setattr(vc, "video_processing_service", MagicMock())
    monkeypatch.setattr(vc, "s3_upload_service", MagicMock())

    # Don't hit the real GPU manager.
    fake_gpu = MagicMock()
    fake_gpu.try_enqueue.return_value = (True, 0)
    fake_gpu.acquire_gpu_slot = MagicMock(
        return_value=_AsyncContextManagerStub()
    )
    fake_gpu.mark_enqueued_started = MagicMock()
    fake_gpu.release_enqueue_on_error = MagicMock()
    monkeypatch.setattr(
        vc, "get_gpu_resource_manager", lambda: fake_gpu
    )

    app = FastAPI()
    app.include_router(vc.router)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/v1/video/process-and-upload — 413 on oversize Content-Length
# ---------------------------------------------------------------------------

def test_process_and_upload_rejects_oversize_content_length(monkeypatch):
    """Spec acceptance criterion 2: ``Content-Length: 6000000000`` → 413,
    no bytes consumed.

    httpx (under TestClient) auto-computes Content-Length from the body,
    so we can't reliably forge the header through ``client.post``. We
    drive the ASGI app directly with a hand-crafted scope whose
    ``content-length`` header advertises 6 GB while the actual body is a
    tiny multipart blob. The handler must reject before reading any
    bytes — i.e. before invoking ``video_processing_service.save_uploaded_video``.
    """
    import asyncio

    from fastapi import FastAPI

    from app.controllers import video_controller as vc

    # Stub heavy collaborators on the controller module.
    monkeypatch.setattr(vc, "video_processing_service", MagicMock())
    monkeypatch.setattr(vc, "s3_upload_service", MagicMock())

    fake_gpu = MagicMock()
    fake_gpu.try_enqueue.return_value = (True, 0)
    fake_gpu.acquire_gpu_slot = MagicMock(
        return_value=_AsyncContextManagerStub()
    )
    monkeypatch.setattr(vc, "get_gpu_resource_manager", lambda: fake_gpu)

    app = FastAPI()
    app.include_router(vc.router)

    boundary = "----testboundary0008"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"tripId\"\r\n\r\n"
        f"test-trip-001\r\n"
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"video_file\"; "
        f"filename=\"x.mp4\"\r\n"
        f"Content-Type: video/mp4\r\n\r\n"
        f"AAAA\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    oversize = 6_000_000_000  # 6e9 > 5 GiB max_upload_size

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/video/process-and-upload",
        "raw_path": b"/api/v1/video/process-and-upload",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (
                b"content-type",
                f"multipart/form-data; boundary={boundary}".encode(),
            ),
            # Forged content-length: advertises 6 GB while the body is
            # only a few hundred bytes. The pre-stream guard must trip
            # on this header alone.
            (b"content-length", str(oversize).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    sent_messages: list[dict] = []
    body_sent = {"done": False}

    async def receive():
        if not body_sent["done"]:
            body_sent["done"] = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message):
        sent_messages.append(message)

    asyncio.run(app(scope, receive, send))

    start_msgs = [m for m in sent_messages if m["type"] == "http.response.start"]
    assert start_msgs, f"no response started; got {sent_messages!r}"
    status = start_msgs[0]["status"]
    body_msgs = [m for m in sent_messages if m["type"] == "http.response.body"]
    body_text = b"".join(m.get("body", b"") for m in body_msgs).decode(
        "utf-8", errors="replace"
    )

    assert status == 413, (
        f"expected 413 on oversize Content-Length, got {status}: {body_text}"
    )
    assert "max size" in body_text.lower()

    # The 413 must fire BEFORE save_uploaded_video runs.
    assert not vc.video_processing_service.save_uploaded_video.called, (
        "save_uploaded_video should not be invoked when Content-Length "
        "exceeds max_upload_size"
    )


# ---------------------------------------------------------------------------
# /api/video/analyze — 400 on SSRF metadata-IP videoUrl
# ---------------------------------------------------------------------------

def test_analyze_rejects_metadata_ip_videourl(client):
    """Spec acceptance criterion 3: ``videoUrl=http://169.254.169.254/...``
    → 400 (URL rejected by validate_external_url).
    """
    from app.controllers import video_controller as vc

    data = {
        "tripId": "test-trip-002",
        "videoUrl": "http://169.254.169.254/latest/meta-data/",
    }
    response = client.post("/api/video/analyze", data=data)

    assert response.status_code == 400, (
        f"expected 400 on metadata-IP videoUrl, got "
        f"{response.status_code}: {response.text}"
    )
    assert (
        "videourl rejected" in response.text.lower()
        or "169.254" in response.text
    )

    # The SSRF guard runs BEFORE save_uploaded_video, so it must have
    # stayed untouched for this rejection path.
    assert not vc.video_processing_service.save_uploaded_video.called


def test_analyze_rejects_off_allowlist_videourl(client):
    """Defense-in-depth: a perfectly-public URL that isn't in
    ``minio_allowed_hosts`` must still be rejected with 400."""
    data = {
        "tripId": "test-trip-003",
        "videoUrl": "https://example.com/cvss/x.mp4",
    }
    response = client.post("/api/video/analyze", data=data)

    assert response.status_code == 400
    assert (
        "videourl rejected" in response.text.lower()
        or "host not in allowlist" in response.text.lower()
    )
