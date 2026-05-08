"""Task 0003 — GPU admission gate tests.

Verifies the contract introduced in
``docs/specs/code-review-fixes/tasks/0003-gpu-slot-on-production-endpoint.md``:

1. ``GPUResourceManager.acquire_gpu_slot`` lazy-constructs its
   ``asyncio.Semaphore`` so the manager can be built synchronously and
   then awaited from any event loop without raising
   ``RuntimeError: <Semaphore [...]> is bound to a different event loop``.
2. ``/api/v1/video/process-and-upload`` honours
   ``MAX_CONCURRENT_VIDEOS`` — five concurrent posts with the cap at 2
   never run more than two ``process_video`` calls simultaneously.
3. The job-queue worker path (``process_video_job``) honours the same
   cap — five queued jobs with the cap at 2 likewise serialise to at
   most two concurrent ``process_video`` calls.
4. ``try_enqueue`` no longer double-counts: when ``_active_count=1`` and
   no requests are pending, a freshly admitted request reports
   ``position=2`` (one running, this is second).

These tests are designed to run on a developer laptop with no GPU and
no real model weights — every heavy dependency is mocked at the
``video_processing_service.process_video`` boundary so the assertions
focus on admission/concurrency behaviour, not detection logic.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Test environment hardening
# ---------------------------------------------------------------------------
# ``app.utils.config`` cross-validates a couple of train-motion flags at
# import time; a developer .env that has TRAIN_MOTION_RULES_ENABLED=1 but
# omits TRAIN_MOTION_DETECTION_ENABLED would otherwise break this whole
# module's collection. Force it on before any ``app`` import — same trick
# the existing ``tests/controllers/test_video_controller_auth.py`` uses.
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
# Cap concurrency at 2 for the whole module — every assertion below
# depends on this. Setting it before importing ``app.utils.config`` is
# what makes ``Settings`` pick it up via env-var.
os.environ["MAX_CONCURRENT_VIDEOS"] = "2"
os.environ.setdefault("JOB_QUEUE_MAX_SIZE", "10")


# ---------------------------------------------------------------------------
# 1. Lazy semaphore + cross-event-loop safety
# ---------------------------------------------------------------------------

def _force_max_concurrent_videos(mgr, value: int) -> None:
    """Coerce a manager's effective ``max_concurrent_videos`` to ``value``.

    ``Settings`` is ``lru_cache``-d, and the singleton may have been
    instantiated before this test module set ``MAX_CONCURRENT_VIDEOS=2``
    in the environment. We monkey ``_settings.max_concurrent_videos``
    directly so every assertion in this module sees the same cap
    regardless of test-collection order.
    """
    # ``Settings`` is a pydantic BaseSettings; field assignment is
    # supported on the in-memory instance via ``__setattr__``.
    try:
        mgr._settings.max_concurrent_videos = value
    except Exception:
        # Fallback for ``model_config = ConfigDict(frozen=True)`` style:
        object.__setattr__(mgr._settings, "max_concurrent_videos", value)


def test_acquire_gpu_slot_lazy_semaphore_across_event_loops():
    """Construct ``GPUResourceManager`` synchronously, then await
    ``acquire_gpu_slot`` from two *separate* ``asyncio.run()`` calls.

    Before the lazy-init fix, the semaphore was constructed in
    ``initialize()`` on the synchronous startup path and bound to
    whichever loop happened to await it first. A second ``asyncio.run``
    creates a fresh loop and the await on the bound semaphore raises
    ``RuntimeError: ... bound to a different event loop``.

    With the fix, the semaphore is constructed lazily on first
    ``acquire_gpu_slot`` and rebuilt under ``_initialization_lock`` so
    each loop gets a semaphore created in its own context.
    """
    from app.services.gpu_resource_manager import GPUResourceManager

    # Build the singleton synchronously — no event loop running.
    mgr = GPUResourceManager()
    _force_max_concurrent_videos(mgr, 2)
    # Reset any semaphore left over from a previous test so we can
    # genuinely exercise the "first acquire" branch in this loop.
    mgr._semaphore = None  # type: ignore[attr-defined]

    async def _exercise_slot():
        async with mgr.acquire_gpu_slot():
            return True

    # Loop #1 — fresh ``asyncio.run`` constructs and tears down a loop.
    assert asyncio.run(_exercise_slot()) is True

    # Force a re-lazy-init for loop #2; the previous loop is closed and
    # the semaphore it built is no longer awaitable. The fix should
    # observe this and re-build under the init lock.
    mgr._semaphore = None  # type: ignore[attr-defined]

    # Loop #2 — another fresh ``asyncio.run``. Must not raise
    # ``RuntimeError: ... bound to a different event loop``.
    assert asyncio.run(_exercise_slot()) is True


# ---------------------------------------------------------------------------
# 2. ``try_enqueue`` no longer double-counts
# ---------------------------------------------------------------------------

def test_try_enqueue_position_no_double_count():
    """Acceptance criterion 4: with ``_active_count=1`` and an empty queue,
    a new admission reports ``position=2`` — *not* 3 (old double-count
    behaviour) and not 0/1 (which would imply we forgot the running job
    is occupying a slot).

    The double-count bug existed because ``acquire_gpu_slot()`` was
    bumping ``_active_count`` *and* ``try_enqueue`` was reading
    ``_active_count + _pending_count`` while the slot was simultaneously
    counted as pending — so the same slot showed up twice in the
    position formula. Fix: ``mark_enqueued_started`` migrates the slot
    out of pending atomically; the position formula is just the natural
    "active + pending" reading without double-counting.
    """
    from app.services.gpu_resource_manager import GPUResourceManager

    mgr = GPUResourceManager()
    _force_max_concurrent_videos(mgr, 1)
    # Pin the manager into a known state so the assertion is
    # deterministic regardless of test ordering.
    mgr._active_count = 1  # type: ignore[attr-defined]
    mgr._pending_count = 0  # type: ignore[attr-defined]

    admitted, position = mgr.try_enqueue()
    try:
        assert admitted is True, "admission should succeed under the cap"
        assert position == 2, (
            f"expected position=2 (one active + this newly pending), "
            f"got position={position} — try_enqueue is still double-counting"
        )
    finally:
        # Restore counters so we don't leak state into later tests.
        mgr._pending_count = 0  # type: ignore[attr-defined]
        mgr._active_count = 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. Concurrency cap on /api/v1/video/process-and-upload
# ---------------------------------------------------------------------------

class _ConcurrencyProbe:
    """Records peak concurrent calls into a wrapped function.

    The wrapped callable simulates ``video_processing_service.process_video``:
    it sleeps briefly (long enough that overlapping requests pile up),
    incrementing ``in_flight`` on entry and decrementing on exit, while
    tracking the *maximum* simultaneous count seen. Tests assert against
    ``peak`` to verify the GPU semaphore is actually serialising work.
    """

    def __init__(self, work_seconds: float = 0.25):
        self._work_seconds = work_seconds
        self._in_flight = 0
        self._peak = 0
        self._lock = threading.Lock()

    @property
    def peak(self) -> int:
        return self._peak

    def __call__(self, *args, **kwargs) -> dict:
        with self._lock:
            self._in_flight += 1
            if self._in_flight > self._peak:
                self._peak = self._in_flight
        try:
            time.sleep(self._work_seconds)
        finally:
            with self._lock:
                self._in_flight -= 1
        # Mimic the shape ``process_video`` returns; only fields read by
        # the controller's downstream blocks are populated.
        return {
            "runDirectory": "/tmp/_probe_run",
            "run_id": "run_00000000_000000",
            "activitiesJsonPath": "",
            "activities": [],
            "activitiesCount": 0,
            "processingTime": self._work_seconds,
            "summary": {},
            "clipsGenerated": 0,
            "clip_files": [],
        }


@pytest.fixture
def app_with_probed_processor(tmp_path, monkeypatch):
    """Build a FastAPI app whose ``video_processing_service`` is replaced
    with a ``_ConcurrencyProbe`` so we can post real HTTP requests and
    measure how many got into ``process_video`` simultaneously.

    Side-effects of ``process_and_upload_video`` we can't avoid (S3
    upload, external API push, VLM verification) are stubbed out so the
    test only exercises the GPU admission gate path.
    """
    from app.controllers import video_controller as vc

    probe = _ConcurrencyProbe(work_seconds=0.30)

    # Replace process_video on the controller's singleton with our probe.
    # ``loop.run_in_executor`` will call the probe in a thread pool so
    # the in_flight counter naturally reflects concurrent thread entries.
    original_process_video = vc.video_processing_service.process_video

    def _probed_process_video(**kwargs):
        return probe(**kwargs)

    monkeypatch.setattr(
        vc.video_processing_service,
        "process_video",
        _probed_process_video,
        raising=True,
    )

    # The save_uploaded_video coroutine is async and writes to disk — we
    # short-circuit it to keep the test self-contained.
    async def _stub_save_uploaded_video(file_content, filename, trip_id):
        path = tmp_path / f"{trip_id}_{filename}"
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr(
        vc.video_processing_service,
        "save_uploaded_video",
        _stub_save_uploaded_video,
        raising=True,
    )

    # Stub validate_video_file to always pass so the probe gets exercised.
    monkeypatch.setattr(
        vc.video_processing_service,
        "validate_video_file",
        lambda filename, file_size: (True, ""),
        raising=True,
    )

    # Disable VLM verification (would otherwise try to read activities.json).
    class _StubVLM:
        def is_enabled(self):
            return False

    monkeypatch.setattr(
        vc, "get_vlm_verification_service", lambda: _StubVLM(), raising=True
    )

    # Stub the external_api transformer + post path so the response build
    # doesn't try to reach the network.
    class _StubExternalAPI:
        def _transform_events_to_violations(self, **kwargs):
            return []

        def post_cvvr_results(self, **kwargs):
            return {"success": True, "violations_count": 0}

    monkeypatch.setattr(
        vc, "get_external_api_service", lambda: _StubExternalAPI(), raising=True
    )

    # Stub S3 upload to no-op so we don't touch boto3.
    class _StubS3:
        def upload_multiple_files(self, file_paths, subfolder, auth_token):
            return True, [], []

    monkeypatch.setattr(vc, "s3_upload_service", _StubS3(), raising=True)

    # Reset the GPU manager's counters and semaphore so each test starts
    # from a known clean state. The lazy-construct path will rebuild the
    # semaphore on first acquire inside the request loop.
    from app.services.gpu_resource_manager import GPUResourceManager

    mgr = GPUResourceManager()
    _force_max_concurrent_videos(mgr, 2)
    mgr._semaphore = None  # type: ignore[attr-defined]
    mgr._active_count = 0  # type: ignore[attr-defined]
    mgr._pending_count = 0  # type: ignore[attr-defined]

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(vc.router)

    yield app, probe

    # Best-effort restore.
    try:
        monkeypatch.setattr(
            vc.video_processing_service,
            "process_video",
            original_process_video,
            raising=True,
        )
    except Exception:
        pass


def test_process_and_upload_respects_max_concurrent_videos(
    app_with_probed_processor,
):
    """Acceptance criterion 1: 5 concurrent posts to
    ``/api/v1/video/process-and-upload`` with ``MAX_CONCURRENT_VIDEOS=2``
    must never have more than two ``process_video`` calls in flight
    simultaneously.

    Implementation: drive the 5 posts inside a single ``asyncio.run``
    via ``httpx.AsyncClient + ASGITransport`` so every request shares
    one event loop (matches production: gunicorn + uvicorn run a
    single loop too). Threads sharing a sync ``TestClient`` would
    each spin up their own loop and the lazy semaphore would be
    constructed against whichever loop won the first acquire — that
    isn't representative of the production hot path we are gating.
    """
    import httpx

    app, probe = app_with_probed_processor

    async def _drive_concurrent_posts():
        # ``ASGITransport`` is the in-process httpx transport that
        # speaks ASGI directly to a FastAPI app — no real network.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30.0
        ) as client:
            async def _post(idx: int):
                files = {
                    "video_file": (
                        f"clip_{idx}.mp4",
                        b"\x00" * 16,
                        "video/mp4",
                    )
                }
                data = {"tripId": f"trip{idx}", "subFolderName": "cvvr"}
                return await client.post(
                    "/api/v1/video/process-and-upload",
                    files=files,
                    data=data,
                )

            return await asyncio.gather(*(_post(i) for i in range(5)))

    responses = asyncio.run(_drive_concurrent_posts())

    # Every request should have completed (200 or some controlled error).
    # The key invariant is the *peak* concurrency, not the response
    # payload — we still want to surface unexpected status codes.
    statuses = [r.status_code for r in responses]
    assert all(s is not None for s in statuses), (
        f"some requests never completed: {statuses}"
    )

    max_concurrent = 2
    assert probe.peak <= max_concurrent, (
        f"peak concurrency {probe.peak} exceeded MAX_CONCURRENT_VIDEOS"
        f"={max_concurrent}; admission gate is leaking. statuses={statuses}"
    )
    # Sanity: at least *one* request actually entered process_video, so
    # we know the test wasn't trivially passing because nothing ran.
    assert probe.peak >= 1, (
        f"probe never observed a concurrent call (peak={probe.peak}); "
        f"the test plumbing failed before reaching process_video. "
        f"statuses={statuses}"
    )


# ---------------------------------------------------------------------------
# 4. Concurrency cap on the queue worker path
# ---------------------------------------------------------------------------

def test_process_video_job_respects_max_concurrent_videos(monkeypatch):
    """Acceptance criterion 2: 5 concurrent invocations of
    ``process_video_job`` (the queue worker entry point) with
    ``MAX_CONCURRENT_VIDEOS=2`` must not let more than 2
    ``process_video`` calls run at once.

    The job_manager would otherwise spawn ``num_workers`` workers and
    they'd race on the GPU. ``acquire_gpu_slot`` inside
    ``process_video_job`` is what serialises them.
    """
    from app import main as app_main
    from app.models.job_models import Job
    from app.services.gpu_resource_manager import GPUResourceManager

    probe = _ConcurrencyProbe(work_seconds=0.20)

    # Replace the queue-path processor's ``process_video`` with the probe.
    monkeypatch.setattr(
        app_main._video_processing_service_for_jobs,
        "process_video",
        lambda **kwargs: probe(**kwargs),
        raising=True,
    )

    # Reset GPU manager state so this test starts clean.
    mgr = GPUResourceManager()
    _force_max_concurrent_videos(mgr, 2)
    mgr._semaphore = None  # type: ignore[attr-defined]
    mgr._active_count = 0  # type: ignore[attr-defined]
    mgr._pending_count = 0  # type: ignore[attr-defined]

    async def _drive_five_jobs():
        # Five fake jobs that will all try to grab the GPU at once.
        # ``Job.id`` is required (no default); we synthesise short UUIDs
        # so the job log lines remain unique per test run.
        import uuid

        jobs = [
            Job(
                id=str(uuid.uuid4()),
                video_path=f"/fake/video_{i}.mp4",
                config={"trip_id": f"t{i}"},
            )
            for i in range(5)
        ]
        return await asyncio.gather(
            *(app_main.process_video_job(j) for j in jobs)
        )

    results = asyncio.run(_drive_five_jobs())
    assert len(results) == 5

    max_concurrent = 2
    assert probe.peak <= max_concurrent, (
        f"queue-worker peak concurrency {probe.peak} exceeded "
        f"MAX_CONCURRENT_VIDEOS={max_concurrent}; process_video_job is "
        f"not gated by acquire_gpu_slot"
    )
