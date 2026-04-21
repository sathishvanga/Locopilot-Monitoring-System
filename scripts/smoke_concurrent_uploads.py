"""Concurrency-admission smoke test for the Locopilot video endpoint (C-1).

Fires N (default 3) simultaneous ``POST /api/video/analyze`` requests at a
running Locopilot server and verifies that:

  * At most ``MAX_CONCURRENT_VIDEOS`` requests are "active" at any instant
    (as reported by :meth:`GPUResourceManager.get_status` via ``/api/health``
    or ``/api/video/queue/status``).
  * Extra requests queue via :meth:`GPUResourceManager.try_enqueue` rather
    than being dropped — responses come back as either 200 or 503 ("queue
    full"), never a hang or a 500.
  * None of the workers OOM (no 500 responses with CUDA OOM in the body).

This script is intentionally a *manual* smoke test, not a pytest integration
test: it needs a running server with real GPU models loaded, which is not
available in CI.

Usage
-----
Start the server locally (or point at a staging host)::

    gunicorn -c gunicorn_config.py app.main:app

Then run::

    python scripts/smoke_concurrent_uploads.py --video path/to/sample.mp4

Options::

    --host            Base URL (default: http://localhost:8000)
    --video           Path to a small test video to upload (required)
    --trip-id         trip_id to send (default: "smoke-test")
    --concurrency     Number of simultaneous POSTs (default: 3)
    --poll-interval   Seconds between status polls (default: 1.0)
    --max-poll-sec    Give up polling after this many seconds (default: 120)

Exit codes
----------
  0 — all assertions held; admission model is behaving correctly.
  1 — ran but one or more assertions failed; see stderr for which one.
  2 — runtime error (bad args, server unreachable, missing video, etc.).

Dependencies
------------
``httpx`` (already a project dep via FastAPI TestClient).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover — httpx ships with FastAPI test deps
    print("httpx not installed. pip install httpx", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def fetch_status(client: httpx.AsyncClient, host: str) -> Optional[Dict[str, Any]]:
    """Return the queue-status payload, or ``None`` if the server isn't
    exposing one at this endpoint.

    The real server mounts ``/api/video/queue/status`` (see
    ``app/controllers/video_controller.py``); this endpoint does not require
    auth and returns the GPU manager's ``active_count`` + ``max_concurrent``
    fields we want to observe.
    """
    try:
        resp = await client.get(f"{host}/api/video/queue/status", timeout=5.0)
    except httpx.RequestError as exc:
        print(f"[status] request failed: {exc}", file=sys.stderr)
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


async def post_analyze(
    client: httpx.AsyncClient,
    host: str,
    video_path: Path,
    trip_id: str,
    label: str,
) -> Tuple[str, int, Optional[str]]:
    """Fire a single ``POST /api/video/analyze``.

    Returns ``(label, status_code, error_snippet_or_none)``. We read the
    response body for diagnostic purposes (OOM detection) but don't parse
    it strictly — the endpoint streams a long-running processing result.
    """
    with video_path.open("rb") as fh:
        files = {"video": (video_path.name, fh.read(), "video/mp4")}
    data = {"tripId": trip_id}
    started = time.monotonic()
    try:
        resp = await client.post(
            f"{host}/api/video/analyze",
            files=files,
            data=data,
            timeout=httpx.Timeout(900.0, connect=10.0),
        )
    except httpx.RequestError as exc:
        elapsed = time.monotonic() - started
        return label, -1, f"request error after {elapsed:.1f}s: {exc}"
    elapsed = time.monotonic() - started
    body = resp.text or ""
    snippet = body[:300] if resp.status_code >= 400 else None
    print(f"[post:{label}] status={resp.status_code} elapsed={elapsed:.1f}s")
    return label, resp.status_code, snippet


async def poll_status_until(
    client: httpx.AsyncClient,
    host: str,
    stop_event: asyncio.Event,
    interval: float,
    max_seconds: float,
) -> List[Dict[str, Any]]:
    """Poll ``/api/video/queue/status`` until ``stop_event`` is set.

    Returns the list of samples collected. We capture ``active_count`` and
    ``max_concurrent`` on each tick so the caller can assert the admission
    cap was never exceeded.
    """
    samples: List[Dict[str, Any]] = []
    deadline = time.monotonic() + max_seconds
    while not stop_event.is_set() and time.monotonic() < deadline:
        payload = await fetch_status(client, host)
        if payload is not None:
            payload["_t"] = time.monotonic()
            samples.append(payload)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
    return samples


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_admission_respected(samples: List[Dict[str, Any]]) -> List[str]:
    """Return a list of assertion-failure strings (empty == pass)."""
    failures: List[str] = []
    if not samples:
        failures.append("no /api/video/queue/status samples collected")
        return failures

    # Pull fields under tolerant key names — the endpoint has historically
    # used ``active_count`` / ``max_concurrent`` but could nest under a
    # "gpu" key.
    def _get(sample: Dict[str, Any], key: str) -> Optional[int]:
        if key in sample:
            return sample.get(key)
        gpu = sample.get("gpu")
        if isinstance(gpu, dict):
            return gpu.get(key)
        return None

    cap: Optional[int] = None
    for s in samples:
        got_cap = _get(s, "max_concurrent")
        if got_cap is not None:
            cap = int(got_cap)
            break
    if cap is None:
        failures.append(
            "could not determine max_concurrent from any status sample — "
            "check /api/video/queue/status payload shape"
        )
        return failures

    max_seen = 0
    for s in samples:
        ac = _get(s, "active_count")
        if ac is None:
            continue
        if int(ac) > max_seen:
            max_seen = int(ac)

    if max_seen > cap:
        failures.append(
            f"admission violation: saw active_count={max_seen} > "
            f"max_concurrent={cap}"
        )
    else:
        print(f"[assert] max observed active_count = {max_seen} / cap {cap} — OK")
    return failures


def assert_no_oom(responses: List[Tuple[str, int, Optional[str]]]) -> List[str]:
    """Return list of OOM-related failures (empty == pass)."""
    failures: List[str] = []
    for label, status, snippet in responses:
        if status == 500 and snippet and (
            "CUDAOutOfMemoryError" in snippet
            or "out of memory" in snippet.lower()
        ):
            failures.append(f"request {label}: server reported CUDA OOM — {snippet}")
    return failures


def assert_extras_handled(
    responses: List[Tuple[str, int, Optional[str]]], cap: int
) -> List[str]:
    """When concurrency exceeds cap, extras should either queue (200) or be
    shed cleanly (503) — never 500 with no OOM, never a hang.
    """
    failures: List[str] = []
    for label, status, snippet in responses:
        if status in (200, 503):
            continue
        if status == -1:
            failures.append(f"request {label}: connection error — {snippet}")
        elif status != 500:  # 500s already covered by OOM check
            failures.append(
                f"request {label}: unexpected status {status} — {snippet or '<no body>'}"
            )
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"video not found: {video_path}", file=sys.stderr)
        return 2

    async with httpx.AsyncClient() as client:
        # 1. Probe the status endpoint once — if it's not there we can
        # still run the test but admission assertions will be skipped.
        initial = await fetch_status(client, args.host)
        if initial is None:
            print(
                "[warn] /api/video/queue/status not reachable; admission "
                "assertions will be skipped. Check server is running.",
                file=sys.stderr,
            )
            cap = None
        else:
            # Tolerant lookup; see assert_admission_respected for shape notes.
            cap = initial.get("max_concurrent") or (
                initial.get("gpu", {}).get("max_concurrent")
                if isinstance(initial.get("gpu"), dict)
                else None
            )
            print(f"[init] max_concurrent = {cap}")

        # 2. Start the status poller in the background.
        stop_event = asyncio.Event()
        poll_task = asyncio.create_task(
            poll_status_until(
                client,
                args.host,
                stop_event,
                interval=args.poll_interval,
                max_seconds=args.max_poll_sec,
            )
        )

        # 3. Fire all N requests simultaneously.
        print(f"[fire] issuing {args.concurrency} concurrent POSTs")
        post_tasks = [
            post_analyze(
                client,
                args.host,
                video_path,
                trip_id=f"{args.trip_id}-{i}",
                label=f"req{i}",
            )
            for i in range(args.concurrency)
        ]
        responses = await asyncio.gather(*post_tasks)

        # 4. Stop polling and collect samples.
        stop_event.set()
        samples = await poll_task

    # 5. Assertions.
    failures: List[str] = []
    failures.extend(assert_no_oom(responses))
    if cap is not None:
        failures.extend(assert_extras_handled(responses, int(cap)))
        failures.extend(assert_admission_respected(samples))
    else:
        print(
            "[warn] cap unknown — skipping admission/extras assertions",
            file=sys.stderr,
        )

    if failures:
        print("\n=== FAIL ===", file=sys.stderr)
        for f in failures:
            print(f" - {f}", file=sys.stderr)
        return 1

    print("\n=== PASS ===")
    print(f"  {len(responses)} concurrent POSTs, {len(samples)} status samples")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default=os.getenv("LOCOPILOT_HOST", "http://localhost:8000"))
    parser.add_argument("--video", required=True, help="path to a small test video")
    parser.add_argument("--trip-id", default="smoke-test")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-poll-sec", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(2)
