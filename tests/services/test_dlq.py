"""Tests for the DLQ writer + drain.

Covers acceptance criterion (4) from
``docs/specs/code-review-fixes/tasks/0004-external-api-dlq-and-idempotency.md``:

    * drain_dlq deletes the file on success.
    * drain_dlq retains the file and bumps ``attempts`` on retry-failure.

Also exercises the security-critical guarantee from criterion (2):

    * write_dlq strips Authorization-class headers before persisting.

Spec 0004 places DLQ files under ``<run_dir>/_failed_external_api/`` (per-run
scoping) rather than a global drop directory, so these tests build a fake
run directory and route ``drain_dlq`` at it via the ``output_dir`` setting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """Create a per-test run directory matching the production layout.

    ``drain_dlq`` walks ``<output_dir>/run_*/_failed_external_api/*.json`` so
    the run directory name must match ``run_*`` for the drain glob to find it.
    """
    target = tmp_path / "run_20260508_120000"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def output_dir_setting(tmp_path, monkeypatch):
    """Point ``settings.output_dir`` at the per-test ``tmp_path`` so
    ``drain_dlq`` only sees this test's DLQ files.
    """
    from app.utils.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "output_dir", str(tmp_path), raising=False)
    return tmp_path


def _seed_dlq(run_dir: Path) -> str:
    """Write one well-formed DLQ record under ``run_dir`` and return its path."""
    from app.services.dlq import write_dlq

    path = write_dlq(
        run_dir=str(run_dir),
        url="https://api.example.com/cvvr/cvvrTripViolations/addUpdateBulk",
        payload=[{"tripId": "T1", "types": [5], "startTime": "00:00:06"}],
        headers={
            "Content-Type": "application/json",
            # The token below MUST NOT survive — see assertion below.
            "Authorization": "Bearer secret-token-abc",
            "Cookie": "session=should-not-persist",
        },
        idempotency_key="T1:abc123-stable-key",
        context_label="Violations for trip_id=T1",
        timeout=30,
        trip_id="T1",
        last_error="status=503",
        attempts=3,
    )
    assert path is not None
    return path


# ---------------------------------------------------------------------------
# write_dlq
# ---------------------------------------------------------------------------


def test_write_dlq_strips_authorization_and_cookie(run_dir):
    path = _seed_dlq(run_dir)
    record = json.loads(Path(path).read_text())

    headers = record["headers"]
    lowered = {k.lower() for k in headers}
    assert "authorization" not in lowered
    assert "cookie" not in lowered
    # Non-sensitive headers survive so we can replay the request faithfully.
    assert headers.get("Content-Type") == "application/json"

    # The idempotency key must survive — without it the drain replay would
    # re-introduce the duplicate the customer asked us to dedupe.
    assert record["idempotency_key"] == "T1:abc123-stable-key"
    assert record["trip_id"] == "T1"
    assert record["attempts"] == 3


def test_write_dlq_lands_under_run_failed_external_api(run_dir):
    """Spec 0004 layout: ``<run_dir>/_failed_external_api/<trip>_<ts>_<uuid8>.json``."""
    path = Path(_seed_dlq(run_dir))

    assert path.parent == run_dir / "_failed_external_api", (
        f"DLQ file must land under <run_dir>/_failed_external_api/, got {path}"
    )
    # <trip>_<ts>_<uuid8>.json
    name = path.name
    assert name.startswith("T1_") and name.endswith(".json"), name
    parts = name[: -len(".json")].split("_")
    assert len(parts) == 3, f"Expected <trip>_<ts>_<uuid8> filename, got {name!r}"
    assert len(parts[-1]) == 8, f"Expected 8-char uuid suffix, got {parts[-1]!r}"
    int(parts[-1], 16)  # raises if not hex


def test_write_dlq_evicts_oldest_when_dir_at_cap(run_dir, monkeypatch):
    """At ``MAX_DLQ_FILES_PER_DIR``, the oldest file is evicted on next write.

    The cap stops a runaway producer from filling disk. We bring the cap
    down to 3 so the test stays fast.
    """
    import time as _time
    from app.services import dlq

    monkeypatch.setattr(dlq, "MAX_DLQ_FILES_PER_DIR", 3, raising=False)

    paths = []
    for i in range(3):
        p = _seed_dlq(run_dir)
        # Force monotonic mtimes so eviction order is deterministic regardless
        # of the host's filesystem timestamp resolution.
        os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))
        paths.append(p)

    # All three present before the cap-triggering write.
    assert all(Path(p).exists() for p in paths)

    # Fourth write must evict the oldest (paths[0]).
    fourth = _seed_dlq(run_dir)
    assert Path(fourth).exists()
    assert not Path(paths[0]).exists(), (
        "Oldest DLQ file must be evicted once the directory is at cap"
    )
    # Newer ones survive.
    assert Path(paths[1]).exists()
    assert Path(paths[2]).exists()


def test_write_dlq_no_op_when_run_dir_missing(tmp_path):
    """Spec 0004 forbids a global drop dir; an empty ``run_dir`` must be a no-op."""
    from app.services.dlq import write_dlq

    result = write_dlq(
        run_dir="",
        url="https://api.example.com/cvvr/cvvrTripViolations/addUpdateBulk",
        payload=[{"tripId": "T1"}],
        headers={"Content-Type": "application/json"},
        idempotency_key="T1:k",
        context_label="ctx",
        timeout=30,
        trip_id="T1",
        last_error="boom",
        attempts=1,
    )
    assert result is None, "Empty run_dir must short-circuit and return None"


# ---------------------------------------------------------------------------
# drain_dlq — success path deletes the file
# ---------------------------------------------------------------------------


def test_drain_dlq_deletes_file_on_success(run_dir, output_dir_setting):
    path = _seed_dlq(run_dir)
    assert Path(path).exists()

    from app.services.dlq import drain_dlq

    def _ok(_record: Dict[str, Any]) -> Tuple[bool, None]:
        return True, None

    stats = drain_dlq(repost=_ok)

    assert stats == {"drained": 1, "succeeded": 1, "failed": 0}
    assert not Path(path).exists(), "Successfully drained file must be removed"


# ---------------------------------------------------------------------------
# drain_dlq — failure path retains and bumps attempts
# ---------------------------------------------------------------------------


def test_drain_dlq_retains_and_bumps_attempts_on_failure(run_dir, output_dir_setting):
    path = _seed_dlq(run_dir)
    before = json.loads(Path(path).read_text())
    initial_attempts = before["attempts"]

    from app.services.dlq import drain_dlq

    def _fail(_record: Dict[str, Any]) -> Tuple[bool, str]:
        return False, "status=503"

    stats = drain_dlq(repost=_fail)

    assert stats == {"drained": 1, "succeeded": 0, "failed": 1}
    assert Path(path).exists(), "Failed drain must retain the DLQ file"

    after = json.loads(Path(path).read_text())
    assert (
        after["attempts"] == initial_attempts + 1
    ), f"Expected attempts to bump from {initial_attempts} to {initial_attempts + 1}"
    assert after["last_error"] == "status=503"
    assert "last_attempt_at" in after


def test_drain_dlq_handles_empty_directory(output_dir_setting):
    from app.services.dlq import drain_dlq

    stats = drain_dlq(repost=lambda r: (True, None))
    assert stats == {"drained": 0, "succeeded": 0, "failed": 0}


def test_drain_dlq_isolates_per_record_failures(run_dir, output_dir_setting):
    """A repost callable that raises must not abort sibling records."""
    path1 = _seed_dlq(run_dir)
    # Seed a second record with a distinct trip id so the filename differs.
    from app.services.dlq import write_dlq

    path2 = write_dlq(
        run_dir=str(run_dir),
        url="https://api.example.com/cvvr/cvvrTripViolations/addUpdateBulk",
        payload=[{"tripId": "T2", "types": [5], "startTime": "00:00:09"}],
        headers={"Content-Type": "application/json"},
        idempotency_key="T2:def456",
        context_label="Violations for trip_id=T2",
        timeout=30,
        trip_id="T2",
        last_error="status=503",
        attempts=3,
    )
    assert path2 is not None and path2 != path1

    from app.services.dlq import drain_dlq

    seen = []

    def _flaky(record: Dict[str, Any]) -> Tuple[bool, None]:
        seen.append(record["trip_id"])
        if record["trip_id"] == "T1":
            raise RuntimeError("boom")
        return True, None

    stats = drain_dlq(repost=_flaky)

    assert sorted(seen) == ["T1", "T2"]
    assert stats["drained"] == 2
    # T2 succeeded -> deleted; T1 raised -> retained with bumped attempts.
    assert Path(path1).exists()
    assert not Path(path2).exists()


def test_drain_dlq_walks_multiple_run_dirs(tmp_path, monkeypatch):
    """Spec 0004 ``drain_dlq`` walks every ``run_*/_failed_external_api/``."""
    from app.services.dlq import drain_dlq, write_dlq
    from app.utils.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "output_dir", str(tmp_path), raising=False)

    run_a = tmp_path / "run_20260508_100000"
    run_b = tmp_path / "run_20260508_110000"
    run_a.mkdir()
    run_b.mkdir()

    pa = write_dlq(
        run_dir=str(run_a),
        url="https://api.example.com/x",
        payload=[{}],
        headers={"Content-Type": "application/json"},
        idempotency_key="A:k",
        context_label="ctx",
        timeout=30,
        trip_id="A",
        last_error="boom",
        attempts=1,
    )
    pb = write_dlq(
        run_dir=str(run_b),
        url="https://api.example.com/x",
        payload=[{}],
        headers={"Content-Type": "application/json"},
        idempotency_key="B:k",
        context_label="ctx",
        timeout=30,
        trip_id="B",
        last_error="boom",
        attempts=1,
    )
    assert pa and pb

    seen = []

    def _ok(record: Dict[str, Any]) -> Tuple[bool, None]:
        seen.append(record["trip_id"])
        return True, None

    stats = drain_dlq(repost=_ok)
    assert sorted(seen) == ["A", "B"]
    assert stats == {"drained": 2, "succeeded": 2, "failed": 0}
    assert not Path(pa).exists()
    assert not Path(pb).exists()
