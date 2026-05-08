"""Tests for ``app.utils.json_utils.atomic_write_json`` (Task 0002).

These tests cover the three failure modes the spec calls out:

1. **Numpy serialisation** — Pipeline-1 routinely emits ``np.bool_``,
   ``np.float32``, ``np.int64`` and ``np.ndarray`` values inside an
   activity record. The previous repository encoder was missing
   ``np.bool_`` and would raise ``TypeError`` on real payloads. The
   canonical :class:`NumpyEncoder` here must encode all of them.

2. **Concurrency** — A Pipeline-1 worker writing the file at the same
   time as the Pipeline-2 VLM rewrite hook is a real race. Two threads
   each writing 100 times to the same path must always leave a parseable
   JSON file behind (i.e. no half-written truncation visible to a
   reader).

3. **Crash-mid-write recovery** — If something explodes between the
   ``mkstemp`` and the ``os.replace``, the original ``activities.json``
   must remain intact. ``open(..., 'w')`` would have truncated it.

The tests are deliberately self-contained — no detector / video / GPU
imports — so they run in any environment that has numpy + portalocker.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure the repository root is on sys.path so ``app.utils.*`` imports
# resolve whether pytest is invoked from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.utils.json_utils import NumpyEncoder, atomic_write_json  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Numpy types must serialise without raising TypeError
# ---------------------------------------------------------------------------


def test_atomic_write_handles_numpy_scalars_and_arrays(tmp_path):
    """Pipeline-1 emits np.bool_/np.float32/np.ndarray in activity payloads.

    The previous ``ActivityRepository.NumpyEncoder`` was missing
    ``np.bool_`` and would raise ``TypeError`` on real records. The
    consolidated encoder must round-trip every scalar plus arrays.
    """
    target = tmp_path / "activities.json"
    payload = [
        {
            "activityType": "writing",
            "is_running": np.bool_(True),  # the historical regression
            "score": np.float32(1.5),
            "frame_id": np.int64(42),
            "keypoints": np.array([1, 2, 3]),
            "nested": {"flag": np.bool_(False), "vec": np.array([0.1, 0.2])},
        }
    ]

    # Must not raise.
    atomic_write_json(str(target), payload)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded[0]["is_running"] is True
    assert loaded[0]["score"] == pytest.approx(1.5, rel=1e-5)
    assert loaded[0]["frame_id"] == 42
    assert loaded[0]["keypoints"] == [1, 2, 3]
    assert loaded[0]["nested"]["flag"] is False
    assert loaded[0]["nested"]["vec"] == pytest.approx([0.1, 0.2], rel=1e-5)


def test_numpy_encoder_handles_bool_directly():
    """Direct encoder usage — guards against future drift inside the encoder."""
    encoded = json.dumps(
        {"a": np.bool_(True), "b": np.float32(2.5), "c": np.array([1, 2])},
        cls=NumpyEncoder,
    )
    assert json.loads(encoded) == {"a": True, "b": pytest.approx(2.5), "c": [1, 2]}


# ---------------------------------------------------------------------------
# 2. Two threads racing — file is always parseable
# ---------------------------------------------------------------------------


def test_concurrent_writes_never_leave_a_corrupt_file(tmp_path):
    """Two threads each writing 100 times must always produce parseable JSON.

    Without locking + atomic replace, a reader catching the file mid-write
    would see truncated bytes; here we just assert no thread observed an
    error and the post-condition file parses cleanly.
    """
    target = tmp_path / "activities.json"
    iterations = 100
    errors: list[Exception] = []
    error_lock = threading.Lock()

    def writer(tag: str) -> None:
        try:
            for i in range(iterations):
                atomic_write_json(
                    str(target),
                    [{"writer": tag, "i": i, "flag": np.bool_(i % 2 == 0)}],
                )
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            with error_lock:
                errors.append(exc)

    t1 = threading.Thread(target=writer, args=("A",))
    t2 = threading.Thread(target=writer, args=("B",))
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)

    assert not t1.is_alive() and not t2.is_alive(), "writer threads hung"
    assert errors == [], f"writer threads raised: {errors!r}"

    # The file must exist and parse to one of the two payload shapes.
    assert target.exists()
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(parsed, list) and len(parsed) == 1
    assert parsed[0]["writer"] in {"A", "B"}
    assert isinstance(parsed[0]["i"], int)
    assert isinstance(parsed[0]["flag"], bool)


# ---------------------------------------------------------------------------
# 2b. Multi-process writers — flock is per-process on POSIX
# ---------------------------------------------------------------------------
#
# The thread test above is necessary but not sufficient: ``portalocker`` on
# POSIX uses ``fcntl.flock`` which is per-*process*, so two threads inside
# the same interpreter would never block each other on the lock and the
# atomic-rename ordering happens to serialise them anyway. This test fires
# real OS processes which is what the lock is actually defending against
# (Pipeline-1 worker vs Pipeline-2 VLM rewrite hook in the controller).


def _process_writer(target_path: str, tag: str, iterations: int) -> None:
    """Top-level so ``multiprocessing`` can pickle it on macOS spawn-start.

    Each iteration writes a list of length (i + 1) so the last writer's
    payload length is deterministic given iterations. We assert post-run
    that the file parses cleanly and len(activities) matches one of the
    valid final-write shapes from any of the workers.
    """
    # Re-establish ``app.utils.json_utils`` import inside the spawned
    # interpreter — pytest's sys.path tweak doesn't carry across spawn.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.utils.json_utils import atomic_write_json as _write

    for i in range(iterations):
        payload = [
            {"writer": tag, "i": i, "flag": bool(i % 2 == 0)}
            for _ in range(i + 1)
        ]
        _write(target_path, payload)


def test_concurrent_processes_never_corrupt_the_file(tmp_path):
    """Four real OS processes, each writing 50 times, must serialise cleanly.

    ``flock`` (used by portalocker on POSIX) is enforced per-process, so
    only a multi-process test actually exercises the lock. After the run
    the file must still be parseable and ``len(activities)`` must equal
    the final iteration's payload length (``iterations``) for whichever
    process wrote last — i.e. ``iterations``-many records.
    """
    target = tmp_path / "activities.json"
    iterations = 50
    n_workers = 4

    # ``spawn`` is the default on macOS and the most portable; force it
    # so the test behaves the same on Linux CI.
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_process_writer,
            args=(str(target), f"P{idx}", iterations),
        )
        for idx in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    for p in procs:
        assert not p.is_alive(), "writer process hung past 120s"
        assert p.exitcode == 0, f"writer process exited with {p.exitcode}"

    # The file must parse cleanly — no torn writes.
    assert target.exists()
    parsed = json.loads(target.read_text(encoding="utf-8"))

    # The last writer's payload had length == iterations (i + 1 with
    # i = iterations - 1). All four workers used the same shape so the
    # final on-disk length is deterministic regardless of which process
    # raced last.
    assert isinstance(parsed, list)
    assert len(parsed) == iterations
    last_record = parsed[-1]
    assert last_record["writer"].startswith("P")
    assert last_record["i"] == iterations - 1
    assert isinstance(last_record["flag"], bool)


# ---------------------------------------------------------------------------
# 3. Crash mid-write — original file is left intact
# ---------------------------------------------------------------------------


def test_crash_mid_write_preserves_original_file(tmp_path):
    """Simulate a crash between ``mkstemp`` and ``os.replace``.

    The previous ``open(path, 'w')`` writers truncated the target file
    before writing, so a SIGKILL during ``json.dump`` left an empty or
    half-written file. ``atomic_write_json`` must instead leave the
    pre-existing content intact when the in-flight write fails.
    """
    target = tmp_path / "activities.json"

    # Seed the target with valid pre-existing content.
    original_payload = [{"original": True, "n": 1}]
    target.write_text(json.dumps(original_payload), encoding="utf-8")

    # Inject a failure at ``os.replace`` — this is the precise window the
    # spec calls out (between ``os.fdopen`` writing the temp file and
    # the atomic rename into place).
    with mock.patch(
        "app.utils.json_utils.os.replace",
        side_effect=RuntimeError("simulated crash mid-write"),
    ):
        with pytest.raises(RuntimeError, match="simulated crash mid-write"):
            atomic_write_json(str(target), [{"new": True, "n": 99}])

    # Original content must be intact.
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == original_payload

    # Temp file must be cleaned up — no leaked ``.activities.*.tmp`` files.
    leftovers = [
        name for name in os.listdir(tmp_path) if name.startswith(".activities.")
    ]
    assert leftovers == [], f"temp file leaked: {leftovers!r}"


def test_crash_during_json_dump_preserves_original_file(tmp_path):
    """Same invariant, but the failure happens during ``json.dump`` itself.

    The encoder pukes on a non-serialisable object, so the temp file is
    written but never replaced into place. The original file must still
    be intact and the temp file must be cleaned up.
    """
    target = tmp_path / "activities.json"
    original_payload = [{"original": True}]
    target.write_text(json.dumps(original_payload), encoding="utf-8")

    class _Unserialisable:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(str(target), [{"bad": _Unserialisable()}])

    assert json.loads(target.read_text(encoding="utf-8")) == original_payload
    leftovers = [
        name for name in os.listdir(tmp_path) if name.startswith(".activities.")
    ]
    assert leftovers == [], f"temp file leaked: {leftovers!r}"
