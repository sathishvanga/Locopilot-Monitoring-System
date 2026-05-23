"""
Dead-Letter Queue for the external CVVR API.

When the external violation sink (cvvr/cvvrTripViolations/addUpdateBulk) is
unreachable after the configured retry budget, the entire payload is written
to a JSON file under the per-run DLQ directory rather than being silently
dropped. A startup re-drain task replays these files on the next boot so a
transient mindcoinapps outage cannot lose a trip's worth of violations.

Storage format
--------------
DLQ files live under the run directory, not a global drop:

    <run_dir>/_failed_external_api/<trip>_<ts>_<uuid8>.json

This per-run scoping (spec 0004) keeps a failed run's evidence clips, its
activities.json, and its undelivered API payload colocated — useful when
a developer is triaging a single bad run, and avoids tangling drops from
unrelated trips. The ``<uuid8>`` suffix removes the millisecond-collision
risk that bit us under multi-worker load.

Each record has the shape::

    {
        "url": "https://.../cvvr/cvvrTripViolations/addUpdateBulk",
        "payload": [...],                      # JSON-serialized request body
        "headers": {"Content-Type": "..."},    # Authorization stripped
        "idempotency_key": "TRIP-A:abc123...", # spec format
        "context_label": "Violations for trip_id=...",
        "timeout": 30,
        "trip_id": "TV22_10",
        "first_attempt_at": "2026-05-08T12:34:56",
        "attempts": 3,
        "last_error": "503 Service Unavailable"
    }

Security
--------
``Authorization`` (and any case variant of it) is stripped before write so a
captured DLQ file never leaks the bearer token. The idempotency key is kept
because the customer's API uses it for de-duplication when we eventually
re-post during drain — losing it would re-introduce duplicates.

Capacity safeguards
-------------------
``MAX_DLQ_FILES_PER_DIR`` caps how many DLQ files may accumulate in a single
run directory. Past that, ``write_dlq`` evicts the oldest by mtime so a
runaway loop can't fill the disk. A free-space precheck refuses to write at
all when the target filesystem has less than 1 GiB available — better to
keep the activity in memory than to brick the host.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.config import get_settings
from ..utils.logger import get_logger


logger = get_logger(__name__)


# Header keys we never persist to disk. Compared case-insensitively.
_SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})

# Lock serializing writes/reads against the DLQ directory.
_dlq_lock = threading.Lock()

# Per-directory file cap. Past this, ``write_dlq`` evicts the oldest by mtime
# so a runaway failure loop can't fill the disk.
MAX_DLQ_FILES_PER_DIR = 1000

# Refuse to write a new DLQ record when the target filesystem has less than
# this many free bytes available. The caller is expected to keep the activity
# in memory and decide what to do — better than bricking the host.
_MIN_FREE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


def _run_dlq_dir(run_dir: str) -> str:
    """Return the per-run DLQ directory, creating it on first use.

    Spec 0004 places each run's DLQ files under
    ``<run_dir>/_failed_external_api/`` so a failed run's evidence clips,
    activities.json, and undelivered payload stay colocated.
    """
    base = os.path.join(run_dir, "_failed_external_api")
    os.makedirs(base, exist_ok=True)
    return base


def _strip_sensitive_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Drop Authorization-class headers so DLQ files never leak credentials."""
    if not headers:
        return {}
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _SENSITIVE_HEADERS
    }


def _safe_filename(prefix: str) -> str:
    """Return a filesystem-safe filename — alphanum, dash, dot, underscore."""
    return "".join(c if c.isalnum() or c in ("-", ".", "_") else "_" for c in prefix)


def _free_bytes(path: str) -> Optional[int]:
    """Return free bytes on the filesystem hosting ``path``, or None on error.

    Wrapped in a try/except because ``statvfs`` is POSIX-only and we want the
    DLQ writer to work on any platform — the precheck just falls back to
    "no opinion" rather than refusing to write.
    """
    try:
        st = os.statvfs(path)
        return int(st.f_bavail) * int(st.f_frsize)
    except (AttributeError, OSError):
        return None


def _evict_oldest_if_over_cap(directory: str) -> None:
    """If the directory holds >= MAX_DLQ_FILES_PER_DIR JSON files, drop the
    oldest by mtime until the count is back under the cap.

    Logs a WARN per eviction so the operator notices a runaway producer.
    """
    try:
        entries = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".json")
        ]
    except FileNotFoundError:
        return
    if len(entries) < MAX_DLQ_FILES_PER_DIR:
        return

    # Stat each file once and sort by mtime ascending (oldest first).
    by_mtime: List[Tuple[float, str]] = []
    for p in entries:
        try:
            by_mtime.append((os.path.getmtime(p), p))
        except OSError:
            continue
    by_mtime.sort()

    overflow = len(by_mtime) - (MAX_DLQ_FILES_PER_DIR - 1)
    for _, victim in by_mtime[:overflow]:
        try:
            os.remove(victim)
            logger.warning(
                f"[dlq] Evicted oldest DLQ file due to per-dir cap "
                f"(MAX_DLQ_FILES_PER_DIR={MAX_DLQ_FILES_PER_DIR}): {victim}"
            )
        except OSError as e:
            logger.error(f"[dlq] Failed to evict {victim}: {e}")


def write_dlq(
    *,
    run_dir: str,
    url: str,
    payload: Any,
    headers: Optional[Dict[str, str]],
    idempotency_key: str,
    context_label: str,
    timeout: int,
    trip_id: str,
    last_error: Optional[str] = None,
    attempts: int = 0,
) -> Optional[str]:
    """Persist a failed external-API request to the per-run DLQ.

    Returns the absolute path of the file written, or ``None`` if the write
    itself failed (in which case the error is logged — DLQ failures are
    never raised so they cannot mask the original API failure).

    ``run_dir`` scopes the DLQ to the current run as required by spec 0004.

    Authorization-class headers are stripped before serialization so a
    captured DLQ file cannot leak the customer's bearer token.

    Capacity guards:
      * if the target filesystem has less than 1 GiB free, refuse to write
        and log ERROR — caller can keep the activity in memory.
      * if the directory already holds ``MAX_DLQ_FILES_PER_DIR`` records,
        evict oldest-by-mtime until under the cap.
    """
    if not run_dir:
        logger.error("[dlq] write_dlq called with empty run_dir; refusing to write")
        return None

    record: Dict[str, Any] = {
        "url": url,
        "payload": payload,
        "headers": _strip_sensitive_headers(headers),
        "idempotency_key": idempotency_key,
        "context_label": context_label,
        "timeout": int(timeout),
        "trip_id": trip_id,
        "first_attempt_at": datetime.utcnow().isoformat(timespec="seconds"),
        "attempts": int(attempts),
        "last_error": last_error,
    }

    try:
        directory = _run_dlq_dir(run_dir)
    except Exception as e:  # pragma: no cover - cannot create dir
        logger.error(f"[dlq] Failed to resolve DLQ directory under {run_dir!r}: {e}", exc_info=True)
        return None

    # Soft free-disk-space gate. Anything under 1 GiB and we'd rather lose
    # the activity in memory than wedge the host.
    free = _free_bytes(directory)
    if free is not None and free < _MIN_FREE_BYTES:
        logger.error(
            f"[dlq] Refusing DLQ write: free space {free} bytes on {directory} "
            f"is below floor {_MIN_FREE_BYTES} (1 GiB). "
            f"trip_id={trip_id} attempts={attempts}"
        )
        return None

    # Per-directory eviction cap so a runaway loop can't fill the disk.
    _evict_oldest_if_over_cap(directory)

    # ``<trip>_<ts>_<uuid8>.json`` per spec; uuid suffix removes the
    # millisecond-collision risk under multi-worker load.
    suffix = uuid.uuid4().hex[:8]
    fname = _safe_filename(f"{trip_id or 'no_trip'}_{int(time.time())}_{suffix}.json")
    path = os.path.join(directory, fname)

    try:
        with _dlq_lock:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, default=str)
            os.replace(tmp_path, path)
        logger.warning(
            f"[dlq] Wrote DLQ record for trip_id={trip_id} attempts={attempts} "
            f"path={path}"
        )
        return path
    except Exception as e:
        logger.error(f"[dlq] Failed to write DLQ file: {e}", exc_info=True)
        return None


def _list_all_dlq_files() -> List[str]:
    """Return every DLQ JSON file under ``<output_dir>/run_*/_failed_external_api/``.

    Files are sorted by path so the drain order is deterministic across runs.
    """
    settings = get_settings()
    output_dir = settings.output_dir
    pattern = os.path.join(output_dir, "run_*", "_failed_external_api", "*.json")
    try:
        return sorted(glob.glob(pattern))
    except Exception as e:
        logger.error(f"[dlq] Failed to list DLQ files under {output_dir}: {e}")
        return []


def _load_record(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.error(f"[dlq] Failed to read DLQ file {path}: {e}")
        return None


def _bump_attempts(path: str, record: Dict[str, Any], last_error: Optional[str]) -> None:
    """Persist an incremented attempt counter back to the same DLQ file."""
    record["attempts"] = int(record.get("attempts", 0)) + 1
    record["last_error"] = last_error
    record["last_attempt_at"] = datetime.utcnow().isoformat(timespec="seconds")
    try:
        with _dlq_lock:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, default=str)
            os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"[dlq] Failed to bump attempt count for {path}: {e}")


def _delete_record(path: str) -> None:
    try:
        with _dlq_lock:
            os.remove(path)
    except Exception as e:
        logger.error(f"[dlq] Failed to remove DLQ file {path}: {e}")


# A repost callable takes the loaded record and returns ``(success, error)``.
RepostFn = Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]


def drain_dlq(repost: Optional[RepostFn] = None) -> Dict[str, int]:
    """Re-attempt every DLQ record once and report stats.

    Walks ``<output_dir>/run_*/_failed_external_api/*.json`` (per-run scoping
    per spec 0004). For each file:
      * call ``repost(record)`` (default: a real ``requests.post`` to the
        recorded URL with the recorded payload + idempotency key);
      * on success — delete the file;
      * on failure — keep the file and increment ``attempts``.

    Returns counters: ``{"drained": N, "succeeded": N, "failed": N}``.
    Errors during a single record's drain never abort the loop.
    """
    if repost is None:
        repost = _default_repost

    paths = _list_all_dlq_files()
    stats = {"drained": 0, "succeeded": 0, "failed": 0}

    for path in paths:
        record = _load_record(path)
        if record is None:
            stats["failed"] += 1
            continue
        stats["drained"] += 1

        try:
            success, err = repost(record)
        except Exception as e:
            logger.error(f"[dlq] Drain raised on {path}: {e}", exc_info=True)
            success, err = False, str(e)

        if success:
            _delete_record(path)
            stats["succeeded"] += 1
            logger.info(
                f"[dlq] Drained record path={path} "
                f"trip_id={record.get('trip_id')} after {record.get('attempts', 0) + 1} attempts"
            )
        else:
            _bump_attempts(path, record, err)
            stats["failed"] += 1
            logger.warning(
                f"[dlq] Drain failed for path={path} trip_id={record.get('trip_id')} "
                f"error={err}; record retained"
            )

    if stats["drained"]:
        logger.info(
            f"[dlq] drain complete drained={stats['drained']} "
            f"succeeded={stats['succeeded']} failed={stats['failed']}"
        )
    return stats


def _default_repost(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Default re-post implementation used by ``drain_dlq``.

    Imports ``requests`` lazily so unit tests can pass a fake repost without
    requiring the optional dependency. Reuses the original idempotency key
    so the customer's API will dedupe if it has already accepted the work.
    """
    import requests  # local import keeps top-level imports light

    settings = get_settings()
    url = record.get("url")
    payload = record.get("payload")
    headers = dict(record.get("headers") or {})
    idempotency_key = record.get("idempotency_key")
    timeout = int(record.get("timeout") or settings.cvvr_api_timeout)

    # Re-attach Authorization at drain time (it was stripped at write time).
    if settings.cvvr_api_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {settings.cvvr_api_token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if resp.status_code in (200, 201):
        return True, None
    return False, f"status={resp.status_code} body={resp.text[:200]}"
