# Task 0004 — DLQ + idempotency key on the external mindcoinapps API

**Severity:** CRITICAL
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #5, top-fix #4.
**Estimated effort:** Half day.

---

## Problem

`app/services/external_api_service.py` is the single point that posts confirmed violations to the customer's `cvvr/cvvrTripViolations/addUpdateBulk` endpoint. Two correctness gaps:

### 1. Silent data loss on outage (no DLQ)

`external_api_service.py:351-400`. If 5xx persists past 3 retries, the call returns `{"success": False, ...}` and the caller has no recovery mechanism. **A whole trip's violations are silently lost on a 5-minute upstream hiccup.** The log line literally says "Consider dead-letter queue recovery" — but nothing actually persists the payload.

### 2. Duplicate violations on retry (no idempotency)

`external_api_service.py:91-103`. Retries on `429/500/502/503/504`. If the server **accepted and committed** the payload but the response was lost (timeout, 502 from a load balancer), the retry creates duplicate violations. The local `_deduplicate_violations` (line 614-618) only protects against same-batch dups, not retry-induced dups across requests.

`_deduplicate_violations` itself is broken: keys use `startTime` string verbatim, so `"6.00"`, `"6.0"`, and `"00:00:06"` dedupe as different keys (review finding H17).

---

## Files to change

- `app/services/external_api_service.py:91-120, 351-400, 614-618`
- `app/services/dlq.py` — **NEW**
- `app/main.py` lifespan — startup re-drain hook

---

## Fix

### Idempotency key

In `_request_with_retry`, compute `idempotency_key` ONCE per logical call (before the retry loop):

```python
import hashlib

def _idempotency_key(trip_id: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{trip_id}:{hashlib.sha256(body).hexdigest()}"
```

Pass it as `Idempotency-Key` header on every retry attempt (server team must honor — open a ticket if not).

### Dead letter queue

```python
# app/services/dlq.py
from pathlib import Path
import json, time, uuid

def write_dlq(run_dir: str, trip_id: str, payload: dict, headers: dict, last_status: int) -> Path:
    dlq_dir = Path(run_dir) / "_failed_external_api"
    dlq_dir.mkdir(parents=True, exist_ok=True)
    name = f"{trip_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    path = dlq_dir / name
    record = {
        "trip_id": trip_id,
        "payload": payload,
        "headers": {k: v for k, v in headers.items() if k.lower() != "authorization"},
        "last_status": last_status,
        "created_at": time.time(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path

def drain_dlq(post_fn) -> int:
    """Iterate <output_dir>/run_*/_failed_external_api/*.json; on success, delete."""
    ...
```

In `_post_violations` after retries exhaust:

```python
if not response_ok:
    dlq_path = write_dlq(run_dir, trip_id, payload, headers, last_status)
    logger.error("[external_api] DLQ wrote %s after retries exhausted", dlq_path)
    return {"success": False, "dlq_path": str(dlq_path)}
```

### Startup re-drain

In FastAPI lifespan (`app/main.py`), spawn a background task that calls `drain_dlq(external_api_service._post_violations)` once at startup.

### Fix `_deduplicate_violations`

Normalize `startTime` via the existing `time_to_seconds` helper before hashing:

```python
key = (
    v["tripId"],
    tuple(sorted(v["violationTypes"])),
    round(time_to_seconds(v["startTime"]), 2),
)
```

---

## Acceptance criteria

1. `tests/services/test_external_api.py`:
   - Test: 3× 503 then 200 with the same `Idempotency-Key` across attempts. Assert header present on every attempt.
   - Test: 4× 503 (exhaust retries). Assert DLQ file written under `<run_dir>/_failed_external_api/` with the original payload, no `Authorization` header captured.
   - Test: `_deduplicate_violations` collapses `{"startTime": "6.00"}` and `{"startTime": "00:00:06"}` to one entry.
2. `tests/services/test_dlq.py`:
   - `drain_dlq` succeeds on first try → file deleted.
   - `drain_dlq` fails again → file retained, attempt count bumped in record.
3. Manual: kill the upstream API, run a video, confirm DLQ file present; restart service, observe re-drain log + file deletion when API recovers.

---

## Out of scope

- Persisting DLQ outside the run dir (e.g., to a queue service). Filesystem is fine for now.
- Re-architecting `external_api_service` to async (Task 0010 covers session pooling).
