# Task 0010 — Standardize logging + redact `Authorization` + opaque 5xx detail

**Severity:** HIGH (security + maintainability)
**Source:** `docs/code-review-2026-05-08.md` cross-cutting themes #6 + #7, top-fix #10.
**Estimated effort:** Half day.

---

## Problem

### 1. Three parallel logging setups, drifted

- `app/utils/logger.py` — the canonical setup with `TimedRotatingFileHandler` + `RequestFormatter`.
- `locopilot_monitor.py:160-191` — `_setup_module_logger`, plain `FileHandler`, no rotation, writes to the same `logs/LocopilotMonitoring.log`. Result: duplicate log lines + the file doesn't rotate cleanly (two handlers hold open the same path).
- `app/services/yolo_pose_adapter.py:25-44` — yet another roll-your-own with hardcoded format string.
- `app/core/activity_tracker.py:17-36` and `app/core/models/yolo_handler.py:48-67` — two more `_setup_module_logger` definitions, identical bodies.

The `TimedRotatingFileHandler` dedup check in `logger.py:140-144` only catches its own type — the plain `FileHandler` from the legacy setup slips through and accumulates one handler per logger creation.

Inconsistent logger acquisition (`logging.getLogger(__name__)` vs `get_logger(__name__)`) means a future "strip secrets" filter applied to the project logger won't catch every call site.

### 2. Bearer tokens in request context

`app/middleware/logging_middleware.py:47, 58` captures the full `Authorization` header value into the per-request context dict. Every later log line that interpolates context fields writes the bearer to disk.

### 3. Internal error detail leaks

`app/controllers/video_controller.py:1021-1032` and several other handlers raise `HTTPException(500, detail=f"Failed to process video: {str(e)}")`. The `app/main.py:308-315` http-exception handler does sanitize the *response* for 5xx, but the *log* still includes raw `exc.errors()` which can include the bad input value (e.g., a token in a malformed header).

### 4. Emojis in log messages defeat grep

`external_api_service.py` (`📤📦✅❌⚠️`), `logging_middleware.py` (`📥📤💥`), `gunicorn_config.py` (`✅`), `start_server.sh` (`✅`), various `[OK]` lines in `video_controller.py`. CLAUDE.md says: "No emojis in commits or code; logs use bracketed prefixes."

---

## Files to change

- `locopilot_monitor.py:160-191` — delete `_setup_module_logger`; switch to `from app.utils.logger import get_logger`.
- `app/services/yolo_pose_adapter.py:25-44` — same.
- `app/core/activity_tracker.py:17-36` — same.
- `app/core/models/yolo_handler.py:48-67` — same.
- `app/middleware/logging_middleware.py:47, 58` — redact `Authorization`.
- `app/main.py:308-345` — opaque 5xx + scrubbed validation log.
- `app/controllers/video_controller.py` — replace all `HTTPException(500, str(e))` with opaque `internal_error`.
- All emoji log strings.

---

## Fix

### One canonical `get_logger`

Keep `app/utils/logger.py` as the only source. Delete every other `_setup_module_logger`. Every module starts with:

```python
from app.utils.logger import get_logger
logger = get_logger(__name__)
```

The dedup check in `logger.py:140-144` should be tightened to "any handler with `baseFilename` pointing at the project log file" (not type-keyed):

```python
if any(getattr(h, "baseFilename", None) == log_path for h in logger.handlers):
    return logger
```

### Sensitive-field filter

Add a `logging.Filter` that strips Authorization-like fields from `LogRecord.args` and from any `extra` dict:

```python
SENSITIVE = {"authorization", "auth_token", "bearer", "x-api-key",
             "secret_key", "password", "minio_secret_key"}

class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Redact extras
        for key in list(getattr(record, "__dict__", {}).keys()):
            if key.lower() in SENSITIVE:
                setattr(record, key, "***")
        # Best-effort scan of msg+args
        if isinstance(record.msg, str):
            for term in SENSITIVE:
                if term in record.msg.lower():
                    record.msg = record.msg.replace(term, term + "=***")
        return True
```

Install on the root logger in `setup_logging`.

### Middleware redact

```python
auth_header = request.headers.get("authorization", "")
context["authorization"] = "***" if auth_header else "None"
```

Never store the raw value into context.

### Opaque 5xx

```python
# In every handler that currently does HTTPException(500, str(e)):
logger.exception("processing failed for trip=%s", trip_id)
raise HTTPException(status_code=500, detail="internal_error")
```

In `app/main.py:341` `RequestValidationError` handler, log a sanitized version of `exc.errors()` (drop `input` key) — the raw dict can contain credentials.

### Emoji removal

Replace each occurrence:

| Emoji | Replacement |
|-------|-------------|
| 📥 | `[REQ]` |
| 📤 | `[RES]` or `[external_api]` |
| 💥 | `[ERR]` |
| ✅ | `[OK]` |
| ❌ | `[FAIL]` |
| ⚠️ | `[WARN]` |
| 📦 | `[PAYLOAD]` |
| 🚂 | `[TRAIN]` |

---

## Acceptance criteria

1. `grep -rn "_setup_module_logger\|FileHandler(" app/ locopilot_monitor.py | grep -v "app/utils/logger.py"` returns zero hits.
2. Tail `logs/LocopilotMonitoring.log` after a request that includes `Authorization: Bearer abc123`. The string `abc123` does NOT appear in any log line.
3. `tests/middleware/test_logging_redact.py`:
   - Asserts `request.headers["authorization"]` is never written to disk via the formatter.
   - Asserts `extra={"authorization": "Bearer foo"}` becomes `***` in the rendered log line.
4. `grep -rEn "[😀-🙏]|[🌀-🗿]|[🚀-🛿]|[🤀-🧿]|[✅❌⚠️📥📤📦💥🚂]" app/ locopilot_monitor.py *.sh *.py` returns zero hits in non-test code.
5. Posting an invalid request that triggers a 500 returns body `{"detail": "internal_error"}` (no `str(e)` content). The log line for the same request includes the full traceback + `trip_id` for debugging.

---

## Out of scope

- Migrating to structured JSON logs (separate task).
- Adding distributed tracing.
