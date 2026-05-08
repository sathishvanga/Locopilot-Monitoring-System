"""
Unit tests for ``LoggingMiddleware`` Authorization-header redaction.

Task 0005 (rotate-secrets-scrub-source) acceptance criterion 3:
    > Mock a request with ``Authorization: Bearer secret123``, render the
    > formatted log line, assert ``"secret123"`` does NOT appear.

The middleware is exercised end-to-end via Starlette's TestClient so the
real ``dispatch`` path runs (header extraction → request_context → log
emission). A capturing log handler scoops up every record produced during
the request; the assertions check both the formatted log lines AND the
stored request-context dict.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable when pytest is invoked from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The project's logger module loads Settings, which can fail-fast on
# dev .env configurations. Mirror the env hardening other tests use.
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")


_BEARER_TOKEN = "secret123"
_AUTH_HEADER_VALUE = f"Bearer {_BEARER_TOKEN}"


@pytest.fixture
def app_with_logging_middleware():
    """Build a minimal FastAPI app wired with ``LoggingMiddleware``."""
    from fastapi import FastAPI

    from app.middleware.logging_middleware import LoggingMiddleware

    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    def _ping():
        # Reach into the request_context inside the handler so the test
        # can assert on what the middleware stored, not just on the
        # emitted log line.
        from app.utils.request_context import get_request_context
        ctx = get_request_context()
        return {
            "authorization": ctx.get("authorization"),
            "user_id": ctx.get("user_id"),
        }

    return app


@pytest.fixture
def captured_logs():
    """Attach a capturing handler to the middleware's logger.

    Returns a list of formatted log records emitted while the test
    exercises the middleware. Cleanup is automatic (handler is removed
    on teardown) so unrelated tests are not polluted.

    The handler uses the project's production ``RequestFormatter`` (from
    ``app.utils.logger``) with the same format string ``setup_logging``
    installs on the file handler, so the redaction assertions verify the
    real production formatter — not a stub formatter that would happen to
    drop the bearer token simply because it never references the
    request_context fields.
    """
    from app.utils.logger import RequestFormatter

    target_logger = logging.getLogger("app.middleware.logging_middleware")
    previous_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)

    records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            records.append(self.format(record))

    handler = _ListHandler(level=logging.DEBUG)
    # Mirror the exact format string installed by ``setup_logging`` so the
    # captured lines are byte-for-byte identical to what would be written
    # to LocopilotMonitoring.log in production.
    handler.setFormatter(
        RequestFormatter(
            "%(asctime)s [%(user_id)s] [%(cookie_id)s] [%(source_request_id)s] "
            "[%(request_id)s] [%(levelname)s] [%(name)s] [%(method)s %(url)s] %(message)s"
        )
    )
    target_logger.addHandler(handler)
    try:
        yield records
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)


def test_authorization_token_not_logged(app_with_logging_middleware, captured_logs):
    """
    Sending ``Authorization: Bearer secret123`` must not result in
    ``secret123`` appearing in any log record produced during the
    request, nor in the request_context dict the middleware stores.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app_with_logging_middleware)
    response = client.get(
        "/ping",
        headers={"Authorization": _AUTH_HEADER_VALUE},
    )

    assert response.status_code == 200
    body = response.json()

    # 1) The middleware must store a redacted sentinel ("***"), not the
    #    raw header value, when an Authorization header is present.
    assert body["authorization"] == "***", (
        f"Expected redacted '***' in request_context, got {body['authorization']!r}"
    )

    # 2) None of the formatted log lines may contain the raw token or the
    #    full Authorization header value.
    joined_logs = "\n".join(captured_logs)
    assert _BEARER_TOKEN not in joined_logs, (
        "Bearer token leaked into logs:\n" + joined_logs
    )
    assert _AUTH_HEADER_VALUE not in joined_logs, (
        "Full Authorization header leaked into logs:\n" + joined_logs
    )


def test_missing_authorization_recorded_as_none(app_with_logging_middleware, captured_logs):
    """
    When no Authorization header is sent the request_context must record
    the literal string ``"None"`` (preserves the existing log-format
    contract for downstream parsers).
    """
    from fastapi.testclient import TestClient

    client = TestClient(app_with_logging_middleware)
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json()["authorization"] == "None"


def test_authorization_value_with_special_chars_not_logged(
    app_with_logging_middleware, captured_logs
):
    """
    Token values with quotes / spaces / special characters must still be
    fully redacted — the middleware does not attempt partial masking.
    """
    from fastapi.testclient import TestClient

    funky_token = 'Bearer abc.def-ghi_jkl="quoted value" & more'
    client = TestClient(app_with_logging_middleware)
    response = client.get("/ping", headers={"Authorization": funky_token})
    assert response.status_code == 200
    assert response.json()["authorization"] == "***"

    joined_logs = "\n".join(captured_logs)
    assert "abc.def-ghi_jkl" not in joined_logs
    assert funky_token not in joined_logs
