"""HTTP client + resilience helpers for the VLM verifier.

Currently owns just the shared :class:`_CircuitBreaker` — the actual
HTTP dispatch is still inlined in :class:`VlmVerificationService` to
preserve byte-for-byte behaviour during the package split.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from ...utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker — protects vLLM during cold-starts / GC pauses
# ---------------------------------------------------------------------------
class _CircuitBreaker:
    """Simple threshold-based circuit breaker for the VLM HTTP client.

    After ``threshold`` consecutive failures the breaker opens; subsequent
    calls short-circuit to ``SKIPPED_VLM_UNAVAILABLE`` for ``cooldown_s``
    seconds, after which it resets.  A single success closes the breaker
    early.  Used by :class:`VlmVerificationService` to avoid wasting
    ``N x timeout_seconds`` per video during a vLLM cold-start window.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._fail_count = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at > self.cooldown_s:
                self._opened_at = None
                self._fail_count = 0
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._fail_count >= self.threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                logger.warning(
                    "[vlm] circuit breaker OPEN after %d consecutive failures; "
                    "skipping VLM calls for %.1fs",
                    self._fail_count, self.cooldown_s,
                )

    def record_success(self) -> None:
        with self._lock:
            if self._fail_count > 0 or self._opened_at is not None:
                logger.info("[vlm] circuit breaker CLOSED (success after failures)")
            self._fail_count = 0
            self._opened_at = None

    def reset(self) -> None:
        with self._lock:
            self._fail_count = 0
            self._opened_at = None
