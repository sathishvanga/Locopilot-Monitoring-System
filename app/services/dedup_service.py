"""Content-hash dedup cache for /api/video/* endpoints.

Single-worker (gunicorn workers=1, see C-1 in CLAUDE.md) so an in-process
dict with a re-entrant lock is sufficient. Keys on (sha256_hex, trip_id) so
the same content uploaded under a different trip still re-processes — that
preserves operational separation between customers/trips.

Motivation: the 50-video load test on 2026-05-20 showed three pairs of
duplicate uploads (CH2_0000010121900000.mp4 vs CH2_00000101219000000.mp4,
etc.) being processed in full, burning ~3 minutes of GPU each. Same trip,
same content — cached result is safe to return.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, Optional, Tuple


class VideoResultCache:
    """TTL cache of processed-video results keyed on (sha256, trip_id)."""

    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._lock = threading.Lock()
        # key -> (insert_ts, result_dict)
        self._store: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}

    def get(self, sha256_hex: str, trip_id: str) -> Optional[Dict[str, Any]]:
        if self._ttl <= 0:
            return None
        key = (sha256_hex, trip_id)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            inserted, result = entry
            if now - inserted > self._ttl:
                # Stale — evict and miss.
                self._store.pop(key, None)
                return None
            return result

    def put(self, sha256_hex: str, trip_id: str, result: Dict[str, Any]) -> None:
        if self._ttl <= 0:
            return
        key = (sha256_hex, trip_id)
        now = time.time()
        with self._lock:
            # Evict expired entries opportunistically before insert. Bounded
            # max_entries is a backstop in case TTL is set very high.
            if len(self._store) >= self._max:
                cutoff = now - self._ttl
                stale = [k for k, (ts, _) in self._store.items() if ts < cutoff]
                for k in stale:
                    self._store.pop(k, None)
                if len(self._store) >= self._max:
                    # Hard eviction: drop the oldest entry.
                    oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                    self._store.pop(oldest, None)
            self._store[key] = (now, result)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "ttl_seconds": self._ttl,
                "max_entries": self._max,
            }


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


_singleton: Optional[VideoResultCache] = None
_singleton_lock = threading.Lock()


def get_dedup_cache(ttl_seconds: Optional[float] = None) -> VideoResultCache:
    """Process-wide singleton. First caller sets the TTL; later callers reuse."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            if ttl_seconds is None:
                from app.utils.config import get_settings
                ttl_seconds = getattr(get_settings(), "video_dedup_ttl_seconds", 0.0)
            _singleton = VideoResultCache(ttl_seconds=ttl_seconds)
    return _singleton
