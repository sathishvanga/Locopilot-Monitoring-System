"""Unit tests for app.services.dedup_service.

Covers: TTL behavior, key separation, eviction at capacity, and the
sha256 helpers used by the controller.
"""

import time

import pytest

from app.services.dedup_service import (
    VideoResultCache,
    sha256_of_bytes,
    sha256_of_file,
)


def test_sha256_of_bytes_is_stable():
    assert sha256_of_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_of_bytes(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_of_file_matches_bytes(tmp_path):
    payload = b"locopilot dedup smoke test"
    f = tmp_path / "video.mp4"
    f.write_bytes(payload)
    assert sha256_of_file(str(f)) == sha256_of_bytes(payload)


def test_cache_hit_returns_same_value():
    c = VideoResultCache(ttl_seconds=60)
    c.put("abc", "trip1", {"result": "ok"})
    assert c.get("abc", "trip1") == {"result": "ok"}


def test_cache_miss_returns_none():
    c = VideoResultCache(ttl_seconds=60)
    assert c.get("abc", "trip1") is None


def test_different_trip_id_does_not_hit():
    """Same content under a different trip must NOT cache-hit.

    Production reasoning: trips have different crew metadata, external API
    targets, and operational context — sharing pipeline results across them
    would post the wrong evidence to the wrong customer.
    """
    c = VideoResultCache(ttl_seconds=60)
    c.put("abc", "trip1", {"result": "ok"})
    assert c.get("abc", "trip2") is None


def test_ttl_expiry_evicts():
    c = VideoResultCache(ttl_seconds=0.05)  # 50ms TTL
    c.put("abc", "trip1", {"result": "ok"})
    assert c.get("abc", "trip1") == {"result": "ok"}
    time.sleep(0.07)
    assert c.get("abc", "trip1") is None


def test_ttl_zero_disables_cache():
    c = VideoResultCache(ttl_seconds=0)
    c.put("abc", "trip1", {"result": "ok"})
    assert c.get("abc", "trip1") is None


def test_cache_max_entries_evicts_oldest():
    c = VideoResultCache(ttl_seconds=3600, max_entries=3)
    c.put("h1", "t", {"i": 1})
    time.sleep(0.001)
    c.put("h2", "t", {"i": 2})
    time.sleep(0.001)
    c.put("h3", "t", {"i": 3})
    time.sleep(0.001)
    # Adding a 4th should evict the oldest (h1).
    c.put("h4", "t", {"i": 4})
    assert c.get("h1", "t") is None
    assert c.get("h4", "t") == {"i": 4}


def test_stats_reports_size_and_ttl():
    c = VideoResultCache(ttl_seconds=42.0, max_entries=10)
    c.put("h", "t", {})
    s = c.stats()
    assert s["entries"] == 1
    assert s["ttl_seconds"] == 42.0
    assert s["max_entries"] == 10
