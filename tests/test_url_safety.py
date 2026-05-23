"""Unit tests for ``app.utils.url_safety.validate_external_url`` (Task 0008).

The acceptance criteria from
``docs/specs/code-review-fixes/tasks/0008-lock-down-ssrf-surface.md`` say the
helper must reject:

  * cloud-metadata link-local IP (``http://169.254.169.254/...``)
  * loopback (``http://localhost:8000/``)
  * RFC1918 (``http://192.168.1.1/x.mp4``)
  * non-http schemes (``file:///etc/passwd``)
  * any host not in the allowlist (``https://example.com/x.mp4``)

and accept the single allow-listed host:

  * ``https://mind.snikbtel.uk:9000/bucket/x.mp4`` -> returns the URL.

The DNS lookup for the allowed host is patched out so the test suite stays
hermetic and never touches the network.
"""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest

# Same convention as ``tests/controllers/test_video_controller_auth.py``:
# avoid Settings tripping on dev-machine .env coupling.
os.environ.setdefault("TRAIN_MOTION_DETECTION_ENABLED", "1")
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")

from app.utils.url_safety import URLNotAllowed, validate_external_url


ALLOWED = {"mind.snikbtel.uk"}


def _fake_resolve_to(public_ip: str):
    """Return a ``getaddrinfo`` patch that resolves any host to ``public_ip``."""
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (public_ip, 0))]
    return fake


# ----- Reject cases ---------------------------------------------------------

def test_rejects_aws_metadata_ip():
    """AWS / GCP cloud metadata IP must be rejected."""
    with pytest.raises(URLNotAllowed):
        validate_external_url(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            ALLOWED,
        )


def test_rejects_localhost():
    """Loopback hostname must be rejected even if scheme is fine."""
    with pytest.raises(URLNotAllowed):
        validate_external_url("http://localhost:8000/x.mp4", ALLOWED)


def test_rejects_rfc1918_ip():
    """RFC1918 private IP literal must be rejected."""
    with pytest.raises(URLNotAllowed):
        validate_external_url("http://192.168.1.1/x.mp4", ALLOWED)


def test_rejects_file_scheme():
    """Non-http(s) schemes (file://) must be rejected outright."""
    with pytest.raises(URLNotAllowed):
        validate_external_url("file:///etc/passwd", ALLOWED)


def test_rejects_off_allowlist_host():
    """A perfectly-public host that isn't in the allowlist is still rejected."""
    with pytest.raises(URLNotAllowed):
        validate_external_url("https://example.com/x.mp4", ALLOWED)


def test_rejects_allowed_host_resolving_to_private_ip():
    """DNS rebind defense: even an allow-listed host must not resolve into
    a blocked range. Simulates a hostile DNS answer pointing at 127.0.0.1."""
    with patch(
        "app.utils.url_safety.socket.getaddrinfo",
        side_effect=_fake_resolve_to("127.0.0.1"),
    ):
        with pytest.raises(URLNotAllowed):
            validate_external_url(
                "https://mind.snikbtel.uk:9000/cvss/x.mp4", ALLOWED
            )


def test_rejects_dns_failure():
    """getaddrinfo failures (NXDOMAIN, no network) surface as URLNotAllowed."""
    with patch(
        "app.utils.url_safety.socket.getaddrinfo",
        side_effect=socket.gaierror("nope"),
    ):
        with pytest.raises(URLNotAllowed):
            validate_external_url(
                "https://mind.snikbtel.uk:9000/cvss/x.mp4", ALLOWED
            )


def test_rejects_empty_url():
    with pytest.raises(URLNotAllowed):
        validate_external_url("", ALLOWED)


def test_rejects_missing_host():
    with pytest.raises(URLNotAllowed):
        validate_external_url("http:///path", ALLOWED)


# ----- H1+H2: IPv4-mapped IPv6 + extended blocked ranges -------------------
#
# Reviewer findings H1 (IPv4-mapped IPv6 bypass) and H2 (missing 0.0.0.0/8,
# 100.64/10, broadcast, multicast). These tests pin the new behavior of
# ``_is_blocked_ip`` (now built on the ``ipaddress`` module's predicates with
# an explicit IPv4-mapped IPv6 fold). Each resolved IP is patched in via
# ``getaddrinfo`` so the test stays hermetic.

@pytest.mark.parametrize(
    "resolved_ip,family",
    [
        # IPv4-mapped IPv6 forms — the previous code skipped these because of
        # the ``ip.version != net.version`` short-circuit.
        ("::ffff:127.0.0.1", socket.AF_INET6),
        ("::ffff:169.254.169.254", socket.AF_INET6),
        # Ranges the old hand-rolled list omitted entirely.
        ("0.0.0.0", socket.AF_INET),            # unspecified / loopback alias
        ("100.64.0.1", socket.AF_INET),         # CGNAT
        ("255.255.255.255", socket.AF_INET),    # limited broadcast (reserved)
        ("224.0.0.1", socket.AF_INET),          # multicast
    ],
)
def test_rejects_extended_blocked_ranges(resolved_ip, family):
    """Allow-listed host that resolves into any newly-blocked range is rejected."""
    def fake(host, port, *args, **kwargs):
        return [(family, socket.SOCK_STREAM, 0, "", (resolved_ip, 0))]

    with patch("app.utils.url_safety.socket.getaddrinfo", side_effect=fake):
        with pytest.raises(URLNotAllowed):
            validate_external_url(
                "https://mind.snikbtel.uk:9000/cvss/x.mp4", ALLOWED
            )


# ----- Accept case ----------------------------------------------------------

def test_accepts_allowed_host_resolving_publicly():
    """Allow-listed host that resolves to a public IP returns the URL unchanged."""
    url = "https://mind.snikbtel.uk:9000/cvss/x.mp4"
    # Use a globally-routable IP. After H1+H2 the predicate now leans on
    # ``ipaddress.IPv4Address.is_private`` which (in Python >=3.4) ALSO
    # treats TEST-NET ranges (``203.0.113.0/24`` etc.) as private — so we
    # cannot use those as the "public" stand-in like the original test did.
    with patch(
        "app.utils.url_safety.socket.getaddrinfo",
        side_effect=_fake_resolve_to("8.8.8.8"),  # globally routable
    ):
        assert validate_external_url(url, ALLOWED) == url


def test_allowlist_match_is_case_insensitive():
    """Hostnames are normalized to lowercase before allowlist comparison."""
    url = "https://MIND.SNIKBTEL.UK/x.mp4"
    with patch(
        "app.utils.url_safety.socket.getaddrinfo",
        side_effect=_fake_resolve_to("8.8.8.8"),
    ):
        assert validate_external_url(url, ALLOWED) == url
