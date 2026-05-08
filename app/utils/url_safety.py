"""
URL safety helpers for external resource fetches.

Task 0008 — Lock down ``videoUrl`` SSRF surface.

The ``/api/video/analyze`` endpoint accepts a ``videoUrl`` form field that
is fetched server-side. (``/api/v1/video/process-and-upload`` only accepts a
multipart ``video_file`` — it has no ``videoUrl`` parameter.) Without
validation, an attacker can ask the GPU box to fetch arbitrary URLs:

  * AWS / GCP cloud-metadata IPs (``169.254.169.254``,
    ``metadata.google.internal``) leaking IAM credentials.
  * Intranet hosts (RFC1918, loopback) for reconnaissance.
  * ``file://`` URIs to read local files (depending on the underlying client).

``validate_external_url`` enforces:

  1. Scheme allowlist (``http`` / ``https`` only).
  2. Hostname allowlist (set in ``Settings.minio_allowed_hosts``).
  3. DNS resolution check — every resolved IP must NOT be loopback,
     private (RFC1918 + CGNAT + ``0.0.0.0/8``), link-local (incl. AWS/GCP
     metadata IP), multicast, broadcast, reserved, or unspecified, in
     either IPv4 or IPv6 (with IPv4-mapped IPv6 folded to IPv4 first).

A pure pre-check is still vulnerable to DNS rebinding (host resolves to a
public IP at validation, private IP at fetch). For the current single-trusted-
host configuration that is acceptable; full mitigation (resolve once, then
connect by IP with a forced ``Host:`` header) is tracked separately.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlparse


class URLNotAllowed(ValueError):
    """Raised when a candidate URL fails the SSRF allowlist checks."""


_ALLOWED_SCHEMES = frozenset({"http", "https"})

# CGNAT (carrier-grade NAT) — RFC 6598. ipaddress.is_private only learned
# this range in CPython 3.13 (bpo-105631), so we check it explicitly to
# guarantee coverage on 3.11 / 3.12.
_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """Return True if ``ip`` is unsafe for server-side fetches.

    Reviewer finding H1+H2: the previous hand-rolled list-of-networks
    walked a small subset (RFC1918, ``127.0.0.0/8``, ``169.254/16``,
    ``::1/128``, ``fc00::/7``, ``fe80::/10``) and short-circuited on
    ``ip.version != net.version``, so an IPv4-mapped IPv6 literal like
    ``::ffff:127.0.0.1`` or ``::ffff:169.254.169.254`` slipped through
    every check. The list also missed ``0.0.0.0/8`` (loopback alias on
    Linux/macOS), ``100.64/10`` (CGNAT), broadcast (``255.255.255.255``),
    multicast (``224.0.0.0/4``), and other reserved ranges.

    Fix: normalize IPv4-mapped IPv6 to its embedded IPv4 first, then defer
    to the ``ipaddress`` built-in predicates which already cover loopback,
    private (RFC1918 + CGNAT + ``0.0.0.0/8``), link-local (incl. cloud
    metadata IP ``169.254.169.254``), multicast, reserved (incl.
    ``240.0.0.0/4`` and broadcast), and unspecified (``0.0.0.0`` /
    ``::``) for both v4 and v6.
    """
    # Fold IPv4-mapped IPv6 (``::ffff:1.2.3.4``) to IPv4 first; otherwise
    # the IPv6 predicates ``is_loopback`` / ``is_private`` would not flag
    # IPv4 loopback or RFC1918 addresses tunneled through IPv6.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # Python 3.11's ``is_private`` does NOT include 100.64/10 (CGNAT) — that
    # was only fixed in CPython 3.13 (bpo-105631). Check it explicitly so the
    # gap doesn't widen on older interpreters.
    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NET:
        return True
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_external_url(url: str, allowed_hosts: Iterable[str]) -> str:
    """Validate that ``url`` is safe to fetch server-side.

    Args:
        url: Candidate external URL.
        allowed_hosts: Iterable of hostnames the caller deems safe (matched
            case-insensitively against ``urlparse(url).hostname``).

    Returns:
        The original ``url`` string when all checks pass.

    Raises:
        URLNotAllowed: If the URL fails any check (bad scheme, host not in
            allowlist, DNS resolution fails, or any resolved IP is in a
            blocked range).
    """
    if not isinstance(url, str) or not url.strip():
        raise URLNotAllowed("empty url")

    parsed = urlparse(url.strip())

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise URLNotAllowed(f"scheme not allowed: {scheme!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise URLNotAllowed("missing host")

    allow_lc = {h.strip().lower() for h in allowed_hosts if h and h.strip()}
    if host not in allow_lc:
        raise URLNotAllowed(f"host not in allowlist: {host}")

    # DNS rebind / metadata-IP defense: resolve and re-check every returned
    # address. ``getaddrinfo`` returns one tuple per (family, socktype,
    # proto, canonname, sockaddr); we only care about the address.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise URLNotAllowed(f"DNS resolution failed for {host!r}: {e}") from e

    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            # Skip un-parseable scope-id'd IPv6 etc.; if any other resolved
            # address is OK we still pass.
            continue
        if _is_blocked_ip(ip):
            raise URLNotAllowed(
                f"host {host!r} resolved to private/loopback range: {ip}"
            )

    return url
