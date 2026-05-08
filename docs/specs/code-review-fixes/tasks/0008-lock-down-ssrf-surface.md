# Task 0008 — Lock down `videoUrl` SSRF surface + upload validation

**Severity:** CRITICAL (security)
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #5, top-fix #8.
**Estimated effort:** 2 hours.

---

## Problem

`/api/v1/video/process-and-upload` (and the analyze path) accepts a `videoUrl` Form field that is fetched server-side. The validation is currently only:

> `videoUrl` "starts with http:// or https://"

`app/controllers/video_controller.py:239-329`.

The URL is passed straight into `minio_service.parse_minio_url` + `download_video`. Failure modes:

1. **Open server-side fetch** — anyone who can reach the endpoint can ask the GPU box to fetch any HTTP/S URL.
2. **Cloud metadata exfiltration** — `http://169.254.169.254/latest/meta-data/` (AWS) or `http://metadata.google.internal/` reveals IAM credentials, instance metadata.
3. **Intranet reconnaissance** — `http://10.0.0.X/`, `http://192.168.X.X/`, `http://localhost:8000/` from the GPU server's perspective.
4. **Local file scheme drift** — depending on underlying client behavior, `file://` may be honored.
5. **No download size cap** — a hostile URL can return a 100GB stream and exhaust disk.

Additionally, the production endpoint `/api/v1/video/process-and-upload` performs **zero file-size validation** before reading: only the filename extension is checked at line 702 (review C3). DoS by upload.

---

## Files to change

- `app/utils/url_safety.py` — **NEW**
- `app/controllers/video_controller.py:239-329, 692-713` — call `validate_external_url` and `validate_video_file`
- `app/utils/config.py` — add `minio_allowed_hosts` setting
- `app/services/minio_service.py:download_video` — enforce size cap on stream

---

## Fix

### URL safety helper

```python
# app/utils/url_safety.py
import ipaddress
import socket
from urllib.parse import urlparse

class URLNotAllowed(ValueError): ...

_PRIVATE_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

def validate_external_url(url: str, allowed_hosts: set[str]) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise URLNotAllowed(f"scheme not allowed: {parsed.scheme}")
    host = parsed.hostname or ""
    if host not in allowed_hosts:
        raise URLNotAllowed(f"host not in allowlist: {host}")
    # DNS rebind defense: resolve and re-check
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise URLNotAllowed(f"DNS resolution failed: {e}")
    for family, _, _, _, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for net in _PRIVATE_NETS:
            if ip in net:
                raise URLNotAllowed(f"resolved to private/loopback range: {ip}")
    return url
```

### Settings

```python
minio_allowed_hosts: list[str] = Field(
    default_factory=lambda: ["mind.snikbtel.uk"],
    description="Hostnames allowed for videoUrl downloads",
)
max_external_download_bytes: int = 5 * 1024**3  # 5 GiB
```

### Controller

In both `/api/v1/video/process-and-upload` and the analyze path:

```python
from app.utils.url_safety import validate_external_url, URLNotAllowed

if videoUrl:
    try:
        validate_external_url(videoUrl, set(settings.minio_allowed_hosts))
    except URLNotAllowed as e:
        raise HTTPException(status_code=400, detail=f"videoUrl rejected: {e}")
```

For uploaded files (`process-and-upload`), reject `Content-Length > settings.max_upload_size` with 413 BEFORE reading any bytes:

```python
content_length = int(request.headers.get("content-length", "0"))
if content_length and content_length > settings.max_upload_size:
    raise HTTPException(status_code=413, detail="upload exceeds max size")
```

After streaming to disk, call `video_processing_service.validate_video_file(filename, file_size)`.

### Stream cap in download

In `MinioService.download_video`, accumulate bytes and abort if exceeded:

```python
total = 0
with open(out_path, "wb") as f:
    for chunk in resp.stream(8 * 1024 * 1024):
        total += len(chunk)
        if total > settings.max_external_download_bytes:
            raise URLNotAllowed("download exceeded max bytes")
        f.write(chunk)
```

### Resolve-once race mitigation

A pure pre-check is vulnerable to DNS rebinding (host resolves to public IP at check, private at fetch). To mitigate: resolve once, then connect by IP (set `Host:` header to original hostname). This is more invasive — defer if `minio_allowed_hosts` is small (a single trusted host) which is the current case.

---

## Acceptance criteria

1. `tests/test_url_safety.py`:
   - `http://169.254.169.254/...` → `URLNotAllowed`
   - `http://localhost:8000/` → `URLNotAllowed`
   - `http://192.168.1.1/x.mp4` → `URLNotAllowed`
   - `file:///etc/passwd` → `URLNotAllowed`
   - `https://example.com/x.mp4` (not in allowlist) → `URLNotAllowed`
   - `https://mind.snikbtel.uk:9000/bucket/x.mp4` → returns the URL.
2. `tests/controllers/test_video_endpoints.py`:
   - POST with `videoUrl=http://169.254.169.254/...` → 400.
   - POST with `Content-Length: 6000000000` (with `max_upload_size=5GB`) → 413, no bytes consumed.
3. Manual: `curl -X POST -F 'videoUrl=http://169.254.169.254/latest/meta-data/' http://gpu-host:8000/api/v1/video/process-and-upload` returns 400.

---

## Out of scope

- mTLS to the MinIO server.
- Migrating uploads to direct-to-S3 presigned PUTs.
