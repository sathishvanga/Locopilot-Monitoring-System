# Task 0005 — Rotate MinIO creds + scrub committed secrets

**Severity:** CRITICAL (security)
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #6, top-fix #5.
**Estimated effort:** 1 hour code change + coordinated rotation window.

---

## Problem

Three secrets are exposed:

### 1. MinIO creds in source defaults AND in `.env.example`

`app/utils/config.py:166-167`:
```python
minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin")
minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "login123")
```

`.env.example:169-172` ships the same values. `.gitignore` whitelists `.env.example` (`!.env.example`), so these credentials are in version control and on every dev machine that clones the repo. Worse, the defaults mean the service silently comes up with these creds if env vars are unset.

### 2. SSH/sudo password in `deploy-gpu.sh`

`deploy-gpu.sh:12` contains a hardcoded `SERVER_PASS='...'` used with `sshpass` and `sudo -S` (lines 104, 149). Anyone with this file has root on the GPU box. Although `deploy-gpu.sh` is in `.gitignore` (line 29), it lives on every developer machine and was likely committed at some point in history.

### 3. Bearer tokens captured into log context

`app/middleware/logging_middleware.py:47, 58` stores the full `Authorization` header value into the request context dict. Any later `logger.info` / `logger.error` that interpolates context fields writes the bearer to `/opt/poc2/logs/LocopilotMonitoring.log`.

---

## Files to change

- `app/utils/config.py:166-167`
- `.env.example:169-172`
- `deploy-gpu.sh:12, 104, 149`
- `app/middleware/logging_middleware.py:47, 58`

---

## Fix

### Config defaults fail-closed

```python
minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "")
minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "")

@model_validator(mode="after")
def _require_minio_creds_in_prod(self):
    if self.environment == "production" and (not self.minio_access_key or not self.minio_secret_key):
        raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in production")
    return self
```

### `.env.example` placeholders

```
MINIO_ACCESS_KEY=changeme
MINIO_SECRET_KEY=changeme
```

### `deploy-gpu.sh`

```bash
SERVER_PASS="${LOCOPILOT_DEPLOY_PASS:?set LOCOPILOT_DEPLOY_PASS env var}"
```

Long term: switch to SSH key auth and remove `sshpass` entirely.

### Middleware redaction

```python
auth_header = request.headers.get("authorization", "")
auth_present = "***" if auth_header else "None"
context["authorization"] = auth_present  # never the raw value
```

### History scrub

After rotation, run:

```bash
git log --all --full-history -- .env.production deploy-gpu.sh | head -50
```

If anything other than `.env.example` ever touched the index, use `git filter-repo` to purge those paths from history and force-push (coordinate with team — destructive operation).

---

## Coordinated rotation procedure

**Do not start until every step is scheduled.** Order matters:

1. Rotate MinIO password on the MinIO server (`mind.snikbtel.uk:9000`). Generate a 32-char random password.
2. Update `.env.production` on the GPU box AND on the deployer machines with the new password.
3. Restart `locopilot.service`.
4. Land the source code changes (this task).
5. Rotate the SSH/sudo password for the deploy user on the GPU box.
6. Update `LOCOPILOT_DEPLOY_PASS` in each deployer's local environment.
7. Run `git filter-repo` if history scan turned up anything sensitive.
8. Notify team to re-clone (history rewrite invalidates local clones).

---

## Acceptance criteria

1. `git ls-files | grep -E '^\.env'` returns only `.env.example`.
2. `git log --all --full-history -- .env .env.production deploy-gpu.sh` returns no commits, OR a follow-up `filter-repo` PR is merged.
3. `grep -E "(login123|admin)" .env.example app/utils/config.py` returns zero hits.
4. Starting the service with `ENVIRONMENT=production` and unset MinIO env vars fails fast with a clear error.
5. Tail `LocopilotMonitoring.log` after a request with a Bearer token. The token must NOT appear in any log line.
6. `deploy-gpu.sh` runs successfully when `LOCOPILOT_DEPLOY_PASS` is exported, fails clearly when it is not.

---

## Out of scope

- Migrating the API from Bearer-token auth to mTLS or another scheme.
- Replacing MinIO with a different object store.
