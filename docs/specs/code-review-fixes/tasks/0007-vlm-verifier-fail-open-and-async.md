# Task 0007 — VLM verifier: per-activity try/except + async client + circuit breaker

**Severity:** CRITICAL (correctness) + HIGH (latency)
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #5, top-fix #7.
**Estimated effort:** 1 day.

---

## Problem

The Pipeline-2 VLM verifier (`app/services/vlm_verification_service.py`) is documented as **fail-open**: if vLLM is down, Pipeline-1 verdicts pass through unchanged. The implementation breaks this contract in three ways.

### 1. Fail-open is actually fail-destructive

`vlm_verification_service.py:991-1085`. The exception handler at line 1046 catches only `URLError/HTTPError/TimeoutError/OSError`. It does NOT catch:
- `json.JSONDecodeError` from `json.loads(resp.read())` at line 1045 (vLLM returns malformed body)
- `KeyError` / `AttributeError` from `act.get(...)` patterns elsewhere
- `cv2` / `numpy` failures inside `_stitch_keyframes`

When an unexpected exception fires, it propagates out of `verify_activities` to `video_processing_service.py:357`. **The `kept` list discards any activities not yet processed.** Pipeline-1 violations are silently truncated mid-loop. This violates the documented invariant: "Pipeline-2 only filters; it never adds violations and never silently drops them."

### 2. Blocking HTTP pins the event loop

`vlm_verification_service.py:1042` uses `urllib.request.urlopen` synchronously. Each call blocks the calling thread for up to `vlm_timeout_seconds` (15s prod). All 6+ activities per video serialize → up to 90s of head-of-line blocking. No connection pooling — fresh TCP+TLS per activity.

### 3. No retry / no circuit breaker on cold start

A single 503 from vLLM (model warm-up, GC pause) skips verification entirely. During a 60-second cold-start window, every activity becomes `SKIPPED_VLM_UNAVAILABLE`, costing `N × timeout_seconds` per video. No "last-failure-at" short-circuit.

### 4. Prompt-injection surface in motion override

`vlm_verification_service.py:718-721, 735-747`. The VLM emits `motion_evidence` free-text; this string is passed to `_has_hard_stopped_cue` whose regex matches `"door open"` / `"platform"` in any clause. A jailbroken VLM (or upstream prompt injection) emitting `motion_evidence: "platform"` for a RUNNING-train activity flips `motionState` to STOPPED and silently DROPS a real violation.

---

## Files to change

- `app/services/vlm_verification_service.py:838-968` — per-activity exception envelope
- `app/services/vlm_verification_service.py:1041-1058` — async client + retry + circuit breaker
- `app/services/vlm_verification_service.py:718-747` — tighten motion override
- `app/services/vlm_verification_service.py:752-764` — verdict enum validation
- `requirements.txt` — add `httpx>=0.27.0`

---

## Fix

### Per-activity exception envelope

```python
async def verify_activities(self, activities: list[dict], ...) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    stats = {"verified": 0, "dropped_fp": 0, "skipped_vlm_unavailable": 0,
             "skipped_parse_error": 0, "skipped_cap": 0}

    for act in activities:
        try:
            verdict = await self._verify_one(act, ...)
            if verdict.should_drop:
                stats["dropped_fp"] += 1
                continue  # do NOT append to kept
            act["vlm_review"] = verdict.to_dict()
            stats["verified"] += 1
            kept.append(act)
        except Exception as e:
            logger.exception("[vlm] verify_one failed for activity %s", act.get("id"))
            act["vlm_review"] = {"verdict": "SKIPPED_VLM_UNAVAILABLE", "reason": type(e).__name__}
            stats["skipped_vlm_unavailable"] += 1
            kept.append(act)  # CRITICAL: never lose a Pipeline-1 violation
    return kept, stats
```

### Async client + connection pool

Replace `urllib.request.urlopen` with a singleton `httpx.AsyncClient`:

```python
class VlmVerificationService:
    _client: httpx.AsyncClient | None = None

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return cls._client
```

Dispatch all activities concurrently with a bounded `asyncio.Semaphore(4)`:

```python
sem = asyncio.Semaphore(4)
async def bounded(act): 
    async with sem: return await self._verify_one(act, ...)
results = await asyncio.gather(*(bounded(a) for a in activities), return_exceptions=True)
```

### Retry with circuit breaker

```python
class _CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0):
        self._fail_count = 0
        self._opened_at: float | None = None
        self.threshold = threshold
        self.cooldown_s = cooldown_s

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at > self.cooldown_s:
            self._opened_at = None
            self._fail_count = 0
            return False
        return True

    def record_failure(self):
        self._fail_count += 1
        if self._fail_count >= self.threshold:
            self._opened_at = time.monotonic()

    def record_success(self):
        self._fail_count = 0
        self._opened_at = None
```

In `_verify_one`: if breaker open, immediately return `SKIPPED_VLM_UNAVAILABLE`. Single retry with 0.5s backoff on connect/timeout errors before recording failure.

### Tighten motion override

Refuse to flip RUNNING→STOPPED purely from VLM. Require:
1. Structured `train_appears_to_be == "stopped"` (not regex over free text).
2. AND Pipeline-1's own `train_motion_detector` corroboration on the keyframe.

Use VLM motion only as a tiebreaker on `UNCERTAIN`, never as an override of `RUNNING`.

### Verdict enum validation

```python
ALLOWED = {"TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN"}
if verdict not in ALLOWED:
    raise ValueError(f"verdict not in enum: {verdict!r}")
```

Increment `stats["skipped_parse_error"]` on this branch.

### Logging on `cv2.imread` failure

`vlm_verification_service.py:671-672`:

```python
img = cv2.imread(p)
if img is None:
    logger.warning("[vlm] cv2.imread returned None for %s", p)
    continue
```

---

## Acceptance criteria

1. `tests/services/test_vlm_fail_open.py`:
   - 5 activities; mock `_verify_one` to raise `RuntimeError` on activity #2. Assert all 5 are in `kept` (with #2 marked `SKIPPED_VLM_UNAVAILABLE`).
   - Mock vLLM returning malformed JSON. Assert all activities are kept.
2. `tests/services/test_vlm_circuit_breaker.py`:
   - 3 consecutive timeouts open the breaker; subsequent calls return `SKIPPED_VLM_UNAVAILABLE` immediately (no HTTP attempt). After 30s, breaker closes.
3. `tests/services/test_vlm_motion_override.py`:
   - VLM emits `motion_evidence: "platform"` while train_motion_detector says RUNNING. Assert NO drop.
4. Latency: with `MAX_CONCURRENT_VIDEOS=1` and 6 activities, total VLM time ≤ `2 × vlm_timeout_seconds` (was `6 × vlm_timeout_seconds`).
5. `grep -n "urllib.request.urlopen" app/services/vlm_verification_service.py` returns zero hits.

---

## Out of scope

- Splitting `vlm_verification_service.py` into prompts / image utils / http client / orchestrator (separate refactor task).
- Re-prompting vLLM on parse errors.
