# Task 0002 — Atomic, locked, single-writer `activities.json`

**Severity:** CRITICAL
**Source:** `docs/code-review-2026-05-08.md` cross-cutting theme #1, top-fix #2.
**Estimated effort:** 2 hours.

---

## Problem

`activities.json` is the source-of-truth artifact for every video run. It currently has **three independent writers**, each with a **different `NumpyEncoder`**, **none atomic**, **none locked**:

1. `app/repositories/activity_repository.py:117-118` — `ActivityRepository.save_activities`
2. `app/utils/video_multiprocessing.py:1080-1084` — `process_video_parallel`
3. `locopilot_monitor.py:4313-4316` — legacy monolith `generate_summary_report`

Concrete failure modes observed in review:

- **Repository's encoder is missing `np.bool_`** (`activity_repository.py:20-29` vs `video_multiprocessing.py:35-46`). Pipeline-1 produces `np.bool_` flags routinely → `repository.save_activities(...)` will raise `TypeError: Object of type bool_ is not JSON serializable` on real payloads.
- **No `tmp + os.replace`**: `open(..., 'w')` truncates the file before writing. A crash/SIGKILL mid-`json.dump` leaves a half-written or empty file.
- **No file lock**: a Pipeline-1 worker writing the file at the same time as the Pipeline-2 VLM rewrite hook (post-verification) is a multi-writer race.

---

## Files to change

- `app/utils/json_utils.py` — **NEW** module containing the canonical encoder + atomic-write helper.
- `app/repositories/activity_repository.py` — delete duplicate `NumpyEncoder`; route writes through new helper.
- `app/utils/video_multiprocessing.py:1077-1084` — delete duplicate `NumpyEncoder`; call `ActivityRepository.save_activities`.
- `locopilot_monitor.py:4307-4317` — delete duplicate encoder; call `ActivityRepository.save_activities`.
- `app/services/video_processing_service.py:350-351` — same swap.

---

## Fix

### `app/utils/json_utils.py` (new)

```python
import json
import os
import tempfile
from typing import Any

import numpy as np
import portalocker


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "item"):  # covers np.bool_, np.float32, np.int64, np.datetime64
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def atomic_write_json(path: str, payload: Any, *, indent: int = 2) -> None:
    """Crash-safe + cross-process-safe JSON write."""
    directory = os.path.dirname(path) or "."
    lock_path = path + ".lock"
    with portalocker.Lock(lock_path, flags=portalocker.LOCK_EX, timeout=30):
        fd, tmp = tempfile.mkstemp(prefix=".activities.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=indent, ensure_ascii=False, cls=NumpyEncoder)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            finally:
                raise
```

### `ActivityRepository.save_activities`

Replace direct `open(...).write` with `atomic_write_json(path, activities)`. Delete the local `NumpyEncoder`.

### Other writers

Delete their inline `NumpyEncoder` classes and route through `ActivityRepository().save_activities(...)`.

---

## Acceptance criteria

1. `grep -rn "class NumpyEncoder" app/ locopilot_monitor.py` returns exactly one hit (`app/utils/json_utils.py`).
2. `grep -rn "json.dump" app/ locopilot_monitor.py | grep -v test` returns no hits writing `activities.json` directly.
3. Add `tests/test_atomic_write.py` that:
   - Writes a payload containing `np.bool_(True)`, `np.float32(1.5)`, `np.array([1,2,3])` and asserts no `TypeError`.
   - Spawns two threads each writing 100 times to the same path; asserts the file is always parseable.
   - Simulates a crash mid-write (kill the writer between `os.fdopen` and `os.replace`) and asserts the original file content is intact.
4. `requirements.txt` adds `portalocker>=2.8.2` (pin once Task 0009 lands).

---

## Out of scope

- Migrating from JSON to a database. Stays JSON for now.
- Schema changes to the activity records.
