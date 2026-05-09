# Task 0001 — Delete dormant pipeline scaffolding

**Severity:** HIGH (silent drift risk)
**Source:** Architecture review 2026-05-09, finding #1.
**Estimated effort:** 0.5 day.

---

## Problem

`app/core/frame_pipeline.py` and the 14 stage files in `app/core/pipeline/stages/` were extracted from `LocopilotActivityMonitor._process_frames_core` four months ago, but `_process_frames_core` (`locopilot_monitor.py:3411`) was never cut over to call `FramePipeline`. Since extraction, four commits have edited `_process_frames_core` directly (phone/sleep/microsleep/train-motion fixes). The stage files therefore contain a stale, parallel implementation of the frame loop that diverges from the live behavior.

Verified via grep: nothing imports `FramePipeline` outside the dead files themselves; `FrameState` is imported only by sibling stage files. The scaffolding's own docstring at `app/core/frame_pipeline.py:13–27` documents the situation and warns that cutting over now would silently regress those fixes.

User decision (2026-05-09): **delete, do not cut over.**

---

## Files to change

**Delete entirely:**
- `app/core/frame_pipeline.py`
- `app/core/pipeline/stages/` (entire directory, including `__init__.py` and all 14 stage files)

**Keep (these are live):**
- `app/core/pipeline/frame_sampling.py` — imported at `locopilot_monitor.py:61`.
- `app/core/pipeline/pose_batch.py` — imported at `locopilot_monitor.py:62`.

**Update:**
- `app/core/pipeline/__init__.py` — remove any references to `frame_pipeline` or `stages`. Keep references to the live submodules only.

**Audit + clean up imports:**
Before deleting, run:
```
grep -rn "from app.core.frame_pipeline\|from app.core.pipeline.stages\|FramePipeline\|FrameState" --include="*.py" .
```
Every match outside the deleted tree must be removed. Expected matches as of 2026-05-09:
- `app/core/pipeline/__init__.py` — comment-only references; rewrite the module docstring.
- Each stage file imports `FrameState` from `app.core.frame_pipeline` — these all get deleted, so no follow-up.
- Tests under `tests/refactor/` may reference the stages — delete or skip those tests.

---

## Fix

1. `git rm app/core/frame_pipeline.py`.
2. `git rm -r app/core/pipeline/stages/`.
3. Rewrite `app/core/pipeline/__init__.py` to a minimal docstring describing the two surviving live helpers (`frame_sampling`, `pose_batch`).
4. Remove any tests under `tests/refactor/` that import from the deleted modules. Tests that exercise live code (frame sampling, pose batching) stay.
5. Run `grep -rn "frame_pipeline\|pipeline.stages\|FramePipeline\|FrameState" --include="*.py" .` — must return zero hits outside this commit's deletions.

---

## Acceptance criteria

1. `find app/core/pipeline/stages -type f 2>/dev/null` is empty.
2. `test ! -f app/core/frame_pipeline.py`.
3. `python -c "from app.core.pipeline import frame_sampling, pose_batch"` succeeds.
4. `python -c "from locopilot_monitor import LocopilotActivityMonitor"` succeeds.
5. `pytest tests/` is green.
6. Grep for `FramePipeline`, `frame_pipeline` (excluding `__pycache__`) returns zero hits.

---

## Out of scope

- Changing `_process_frames_core` in any way.
- Touching `frame_sampling.py` or `pose_batch.py`.
- Refactoring detectors.
