# Architecture cleanup — execution plan

**Source:** Architecture review 2026-05-09 (post-merge to feature/cropping-image-applying-yolo).

**Driver:** The codebase decomposed cleanly on paper but has dual implementations of the frame loop, a 4,407-line god-class still doing the heavy lifting, and ~5,000 lines of code that can be deleted with no behavior change.

**User-confirmed decisions (2026-05-09):**

1. Delete the dormant `app/core/pipeline/stages/` + `frame_pipeline.py` — *do not* attempt the cutover.
2. `train_motion_rules_enabled` / etrain / trip_data are dead in production — delete them.
3. e2e tests on real videos exist and are authoritative.
4. One epic spec per fix; agents in parallel where safe.

---

## Waves

### Wave 1 — Parallel-safe (4 agents)

| # | Task | Files touched | Risk |
|---|---|---|---|
| 0001 | Delete dormant pipeline scaffolding | `app/core/frame_pipeline.py`, `app/core/pipeline/stages/`, `app/core/pipeline/__init__.py` | Low — verified dormant |
| 0002 | Split `vlm_verification_service.py` into a package | `app/services/vlm_verification_service.py` (→ `app/services/vlm/*.py`) | Medium — covered by `tests/regression/vlm_fixture/` |
| 0003 | Collapse `/api/video/analyze` + `/api/v1/video/process-and-upload` handler bodies | `app/controllers/video_controller.py` | Low — single file |
| 0004 | Make `ActivityConfig` a view over `ACTIVITY_REGISTRY` | `app/core/activity_tracker.py`, `app/core/activity_registry.py` | Low — small surface |

These agents touch disjoint files, so merge conflicts are impossible.

### Wave 2 — Sequential (cascade)

| # | Task | Risk |
|---|---|---|
| 0005 | Conservative config flag cull (only flags with zero readers in non-test code) | Low |
| 0006 | Delete `etrain_delay_service.py`, `trip_data_service.py`, and the `train_motion_rules_enabled` rule-engine path. Keep `TrainMotionDetector` (vibration). | Medium |

### Wave 3 — Sequential, e2e tested after each (highest blast radius)

| # | Task | Risk |
|---|---|---|
| 0007 | Split `sleep_detector.py` (1,614 LOC) into 3 sibling modules | Medium |
| 0008 | Trim `gpu_resource_manager.py` (1,208 → ~400 LOC); keep semaphore + OOM backoff + model loading; delete per-worker CUDA streams | Medium |
| 0009 | Extract `process_all_persons_activities` (~1,200 LOC inside `LocopilotActivityMonitor`) into its own runner class | High |

---

## Out of scope for this cleanup
- `video_multiprocessing.py` deletion. Needs a benchmark first; spec deferred.
- Renaming the monolith file or breaking the public `LocopilotActivityMonitor` constructor.
- Any change to detector logic (sleep, activity, gesture, mind diversion). Behavior must be byte-identical post-refactor.

## Verification gate (every wave)
1. `pytest tests/` is green.
2. `pytest tests/regression/` is green.
3. Determinism contract test (`tests/test_determinism.py`) is green.
4. `python -c "from app.controllers.video_controller import video_router"` and `from locopilot_monitor import LocopilotActivityMonitor` import cleanly.
