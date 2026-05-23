# Task 0005 — Delete dead trip-schedule services + cull provably-unread config flags

**Severity:** MEDIUM (dead code)
**Source:** Architecture review 2026-05-09, findings #5 + #6.
**Estimated effort:** 1 day.

---

## Problem

Two ~450-line services are wired in but never instantiated in production:

- `app/services/etrain_delay_service.py` (`EtrainDelayService`, fetches train delays from etrain.info)
- `app/services/trip_data_service.py` (`TripDataService`, fetches train schedules from RailRadar API)

Both are gated on `train_motion_rules_enabled` (default `0` in `app/utils/config.py:601`). User confirmed (2026-05-09): these are dead in prod. The fetch path in `video_processing_service.py:215–275` therefore never runs in default config. The matching test suite (`tests/unit/test_settings_validator.py`) exercises a coherence rule that becomes vacuous once the flag is gone.

Separately, `vlm_self_consistency_enabled` (`app/utils/config.py:804`) was confirmed via grep to have **zero readers** anywhere in the codebase.

---

## Behavior to PRESERVE (this is the trickiest part)

**Critical invariant:** `locopilot_monitor.py:3895` reads `self.suppress_no_person_without_schedule and self.trip_schedule is None` to suppress `no_person_detected` violations when no schedule is loaded. In production today, `trip_schedule` is always `None` (services never instantiate), so `no_person_detected` is **always suppressed in prod**.

After this task, `trip_schedule` will still be `None` (just for a different reason — nobody calls `set_trip_schedule()` anymore). The suppression behavior must remain identical. **Do NOT delete `set_trip_schedule()`, `self.trip_schedule`, or the gate at line 3895 from the monolith.** Out-of-scope per Wave 3 plan.

---

## Files to change

**Delete entirely:**
- `app/services/etrain_delay_service.py`
- `app/services/trip_data_service.py`

**Modify:**
- `app/services/video_processing_service.py` — remove the `from .trip_data_service import get_trip_data_service` import (line 21–23) and the entire trip-schedule fetch block (lines ~215–275). The "skip" branch's logging that mentions `train_motion_rules_enabled` goes too. Replace the whole block with a one-line comment noting it was removed (no logging needed; users never see it).
- `app/utils/config.py`:
  - Delete `train_motion_rules_enabled` (line 601).
  - Delete `etrain_enabled` (line 718) and `etrain_base_url` (line 721).
  - Delete `trip_api_url` (line 695) and `trip_api_timeout` (line 696).
  - Delete `vlm_self_consistency_enabled` (line 804).
  - Delete the flag-coherence validator block that mentions `train_motion_rules_enabled` (lines ~876, 918–922). Verify the rest of `_validate_flag_combinations` stays intact for other flags it checks.
- `app/utils/video_multiprocessing.py` — remove the `trip_schedule_dict` parameter and the serialization/reconstruction code (lines 535, 568, 655–662, 874, 895, 940–942). The worker no longer calls `monitor.set_trip_schedule()`; the monolith's `self.trip_schedule = None` default stays untouched.
- `tests/unit/test_settings_validator.py` — delete the three tests that exercise `train_motion_rules_enabled` (`test_rules_enabled_without_detection_rejected`, `test_rules_enabled_with_detection_truthy_ok`, `test_rules_disabled_with_any_detection_ok`) and the `assert hasattr(settings, "train_motion_rules_enabled")` line in `test_default_settings_constructs`. Other tests in this file (path checks, MinIO, pose model) must keep working.

**Sweep (one-line cleanup deferred from Wave 1):**
- `locopilot_monitor.py:4035` — a comment string mentions `app/core/pipeline/stages/train_motion_suppress_stage.py`. That file no longer exists. Update the comment to reference the live equivalent (`app/core/gates.py:apply_train_stopped_suppression`) or delete the comment if it's not load-bearing. **This is the ONLY change permitted in `locopilot_monitor.py` for this task.**

**Do NOT touch:**
- `app/core/detectors/train_motion_detector.py` — vibration-based detector, completely separate from the rules engine. Stays.
- `train_motion_detection_enabled` config flag — different from `train_motion_rules_enabled`. Stays.
- `set_trip_schedule()`, `self.trip_schedule`, `self.suppress_no_person_without_schedule` in `locopilot_monitor.py` — all stay; preserves behavior.
- Anything else in `locopilot_monitor.py` other than the line 4035 comment.

---

## Verification commands

After the changes, every one of these must return zero hits (excluding `__pycache__`, this spec file, and `PLAN.md`):

```
grep -rn "train_motion_rules_enabled" --include="*.py" .
grep -rn "etrain_delay_service\|EtrainDelayService\|etrain_enabled\|etrain_base_url" --include="*.py" .
grep -rn "trip_data_service\|TripDataService\|trip_api_url\|trip_api_timeout" --include="*.py" .
grep -rn "vlm_self_consistency_enabled" --include="*.py" .
grep -rn "trip_schedule_dict" --include="*.py" .
grep -rn "pipeline/stages/train_motion" --include="*.py" .
```

Expected total LOC deleted: ~900–1100.

---

## Acceptance criteria

1. `pytest tests/` is green (5+ tests will be removed; the rest pass).
2. `pytest tests/unit/test_settings_validator.py` is green for the surviving tests.
3. `python -c "from app.utils.config import Settings; s = Settings(_env_file=None); print('config ok')"` succeeds. `LOCOPILOT_SKIP_PATH_CHECKS=1` if needed.
4. `python -c "from app.services.video_processing_service import VideoProcessingService; print('vps ok')"` succeeds.
5. `python -c "from locopilot_monitor import LocopilotActivityMonitor; print('monolith ok')"` succeeds.
6. `pytest tests/regression/` is green — behavior on real-video fixtures preserved.
7. The grep audit above returns zero hits.
8. No new logging messages added to user-facing paths.

---

## Out of scope

- Deleting `set_trip_schedule()` / `self.trip_schedule` plumbing from the monolith. Behavior-preserving leave-alone; a Wave 3 task may revisit.
- Deleting `suppress_no_person_without_schedule` flag. Behavior-preserving leave-alone.
- Touching `_process_frames_core` or any other monolith internals besides the line-4035 comment.
- Refactoring `_validate_flag_combinations` beyond the deletions specified.
- Touching `TrainMotionDetector` (vibration detector) or its config flag `train_motion_detection_enabled`.
- Any other config flag cleanup. Restrictive list above is intentional — `ir_forward_lean_enabled` (2 readers in monolith), `use_unsharp_masking` (6 readers), `mp_overlap_seconds` (6 readers), `vlm_disagreement_log_enabled` (1 reader) are all live; do NOT delete them in this task.

---

## Notes for the executing agent

- The "skip" branch logging in `video_processing_service.py` includes nice-looking diagnostics like `f"[MOTION-RULES] [SKIP] ... All detected activities will be treated as violations."`. **Do not preserve this message** — it documents the behavior of the deleted feature. A short comment (`# Trip-schedule motion rules removed (2026-05-09); see docs/specs/architecture-cleanup/`) is enough.
- After deletion, `trip_schedule` is unconditionally `None` at the call site. The downstream consumer (the monolith's gate at line 3895) keeps suppressing `no_person_detected` exactly as before — this is correct and intentional.
- When updating `video_multiprocessing.py`, the worker function signatures change. Make sure the orchestrator's call site is updated too. Search for every occurrence of `trip_schedule` in that file before editing.
- Karpathy guidelines apply: this is a deletion task. Do not "improve" surrounding code, do not add error handling, do not rename anything.
