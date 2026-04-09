# Task 0001: Consolidate activity metadata into a single `ActivityConfig`

- **Issue ID:** ARCH-01
- **Priority:** High-impact, medium-effort (do first)
- **Severity:** HIGH — live drift already present
- **Category:** Duplication / Extensibility
- **Files:**
  - `locopilot_monitor.py:59-152` (`ACTIVITY_REGISTRY`)
  - `locopilot_monitor.py:697-742` (`activity_type_map`, `activity_descriptions`, `evidence_rules`)
  - `locopilot_monitor.py:3637-3669` (duplicated `activity_key_map` — see also task 0008)
  - `app/services/activity_detection_service.py:31-68`
  - `app/models/activity_models.py:10-24` (`ActivityTypeEnum`)

## Description

Activity metadata is defined in **three** places today:

1. `locopilot_monitor.py` — the real runtime maps (12 activities).
2. `app/services/activity_detection_service.py` — a mock service used by
   controllers that has **only 10 activities**; it is already missing
   `eating_drinking` and `alp_not_standing`.
3. `app/models/activity_models.py` — the `ActivityTypeEnum` Pydantic enum.

`ACTIVITY_REGISTRY` partially consolidates tracking thresholds but does not
hold `type_code`, `description`, `evidence_rule`, `voting_key`, or
`triggering_role`, so those still live as parallel dicts on the monitor
instance.

Every new activity today requires touching ≥12 places (see task 0008 for the
copy-pasted `activity_key_map`). The mock service silently diverges.

## Fix

Extend `ActivityConfig` (`locopilot_monitor.py:40-53`) with:

```python
@dataclass
class ActivityConfig:
    type_code: int
    description: str
    evidence_rule: str
    voting_key: Optional[str] = None
    triggering_role: Optional[str] = None  # "LP" | "ALP" | None
    # existing fields:
    min_duration: float = 0.0
    required_consecutive: int = 1
    margin: Optional[int] = None
    grace_frames: int = 5
    region_margin: Optional[int] = None
    wrist_inside_margin: Optional[int] = None
    sustained_proximity_seconds: Optional[float] = None
```

1. Move `ActivityConfig` + `ACTIVITY_REGISTRY` from `locopilot_monitor.py` into
   a new module `app/core/activity_registry.py` so it can be imported without
   pulling in the monitor.
2. Seed each entry with `type_code`, `description`, and `evidence_rule` from
   the monitor's current dicts.
3. Rewrite `ActivityTypeEnum` in `activity_models.py` to derive from the
   registry (generate the enum at import time from `ACTIVITY_REGISTRY`), or
   add an assertion at module load that both are consistent.
4. Delete `self.activity_type_map`, `self.activity_descriptions`, and
   `self.evidence_rules` from the monitor. Replace usages at
   `locopilot_monitor.py:4196-4214` with `ACTIVITY_REGISTRY[activity_name].type_code`,
   `.description`, and `.evidence_rule`.
5. Delete the mock maps in `activity_detection_service.py:31-68` and have the
   mock import from the registry.
6. Remove the copy-pasted `activity_key_map` blocks at
   `locopilot_monitor.py:3637-3645` and `3659-3667` — fold into the registry
   (task 0008 can be absorbed here).

## Acceptance criteria

- [ ] `ACTIVITY_REGISTRY` contains `type_code`, `description`, `evidence_rule`
      for all 12 activities.
- [ ] `grep -rn "activity_type_map\|activity_descriptions\|evidence_rules" app/ locopilot_monitor.py`
      returns only the registry definition and reads, no parallel dicts.
- [ ] `ActivityTypeEnum` and `ACTIVITY_REGISTRY` cannot drift — enforced by a
      module-level assertion or by generating the enum from the registry.
- [ ] `activity_detection_service.py` imports the registry instead of
      redefining metadata; `eating_drinking` and `alp_not_standing` are
      present wherever the mock is used.
- [ ] Existing integration behavior is unchanged — sample runs produce
      identical JSON `activityType`, `des`, and `evidence.rule` fields.

## Implementation status

Implemented on branch
`feat/arch-review-2026-04/0001-consolidate-activity-metadata` (worktree
`agent-af4b8528`, commit pending) against base commit `929272b`.

### Files created

- `app/core/activity_registry.py` — new single-source-of-truth module. Defines
  the extended `ActivityConfig` dataclass (legacy threshold fields preserved +
  `type_code`, `description`, `evidence_rule`, `voting_key`, `triggering_role`)
  and the 12-entry `ACTIVITY_REGISTRY` plus a `rebuild_activity_registry()`
  helper.
- `tests/unit/test_activity_registry.py` — 12 new pytest tests covering
  registry shape, enum/registry parity, mock-service parity, triggering-role
  coverage, voting-key dispatch, and rebuild invariance. Passes under the
  project `.venv` (`python -m pytest tests/unit/test_activity_registry.py -q`
  → 12 passed).
- `tests/__init__.py`, `tests/unit/__init__.py` — empty package markers so the
  pytest collection path works without an editable install.

### Files modified

- `locopilot_monitor.py`
  - Removed the in-file `ActivityConfig` dataclass and
    `_build_activity_registry()` function (≈120 lines) and the stale
    `ACTIVITY_REGISTRY = _build_activity_registry()` call.
  - Imports `ACTIVITY_REGISTRY` and `ActivityConfig` from
    `app.core.activity_registry` instead.
  - Deleted the three parallel reporting dicts
    (`self.activity_type_map`, `self.activity_descriptions`,
    `self.evidence_rules`) from `LocopilotActivityMonitor.__init__`.
  - Rewrote the JSON-building block (was ~L4196; now in the 3760s after the
    deletions) to read `ACTIVITY_REGISTRY[activity_name].type_code /
    .description / .evidence_rule`.
  - Collapsed the two copy-pasted `activity_key_map` identity maps in the
    voting-batch dispatch (was L3637 and L3659) into a single
    `ACTIVITY_REGISTRY[activity_type].voting_key` lookup with raw-type
    fallback. Absorbs task 0008 for these call sites.
  - Net delta: +/- per `git diff --stat` ≈ `223 -` vs `60 +`.
- `app/models/activity_models.py`
  - `ActivityTypeEnum` is now generated at import time from
    `ACTIVITY_REGISTRY` via `_build_activity_type_enum_members()`. Adds the
    `_REGISTRY_TO_ENUM_NAME` mapping so legacy member names (e.g.
    `LP_NOT_EXCHANGING_HAND_GESTURE`) are preserved for external consumers.
  - Adds `_assert_enum_matches_registry()` called at module load as a
    belt-and-braces safety net in case the mapping is hand-edited.
- `app/services/activity_detection_service.py`
  - Deleted the hand-written `activity_type_map` / `activity_descriptions` /
    `evidence_rules` dicts (which were missing `eating_drinking` and
    `alp_not_standing`). They are now derived from `ACTIVITY_REGISTRY` so the
    mock sees all 12 activities and can never drift again.
  - Imports `ACTIVITY_REGISTRY` from `app.core.activity_registry`.

### Acceptance criteria

- **DONE** — `ACTIVITY_REGISTRY` contains `type_code`, `description`,
  `evidence_rule` for all 12 activities. Verified by
  `test_registry_contains_all_twelve_activities` and
  `test_every_registry_entry_has_required_reporting_fields`.
- **DONE** — `grep -rn "activity_type_map\|activity_descriptions\|
  evidence_rules" app/ locopilot_monitor.py` only returns (a) the registry
  definition/docstrings, (b) comments in `locopilot_monitor.py`, and (c) the
  mock service's derived dicts (still named for backward API compatibility
  but built from `ACTIVITY_REGISTRY`). No hand-maintained parallel dict
  remains.
- **DONE** — `ActivityTypeEnum` cannot drift from `ACTIVITY_REGISTRY`: the
  enum is generated from the registry, plus
  `_assert_enum_matches_registry()` runs at import time. Verified by
  `test_activity_type_enum_matches_registry_for_every_entry`,
  `test_activity_type_enum_covers_every_registry_entry`, and
  `test_activity_type_enum_has_no_extra_members_beyond_unknown`.
- **DONE** — `activity_detection_service.py` imports from the registry and
  `eating_drinking` + `alp_not_standing` are present. Verified by
  `test_mock_service_sees_every_registry_activity` and
  `test_mock_service_values_match_registry`.
- **PARTIAL** — "Existing integration behavior unchanged": verified
  statically by asserting that the registry values match the legacy
  `activity_type_map` / `activity_descriptions` / `evidence_rules` entries
  byte-for-byte and by `test_type_codes_match_legacy_assignments`. A full
  end-to-end sample run against a video was **not** executed because the
  pipeline requires GPU models and the `.venv` is CPU-only; this is
  documented in the task constraints.

### Out of scope / follow-ups

- `app/services/temporal_filtering_service.py` and
  `app/utils/video_multiprocessing.py` still contain their own local copies
  of `activity_type_map`/`descriptions`/`evidence_rules` (noted in MEMORY.md
  under "Two-Pass Deterministic Pipeline"). They were excluded from this task
  per the spec's file list. A follow-up task should point them at
  `ACTIVITY_REGISTRY` so the multiprocessing path also uses the single
  source of truth.
- The old `ActivityConfig` in `app/core/activity_tracker.py` still exists
  (unused now that the alias import was removed from the monitor). A later
  cleanup can either delete it or re-export the new dataclass.
- The `activity_key_map` dedupe here only covers the two voting-dispatch
  call sites in `locopilot_monitor.py`. Other `activity_key_map` references
  in `tasks/all-review-issues.md` describe a **different** map (name→key
  translation such as `packing`→`packing_bags`) and are unaffected.
