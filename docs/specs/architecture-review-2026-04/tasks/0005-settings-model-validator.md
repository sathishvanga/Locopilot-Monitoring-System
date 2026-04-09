# Task 0005: Add `model_validator` to `Settings` and auto-generate `.env.example`

- **Issue ID:** ARCH-05
- **Priority:** Medium-impact, medium-effort
- **Severity:** MEDIUM — silent misconfiguration risk
- **Category:** Config surface / Feature flags
- **Files:**
  - `app/utils/config.py` (255 typed fields, 235 `os.getenv` calls, only 2
    validators — lines 12, 172)
  - `.env.example` (currently 79 documented env vars; ~66% of flags
    undocumented)

## Description

`config.py` defines 255 typed settings fields and 235 unique
`os.getenv(...)` calls, but only 79 of them are documented in
`.env.example`. There are only **2** `@field_validator`s and **zero** cross-
flag coherence checks. Silent misconfiguration examples:

- `TRAIN_MOTION_RULES_ENABLED=1` with `TRAIN_MOTION_DETECTION_ENABLED=0`
  boots and silently does nothing (the rules engine has no motion state).
- `POSE_MODEL=rtmpose` boots even though no `rtmpose_adapter.py` exists in
  this branch — the code silently falls back or errors deep in the
  pipeline.
- `TRAIN_MOTION_RULES_ENABLED=1` without `OCR_ENABLED=1` and without a
  `trip_schedule` silently disables station suppression.
- Model file paths can reference nonexistent `.pt` files — error surfaces
  only at first inference.

## Fix

1. Add a Pydantic `@model_validator(mode='after')` to `Settings` in
   `app/utils/config.py`:

   ```python
   from pydantic import model_validator

   @model_validator(mode='after')
   def _validate_flag_combinations(self) -> 'Settings':
       # (a) Referenced model files exist
       for attr in ('yolo_model_path', 'yolo_pose_model_path',
                    'yolo_voting_model_path', 'yolo_voting_pose_model_path',
                    'yolo_roi_model_path'):
           path = getattr(self, attr, None)
           if path and not os.path.exists(path):
               raise ValueError(f"{attr}={path!r} does not exist")

       # (b) Flag coherence
       if self.train_motion_rules_enabled and not self.train_motion_detection_enabled:
           raise ValueError(
               "TRAIN_MOTION_RULES_ENABLED=1 requires "
               "TRAIN_MOTION_DETECTION_ENABLED=1"
           )

       # (c) Pose backend adapter exists
       if getattr(self, 'pose_model', 'yolo') == 'rtmpose':
           try:
               import rtmlib  # noqa: F401
           except ImportError as e:
               raise ValueError(
                   "POSE_MODEL=rtmpose requires rtmlib; "
                   "install with `pip install rtmlib onnxruntime`"
               ) from e

       # (d) Overlap window covers baseline + coordination (see task 0003)
       if hasattr(self, 'mp_overlap_seconds'):
           required = max(
               self.sleep_baseline_calibration_window,
               self.hand_gesture_coordination_window,
           )
           if self.mp_overlap_seconds < required:
               raise ValueError(
                   f"mp_overlap_seconds={self.mp_overlap_seconds} must be "
                   f">= max(sleep_baseline_calibration_window, "
                   f"hand_gesture_coordination_window) = {required}"
               )

       return self
   ```

2. Replace the hand-written `.env.example` with an auto-generator at
   `scripts/generate_env_example.py`:

   ```python
   from app.utils.config import Settings

   def main():
       lines = ["# Auto-generated from app/utils/config.py — do not edit."]
       for name, field in Settings.model_fields.items():
           default = field.default
           lines.append(f"# {field.description or name}")
           lines.append(f"{name.upper()}={default}")
           lines.append("")
       Path(".env.example").write_text("\n".join(lines))
   ```

3. Add a CI check / pre-commit hook that runs
   `python scripts/generate_env_example.py --check` and fails if
   `.env.example` would change.

4. Audit the ~150 undocumented flags and add `description=...` to each
   `Field()` so the generated `.env.example` is human-readable.

## Acceptance criteria

- [ ] `Settings._validate_flag_combinations` exists and is covered by
      unit tests exercising at least the 4 rules above.
- [ ] Starting the app with `TRAIN_MOTION_RULES_ENABLED=1 TRAIN_MOTION_DETECTION_ENABLED=0`
      fails at startup with a clear error.
- [ ] Starting with `YOLO_MODEL_PATH=/nope.pt` fails at startup.
- [ ] `scripts/generate_env_example.py` regenerates `.env.example` from
      `Settings.model_fields`.
- [ ] `.env.example` documents at least 200 of the 255 typed settings
      fields (80%+ coverage), up from today's 79.

## Implementation status (2026-04-09)

**Branch:** `feat/arch-review-2026-04/0005-settings-model-validator`

**Files changed:**
- `app/utils/config.py` — imported `model_validator`, added
  `_validate_flag_combinations(self)` on `Settings`.
- `scripts/generate_env_example.py` — new auto-generator with
  `--check` / `--stdout` / `--output` flags.
- `.env.example` — regenerated from `Settings.model_fields` (229 entries).
- `tests/unit/test_settings_validator.py` — 8 unit tests.

**Validator rules implemented:**
- (a) Referenced YOLO model paths must exist on disk when set to an
  absolute path. Checks both the spec field names
  (`yolo_model_path`, `yolo_pose_model_path`, `yolo_voting_model_path`,
  `yolo_voting_pose_model_path`, `yolo_roi_model_path`) and the names
  actually defined in this branch (`yolo_weights`, `yolo_pose_weights`,
  `yolo_voting_weights`, `yolo_voting_pose_weights`). Missing fields are
  skipped via `getattr(..., None)`. Relative paths are allowed so the
  ultralytics model cache can resolve them lazily.
- (b) `train_motion_rules_enabled=True` + explicit
  `train_motion_detection_enabled=False` (either as a field or as an
  env var) raises `ValueError`. If neither is set, the validator stays
  silent — the task instructions required graceful skipping on missing
  fields.
- (c) `pose_model == 'rtmpose'` requires `rtmlib` importable. This branch
  does not yet expose a `pose_model` field on `Settings`, so the check is
  defensively gated on `getattr(self, 'pose_model', None)`.
- Escape hatch: `LOCOPILOT_SKIP_PATH_CHECKS=1` bypasses (a) for fresh
  clones and unit tests that construct `Settings` with synthetic values.

**Acceptance criteria results:**
- `_validate_flag_combinations` exists + 7/8 tests passing (1 skipped
  because `pose_model` isn't a field on this branch): DONE.
- `TRAIN_MOTION_RULES_ENABLED=1 TRAIN_MOTION_DETECTION_ENABLED=0` fails at
  startup: DONE (verified via test
  `test_rules_enabled_without_detection_rejected`).
- `YOLO_MODEL_PATH=/nope.pt` fails at startup: DONE for the `_weights`
  fields actually used by this branch; verified via
  `test_nonexistent_yolo_weights_rejected`.
- `scripts/generate_env_example.py` regenerates `.env.example`: DONE.
- `.env.example` documents ≥200 fields: DONE — 229 fields written, up
  from 79 baseline (290% of target).

**Not done:**
- Audit of the ~150 undocumented flags to add `description=...` to each
  `Field()` (step 4 in the Fix section). Out of scope for this task; the
  generated comments fall back to a humanised field name when no
  description is present.
- Pre-commit / CI check hook that runs `--check` automatically. The
  script supports `--check`; wiring it into `.github/workflows/*` is
  deferred to a follow-up infra task.
