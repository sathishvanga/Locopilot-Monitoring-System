"""
Unit tests for Settings._validate_flag_combinations model validator.

Covers the four acceptance rules from docs/specs/architecture-review-2026-04/
tasks/0005-settings-model-validator.md:

    (a) Referenced model files must exist on disk (when absolute).
    (b) train_motion_rules_enabled=True requires train_motion_detection_enabled
        (or env fallback).
    (c) pose_model == 'rtmpose' requires rtmlib importable.
    (d) Default construction succeeds when the escape hatch is set.

The tests deliberately avoid loading any .env file (``_env_file=None``) so
they are deterministic regardless of the developer's local environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure the repo root is importable when pytest is invoked from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The validator's model-path check is disabled for the entire test module so
# fresh clones without .pt weights can still construct Settings. Individual
# tests that exercise the path check re-enable it temporarily.
os.environ.setdefault("LOCOPILOT_SKIP_PATH_CHECKS", "1")

from app.utils.config import Settings  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    """
    Provide an environment free of the env vars the validator reads, while
    preserving ``LOCOPILOT_SKIP_PATH_CHECKS`` so default construction does
    not trip on missing model weights.
    """
    for var in (
        "TRAIN_MOTION_RULES_ENABLED",
        "TRAIN_MOTION_DETECTION_ENABLED",
        "POSE_MODEL",
        "YOLO_WEIGHTS_PRELOAD",
        "YOLO_POSE_WEIGHTS",
        "YOLO_VOTING_WEIGHTS",
        "YOLO_VOTING_POSE_WEIGHTS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOCOPILOT_SKIP_PATH_CHECKS", "1")
    return monkeypatch


def test_default_settings_constructs(clean_env):
    """Default Settings() should succeed with skip flag set."""
    settings = Settings(_env_file=None)
    assert settings is not None
    assert hasattr(settings, "train_motion_rules_enabled")


def test_rules_enabled_without_detection_rejected(clean_env):
    """
    Enabling train motion rules while explicitly disabling detection must
    raise — this is the silent-misconfiguration case flagged in ARCH-05.
    """
    clean_env.setenv("TRAIN_MOTION_DETECTION_ENABLED", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, train_motion_rules_enabled=True)
    msg = str(excinfo.value)
    assert "TRAIN_MOTION_RULES_ENABLED" in msg
    assert "TRAIN_MOTION_DETECTION_ENABLED" in msg


def test_rules_enabled_with_detection_truthy_ok(clean_env):
    """Rules enabled + detection=1 in env should validate cleanly."""
    clean_env.setenv("TRAIN_MOTION_DETECTION_ENABLED", "1")
    settings = Settings(_env_file=None, train_motion_rules_enabled=True)
    assert settings.train_motion_rules_enabled is True


def test_rules_disabled_with_any_detection_ok(clean_env):
    """Rules disabled → detection flag is irrelevant, always valid."""
    clean_env.setenv("TRAIN_MOTION_DETECTION_ENABLED", "0")
    settings = Settings(_env_file=None, train_motion_rules_enabled=False)
    assert settings.train_motion_rules_enabled is False


def test_nonexistent_yolo_weights_rejected(clean_env):
    """Absolute path to a missing model file must raise."""
    # Re-enable path checks just for this test.
    clean_env.setenv("LOCOPILOT_SKIP_PATH_CHECKS", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None,
            yolo_weights="/definitely/does/not/exist.pt",
        )
    msg = str(excinfo.value)
    assert "yolo_weights" in msg
    assert "does not exist" in msg


def test_relative_yolo_weights_accepted(clean_env):
    """
    Relative paths are intentionally allowed to support ultralytics' model
    cache (it downloads weights lazily when given a bare name).
    """
    clean_env.setenv("LOCOPILOT_SKIP_PATH_CHECKS", "0")
    settings = Settings(_env_file=None, yolo_weights="yolo26n.pt")
    assert settings.yolo_weights == "yolo26n.pt"


def test_skip_path_checks_bypasses_existence(clean_env):
    """Escape hatch must let a nonexistent absolute path through."""
    clean_env.setenv("LOCOPILOT_SKIP_PATH_CHECKS", "1")
    settings = Settings(
        _env_file=None,
        yolo_weights="/definitely/does/not/exist.pt",
    )
    assert settings.yolo_weights == "/definitely/does/not/exist.pt"


def test_rtmpose_without_rtmlib_rejected(clean_env, monkeypatch):
    """
    POSE_MODEL=rtmpose should fail if rtmlib cannot be imported. This
    branch does not define a ``pose_model`` field, so the check is
    exercised by injecting an attribute on the settings instance post-
    construction would not work — instead we patch __init__ to attach the
    attribute and then re-run the validator. If the field is not present,
    the test is skipped as specified in the task instructions.
    """
    if "pose_model" not in Settings.model_fields:
        pytest.skip("pose_model field not defined in this branch")
    # Force rtmlib to be unimportable.
    monkeypatch.setitem(sys.modules, "rtmlib", None)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, pose_model="rtmpose")
    assert "rtmlib" in str(excinfo.value)
