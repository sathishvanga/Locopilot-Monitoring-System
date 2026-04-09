"""Unit tests for the consolidated activity metadata registry.

Task 0001 (architecture-review-2026-04) consolidated four parallel metadata
dictionaries into ``app.core.activity_registry.ACTIVITY_REGISTRY``. These
tests guard against regressions that would either:

* drop an activity from the 12-entry canonical list
* let ``ActivityTypeEnum`` drift from the registry ``type_code`` values
* let the mock ``ActivityDetectionService`` omit activities the real
  runtime monitor can emit (the previous bug: ``eating_drinking`` and
  ``alp_not_standing`` were missing from the mock).

They run without GPU, without OpenCV/YOLO, and without network access.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the project root is importable when pytest is invoked from the
# worktree root without an editable install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


EXPECTED_ACTIVITIES = {
    "cell_phone",
    "microsleep",
    "sleep",
    "writing",
    "packing_bags",
    "group_detected",
    "lp_hand_gesture",
    "alp_hand_gesture",
    "mind_diversion",
    "no_person_detected",
    "alp_not_standing",
    "eating_drinking",
}


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_contains_all_twelve_activities():
    from app.core.activity_registry import ACTIVITY_REGISTRY

    assert set(ACTIVITY_REGISTRY.keys()) == EXPECTED_ACTIVITIES
    assert len(ACTIVITY_REGISTRY) == 12


def test_every_registry_entry_has_required_reporting_fields():
    from app.core.activity_registry import ACTIVITY_REGISTRY

    for name, cfg in ACTIVITY_REGISTRY.items():
        assert cfg.type_code > 0, f"{name} missing type_code"
        assert cfg.description, f"{name} missing description"
        assert cfg.evidence_rule, f"{name} missing evidence_rule"


def test_type_codes_are_unique():
    from app.core.activity_registry import ACTIVITY_REGISTRY

    codes = [cfg.type_code for cfg in ACTIVITY_REGISTRY.values()]
    assert len(codes) == len(set(codes)), f"duplicate type_codes: {codes}"


def test_type_codes_match_legacy_assignments():
    """Guard against silent renumbering of existing activities.

    These values are part of the public JSON contract (``activityType`` field
    in the evidence output) and must not change without coordination with
    downstream consumers.
    """

    from app.core.activity_registry import ACTIVITY_REGISTRY

    expected = {
        "cell_phone": 2,
        "microsleep": 3,
        "sleep": 4,
        "writing": 5,
        "packing_bags": 6,
        "group_detected": 7,
        "lp_hand_gesture": 8,
        "alp_hand_gesture": 9,
        "mind_diversion": 10,
        "no_person_detected": 11,
        "alp_not_standing": 12,
        "eating_drinking": 13,
    }
    for name, code in expected.items():
        assert ACTIVITY_REGISTRY[name].type_code == code, (
            f"{name} type_code drifted: "
            f"expected {code}, got {ACTIVITY_REGISTRY[name].type_code}"
        )


# ---------------------------------------------------------------------------
# Enum / registry consistency
# ---------------------------------------------------------------------------


def test_activity_type_enum_matches_registry_for_every_entry():
    from app.core.activity_registry import ACTIVITY_REGISTRY
    from app.models.activity_models import (
        ActivityTypeEnum,
        _REGISTRY_TO_ENUM_NAME,
    )

    for key, cfg in ACTIVITY_REGISTRY.items():
        enum_name = _REGISTRY_TO_ENUM_NAME[key]
        member = ActivityTypeEnum[enum_name]
        assert int(member) == cfg.type_code, (
            f"ActivityTypeEnum.{enum_name} ({int(member)}) != "
            f"ACTIVITY_REGISTRY['{key}'].type_code ({cfg.type_code})"
        )


def test_activity_type_enum_covers_every_registry_entry():
    from app.core.activity_registry import ACTIVITY_REGISTRY
    from app.models.activity_models import _REGISTRY_TO_ENUM_NAME

    assert set(_REGISTRY_TO_ENUM_NAME.keys()) == set(ACTIVITY_REGISTRY.keys())


def test_activity_type_enum_has_no_extra_members_beyond_unknown():
    from app.core.activity_registry import ACTIVITY_REGISTRY
    from app.models.activity_models import ActivityTypeEnum

    # UNKNOWN(1) is the historical sentinel; every other member should map
    # 1:1 onto a registry entry.
    non_unknown = [m for m in ActivityTypeEnum if m.name != "UNKNOWN"]
    assert len(non_unknown) == len(ACTIVITY_REGISTRY)


# ---------------------------------------------------------------------------
# Mock service parity
# ---------------------------------------------------------------------------


def test_mock_service_sees_every_registry_activity():
    """Regression guard for the pre-task-0001 bug where the mock was
    missing ``eating_drinking`` and ``alp_not_standing`` (only 10 of 12).
    """

    from app.core.activity_registry import ACTIVITY_REGISTRY
    from app.services.activity_detection_service import ActivityDetectionService

    service = ActivityDetectionService()

    assert set(service.activity_type_map.keys()) == set(ACTIVITY_REGISTRY.keys())
    assert set(service.activity_descriptions.keys()) == set(ACTIVITY_REGISTRY.keys())
    assert set(service.evidence_rules.keys()) == set(ACTIVITY_REGISTRY.keys())

    # Explicitly assert the two previously missing activities are now present
    # so the regression is called out in the test name.
    assert "eating_drinking" in service.activity_type_map
    assert "alp_not_standing" in service.activity_type_map


def test_mock_service_values_match_registry():
    from app.core.activity_registry import ACTIVITY_REGISTRY
    from app.models.activity_models import ActivityTypeEnum
    from app.services.activity_detection_service import ActivityDetectionService

    service = ActivityDetectionService()

    for name, cfg in ACTIVITY_REGISTRY.items():
        assert service.activity_type_map[name] == ActivityTypeEnum(cfg.type_code)
        assert service.activity_descriptions[name] == cfg.description
        assert service.evidence_rules[name] == cfg.evidence_rule


# ---------------------------------------------------------------------------
# ActivityConfig shape / triggering_role coverage
# ---------------------------------------------------------------------------


def test_triggering_role_is_set_for_role_specific_activities():
    from app.core.activity_registry import ACTIVITY_REGISTRY

    # Role-specific activities should carry a hint for downstream consumers
    # that want to attribute the detection to LP vs ALP.
    assert ACTIVITY_REGISTRY["lp_hand_gesture"].triggering_role == "LP"
    assert ACTIVITY_REGISTRY["alp_hand_gesture"].triggering_role == "ALP"
    assert ACTIVITY_REGISTRY["alp_not_standing"].triggering_role == "ALP"

    # The rest should not assert a role (None is the neutral default).
    neutral = {
        "microsleep",
        "sleep",
        "cell_phone",
        "writing",
        "packing_bags",
        "group_detected",
        "mind_diversion",
        "no_person_detected",
        "eating_drinking",
    }
    for name in neutral:
        assert ACTIVITY_REGISTRY[name].triggering_role is None, name


def test_voting_keys_resolve_without_the_copypasted_map():
    """The two copy-pasted ``activity_key_map`` blocks in the monitor used
    to hard-code a 7-entry identity map. After task 0001 that dispatch is
    done via ``ACTIVITY_REGISTRY[name].voting_key`` — make sure the seven
    voting-verification activities resolve to non-empty keys.
    """

    from app.core.activity_registry import ACTIVITY_REGISTRY

    voting_activities = {
        "mind_diversion",
        "cell_phone",
        "writing",
        "packing_bags",
        "lp_hand_gesture",
        "alp_hand_gesture",
        "eating_drinking",
    }
    for name in voting_activities:
        cfg = ACTIVITY_REGISTRY[name]
        assert cfg.voting_key, f"{name} missing voting_key"
        # In the current pipeline these are identity; the registry allows the
        # field to diverge, but if it ever does, the monitor's dispatch code
        # also needs to change.
        assert cfg.voting_key == name, f"{name} voting_key drifted: {cfg.voting_key}"


# ---------------------------------------------------------------------------
# rebuild_activity_registry invariant
# ---------------------------------------------------------------------------


def test_rebuild_activity_registry_preserves_keys():
    from app.core import activity_registry

    before_keys = set(activity_registry.ACTIVITY_REGISTRY.keys())
    rebuilt = activity_registry.rebuild_activity_registry()
    assert set(rebuilt.keys()) == before_keys == EXPECTED_ACTIVITIES
