"""Unit tests for ARCH-08b: train-STOPPED suppression gate.

These tests verify that :func:`app.core.gates.apply_train_stopped_suppression`
zeroes both the aggregated boolean flags AND the per-person activity dicts
consistently, while preserving safety-critical activities (``microsleep``,
``cell_phone``).
"""

from __future__ import annotations

import os
import sys

import pytest


# Ensure the repo root is on sys.path so "app.core.gates" imports cleanly
# regardless of where pytest is invoked from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.core.gates import (  # noqa: E402
    DEFAULT_SUPPRESSED_WHEN_STOPPED,
    apply_train_stopped_suppression,
)


def _make_fixture():
    """Build a representative aggregated + persons_data pair.

    Everything is set to ``True`` so we can verify *which* keys flip and
    which remain untouched.
    """
    aggregated = {
        'sleep_detected': True,
        'microsleep_detected': True,  # must remain True (safety-critical)
        'writing_detected': True,
        'packing_detected': True,  # note: unrelated name, should stay True
        'packing_bags_detected': True,
        'lp_hand_gesture_detected': True,
        'alp_hand_gesture_detected': True,
        'mind_diversion_detected': True,
        'eating_drinking_detected': True,
        'cell_phone_detected': True,  # must remain True (safety-critical)
    }

    persons_data = {
        0: {
            'role': 'LP',
            'activities': {
                'sleep': True,
                'microsleep': True,
                'writing': True,
                'packing_bags': True,
                'lp_hand_gesture': True,
                'alp_hand_gesture': False,
                'mind_diversion': True,
                'eating_drinking': True,
                'cell_phone': True,
            },
        },
        1: {
            'role': 'ALP',
            'activities': {
                'sleep': True,
                'microsleep': False,
                'writing': False,
                'packing_bags': True,
                'lp_hand_gesture': False,
                'alp_hand_gesture': True,
                'mind_diversion': True,
                'eating_drinking': False,
                'cell_phone': True,
            },
        },
    }
    return aggregated, persons_data


def test_aggregated_flags_zeroed_except_safety_critical():
    aggregated, persons_data = _make_fixture()

    apply_train_stopped_suppression(aggregated, persons_data)

    # Suppressed aggregated flags must be False.
    assert aggregated['sleep_detected'] is False
    assert aggregated['writing_detected'] is False
    assert aggregated['packing_bags_detected'] is False
    assert aggregated['lp_hand_gesture_detected'] is False
    assert aggregated['alp_hand_gesture_detected'] is False
    assert aggregated['mind_diversion_detected'] is False
    assert aggregated['eating_drinking_detected'] is False

    # Safety-critical flags must remain True.
    assert aggregated['microsleep_detected'] is True
    assert aggregated['cell_phone_detected'] is True

    # A differently-named key that doesn't match any suppressed prefix
    # must not be touched.
    assert aggregated['packing_detected'] is True


def test_per_person_activity_flags_zeroed_except_safety_critical():
    aggregated, persons_data = _make_fixture()

    apply_train_stopped_suppression(aggregated, persons_data)

    for pidx, pdata in persons_data.items():
        acts = pdata['activities']

        # Suppressed per-person flags must be False.
        for suppressed_name in DEFAULT_SUPPRESSED_WHEN_STOPPED:
            assert acts.get(suppressed_name) is False, (
                f"person {pidx} activity {suppressed_name} should be False "
                f"after suppression but was {acts.get(suppressed_name)!r}"
            )

        # Safety-critical per-person flags must be untouched.
        # (We set them True in the fixture where applicable.)
        assert acts['cell_phone'] is True
        # microsleep was True for person 0 and False for person 1 — either
        # way the helper must not flip its value.
        if pidx == 0:
            assert acts['microsleep'] is True
        else:
            assert acts['microsleep'] is False


def test_aggregated_and_per_person_are_consistent_after_gate():
    """The whole point of ARCH-08b: both structures agree."""
    aggregated, persons_data = _make_fixture()

    apply_train_stopped_suppression(aggregated, persons_data)

    for act in DEFAULT_SUPPRESSED_WHEN_STOPPED:
        agg_key = f'{act}_detected'
        assert aggregated.get(agg_key) is False, (
            f"aggregated[{agg_key}] should be False"
        )
        for pidx, pdata in persons_data.items():
            acts = pdata['activities']
            if act in acts:
                assert acts[act] is False, (
                    f"persons_data[{pidx}]['activities'][{act}] should be False"
                )


def test_handles_empty_persons_data():
    aggregated = {
        'sleep_detected': True,
        'microsleep_detected': True,
        'cell_phone_detected': True,
    }
    persons_data = {}

    apply_train_stopped_suppression(aggregated, persons_data)

    assert aggregated['sleep_detected'] is False
    assert aggregated['microsleep_detected'] is True
    assert aggregated['cell_phone_detected'] is True


def test_handles_person_without_activities_dict():
    aggregated = {'sleep_detected': True}
    persons_data = {
        0: {'role': 'LP'},  # no 'activities' key at all
        1: {'role': 'ALP', 'activities': None},  # wrong type
        2: {'role': 'ALP', 'activities': {'sleep': True, 'cell_phone': True}},
    }

    # Must not raise.
    apply_train_stopped_suppression(aggregated, persons_data)

    assert aggregated['sleep_detected'] is False
    assert persons_data[2]['activities']['sleep'] is False
    assert persons_data[2]['activities']['cell_phone'] is True


def test_custom_suppressed_set_override():
    aggregated = {
        'sleep_detected': True,
        'writing_detected': True,
    }
    persons_data = {
        0: {'activities': {'sleep': True, 'writing': True}},
    }

    # Override: only suppress 'writing', leave 'sleep' alone.
    apply_train_stopped_suppression(
        aggregated, persons_data, suppressed={'writing'}
    )

    assert aggregated['sleep_detected'] is True
    assert aggregated['writing_detected'] is False
    assert persons_data[0]['activities']['sleep'] is True
    assert persons_data[0]['activities']['writing'] is False


def test_monitor_class_constants_match_helper_default():
    """Guard against drift between locopilot_monitor.SUPPRESSED_WHEN_STOPPED
    and app.core.gates.DEFAULT_SUPPRESSED_WHEN_STOPPED.

    We import the monitor module directly to grab the class constant without
    instantiating the heavy ``LocopilotActivityMonitor``.
    """
    import importlib.util

    monitor_path = os.path.join(REPO_ROOT, 'locopilot_monitor.py')
    if not os.path.exists(monitor_path):
        pytest.skip("locopilot_monitor.py not present")

    # locopilot_monitor imports heavy ML deps at module load. If those
    # aren't installed in the test env, skip the drift check rather than
    # failing — the helper itself is already covered by the other tests.
    try:
        spec = importlib.util.spec_from_file_location(
            'locopilot_monitor', monitor_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"locopilot_monitor import unavailable: {exc}")

    monitor_cls = getattr(module, 'LocopilotActivityMonitor', None)
    assert monitor_cls is not None, (
        "LocopilotActivityMonitor class not found in locopilot_monitor.py"
    )

    assert hasattr(monitor_cls, 'SUPPRESSED_WHEN_STOPPED'), (
        "ARCH-08b: LocopilotActivityMonitor.SUPPRESSED_WHEN_STOPPED missing"
    )
    # NOTE: ARCH-08a's VOTING_ACTIVITY_KEY_MAP class constant was superseded
    # by ARCH-01's ACTIVITY_REGISTRY.voting_key field during the merge —
    # voting key lookups now flow through the registry directly. The drift
    # guard for that mapping lives in test_activity_registry.py instead.

    assert (
        frozenset(monitor_cls.SUPPRESSED_WHEN_STOPPED)
        == DEFAULT_SUPPRESSED_WHEN_STOPPED
    ), (
        "Monitor class constant and app.core.gates default must stay in sync"
    )
