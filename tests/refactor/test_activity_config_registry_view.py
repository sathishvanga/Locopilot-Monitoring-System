"""Task 0004: prove ``ActivityTracker`` observes per-activity tunables sourced
from the canonical ``ACTIVITY_REGISTRY`` rather than from a duplicate dataclass.

Before this task there were two parallel ``ActivityConfig`` definitions — one in
``app.core.activity_registry`` (canonical) and a strict subset in
``app.core.activity_tracker`` — which created drift risk: editing one without
the other silently changed behavior.

This test flips ``min_duration`` for a single activity in the live registry
under ``monkeypatch`` and asserts ``ActivityTracker`` reads the new value
through the same ``ActivityConfig`` field set, without any edit to
``activity_tracker.py``.
"""

from app.core import activity_registry
from app.core.activity_registry import ACTIVITY_REGISTRY, ActivityConfig
from app.core.activity_tracker import ActivityConfig as TrackerActivityConfig
from app.core.activity_tracker import ActivityTracker


def test_activity_config_is_registry_view():
    # The tracker module re-exports the registry dataclass — a single source
    # of truth, so they must be the same object.
    assert TrackerActivityConfig is ActivityConfig


def test_tracker_observes_registry_min_duration_change(monkeypatch):
    activity_name = 'cell_phone'
    original_cfg = ACTIVITY_REGISTRY[activity_name]
    new_min_duration = original_cfg.min_duration + 99.0

    # Build a patched registry where exactly one activity has a new
    # ``min_duration``. All other fields and entries are preserved verbatim.
    patched_cfg = ActivityConfig(
        type_code=original_cfg.type_code,
        description=original_cfg.description,
        evidence_rule=original_cfg.evidence_rule,
        triggering_role=original_cfg.triggering_role,
        min_duration=new_min_duration,
        required_consecutive=original_cfg.required_consecutive,
        margin=original_cfg.margin,
        grace_frames=original_cfg.grace_frames,
        region_margin=original_cfg.region_margin,
        wrist_inside_margin=original_cfg.wrist_inside_margin,
        sustained_proximity_seconds=original_cfg.sustained_proximity_seconds,
    )
    patched_registry = dict(ACTIVITY_REGISTRY)
    patched_registry[activity_name] = patched_cfg

    monkeypatch.setattr(activity_registry, 'ACTIVITY_REGISTRY', patched_registry)

    # Construct the tracker from the patched registry — exactly the path the
    # monolith uses (``ActivityTracker(activity_configs=ACTIVITY_REGISTRY,...)``).
    tracker = ActivityTracker(
        activity_configs=patched_registry,
        fps=1.0,
    )

    assert tracker.activity_thresholds[activity_name]['min_duration'] == new_min_duration
    assert tracker.get_threshold(activity_name, 'min_duration') == new_min_duration
    # Other entries remain untouched.
    other = next(name for name in ACTIVITY_REGISTRY if name != activity_name)
    assert (
        tracker.activity_thresholds[other]['min_duration']
        == ACTIVITY_REGISTRY[other].min_duration
    )
