"""Single source of truth for activity metadata.

This module consolidates all per-activity configuration (tracking thresholds,
type codes, descriptions, evidence rules, triggering roles) into one registry
so downstream consumers (monitor, mock detection service, Pydantic enum,
multiprocessing workers) cannot silently drift.

Extracted from ``locopilot_monitor.py`` during the 2026-04 architecture review
(task 0001). Previously this metadata lived in four hand-written parallel
dicts on ``LocopilotActivityMonitor.__init__`` plus a partial ``ACTIVITY_REGISTRY``
dataclass inside the monolith, plus a divergent copy in
``app/services/activity_detection_service.py`` that was missing
``eating_drinking`` and ``alp_not_standing``.

Design notes:

* ``ActivityConfig`` is a strict superset of the older
  ``app/core/activity_tracker.ActivityConfig`` — the original runtime fields
  (``min_duration``, ``required_consecutive``, ``margin``, ``grace_frames``,
  ``region_margin``, ``wrist_inside_margin``, ``sustained_proximity_seconds``)
  are preserved verbatim so ``ActivityTracker`` continues to accept the same
  objects via duck typing.
* Reporting metadata fields (``type_code``, ``description``, ``evidence_rule``,
  ``triggering_role``) are the metadata the monitor used to keep in parallel
  dicts.
* ``ACTIVITY_REGISTRY`` is built by a function (not a module-level literal) so
  the config-driven margins resolve lazily via ``get_settings()``. The
  resulting dict is cached on first access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:  # Real runtime import.
    from app.utils.config import get_settings
except Exception:  # pragma: no cover - defensive fallback for partial envs.
    get_settings = None  # type: ignore[assignment]


@dataclass
class ActivityConfig:
    """Per-activity configuration used to seed runtime tracking dicts.

    The dataclass intentionally groups three concerns that used to be spread
    across four parallel dicts:

    1. Temporal filtering thresholds (``min_duration``, ``required_consecutive``,
       ``grace_frames``).
    2. Proximity margins (``margin``, ``region_margin``, ``wrist_inside_margin``,
       ``sustained_proximity_seconds``).
    3. Reporting metadata (``type_code``, ``description``, ``evidence_rule``,
       ``triggering_role``) previously in the monitor's
       ``activity_type_map``/``activity_descriptions``/``evidence_rules`` dicts.
    """

    # --- Reporting metadata (new in task 0001) ---------------------------
    # ``type_code`` must match ``app.models.activity_models.ActivityTypeEnum``;
    # an assertion in ``activity_models`` enforces this invariant at import.
    type_code: int = 0
    description: str = ""
    evidence_rule: str = ""
    triggering_role: Optional[str] = None  # "LP" | "ALP" | None

    # --- Temporal filtering thresholds ----------------------------------
    min_duration: float = 0.0
    required_consecutive: int = 1
    margin: Optional[int] = None
    grace_frames: int = 5

    # --- Extra threshold fields used by specific activities -------------
    region_margin: Optional[int] = None
    wrist_inside_margin: Optional[int] = None
    sustained_proximity_seconds: Optional[float] = None


def _build_activity_registry() -> Dict[str, ActivityConfig]:
    """Construct the canonical activity registry.

    Called lazily so ``get_settings()`` can supply config-driven margins
    without forcing import-time settings resolution.
    """

    try:
        _settings = get_settings() if get_settings is not None else None
    except Exception:
        _settings = None

    cell_phone_margin = _settings.activity_cell_phone_margin if _settings else 180
    writing_margin = _settings.activity_writing_margin if _settings else 180
    packing_margin = _settings.activity_packing_margin if _settings else 100
    packing_region_margin = (
        _settings.activity_packing_region_margin if _settings else 150
    )
    packing_wrist_inside_margin = (
        _settings.activity_packing_wrist_inside_margin if _settings else 80
    )

    return {
        # Reporting metadata columns must mirror the values that used to live
        # in ``LocopilotActivityMonitor.__init__`` at lines 697-742 of
        # ``locopilot_monitor.py`` before task 0001.
        'microsleep': ActivityConfig(
            type_code=3,
            description='Micro-sleep detected (5+ seconds)',
            evidence_rule='pose_indicators',
            triggering_role=None,
            # F2 (2026-04): eyes closed > 5 sec; at 0.5 fps, 3 consecutive
            # frames span >=4 sec, and min_duration=5.0 enforces the full
            # 5-second threshold after onset.
            min_duration=5.0,
            required_consecutive=3,
            margin=None,
            grace_frames=10,
        ),
        'sleep': ActivityConfig(
            type_code=4,
            description='Sleep detected (30+ seconds)',
            evidence_rule='pose_indicators',
            triggering_role=None,
            min_duration=2.0,
            required_consecutive=1,
            margin=None,
            grace_frames=10,
        ),
        'cell_phone': ActivityConfig(
            type_code=2,
            description='Using mobile phone',
            evidence_rule='phone_in_hand',
            triggering_role=None,
            min_duration=0.1,
            # Raised 2026-04-20 from 1 → 3 after pose-fallback addition:
            # at 1, momentary hand-to-face (eye rubs, scratches) fired the
            # fallback as cell_phone (26 detections vs ~4 real events in
            # all_activities.mp4). At 3 (6s at 0.5fps), real phone calls
            # (15-48s in ground-truth) easily pass; brief gestures don't.
            required_consecutive=3,
            margin=cell_phone_margin,
            grace_frames=8,
        ),
        'writing': ActivityConfig(
            type_code=5,
            description='WRITING LOG BOOK WHILE RUNNING',
            evidence_rule='hand_near_book_or_wrist_proximity',
            triggering_role=None,
            # 2026-04-23: consecutive 2->1. Writing v4 rule has two high-precision
            # paths (book-bbox-inside AND pose-only wrists-together-in-lap).
            # Each fire is trustworthy — requiring 2 consecutive samples drops
            # the sparse short GT events (TV22.5_0447 5s, TV22.9 5s, TV22.8 3s).
            min_duration=2.0,
            required_consecutive=1,
            margin=writing_margin,
            grace_frames=10,
        ),
        'packing_bags': ActivityConfig(
            type_code=6,
            description='Packing bags activity detected',
            evidence_rule='wrist_inside_backpack_bbox_or_hand_near_backpack',
            triggering_role=None,
            min_duration=0.0,
            required_consecutive=1,
            margin=packing_margin,
            grace_frames=5,
            region_margin=packing_region_margin,
            wrist_inside_margin=packing_wrist_inside_margin,
            sustained_proximity_seconds=4.0,
        ),
        'group_detected': ActivityConfig(
            type_code=7,
            description='More than 2 people (group) detected',
            evidence_rule='more_than_2_deduplicated_persons',
            triggering_role=None,
            min_duration=0.0,
            required_consecutive=3,
            margin=None,
            grace_frames=8,
        ),
        'lp_hand_gesture': ActivityConfig(
            type_code=8,
            description='LP not exchanging hand gesture',
            evidence_rule='lp_hand_raised_gesture_detected',
            triggering_role='LP',
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'alp_hand_gesture': ActivityConfig(
            type_code=9,
            description='ALP not exchanging hand gesture',
            evidence_rule='alp_hand_raised_gesture_detected',
            triggering_role='ALP',
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'mind_diversion': ActivityConfig(
            type_code=10,
            # Sub-type (looking_sideways, looking_down_distracted,
            # looking_away_combined) is stored in evidence details.
            description='Mind diversion - attention diverted from controls',
            evidence_rule='attention_diverted_from_controls',
            triggering_role=None,
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'no_person_detected': ActivityConfig(
            type_code=11,
            description='No person detected in frame',
            evidence_rule='zero_persons_in_frame',
            triggering_role=None,
            # F4 (2026-04-06): consecutive 3->5, min_duration 5->10 to
            # suppress intermittent YOLO recall drops on non-canonical poses.
            min_duration=10.0,
            required_consecutive=5,
            margin=None,
            grace_frames=3,
        ),
        'alp_not_standing': ActivityConfig(
            type_code=12,
            description='ALP not standing during pre-arrival window',
            evidence_rule='alp_seated_during_pre_arrival_window',
            triggering_role='ALP',
            required_consecutive=2,
            grace_frames=3,
        ),
        'eating_drinking': ActivityConfig(
            type_code=13,
            description='Eating or drinking detected',
            evidence_rule='cup_or_bottle_near_face',
            triggering_role=None,
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'solo_person': ActivityConfig(
            type_code=14,
            description='Only one person in cabin while train running',
            evidence_rule='exactly_one_deduplicated_person_while_running',
            triggering_role=None,
            # Mirror no_person_detected timing: 10s sustained at 0.5 fps
            # (5 consecutive samples) so brief ALP-turning-away-from-camera
            # frames don't fire. Suppressed while train is STOPPED — see
            # ``DEFAULT_SUPPRESSED_WHEN_STOPPED`` in ``app/core/gates.py``.
            min_duration=10.0,
            required_consecutive=5,
            margin=None,
            grace_frames=3,
        ),
    }


# Cached canonical registry. Built at import time so downstream modules
# (including the Pydantic enum and the mock service) can import synchronously.
ACTIVITY_REGISTRY: Dict[str, ActivityConfig] = _build_activity_registry()


def rebuild_activity_registry() -> Dict[str, ActivityConfig]:
    """Rebuild the cached registry in-place.

    Provided for tests and live config reloads that need to pick up new
    margin values without importing a fresh module.
    """

    global ACTIVITY_REGISTRY
    ACTIVITY_REGISTRY = _build_activity_registry()
    return ACTIVITY_REGISTRY


def is_activity_enabled(activity_name: str) -> bool:
    """Return True iff ``activity_name`` is enabled by the operator.

    Thin wrapper around ``Settings.is_activity_enabled`` so callers can
    import the gate from the same module that owns ``ACTIVITY_REGISTRY``
    without having to plumb a settings object through every call site.

    Falls back to ``True`` when settings cannot be resolved (partial test
    envs) — fail-open matches the registry's "all activities run by
    default" contract.
    """

    if get_settings is None:
        return True
    try:
        settings = get_settings()
    except Exception:
        return True
    return settings.is_activity_enabled(activity_name)


__all__ = [
    "ActivityConfig",
    "ACTIVITY_REGISTRY",
    "rebuild_activity_registry",
    "is_activity_enabled",
]
