"""Rule gates applied to raw activity detections.

This module hosts small, pure functions that apply business-rule gates
on top of the raw per-frame activity outputs. They are intentionally kept
free of monitor/state dependencies so they can be unit-tested in isolation.

ARCH-08b: ``apply_train_stopped_suppression`` fixes a latent bug where the
train-STOPPED gate previously only zeroed the aggregated boolean flags
and left each ``persons_data[pidx]['activities']`` dict positive. Any
downstream consumer reading ``persons_data`` (annotation, debug overlays)
would see stale positives. This helper enforces a single consistent
suppression rule across both structures.

``microsleep`` and ``cell_phone`` are intentionally preserved — they
remain safety-critical even when the train is at rest at a station.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)


# Activities that must be suppressed while the train is stopped.
# microsleep and cell_phone are intentionally excluded.
DEFAULT_SUPPRESSED_WHEN_STOPPED: FrozenSet[str] = frozenset({
    'sleep',
    'writing',
    'packing_bags',
    'lp_hand_gesture',
    'alp_hand_gesture',
    'mind_diversion',
    'eating_drinking',
})


# Type alias for the per-detector reset hook.
# Detectors expose ``on_suppressed(person_idx, activity_name)`` to clear
# any internal counters / timestamps tied to the suppressed activity.
SuppressionHook = Callable[[Optional[Any], str], None]


# Mapping of suppressed activity names to the detector keys that own them.
# When the train-stopped gate fires for an activity, every detector listed
# here gets its ``on_suppressed`` hook called. Multiple detectors may want
# the notification (e.g. ``writing`` clears ``mind_diversion`` grace cache)
# so values are lists, not single names.
ACTIVITY_TO_DETECTOR_HOOKS: Mapping[str, List[str]] = {
    'sleep': ['sleep'],
    'writing': ['mind_diversion'],
    'packing_bags': ['activity'],
    'lp_hand_gesture': ['gesture'],
    'alp_hand_gesture': ['gesture'],
    'mind_diversion': ['mind_diversion'],
    # eating_drinking has no per-detector counter to clear today; the
    # aggregated/per-person flag flip is sufficient.
}


def apply_train_stopped_suppression(
    aggregated: MutableMapping[str, bool],
    persons_data: MutableMapping[Any, MutableMapping[str, Any]],
    suppressed: Optional[Iterable[str]] = None,
    detectors: Optional[Mapping[str, Any]] = None,
    previously_suppressed: Optional[
        MutableMapping[Tuple[Any, str], bool]
    ] = None,
) -> None:
    """Zero out suppressed activities in both aggregated + per-person dicts.

    This is the *single* place that applies the train-STOPPED gate. Call it
    whenever the train is known to be stopped (e.g. at a station) so that
    both the aggregated booleans used for downstream logging AND the
    per-person activity dicts used for annotation stay consistent.

    Args:
        aggregated: Flat dict of aggregated boolean flags (e.g.
            ``{'sleep_detected': True, 'writing_detected': False, ...}``).
            Any key ending in ``_detected`` whose prefix is in ``suppressed``
            will be set to ``False``. Plain keys matching a suppressed name
            are also zeroed for flexibility.
        persons_data: Mapping of ``person_idx -> person_dict`` where each
            ``person_dict`` has an ``'activities'`` sub-dict. Each
            suppressed activity found in that sub-dict is set to ``False``.
        suppressed: Optional override of the suppressed activity names.
            Defaults to :data:`DEFAULT_SUPPRESSED_WHEN_STOPPED`.
        detectors: Optional mapping of detector key -> detector instance
            (e.g. ``{'sleep': sleep_detector, 'gesture': gesture_detector,
            ...}``). When provided, every detector that defines an
            ``on_suppressed(person_idx, activity_name)`` method has it
            invoked for each (person, suppressed-activity) pair whose
            activity flag was actually zeroed by this call. This prevents
            internal state machines (sleep duration timers, gesture
            last-raise timestamps, packing direction-change counters,
            mind-diversion writing grace cache) from maturing while the
            activity is suppressed and producing instant false-positives
            on train resume.
        previously_suppressed: Optional mutable mapping of
            ``(person_idx, activity_name) -> True`` tracking which
            (person, activity) pairs already had their ``on_suppressed``
            hook fired in the current STOPPED window. When provided, the
            gate consults+updates this map so each hook fires at most once
            per ``True -> False`` flag flip per STOPPED window — even
            though the underlying detectors may continue to compute the
            activity as positive on every frame the train is stopped. The
            caller is responsible for resetting (clearing) this map when
            the train transitions out of STOPPED so a subsequent stop
            starts fresh. When omitted, hooks fire on every call where the
            flag flipped from True to False (the original semantics).

    Returns:
        None. Both mappings (and any detector state) are mutated in place.
    """
    suppressed_set: FrozenSet[str] = (
        frozenset(suppressed) if suppressed is not None
        else DEFAULT_SUPPRESSED_WHEN_STOPPED
    )

    # 1. Aggregated flat booleans. We accept either ``{name}_detected`` or
    #    bare ``{name}`` forms so callers with different naming conventions
    #    can share the helper.
    for act in suppressed_set:
        detected_key = f'{act}_detected'
        if detected_key in aggregated:
            aggregated[detected_key] = False
        if act in aggregated:
            aggregated[act] = False

    # 2. Per-person activity dicts. Leave unrelated keys (microsleep,
    #    cell_phone, debug fields) untouched. While iterating, fan out to
    #    each detector's ``on_suppressed`` hook so internal counters get
    #    cleared in lockstep with the public flag flip.
    for pidx, pdata in persons_data.items():
        pactivities = pdata.get('activities')
        if not isinstance(pactivities, dict):
            continue
        for act in suppressed_set:
            if act not in pactivities:
                continue
            # Only fire hooks when the flag was actually positive — there's
            # no counter to roll back when the activity was already False.
            had_positive = bool(pactivities.get(act))
            pactivities[act] = False
            if not (had_positive and detectors):
                continue
            # When the caller is tracking already-fired (person, activity)
            # pairs across consecutive STOPPED frames, skip hook fan-out
            # if this pair already fired in the current STOPPED window.
            # The detector state was already cleared on the first
            # True -> False flip; firing again is wasted work.
            if previously_suppressed is not None:
                key = (pidx, act)
                if previously_suppressed.get(key):
                    continue
                previously_suppressed[key] = True
            for detector_key in ACTIVITY_TO_DETECTOR_HOOKS.get(act, ()):
                detector = detectors.get(detector_key)
                hook = getattr(detector, 'on_suppressed', None)
                if callable(hook):
                    try:
                        hook(pidx, act)
                    except Exception:
                        # Hook failures must never break the gate. The
                        # public flag flip already happened above.
                        continue


__all__ = [
    'ACTIVITY_TO_DETECTOR_HOOKS',
    'DEFAULT_SUPPRESSED_WHEN_STOPPED',
    'SuppressionHook',
    'apply_train_stopped_suppression',
]
