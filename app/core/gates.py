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

from typing import Any, Dict, FrozenSet, Iterable, MutableMapping, Optional


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


def apply_train_stopped_suppression(
    aggregated: MutableMapping[str, bool],
    persons_data: MutableMapping[Any, MutableMapping[str, Any]],
    suppressed: Optional[Iterable[str]] = None,
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

    Returns:
        None. Both mappings are mutated in place.
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
    #    cell_phone, debug fields) untouched.
    for _pidx, pdata in persons_data.items():
        pactivities = pdata.get('activities')
        if not isinstance(pactivities, dict):
            continue
        for act in suppressed_set:
            if act in pactivities:
                pactivities[act] = False


__all__ = [
    'DEFAULT_SUPPRESSED_WHEN_STOPPED',
    'apply_train_stopped_suppression',
]
