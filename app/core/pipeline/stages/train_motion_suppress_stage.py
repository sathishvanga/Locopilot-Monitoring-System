"""Stage 11: Train-STOPPED activity suppression.

Delegates to :func:`app.core.gates.apply_train_stopped_suppression` so that
the aggregated boolean flags AND each ``persons_data[*]['activities']``
sub-dict are zeroed in lockstep, and so per-detector state machines get
their ``on_suppressed`` hooks invoked. This is the single source of truth
for the train-STOPPED gate — see ``app/core/gates.py``.

``microsleep`` and ``cell_phone`` are intentionally left active because
they remain safety-critical even at stations.
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.frame_pipeline import FrameState
from app.core.gates import apply_train_stopped_suppression


class TrainMotionSuppressStage:
    name = "train_motion_suppress"

    def _collect_detectors(self, monitor: Any) -> Dict[str, Any]:
        """Build the ``detectors`` map passed to the gate helper.

        Detector attributes are looked up defensively so the stage stays
        no-op when running against a partial monitor (e.g. unit tests that
        only populate ``train_motion_detector``).
        """
        keys = (
            ('sleep', 'sleep_detector'),
            ('gesture', 'gesture_detector'),
            ('mind_diversion', 'mind_diversion_detector'),
            ('activity', 'activity_detector'),
        )
        out: Dict[str, Any] = {}
        for short_key, attr in keys:
            det = getattr(monitor, attr, None)
            if det is not None:
                out[short_key] = det
        return out

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        detector = getattr(monitor, "train_motion_detector", None)
        current_motion_state = getattr(monitor, "current_motion_state", None)
        if detector is None or current_motion_state != "STOPPED":
            # Train is no longer stopped (or detector absent): clear the
            # per-window already-fired (person, activity) tracker so the
            # NEXT entry into a STOPPED window starts fresh and fires the
            # ``on_suppressed`` hooks exactly once per flag flip again.
            tracker = getattr(monitor, "_train_stopped_suppressed_pairs", None)
            if tracker:
                tracker.clear()
            return state
        # When TRAIN_MOTION_SUPPRESS_WHEN_STOPPED=0 the operator wants
        # stopped-state activities to flow through (tagged motionState=STOPPED
        # in the posted record) instead of being zeroed here.
        settings = getattr(monitor, "settings", None)
        if settings is not None and not getattr(
            settings, "train_motion_suppress_when_stopped", True
        ):
            return state

        # Maintain a per-monitor mapping that records which
        # (person_idx, activity) pairs already had their ``on_suppressed``
        # hook fired in the current contiguous STOPPED window. The gate
        # consults+updates this map so each detector hook fires once per
        # True -> False flag flip per window — without it, every frame the
        # detector recomputes the activity as positive and we'd re-fire
        # the hook on every frame the train is stopped (wasted work, and
        # log noise).
        tracker = getattr(monitor, "_train_stopped_suppressed_pairs", None)
        if tracker is None:
            tracker = {}
            monitor._train_stopped_suppressed_pairs = tracker

        # Single-call gate: zero aggregated + per-person flags AND fan out
        # the per-detector ``on_suppressed`` hook so internal counters
        # (sleep duration, gesture last-raise time, packing direction
        # changes, mind-diversion writing grace) reset in lockstep — and
        # only on the first flag-flip per STOPPED window.
        apply_train_stopped_suppression(
            state.activity_flags,
            state.persons_data,
            detectors=self._collect_detectors(monitor),
            previously_suppressed=tracker,
        )
        # Compatibility bridge — required because ``per_person_activities_stage``
        # emits the legacy key ``'packing_detected'`` (no ``_bags`` suffix)
        # alongside the canonical ``'packing_bags_detected'``. The canonical
        # gate flips the registry-keyed ``'packing_bags_detected'`` flag, but
        # any downstream stage that still reads the legacy alias would
        # otherwise see a stale True during a STOPPED window. Mirror the flip
        # here so both forms agree. (See per_person_activities_stage.py:58-79
        # for the dual-emit rationale.)
        if 'packing_detected' in state.activity_flags:
            state.activity_flags['packing_detected'] = False

        # group_detected: raise threshold when stopped (only when the newer
        # ``train_motion_stopped_group_threshold`` attribute is present).
        stopped_threshold = getattr(
            monitor, "train_motion_stopped_group_threshold", None
        )
        if stopped_threshold is not None and state.group_detected_flag:
            person_count = len(state.detections.get('deduplicated_person', []))
            if person_count <= stopped_threshold:
                state.group_detected_flag = False

        monitor.logger.debug(
            f"[{state.timestamp}] [Frame {state.frame_idx}] Train STOPPED - "
            f"sleep/writing/packing/gesture/mind_diversion/eating suppressed; "
            f"microsleep and cell_phone remain active (safety-critical)"
        )
        return state
