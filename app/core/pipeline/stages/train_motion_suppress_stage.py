"""Stage 11: Train-STOPPED activity suppression.

This stage is present to match the 14-stage contract defined in task
0002. In the code revision this worktree is based on, the train-STOPPED
suppression gate does not exist in ``_process_frames_core``; it will
land in a later revision. The stage is a no-op when
``monitor.train_motion_detector`` is absent, and is forward-compatible
— once rebased onto a revision that defines the detector and a
``current_motion_state == 'STOPPED'`` branch, this stage will light up
automatically.

See task 0008 for the "also zero ``persons_data[*]['activities']``"
follow-up, which is left as a TODO here.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class TrainMotionSuppressStage:
    name = "train_motion_suppress"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        detector = getattr(monitor, "train_motion_detector", None)
        current_motion_state = getattr(monitor, "current_motion_state", None)
        if detector is None or current_motion_state != "STOPPED":
            return state
        # When TRAIN_MOTION_SUPPRESS_WHEN_STOPPED=0 the operator wants
        # stopped-state activities to flow through (tagged motionState=STOPPED
        # in the posted record) instead of being zeroed here.
        settings = getattr(monitor, "settings", None)
        if settings is not None and not getattr(
            settings, "train_motion_suppress_when_stopped", True
        ):
            return state

        flags = state.activity_flags

        # Suppress motion-dependent activities when the train is stopped.
        # Per spec: microsleep and cell_phone MUST trigger in both
        # stationary and moving states, so they are NOT zeroed here.
        flags['sleep_detected'] = False
        flags['writing_detected'] = False
        flags['packing_detected'] = False
        flags['lp_hand_gesture_detected'] = False
        flags['alp_hand_gesture_detected'] = False
        flags['mind_diversion_detected'] = False
        flags['eating_drinking_detected'] = False
        # NOTE: lp_not_coordinating / alp_not_coordinating are intentionally
        # NOT reset here — this matches the pre-refactor behavior in newer
        # revisions of the monitor where the coordination-failure flags
        # derived earlier from the hand-gesture signals continue to flow
        # into activities_map when the train is stopped. Changing this
        # would be a behavior change and is deferred to task 0008.

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
        # TODO(task 0008): Also zero persons_data[*]['activities'] so
        # downstream consumers don't see stale per-person flags after
        # suppression. Kept out of this scaffolding pass.
        return state
