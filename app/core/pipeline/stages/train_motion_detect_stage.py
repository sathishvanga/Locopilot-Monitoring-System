"""Stage 6: Vibration-based train motion detection.

This stage is present to match the 14-stage contract defined in task
0002. In the code revision this worktree is based on, train motion
detection has not yet landed — ``monitor.train_motion_detector`` does
not exist. The stage therefore no-ops on missing attributes and is
forward-compatible: once the monitor is rebased onto a revision that
defines ``train_motion_detector``, this stage will automatically start
running without any further edits.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class TrainMotionDetectStage:
    name = "train_motion_detect"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        detector = getattr(monitor, "train_motion_detector", None)
        if detector is None:
            return state

        person_bboxes_for_motion = [
            p[:4].astype(int) if hasattr(p, 'astype') else [int(c) for c in p[:4]]
            for p in state.detections.get('person', [])
        ]
        motion_state, motion_conf, motion_diag = detector.process_frame(
            state.frame, person_bboxes_for_motion
        )
        monitor.current_motion_state = motion_state
        state.motion_state = motion_state
        if monitor.logger:
            monitor.logger.debug(
                f"[{state.timestamp}] [Frame {state.frame_idx}] Train motion: {motion_state} "
                f"(conf={motion_conf:.2f}, vib={motion_diag['vib_mean']:.2f}, "
                f"combined={motion_diag['combined_score']:.3f})"
            )
        return state
