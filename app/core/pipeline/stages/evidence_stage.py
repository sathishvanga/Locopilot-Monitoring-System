"""Stage 14: Stale-person cleanup + progress logging.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4482-4500 (task 0002). In the original function this was the last
body of work before the try/finally cleanup. Named ``EvidenceStage`` for
consistency with the task spec; the actual evidence writing (start /
end_activity calls) lives in ``TemporalFilterStage``.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class EvidenceStage:
    name = "evidence"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        # CR-012: Clean up stale per-person tracking dicts after each frame
        if state.enable_stale_cleanup and state.persons_data:
            monitor._cleanup_stale_person_tracking(set(state.persons_data.keys()))

        # Display progress with detection status (every 50 sampled frames)
        if state.sample_idx % 50 == 0:
            progress = (state.frame_idx / state.total_frames) * 100
            monitor.logger.info(
                f"Progress: {state.sample_idx} samples processed "
                f"(frame {state.frame_idx}/{state.total_frames}, {progress:.1f}%)"
            )

            # Show current detection counts for debugging
            active_detections = []
            for act_name, count in monitor.consecutive_detections.items():
                if count > 0:
                    threshold = monitor.activity_thresholds[act_name]['required_consecutive']
                    status = (
                        "RECORDING"
                        if monitor.activities[act_name]['active']
                        else f"building {count}/{threshold}"
                    )
                    active_detections.append(f"{act_name}: {status}")

            if active_detections:
                monitor.logger.debug(
                    f"  Active detections: {', '.join(active_detections)}"
                )
        return state
