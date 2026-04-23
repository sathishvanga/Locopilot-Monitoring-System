"""Stage 5: Group detection (count-based).

Flags ``state.group_detected_flag = True`` when the deduplicated person
count exceeds the configured threshold (``TRAIN_MOTION_RUNNING_GROUP_THRESHOLD``,
default 5 → fires at 6+). Extracted from
``LocopilotActivityMonitor._process_frames_core`` lines ~4154-4172 (task 0002).

History: this stage previously ran a voting-verification pass to confirm
the group-count trigger before raising the flag. The voting layer was
removed in 2026-04-18 once the domain-trained YOLO v8 weights made the
re-verification layer redundant; the stage now trips purely on the
deduplicated-count threshold.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class GroupDetectStage:
    name = "group_detect"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        deduplicated_persons = state.detections.get('deduplicated_person', [])
        if not deduplicated_persons:
            return state

        deduplicated_count = len(deduplicated_persons)
        threshold = getattr(monitor, 'train_motion_running_group_threshold', 5)
        if deduplicated_count <= threshold:
            return state

        state.group_detected_flag = True

        if monitor.consecutive_detections['group_detected'] == 0:
            monitor.logger.info(
                f"[{state.timestamp}] Group detected - {deduplicated_count} people "
                f"(de-duplicated from {len(state.detections['person'])} raw detections)"
            )
        return state
