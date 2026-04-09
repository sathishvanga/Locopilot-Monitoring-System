"""Stage 3: Person box deduplication.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4124-4180 (task 0002). Sets ``state.people_count``,
``state.detections['deduplicated_person']`` and initialises
``state.person_roles`` (left empty here — filled in ``RoleIdentifyStage``).
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState
from app.core.utils.geometry import deduplicate_person_boxes


class PersonDedupStage:
    name = "person_dedup"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        detections = state.detections

        # STEP 3: Identify person roles and count people
        people_count = len(detections.get('person', []))
        if people_count == 0:
            people_count = 1  # Default to 1 if no person detected
        state.people_count = people_count

        # De-duplicate person boxes and identify roles
        state.person_roles = {}

        if len(detections.get('person', [])) > 0:
            # De-duplicate person boxes to get accurate count.
            # Increased IOU threshold from 0.3 to 0.5 to better filter duplicate detections.
            deduplicated_persons = deduplicate_person_boxes(
                detections['person'], iou_threshold=0.5
            )

            # Store deduplicated boxes in detections for visualization.
            # NOTE: Removed pose validation as it was filtering out legitimate people
            # (MediaPipe struggles with back views, partial occlusions, overhead cameras).
            detections['deduplicated_person'] = deduplicated_persons
        else:
            # No person detected at all
            detections['deduplicated_person'] = []
            state.person_roles = {}
            # DEBUG: Log when no person is detected (will be tracked as activity)
            if monitor.consecutive_detections['no_person_detected'] == 0:
                raw_detections = len(detections.get('person', []))
                monitor.logger.debug(
                    f"[{state.timestamp}] NO PERSON detected in frame "
                    f"(raw YOLO detections: {raw_detections})"
                )
        return state
