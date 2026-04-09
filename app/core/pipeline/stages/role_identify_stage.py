"""Stage 4: Person role identification (LP / ALP / etc.).

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4144-4152 (task 0002). Runs only when deduplicated persons exist —
otherwise ``state.person_roles`` was already set to ``{}`` by
``PersonDedupStage``.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class RoleIdentifyStage:
    name = "role_identify"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        deduplicated_persons = state.detections.get('deduplicated_person', [])
        if not deduplicated_persons:
            return state

        # Identify person roles (LP, ALP, etc.)
        state.person_roles = monitor.identify_person_roles(
            state.frame, deduplicated_persons, state.detections
        )

        # Log role identification (only once per detection cycle)
        if (
            monitor.consecutive_detections['group_detected'] == 0
            and state.person_roles
        ):
            monitor.logger.debug(f"[{state.timestamp}] Person roles identified:")
            for person_idx in sorted(state.person_roles.keys()):
                role_info = state.person_roles[person_idx]
                monitor.logger.debug(
                    f"  Person {person_idx+1}: {role_info['role_name']} "
                    f"(bbox_area: {role_info.get('bbox_area', 0):.0f})"
                )
        return state
