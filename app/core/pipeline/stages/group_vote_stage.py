"""Stage 5: Group detection + voting verification.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4154-4172 (task 0002). Only fires when the deduplicated person
count exceeds 2.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class GroupVoteStage:
    name = "group_vote"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        deduplicated_persons = state.detections.get('deduplicated_person', [])
        if not deduplicated_persons:
            return state

        deduplicated_count = len(deduplicated_persons)
        if deduplicated_count <= 2:
            return state

        # Stage 2: Voting verification for group_detected (if enabled)
        if monitor.voting_service is not None:
            is_confirmed, vote_details = monitor.voting_service.verify_activity(
                video_path=monitor.current_video_path,
                timestamp_sec=state.timestamp_sec,
                activity_type='group_detected',
                # Full frame for group detection
                person_bbox=[0, 0, state.frame.shape[1], state.frame.shape[0]],
            )
            if is_confirmed:
                state.group_detected_flag = True
                monitor.logger.info(
                    f"[VOTING] group_detected CONFIRMED: "
                    f"{vote_details.get('vote_breakdown', [])}"
                )
            else:
                state.group_detected_flag = False
                monitor.logger.info(
                    f"[VOTING] group_detected REJECTED: "
                    f"{vote_details.get('vote_breakdown', [])}"
                )
        else:
            state.group_detected_flag = True

        if (
            state.group_detected_flag
            and monitor.consecutive_detections['group_detected'] == 0
        ):
            monitor.logger.info(
                f"[{state.timestamp}] Group detected - {deduplicated_count} people "
                f"(de-duplicated from {len(state.detections['person'])} raw detections)"
            )
        return state
