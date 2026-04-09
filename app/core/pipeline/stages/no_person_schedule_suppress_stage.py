"""Stage 12: Build ``activities_map`` + no-person / trip-schedule suppression.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4406-4429 (task 0002). Combines the flat activity flags into the
``activities_map`` dict and then applies the trip-schedule suppression
for ``no_person_detected`` (which must only trigger when we have a
schedule that lets us distinguish station halts from true no-person
frames).
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class NoPersonScheduleSuppressStage:
    name = "no_person_schedule_suppress"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        flags = state.activity_flags
        microsleep_detected = flags.get('microsleep_detected', False)
        sleep_detected = flags.get('sleep_detected', False)

        # Update activity states with temporal filtering
        state.activities_map = {
            'microsleep': microsleep_detected and not sleep_detected,
            'sleep': sleep_detected,
            'cell_phone': flags.get('cell_phone_detected', False),
            'writing': flags.get('writing_detected', False),
            'packing_bags': flags.get('packing_detected', False),
            'group_detected': state.group_detected_flag,
            # LP fails to respond when ALP raises hand
            'lp_hand_gesture': state.lp_not_coordinating,
            # ALP fails to respond when LP raises hand
            'alp_hand_gesture': state.alp_not_coordinating,
            'mind_diversion': flags.get('mind_diversion_detected', False),
            'eating_drinking': flags.get('eating_drinking_detected', False),
            'no_person_detected': state.no_person_detected_flag,
            'alp_not_standing': False,
        }

        # 3.5. Suppress no_person_detected when trip schedule is unavailable
        if monitor.suppress_no_person_without_schedule and monitor.trip_schedule is None:
            if state.activities_map.get('no_person_detected', False):
                state.activities_map['no_person_detected'] = False
                monitor.logger.debug(
                    f"[{state.timestamp}] no_person_detected suppressed - "
                    f"no trip schedule available to distinguish station halts"
                )
        return state
