"""Stage 8: Per-person sleep state-machine re-gate (H-02 fix).

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4257-4280 (task 0002). The primary per-person gate is applied
inside ``process_all_persons_activities`` before aggregation. This
secondary gate verifies that at least one person with active
sleep/microsleep has their own state machine in DROWSY or beyond.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class StateMachineGateStage:
    name = "state_machine_gate"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        flags = state.activity_flags
        microsleep_detected = flags.get('microsleep_detected', False)
        sleep_detected = flags.get('sleep_detected', False)

        if not (microsleep_detected or sleep_detected):
            return state

        state_machine_ready = False
        for _pidx, _pdata in state.persons_data.items():
            _activities = _pdata.get('activities', {})
            if not (_activities.get('sleep') or _activities.get('microsleep')):
                continue  # Skip persons without active sleep/microsleep
            _sleep_info = _pdata.get('debug_info', {}).get('sleep_info', {})
            _state = _sleep_info.get('sleep_state', 'ALERT')
            if _state in ('DROWSY', 'MICROSLEEP', 'SLEEPING'):
                state_machine_ready = True
                break

        if not state_machine_ready:
            monitor.logger.debug(
                f"[{state.timestamp}] [Frame {state.frame_idx}] "
                f"Sleep/microsleep SUPPRESSED - no person with active sleep/microsleep "
                f"has state machine in DROWSY/MICROSLEEP/SLEEPING"
            )
            flags['microsleep_detected'] = False
            flags['sleep_detected'] = False
        return state
