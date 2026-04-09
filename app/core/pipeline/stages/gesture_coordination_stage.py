"""Stage 10: Hand-gesture coordination check + OCR + no-person flag.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4383-4404 (task 0002). Runs the LP/ALP coordination-failure check
and extracts the OCR timestamp from the frame.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class GestureCoordinationStage:
    name = "gesture_coordination"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        flags = state.activity_flags

        # CRITICAL: Hand gesture coordination check
        # Activity Type 8 (LP not exchanging): Triggers when ALP raises hand BUT LP does NOT
        # Activity Type 9 (ALP not exchanging): Triggers when LP raises hand BUT ALP does NOT
        # This ensures we detect COORDINATION FAILURES, not individual gestures
        # Uses temporal window to prevent false positives when both people raise hands within window
        lp_not_coordinating, alp_not_coordinating = monitor._check_hand_gesture_coordination(
            flags.get('lp_hand_gesture_detected', False),
            flags.get('alp_hand_gesture_detected', False),
            state.timestamp_sec,
        )
        state.lp_not_coordinating = lp_not_coordinating
        state.alp_not_coordinating = alp_not_coordinating

        # Debug logging for coordination check
        if lp_not_coordinating and monitor.consecutive_detections['lp_hand_gesture'] == 0:
            monitor.logger.info(
                f"[{state.timestamp}] [Frame {state.frame_idx}] "
                f"COORDINATION FAILURE: ALP raised hand but LP did NOT respond"
            )
        if alp_not_coordinating and monitor.consecutive_detections['alp_hand_gesture'] == 0:
            monitor.logger.info(
                f"[{state.timestamp}] [Frame {state.frame_idx}] "
                f"COORDINATION FAILURE: LP raised hand but ALP did NOT respond"
            )

        # Detect when no person is in frame
        state.no_person_detected_flag = (
            len(state.detections.get('deduplicated_person', [])) == 0
        )

        # Extract OCR timestamp from frame (if enabled)
        state.ocr_timestamp = monitor._extract_ocr_timestamp(state.frame)
        return state
