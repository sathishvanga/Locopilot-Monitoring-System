"""Stage 1: MediaPipe Face Mesh + frame-buffer bookkeeping.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4098-4109 (task 0002).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import cv2

from app.core.frame_pipeline import FrameState


class FaceMeshStage:
    name = "face_mesh"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        # Convert timestamp to HH:MM:SS format
        state.timestamp = str(timedelta(seconds=state.timestamp_sec))

        # Add frame to buffer
        monitor.frame_buffer.append(state.frame.copy())
        # CR-005: Track frame indices in parallel buffer for activity frame storage
        monitor.frame_idx_buffer.append(state.frame_idx)

        # STEP 1: Run MediaPipe Face Mesh on full frame
        state.rgb_frame = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        state.face_results = monitor.face_mesh.process(state.rgb_frame)
        return state
