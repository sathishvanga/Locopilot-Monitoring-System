"""Stage 2: YOLO object detection.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4111-4122 (task 0002). Uses the pre-computed batch detection if
available, otherwise falls back to per-frame detection (with a dark/IR
preprocess pass when batch mode is active but this particular frame is a
fallback).
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class ObjectDetectStage:
    name = "object_detect"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        # STEP 2: Detect objects with YOLO
        # GPU BATCH: Use pre-computed detections from batch inference if available
        if (
            state.batch_object_detections is not None
            and state.batch_idx < len(state.batch_object_detections)
        ):
            state.detections = state.batch_object_detections[state.batch_idx]
        else:
            # Per-frame detection (process_video path or fallback)
            # Preprocess single frame for dark/IR conditions if batch mode fallback
            if state.batch_object_detections is not None:
                detection_frame = monitor.object_detector._preprocess_frames_for_detection(
                    [state.frame]
                )[0]
                state.detections = monitor.object_detector.detect_objects(
                    detection_frame, None, use_pose_guided=False
                )
            else:
                state.detections = monitor.object_detector.detect_objects(
                    state.frame, None, use_pose_guided=False
                )
        return state
