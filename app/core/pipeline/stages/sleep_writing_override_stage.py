"""Stage 9: Sleep vs writing override + annotated-frame rendering.

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4282-4381 (task 0002). Covers:

* The chicken-and-egg sleep-vs-writing override block.
* Debug logging after the override check.
* The ``annotated_frame_for_activity`` rendering chain (draw boxes,
  multi-person mediapipe outputs, per-person sleep debug overlays).
* Periodic annotated-frame saving when enabled.

These were all in the same contiguous region of the original function and
share the same per-frame scope, so they stay together rather than being
split into four micro-stages. If this becomes a pain point during
follow-up per-activity decomposition we can split it further.
"""

from __future__ import annotations

import os
from typing import Any

import cv2

from app.core.frame_pipeline import FrameState


class SleepWritingOverrideStage:
    name = "sleep_writing_override"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        flags = state.activity_flags
        sleep_detected = flags.get('sleep_detected', False)
        microsleep_detected = flags.get('microsleep_detected', False)
        writing_detected = flags.get('writing_detected', False)
        cell_phone_detected = flags.get('cell_phone_detected', False)
        packing_detected = flags.get('packing_detected', False)
        timestamp = state.timestamp

        # CRITICAL: Exclude sleep detection if person is holding objects or in active posture
        # If someone has a phone, book, or backpack in hand, they're clearly NOT sleeping
        # EXCEPTION: If the sleep state machine is in DROWSY/MICROSLEEP/SLEEPING, don't let
        # writing suppress sleep — during microsleep, hands-in-lap + head-down can look like
        # writing posture but the state machine has already determined the person is drowsy.
        # FIX: Also check raw drowsiness indicators from pose_sleep_info directly, so that
        # on the first frame of sleep onset (when state machine is still ALERT), strong
        # drowsiness signals can override writing suppression and allow the state machine
        # to advance. This prevents the chicken-and-egg problem where writing suppresses
        # sleep before the state machine ever reaches DROWSY.
        sleep_state_overrides_writing = False
        if sleep_detected or microsleep_detected:
            # H-02 fix: Only check persons who have active sleep/microsleep,
            # not all persons. This prevents cross-person state leakage where
            # person 0's SLEEPING state overrides writing suppression for person 1.
            for _pidx, _pdata in state.persons_data.items():
                _activities = _pdata.get('activities', {})
                if not (_activities.get('sleep') or _activities.get('microsleep')):
                    continue  # Skip persons without active sleep/microsleep
                _sleep_info = _pdata.get('debug_info', {}).get('sleep_info', {})
                _state = _sleep_info.get('sleep_state', 'ALERT')
                if _state in ('DROWSY', 'MICROSLEEP', 'SLEEPING'):
                    sleep_state_overrides_writing = True
                    break
                # Check raw drowsiness indicators even when state machine is still ALERT.
                # head_drop_detected or significant nose_y_drop indicate the person's head
                # is dropping -- a strong physical signal that should not be suppressed by
                # writing detection. haar_eye_closed similarly indicates closed eyes.
                _head_drop = _sleep_info.get('head_drop_detected', False)
                _nose_y_drop = _sleep_info.get('nose_y_drop', 0.0)
                _haar_eye_closed = _sleep_info.get('haar_eye_closed', False)
                if _head_drop or _nose_y_drop > 0.05 or _haar_eye_closed:
                    sleep_state_overrides_writing = True
                    monitor.logger.debug(
                        f"[{timestamp}] Writing suppression overridden by raw drowsiness "
                        f"indicators: head_drop={_head_drop}, nose_y_drop={_nose_y_drop:.4f}, "
                        f"haar_eye_closed={_haar_eye_closed}"
                    )
                    break

        state.sleep_state_overrides_writing = sleep_state_overrides_writing

        suppress_activities = cell_phone_detected or packing_detected
        if writing_detected and not sleep_state_overrides_writing:
            suppress_activities = True
        if suppress_activities:
            if state.log_per_person_detections and (microsleep_detected or sleep_detected):
                reason = []
                if cell_phone_detected:
                    reason.append("phone")
                if writing_detected and not sleep_state_overrides_writing:
                    reason.append("book")
                if packing_detected:
                    reason.append("backpack")
                monitor.logger.debug(
                    f"[{timestamp}] Sleep detection OVERRIDDEN - "
                    f"person active ({', '.join(reason)})"
                )
            microsleep_detected = False
            sleep_detected = False
            flags['microsleep_detected'] = False
            flags['sleep_detected'] = False

        # Debug: log sleep detection state after override check
        if sleep_detected or microsleep_detected:
            monitor.logger.info(
                f"[{timestamp}] [Frame {state.frame_idx}] SLEEP/MICROSLEEP PASSED override check: "
                f"sleep={sleep_detected}, microsleep={microsleep_detected}, "
                f"writing={writing_detected}, override={sleep_state_overrides_writing}"
            )

        # Create annotated frame with all detections (pose landmarks + YOLO boxes)
        # This annotated frame will be used for BOTH activity clips AND periodic frame saving
        annotated_frame_for_activity = monitor.frame_annotator.draw_bounding_boxes(
            state.frame, state.detections, show_roi_boxes=True, person_roles=state.person_roles
        )
        # NEW: Draw MediaPipe outputs for ALL persons (not just one)
        annotated_frame_for_activity = monitor.draw_multi_person_mediapipe_outputs(
            annotated_frame_for_activity,
            state.persons_data,  # All persons' pose landmarks and activities
            state.face_results,
        )

        # Draw sleep detection debug overlay for each person
        for pidx, pdata in state.persons_data.items():
            sleep_info = pdata.get('debug_info', {}).get('sleep_info')
            if sleep_info:
                annotated_frame_for_activity = monitor.frame_annotator.draw_sleep_debug_overlay(
                    annotated_frame_for_activity, sleep_info, pidx,
                    pdata.get('activities', {}), state.timestamp_sec,
                )

        state.annotated_frame_for_activity = annotated_frame_for_activity

        # Save annotated frames periodically if enabled (AFTER all detections)
        if (
            monitor.save_annotated_frames
            and monitor.frames_dir is not None
            and state.sample_idx % monitor.frame_save_interval == 0
        ):
            try:
                # Save frame with unique filename
                frame_filename = f"frame_{state.frame_idx:08d}.jpg"
                frame_path = os.path.join(monitor.frames_dir, frame_filename)

                # Ensure directory exists (for multiprocessing safety)
                os.makedirs(monitor.frames_dir, exist_ok=True)

                # Save with high quality
                cv2.imwrite(
                    frame_path,
                    annotated_frame_for_activity,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            except Exception as e:
                monitor.logger.error(
                    f"[{timestamp}] Error saving frame {state.frame_idx}: {e}"
                )
        return state
