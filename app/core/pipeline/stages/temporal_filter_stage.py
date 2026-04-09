"""Stage 13: Per-frame temporal filtering (consecutive counters + grace).

Extracted verbatim from ``LocopilotActivityMonitor._process_frames_core``
lines ~4431-4480 (task 0002). Increments consecutive-frame counters for
each detected activity, starts/continues activity recordings once the
required-consecutive threshold is met, and ends activities when grace
frames are exceeded.

This stage mutates monitor-side state (``consecutive_detections``,
``grace_counters``, ``activities``) but reads exclusively from
``state.activities_map``, so reordering earlier stages does not break it.
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class TemporalFilterStage:
    name = "temporal_filter"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        activities_map = state.activities_map
        timestamp = state.timestamp
        people_count = state.people_count

        for activity_name, detected in activities_map.items():
            if detected:
                # Activity detected - increment consecutive counter and reset grace period
                monitor.consecutive_detections[activity_name] += 1
                monitor.grace_counters[activity_name] = 0  # Reset grace period

                # Only start recording after required consecutive frames threshold is met
                required_consecutive = monitor.activity_thresholds[activity_name][
                    'required_consecutive'
                ]

                if monitor.consecutive_detections[activity_name] >= required_consecutive:
                    # Start activity if not already active
                    if not monitor.activities[activity_name]['active']:
                        monitor.start_activity(
                            activity_name,
                            timestamp,
                            state.fps,
                            state.frame_idx,
                            person_roles=state.person_roles,
                            ocr_timestamp=state.ocr_timestamp,
                        )

                    # Continue recording frames ONLY when activity is actively detected
                    if monitor.activities[activity_name]['active']:
                        # CR-005: Store frame index instead of frame copy to reduce memory usage
                        monitor.activities[activity_name]['frames'].append(state.frame_idx)
                        monitor.activities[activity_name]['last_frame_count'] = state.frame_idx
                        # Track last actual detection
                        monitor.activities[activity_name]['last_detected_frame'] = state.frame_idx
                        # Track for precise clip duration
                        monitor.activities[activity_name]['last_detection_time'] = timestamp
                        # Update person roles (in case they change during activity)
                        if state.person_roles:
                            monitor.activities[activity_name]['person_roles'] = state.person_roles
            else:
                # Activity not detected - use grace period before resetting
                if (
                    monitor.consecutive_detections[activity_name] > 0
                    or monitor.activities[activity_name]['active']
                ):
                    # Increment grace counter
                    monitor.grace_counters[activity_name] += 1
                    grace_frames = monitor.activity_thresholds[activity_name]['grace_frames']

                    # If still within grace period, keep activity alive but DON'T add frames
                    if monitor.grace_counters[activity_name] <= grace_frames:
                        # Still in grace period - keep activity active but don't record frames
                        # This allows brief interruptions without ending the activity
                        pass
                    else:
                        # Grace period exceeded - end activity and reset counters
                        if monitor.activities[activity_name]['active']:
                            # No ocr_timestamp: current frame is post-grace-period, not when
                            # activity ended. end_activity computes ocr_end from
                            # ocr_start + duration instead (more accurate).
                            if state.save_clips:
                                monitor.end_activity(
                                    activity_name, timestamp, state.fps,
                                    state.frame_idx, people_count,
                                )
                            else:
                                monitor.end_activity(
                                    activity_name, timestamp, state.fps,
                                    state.frame_idx, people_count,
                                    save_clips=state.save_clips,
                                )
                        monitor.consecutive_detections[activity_name] = 0
                        monitor.grace_counters[activity_name] = 0
                else:
                    # Reset grace counter if nothing is being tracked
                    monitor.grace_counters[activity_name] = 0
        return state
