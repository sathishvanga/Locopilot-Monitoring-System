"""Stage 7: Per-person activity dispatch.

Wraps ``LocopilotActivityMonitor.process_all_persons_activities`` — a thin
shim for task 0002. Per-activity decomposition is an explicit follow-up.

Extracted verbatim from ``_process_frames_core`` lines ~4182-4255. In
addition to calling ``process_all_persons_activities`` the stage also
populates ``state.activity_flags`` with the same flat flag names the
original function used, and performs the per-person logging (only when
``log_per_person_detections`` is set).
"""

from __future__ import annotations

from typing import Any

from app.core.frame_pipeline import FrameState


class PerPersonActivitiesStage:
    name = "per_person_activities"

    def run(self, state: FrameState, monitor: Any) -> FrameState:
        # STEP 4: *** MULTI-PERSON PROCESSING ***
        # Process ALL persons individually for ALL activities.
        # GPU BATCH: Pass pre-computed pose results if available.
        precomputed_poses = None
        if (
            state.batch_pose_results is not None
            and state.batch_idx < len(state.batch_pose_results)
        ):
            precomputed_poses = state.batch_pose_results[state.batch_idx]

        precomputed_sleep_poses = None
        if (
            state.batch_sleep_pose_results is not None
            and state.batch_idx < len(state.batch_sleep_pose_results)
        ):
            precomputed_sleep_poses = state.batch_sleep_pose_results[state.batch_idx]

        multi_person_results = monitor.process_all_persons_activities(
            state.frame,
            state.detections,
            state.person_roles,
            state.timestamp_sec,
            state.face_results,
            state.frame_idx,
            precomputed_pose_results=precomputed_poses,
            precomputed_sleep_pose_results=precomputed_sleep_poses,
        )

        # Extract aggregated detection flags
        state.persons_data = multi_person_results['persons']
        state.aggregated = multi_person_results['aggregated']
        aggregated = state.aggregated

        # Initialize detection flags from aggregated results
        state.activity_flags = {
            'microsleep_detected': aggregated['microsleep_detected'],
            'sleep_detected': aggregated['sleep_detected'],
            'cell_phone_detected': aggregated['cell_phone_detected'],
            'writing_detected': aggregated['writing_detected'],
            'packing_detected': aggregated['packing_detected'],
            'lp_hand_gesture_detected': aggregated['lp_hand_gesture_detected'],
            'alp_hand_gesture_detected': aggregated['alp_hand_gesture_detected'],
            'mind_diversion_detected': aggregated['mind_diversion_detected'],
            'eating_drinking_detected': aggregated.get('eating_drinking_detected', False),
        }

        # Log detections for each person (only on first detection)
        if state.log_per_person_detections:
            timestamp = state.timestamp
            for person_idx, person_data in state.persons_data.items():
                activities = person_data['activities']
                role_name = person_data['role_name']
                debug_info = person_data['debug_info']

                # Log mind diversion
                if (
                    activities.get('mind_diversion', False)
                    and monitor.consecutive_detections.get('mind_diversion', 0) == 0
                ):
                    head_pose = debug_info.get('head_pose', {})
                    yaw = head_pose.get('yaw', 0)
                    pitch = head_pose.get('pitch', 0)
                    method = head_pose.get('method', 'unknown')
                    monitor.logger.info(
                        f"[{timestamp}] MIND DIVERSION detected for {role_name} "
                        f"(Person {person_idx+1}) - Yaw={yaw:.1f}, Pitch={pitch:.1f} "
                        f"(method: {method})"
                    )

                # Log sleep detection
                if activities['sleep'] and monitor.consecutive_detections['sleep'] == 0:
                    monitor.logger.info(
                        f"[{timestamp}] SLEEP detected for {role_name} "
                        f"(Person {person_idx+1}) - pose-based"
                    )

                # Log microsleep detection
                if (
                    activities['microsleep']
                    and monitor.consecutive_detections['microsleep'] == 0
                ):
                    monitor.logger.info(
                        f"[{timestamp}] MICROSLEEP detected for {role_name} "
                        f"(Person {person_idx+1}) - pose-based"
                    )

                # Log hand gestures
                if (
                    activities['lp_hand_gesture']
                    and monitor.consecutive_detections['lp_hand_gesture'] == 0
                ):
                    gesture_debug = debug_info.get('gesture_debug', {})
                    monitor.logger.info(
                        f"[{timestamp}] LP hand gesture detected for {role_name} "
                        f"(Person {person_idx+1}) - "
                        f"{gesture_debug.get('hand_raised', 'unknown')} hand raised"
                    )

                if (
                    activities['alp_hand_gesture']
                    and monitor.consecutive_detections['alp_hand_gesture'] == 0
                ):
                    gesture_debug = debug_info.get('gesture_debug', {})
                    monitor.logger.info(
                        f"[{timestamp}] ALP hand gesture detected for {role_name} "
                        f"(Person {person_idx+1}) - "
                        f"{gesture_debug.get('hand_raised', 'unknown')} hand raised"
                    )

                # Log cell phone, writing, packing
                if (
                    activities['cell_phone']
                    and monitor.consecutive_detections['cell_phone'] == 0
                ):
                    monitor.logger.info(
                        f"[{timestamp}] Cell phone ACTIVELY USED by {role_name} "
                        f"(Person {person_idx+1})"
                    )

                if activities['writing'] and monitor.consecutive_detections['writing'] == 0:
                    monitor.logger.info(
                        f"[{timestamp}] WRITING detected for {role_name} "
                        f"(Person {person_idx+1})"
                    )

                if (
                    activities.get('packing_bags', False)
                    and monitor.consecutive_detections.get('packing_bags', 0) == 0
                ):
                    monitor.logger.info(
                        f"[{timestamp}] PACKING detected for {role_name} "
                        f"(Person {person_idx+1})"
                    )
        return state
