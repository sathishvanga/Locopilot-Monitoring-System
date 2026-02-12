"""
Temporal Filtering Service — Pass 2 of the two-pass deterministic pipeline.

Pass 1 (parallel workers) runs GPU-heavy detection (YOLO, pose, face mesh) and
returns raw per-frame detection flags (activities_map).

Pass 2 (this service) applies temporal filtering (consecutive_detections,
grace_counters) sequentially across ALL frames in order, then generates
evidence clips. Because it runs single-threaded over the globally-sorted
frame list, temporal state never spans chunk boundaries and the result is
fully deterministic.

The temporal filtering logic is an exact mirror of the loop at
locopilot_monitor.py lines 4266-4325:
    - consecutive_detections counter per activity
    - grace_counters per activity
    - required_consecutive threshold to start an activity
    - grace_frames threshold to end an activity
    - min_duration check before producing evidence
"""

import os
import subprocess
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import cv2

try:
    from app.services.vlm_verification_service import VLMVerificationService, parse_reclassify_target
except ImportError:
    VLMVerificationService = None
    parse_reclassify_target = None


class TemporalFilteringService:
    """Applies sequential temporal filtering over raw per-frame detections.

    Produces finalized activity records (same JSON shape as
    LocopilotActivityMonitor.end_activity) ready for the grouping service.
    """

    def __init__(
        self,
        activity_thresholds: Dict[str, Dict[str, Any]],
        activity_type_map: Dict[str, int],
        activity_descriptions: Dict[str, str],
        evidence_rules: Dict[str, str],
        sample_fps: float,
        logger: logging.Logger,
        vlm_service=None,
        clip_buffer_before: float = 1.0,
        clip_buffer_after: float = 1.0,
        settings=None,
    ):
        self.activity_thresholds = activity_thresholds
        self.activity_type_map = activity_type_map
        self.activity_descriptions = activity_descriptions
        self.evidence_rules = evidence_rules
        self.sample_fps = sample_fps
        self.logger = logger
        self.vlm_service = vlm_service
        self.clip_buffer_before = clip_buffer_before
        self.clip_buffer_after = clip_buffer_after
        self.settings = settings

        # VLM activity-name → vlm-type mapping (mirrors _ACT_TO_VLM in monitor)
        self._ACT_TO_VLM = {
            'mind_diversion': 'mind_diversion',
            'sleep': 'sleeping',
            'microsleep': 'microsleep',
            'cell_phone': 'cell_phone',
            'writing': 'writing',
            'packing_bags': 'packing_bags',
            'lp_hand_gesture': 'lp_hand_gesture',
            'alp_hand_gesture': 'alp_hand_gesture',
            'group_detected': 'group_detected',
            'no_person_detected': 'no_person_detected',
            'alp_not_standing': 'alp_not_standing',
            'eating_drinking': 'eating_drinking',
        }
        self._ACT_TO_PA = {
            'mind_diversion': 'mind_diversion',
            'sleep': 'sleep',
            'microsleep': 'microsleep',
            'cell_phone': 'cell_phone',
            'writing': 'writing',
            'packing_bags': 'packing',
            'lp_hand_gesture': 'lp_hand_gesture',
            'alp_hand_gesture': 'alp_hand_gesture',
            'eating_drinking': 'eating_drinking',
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_temporal_filtering(
        self,
        frame_detections: List[Dict],
        video_path: str,
        run_dir: str,
        trip_id: str,
        fps: float,
        sample_fps: float,
        crew_name: str,
        crew_id: str,
        crew_role: int,
        crew_members: Dict,
        camera_angle: int,
        save_clips: bool = True,
    ) -> List[Dict]:
        """Run Pass 2: sequential temporal filtering + evidence generation.

        Args:
            frame_detections: Sorted list of per-frame raw detection dicts
                (from workers' process_video_range_raw).
            video_path: Path to source video.
            run_dir: Run directory for saving clips/images.
            trip_id: Trip identifier.
            fps: Native FPS of the video.
            sample_fps: Sampling rate used during detection.
            crew_name, crew_id, crew_role: Default crew info.
            crew_members: Dict mapping role -> {name, id}.
            camera_angle: 1=LP side, 2=ALP side.
            save_clips: Whether to extract evidence clips.

        Returns:
            List of activity dicts (same shape as monitor's all_activities).
        """
        if not frame_detections:
            return []

        # Ensure sorted by frame_idx for determinism
        frame_detections.sort(key=lambda d: d['frame_idx'])

        self.logger.info(
            f"[PASS 2] Starting temporal filtering over {len(frame_detections)} frames "
            f"(video={os.path.basename(video_path)})"
        )

        # --- Temporal state ---
        activity_names = list(self.activity_thresholds.keys())
        consecutive_detections = {name: 0 for name in activity_names}
        grace_counters = {name: 0 for name in activity_names}

        # Rolling buffer of recent frame indices (mirrors monitor's frame_idx_buffer)
        # Used to populate pre-activity frames when starting an activity
        buffer_size = max(5, int(5 * sample_fps))  # 5 seconds of context
        frame_idx_buffer = deque(maxlen=buffer_size)
        activities = {
            name: {
                'active': False,
                'start_time': None,
                'start_frame': None,
                'frames': [],
                'person_roles': {},
                'first_detection_time': None,
                'last_detection_time': None,
                'last_detected_frame': None,
            }
            for name in activity_names
        }

        all_activities: List[Dict] = []

        # VLM sleep screening: per-person sustained stillness tracker (mirrors monitor)
        vlm_sleep_stillness_tracker: Dict[str, Dict] = {}
        vlm_sleep_enabled = (
            self.vlm_service is not None
            and getattr(self.settings, 'vlm_sleep_screening_enabled', False)
        )

        # Evidence output directory
        clips_dir = os.path.join(run_dir, 'clips') if run_dir else None
        if clips_dir:
            os.makedirs(clips_dir, exist_ok=True)

        evidence_counter = 0

        # Video metadata (cached)
        cap_meta = cv2.VideoCapture(video_path)
        total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
        video_duration_seconds = total_frames / video_fps
        cap_meta.release()

        video_duration_formatted = str(timedelta(seconds=int(video_duration_seconds)))
        video_filename = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]

        # ---------------------------------------------------------------
        # Helper: end an activity and produce evidence
        # ---------------------------------------------------------------
        def _end_activity(activity_name: str, timestamp_str: str, frame_idx: int):
            nonlocal evidence_counter

            act = activities[activity_name]
            if not act['active']:
                return
            act['active'] = False

            total_clip_frames = len(act['frames'])
            actual_clip_duration = total_clip_frames / sample_fps

            min_duration = self.activity_thresholds[activity_name]['min_duration']
            if actual_clip_duration < min_duration:
                self.logger.info(
                    f"[PASS 2] Activity DISCARDED: {activity_name} — too short "
                    f"({actual_clip_duration:.2f}s < {min_duration}s min, {total_clip_frames} frames)"
                )
                act['frames'] = []
                consecutive_detections[activity_name] = 0
                grace_counters[activity_name] = 0
                return

            start_time_str = act['start_time']
            first_detection = act['first_detection_time'] or start_time_str
            last_detection = act['last_detection_time'] or start_time_str

            first_det_sec = _time_to_seconds(first_detection)
            last_det_sec = _time_to_seconds(last_detection)

            activity_start_sec = max(0, first_det_sec - self.clip_buffer_before)
            activity_end_sec = last_det_sec + self.clip_buffer_after
            if activity_end_sec - activity_start_sec < min_duration:
                activity_end_sec = activity_start_sec + min_duration

            # --- File names ---
            start_frame = act.get('start_frame', 0) or 0
            clip_filename = (
                f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}"
                f"_{evidence_counter:03d}_clip.mp4"
            )
            image_filename = (
                f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}"
                f"_{evidence_counter:03d}_activity.jpg"
            )

            clip_path = os.path.join(clips_dir, clip_filename) if clips_dir else clip_filename
            image_path = os.path.join(clips_dir, image_filename) if clips_dir else image_filename

            # --- Extract evidence ---
            if save_clips and clips_dir:
                _extract_clip(video_path, clip_path, activity_start_sec, activity_end_sec)
                _save_activity_image(video_path, act['frames'], image_path)

            # --- Determine performing crew member ---
            activity_crew_name = crew_name
            activity_crew_id = crew_id
            activity_crew_role = crew_role
            performing_role = 'LP'

            person_roles = act.get('person_roles', {})
            if person_roles and crew_members:
                first_person_idx = min(person_roles.keys())
                first_person_role = person_roles[first_person_idx].get('role', 'LP')
                performing_role = first_person_role
                if first_person_role in crew_members:
                    activity_crew_name = crew_members[first_person_role]['name']
                    activity_crew_id = crew_members[first_person_role]['id']
                    activity_crew_role = 1 if first_person_role == 'LP' else 2

            # --- Build JSON record ---
            now = datetime.now()
            json_data = {
                "tripId": trip_id,
                "activityType": self.activity_type_map.get(activity_name, 0),
                "des": self.activity_descriptions.get(activity_name, ''),
                "objectType": activity_name.replace('_', ' '),
                "fileUrl": os.path.abspath(video_path),
                "fileDuration": video_duration_formatted,
                "activityStartTime": f"{activity_start_sec:.2f}",
                "activityEndTime": f"{activity_end_sec:.2f}",
                "videoStartTime": f"{activity_start_sec:.2f}",
                "videoEndTime": f"{activity_end_sec:.2f}",
                "crewName": activity_crew_name,
                "crewId": activity_crew_id,
                "crewRole": activity_crew_role,
                "performingRole": performing_role,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "filename": video_filename,
                "peopleCount": len(person_roles) if person_roles else 1,
                "evidence": {"rule": self.evidence_rules.get(activity_name, '')},
                "activityImage": os.path.abspath(image_path) if clips_dir else image_filename,
                "activityClip": os.path.abspath(clip_path) if clips_dir else clip_filename,
            }

            if person_roles:
                person_roles_list = []
                for pidx in sorted(person_roles.keys()):
                    role_info = person_roles[pidx]
                    person_roles_list.append({
                        "personIndex": pidx,
                        "role": role_info.get('role', 'LP'),
                        "roleName": role_info.get('role_name', 'Loco Pilot'),
                        "bboxArea": role_info.get('bbox_area', 0),
                    })
                json_data["personRoles"] = person_roles_list

            all_activities.append(json_data)

            self.logger.info(
                f"[PASS 2] Activity ENDED: {activity_name} "
                f"(span={first_det_sec:.2f}s–{last_det_sec:.2f}s, "
                f"duration={actual_clip_duration:.2f}s, "
                f"{total_clip_frames} frames, clip: {clip_filename})"
            )

            act['frames'] = []
            consecutive_detections[activity_name] = 0
            grace_counters[activity_name] = 0
            evidence_counter += 1

        # ---------------------------------------------------------------
        # Main temporal filtering loop  (mirrors lines 4266-4325 exactly)
        # ---------------------------------------------------------------
        for det in frame_detections:
            frame_idx = det['frame_idx']
            timestamp_sec = det['timestamp_sec']
            timestamp_str = str(timedelta(seconds=timestamp_sec))
            activities_map = det['activities_map']
            person_roles = det.get('person_roles', {})

            # Maintain rolling buffer of recent frame indices (mirrors monitor's frame_idx_buffer)
            frame_idx_buffer.append(frame_idx)

            # VLM sleep screening: track per-person stillness from raw sleep_debug
            if vlm_sleep_enabled:
                persons_summary = det.get('persons_data_summary', {})
                for pidx, pdata in persons_summary.items():
                    sleep_dbg = pdata.get('sleep_debug', {})
                    avg_vel = sleep_dbg.get('avg_wrist_velocity')
                    movement = sleep_dbg.get('movement')
                    if avg_vel is None or movement is None:
                        continue

                    # Skip if sleep/microsleep already detected for this person
                    p_acts = pdata.get('activities', {})
                    if p_acts.get('sleep') or p_acts.get('microsleep'):
                        continue

                    vel_thresh = getattr(self.settings, 'vlm_sleep_stillness_velocity', 0.01)
                    mov_thresh = getattr(self.settings, 'vlm_sleep_stillness_movement', 5.0)

                    tkey = f"vlm_sleep_{pidx}"
                    if tkey not in vlm_sleep_stillness_tracker:
                        vlm_sleep_stillness_tracker[tkey] = {
                            'consecutive_still': 0, 'last_vlm_time': 0,
                        }
                    trk = vlm_sleep_stillness_tracker[tkey]

                    if avg_vel < vel_thresh and movement < mov_thresh:
                        trk['consecutive_still'] += 1
                    else:
                        trk['consecutive_still'] = 0

                    req_frames = getattr(self.settings, 'vlm_sleep_stillness_frames', 30)
                    cooldown = getattr(self.settings, 'vlm_sleep_screening_cooldown', 60.0)

                    if (trk['consecutive_still'] >= req_frames
                            and timestamp_sec - trk['last_vlm_time'] > cooldown):
                        self.logger.info(
                            f"[PASS 2] [{timestamp_str}] [VLM SLEEP SCREEN] Person {pidx}: "
                            f"{trk['consecutive_still']} frames still "
                            f"(vel={avg_vel:.4f}, mov={movement:.2f}) → calling VLM"
                        )
                        try:
                            cap = cv2.VideoCapture(video_path)
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, vlm_frame = cap.read()
                            finally:
                                cap.release()
                            if ret and vlm_frame is not None:
                                p_bbox = pdata.get('bbox', [0, 0, vlm_frame.shape[1], vlm_frame.shape[0]])
                                is_confirmed, details = self.vlm_service.verify_detection_sync(
                                    vlm_frame, list(p_bbox), 'sleeping', pidx
                                )
                                trk['last_vlm_time'] = timestamp_sec
                                if is_confirmed:
                                    # Inject microsleep into activities_map so temporal filtering picks it up
                                    activities_map['microsleep'] = True
                                    self.logger.info(
                                        f"[PASS 2] [{timestamp_str}] [VLM SLEEP SCREEN] Person {pidx}: "
                                        f"VLM CONFIRMED → microsleep=True"
                                    )
                                else:
                                    trk['consecutive_still'] = 0
                                    self.logger.info(
                                        f"[PASS 2] [{timestamp_str}] [VLM SLEEP SCREEN] Person {pidx}: "
                                        f"VLM REJECTED → reset"
                                    )
                        except Exception as e:
                            self.logger.error(f"[PASS 2] [VLM SLEEP SCREEN] Error: {e}")

            # --- Fix 5: Conflicting activity suppression ---
            # packing_bags takes priority over writing when both detected in same frame
            if activities_map.get('packing_bags') and activities_map.get('writing'):
                activities_map['writing'] = False
                self.logger.debug(
                    f"[PASS 2] [{timestamp_str}] Suppressed writing (packing_bags active in same frame)"
                )

            # --- Fix 3: Writing max duration cap ---
            # Force-end excessively long writing to allow re-evaluation
            # (may actually be sleeping/resting misclassified as writing)
            _writing_max_dur = getattr(self.settings, 'writing_max_duration', 120.0) if self.settings else 120.0
            if activities['writing']['active'] and activities['writing']['first_detection_time']:
                writing_start_sec = _time_to_seconds(activities['writing']['first_detection_time'])
                if timestamp_sec - writing_start_sec > _writing_max_dur:
                    self.logger.info(
                        f"[PASS 2] [{timestamp_str}] Writing max duration ({_writing_max_dur:.0f}s) exceeded "
                        f"→ force-ending for re-evaluation"
                    )
                    _end_activity('writing', timestamp_str, frame_idx)
                    # Suppress writing for this frame so it doesn't immediately restart
                    activities_map['writing'] = False

            # Track reclassified activities for post-loop processing (Fix 1)
            reclassified_this_frame = set()

            for activity_name, detected in activities_map.items():
                if activity_name not in consecutive_detections:
                    continue

                if detected:
                    consecutive_detections[activity_name] += 1
                    grace_counters[activity_name] = 0

                    required = self.activity_thresholds[activity_name]['required_consecutive']

                    # Log consecutive frame buildup
                    if not activities[activity_name]['active']:
                        self.logger.debug(
                            f"[PASS 2] [{timestamp_str}] [TEMPORAL] {activity_name}: consecutive "
                            f"{consecutive_detections[activity_name]}/{required}"
                        )

                    if consecutive_detections[activity_name] >= required:
                        if not activities[activity_name]['active']:
                            self.logger.info(
                                f"[PASS 2] [{timestamp_str}] [TEMPORAL] {activity_name}: threshold reached "
                                f"({consecutive_detections[activity_name]}/{required} consecutive frames)"
                            )
                            # VLM one-shot verification before starting
                            if self.vlm_service is not None:
                                vlm_confirmed, vlm_reason = self._vlm_verify_at_start(
                                    activity_name, video_path, frame_idx,
                                    det.get('persons_data_summary', {})
                                )
                                if not vlm_confirmed:
                                    # Extract reclassification target once
                                    reclassify_target = parse_reclassify_target(vlm_reason) if parse_reclassify_target else None

                                    # Fix 4: Self-reclassification = VLM confusion
                                    if reclassify_target and reclassify_target == activity_name:
                                        self.logger.info(
                                            f"[PASS 2] [{timestamp_str}] [TEMPORAL] {activity_name}: "
                                            f"VLM self-reclassified → treating as rejection"
                                        )
                                        reclassify_target = None  # Don't process as reclassification

                                    self.logger.info(
                                        f"[PASS 2] [{timestamp_str}] [TEMPORAL] {activity_name}: VLM rejected "
                                        f"→ resetting consecutive counter (was {consecutive_detections[activity_name]})"
                                    )
                                    # Reclassify to a different activity if VLM suggested one
                                    if reclassify_target and reclassify_target in consecutive_detections:
                                        activities_map[reclassify_target] = True
                                        consecutive_detections[reclassify_target] = consecutive_detections.get(reclassify_target, 0) + 1
                                        grace_counters[reclassify_target] = 0
                                        reclassified_this_frame.add(reclassify_target)
                                        self.logger.info(
                                            f"[PASS 2] [{timestamp_str}] [TEMPORAL] {activity_name}: "
                                            f"VLM reclassified → {reclassify_target} "
                                            f"(consecutive={consecutive_detections[reclassify_target]})"
                                        )
                                    consecutive_detections[activity_name] = 0
                                    grace_counters[activity_name] = 0
                                    continue

                            # Start activity (initialize frames with buffer for pre-context)
                            activities[activity_name]['active'] = True
                            activities[activity_name]['start_time'] = timestamp_str
                            activities[activity_name]['start_frame'] = frame_idx
                            activities[activity_name]['frames'] = list(frame_idx_buffer)
                            activities[activity_name]['person_roles'] = person_roles
                            activities[activity_name]['first_detection_time'] = timestamp_str
                            activities[activity_name]['last_detection_time'] = timestamp_str
                            activities[activity_name]['last_detected_frame'] = frame_idx
                            roles_str = ""
                            if person_roles:
                                roles_str = ", persons=" + "+".join(
                                    f"p{pidx}({info.get('role', '?')})" for pidx, info in sorted(person_roles.items())
                                )
                            vlm_str = " [VLM confirmed]" if self.vlm_service is not None else ""
                            self.logger.info(
                                f"[PASS 2] [{timestamp_str}] Activity STARTED: {activity_name}{vlm_str}"
                                f" (frame={frame_idx}{roles_str})"
                            )

                        if activities[activity_name]['active']:
                            activities[activity_name]['frames'].append(frame_idx)
                            activities[activity_name]['last_detected_frame'] = frame_idx
                            activities[activity_name]['last_detection_time'] = timestamp_str
                            if person_roles:
                                activities[activity_name]['person_roles'] = person_roles
                else:
                    if consecutive_detections[activity_name] > 0 or activities[activity_name]['active']:
                        grace_counters[activity_name] += 1
                        grace_frames = self.activity_thresholds[activity_name]['grace_frames']

                        if grace_counters[activity_name] <= grace_frames:
                            # Log first entry into grace period
                            if grace_counters[activity_name] == 1:
                                state = "ACTIVE" if activities[activity_name]['active'] else f"building ({consecutive_detections[activity_name]} consecutive)"
                                self.logger.debug(
                                    f"[PASS 2] [{timestamp_str}] [GRACE] {activity_name}: not detected, "
                                    f"entering grace period (1/{grace_frames} frames, state={state})"
                                )
                        else:
                            if activities[activity_name]['active']:
                                self.logger.info(
                                    f"[PASS 2] [{timestamp_str}] [GRACE] {activity_name}: grace period exceeded "
                                    f"({grace_counters[activity_name]}/{grace_frames}) → ending activity"
                                )
                                _end_activity(activity_name, timestamp_str, frame_idx)
                            else:
                                self.logger.debug(
                                    f"[PASS 2] [{timestamp_str}] [GRACE] {activity_name}: grace period exceeded "
                                    f"→ resetting counter (was {consecutive_detections[activity_name]} consecutive)"
                                )
                            consecutive_detections[activity_name] = 0
                            grace_counters[activity_name] = 0
                    else:
                        grace_counters[activity_name] = 0

            # --- Fix 1: Post-loop reclassification check ---
            # When an activity (e.g. writing) is reclassified to another (e.g. sleep)
            # whose dict iteration already passed, the target's threshold check was skipped.
            # Process reclassified targets here to start activities immediately.
            for target in reclassified_this_frame:
                if target not in consecutive_detections:
                    continue
                if activities[target]['active']:
                    # Already active — just track this frame as a detection
                    activities[target]['frames'].append(frame_idx)
                    activities[target]['last_detected_frame'] = frame_idx
                    activities[target]['last_detection_time'] = timestamp_str
                    if person_roles:
                        activities[target]['person_roles'] = person_roles
                    continue

                required = self.activity_thresholds[target]['required_consecutive']
                if consecutive_detections[target] < required:
                    continue

                self.logger.info(
                    f"[PASS 2] [{timestamp_str}] [RECLASSIFY] {target}: threshold reached "
                    f"({consecutive_detections[target]}/{required}) → starting activity"
                )
                # Start activity (VLM already suggested this — skip re-verification)
                activities[target]['active'] = True
                activities[target]['start_time'] = timestamp_str
                activities[target]['start_frame'] = frame_idx
                activities[target]['frames'] = list(frame_idx_buffer)
                activities[target]['person_roles'] = person_roles
                activities[target]['first_detection_time'] = timestamp_str
                activities[target]['last_detection_time'] = timestamp_str
                activities[target]['last_detected_frame'] = frame_idx
                activities[target]['frames'].append(frame_idx)
                roles_str = ""
                if person_roles:
                    roles_str = ", persons=" + "+".join(
                        f"p{pidx}({info.get('role', '?')})" for pidx, info in sorted(person_roles.items())
                    )
                self.logger.info(
                    f"[PASS 2] [{timestamp_str}] Activity STARTED (via reclassification): "
                    f"{target} (frame={frame_idx}{roles_str})"
                )

        # End any remaining active activities
        if frame_detections:
            last_det = frame_detections[-1]
            final_ts = str(timedelta(seconds=last_det['timestamp_sec']))
            final_frame = last_det['frame_idx']
            for activity_name in activity_names:
                if activities[activity_name]['active']:
                    _end_activity(activity_name, final_ts, final_frame)

        self.logger.info(
            f"[PASS 2] Temporal filtering complete: {len(all_activities)} activities detected"
        )
        return all_activities

    # ------------------------------------------------------------------
    # VLM verification (mirrors monitor._vlm_verify_at_start)
    # ------------------------------------------------------------------

    def _vlm_verify_at_start(
        self,
        activity_name: str,
        video_path: str,
        frame_idx: int,
        persons_data_summary: Dict,
    ) -> Tuple[bool, str]:
        """One-shot VLM verification when temporal threshold is met.

        Extracts the frame from the video, crops the person bbox, and
        calls the VLM service to verify.

        Returns:
            Tuple of (is_confirmed, reason_string)
        """
        vlm_type = self._ACT_TO_VLM.get(activity_name)
        if vlm_type is None:
            return True, ""

        # Extract frame from video
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            self.logger.warning(f"[VLM PASS2] Could not read frame {frame_idx}")
            return True, ""  # On error, let it through

        # Determine bbox: find the person who triggered this activity
        pa_key = self._ACT_TO_PA.get(activity_name)
        person_bbox = None
        person_idx = 0

        # Hand gesture is a two-person coordination activity — VLM needs to
        # see both LP and ALP to judge whether coordination is happening.
        is_hand_gesture = activity_name in ('lp_hand_gesture', 'alp_hand_gesture')

        if is_hand_gesture and persons_data_summary and len(persons_data_summary) >= 2:
            # Compute a combined bbox encompassing all persons
            all_bboxes = [
                pdata.get('bbox') for pdata in persons_data_summary.values()
                if pdata.get('bbox')
            ]
            if all_bboxes:
                person_bbox = [
                    min(b[0] for b in all_bboxes),
                    min(b[1] for b in all_bboxes),
                    max(b[2] for b in all_bboxes),
                    max(b[3] for b in all_bboxes),
                ]
                person_idx = 0  # Not person-specific
        elif pa_key and persons_data_summary:
            # First try to find the person who triggered this specific activity
            for pidx, pdata in persons_data_summary.items():
                if pdata.get('activities', {}).get(pa_key, False):
                    bbox = pdata.get('bbox')
                    if bbox:
                        person_bbox = list(bbox)
                        person_idx = pidx
                        break
            # Fallback: use first person's bbox
            if person_bbox is None:
                for pidx, pdata in persons_data_summary.items():
                    bbox = pdata.get('bbox')
                    if bbox:
                        person_bbox = list(bbox)
                        person_idx = pidx
                        break

        if person_bbox is None:
            person_bbox = [0, 0, frame.shape[1], frame.shape[0]]

        try:
            is_confirmed, details = self.vlm_service.verify_detection_sync(
                frame, person_bbox, vlm_type, person_idx
            )
            reason = details.get('reason', '')
            if is_confirmed:
                self.logger.info(
                    f"[VLM PASS2] {vlm_type} p{person_idx} CONFIRMED "
                    f"(conf={details.get('confidence', '?')})"
                )
            else:
                self.logger.info(
                    f"[VLM PASS2] {vlm_type} p{person_idx} REJECTED "
                    f"(conf={details.get('confidence', '?')}, reason={reason})"
                )
            return is_confirmed, reason
        except Exception as e:
            self.logger.error(f"[VLM PASS2] Error verifying {activity_name}: {e}")
            return True, ""


# ------------------------------------------------------------------
# Module-level helpers (no state)
# ------------------------------------------------------------------

try:
    from app.utils.time_utils import time_to_seconds as _time_to_seconds
except ImportError:
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS[.microseconds] to seconds."""
        parts = time_str.split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def _extract_clip(
    video_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
) -> bool:
    """Extract a video clip via ffmpeg (H.264 for browser compat)."""
    duration = end_seconds - start_seconds
    if duration <= 0:
        return False

    ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
    cmd = [
        ffmpeg_path, '-y',
        '-ss', str(start_seconds),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',
        '-movflags', '+faststart',
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def _save_activity_image(
    video_path: str,
    frame_indices: List[int],
    output_path: str,
) -> bool:
    """Save the middle frame of an activity as a JPEG image."""
    if not frame_indices:
        return False

    middle_idx = frame_indices[len(frame_indices) // 2]
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame = cap.read()
    finally:
        cap.release()

    if ret and frame is not None:
        cv2.imwrite(output_path, frame)
        return True
    return False
