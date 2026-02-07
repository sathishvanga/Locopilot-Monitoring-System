"""Person tracking and role assignment for locomotive cab monitoring.

This module extracts person tracking logic from locopilot_monitor.py,
providing a reusable PersonTracker class for identifying crew members
(LP/ALP) and maintaining temporal consistency of role assignments.

Role Assignment Logic:
- camera_angle=1 (LP Side): closest person (largest bbox) = LP, further = ALP
- camera_angle=2 (ALP Side): closest person (largest bbox) = ALP, further = LP
- Third+ persons = "Visitor"

Temporal Tracking:
- Uses IoU matching to prevent role flipping between frames
- Maintains previous frame's boxes and roles for consistency
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

# Import IoU calculation from geometry utils
from app.core.utils.geometry import calculate_iou


class PersonTracker:
    """Tracks persons across frames and assigns LP/ALP/Visitor roles.

    This class provides:
    - Role assignment based on camera angle and person proximity
    - Temporal tracking to prevent role flipping between frames
    - Pose-to-role matching using bounding box IoU

    Attributes:
        camera_angle: Camera angle (1=LP side, 2=ALP side)
        logger: Logger instance for debug output
        _prev_person_boxes: Previous frame's person bounding boxes
        _prev_person_roles: Previous frame's person role assignments
    """

    def __init__(
        self,
        camera_angle: int = 1,
        logger: Optional[logging.Logger] = None,
        iou_threshold: float = 0.3
    ):
        """Initialize the PersonTracker.

        Args:
            camera_angle: Camera position (1=LP side, 2=ALP side).
                         Determines which role is assigned to the closest person.
            logger: Optional logger instance. If None, creates a new one.
            iou_threshold: Minimum IoU for temporal role tracking (default 0.3)
        """
        self.camera_angle = camera_angle
        self.logger = logger or logging.getLogger(__name__)
        self.iou_threshold = iou_threshold

        # Temporal tracking state
        self._prev_person_boxes: List[List[float]] = []
        self._prev_person_roles: Dict[int, Dict[str, str]] = {}

    def reset_tracking(self) -> None:
        """Reset temporal tracking state.

        Call this when starting a new video or when tracking should be reset.
        """
        self._prev_person_boxes = []
        self._prev_person_roles = {}

    def identify_person_roles(
        self,
        person_boxes: List[List[int]],
        frame: Optional[Any] = None,
        detections: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Identify LP (Loco Pilot) and ALP (Assistant Loco Pilot) based on camera angle.

        Logic:
        - camera_angle=1 (LP Side): closest person (largest bbox) = LP, further = ALP
        - camera_angle=2 (ALP Side): closest person (largest bbox) = ALP, further = LP
        - Third+ persons = "Visitor"

        Args:
            person_boxes: List of de-duplicated person bounding boxes [[x1, y1, x2, y2], ...]
            frame: Current video frame (optional, for future use)
            detections: Dictionary of all detected objects (optional, for future use)

        Returns:
            Dictionary mapping person index to role info: {
                0: {'role': 'LP', 'role_name': 'Loco Pilot', 'bbox': [...], 'bbox_area': 12345.0},
                1: {'role': 'ALP', 'role_name': 'Assistant Loco Pilot', 'bbox': [...], 'bbox_area': 9876.0},
                ...
            }
        """
        if len(person_boxes) == 0:
            return {}

        def get_bbox_area(bbox):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width * height

        # Build person list with bbox areas
        persons = []
        for person_idx, person_bbox in enumerate(person_boxes):
            persons.append({
                'person_idx': person_idx,
                'bbox': person_bbox,
                'bbox_area': get_bbox_area(person_bbox)
            })

        # Sort by bounding box area (largest first = closest to camera)
        sorted_persons = sorted(persons, key=lambda x: x['bbox_area'], reverse=True)

        # Determine role assignment based on camera angle
        is_lp_side = self.camera_angle == 1
        closest_role = 'LP' if is_lp_side else 'ALP'
        closest_role_name = 'Loco Pilot' if is_lp_side else 'Assistant Loco Pilot'
        further_role = 'ALP' if is_lp_side else 'LP'
        further_role_name = 'Assistant Loco Pilot' if is_lp_side else 'Loco Pilot'

        camera_side = "LP Side" if is_lp_side else "ALP Side"

        # Assign roles based on camera proximity
        person_roles = {}

        if len(sorted_persons) == 1:
            # Only one person - assign based on camera side
            person_roles[0] = {
                'role': closest_role,
                'role_name': closest_role_name,
                'bbox': sorted_persons[0]['bbox'],
                'bbox_area': sorted_persons[0]['bbox_area']
            }

        elif len(sorted_persons) == 2:
            self.logger.debug(f"Role assignment (camera: {camera_side}): "
                            f"Person {sorted_persons[0]['person_idx']} (area={sorted_persons[0]['bbox_area']:.0f}) -> {closest_role}, "
                            f"Person {sorted_persons[1]['person_idx']} (area={sorted_persons[1]['bbox_area']:.0f}) -> {further_role}")

            person_roles[sorted_persons[0]['person_idx']] = {
                'role': closest_role,
                'role_name': closest_role_name,
                'bbox': sorted_persons[0]['bbox'],
                'bbox_area': sorted_persons[0]['bbox_area']
            }

            person_roles[sorted_persons[1]['person_idx']] = {
                'role': further_role,
                'role_name': further_role_name,
                'bbox': sorted_persons[1]['bbox'],
                'bbox_area': sorted_persons[1]['bbox_area']
            }

        else:
            # Three or more people
            self.logger.debug(f"Role assignment (3+ people, camera: {camera_side}) - areas: {[p['bbox_area'] for p in sorted_persons]}")

            # First person (largest bbox / closest to camera)
            person_roles[sorted_persons[0]['person_idx']] = {
                'role': closest_role,
                'role_name': closest_role_name,
                'bbox': sorted_persons[0]['bbox'],
                'bbox_area': sorted_persons[0]['bbox_area']
            }

            # Second person
            person_roles[sorted_persons[1]['person_idx']] = {
                'role': further_role,
                'role_name': further_role_name,
                'bbox': sorted_persons[1]['bbox'],
                'bbox_area': sorted_persons[1]['bbox_area']
            }

            # Additional people - assign as Visitor
            for i in range(2, len(sorted_persons)):
                person_idx = sorted_persons[i]['person_idx']
                person_roles[person_idx] = {
                    'role': 'VISITOR',
                    'role_name': 'Visitor',
                    'bbox': sorted_persons[i]['bbox'],
                    'bbox_area': sorted_persons[i]['bbox_area']
                }

        # Apply temporal role tracking to prevent role flipping
        person_roles = self._apply_temporal_tracking(person_roles)

        # Update tracking state for next frame
        self._update_tracking_state(person_roles)

        return person_roles

    def _apply_temporal_tracking(
        self,
        person_roles: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Apply temporal tracking to prevent role flipping between frames.

        Uses IoU matching between current and previous frame's bounding boxes
        to maintain role consistency.

        Args:
            person_roles: Current frame's role assignments

        Returns:
            Updated role assignments with temporal corrections applied
        """
        if not self._prev_person_boxes or not self._prev_person_roles or len(person_roles) < 2:
            return person_roles

        # Build IoU matrix: current person_idx -> best matching prev person_idx
        current_to_prev_match = {}
        for curr_idx, curr_info in person_roles.items():
            curr_box = curr_info['bbox']
            best_iou = 0.0
            best_prev_idx = None
            for prev_idx, prev_box in enumerate(self._prev_person_boxes):
                iou = calculate_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_idx = prev_idx
            if best_iou >= self.iou_threshold and best_prev_idx is not None:
                current_to_prev_match[curr_idx] = (best_prev_idx, best_iou)

        # Check if any LP/ALP roles need to be swapped based on temporal tracking
        lp_alp_indices = [idx for idx, info in person_roles.items() if info['role'] in ('LP', 'ALP')]
        matched_lp_alp = [idx for idx in lp_alp_indices if idx in current_to_prev_match]

        if len(matched_lp_alp) == 2 and len(lp_alp_indices) == 2:
            # Both LP/ALP persons have good IoU matches to previous frame
            idx_a, idx_b = matched_lp_alp
            prev_idx_a = current_to_prev_match[idx_a][0]
            prev_idx_b = current_to_prev_match[idx_b][0]

            # Get previous roles for the matched previous persons
            prev_role_a = self._prev_person_roles.get(prev_idx_a, {}).get('role')
            prev_role_b = self._prev_person_roles.get(prev_idx_b, {}).get('role')

            # Check if current assignment differs from previous and needs correction
            curr_role_a = person_roles[idx_a]['role']
            curr_role_b = person_roles[idx_b]['role']

            if (prev_role_a in ('LP', 'ALP') and prev_role_b in ('LP', 'ALP') and
                    prev_role_a != prev_role_b):
                # Previous frame had valid distinct LP/ALP assignments
                # Check if current frame flipped them
                if curr_role_a != prev_role_a or curr_role_b != prev_role_b:
                    # Roles flipped - restore previous assignment
                    role_name_map = {'LP': 'Loco Pilot', 'ALP': 'Assistant Loco Pilot'}
                    self.logger.debug(
                        f"Temporal role correction - restoring "
                        f"Person {idx_a} as {prev_role_a} and Person {idx_b} as {prev_role_b} "
                        f"(IoU: {current_to_prev_match[idx_a][1]:.2f}, {current_to_prev_match[idx_b][1]:.2f})"
                    )
                    person_roles[idx_a]['role'] = prev_role_a
                    person_roles[idx_a]['role_name'] = role_name_map[prev_role_a]
                    person_roles[idx_b]['role'] = prev_role_b
                    person_roles[idx_b]['role_name'] = role_name_map[prev_role_b]

        return person_roles

    def _update_tracking_state(self, person_roles: Dict[int, Dict[str, Any]]) -> None:
        """Update tracking state for the next frame.

        Args:
            person_roles: Current frame's role assignments
        """
        self._prev_person_boxes = [info['bbox'] for idx, info in sorted(person_roles.items())]
        self._prev_person_roles = {idx: {'role': info['role'], 'role_name': info['role_name']}
                                   for idx, info in person_roles.items()}

    def match_pose_to_roles(
        self,
        yolo_pose_results: Dict[int, Dict[str, Any]],
        person_roles: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Any]:
        """Match YOLOv8-Pose detections to identified person roles by bounding box IoU.

        Enhanced with torso-center fallback for cases where bboxes overlap significantly.

        Args:
            yolo_pose_results: Dict from YoloPoseAdapter.process() containing:
                {person_idx: {'bbox': [...], 'keypoints': YoloPoseLandmarks}}
            person_roles: Dict from identify_person_roles() containing:
                {person_idx: {'bbox': [...], 'role': 'LP'/'ALP', ...}}

        Returns:
            Dict mapping person_idx (from person_roles) to YoloPoseLandmarks
        """
        matched = {}
        used_yolo_indices = set()

        self.logger.info(f"[POSE MATCHING] Matching {len(yolo_pose_results)} YOLO poses to {len(person_roles)} person roles")

        for person_idx, role_data in person_roles.items():
            if 'bbox' not in role_data:
                continue

            role_bbox = role_data['bbox']
            role_center_x = (role_bbox[0] + role_bbox[2]) / 2
            role_center_y = (role_bbox[1] + role_bbox[3]) / 2
            role_name = role_data.get('role', 'UNKNOWN')

            # Collect all candidates with their IoU scores
            candidates = []
            for yolo_idx, yolo_data in yolo_pose_results.items():
                if yolo_idx in used_yolo_indices:
                    continue

                iou = calculate_iou(role_bbox, yolo_data['bbox'])
                candidates.append({
                    'yolo_idx': yolo_idx,
                    'iou': iou,
                    'keypoints': yolo_data['keypoints'],
                    'bbox': yolo_data['bbox']
                })

            # Sort by IoU descending
            candidates.sort(key=lambda x: x['iou'], reverse=True)

            # Log all candidates
            for c in candidates:
                self.logger.debug(f"  [{role_name}] Candidate YOLO {c['yolo_idx']}: IoU={c['iou']:.3f}")

            if not candidates:
                self.logger.warning(f"[POSE MATCHING] No candidates for {role_name} (person {person_idx})")
                continue

            best_candidate = candidates[0]

            # Check if top two candidates have similar IoU (within 0.15) - use torso center as tiebreaker
            if len(candidates) >= 2 and candidates[0]['iou'] - candidates[1]['iou'] < 0.15:
                self.logger.info(f"[POSE MATCHING] Close IoU scores for {role_name}: {candidates[0]['iou']:.3f} vs {candidates[1]['iou']:.3f} - using torso center")

                # Calculate torso center for each candidate using shoulders
                best_dist = float('inf')
                for c in candidates[:2]:  # Only compare top 2
                    keypoints = c['keypoints']
                    if len(keypoints.landmark) >= 7:  # Need at least shoulders
                        # Get shoulder positions (indices 5 and 6 for left/right shoulder)
                        left_shoulder = keypoints.landmark[5]
                        right_shoulder = keypoints.landmark[6]

                        # Get frame dimensions from bbox (approximate)
                        bbox = c['bbox']
                        frame_w = max(bbox[2], 1920)  # Estimate frame width
                        frame_h = max(bbox[3], 1080)  # Estimate frame height

                        # Calculate torso center in pixel coords
                        torso_x = ((left_shoulder.x + right_shoulder.x) / 2) * frame_w
                        torso_y = ((left_shoulder.y + right_shoulder.y) / 2) * frame_h

                        # Distance from role bbox center to torso center
                        dist = ((torso_x - role_center_x) ** 2 + (torso_y - role_center_y) ** 2) ** 0.5

                        self.logger.debug(f"    Candidate {c['yolo_idx']}: torso=({torso_x:.0f}, {torso_y:.0f}), dist={dist:.0f}px")

                        if dist < best_dist:
                            best_dist = dist
                            best_candidate = c

                self.logger.info(f"[POSE MATCHING] Torso-based selection: YOLO {best_candidate['yolo_idx']} (dist={best_dist:.0f}px)")

            # Match if IoU is above threshold (0.2 for overlapping cases)
            if best_candidate['iou'] > 0.2:
                matched[person_idx] = best_candidate['keypoints']
                used_yolo_indices.add(best_candidate['yolo_idx'])
                self.logger.info(f"[POSE MATCHING] Matched {role_name} (person {person_idx}) -> YOLO {best_candidate['yolo_idx']} (IoU={best_candidate['iou']:.3f})")
            else:
                self.logger.warning(f"[POSE MATCHING] No match for {role_name} (person {person_idx}): best IoU={best_candidate['iou']:.3f} < 0.2")

        return matched

    def get_role_for_person(
        self,
        person_idx: int,
        person_roles: Dict[int, Dict[str, Any]]
    ) -> Optional[str]:
        """Get the role for a specific person index.

        Args:
            person_idx: Person index to look up
            person_roles: Dictionary of person roles

        Returns:
            Role string ('LP', 'ALP', 'VISITOR') or None if not found
        """
        if person_idx in person_roles:
            return person_roles[person_idx].get('role')
        return None

    def get_role_name_for_person(
        self,
        person_idx: int,
        person_roles: Dict[int, Dict[str, Any]]
    ) -> Optional[str]:
        """Get the role name for a specific person index.

        Args:
            person_idx: Person index to look up
            person_roles: Dictionary of person roles

        Returns:
            Role name string or None if not found
        """
        if person_idx in person_roles:
            return person_roles[person_idx].get('role_name')
        return None
