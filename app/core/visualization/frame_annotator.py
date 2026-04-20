"""Frame annotation for video visualization.

This module provides a FrameAnnotator class for drawing detection results,
pose landmarks, and debug overlays on video frames.

Drawing Functions:
- Bounding boxes for detected objects
- ROI regions for pose-guided detection
- Person role labels (LP/ALP/Visitor)
- Pose skeleton connections
- Sleep detection debug overlay
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import cv2
import numpy as np


class FrameAnnotator:
    """Annotates video frames with detections, poses, and debug info.

    This class handles all frame visualization including:
    - Object detection bounding boxes
    - ROI visualization for pose-guided detection
    - Person role labels and count overlays
    - Pose skeleton drawing
    - Sleep detection debug overlays

    Attributes:
        colors: Color mapping for different object types and roles
        logger: Logger instance for debug output
    """

    # Default color palette for annotations (BGR format)
    DEFAULT_COLORS = {
        'person': (0, 255, 0),
        'cell_phone': (0, 0, 255),
        'book': (255, 0, 0),
        'backpack': (0, 255, 255),
        'deduplicated_person': (0, 255, 0),
        'LP': (0, 255, 255),      # Yellow for Loco Pilot
        'ALP': (255, 165, 0),     # Orange for Assistant Loco Pilot
        'SUPERVISOR': (128, 0, 128),
        'TRAINEE': (0, 255, 255),
        'VISITOR': (128, 128, 128)
    }

    # YOLO skeleton connections (COCO format)
    SKELETON_CONNECTIONS = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
    ]

    # Key landmarks to label
    KEY_LANDMARKS = [
        ('nose', "Nose"),
        ('left_shoulder', "L Shoulder"),
        ('right_shoulder', "R Shoulder"),
        ('left_elbow', "L Elbow"),
        ('right_elbow', "R Elbow"),
        ('left_wrist', "L Wrist"),
        ('right_wrist', "R Wrist"),
        ('left_hip', "L Hip"),
        ('right_hip', "R Hip"),
    ]

    def __init__(
        self,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the FrameAnnotator.

        Args:
            colors: Custom color mapping for object types. If None, uses defaults.
            logger: Optional logger instance. If None, creates a new one.
        """
        self.colors = colors if colors is not None else self.DEFAULT_COLORS.copy()
        self.logger = logger or logging.getLogger(__name__)

    def draw_bounding_boxes(
        self,
        frame: np.ndarray,
        detections: Dict[str, List[Any]],
        show_roi_boxes: bool = True,
        person_roles: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> np.ndarray:
        """Draw bounding boxes on frame for detected objects and ROI regions.

        Args:
            frame: Input frame (BGR numpy array)
            detections: Dictionary with detection results
            show_roi_boxes: Whether to show ROI boxes (default True)
            person_roles: Dictionary of person roles (optional)

        Returns:
            Annotated frame with bounding boxes drawn
        """
        annotated_frame = frame.copy()

        # Draw ROI boxes (semi-transparent cyan boxes)
        if show_roi_boxes and 'roi_boxes' in detections:
            for keypoint_name, roi_bbox in detections['roi_boxes']:
                x1, y1, x2, y2 = roi_bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                label = keypoint_name.replace('_', ' ')
                cv2.putText(annotated_frame, label,
                           (x1 + 5, y1 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 255, 0), 1)

        # Draw ROI detections (objects found via pose-guided detection)
        if 'roi_detections' in detections:
            for roi_det in detections['roi_detections']:
                bbox = roi_det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                color = (255, 0, 255)  # Magenta for pose-guided detections
                thickness = 3

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                label = f"{roi_det['class']} {roi_det['confidence']:.2f} (ROI: {roi_det['keypoint']})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_w, label_h = label_size

                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_h - 10),
                            (x1 + label_w + 10, y1),
                            color, -1)

                cv2.putText(annotated_frame, label,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 255, 255), 1)

        # Draw regular detections
        for obj_type, bboxes in detections.items():
            # Skip non-bbox entries: ROI metadata, dedup cache, and diagnostic
            # fields. ``_raw_v8_detections`` holds (class_name, conf) tuples
            # used by the [V8 DETECTIONS] INFO log — iterating it as bboxes
            # crashes with int('person').
            if obj_type in ('roi_detections', 'roi_boxes', 'deduplicated_person'):
                continue
            if obj_type.startswith('_'):
                continue

            color = self.colors.get(obj_type, (255, 255, 255))
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label = obj_type.replace('_', ' ').title()
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_w, label_h = label_size

                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_h - 10),
                            (x1 + label_w + 10, y1),
                            color, -1)

                cv2.putText(annotated_frame, label,
                           (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 2)

        # Draw deduplicated person boxes with role labels
        if 'deduplicated_person' in detections and len(detections['deduplicated_person']) > 0:
            self._draw_person_boxes(annotated_frame, detections['deduplicated_person'],
                                   person_roles, frame.shape)

        return annotated_frame

    def _draw_person_boxes(
        self,
        frame: np.ndarray,
        person_boxes: List[Any],
        person_roles: Optional[Dict[int, Dict[str, Any]]],
        frame_shape: Tuple[int, ...]
    ) -> None:
        """Draw person bounding boxes with role labels.

        Args:
            frame: Frame to draw on (modified in place)
            person_boxes: List of person bounding boxes
            person_roles: Dictionary of person roles (optional)
            frame_shape: Shape of the frame
        """
        person_count = len(person_boxes)

        for idx, bbox in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, bbox)

            if person_roles and idx in person_roles:
                role_info = person_roles[idx]
                role = role_info['role']
                role_name = role_info['role_name']
                bbox_area = role_info.get('bbox_area', 0)
                box_color = self.colors.get(role, (0, 255, 0))
                label = f"{role_name} (area:{bbox_area:.0f})"
            else:
                box_color = (0, 255, 0)
                label = f"Person {idx+1}"

            # Thicker border for persons
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = label_size

            cv2.rectangle(frame,
                        (x1, y1 - label_h - 10),
                        (x1 + label_w + 10, y1),
                        box_color, -1)

            cv2.putText(frame, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)

        # Add person count overlay
        if person_count > 2:
            count_text = f"GROUP DETECTED: {person_count} PEOPLE"
            count_color = (0, 0, 255)  # Red for group alert
        else:
            count_text = f"People Count: {person_count}"
            count_color = (0, 255, 0)

        cv2.putText(frame, count_text,
                   (frame_shape[1] - 400, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, count_color, 2, cv2.LINE_AA)

        # Add role summary
        if person_roles:
            y_offset = 60
            for idx in sorted(person_roles.keys()):
                role_info = person_roles[idx]
                role_text = f"{role_info['role_name']} (area:{role_info.get('bbox_area', 0):.0f})"
                role_color = self.colors.get(role_info['role'], (255, 255, 255))
                cv2.putText(frame, role_text,
                           (frame_shape[1] - 400, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, role_color, 2, cv2.LINE_AA)
                y_offset += 25

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose_landmarks: Any,
        get_keypoint_func: Callable,
        visibility_threshold: float = 0.5,
        connection_color: Tuple[int, int, int] = (0, 255, 255),
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 3,
        keypoint_radius: int = 8
    ) -> np.ndarray:
        """Draw pose skeleton on frame.

        Args:
            frame: Input frame (BGR numpy array)
            pose_landmarks: Pose landmarks object
            get_keypoint_func: Function to get keypoint by name
            visibility_threshold: Minimum visibility to draw landmark
            connection_color: Color for skeleton connections
            keypoint_color: Color for keypoints
            line_thickness: Thickness of connection lines
            keypoint_radius: Radius of keypoint circles

        Returns:
            Frame with skeleton drawn
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        # Draw skeleton connections
        for start_name, end_name in self.SKELETON_CONNECTIONS:
            try:
                start = get_keypoint_func(pose_landmarks, start_name)
                end = get_keypoint_func(pose_landmarks, end_name)

                if start.visibility > visibility_threshold and end.visibility > visibility_threshold:
                    start_pt = (int(start.x * w), int(start.y * h))
                    end_pt = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_frame, start_pt, end_pt, connection_color, line_thickness)
            except Exception:
                continue

        # Draw keypoints
        for i in range(17):
            try:
                landmark = pose_landmarks.landmark[i]
                if landmark.visibility > visibility_threshold:
                    pt = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(annotated_frame, pt, keypoint_radius, keypoint_color, -1)
            except Exception:
                continue

        return annotated_frame

    def draw_sleep_debug_overlay(
        self,
        frame: np.ndarray,
        sleep_info: Dict[str, Any],
        person_idx: int,
        activities: Dict[str, bool],
        timestamp_sec: float,
        sleep_score_threshold: int = 5
    ) -> np.ndarray:
        """Draw sleep detection debug overlay on frame.

        Shows sleep score, state machine state, eye visibility, wrist velocity,
        and other key signals as a semi-transparent panel.

        Args:
            frame: Frame to annotate (modified in place)
            sleep_info: Sleep detection information dictionary
            person_idx: Person index (0 for LP side, 1 for ALP side)
            activities: Dictionary of activity states
            timestamp_sec: Current timestamp
            sleep_score_threshold: Threshold for sleep score display

        Returns:
            Frame with debug overlay
        """
        if frame is None or sleep_info is None:
            return frame

        h, w = frame.shape[:2]
        x_offset = 5 if person_idx == 0 else w - 340
        y_start = 55

        # Extract sleep info
        sleep_score = sleep_info.get('sleep_score', 0)
        score_thresh = sleep_info.get('sleep_score_threshold', sleep_score_threshold)
        state = sleep_info.get('sleep_state', 'N/A')
        eye_vis = sleep_info.get('avg_eye_vis')
        wrist_vel = sleep_info.get('avg_wrist_velocity', 0)
        head_bob = sleep_info.get('head_bob_detected', False)
        slump_rate = sleep_info.get('shoulder_slump_rate', 0)
        is_slumping = sleep_info.get('is_shoulder_slumping', False)
        is_wrists_still = sleep_info.get('is_wrists_still', False)
        is_wrists_active = sleep_info.get('is_wrists_active', False)
        face_gone = sleep_info.get('is_face_not_visible', False)
        face_gone_body = sleep_info.get('is_face_gone_with_body_signals', False)
        is_sustained_low_eyes = sleep_info.get('is_sustained_low_eyes', False)
        is_sustained_stillness = sleep_info.get('is_sustained_stillness', False)
        is_hands_clasped = sleep_info.get('is_hands_clasped', False)
        is_sleeping = activities.get('sleep', False)
        is_microsleep = activities.get('microsleep', False)
        status = sleep_info.get('status', '')
        movement = sleep_info.get('avg_movement', 0)
        duration = sleep_info.get('pose_sleep_duration', 0)

        # Build lines
        lines = []
        lines.append(f"P{person_idx} SLEEP DEBUG")
        lines.append(f"Score: {sleep_score}/{score_thresh}")
        lines.append(f"State: {state}")
        lines.append(f"Eye vis: {eye_vis:.3f}" if eye_vis is not None else "Eye vis: N/A")
        lines.append(f"Wrist vel: {wrist_vel:.4f}")
        lines.append(f"Movement: {movement:.4f}" if movement else "Movement: N/A")
        lines.append(f"Head bob: {head_bob}")
        lines.append(f"Slump: {slump_rate:.5f} {'[!]' if is_slumping else ''}")
        lines.append(f"Wrists: {'STILL' if is_wrists_still else 'ACTIVE' if is_wrists_active else 'normal'}")
        lines.append(f"Face gone: {face_gone} body:{face_gone_body}")
        lines.append(f"Low eyes: {is_sustained_low_eyes} Still: {is_sustained_stillness}")
        lines.append(f"Clasped: {is_hands_clasped}")
        if duration:
            lines.append(f"Duration: {duration:.1f}s")
        if status:
            lines.append(f"Status: {status}")

        # Determine panel color based on state
        if is_sleeping:
            panel_color = (0, 0, 180)
            text_color = (255, 255, 255)
        elif is_microsleep:
            panel_color = (0, 80, 200)
            text_color = (255, 255, 255)
        elif state == 'DROWSY':
            panel_color = (0, 140, 220)
            text_color = (0, 0, 0)
        elif state == 'LOOKING_DOWN_WORKING':
            panel_color = (140, 100, 40)
            text_color = (255, 255, 255)
        elif sleep_score >= score_thresh:
            panel_color = (30, 80, 160)
            text_color = (255, 255, 255)
        else:
            panel_color = (60, 60, 60)
            text_color = (200, 200, 200)

        # Draw semi-transparent panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        line_height = 16
        panel_h = len(lines) * line_height + 10
        panel_w = 335

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_start), (x_offset + panel_w, y_start + panel_h), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        for i, line in enumerate(lines):
            y = y_start + 14 + i * line_height
            color = text_color
            if 'Score:' in line and sleep_score >= score_thresh:
                color = (0, 255, 255)
            elif 'SLEEPING' in str(state) or 'MICROSLEEP' in str(state):
                color = (0, 0, 255)
            elif 'Head bob: True' in line:
                color = (0, 255, 0)
            elif 'Face gone: True' in line:
                color = (0, 200, 255)
            cv2.putText(frame, line, (x_offset + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Draw score bar
        bar_y = y_start + panel_h + 2
        bar_w = min(int((sleep_score / max(score_thresh + 3, 1)) * panel_w), panel_w)
        bar_color = (0, 255, 0) if sleep_score < score_thresh else (0, 0, 255)
        cv2.rectangle(frame, (x_offset, bar_y), (x_offset + bar_w, bar_y + 6), bar_color, -1)
        cv2.rectangle(frame, (x_offset, bar_y), (x_offset + panel_w, bar_y + 6), (100, 100, 100), 1)
        thresh_x = x_offset + int((score_thresh / max(score_thresh + 3, 1)) * panel_w)
        cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + 6), (255, 255, 255), 2)

        return frame

    def draw_text_overlay(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.7,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        background: bool = True,
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """Draw text with optional background on frame.

        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position
            font_scale: Font scale
            color: Text color
            thickness: Text thickness
            background: Whether to draw background
            bg_color: Background color

        Returns:
            Frame with text drawn
        """
        x, y = position

        if background:
            label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_w, label_h = label_size
            cv2.rectangle(frame, (x - 2, y - label_h - 4), (x + label_w + 2, y + 4), bg_color, -1)

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return frame
