"""MediaPipe pose / face-mesh overlay helpers extracted from the monolith.

These functions are pure / stateless: they take a frame plus the MediaPipe
module references the monolith holds on ``self`` (``mp_drawing``,
``mp_pose``, ``mp_face_mesh``, ``mp_drawing_styles``) by parameter, and
return the annotated frame. Behavior is byte-identical to the original
``LocopilotActivityMonitor.draw_mediapipe_outputs`` and
``draw_multi_person_mediapipe_outputs`` methods — log strings and drawing
parameters are preserved verbatim.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import cv2
import mediapipe as mp


_DEFAULT_LOGGER = logging.getLogger(__name__)


def draw_mediapipe_outputs(
    frame: Any,
    pose_results: Any,
    face_results: Any,
    *,
    mp_drawing: Any,
    mp_pose: Any,
    mp_face_mesh: Any,
    mp_drawing_styles: Any,
    pose_sleep_info: Optional[Dict[str, Any]] = None,
    head_pose_info: Optional[Dict[str, Any]] = None,
    sleep_strong_duration_sec: float = 2.0,
    sleep_microsleep_duration_sec: float = 2.0,
) -> Any:
    """Draw MediaPipe pose and face mesh landmarks on frame"""
    annotated_frame = frame.copy()

    face_detected = face_results.multi_face_landmarks is not None and len(face_results.multi_face_landmarks) > 0

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    if face_detected:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    if face_detected:
        cv2.putText(annotated_frame, "FACE DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated_frame, "FACE NOT DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    # Display pose-based sleep detection info
    if pose_sleep_info and pose_results.pose_landmarks:
        y_offset = 60 if not face_detected else 120

        # Head tilt angle
        if 'head_tilt' in pose_sleep_info and pose_sleep_info['head_tilt'] is not None:
            head_tilt = pose_sleep_info['head_tilt']
            tilt_color = (0, 0, 255) if head_tilt < -15 else (0, 255, 0)
            tilt_text = f"Head Tilt: {head_tilt:.1f}deg"
            cv2.putText(annotated_frame, tilt_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 2, cv2.LINE_AA)
            y_offset += 30

        # Movement score
        if 'avg_movement' in pose_sleep_info:
            movement = pose_sleep_info['avg_movement']
            movement_color = (0, 0, 255) if movement < 0.02 else (0, 255, 0)  # Updated threshold
            movement_text = f"Movement: {movement:.4f}"
            cv2.putText(annotated_frame, movement_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement_color, 2, cv2.LINE_AA)
            y_offset += 30

        # Pose sleep duration
        if 'pose_sleep_duration' in pose_sleep_info and pose_sleep_info['pose_sleep_duration'] > 0:
            duration = pose_sleep_info['pose_sleep_duration']
            duration_text = f"Pose Sleep: {duration:.1f}s"

            if duration >= sleep_strong_duration_sec:
                duration_text += " - SLEEP DETECTED!"
                duration_color = (0, 0, 255)
            elif duration >= sleep_microsleep_duration_sec:
                duration_text += " - MICROSLEEP!"
                duration_color = (0, 140, 255)
            else:
                duration_color = (0, 165, 255)

            cv2.putText(annotated_frame, duration_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2, cv2.LINE_AA)

    # Display head pose angles for mind diversion detection
    if head_pose_info and head_pose_info.get('method') != 'none':
        y_offset = 60 if not face_detected else (120 if not pose_sleep_info else 180)

        yaw = head_pose_info.get('yaw', 0)
        pitch = head_pose_info.get('pitch', 0)
        detected = head_pose_info.get('detected', False)
        method = head_pose_info.get('method', 'unknown')

        # Display yaw (side turn)
        yaw_direction = "RIGHT" if yaw > 0 else "LEFT"
        yaw_color = (0, 0, 255) if abs(yaw) > 45 else (0, 255, 0)
        yaw_text = f"Head Yaw: {abs(yaw):.1f}° {yaw_direction}"
        cv2.putText(annotated_frame, yaw_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, yaw_color, 2, cv2.LINE_AA)

        # Display pitch (up/down tilt)
        pitch_direction = "DOWN" if pitch > 0 else "UP"
        pitch_color = (0, 0, 255) if pitch > 15 else (0, 255, 0)
        pitch_text = f"Head Pitch: {abs(pitch):.1f}° {pitch_direction}"
        cv2.putText(annotated_frame, pitch_text, (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_color, 2, cv2.LINE_AA)

        # Display mind diversion alert if detected
        if detected:
            alert_text = "[WARN] MIND DIVERSION - ATTENTION DIVERTED!"
            cv2.putText(annotated_frame, alert_text, (10, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Show detection method
            method_text = f"(Method: {method})"
            cv2.putText(annotated_frame, method_text, (10, y_offset + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated_frame


def draw_multi_person_mediapipe_outputs(
    frame: Any,
    persons_data: Dict[int, Dict[str, Any]],
    face_results: Any,
    *,
    mp_drawing: Any,
    mp_face_mesh: Any,
    mp_drawing_styles: Any,
    get_keypoint: Callable[[Any, str], Any],
) -> Any:
    """Draw MediaPipe pose landmarks for ALL detected persons

    Args:
        frame: The frame image
        persons_data: Dictionary of person data from process_all_persons_activities()
                     Format: {person_idx: {'pose_landmarks': landmarks, 'role': 'LP', 'activities': {...}, ...}}
        face_results: MediaPipe face mesh results

    Returns:
        Annotated frame with all persons' pose landmarks drawn
    """
    annotated_frame = frame.copy()

    # Custom drawing specs for more visible landmarks
    landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 255, 0),      # Green color for landmarks
        thickness=3,             # Thicker circles (increased from default 2)
        circle_radius=5          # Larger circles (increased from default 2)
    )

    connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 255, 255),     # Cyan color for connections
        thickness=3              # Thicker connections (increased from default 2)
    )

    # Key landmarks to label (using YOLO keypoint names)
    key_landmarks_to_label = [
        ('nose', "Nose"),
        ('left_shoulder', "L Shoulder"),
        ('right_shoulder', "R Shoulder"),
        ('left_elbow', "L Elbow"),
        ('right_elbow', "R Elbow"),
        ('left_wrist', "L Wrist"),
        ('right_wrist', "R Wrist"),
        ('left_hip', "L Hip"),
        ('right_hip', "R Hip"),
        ('left_knee', "L Knee"),
        ('right_knee', "R Knee"),
        ('left_ankle', "L Ankle"),
        ('right_ankle', "R Ankle"),
    ]

    # YOLO skeleton connections (COCO format)
    skeleton_connections = [
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

    # Draw pose landmarks for ALL persons
    for person_idx, person_data in persons_data.items():
        pose_landmarks = person_data.get('pose_landmarks')
        if pose_landmarks:
            h, w = annotated_frame.shape[:2]

            # Draw skeleton connections
            for start_name, end_name in skeleton_connections:
                try:
                    start = get_keypoint(pose_landmarks, start_name)
                    end = get_keypoint(pose_landmarks, end_name)

                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_pt = (int(start.x * w), int(start.y * h))
                        end_pt = (int(end.x * w), int(end.y * h))
                        cv2.line(annotated_frame, start_pt, end_pt, (0, 255, 255), 3)
                except (cv2.error, ValueError, TypeError):
                    continue

            # Draw keypoints
            for i in range(17):
                try:
                    landmark = pose_landmarks.landmark[i]
                    if landmark.visibility > 0.5:
                        pt = (int(landmark.x * w), int(landmark.y * h))
                        cv2.circle(annotated_frame, pt, 8, (0, 255, 0), -1)
                except (cv2.error, ValueError, TypeError):
                    continue

            # Draw labels for key landmarks
            for keypoint_name, label_name in key_landmarks_to_label:
                try:
                    landmark = get_keypoint(pose_landmarks, keypoint_name)

                    # Only draw if landmark is visible enough
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        # Draw small label with background
                        label_size, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        label_w, label_h = label_size

                        # Background rectangle for better visibility
                        cv2.rectangle(annotated_frame,
                                    (x - 2, y - label_h - 4),
                                    (x + label_w + 2, y + 2),
                                    (0, 0, 0), -1)

                        # Label text in white
                        cv2.putText(annotated_frame, label_name,
                                   (x, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                except (cv2.error, ValueError, TypeError):
                    pass

            # Add person label near their head
            try:
                nose = get_keypoint(pose_landmarks, 'nose')
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                role_name = person_data.get('role_name', 'Unknown')
                label = f"{role_name} ({person_idx+1})"

                # Draw label with background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_w, label_h = label_size

                # Background rectangle
                cv2.rectangle(annotated_frame,
                            (nose_x - 5, nose_y - label_h - 15),
                            (nose_x + label_w + 5, nose_y - 5),
                            (0, 0, 0), -1)

                # Label text
                cv2.putText(annotated_frame, label,
                           (nose_x, nose_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            except (cv2.error, ValueError, TypeError):
                pass

    # Draw face mesh (same as before)
    face_detected = face_results.multi_face_landmarks is not None and len(face_results.multi_face_landmarks) > 0

    if face_detected:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # Draw face detection status
    if face_detected:
        cv2.putText(annotated_frame, "FACE DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated_frame, "FACE NOT DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    # Draw activity warnings for each person
    y_offset = 100
    for person_idx, person_data in persons_data.items():
        activities = person_data.get('activities', {})
        role_name = person_data.get('role_name', 'Unknown')
        debug_info = person_data.get('debug_info', {})

        # Mind diversion
        if activities.get('mind_diversion'):
            head_pose = debug_info.get('head_pose', {})
            yaw = head_pose.get('yaw', 0)
            pitch = head_pose.get('pitch', 0)

            alert_text = f"!!! {role_name}: MIND DIVERSION - ATTENTION DIVERTED!"
            cv2.putText(annotated_frame, alert_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            details_text = f"    Yaw={abs(yaw):.1f}°, Pitch={abs(pitch):.1f}°"
            cv2.putText(annotated_frame, details_text, (10, y_offset + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 55

        # Sleep/microsleep
        if activities.get('sleep'):
            alert_text = f"!!! {role_name}: SLEEP DETECTED"
            cv2.putText(annotated_frame, alert_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y_offset += 30
        elif activities.get('microsleep'):
            alert_text = f"!!! {role_name}: MICROSLEEP DETECTED"
            cv2.putText(annotated_frame, alert_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
            y_offset += 30

        # Cell phone
        if activities.get('cell_phone'):
            alert_text = f"! {role_name}: CELL PHONE IN USE"
            cv2.putText(annotated_frame, alert_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            y_offset += 25

        # Writing
        if activities.get('writing'):
            alert_text = f"! {role_name}: WRITING"
            cv2.putText(annotated_frame, alert_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

    # Hand gesture alerts - only show if NOT coordinated (one raised but not the other)
    # Check if BOTH LP and ALP raised hands across all persons
    any_lp_gesture = any(p.get('activities', {}).get('lp_hand_gesture', False) for p in persons_data.values())
    any_alp_gesture = any(p.get('activities', {}).get('alp_hand_gesture', False) for p in persons_data.values())
    both_raised = any_lp_gesture and any_alp_gesture

    # Only show hand gesture alerts if it's a coordination failure (one raised, other didn't)
    if not both_raised:
        for person_idx, person_data in persons_data.items():
            activities = person_data.get('activities', {})
            role_name = person_data.get('role_name', 'Unknown')

            if activities.get('lp_hand_gesture'):
                alert_text = f"! {role_name}: LP HAND GESTURE (ALP not responding)"
                cv2.putText(annotated_frame, alert_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                y_offset += 25

            if activities.get('alp_hand_gesture'):
                alert_text = f"! {role_name}: ALP HAND GESTURE (LP not responding)"
                cv2.putText(annotated_frame, alert_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)
                y_offset += 25

    return annotated_frame
