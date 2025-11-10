import cv2
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
import os

class LocopilotActivityMonitor:
    def __init__(self, video_path, output_dir="evidence", save_annotated_frames=False, frame_save_interval=1, sample_fps=1.0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.evidence_clips_dir = os.path.join(output_dir, "clips")
        self.json_dir = os.path.join(output_dir, "json")
        
        # Frame sampling configuration
        self.sample_fps = sample_fps  # Sample frames at this rate (e.g., 0.5 = 1 frame every 2 seconds)
        
        # Control annotated frame saving
        self.save_annotated_frames = save_annotated_frames
        self.frame_save_interval = frame_save_interval  # Save 1 frame every N sampled frames (1 = save all sampled frames)
        
        if self.save_annotated_frames:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.frames_dir = os.path.join(output_dir, "frames", f"run_{self.run_timestamp}")
            os.makedirs(self.frames_dir, exist_ok=True)
        
        # Create directories
        os.makedirs(self.evidence_clips_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Initialize models
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolo11s.pt')
        print("Initializing MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # NOTE: MediaPipe Pose only supports single-person tracking
        # For multi-person scenarios, hands from the "primary" detected person are tracked
        # Face mesh supports multiple people (up to 2) for microsleep detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,  # Track up to 2 faces (both loco pilots)
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Activity tracking with temporal filtering
        self.activities = {
            'microsleep': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'sleep': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'cell_phone': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'writing': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'packing_bags': {'active': False, 'start_time': None, 'frames': [], 'duration': 0}
        }
        
        # Activity thresholds: minimum duration and required consecutive frames before recording starts
        self.activity_thresholds = {
            'packing_bags': {
                'min_duration': 3.0,          # Must last 3 seconds minimum (reduced from 5.0)
                'required_consecutive': 60,   # 2 seconds @ 30fps (reduced from 150)
                'margin': 50,                 # More lenient proximity (increased from 30)
                'grace_frames': 15            # Allow 15 frames (0.5s) of non-detection
            },
            'writing': {
                'min_duration': 2.0,          # Must last 2 seconds minimum (reduced from 3.0)
                'required_consecutive': 45,   # 1.5 seconds @ 30fps (reduced from 90)
                'margin': 60,                 # More lenient proximity (increased from 50)
                'grace_frames': 15            # Allow 15 frames (0.5s) of non-detection
            },
            'cell_phone': {
                'min_duration': 0.0,          # NO minimum duration - any detection creates activity
                'required_consecutive': 1,    # Just 1 sample detection required
                'margin': 80,                 # Very lenient proximity for detecting phone near hand
                'grace_frames': 5             # Allow 5 samples (~10s) gap to group nearby detections into same activity
            },
            'microsleep': {
                'min_duration': 5.0,          # Must last 5 seconds minimum (reduced for early detection)
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds (reduced from 10)
                'margin': None,               # N/A for eye-based detection
                'grace_frames': 10            # Allow 10 frames (~20s) of non-detection
            },
            'sleep': {
                'min_duration': 30.0,         # Must last 30 seconds minimum (reduced from 180s)
                'required_consecutive': 5,    # 5 samples @ 0.5fps = 10 seconds (reduced from 30)
                'margin': None,               # N/A for eye-based detection
                'grace_frames': 10            # Allow 10 frames (~20s) of non-detection
            }
        }
        
        # Consecutive detection counters for temporal filtering
        self.consecutive_detections = {
            'microsleep': 0,
            'sleep': 0,
            'cell_phone': 0,
            'writing': 0,
            'packing_bags': 0
        }
        
        # Grace period counters - allows brief interruptions without resetting
        self.grace_counters = {
            'microsleep': 0,
            'sleep': 0,
            'cell_phone': 0,
            'writing': 0,
            'packing_bags': 0
        }
        
        # Buffer for pre-activity frames (5 seconds before at sampled rate)
        # Calculate buffer size based on sample_fps: 5 seconds * sample_fps
        buffer_size = max(5, int(5 * self.sample_fps))  # At least 5 frames
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Eye closure tracking
        self.eye_closure_start = None
        self.eye_closure_duration = 0
        
        # Pose-based sleep detection tracking
        self.pose_sleep_start = None
        self.pose_sleep_duration = 0
        self.previous_pose_landmarks = None
        self.movement_history = deque(maxlen=int(30 * self.sample_fps))  # 30 seconds of movement data
        self.head_tilt_history = deque(maxlen=int(10 * self.sample_fps))  # 10 seconds of head tilt data
        
        # Evidence counter
        self.evidence_counter = 0
        
        # Activity type mappings for JSON output
        self.activity_type_map = {
            'cell_phone': 2,
            'microsleep': 3,
            'sleep': 4,
            'writing': 5,
            'packing_bags': 6
        }
        
        # Activity descriptions
        self.activity_descriptions = {
            'cell_phone': 'Using mobile phone',
            'microsleep': 'Micro-sleep detected (5+ seconds)',
            'sleep': 'Sleep detected (30+ seconds)',
            'writing': 'Writing activity detected',
            'packing_bags': 'Packing bags activity detected'
        }
        
        # Evidence rules
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'eyes_closed_5s_or_pose_indicators',
            'sleep': 'eyes_closed_30s_or_pose_indicators',
            'writing': 'hand_near_book',
            'packing_bags': 'hand_near_backpack'
        }
        
        # Default crew/trip information
        self.trip_id = "TRIP-123"
        self.crew_name = "John Doe"
        self.crew_id = "C-001"
        self.crew_role = 1  # 1 for primary loco pilot
        
        # Store all activities for final JSON array output
        self.all_activities = []
    
    def sample_video_frames(self, video_path):
        """Sample frames at fixed intervals based on sample_fps.
        
        Yields tuples: (sample_index, timestamp_sec, frame_bgr, frame_idx)
        
        Args:
            video_path: Path to video file
            
        Yields:
            sample_index: Sequential index of sampled frames (0, 1, 2, ...)
            timestamp_sec: Timestamp in seconds from video start
            frame_bgr: BGR frame from OpenCV
            frame_idx: Original frame index in the video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Calculate stride: how many frames to skip between samples
        step = max(1, int(round(native_fps / max(1e-6, float(self.sample_fps)))))
        
        print(f"[Frame Sampling] Native FPS: {native_fps:.2f}, Sample FPS: {self.sample_fps}")
        print(f"[Frame Sampling] Step: {step} (sampling 1 frame every {step} frames)")
        print(f"[Frame Sampling] Expected sampled frames: ~{(total_frames // step)}")
        
        sampled_idx = 0
        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_idx / native_fps
            yield sampled_idx, timestamp, frame, frame_idx
            sampled_idx += 1
        
        cap.release()
        print(f"[Frame Sampling] Completed sampling, total samples: {sampled_idx}")
        
    def calculate_eye_aspect_ratio(self, landmarks):
        """Calculate Eye Aspect Ratio (EAR) for drowsiness detection"""
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            def get_ear(eye_indices):
                points = [landmarks[i] for i in eye_indices]
                v1 = np.linalg.norm(np.array([points[1].x, points[1].y]) - 
                                   np.array([points[5].x, points[5].y]))
                v2 = np.linalg.norm(np.array([points[2].x, points[2].y]) - 
                                   np.array([points[4].x, points[4].y]))
                h = np.linalg.norm(np.array([points[0].x, points[0].y]) - 
                                  np.array([points[3].x, points[3].y]))
                
                if h == 0:
                    return 0.3
                
                ear = (v1 + v2) / (2.0 * h)
                return ear
            
            left_ear = get_ear(left_eye_indices)
            right_ear = get_ear(right_eye_indices)
            avg_ear = (left_ear + right_ear) / 2.0
            
            return max(0.0, min(0.5, avg_ear))
            
        except Exception as e:
            return None
    
    def calculate_head_tilt_angle(self, landmarks):
        """Calculate head tilt angle from pose landmarks.
        
        Returns:
            float: Head tilt angle in degrees (0 = upright, negative = tilted forward/down)
        """
        try:
            # Use nose, neck (midpoint of shoulders), and reference points
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate neck position (midpoint between shoulders)
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Calculate angle from vertical
            # Positive y goes down in image coordinates
            delta_y = nose.y - neck_y
            delta_x = nose.x - neck_x
            
            # Calculate angle in degrees
            # Negative angle = head tilted forward/down (sleeping position)
            angle = np.arctan2(delta_y, delta_x) * 180 / np.pi - 90
            
            return angle
            
        except Exception as e:
            return None
    
    def calculate_movement_score(self, current_landmarks, previous_landmarks):
        """Calculate movement score between two sets of pose landmarks.
        
        Returns:
            float: Movement score (0 = no movement, higher = more movement)
        """
        if previous_landmarks is None:
            return 0.0
        
        try:
            # Key landmarks to track for movement (upper body)
            key_landmarks = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ]
            
            total_movement = 0.0
            for landmark_id in key_landmarks:
                curr = current_landmarks[landmark_id]
                prev = previous_landmarks[landmark_id]
                
                # Calculate Euclidean distance
                distance = np.sqrt(
                    (curr.x - prev.x) ** 2 + 
                    (curr.y - prev.y) ** 2
                )
                total_movement += distance
            
            # Normalize by number of landmarks
            movement_score = total_movement / len(key_landmarks)
            
            return movement_score
            
        except Exception as e:
            return 0.0
    
    def detect_pose_based_sleep(self, pose_landmarks, timestamp_sec):
        """Detect sleep based on pose analysis when face detection fails.
        
        Criteria:
        - Head tilted significantly forward/down (< -20 degrees)
        - Minimal movement over extended period
        - Stable posture maintained
        
        Returns:
            tuple: (is_sleeping, is_microsleeping, debug_info)
        """
        if not pose_landmarks:
            return False, False, {}
        
        # Calculate head tilt angle
        head_tilt = self.calculate_head_tilt_angle(pose_landmarks.landmark)
        
        # Calculate movement score
        movement_score = self.calculate_movement_score(
            pose_landmarks.landmark,
            self.previous_pose_landmarks
        )
        
        # Update history
        if head_tilt is not None:
            self.head_tilt_history.append(head_tilt)
        
        self.movement_history.append(movement_score)
        
        # Store current landmarks for next frame
        self.previous_pose_landmarks = pose_landmarks.landmark
        
        # Need sufficient history to make determination
        min_samples = max(5, int(5 * self.sample_fps))  # At least 5 seconds
        
        if len(self.head_tilt_history) < min_samples or len(self.movement_history) < min_samples:
            return False, False, {
                'head_tilt': head_tilt,
                'movement': movement_score,
                'status': 'building_history'
            }
        
        # Calculate average head tilt over recent period
        avg_head_tilt = np.mean(list(self.head_tilt_history))
        
        # Calculate average movement over recent period
        avg_movement = np.mean(list(self.movement_history))
        
        # Sleep indicators:
        # 1. Head tilted forward significantly (< -15 degrees)
        # 2. Very low movement (< 0.02) - increased from 0.01 for more realistic detection
        # 3. Consistent over time (low variance)
        
        head_tilt_variance = np.var(list(self.head_tilt_history))
        movement_variance = np.var(list(self.movement_history))
        
        is_head_down = avg_head_tilt < -15
        is_minimal_movement = avg_movement < 0.02  # Increased from 0.01
        is_stable_posture = head_tilt_variance < 100  # Low variance = stable position (increased from 50)
        
        debug_info = {
            'head_tilt': head_tilt,
            'avg_head_tilt': avg_head_tilt,
            'movement': movement_score,
            'avg_movement': avg_movement,
            'head_tilt_variance': head_tilt_variance,
            'is_head_down': is_head_down,
            'is_minimal_movement': is_minimal_movement,
            'is_stable_posture': is_stable_posture
        }
        
        # Detect sleep condition
        sleep_indicators_met = is_head_down and is_minimal_movement and is_stable_posture
        
        if sleep_indicators_met:
            if self.pose_sleep_start is None:
                self.pose_sleep_start = timestamp_sec
                print(f"[Pose-Based Sleep] Started tracking - head_tilt={avg_head_tilt:.1f}°, movement={avg_movement:.4f}")
            
            self.pose_sleep_duration = timestamp_sec - self.pose_sleep_start
            
            # Check thresholds
            is_sleeping = self.pose_sleep_duration >= 30  # 30 seconds
            is_microsleeping = self.pose_sleep_duration >= 5 and not is_sleeping  # 5 seconds
            
            debug_info['pose_sleep_duration'] = self.pose_sleep_duration
            
            return is_sleeping, is_microsleeping, debug_info
        else:
            # Reset if conditions not met
            if self.pose_sleep_start is not None:
                print(f"[Pose-Based Sleep] Stopped - indicators not met")
            self.pose_sleep_start = None
            self.pose_sleep_duration = 0
            
            return False, False, debug_info
    
    def detect_objects(self, frame):
        """Detect objects using YOLO - relevant items for locomotive cabin"""
        results = self.yolo_model(frame, verbose=False)  # Suppress YOLO output
        detections = {
            'person': [],
            'cell_phone': [],
            'book': [],
            'backpack': []
        }
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = self.yolo_model.names[cls]
                if class_name == 'person' and conf > 0.5:
                    detections['person'].append(xyxy)
                elif class_name == 'cell phone' and conf > 0.5:  # Lowered from 0.5 for better detection
                    detections['cell_phone'].append(xyxy)
                elif class_name == 'book' and conf > 0.1:  # Lowered threshold for better book detection
                    detections['book'].append(xyxy)
                elif class_name == 'backpack' and conf > 0.4:
                    detections['backpack'].append(xyxy)
        
        return detections
    
    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes on frame for detected objects"""
        annotated_frame = frame.copy()
        
        colors = {
            'person': (0, 255, 0),
            'cell_phone': (0, 0, 255),
            'book': (255, 0, 0),
            'backpack': (0, 255, 255)
        }
        
        for obj_type, bboxes in detections.items():
            color = colors.get(obj_type, (255, 255, 255))
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
        
        return annotated_frame
    
    def draw_mediapipe_outputs(self, frame, pose_results, face_results, ear_value=None, eye_closure_duration=0, pose_sleep_info=None):
        """Draw MediaPipe pose and face mesh landmarks on frame"""
        annotated_frame = frame.copy()
        
        face_detected = face_results.multi_face_landmarks is not None and len(face_results.multi_face_landmarks) > 0
        
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if face_detected:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        if face_detected:
            if ear_value is not None:
                if ear_value < 0.2:
                    status = "EYES CLOSED"
                    color = (0, 0, 255)
                else:
                    status = "EYES OPEN"
                    color = (0, 255, 0)
                
                ear_text = f"EAR: {ear_value:.3f} - {status}"
                cv2.putText(annotated_frame, ear_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                threshold_text = "Threshold: < 0.2 = Closed"
                cv2.putText(annotated_frame, threshold_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                if eye_closure_duration > 0:
                    duration_text = f"Closed Duration: {eye_closure_duration:.1f}s"
                    duration_color = (0, 165, 255)
                    
                    if eye_closure_duration >= 30:
                        duration_text += " - SLEEP ALERT!"
                        duration_color = (0, 0, 255)
                    elif eye_closure_duration >= 5:
                        duration_text += " - MICROSLEEP!"
                        duration_color = (0, 140, 255)
                    
                    cv2.putText(annotated_frame, duration_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2, cv2.LINE_AA)
            else:
                warning_text = "FACE DETECTED - EAR CALC ISSUE"
                cv2.putText(annotated_frame, warning_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            no_face_text = "FACE NOT DETECTED"
            cv2.putText(annotated_frame, no_face_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
        
        # Display pose-based sleep detection info (when face not detected or as backup)
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
                
                if duration >= 30:
                    duration_text += " - SLEEP DETECTED!"
                    duration_color = (0, 0, 255)
                elif duration >= 5:
                    duration_text += " - MICROSLEEP!"
                    duration_color = (0, 140, 255)
                else:
                    duration_color = (0, 165, 255)
                
                cv2.putText(annotated_frame, duration_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2, cv2.LINE_AA)
        
        return annotated_frame
    
    def check_hand_object_interaction(self, hand_coords, object_bbox, margin=50):
        """Check if hand is interacting with an object
        
        Args:
            hand_coords: (x, y) coordinates of hand
            object_bbox: [x1, y1, x2, y2] bounding box of object
            margin: proximity margin in pixels (default 50, use 30 for tighter checks)
        """
        if hand_coords is None or object_bbox is None:
            return False
        
        hx, hy = hand_coords
        x1, y1, x2, y2 = object_bbox
        return (x1 - margin <= hx <= x2 + margin and 
                y1 - margin <= hy <= y2 + margin)
    
    def start_activity(self, activity_name, timestamp, fps, frame_count):
        """Start tracking an activity"""
        if not self.activities[activity_name]['active']:
            self.activities[activity_name]['active'] = True
            self.activities[activity_name]['start_time'] = timestamp
            self.activities[activity_name]['start_frame_count'] = frame_count
            self.activities[activity_name]['last_frame_count'] = frame_count
            self.activities[activity_name]['frames'] = list(self.frame_buffer)
            self.activities[activity_name]['duration'] = 0
            print(f"[{timestamp}] Activity started: {activity_name}")
    
    def end_activity(self, activity_name, timestamp, fps, frame_count, people_count=1):
        """End tracking an activity and save evidence (only if meets minimum duration)"""
        if self.activities[activity_name]['active']:
            activity = self.activities[activity_name]
            activity['active'] = False
            
            start_frame = activity.get('start_frame_count', frame_count)
            duration = (frame_count - start_frame) / fps
            
            # Check if activity meets minimum duration threshold
            min_duration = self.activity_thresholds[activity_name]['min_duration']
            
            if duration < min_duration:
                print(f"[{timestamp}] Activity '{activity_name}' too short ({duration:.2f}s < {min_duration}s) - discarded")
                activity['frames'] = []
                activity['duration'] = 0
                self.consecutive_detections[activity_name] = 0
                self.grace_counters[activity_name] = 0
                return
            
            start_time_str = activity['start_time']
            end_time_str = timestamp
            
            # Parse activity start and end times in seconds
            def time_to_seconds(time_str):
                """Convert HH:MM:SS.microseconds to seconds"""
                parts = time_str.split(':')
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            
            activity_start_seconds = time_to_seconds(start_time_str)
            activity_end_seconds = time_to_seconds(end_time_str)
            
            # Generate filenames
            video_filename = os.path.basename(self.video_path)
            video_name_without_ext = os.path.splitext(video_filename)[0]
            
            clip_filename = f"{video_name_without_ext}_ts{int(activity_start_seconds):04d}_{self.evidence_counter:03d}_clip.mp4"
            image_filename = f"{video_name_without_ext}_ts{int(activity_start_seconds):04d}_{self.evidence_counter:03d}_activity.jpg"
            
            clip_path = os.path.join(self.evidence_clips_dir, clip_filename)
            image_path = os.path.join(self.evidence_clips_dir, image_filename)
            
            # Save video clip
            self.save_video_clip(activity['frames'], clip_path, fps)
            
            # Save activity image (middle frame of the activity)
            if len(activity['frames']) > 0:
                middle_frame_idx = len(activity['frames']) // 2
                activity_image = activity['frames'][middle_frame_idx]
                cv2.imwrite(image_path, activity_image)
            
            total_clip_frames = len(activity['frames'])
            total_clip_duration = total_clip_frames / fps
            
            # Get video duration in HH:MM:SS format
            cap = cv2.VideoCapture(self.video_path)
            video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration_seconds = video_total_frames / fps
            cap.release()
            
            video_duration_formatted = str(timedelta(seconds=int(video_duration_seconds)))
            
            # Get current date and time
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            
            # Create JSON data in the required format
            json_data = {
                "tripId": self.trip_id,
                "activityType": self.activity_type_map[activity_name],
                "des": self.activity_descriptions[activity_name],
                "objectType": activity_name.replace('_', ' '),
                "fileUrl": os.path.abspath(self.video_path),
                "fileDuration": video_duration_formatted,
                "activityStartTime": f"{activity_start_seconds:.2f}",
                "activityEndTime": f"{activity_end_seconds:.2f}",
                "crewName": self.crew_name,
                "crewId": self.crew_id,
                "crewRole": self.crew_role,
                "date": current_date,
                "time": current_time,
                "filename": video_filename,
                "peopleCount": people_count,
                "evidence": {"rule": self.evidence_rules[activity_name]},
                "activityImage": image_filename,
                "activityClip": clip_filename
            }
            
            # Add to all activities list
            self.all_activities.append(json_data)
            
            # Also save individual JSON file for backward compatibility
            json_filename = f"{activity_name}_{self.evidence_counter:04d}.json"
            json_path = os.path.join(self.json_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"[{end_time_str}] Activity ended: {activity_name}")
            print(f"  Activity Duration: {duration:.2f}s | Total Clip: {total_clip_duration:.2f}s")
            print(f"  Min Duration Threshold: {min_duration}s | Required Consecutive: {self.activity_thresholds[activity_name]['required_consecutive']} frames")
            print(f"  Evidence saved: {clip_filename} ({total_clip_frames} frames)")
            print(f"  Activity image: {image_filename}")
            
            activity['frames'] = []
            activity['duration'] = 0
            self.consecutive_detections[activity_name] = 0
            self.grace_counters[activity_name] = 0
            
            self.evidence_counter += 1
    
    def save_video_clip(self, frames, output_path, fps):
        """Save frames as video clip"""
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def process_video(self):
        """Main video processing loop - SAMPLES FRAMES AT SPECIFIED RATE"""
        # Get video metadata
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate expected sampled frames
        step = max(1, int(round(fps / max(1e-6, float(self.sample_fps)))))
        expected_samples = (total_frames // step)
        
        print(f"Processing video: {self.video_path}")
        print(f"Native FPS: {fps:.2f}")
        print(f"Sample FPS: {self.sample_fps} (1 frame every {1.0/self.sample_fps:.1f} seconds)")
        print(f"Total frames in video: {total_frames}")
        print(f"Expected duration: {total_frames/fps/60:.2f} minutes")
        print(f"Expected sampled frames: ~{expected_samples}")
        print(f"Processing speed-up: ~{step}x faster")
        if self.save_annotated_frames:
            if self.frame_save_interval == 1:
                print(f"Saving ALL sampled frames (~{expected_samples} frames) to: {self.frames_dir}")
            else:
                print(f"Saving every {self.frame_save_interval}th sampled frame (~{expected_samples//self.frame_save_interval} frames) to: {self.frames_dir}")
        else:
            print("Annotated frame saving is disabled (faster processing)")
        print("-" * 60)
        
        sampled_count = 0
        
        # Use the frame sampling generator
        for sample_idx, timestamp_sec, frame, frame_idx in self.sample_video_frames(self.video_path):
            sampled_count += 1
            
            try:
                # Convert timestamp to HH:MM:SS format
                timestamp = str(timedelta(seconds=timestamp_sec))
                
                # Add frame to buffer
                self.frame_buffer.append(frame.copy())
                
                # Process pose and face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb_frame)
                face_results = self.face_mesh.process(rgb_frame)
                
                # Calculate EAR for all detected faces (check all people)
                ear_value = None
                min_ear_value = None  # Track the lowest EAR (most closed eyes)
                
                if face_results.multi_face_landmarks:
                    # Check all detected faces
                    ear_values = []
                    for face_landmarks in face_results.multi_face_landmarks:
                        ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark)
                        if ear is not None:
                            ear_values.append(ear)
                    
                    # Use the minimum EAR (most closed eyes) for microsleep detection
                    if ear_values:
                        min_ear_value = min(ear_values)
                        ear_value = min_ear_value  # For display purposes
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Count people in frame
                people_count = len(detections['person'])
                if people_count == 0:
                    people_count = 1  # Default to 1 if no person detected
                
                # Run pose-based sleep detection (always, as backup or primary method)
                pose_sleep_detected = False
                pose_microsleep_detected = False
                pose_sleep_info = {}
                
                if pose_results.pose_landmarks:
                    pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.detect_pose_based_sleep(
                        pose_results.pose_landmarks, timestamp_sec
                    )
                
                # Save annotated frames periodically if enabled
                if self.save_annotated_frames and sample_idx % self.frame_save_interval == 0:
                    annotated_frame = self.draw_bounding_boxes(frame, detections)
                    annotated_frame = self.draw_mediapipe_outputs(
                        annotated_frame, pose_results, face_results, 
                        ear_value, self.eye_closure_duration, pose_sleep_info
                    )
                    frame_filename = f"frame_{frame_idx:08d}.jpg"
                    frame_path = os.path.join(self.frames_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)
                
                # Initialize detection flags
                microsleep_detected = False
                sleep_detected = False
                cell_phone_detected = False
                writing_detected = False
                packing_detected = False
                
                # Check for sleep/microsleep
                # Priority 1: Face-based detection (most accurate when face is visible)
                if face_results.multi_face_landmarks and ear_value is not None:
                    if ear_value < 0.2:
                        if self.eye_closure_start is None:
                            self.eye_closure_start = timestamp_sec
                        
                        self.eye_closure_duration = timestamp_sec - self.eye_closure_start
                        
                        if self.eye_closure_duration >= 30:
                            sleep_detected = True
                        elif self.eye_closure_duration >= 5:
                            microsleep_detected = True
                    else:
                        self.eye_closure_start = None
                        self.eye_closure_duration = 0
                
                # Priority 2: Pose-based detection (fallback when face not detected)
                # OR use pose detection to confirm face-based detection
                else:
                    # Face not detected, use pose-based detection
                    if pose_sleep_detected:
                        sleep_detected = True
                    elif pose_microsleep_detected:
                        microsleep_detected = True
                
                # Check for cell phone usage
                if pose_results.pose_landmarks and len(detections['cell_phone']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    
                    margin = self.activity_thresholds['cell_phone']['margin']
                    for phone_bbox in detections['cell_phone']:
                        if (self.check_hand_object_interaction(right_hand_coords, phone_bbox, margin) or
                            self.check_hand_object_interaction(left_hand_coords, phone_bbox, margin)):
                            cell_phone_detected = True
                            
                            # Log cell phone in hand detection
                            if self.consecutive_detections['cell_phone'] == 0:
                                print(f"[{timestamp}] Cell phone detected in hand (frame {frame_idx}, sample {sample_idx})")
                            
                            break
                
                # Check for writing
                if pose_results.pose_landmarks and len(detections['book']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    
                    margin = self.activity_thresholds['writing']['margin']
                    for book_bbox in detections['book']:
                        if self.check_hand_object_interaction(right_hand_coords, book_bbox, margin):
                            writing_detected = True
                            break
                
                # Check for packing bags - backpacks only
                if pose_results.pose_landmarks and len(detections['backpack']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    
                    # Check interaction with backpacks
                    margin = self.activity_thresholds['packing_bags']['margin']
                    
                    for backpack_bbox in detections['backpack']:
                        if (self.check_hand_object_interaction(right_hand_coords, backpack_bbox, margin) or
                            self.check_hand_object_interaction(left_hand_coords, backpack_bbox, margin)):
                            packing_detected = True
                            break
                
                # Update activity states with temporal filtering
                activities_map = {
                    'microsleep': microsleep_detected and not sleep_detected,
                    'sleep': sleep_detected,
                    'cell_phone': cell_phone_detected,
                    'writing': writing_detected,
                    'packing_bags': packing_detected
                }
                
                for activity_name, detected in activities_map.items():
                    if detected:
                        # Activity detected - increment consecutive counter and reset grace period
                        self.consecutive_detections[activity_name] += 1
                        self.grace_counters[activity_name] = 0  # Reset grace period
                        
                        # Only start recording after required consecutive frames threshold is met
                        required_consecutive = self.activity_thresholds[activity_name]['required_consecutive']
                        
                        if self.consecutive_detections[activity_name] >= required_consecutive:
                            # Start activity if not already active
                            if not self.activities[activity_name]['active']:
                                self.start_activity(activity_name, timestamp, fps, frame_idx)
                            
                            # Continue recording frames
                            if self.activities[activity_name]['active']:
                                self.activities[activity_name]['frames'].append(frame.copy())
                                self.activities[activity_name]['last_frame_count'] = frame_idx
                    else:
                        # Activity not detected - use grace period before resetting
                        if self.consecutive_detections[activity_name] > 0 or self.activities[activity_name]['active']:
                            # Increment grace counter
                            self.grace_counters[activity_name] += 1
                            grace_frames = self.activity_thresholds[activity_name]['grace_frames']
                            
                            # If still within grace period, continue as if detected
                            if self.grace_counters[activity_name] <= grace_frames:
                                # Still in grace period - continue recording if active
                                if self.activities[activity_name]['active']:
                                    self.activities[activity_name]['frames'].append(frame.copy())
                                    self.activities[activity_name]['last_frame_count'] = frame_idx
                            else:
                                # Grace period exceeded - end activity and reset counters
                                if self.activities[activity_name]['active']:
                                    self.end_activity(activity_name, timestamp, fps, frame_idx, people_count)
                                self.consecutive_detections[activity_name] = 0
                                self.grace_counters[activity_name] = 0
                        else:
                            # Reset grace counter if nothing is being tracked
                            self.grace_counters[activity_name] = 0
                
                # Display progress with detection status
                if sample_idx % 50 == 0:  # Show progress every 50 sampled frames
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {sample_idx} samples processed (frame {frame_idx}/{total_frames}, {progress:.1f}%)")
                    
                    # Show current detection counts for debugging
                    active_detections = []
                    for act_name, count in self.consecutive_detections.items():
                        if count > 0:
                            threshold = self.activity_thresholds[act_name]['required_consecutive']
                            status = "RECORDING" if self.activities[act_name]['active'] else f"building {count}/{threshold}"
                            active_detections.append(f"{act_name}: {status}")
                    
                    if active_detections:
                        print(f"  Active detections: {', '.join(active_detections)}")
            
            except Exception as e:
                print(f"\nError processing sample {sample_idx} (frame {frame_idx}): {e}")
                continue
        
        # End any remaining active activities
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, 1)  # Default to 1 person for final activities
        
        print(f"\n{'=' * 60}")
        print(f"Processing complete!")
        print(f"Total frames sampled: {sampled_count}/{total_frames}")
        print(f"Sampling rate: {self.sample_fps} FPS (1 frame every {1.0/self.sample_fps:.1f} seconds)")
        print(f"Processing speed-up: ~{step}x faster than full-frame processing")
        print(f"Evidence clips created: {self.evidence_counter}")
        print(f"Evidence saved in: {self.output_dir}")
        if self.save_annotated_frames:
            print(f"Annotated frames saved in: {self.frames_dir}")
        print(f"{'=' * 60}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of all activities in JSON array format"""
        # Save the activities array in the main output directory
        activities_json_path = os.path.join(self.output_dir, "activities.json")
        with open(activities_json_path, 'w') as f:
            json.dump(self.all_activities, f, indent=2)
        
        print(f"\nActivities JSON saved: {activities_json_path}")
        print(f"Total activities detected: {len(self.all_activities)}")
        
        # Also create a detailed summary for reference
        summary = {
            "video_path": self.video_path,
            "processing_date": datetime.now().isoformat(),
            "total_evidence_clips": self.evidence_counter,
            "total_activities": len(self.all_activities),
            "activities_by_type": {}
        }
        
        if self.save_annotated_frames:
            summary["run_id"] = self.run_timestamp
            summary["frames_directory"] = self.frames_dir
        
        # Count activities by type
        for activity in self.all_activities:
            activity_type = activity['des']
            if activity_type not in summary["activities_by_type"]:
                summary["activities_by_type"][activity_type] = 0
            summary["activities_by_type"][activity_type] += 1
        
        summary_path = os.path.join(self.output_dir, "summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved: {summary_path}")
        
        # Print activity breakdown
        if summary["activities_by_type"]:
            print("\nActivity Breakdown:")
            for activity_type, count in summary["activities_by_type"].items():
                print(f"  - {activity_type}: {count}")


# Usage example
if __name__ == "__main__":
    video_path = "example_data/latest_1.mp4"
    
    # Option 1: Sample at 0.5 FPS and save ALL sampled frames
    # This samples 1 frame every 2 seconds, making processing ~60x faster for 30fps videos
    monitor = LocopilotActivityMonitor(
        video_path, 
        output_dir="locopilot_evidence",
        save_annotated_frames=True,  # Enable frame saving
        frame_save_interval=1,  # Save EVERY sampled frame (1 = save all)
        sample_fps=0.5  # Sample at 0.5 FPS (1 frame every 2 seconds)
    )
    
    # Option 2: Sample at 1.0 FPS and save all sampled frames
    # monitor = LocopilotActivityMonitor(
    #     video_path, 
    #     output_dir="locopilot_evidence",
    #     save_annotated_frames=True,
    #     frame_save_interval=1,  # Save EVERY sampled frame
    #     sample_fps=1.0  # Sample at 1.0 FPS (1 frame per second)
    # )
    
    # Option 3: Sample at 2.0 FPS without saving frames (FASTEST for high sample rate)
    # monitor = LocopilotActivityMonitor(
    #     video_path, 
    #     output_dir="locopilot_evidence",
    #     save_annotated_frames=False,  # Disable for maximum speed
    #     sample_fps=2.0  # Sample at 2 FPS (1 frame every 0.5 seconds)
    # )
    
    # Option 4: Save only some sampled frames (e.g., every 10th sampled frame)
    # monitor = LocopilotActivityMonitor(
    #     video_path, 
    #     output_dir="locopilot_evidence",
    #     save_annotated_frames=True,
    #     frame_save_interval=10,  # Save every 10th sampled frame (for storage efficiency)
    #     sample_fps=1.0  # Sample at 1 FPS
    # )
    
    monitor.process_video()