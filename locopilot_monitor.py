import cv2
import json
import numpy as np
import math
from datetime import datetime, timedelta
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
import os

class LocopilotActivityMonitor:
    def __init__(self, video_path, output_dir="evidence", save_annotated_frames=False, frame_save_interval=1, sample_fps=1.0, run_dir=None, create_run_dir=True, enable_gesture_debug=False, gesture_sensitivity='balanced'):
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Frame sampling configuration
        self.sample_fps = sample_fps  # Sample frames at this rate (e.g., 0.5 = 1 frame every 2 seconds)
        
        # Gesture debug and monitoring configuration
        self.enable_gesture_debug = enable_gesture_debug
        self.gesture_sensitivity = gesture_sensitivity
        
        # Control annotated frame saving
        self.save_annotated_frames = save_annotated_frames
        self.frame_save_interval = frame_save_interval  # Save 1 frame every N sampled frames (1 = save all sampled frames)
        
        # Create or use existing run directory
        if run_dir is not None:
            # Use provided run directory (for multiprocessing)
            self.run_dir = run_dir
            self.run_timestamp = os.path.basename(run_dir).replace("run_", "")
        elif create_run_dir:
            # Create new run-specific directory
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(output_dir, f"run_{self.run_timestamp}")
        else:
            # No run directory (for multiprocessing workers)
            self.run_dir = None
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories only if run_dir is set
        if self.run_dir:
            self.evidence_clips_dir = os.path.join(self.run_dir, "clips")
            self.frames_dir = os.path.join(self.run_dir, "frames")
            
            # Create directories
            os.makedirs(self.evidence_clips_dir, exist_ok=True)
            if self.save_annotated_frames:
                os.makedirs(self.frames_dir, exist_ok=True)
        else:
            # No directories for multiprocessing workers (activities in memory only)
            self.evidence_clips_dir = None
            self.frames_dir = None
        
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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Activity tracking with temporal filtering
        self.activities = {
            'microsleep': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'sleep': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'cell_phone': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'writing': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'packing_bags': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'group_detected': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'lp_hand_gesture': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'alp_hand_gesture': {'active': False, 'start_time': None, 'frames': [], 'duration': 0},
            'mind_diversion': {'active': False, 'start_time': None, 'frames': [], 'duration': 0}
        }
        
        # Activity thresholds: minimum duration and required consecutive frames before recording starts
        self.activity_thresholds = {
            'packing_bags': {
                'min_duration': 1.5,          # Must last 1.5 seconds minimum (reduced from 3.0)
                'required_consecutive': 1,    # 1 sample @ 0.5fps = 2 seconds before recording (equivalent to ~50 frames at 30fps)
                'margin': 50,                 # More lenient proximity (increased from 30)
                'grace_frames': 5             # Allow 5 samples (~10s) gap to group nearby detections
            },
            'writing': {
                'min_duration': 2.0,          # Must last 2 seconds minimum
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds before recording
                'margin': 60,                 # More lenient proximity (increased from 50)
                'grace_frames': 5             # Allow 5 samples (~10s) gap to group nearby detections
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
            },
            'group_detected': {
                'min_duration': 2.0,          # Must last 2 seconds minimum
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds before recording
                'margin': None,               # N/A for person count detection
                'grace_frames': 5             # Allow 5 samples (~10s) gap
            },
            'lp_hand_gesture': {
                'min_duration': 2.0,          # Must last 2 seconds minimum
                'required_consecutive': 2,    # 2 samples @ 0.5fps = 4 seconds before recording
                'margin': None,               # N/A for hand gesture detection
                'grace_frames': 3             # Allow 3 samples (~6s) gap to handle multiple raises
            },
            'alp_hand_gesture': {
                'min_duration': 2.0,          # Must last 2 seconds minimum
                'required_consecutive': 2,    # 2 samples @ 0.5fps = 4 seconds before recording
                'margin': None,               # N/A for hand gesture detection
                'grace_frames': 3             # Allow 3 samples (~6s) gap to handle multiple raises
            },
            'mind_diversion': {
                'min_duration': 5.0,          # Must last 5 seconds minimum
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds before recording
                'margin': None,               # N/A for head pose detection
                'grace_frames': 3             # Allow 3 samples (~6s) gap
            }
        }
        
        # Consecutive detection counters for temporal filtering
        self.consecutive_detections = {
            'microsleep': 0,
            'sleep': 0,
            'cell_phone': 0,
            'writing': 0,
            'packing_bags': 0,
            'group_detected': 0,
            'lp_hand_gesture': 0,
            'alp_hand_gesture': 0,
            'mind_diversion': 0
        }
        
        # Grace period counters - allows brief interruptions without resetting
        self.grace_counters = {
            'microsleep': 0,
            'sleep': 0,
            'cell_phone': 0,
            'writing': 0,
            'packing_bags': 0,
            'group_detected': 0,
            'lp_hand_gesture': 0,
            'alp_hand_gesture': 0,
            'mind_diversion': 0
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
            'packing_bags': 6,
            'group_detected': 7,
            'lp_hand_gesture': 8,
            'alp_hand_gesture': 9
        }
        
        # Activity descriptions
        self.activity_descriptions = {
            'cell_phone': 'Using mobile phone',
            'microsleep': 'Micro-sleep detected (5+ seconds)',
            'sleep': 'Sleep detected (30+ seconds)',
            'writing': 'WRITING LOG BOOK WHILE RUNNING',
            'packing_bags': 'Packing bags activity detected',
            'group_detected': 'More than 2 people (group) detected',
            'lp_hand_gesture': 'LP not exchanging hand gesture',
            'alp_hand_gesture': 'ALP not exchanging hand gesture',
            'mind_diversion': 'Mind diversion - attention diverted from controls'
        }
        
        # Evidence rules
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'eyes_closed_5s_or_pose_indicators',
            'sleep': 'eyes_closed_30s_or_pose_indicators',
            'writing': 'hand_near_book',
            'packing_bags': 'hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons',
            'lp_hand_gesture': 'lp_hand_raised_gesture_detected',
            'alp_hand_gesture': 'alp_hand_raised_gesture_detected',
            'mind_diversion': 'head_turned_side_and_down'
        }
        
        # Default crew/trip information
        self.trip_id = "TRIP-123"
        self.crew_name = "John Doe"
        self.crew_id = "C-001"
        self.crew_role = 1  # 1 for primary loco pilot
        
        # Crew members mapping: role (LP/ALP) -> {name, id, role}
        self.crew_members = {}  # Will be populated from API input
        
        # Store all activities for final JSON array output
        self.all_activities = []
        
        # Gesture statistics tracking
        self.gesture_stats = {
            'total_frames_analyzed': 0,
            'successful_detections': {
                'lp': 0,
                'alp': 0,
                'total': 0
            },
            'rejections_by_reason': {
                'control_zone': 0,
                'insufficient_height': 0,
                'insufficient_extension': 0,
                'elbow_position': 0,
                'visibility_low': 0,
                'out_of_bounds': 0,
                'pose_not_matched': 0,
                'wrist_not_in_bbox': 0,
                'elbow_not_below_wrist': 0,
                # v3.0 NEW REJECTION REASONS
                'arm_not_vertical': 0,
                'hand_below_head': 0
            },
            'confidence_distribution': {
                'high': 0,    # >80%
                'medium': 0,  # 50-80%
                'low': 0      # <50%
            }
        }
        
        # Sensitivity mode thresholds
        self.sensitivity_thresholds = {
            'strict': {
                'wrist_shoulder_vertical': 120,    # High threshold
                'wrist_elbow_distance': 60,
                'arm_extension': 80,
                'elbow_shoulder_margin': -30,
                'control_zone_extension': 100,
                'control_zone_elbow_distance': 40
            },
            'balanced': {
                'wrist_shoulder_vertical': 80,     # Current v2.0 values
                'wrist_elbow_distance': 40,
                'arm_extension': 60,
                'elbow_shoulder_margin': -40,
                'control_zone_extension': 120,
                'control_zone_elbow_distance': 50
            },
            'sensitive': {
                'wrist_shoulder_vertical': 60,     # Lower threshold
                'wrist_elbow_distance': 30,
                'arm_extension': 50,
                'elbow_shoulder_margin': -50,
                'control_zone_extension': 140,
                'control_zone_elbow_distance': 60
            }
        }
        
        # Get thresholds for selected sensitivity mode
        self.gesture_thresholds = self.sensitivity_thresholds.get(gesture_sensitivity, self.sensitivity_thresholds['balanced'])
        
        # Log configuration if debug enabled
        if self.enable_gesture_debug:
            print(f"[GESTURE DEBUG] Enabled with sensitivity mode: {gesture_sensitivity}")
            print(f"[GESTURE DEBUG] Thresholds: {self.gesture_thresholds}")
    
    def sample_video_frames(self, video_path, start_frame=None, end_frame=None):
        """Sample frames at fixed intervals based on sample_fps.
        
        Yields tuples: (sample_index, timestamp_sec, frame_bgr, frame_idx)
        
        Args:
            video_path: Path to video file
            start_frame: Optional starting frame index (for range processing)
            end_frame: Optional ending frame index (for range processing)
            
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
        
        # Determine frame range
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else total_frames
        
        # Calculate stride: how many frames to skip between samples
        step = max(1, int(round(native_fps / max(1e-6, float(self.sample_fps)))))
        
        print(f"[Frame Sampling] Native FPS: {native_fps:.2f}, Sample FPS: {self.sample_fps}")
        print(f"[Frame Sampling] Step: {step} (sampling 1 frame every {step} frames)")
        print(f"[Frame Sampling] Frame range: {start_frame} - {end_frame}")
        print(f"[Frame Sampling] Expected sampled frames: ~{((end_frame - start_frame) // step)}")
        
        sampled_idx = 0
        # Start from the beginning of the range, aligned to step
        first_sample_frame = start_frame + (step - (start_frame % step)) % step
        
        for frame_idx in range(first_sample_frame, end_frame, step):
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
        # 1. Head tilted forward VERY significantly (< -100 degrees) - stricter to avoid false positives during normal work
        # 2. Low movement (< 0.05) - allows some minimal working movement
        # 3. Consistent over time (low variance)
        
        head_tilt_variance = np.var(list(self.head_tilt_history))
        movement_variance = np.var(list(self.movement_history))
        
        is_head_down = avg_head_tilt < -100  # Changed from -15 to -100 (stricter)
        is_minimal_movement = avg_movement < 0.1  # Changed from 0.02 to 0.05 (more lenient)
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
    
    def get_roi_around_keypoint(self, keypoint_coords, frame_shape, roi_size=150):
        """Create Region of Interest (ROI) box around a keypoint.
        
        Args:
            keypoint_coords: (x, y) coordinates of keypoint
            frame_shape: (height, width) of frame
            roi_size: Size of ROI box in pixels (default 150x150)
            
        Returns:
            (x1, y1, x2, y2) ROI bounding box, or None if invalid
        """
        if keypoint_coords is None:
            return None
        
        h, w = frame_shape[:2]
        x, y = keypoint_coords
        
        # Create square ROI centered on keypoint
        half_size = roi_size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        # Ensure minimum ROI size
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return None
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def detect_objects_in_roi(self, frame, roi_bbox, target_classes=['cell phone', 'book', 'pen', 'pencil']):
        """Run YOLO detection on a specific ROI region.
        
        Args:
            frame: Full frame
            roi_bbox: (x1, y1, x2, y2) ROI bounding box
            target_classes: List of class names to detect in ROI
            
        Returns:
            List of detections with global coordinates: [(class_name, conf, x1, y1, x2, y2), ...]
        """
        if roi_bbox is None:
            return []
        
        x1, y1, x2, y2 = roi_bbox
        roi_frame = frame[y1:y2, x1:x2]
        
        # Run YOLO on ROI with lower confidence threshold
        results = self.yolo_model(roi_frame, verbose=False, conf=0.1)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy_local = box.xyxy[0].cpu().numpy()
                
                class_name = self.yolo_model.names[cls]
                
                # Check if this is a target class
                if class_name in target_classes:
                    # Convert local ROI coordinates to global frame coordinates
                    global_x1 = xyxy_local[0] + x1
                    global_y1 = xyxy_local[1] + y1
                    global_x2 = xyxy_local[2] + x1
                    global_y2 = xyxy_local[3] + y1
                    
                    detections.append((class_name, conf, global_x1, global_y1, global_x2, global_y2))
        
        return detections
    
    def detect_objects(self, frame, pose_landmarks=None, use_pose_guided=True):
        """Detect objects using YOLO with pose-guided detection.
        
        MULTI-LAYERED DETECTION FLOW:
        1. Full frame detection for:
           - Person (for counting)
           - Backpack (for packing detection)
           - Book (low confidence, only if near person) - OPTION 3
        2. ROI-based detection around landmarks for activity objects:
           - Hands (wrists, index) - 250px radius (OPTION 2 - increased)
           - Lap/Torso (hips) - 280px radius (OPTION 1 - added)
           - Ears, mouth - for phone/eating detection
        3. This provides comprehensive detection while minimizing false positives
        
        Args:
            frame: Input frame
            pose_landmarks: MediaPipe pose landmarks (optional)
            use_pose_guided: Enable pose-guided ROI detection (default True)
            
        Returns:
            Dictionary with detections and ROI information
        """
        # Stage 1: Full frame detection for person, backpack, and books near person
        results = self.yolo_model(frame, verbose=False)
        detections = {
            'person': [],
            'cell_phone': [],
            'book': [],
            'backpack': [],
            'roi_detections': [],  # ROI-based detections (main activity detection)
            'roi_boxes': []  # ROI boxes for visualization
        }
        
        # Store person boxes for proximity checking
        person_boxes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = self.yolo_model.names[cls]
                # Detect person and backpack from full frame
                if class_name == 'person' and conf > 0.5:
                    detections['person'].append(xyxy)
                    person_boxes.append(xyxy)
                elif class_name == 'backpack' and conf > 0.5:
                    detections['backpack'].append(xyxy)
                # OPTION 3: Re-enable book detection in full frame with low confidence
                # But only if book is within reasonable distance of a person
                elif class_name == 'book' and conf > 0.2:  # Low threshold for books
                    # Check if book is near any detected person
                    if len(person_boxes) > 0:
                        book_near_person = False
                        book_center_x = (xyxy[0] + xyxy[2]) / 2
                        book_center_y = (xyxy[1] + xyxy[3]) / 2
                        
                        for person_box in person_boxes:
                            # Check if book center is within expanded person bounding box
                            person_x1, person_y1, person_x2, person_y2 = person_box
                            margin = 200  # 200px margin around person box
                            if (person_x1 - margin <= book_center_x <= person_x2 + margin and
                                person_y1 - margin <= book_center_y <= person_y2 + margin):
                                book_near_person = True
                                break
                        
                        if book_near_person:
                            detections['book'].append(xyxy)
                    else:
                        # No person detected, add book anyway (fallback)
                        detections['book'].append(xyxy)
        
        # Stage 2: Pose-guided ROI detection (if pose landmarks available)
        if use_pose_guided and pose_landmarks is not None:
            landmarks = pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Define keypoints of interest with ROI sizes
            keypoints_of_interest = [
                # Hands (for phone, book, pen, pencil) - INCREASED SIZE
                ('RIGHT_WRIST', self.mp_pose.PoseLandmark.RIGHT_WRIST, 250),  # Increased from 180
                ('LEFT_WRIST', self.mp_pose.PoseLandmark.LEFT_WRIST, 250),    # Increased from 180
                ('RIGHT_INDEX', self.mp_pose.PoseLandmark.RIGHT_INDEX, 200),  # Increased from 150
                ('LEFT_INDEX', self.mp_pose.PoseLandmark.LEFT_INDEX, 200),    # Increased from 150
                
                # Lap/Torso area (for books, reading, writing on lap)
                ('RIGHT_HIP', self.mp_pose.PoseLandmark.RIGHT_HIP, 280),
                ('LEFT_HIP', self.mp_pose.PoseLandmark.LEFT_HIP, 280),
                
                # Ears (for phone calls)
                ('RIGHT_EAR', self.mp_pose.PoseLandmark.RIGHT_EAR, 120),
                ('LEFT_EAR', self.mp_pose.PoseLandmark.LEFT_EAR, 120),
                
                # Mouth (for eating, drinking, phone)
                ('MOUTH_LEFT', self.mp_pose.PoseLandmark.MOUTH_LEFT, 100),
                ('MOUTH_RIGHT', self.mp_pose.PoseLandmark.MOUTH_RIGHT, 100),
            ]
            
            # Create ROIs and run focused detection
            for keypoint_name, keypoint_idx, roi_size in keypoints_of_interest:
                try:
                    landmark = landmarks[keypoint_idx]
                    
                    # Check visibility
                    if landmark.visibility < 0.5:
                        continue
                    
                    keypoint_coords = (int(landmark.x * w), int(landmark.y * h))
                    roi_bbox = self.get_roi_around_keypoint(keypoint_coords, frame.shape, roi_size)
                    
                    if roi_bbox is not None:
                        detections['roi_boxes'].append((keypoint_name, roi_bbox))
                        
                        # Detect objects in ROI
                        roi_detections = self.detect_objects_in_roi(
                            frame, roi_bbox, 
                            target_classes=['cell phone', 'book', 'pen', 'pencil', 'paper', 'bottle', 'cup']
                        )
                        
                        for det in roi_detections:
                            class_name, conf, x1, y1, x2, y2 = det
                            detections['roi_detections'].append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'keypoint': keypoint_name,
                                'source': 'pose_guided_roi'
                            })
                            
                            # Also add to main detection lists
                            if class_name == 'cell phone':
                                detections['cell_phone'].append([x1, y1, x2, y2])
                            elif class_name == 'book':
                                detections['book'].append([x1, y1, x2, y2])
                
                except Exception as e:
                    continue
        
        return detections
    
    def draw_bounding_boxes(self, frame, detections, show_roi_boxes=True, person_roles=None):
        """Draw bounding boxes on frame for detected objects and ROI regions.
        
        Args:
            frame: Input frame
            detections: Dictionary with detection results
            show_roi_boxes: Whether to show ROI boxes (default True)
            person_roles: Dictionary of person roles (optional)
        """
        annotated_frame = frame.copy()
        
        colors = {
            'person': (0, 255, 0),
            'cell_phone': (0, 0, 255),
            'book': (255, 0, 0),
            'backpack': (0, 255, 255),
            'deduplicated_person': (0, 255, 0),  # Green for deduplicated persons
            'LP': (0, 255, 255),  # Yellow for Loco Pilot
            'ALP': (255, 165, 0),  # Orange for Assistant Loco Pilot
            'SUPERVISOR': (128, 0, 128),  # Purple for Supervisor
            'TRAINEE': (0, 255, 255),  # Cyan for Trainee
            'VISITOR': (128, 128, 128)  # Gray for Visitor
        }
        
        # Draw ROI boxes (semi-transparent cyan boxes)
        if show_roi_boxes and 'roi_boxes' in detections:
            for keypoint_name, roi_bbox in detections['roi_boxes']:
                x1, y1, x2, y2 = roi_bbox
                # Draw semi-transparent ROI box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                
                # Add keypoint label
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
                
                # Use magenta color for pose-guided detections
                color = (255, 0, 255)
                thickness = 3  # Thicker border to distinguish from regular detections
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add label with confidence and keypoint
                label = f"{roi_det['class']} {roi_det['confidence']:.2f} (ROI: {roi_det['keypoint']})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_w, label_h = label_size
                
                # Background for label
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
            if obj_type in ['roi_detections', 'roi_boxes', 'deduplicated_person']:
                continue
            
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
        
        # Draw deduplicated person boxes (with thicker border and role labels)
        if 'deduplicated_person' in detections and len(detections['deduplicated_person']) > 0:
            person_count = len(detections['deduplicated_person'])
            for idx, bbox in enumerate(detections['deduplicated_person']):
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get role information if available
                if person_roles and idx in person_roles:
                    role_info = person_roles[idx]
                    role = role_info['role']
                    role_name = role_info['role_name']
                    lp_score = role_info['lp_score']
                    alp_score = role_info['alp_score']
                    
                    # Use role-specific color
                    box_color = colors.get(role, (0, 255, 0))
                    
                    # Create detailed label
                    label = f"{role_name} (LP:{lp_score}/ALP:{alp_score})"
                else:
                    # Default label if no role info
                    box_color = (0, 255, 0)
                    label = f"Person {idx+1}"
                
                # Thicker border for deduplicated persons
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_w, label_h = label_size
                
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_h - 10), 
                            (x1 + label_w + 10, y1), 
                            box_color, -1)
                
                cv2.putText(annotated_frame, label, 
                           (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
            
            # Add person count overlay at top
            if person_count > 2:
                count_text = f"GROUP DETECTED: {person_count} PEOPLE"
                count_color = (0, 0, 255)  # Red for group alert
            else:
                count_text = f"People Count: {person_count}"
                count_color = (0, 255, 0)
            
            cv2.putText(annotated_frame, count_text, 
                       (frame.shape[1] - 400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, count_color, 2, cv2.LINE_AA)
            
            # Add role summary if available
            if person_roles:
                y_offset = 60
                for idx in sorted(person_roles.keys()):
                    role_info = person_roles[idx]
                    role_text = f"{role_info['role_name']}: LP={role_info['lp_score']}, ALP={role_info['alp_score']}"
                    role_color = colors.get(role_info['role'], (255, 255, 255))
                    cv2.putText(annotated_frame, role_text, 
                               (frame.shape[1] - 400, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, role_color, 2, cv2.LINE_AA)
                    y_offset += 25
        
        return annotated_frame
    
    def draw_mediapipe_outputs(self, frame, pose_results, face_results, ear_value=None, eye_closure_duration=0, pose_sleep_info=None, head_pose_info=None, multi_person_results=None):
        """Draw MediaPipe pose and face mesh landmarks on frame
        
        Args:
            frame: Frame to draw on
            pose_results: Single-person pose results (for fallback/legacy)
            face_results: Single-person face results (for fallback/legacy)
            ear_value: EAR value (for single-person mode)
            eye_closure_duration: Eye closure duration (for single-person mode)
            pose_sleep_info: Sleep info dict (for single-person mode)
            head_pose_info: Head pose info dict (for single-person mode)
            multi_person_results: Multi-person detection results from detect_per_person_activities()
        """
        annotated_frame = frame.copy()
        
        # NEW: Draw multi-person pose landmarks if available
        if multi_person_results:
            for person_idx, person_result in multi_person_results.items():
                # Draw pose landmarks for this person
                if person_result['pose_landmarks']:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        person_result['pose_landmarks'],
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Draw face landmarks for this person
                if person_result['face_landmarks']:
                    self.mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=person_result['face_landmarks'],
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    self.mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=person_result['face_landmarks'],
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
            
            # For multi-person, we'll show aggregated info or first person's info
            # Use the first person's metrics for display (or we can enhance this later)
            if len(multi_person_results) > 0:
                first_person = list(multi_person_results.values())[0]
                ear_value = first_person['metrics'].get('ear')
                eye_closure_duration = first_person['metrics'].get('eye_closure_duration', 0)
                pose_sleep_info = first_person['metrics'].get('pose_sleep_info')
                head_pose_info = first_person['metrics'].get('head_pose_info')
                face_detected = first_person['face_landmarks'] is not None
        else:
            # FALLBACK: Single-person mode (legacy)
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
                alert_text = "⚠️ MIND DIVERSION - ATTENTION DIVERTED!"
                cv2.putText(annotated_frame, alert_text, (10, y_offset + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Show detection method
                method_text = f"(Method: {method})"
                cv2.putText(annotated_frame, method_text, (10, y_offset + 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
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
    
    def detect_pose_per_person(self, frame, person_roles):
        """Run MediaPipe Pose detection on each person's cropped bounding box.
        
        This enables multi-person pose detection by running single-person MediaPipe Pose
        on each detected person's region separately.
        
        Args:
            frame: Full video frame (BGR format)
            person_roles: Dictionary of person roles with bounding boxes
                         Format: {person_idx: {'bbox': [x1, y1, x2, y2], 'role': 'LP'/'ALP', ...}}
        
        Returns:
            Dict[int, pose_landmarks]: Dictionary mapping person_idx to their pose landmarks
                                      Returns None for persons where pose detection failed
        """
        if not person_roles:
            return {}
        
        h, w = frame.shape[:2]
        person_poses = {}
        
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                person_poses[person_idx] = None
                continue
            
            bbox = person_data['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # Check if bbox is valid
            if x2 <= x1 or y2 <= y1:
                person_poses[person_idx] = None
                continue
            
            # Crop frame to person's bounding box
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Check if crop is valid
            if cropped_frame.size == 0:
                person_poses[person_idx] = None
                continue
            
            # Convert to RGB for MediaPipe
            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Run MediaPipe Pose on cropped region
            try:
                pose_result = self.pose.process(cropped_rgb)
                
                if pose_result.pose_landmarks:
                    # Translate landmarks from cropped coordinates to full frame coordinates
                    translated_landmarks = self.translate_pose_landmarks(
                        pose_result.pose_landmarks,
                        offset_x=x1,
                        offset_y=y1,
                        crop_width=x2-x1,
                        crop_height=y2-y1,
                        frame_width=w,
                        frame_height=h
                    )
                    person_poses[person_idx] = translated_landmarks
                else:
                    person_poses[person_idx] = None
            except Exception as e:
                print(f"Error processing pose for person {person_idx}: {e}")
                person_poses[person_idx] = None
        
        return person_poses
    
    def translate_pose_landmarks(self, pose_landmarks, offset_x, offset_y, crop_width, crop_height, frame_width, frame_height):
        """Translate pose landmarks from cropped coordinates back to full frame coordinates.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (normalized to crop size)
            offset_x: X offset of crop in full frame
            offset_y: Y offset of crop in full frame
            crop_width: Width of cropped region
            crop_height: Height of cropped region
            frame_width: Full frame width
            frame_height: Full frame height
        
        Returns:
            Translated pose landmarks in full frame coordinates
        """
        import copy
        from mediapipe.framework.formats import landmark_pb2
        
        # Create a deep copy of landmarks
        translated = landmark_pb2.NormalizedLandmarkList()
        
        for landmark in pose_landmarks.landmark:
            new_landmark = translated.landmark.add()
            
            # Convert normalized crop coordinates to pixel coordinates
            pixel_x_in_crop = landmark.x * crop_width
            pixel_y_in_crop = landmark.y * crop_height
            
            # Translate to full frame pixel coordinates
            pixel_x_in_frame = pixel_x_in_crop + offset_x
            pixel_y_in_frame = pixel_y_in_crop + offset_y
            
            # Normalize to full frame dimensions
            new_landmark.x = pixel_x_in_frame / frame_width
            new_landmark.y = pixel_y_in_frame / frame_height
            new_landmark.z = landmark.z  # Z coordinate doesn't need translation
            new_landmark.visibility = landmark.visibility
            
            # Copy presence if it exists
            if hasattr(landmark, 'presence'):
                new_landmark.presence = landmark.presence
        
        return translated
    
    def calculate_gesture_confidence(self, criteria_results, hand_side='right'):
        """Calculate 0-100% confidence score based on which criteria passed/failed.
        
        Args:
            criteria_results: Dictionary with results for each criterion
            hand_side: 'right' or 'left'
            
        Returns:
            tuple: (confidence_score, passed_criteria, failed_criteria, rejection_reasons)
        """
        # Weight each criterion (total should sum to 100)
        # v3.0 ENHANCED with geometric criteria
        weights = {
            'wrist_in_expanded_bbox': 10,      # Person matching
            'not_in_control_zone': 25,         # CRITICAL: Most important for false positive prevention
            'wrist_above_shoulder': 10,        # Core gesture criteria
            'wrist_above_elbow': 10,           # Vertical extension
            'arm_extended': 8,                 # Lateral extension
            'elbow_position': 7,               # Arm angle
            'visibility': 8,                   # Landmark quality
            'in_frame_bounds': 5,              # Basic sanity check
            # v3.0 NEW CRITERIA
            'vertical_arm': 12,                # IMPORTANT: Arm verticality (not angled forward)
            'hand_at_head_level': 5            # Hand is at head level (bonus check)
        }
        
        confidence = 0
        passed_criteria = []
        failed_criteria = []
        rejection_reasons = []
        
        for criterion, passed in criteria_results.items():
            if passed:
                confidence += weights.get(criterion, 0)
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)
                
                # Map criteria to human-readable rejection reasons
                reason_map = {
                    'wrist_in_expanded_bbox': 'wrist_not_in_bbox',
                    'not_in_control_zone': 'control_zone',
                    'wrist_above_shoulder': 'insufficient_height',
                    'wrist_above_elbow': 'elbow_not_below_wrist',
                    'arm_extended': 'insufficient_extension',
                    'elbow_position': 'elbow_position',
                    'visibility': 'visibility_low',
                    'in_frame_bounds': 'out_of_bounds',
                    # v3.0 NEW REJECTION REASONS
                    'vertical_arm': 'arm_not_vertical',
                    'hand_at_head_level': 'hand_below_head'
                }
                
                reason = reason_map.get(criterion, criterion)
                rejection_reasons.append(reason)
        
        return confidence, passed_criteria, failed_criteria, rejection_reasons
    
    def detect_hand_gesture(self, pose_landmarks, frame_shape, person_roles, yolo_person_boxes=None):
        """Detect hand gesture (raised hand) for LP/ALP hand exchange signal.
        
        CRITICAL: This function ensures pose landmarks belong to the SAME person
        we're analyzing by matching pose to YOLO person bounding boxes.
        
        The gesture should be detected only when ONE person is doing it (not both).
        
        ROBUST FALSE POSITIVE PREVENTION:
        - Filters out hands reaching toward/operating control panels
        - Only detects deliberate hand-raising gestures (signaling)
        - Uses control panel proximity, forward reach detection, and arm geometry
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (tracks 1 person)
            frame_shape: (height, width) of the frame
            person_roles: Dictionary of person roles from identify_person_roles()
                         Format: {person_idx: {'bbox': [x1, y1, x2, y2], 'role': 'LP'/'ALP', ...}}
            yolo_person_boxes: List of YOLO person bounding boxes (for validation)
            
        Returns:
            tuple: (lp_gesture_detected, alp_gesture_detected, debug_info)
                   Returns (False, False, {}) if both are gesturing or no one is
        """
        if not pose_landmarks or not person_roles:
            return False, False, {}
        
        h, w = frame_shape[:2]
        landmarks = pose_landmarks.landmark
        
        # Get key body landmarks
        try:
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        except (IndexError, AttributeError):
            return False, False, {}
        
        # Convert to pixel coordinates
        right_wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_elbow_coords = (int(right_elbow.x * w), int(right_elbow.y * h))
        left_elbow_coords = (int(left_elbow.x * w), int(left_elbow.y * h))
        right_hip_coords = (int(right_hip.x * w), int(right_hip.y * h))
        left_hip_coords = (int(left_hip.x * w), int(left_hip.y * h))
        nose_coords = (int(nose.x * w), int(nose.y * h))
        avg_shoulder_y = (right_shoulder_coords[1] + left_shoulder_coords[1]) / 2
        
        # Calculate body centerline (for detecting forward vs upward reach)
        body_center_x = (right_shoulder_coords[0] + left_shoulder_coords[0]) / 2
        
        # CRITICAL: Match MediaPipe Pose to the correct YOLO person bounding box
        # MediaPipe Pose tracks only 1 person. We need to determine which person's box
        # this pose belongs to, to avoid mixing one person's body with another's hands.
        
        # Strategy: Check which person's bounding box contains the pose landmarks
        # We'll use nose/shoulders as the key landmarks to match
        
        matched_person_idx = None
        matched_role = None
        
        # Calculate pose center point (using shoulders and nose)
        pose_center_x = (right_shoulder_coords[0] + left_shoulder_coords[0] + nose_coords[0]) / 3
        pose_center_y = (right_shoulder_coords[1] + left_shoulder_coords[1] + nose_coords[1]) / 3
        
        # Match pose to person bounding box
        best_overlap_score = 0
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue
            
            bbox = person_data['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Check if key pose landmarks fall within this person's bounding box
            # Count how many key landmarks are inside
            landmarks_inside = 0
            total_landmarks = 0
            
            key_points = [
                nose_coords,
                right_shoulder_coords,
                left_shoulder_coords,
                right_elbow_coords,
                left_elbow_coords,
                right_wrist_coords,
                left_wrist_coords,
                right_hip_coords,
                left_hip_coords
            ]
            
            for point in key_points:
                px, py = point
                total_landmarks += 1
                if x1 <= px <= x2 and y1 <= py <= y2:
                    landmarks_inside += 1
            
            overlap_score = landmarks_inside / total_landmarks if total_landmarks > 0 else 0
            
            # Require at least 50% of key landmarks to be within the box
            if overlap_score > best_overlap_score and overlap_score >= 0.5:
                best_overlap_score = overlap_score
                matched_person_idx = person_idx
                matched_role = person_data.get('role', 'UNKNOWN')
        
        # If we can't match pose to any person box with high confidence, reject detection
        if matched_person_idx is None or matched_role is None:
            return False, False, {
                'error': 'pose_not_matched_to_person',
                'pose_center': (pose_center_x, pose_center_y),
                'best_overlap': best_overlap_score
            }
        
        # Additional validation: Check if wrist landmarks are plausibly within or near the matched person's box
        # Allow some margin for extended arms
        matched_bbox = person_roles[matched_person_idx]['bbox']
        mx1, my1, mx2, my2 = matched_bbox
        
        # Expand box by 30% for arm extension tolerance
        box_width = mx2 - mx1
        box_height = my2 - my1
        margin_x = box_width * 0.3
        margin_y = box_height * 0.3
        
        expanded_x1 = mx1 - margin_x
        expanded_y1 = my1 - margin_y
        expanded_x2 = mx2 + margin_x
        expanded_y2 = my2 + margin_y
        
        # Check if wrists are within expanded box (for extended arms)
        right_wrist_in_expanded = (expanded_x1 <= right_wrist_coords[0] <= expanded_x2 and 
                                   expanded_y1 <= right_wrist_coords[1] <= expanded_y2)
        left_wrist_in_expanded = (expanded_x1 <= left_wrist_coords[0] <= expanded_x2 and 
                                  expanded_y1 <= left_wrist_coords[1] <= expanded_y2)
        
        # ==================================================================================
        # ROBUST HAND GESTURE DETECTION LOGIC
        # ==================================================================================
        # Detect when LP/ALP raises their hand in a signaling gesture (extended arm with raised hand)
        # This is the typical hand gesture used for communication signals between crew members
        #
        # FALSE POSITIVE PREVENTION:
        # - Filters out hands reaching toward/operating control panels
        # - Only detects deliberate hand-raising gestures (signaling)
        # - Distinguishes between:
        #   * FORWARD REACH (operating controls) → FALSE
        #   * UPWARD RAISE (signaling) → TRUE
        #
        # Key Detection Criteria:
        # 1. Hand raised significantly above shoulder (minimum 120px above shoulder)
        # 2. Hand is NOT in front of body (control panel region) - must be to the side or above
        # 3. Elbow is below wrist (vertical arm extension, not forward reach)
        # 4. Hand is laterally away from body centerline (not reaching forward)
        # 5. Wrist must be within expanded bounding box of the SAME person (critical for multi-person)
        # 6. Good visibility of landmarks
        # ==================================================================================
        
        # Calculate arm extension (how far hand is from shoulder horizontally)
        right_arm_extension = abs(right_wrist_coords[0] - right_shoulder_coords[0])
        left_arm_extension = abs(left_wrist_coords[0] - left_shoulder_coords[0])
        
        # Calculate vertical distance from wrist to elbow
        right_wrist_elbow_distance = right_elbow_coords[1] - right_wrist_coords[1]  # Positive if wrist above elbow
        left_wrist_elbow_distance = left_elbow_coords[1] - left_wrist_coords[1]
        
        # Calculate wrist to shoulder vertical distance
        right_wrist_shoulder_vertical = right_shoulder_coords[1] - right_wrist_coords[1]  # Positive if wrist above shoulder
        left_wrist_shoulder_vertical = left_shoulder_coords[1] - left_wrist_coords[1]
        
        # ==================================================================================
        # ROBUST CONTROL ZONE DETECTION - v3.0 ENHANCED
        # ==================================================================================
        # CRITICAL: Distinguish FORWARD REACH (controls) from UPWARD RAISE (signaling)
        #
        # FALSE POSITIVE PATTERN: Hand reaching FORWARD to overhead controls
        # - Wrist is "above shoulder" but hand is IN FRONT of body (not ABOVE head)
        # - Arm is angled forward, not vertical
        # - Hand is far from face/head
        #
        # TRUE SIGNAL PATTERN: Hand raised UPWARD for signaling
        # - Wrist is near or above head level
        # - Arm is nearly vertical (>75° angle)
        # - Hand is close to face/head (vertical plane)
        # ==================================================================================
        
        # Calculate additional geometric features for robust detection
        
        # 1. HAND-TO-NOSE DISTANCE (detects if hand is vertically above vs forward)
        right_hand_to_nose_dist = abs(right_wrist_coords[0] - nose_coords[0])
        left_hand_to_nose_dist = abs(left_wrist_coords[0] - nose_coords[0])
        
        # 2. HAND-TO-HEAD VERTICAL DISTANCE (true signals have hand at/above head)
        right_wrist_to_nose_vertical = nose_coords[1] - right_wrist_coords[1]  # Positive if wrist above nose
        left_wrist_to_nose_vertical = nose_coords[1] - left_wrist_coords[1]
        
        # 3. ARM ANGLE FROM VERTICAL (calculate how vertical the arm is)
        # Vertical arm (signaling): wrist directly above elbow
        # Forward arm (controls): wrist forward of elbow
        right_elbow_wrist_horizontal = abs(right_wrist_coords[0] - right_elbow_coords[0])
        left_elbow_wrist_horizontal = abs(left_wrist_coords[0] - left_elbow_coords[0])
        
        # Calculate "verticality score" - higher = more vertical (better for signaling)
        # True signal: wrist mostly above elbow (small horizontal distance)
        # Control reach: wrist significantly forward of elbow (large horizontal distance)
        right_arm_verticality = right_wrist_elbow_distance / max(1, right_elbow_wrist_horizontal)
        left_arm_verticality = left_wrist_elbow_distance / max(1, left_elbow_wrist_horizontal)
        
        # V4.0-B: Pre-calculate mandatory check results (MUST be defined early for debug logging)
        # These will be used in gesture detection logic and debug output
        # 
        # OPTION B: Use SHOULDER reference instead of NOSE (robust against head tilt)
        # Problem: When person looks down at controls, nose lowers → makes wrist_to_nose unreliable
        # Solution: Use shoulder as reference (shoulders don't move with head tilt)
        #
        # Calculate hand height relative to shoulder (positive = hand above shoulder)
        right_wrist_to_shoulder_height = right_shoulder_coords[1] - right_wrist_coords[1]
        left_wrist_to_shoulder_height = left_shoulder_coords[1] - left_wrist_coords[1]
        
        # Hand must be above shoulder (20px = lenient threshold)
        # Lowered from 30px to 20px to capture frame 10400 (per user requirement)
        # This accepts low-to-high reaches while rejecting shoulder-level or below
        right_hand_at_head = right_wrist_to_shoulder_height >= 20  # 20px above shoulder (was 30)
        left_hand_at_head = left_wrist_to_shoulder_height >= 20
        
        # LENIENT verticality: Accept arm going up-and-forward (not just purely vertical)
        # Threshold 1.5 = arm is at least 1.5x more vertical than horizontal
        right_arm_is_vertical = right_arm_verticality >= 1.5  # More lenient for high reaches
        left_arm_is_vertical = left_arm_verticality >= 1.5
        
        # Right hand: ENHANCED control operation zone detection
        right_in_control_zone = (
            # PRIMARY CHECK: Hand is NOT near head level (control operations are below head)
            (right_wrist_to_nose_vertical < -30) and  # Wrist is below nose level (not at head height)
            
            # SECONDARY CHECK: Hand is NOT vertically aligned with body (reaching forward)
            (right_hand_to_nose_dist > 80 or right_arm_verticality < 1.2) and  # Hand far from body centerline OR arm not vertical
            
            # ORIGINAL CHECKS: Hand in control panel region
            (right_wrist_coords[1] > (my1 + (my2 - my1) * 0.3)) and  # Not too high
            (right_wrist_coords[1] < (my1 + (my2 - my1) * 0.7)) and  # Not too low
            (right_arm_extension < self.gesture_thresholds['control_zone_extension']) and  # Not laterally extended
            (right_wrist_elbow_distance < self.gesture_thresholds['control_zone_elbow_distance']) and  # Elbow not far below wrist
            (right_wrist_shoulder_vertical < 100)  # Not far above shoulder
        )
        
        # Left hand: ENHANCED control operation zone detection
        left_in_control_zone = (
            # PRIMARY CHECK: Hand is NOT near head level
            (left_wrist_to_nose_vertical < -30) and
            
            # SECONDARY CHECK: Hand is NOT vertically aligned with body
            (left_hand_to_nose_dist > 80 or left_arm_verticality < 1.2) and
            
            # ORIGINAL CHECKS: Hand in control panel region
            (left_wrist_coords[1] > (my1 + (my2 - my1) * 0.3)) and
            (left_wrist_coords[1] < (my1 + (my2 - my1) * 0.7)) and
            (left_arm_extension < self.gesture_thresholds['control_zone_extension']) and
            (left_wrist_elbow_distance < self.gesture_thresholds['control_zone_elbow_distance']) and
            (left_wrist_shoulder_vertical < 100)
        )
        
        # ==================================================================================
        # CONFIDENCE SCORING AND DEBUG TRACKING
        # ==================================================================================
        # Track statistics
        self.gesture_stats['total_frames_analyzed'] += 1
        
        # Build criteria dictionaries for confidence calculation
        # ENHANCED with v3.0 geometric checks
        right_criteria = {
            'wrist_in_expanded_bbox': right_wrist_in_expanded,
            'not_in_control_zone': not right_in_control_zone,
            'wrist_above_shoulder': right_wrist_shoulder_vertical > self.gesture_thresholds['wrist_shoulder_vertical'],
            'wrist_above_elbow': right_wrist_elbow_distance > self.gesture_thresholds['wrist_elbow_distance'],
            'arm_extended': right_arm_extension > self.gesture_thresholds['arm_extension'],
            'elbow_position': (right_elbow_coords[1] >= right_shoulder_coords[1] + self.gesture_thresholds['elbow_shoulder_margin']),
            'visibility': (right_wrist.visibility > 0.5 and right_elbow.visibility > 0.4 and right_shoulder.visibility > 0.5),
            'in_frame_bounds': (0 < right_wrist_coords[0] < w and 0 < right_wrist_coords[1] < h),
            # v3.0 ENHANCED CRITERIA
            'vertical_arm': right_arm_verticality >= 1.2,  # Arm is vertical (not angled forward)
            'hand_at_head_level': right_wrist_to_nose_vertical >= -30  # Hand is at or above nose level
        }
        
        left_criteria = {
            'wrist_in_expanded_bbox': left_wrist_in_expanded,
            'not_in_control_zone': not left_in_control_zone,
            'wrist_above_shoulder': left_wrist_shoulder_vertical > self.gesture_thresholds['wrist_shoulder_vertical'],
            'wrist_above_elbow': left_wrist_elbow_distance > self.gesture_thresholds['wrist_elbow_distance'],
            'arm_extended': left_arm_extension > self.gesture_thresholds['arm_extension'],
            'elbow_position': (left_elbow_coords[1] >= left_shoulder_coords[1] + self.gesture_thresholds['elbow_shoulder_margin']),
            'visibility': (left_wrist.visibility > 0.5 and left_elbow.visibility > 0.4 and left_shoulder.visibility > 0.5),
            'in_frame_bounds': (0 < left_wrist_coords[0] < w and 0 < left_wrist_coords[1] < h),
            # v3.0 ENHANCED CRITERIA
            'vertical_arm': left_arm_verticality >= 1.2,
            'hand_at_head_level': left_wrist_to_nose_vertical >= -30
        }
        
        # Calculate confidence for both hands
        right_confidence, right_passed, right_failed, right_rejection_reasons = self.calculate_gesture_confidence(right_criteria, 'right')
        left_confidence, left_passed, left_failed, left_rejection_reasons = self.calculate_gesture_confidence(left_criteria, 'left')
        
        # Determine which hand has better confidence
        best_hand = 'right' if right_confidence >= left_confidence else 'left'
        best_confidence = max(right_confidence, left_confidence)
        best_rejection_reasons = right_rejection_reasons if best_hand == 'right' else left_rejection_reasons
        
        # Update confidence distribution statistics
        if best_confidence > 80:
            self.gesture_stats['confidence_distribution']['high'] += 1
        elif best_confidence >= 50:
            self.gesture_stats['confidence_distribution']['medium'] += 1
        else:
            self.gesture_stats['confidence_distribution']['low'] += 1
        
        # Debug logging if enabled
        if self.enable_gesture_debug:
            import logging
            logger = logging.getLogger(__name__)
            
            # Log hand measurements (v3.0 ENHANCED with geometric features)
            logger.debug(f"[GESTURE] Right hand - wrist_shoulder_vert: {right_wrist_shoulder_vertical:.1f}px (req: {self.gesture_thresholds['wrist_shoulder_vertical']}), "
                        f"wrist_elbow_dist: {right_wrist_elbow_distance:.1f}px (req: {self.gesture_thresholds['wrist_elbow_distance']}), "
                        f"arm_ext: {right_arm_extension:.1f}px (req: {self.gesture_thresholds['arm_extension']}), "
                        f"in_control_zone: {right_in_control_zone}, confidence: {right_confidence:.1f}%")
            
            logger.debug(f"[GESTURE v4.0-B SHOULDER-REF] Right hand - arm_verticality: {right_arm_verticality:.2f} (MUST BE ≥1.5), "
                        f"wrist_above_shoulder: {right_wrist_to_shoulder_height:.1f}px (MUST BE ≥20 [LENIENT]), "
                        f"hand_at_head: {right_hand_at_head}, arm_is_vertical: {right_arm_is_vertical}")
            
            logger.debug(f"[GESTURE] Left hand - wrist_shoulder_vert: {left_wrist_shoulder_vertical:.1f}px (req: {self.gesture_thresholds['wrist_shoulder_vertical']}), "
                        f"wrist_elbow_dist: {left_wrist_elbow_distance:.1f}px (req: {self.gesture_thresholds['wrist_elbow_distance']}), "
                        f"arm_ext: {left_arm_extension:.1f}px (req: {self.gesture_thresholds['arm_extension']}), "
                        f"in_control_zone: {left_in_control_zone}, confidence: {left_confidence:.1f}%")
            
            logger.debug(f"[GESTURE v4.0-B SHOULDER-REF] Left hand - arm_verticality: {left_arm_verticality:.2f} (MUST BE ≥1.5), "
                        f"wrist_above_shoulder: {left_wrist_to_shoulder_height:.1f}px (MUST BE ≥20 [LENIENT]), "
                        f"hand_at_head: {left_hand_at_head}, arm_is_vertical: {left_arm_is_vertical}")
            
            if right_in_control_zone or left_in_control_zone:
                logger.debug(f"[GESTURE] Hand in control zone detected - Right: {right_in_control_zone}, Left: {left_in_control_zone}")
            
            if best_rejection_reasons:
                logger.debug(f"[GESTURE] Rejection reasons ({best_hand} hand): {', '.join(best_rejection_reasons)}")
        
        # Track rejection reasons in statistics
        for reason in best_rejection_reasons:
            if reason in self.gesture_stats['rejections_by_reason']:
                self.gesture_stats['rejections_by_reason'][reason] += 1
        
        # ==================================================================================
        # V4.0 CRITICAL FIX: MANDATORY POSITIVE CRITERIA (WHITELIST APPROACH)
        # ==================================================================================
        # PROBLEM with v3.0: Control zone was EXCLUSION logic - if it didn't trigger, 
        #                    false positives still passed through
        #
        # SOLUTION: Add MANDATORY POSITIVE requirements that MUST pass for TRUE signals:
        #   1. Hand MUST be at HEAD level (not just shoulder)
        #   2. Arm MUST be VERTICAL (strict ratio ≥ 2.0)
        #   3. NOT in control zone (existing check)
        #
        # ALL THREE must pass. If ANY fails → REJECT immediately
        # ==================================================================================
        
        # V4.0 GESTURE DETECTION (WHITELIST APPROACH)
        # Note: Mandatory check variables (right_hand_at_head, right_arm_is_vertical, etc.)
        # are defined earlier (after geometric calculations) for use in debug logging
        right_hand_raised = (
            # CRITICAL PERSON MATCHING
            right_wrist_in_expanded and
            
            # ===== V4.0 MANDATORY POSITIVE CRITERIA (ALL MUST PASS) =====
            right_hand_at_head and              # MUST be at head level
            right_arm_is_vertical and           # MUST be vertical arm
            not right_in_control_zone and       # MUST NOT be in control zone
            # ============================================================
            
            # Traditional criteria (still required)
            right_wrist_shoulder_vertical > self.gesture_thresholds['wrist_shoulder_vertical'] and
            right_wrist_elbow_distance > self.gesture_thresholds['wrist_elbow_distance'] and
            right_arm_extension > self.gesture_thresholds['arm_extension'] and
            (right_elbow_coords[1] >= right_shoulder_coords[1] + self.gesture_thresholds['elbow_shoulder_margin']) and
            
            # Visibility
            right_wrist.visibility > 0.5 and
            right_elbow.visibility > 0.4 and
            right_shoulder.visibility > 0.5 and
            
            # Frame bounds
            0 < right_wrist_coords[0] < w and
            0 < right_wrist_coords[1] < h
        )
        
        left_hand_raised = (
            # CRITICAL PERSON MATCHING
            left_wrist_in_expanded and
            
            # ===== V4.0 MANDATORY POSITIVE CRITERIA (ALL MUST PASS) =====
            left_hand_at_head and               # MUST be at head level
            left_arm_is_vertical and            # MUST be vertical arm
            not left_in_control_zone and        # MUST NOT be in control zone
            # ============================================================
            
            # Traditional criteria
            left_wrist_shoulder_vertical > self.gesture_thresholds['wrist_shoulder_vertical'] and
            left_wrist_elbow_distance > self.gesture_thresholds['wrist_elbow_distance'] and
            left_arm_extension > self.gesture_thresholds['arm_extension'] and
            (left_elbow_coords[1] >= left_shoulder_coords[1] + self.gesture_thresholds['elbow_shoulder_margin']) and
            
            # Visibility
            left_wrist.visibility > 0.5 and
            left_elbow.visibility > 0.4 and
            left_shoulder.visibility > 0.5 and
            
            # Frame bounds
            0 < left_wrist_coords[0] < w and
            0 < left_wrist_coords[1] < h
        )
        
        hand_gesture_detected = right_hand_raised or left_hand_raised
        
        # Prepare debug dictionary with confidence data
        gesture_debug = {
            'hand_raised': best_hand if hand_gesture_detected else None,
            'right_confidence': right_confidence,
            'left_confidence': left_confidence,
            'right_rejection_reasons': right_rejection_reasons,
            'left_rejection_reasons': left_rejection_reasons,
            'shoulder_y': avg_shoulder_y,
            'wrist_y': right_wrist_coords[1] if (hand_gesture_detected and best_hand == 'right') else left_wrist_coords[1],
            'person_role': matched_role,
            'matched_person_idx': matched_person_idx,
            'overlap_score': best_overlap_score,
            'detection_version': 'v4.0-B-shoulder-20px',  # Mark with version for tracking
            'measurements': {
                'right': {
                    'wrist_shoulder_vertical': right_wrist_shoulder_vertical,
                    'wrist_elbow_distance': right_wrist_elbow_distance,
                    'arm_extension': right_arm_extension,
                    'in_control_zone': right_in_control_zone,
                    # v3.0 NEW MEASUREMENTS
                    'arm_verticality': right_arm_verticality,
                    'wrist_to_nose_vertical': right_wrist_to_nose_vertical,
                    'hand_to_nose_distance': right_hand_to_nose_dist,
                    # v4.0-B NEW: Shoulder-based height (robust against head tilt)
                    'wrist_to_shoulder_height': right_wrist_to_shoulder_height
                },
                'left': {
                    'wrist_shoulder_vertical': left_wrist_shoulder_vertical,
                    'wrist_elbow_distance': left_wrist_elbow_distance,
                    'arm_extension': left_arm_extension,
                    'in_control_zone': left_in_control_zone,
                    # v3.0 NEW MEASUREMENTS
                    'arm_verticality': left_arm_verticality,
                    'wrist_to_nose_vertical': left_wrist_to_nose_vertical,
                    'hand_to_nose_distance': left_hand_to_nose_dist,
                    # v4.0-B NEW: Shoulder-based height (robust against head tilt)
                    'wrist_to_shoulder_height': left_wrist_to_shoulder_height
                }
            }
        }
        
        if not hand_gesture_detected:
            # Add role-specific confidence to debug dict
            if matched_role == 'LP':
                gesture_debug['lp_confidence'] = best_confidence
                gesture_debug['lp_rejection_reasons'] = best_rejection_reasons
            elif matched_role == 'ALP':
                gesture_debug['alp_confidence'] = best_confidence
                gesture_debug['alp_rejection_reasons'] = best_rejection_reasons
            
            return False, False, gesture_debug
        
        # Update successful detection statistics
        self.gesture_stats['successful_detections']['total'] += 1
        
        # Return result based on the MATCHED person's role
        if matched_role == 'LP':
            self.gesture_stats['successful_detections']['lp'] += 1
            gesture_debug['lp_confidence'] = best_confidence
            gesture_debug['lp_rejection_reasons'] = []  # No rejections since detected
            
            if self.enable_gesture_debug:
                logger.debug(f"[GESTURE] ✓ LP hand gesture DETECTED - Confidence: {best_confidence:.1f}%, Hand: {best_hand}")
            
            return True, False, gesture_debug
        elif matched_role == 'ALP':
            self.gesture_stats['successful_detections']['alp'] += 1
            gesture_debug['alp_confidence'] = best_confidence
            gesture_debug['alp_rejection_reasons'] = []
            
            if self.enable_gesture_debug:
                logger.debug(f"[GESTURE] ✓ ALP hand gesture DETECTED - Confidence: {best_confidence:.1f}%, Hand: {best_hand}")
            
            return False, True, gesture_debug
        
        # Unknown role
        return False, False, gesture_debug
    
    def draw_hand_gesture_debug(self, frame, gesture_debug, lp_detected, alp_detected):
        """Draw confidence scores and rejection reasons on frame.
        
        Args:
            frame: Video frame (numpy array)
            gesture_debug: Debug information from detect_hand_gesture
            lp_detected: Whether LP gesture was detected
            alp_detected: Whether ALP gesture was detected
            
        Returns:
            frame: Frame with debug overlay
        """
        if not self.enable_gesture_debug or not gesture_debug:
            return frame
        
        h, w = frame.shape[:2]
        y_offset = 10
        
        # Draw LP confidence
        if 'lp_confidence' in gesture_debug:
            confidence = gesture_debug['lp_confidence']
            
            # Color coding: green (>80%), yellow (50-80%), red (<50%)
            if confidence > 80:
                color = (0, 255, 0)  # Green
            elif confidence >= 50:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            status = "DETECTED" if lp_detected else "REJECTED"
            text = f"LP Gesture: {confidence:.1f}% [{status}]"
            cv2.putText(frame, text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            y_offset += 30
            
            # Draw rejection reasons if any
            if 'lp_rejection_reasons' in gesture_debug and gesture_debug['lp_rejection_reasons']:
                reasons = gesture_debug['lp_rejection_reasons'][:3]  # Limit to top 3
                for reason in reasons:
                    reason_text = f"  - {reason.replace('_', ' ')}"
                    cv2.putText(frame, reason_text, (10, y_offset + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += 25
        
        # Draw ALP confidence
        if 'alp_confidence' in gesture_debug:
            confidence = gesture_debug['alp_confidence']
            
            if confidence > 80:
                color = (0, 255, 0)
            elif confidence >= 50:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            status = "DETECTED" if alp_detected else "REJECTED"
            text = f"ALP Gesture: {confidence:.1f}% [{status}]"
            cv2.putText(frame, text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            y_offset += 30
            
            if 'alp_rejection_reasons' in gesture_debug and gesture_debug['alp_rejection_reasons']:
                reasons = gesture_debug['alp_rejection_reasons'][:3]
                for reason in reasons:
                    reason_text = f"  - {reason.replace('_', ' ')}"
                    cv2.putText(frame, reason_text, (10, y_offset + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += 25
        
        return frame
    
    def generate_gesture_report(self):
        """Generate gesture analysis report and save to JSON.
        
        Creates a comprehensive statistics report including:
        - Detection rates
        - Rejection breakdown
        - Frame counts
        - Confidence distribution
        
        Returns:
            dict: Report data
        """
        if not self.run_dir:
            return None
        
        stats = self.gesture_stats
        
        # Calculate rates
        total = stats['total_frames_analyzed']
        if total == 0:
            return None
        
        successful = stats['successful_detections']['total']
        detection_rate = (successful / total * 100) if total > 0 else 0
        
        # Calculate rejection rate by reason
        total_rejections = sum(stats['rejections_by_reason'].values())
        rejection_breakdown = {}
        for reason, count in stats['rejections_by_reason'].items():
            percentage = (count / total_rejections * 100) if total_rejections > 0 else 0
            rejection_breakdown[reason] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # Build report
        report = {
            'generated_at': datetime.now().isoformat(),
            'sensitivity_mode': self.gesture_sensitivity,
            'debug_enabled': self.enable_gesture_debug,
            'summary': {
                'total_frames_analyzed': total,
                'successful_detections': successful,
                'detection_rate_percent': round(detection_rate, 2),
                'lp_detections': stats['successful_detections']['lp'],
                'alp_detections': stats['successful_detections']['alp']
            },
            'rejection_analysis': {
                'total_rejections': total_rejections,
                'breakdown_by_reason': rejection_breakdown
            },
            'confidence_distribution': {
                'high_confidence': {
                    'count': stats['confidence_distribution']['high'],
                    'percentage': round(stats['confidence_distribution']['high'] / total * 100, 2) if total > 0 else 0
                },
                'medium_confidence': {
                    'count': stats['confidence_distribution']['medium'],
                    'percentage': round(stats['confidence_distribution']['medium'] / total * 100, 2) if total > 0 else 0
                },
                'low_confidence': {
                    'count': stats['confidence_distribution']['low'],
                    'percentage': round(stats['confidence_distribution']['low'] / total * 100, 2) if total > 0 else 0
                }
            },
            'thresholds_used': self.gesture_thresholds
        }
        
        # Save to JSON
        report_path = os.path.join(self.run_dir, 'gesture_stats_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n{'='*80}")
            print(f"GESTURE ANALYSIS REPORT")
            print(f"{'='*80}")
            print(f"Total Frames Analyzed: {total}")
            print(f"Successful Detections: {successful} ({detection_rate:.1f}%)")
            print(f"  - LP Detections: {stats['successful_detections']['lp']}")
            print(f"  - ALP Detections: {stats['successful_detections']['alp']}")
            print(f"\nConfidence Distribution:")
            print(f"  - High (>80%): {stats['confidence_distribution']['high']} ({stats['confidence_distribution']['high']/total*100:.1f}%)")
            print(f"  - Medium (50-80%): {stats['confidence_distribution']['medium']} ({stats['confidence_distribution']['medium']/total*100:.1f}%)")
            print(f"  - Low (<50%): {stats['confidence_distribution']['low']} ({stats['confidence_distribution']['low']/total*100:.1f}%)")
            print(f"\nTop Rejection Reasons:")
            sorted_rejections = sorted(rejection_breakdown.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            for reason, data in sorted_rejections:
                if data['count'] > 0:
                    print(f"  - {reason}: {data['count']} ({data['percentage']:.1f}%)")
            print(f"\nReport saved to: {report_path}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error saving gesture report: {e}")
        
        return report
    
    def detect_multi_person_pose_and_gestures(self, frame, person_roles):
        """Run MediaPipe Pose on each person's cropped bounding box for multi-person gesture detection.
        
        This allows simultaneous detection of hand gestures from multiple people (LP and ALP)
        by running pose detection on cropped regions for each detected person.
        
        Args:
            frame: The full frame image (BGR format)
            person_roles: Dictionary of person roles from identify_person_roles()
                         Format: {person_idx: {'bbox': [x1, y1, x2, y2], 'role': 'LP'/'ALP', ...}}
        
        Returns:
            dict: Results for each person
                  Format: {person_idx: {'pose_landmarks': landmarks, 'gesture_detected': bool, 
                          'gesture_type': 'lp'/'alp', 'debug_info': {}}}
        """
        if not person_roles or len(person_roles) == 0:
            return {}
        
        results = {}
        h, w = frame.shape[:2]
        
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue
            
            bbox = person_data['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Add padding to bbox for better pose detection (10% on each side)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            # Expand bbox with padding, but stay within frame bounds
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(w, x2 + padding_x)
            y2_padded = min(h, y2 + padding_y)
            
            # Crop frame to this person's region
            try:
                cropped_frame = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                if cropped_frame.size == 0:
                    continue
                
                # Convert to RGB for MediaPipe
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                
                # Run MediaPipe Pose on this cropped region
                pose_result = self.pose.process(cropped_rgb)
                
                if pose_result.pose_landmarks:
                    # Translate landmarks from cropped coordinates back to full frame coordinates
                    translated_landmarks = self.translate_landmarks_to_full_frame(
                        pose_result.pose_landmarks,
                        x1_padded, y1_padded,
                        x2_padded - x1_padded, y2_padded - y1_padded,
                        w, h
                    )
                    
                    # Create a single-person roles dict for this person
                    single_person_roles = {person_idx: person_data}
                    
                    # Detect hand gesture for this specific person
                    lp_gesture, alp_gesture, gesture_debug = self.detect_hand_gesture(
                        translated_landmarks,
                        frame.shape,
                        single_person_roles
                    )
                    
                    # Store results
                    results[person_idx] = {
                        'pose_landmarks': translated_landmarks,
                        'gesture_detected': lp_gesture or alp_gesture,
                        'gesture_type': 'lp' if lp_gesture else ('alp' if alp_gesture else None),
                        'debug_info': gesture_debug,
                        'role': person_data.get('role', 'UNKNOWN')
                    }
                    
            except Exception as e:
                print(f"Error processing person {person_idx}: {e}")
                continue
        
        return results
    
    def detect_per_person_activities(self, frame, person_roles, timestamp_sec):
        """
        Comprehensive multi-person activity detection.
        
        Runs ALL activity detection logic on each person's cropped region:
        - MediaPipe Pose (for hand gestures, sleep detection, head pose)
        - MediaPipe Face Mesh (for EAR calculation, face-based sleep detection)
        - Sleep/microsleep detection
        - Hand gesture detection
        - Head pose/mind diversion detection
        
        Args:
            frame: The full frame image (BGR format)
            person_roles: Dictionary of person roles from identify_person_roles()
                         Format: {person_idx: {'bbox': [x1, y1, x2, y2], 'role': 'LP'/'ALP', ...}}
            timestamp_sec: Current timestamp in seconds
        
        Returns:
            dict: Comprehensive results for each person
                  Format: {
                      person_idx: {
                          'pose_landmarks': landmarks,
                          'face_landmarks': face_landmarks,
                          'role': 'LP'/'ALP',
                          'activities': {
                              'sleep': bool,
                              'microsleep': bool,
                              'hand_gesture': bool,
                              'gesture_type': 'lp'/'alp'/None,
                              'mind_diversion': bool
                          },
                          'metrics': {
                              'ear': float,
                              'eye_closure_duration': float,
                              'pose_sleep_info': dict,
                              'head_pose_info': dict,
                              'gesture_debug': dict
                          }
                      }
                  }
        """
        if not person_roles or len(person_roles) == 0:
            return {}
        
        results = {}
        h, w = frame.shape[:2]
        
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue
            
            bbox = person_data['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            role = person_data.get('role', 'UNKNOWN')
            
            # Add padding to bbox for better detection (10% on each side)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            # Expand bbox with padding, but stay within frame bounds
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(w, x2 + padding_x)
            y2_padded = min(h, y2 + padding_y)
            
            # Crop frame to this person's region
            try:
                cropped_frame = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                if cropped_frame.size == 0:
                    continue
                
                # Convert to RGB for MediaPipe
                cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                
                # Run MediaPipe Pose on this cropped region
                pose_result = self.pose.process(cropped_rgb)
                
                # Run MediaPipe Face Mesh on this cropped region
                face_result = self.face_mesh.process(cropped_rgb)
                
                # Initialize result structure for this person
                person_result = {
                    'pose_landmarks': None,
                    'face_landmarks': None,
                    'role': role,
                    'activities': {
                        'sleep': False,
                        'microsleep': False,
                        'hand_gesture': False,
                        'gesture_type': None,
                        'mind_diversion': False
                    },
                    'metrics': {
                        'ear': None,
                        'eye_closure_duration': 0,
                        'pose_sleep_info': {},
                        'head_pose_info': {},
                        'gesture_debug': {}
                    }
                }
                
                # Process pose landmarks
                if pose_result.pose_landmarks:
                    # Translate landmarks from cropped coordinates back to full frame coordinates
                    translated_pose_landmarks = self.translate_landmarks_to_full_frame(
                        pose_result.pose_landmarks,
                        x1_padded, y1_padded,
                        x2_padded - x1_padded, y2_padded - y1_padded,
                        w, h
                    )
                    person_result['pose_landmarks'] = translated_pose_landmarks
                    
                    # 1. Detect pose-based sleep/microsleep for this person
                    # Note: We need per-person sleep tracking, using person_idx as key
                    if not hasattr(self, 'per_person_sleep_tracking'):
                        self.per_person_sleep_tracking = {}
                    
                    if person_idx not in self.per_person_sleep_tracking:
                        self.per_person_sleep_tracking[person_idx] = {
                            'pose_sleep_start': None,
                            'pose_sleep_duration': 0,
                            'previous_landmarks': None
                        }
                    
                    # Detect pose-based sleep using cropped pose landmarks
                    pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.detect_pose_based_sleep_per_person(
                        translated_pose_landmarks,
                        timestamp_sec,
                        person_idx
                    )
                    
                    person_result['activities']['sleep'] = pose_sleep_detected
                    person_result['activities']['microsleep'] = pose_microsleep_detected
                    person_result['metrics']['pose_sleep_info'] = pose_sleep_info
                    
                    # 2. Detect hand gesture for this person
                    single_person_roles = {person_idx: person_data}
                    lp_gesture, alp_gesture, gesture_debug = self.detect_hand_gesture(
                        translated_pose_landmarks,
                        frame.shape,
                        single_person_roles
                    )
                    
                    person_result['activities']['hand_gesture'] = lp_gesture or alp_gesture
                    person_result['activities']['gesture_type'] = 'lp' if lp_gesture else ('alp' if alp_gesture else None)
                    person_result['metrics']['gesture_debug'] = gesture_debug
                    
                    # 3. Detect head pose/mind diversion for this person
                    head_pose_info = self.calculate_head_pose_angles(
                        translated_pose_landmarks,
                        face_result if face_result.multi_face_landmarks else None,
                        frame.shape
                    )
                    person_result['activities']['mind_diversion'] = head_pose_info.get('detected', False)
                    person_result['metrics']['head_pose_info'] = head_pose_info
                
                # Process face landmarks for EAR (eye aspect ratio) calculation
                if face_result.multi_face_landmarks:
                    # Get the first face (should be the only one in cropped region)
                    face_landmarks = face_result.multi_face_landmarks[0]
                    
                    # Translate face landmarks to full frame coordinates
                    translated_face_landmarks = self.translate_face_landmarks_to_full_frame(
                        face_landmarks,
                        x1_padded, y1_padded,
                        x2_padded - x1_padded, y2_padded - y1_padded
                    )
                    person_result['face_landmarks'] = translated_face_landmarks
                    
                    # Calculate EAR for this person
                    ear_value = self.calculate_eye_aspect_ratio(face_landmarks.landmark)
                    person_result['metrics']['ear'] = ear_value
                    
                    # Track eye closure duration per person
                    if not hasattr(self, 'per_person_eye_tracking'):
                        self.per_person_eye_tracking = {}
                    
                    if person_idx not in self.per_person_eye_tracking:
                        self.per_person_eye_tracking[person_idx] = {
                            'eye_closure_start': None,
                            'eye_closure_duration': 0
                        }
                    
                    # Check for eye closure (EAR < 0.2 = eyes closed)
                    if ear_value is not None and ear_value < 0.2:
                        tracking = self.per_person_eye_tracking[person_idx]
                        if tracking['eye_closure_start'] is None:
                            tracking['eye_closure_start'] = timestamp_sec
                        
                        tracking['eye_closure_duration'] = timestamp_sec - tracking['eye_closure_start']
                        person_result['metrics']['eye_closure_duration'] = tracking['eye_closure_duration']
                        
                        # Override pose-based detection with face-based detection (more accurate)
                        if tracking['eye_closure_duration'] >= 30:
                            person_result['activities']['sleep'] = True
                            person_result['activities']['microsleep'] = False
                        elif tracking['eye_closure_duration'] >= 5:
                            person_result['activities']['microsleep'] = True
                    else:
                        # Eyes open, reset tracking
                        self.per_person_eye_tracking[person_idx]['eye_closure_start'] = None
                        self.per_person_eye_tracking[person_idx]['eye_closure_duration'] = 0
                
                # Store results for this person
                results[person_idx] = person_result
                
            except Exception as e:
                print(f"Error processing person {person_idx} ({role}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def detect_pose_based_sleep_per_person(self, pose_landmarks, timestamp_sec, person_idx):
        """Detect sleep based on pose analysis for a specific person.
        
        Similar to detect_pose_based_sleep but tracks each person separately.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks for this person
            timestamp_sec: Current timestamp in seconds
            person_idx: Index of the person being tracked
        
        Returns:
            tuple: (sleep_detected, microsleep_detected, sleep_info_dict)
        """
        if not pose_landmarks:
            return False, False, {}
        
        tracking = self.per_person_sleep_tracking[person_idx]
        
        landmarks = pose_landmarks.landmark
        h, w = 480, 640  # Use normalized coordinates
        
        # Calculate head tilt (similar to main detect_pose_based_sleep)
        try:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            nose_y = nose.y
            shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2
            
            vertical_diff = nose_y - shoulder_midpoint_y
            head_tilt = math.degrees(math.asin(max(-1, min(1, vertical_diff))))
            
        except:
            head_tilt = None
        
        # Calculate movement (compare with previous frame)
        movement = 0.0
        if tracking['previous_landmarks'] is not None:
            prev_landmarks = tracking['previous_landmarks']
            total_diff = 0.0
            valid_landmarks = 0
            
            for i in range(len(landmarks)):
                if landmarks[i].visibility > 0.5 and prev_landmarks[i].visibility > 0.5:
                    dx = landmarks[i].x - prev_landmarks[i].x
                    dy = landmarks[i].y - prev_landmarks[i].y
                    total_diff += math.sqrt(dx*dx + dy*dy)
                    valid_landmarks += 1
            
            if valid_landmarks > 0:
                movement = total_diff / valid_landmarks
        
        # Store current landmarks for next comparison
        tracking['previous_landmarks'] = landmarks
        
        # Detect sleep indicators
        sleep_indicators = 0
        
        # 1. Head tilted forward/down
        if head_tilt is not None and head_tilt < -15:
            sleep_indicators += 1
        
        # 2. Minimal movement
        if movement < 0.02:
            sleep_indicators += 1
        
        # Track sleep duration
        if sleep_indicators >= 2:
            if tracking['pose_sleep_start'] is None:
                tracking['pose_sleep_start'] = timestamp_sec
            
            tracking['pose_sleep_duration'] = timestamp_sec - tracking['pose_sleep_start']
        else:
            tracking['pose_sleep_start'] = None
            tracking['pose_sleep_duration'] = 0
        
        # Determine if sleep/microsleep detected
        sleep_detected = tracking['pose_sleep_duration'] >= 30
        microsleep_detected = tracking['pose_sleep_duration'] >= 5 and not sleep_detected
        
        sleep_info = {
            'head_tilt': head_tilt,
            'avg_movement': movement,
            'pose_sleep_duration': tracking['pose_sleep_duration'],
            'sleep_indicators': sleep_indicators
        }
        
        return sleep_detected, microsleep_detected, sleep_info
    
    def translate_face_landmarks_to_full_frame(self, face_landmarks, crop_x, crop_y, crop_w, crop_h):
        """Translate face landmarks from cropped coordinates to full frame coordinates.
        
        Args:
            face_landmarks: MediaPipe face landmarks from cropped region
            crop_x, crop_y: Top-left corner of crop in full frame
            crop_w, crop_h: Width and height of crop region
        
        Returns:
            Translated face landmarks object
        """
        # Create a copy of the landmarks
        from mediapipe.framework.formats import landmark_pb2
        
        translated_landmarks = landmark_pb2.NormalizedLandmarkList()
        
        for landmark in face_landmarks.landmark:
            new_landmark = translated_landmarks.landmark.add()
            # Convert from cropped normalized coordinates to full frame normalized coordinates
            new_landmark.x = (landmark.x * crop_w + crop_x) / 1.0  # Will be normalized by frame width later
            new_landmark.y = (landmark.y * crop_h + crop_y) / 1.0  # Will be normalized by frame height later
            new_landmark.z = landmark.z
            new_landmark.visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        
        return translated_landmarks
    
    def calculate_head_pose_angles(self, pose_landmarks, face_landmarks, frame_shape):
        """Calculate head pose angles (yaw and pitch) to detect mind diversion.
        
        Detects when person turns face to side AND looks down.
        
        Uses both pose landmarks (nose, shoulders) and face mesh landmarks for accuracy.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            face_landmarks: MediaPipe face mesh landmarks (can be None)
            frame_shape: (height, width) of frame
            
        Returns:
            dict: {
                'yaw': float,      # Side turn angle in degrees (-90 to +90)
                'pitch': float,    # Up/down tilt angle in degrees (-90 to +90)
                'detected': bool,  # True if mind diversion detected
                'method': str      # Detection method used
            }
        """
        h, w = frame_shape[:2]
        result = {'yaw': 0, 'pitch': 0, 'detected': False, 'method': 'none'}
        
        if not pose_landmarks:
            return result
        
        landmarks = pose_landmarks.landmark
        
        try:
            # Get pose landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Check visibility
            if nose.visibility < 0.5:
                return result
            
            # Convert to pixel coordinates
            nose_coords = np.array([nose.x * w, nose.y * h])
            left_shoulder_coords = np.array([left_shoulder.x * w, left_shoulder.y * h])
            right_shoulder_coords = np.array([right_shoulder.x * w, right_shoulder.y * h])
            left_ear_coords = np.array([left_ear.x * w, left_ear.y * h])
            right_ear_coords = np.array([right_ear.x * w, right_ear.y * h])
            
            # Calculate shoulder midpoint
            shoulder_midpoint = (left_shoulder_coords + right_shoulder_coords) / 2
            shoulder_width = np.linalg.norm(right_shoulder_coords - left_shoulder_coords)
            
            # METHOD 1: Calculate YAW (side turning) using nose offset from shoulder midpoint
            nose_offset_x = nose_coords[0] - shoulder_midpoint[0]
            
            # Normalize by shoulder width and convert to angle
            # Positive = turned right, Negative = turned left
            yaw_normalized = nose_offset_x / (shoulder_width / 2) if shoulder_width > 0 else 0
            yaw_angle = np.clip(yaw_normalized * 45, -90, 90)  # Scale to degrees
            
            # METHOD 2: Calculate PITCH (up/down tilt) using nose position relative to ears
            ear_midpoint = (left_ear_coords + right_ear_coords) / 2
            nose_offset_y = nose_coords[1] - ear_midpoint[1]
            
            # Normalize by head size (ear-to-nose distance) and convert to angle
            # Positive = looking down, Negative = looking up
            head_height = shoulder_midpoint[1] - ear_midpoint[1]
            if head_height > 0:
                pitch_normalized = nose_offset_y / head_height
                pitch_angle = np.clip(pitch_normalized * 30, -45, 45)  # Scale to degrees
            else:
                pitch_angle = 0
            
            result['yaw'] = yaw_angle
            result['pitch'] = pitch_angle
            result['method'] = 'pose_landmarks'
            
            # DETECTION LOGIC: Mind diversion detected if BOTH conditions met:
            # 1. Head turned to side > 45 degrees (either direction)
            # 2. Head looking down > 15 degrees
            yaw_threshold = 45  # degrees
            pitch_threshold = 15  # degrees (looking down is positive)
            
            if abs(yaw_angle) > yaw_threshold and pitch_angle > pitch_threshold:
                result['detected'] = True
            
            # Use face mesh if available for more accurate detection
            if face_landmarks and face_landmarks.multi_face_landmarks:
                try:
                    # Use first detected face
                    face_lm = face_landmarks.multi_face_landmarks[0].landmark
                    
                    # Key face mesh landmarks for 3D pose estimation
                    # Nose tip, chin, left/right face edges
                    nose_tip = face_lm[1]  # Nose tip
                    chin = face_lm[152]     # Chin
                    left_face_edge = face_lm[234]  # Left face edge
                    right_face_edge = face_lm[454]  # Right face edge
                    left_eye = face_lm[33]  # Left eye outer corner
                    right_eye = face_lm[263]  # Right eye outer corner
                    
                    # Convert to pixel coordinates
                    nose_tip_coords = np.array([nose_tip.x * w, nose_tip.y * h])
                    chin_coords = np.array([chin.x * w, chin.y * h])
                    left_edge_coords = np.array([left_face_edge.x * w, left_face_edge.y * h])
                    right_edge_coords = np.array([right_face_edge.x * w, right_face_edge.y * h])
                    left_eye_coords = np.array([left_eye.x * w, left_eye.y * h])
                    right_eye_coords = np.array([right_eye.x * w, right_eye.y * h])
                    
                    # Calculate face width and nose offset for YAW
                    face_width = np.linalg.norm(right_edge_coords - left_edge_coords)
                    face_center_x = (left_edge_coords[0] + right_edge_coords[0]) / 2
                    nose_offset_x_face = nose_tip_coords[0] - face_center_x
                    
                    # YAW angle from face mesh (more accurate)
                    if face_width > 0:
                        yaw_face = (nose_offset_x_face / (face_width / 2)) * 60  # Scale to degrees
                        result['yaw'] = np.clip(yaw_face, -90, 90)
                    
                    # Calculate PITCH using nose tip and eye line
                    eye_midpoint = (left_eye_coords + right_eye_coords) / 2
                    nose_to_eye_dist = np.linalg.norm(nose_tip_coords - eye_midpoint)
                    nose_below_eyes = nose_tip_coords[1] - eye_midpoint[1]
                    
                    # PITCH angle from face mesh
                    if nose_to_eye_dist > 0:
                        pitch_face = (nose_below_eyes / nose_to_eye_dist) * 45
                        result['pitch'] = np.clip(pitch_face, -45, 45)
                    
                    result['method'] = 'face_mesh'
                    
                    # Re-evaluate detection with face mesh data
                    if abs(result['yaw']) > yaw_threshold and result['pitch'] > pitch_threshold:
                        result['detected'] = True
                    else:
                        result['detected'] = False
                        
                except Exception as e:
                    # If face mesh processing fails, keep pose-based result
                    pass
            
            return result
            
        except (IndexError, AttributeError, ZeroDivisionError) as e:
            return {'yaw': 0, 'pitch': 0, 'detected': False, 'method': 'error'}
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2] first bounding box
            bbox2: [x1, y1, x2, y2] second bounding box
            
        Returns:
            float: IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def deduplicate_person_boxes(self, person_boxes, iou_threshold=0.3):
        """De-duplicate overlapping person bounding boxes using Non-Maximum Suppression.
        
        Args:
            person_boxes: List of person bounding boxes [x1, y1, x2, y2]
            iou_threshold: IoU threshold for considering boxes as duplicates (default 0.3)
            
        Returns:
            List of de-duplicated person boxes
        """
        if len(person_boxes) == 0:
            return []
        
        # Convert to list of lists if numpy arrays
        boxes = [list(box) if hasattr(box, 'tolist') else box for box in person_boxes]
        
        # Calculate areas for each box
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        
        # Sort by area (larger boxes first - usually more confident detections)
        sorted_indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        
        keep_boxes = []
        keep_indices = []
        
        while sorted_indices:
            # Take the first box (largest remaining)
            idx = sorted_indices[0]
            keep_boxes.append(boxes[idx])
            keep_indices.append(idx)
            sorted_indices.pop(0)
            
            # Remove boxes that significantly overlap with this box
            remaining_indices = []
            for other_idx in sorted_indices:
                iou = self.calculate_iou(boxes[idx], boxes[other_idx])
                if iou < iou_threshold:
                    # Keep this box (not a duplicate)
                    remaining_indices.append(other_idx)
                # else: discard as duplicate
            
            sorted_indices = remaining_indices
        
        return keep_boxes
    
    def identify_person_roles(self, frame, person_boxes, detections):
        """Identify LP (Loco Pilot) and ALP (Assistant Loco Pilot) based on objects near each person.
        
        Logic:
        - For each person, detect objects in front using YOLO
        - lp_score = monitors + keyboards + cell_phone + panel-like boxes
        - alp_score = book + empty_desk (approximated by lack of control objects)
        - LP = person with higher lp_score
        - ALP = the other person
        - Third person = "Supervisor", "Trainee", or "Visitor"
        
        Args:
            frame: Current video frame
            person_boxes: List of de-duplicated person bounding boxes [[x1, y1, x2, y2], ...]
            detections: Dictionary of all detected objects from YOLO
            
        Returns:
            Dictionary mapping person index to role info: {
                0: {'role': 'LP', 'lp_score': 5, 'alp_score': 1, 'bbox': [x1, y1, x2, y2]},
                1: {'role': 'ALP', 'lp_score': 2, 'alp_score': 4, 'bbox': [x1, y1, x2, y2]},
                ...
            }
        """
        if len(person_boxes) == 0:
            return {}
        
        # Run full-frame YOLO detection to find control objects
        # Look for: tv/monitor, keyboard, mouse, laptop, book, backpack, cell phone
        yolo_results = self.yolo_model(frame, verbose=False, conf=0.3)
        
        # Collect all detected objects with their class names
        all_objects = []
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                class_name = self.yolo_model.names[cls]
                
                all_objects.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                })
        
        # For each person, calculate scores based on nearby objects
        person_scores = []
        
        for person_idx, person_bbox in enumerate(person_boxes):
            px1, py1, px2, py2 = person_bbox
            person_center_x = (px1 + px2) / 2
            person_width = px2 - px1
            person_height = py2 - py1
            
            # Define "in front of person" as region ahead of them
            # Assuming people face the camera/controls, "in front" is area below person's upper body
            # and within reasonable horizontal distance
            search_margin = person_width * 1.5  # Search 1.5x person width on each side
            search_x1 = person_center_x - search_margin
            search_x2 = person_center_x + search_margin
            search_y1 = py1 + (person_height * 0.3)  # Start from chest level
            search_y2 = py2 + (person_height * 0.5)  # Extend below person (desk/console area)
            
            # Count relevant objects in search region
            lp_objects = {
                'tv': 0,
                'laptop': 0, 
                'keyboard': 0,
                'mouse': 0,
                'cell phone': 0,
                'remote': 0  # Can act as control panel
            }
            
            alp_objects = {
                'book': 0,
                'notebook': 0,
                'backpack': 0
            }
            
            nearby_objects = []
            
            for obj in all_objects:
                obj_bbox = obj['bbox']
                ox1, oy1, ox2, oy2 = obj_bbox
                obj_center_x = (ox1 + ox2) / 2
                obj_center_y = (oy1 + oy2) / 2
                
                # Check if object is in the search region
                if (search_x1 <= obj_center_x <= search_x2 and 
                    search_y1 <= obj_center_y <= search_y2):
                    nearby_objects.append(obj)
                    
                    # Count LP-related objects
                    obj_class = obj['class']
                    if obj_class in lp_objects:
                        lp_objects[obj_class] += 1
                    
                    # Count ALP-related objects
                    if obj_class in alp_objects:
                        alp_objects[obj_class] += 1
            
            # Calculate scores
            lp_score = (
                lp_objects['tv'] * 3 +  # Monitors are strong indicators
                lp_objects['laptop'] * 2 +
                lp_objects['keyboard'] * 2 +
                lp_objects['mouse'] * 1 +
                lp_objects['cell phone'] * 1 +
                lp_objects['remote'] * 2  # Control panels/remotes
            )
            
            alp_score = (
                alp_objects['book'] * 3 +  # Books/logs are strong indicators
                alp_objects['notebook'] * 3 +
                alp_objects['backpack'] * 1
            )
            
            # If no LP objects detected, consider "empty desk" as ALP indicator
            if lp_score == 0 and alp_score == 0:
                alp_score = 1  # Slight preference for ALP if nothing detected
            
            person_scores.append({
                'person_idx': person_idx,
                'bbox': person_bbox,
                'lp_score': lp_score,
                'alp_score': alp_score,
                'lp_objects': lp_objects,
                'alp_objects': alp_objects,
                'nearby_objects': nearby_objects
            })
        
        # Assign roles based on scores
        person_roles = {}
        
        if len(person_scores) == 1:
            # Only one person - default to LP
            person_roles[0] = {
                'role': 'LP',
                'role_name': 'Loco Pilot',
                'lp_score': person_scores[0]['lp_score'],
                'alp_score': person_scores[0]['alp_score'],
                'bbox': person_scores[0]['bbox'],
                'objects': person_scores[0]['nearby_objects']
            }
        
        elif len(person_scores) == 2:
            # Two people - assign LP and ALP
            # Sort by lp_score (descending)
            sorted_persons = sorted(person_scores, key=lambda x: x['lp_score'], reverse=True)
            
            # Person with higher lp_score is LP
            person_roles[sorted_persons[0]['person_idx']] = {
                'role': 'LP',
                'role_name': 'Loco Pilot',
                'lp_score': sorted_persons[0]['lp_score'],
                'alp_score': sorted_persons[0]['alp_score'],
                'bbox': sorted_persons[0]['bbox'],
                'objects': sorted_persons[0]['nearby_objects']
            }
            
            # Other person is ALP
            person_roles[sorted_persons[1]['person_idx']] = {
                'role': 'ALP',
                'role_name': 'Assistant Loco Pilot',
                'lp_score': sorted_persons[1]['lp_score'],
                'alp_score': sorted_persons[1]['alp_score'],
                'bbox': sorted_persons[1]['bbox'],
                'objects': sorted_persons[1]['nearby_objects']
            }
        
        else:
            # Three or more people
            # Sort by lp_score (descending)
            sorted_persons = sorted(person_scores, key=lambda x: x['lp_score'], reverse=True)
            
            # First person is LP
            person_roles[sorted_persons[0]['person_idx']] = {
                'role': 'LP',
                'role_name': 'Loco Pilot',
                'lp_score': sorted_persons[0]['lp_score'],
                'alp_score': sorted_persons[0]['alp_score'],
                'bbox': sorted_persons[0]['bbox'],
                'objects': sorted_persons[0]['nearby_objects']
            }
            
            # Second person is ALP
            person_roles[sorted_persons[1]['person_idx']] = {
                'role': 'ALP',
                'role_name': 'Assistant Loco Pilot',
                'lp_score': sorted_persons[1]['lp_score'],
                'alp_score': sorted_persons[1]['alp_score'],
                'bbox': sorted_persons[1]['bbox'],
                'objects': sorted_persons[1]['nearby_objects']
            }
            
            # Additional people - assign contextual roles
            for i in range(2, len(sorted_persons)):
                person_idx = sorted_persons[i]['person_idx']
                
                # Determine role based on context
                # If they have books/backpacks, likely trainee
                # If they have control objects, likely supervisor
                # Otherwise, visitor
                if sorted_persons[i]['alp_score'] > 0:
                    role = 'TRAINEE'
                    role_name = 'Trainee'
                elif sorted_persons[i]['lp_score'] > 2:
                    role = 'SUPERVISOR'
                    role_name = 'Supervisor'
                else:
                    role = 'VISITOR'
                    role_name = 'Visitor'
                
                person_roles[person_idx] = {
                    'role': role,
                    'role_name': role_name,
                    'lp_score': sorted_persons[i]['lp_score'],
                    'alp_score': sorted_persons[i]['alp_score'],
                    'bbox': sorted_persons[i]['bbox'],
                    'objects': sorted_persons[i]['nearby_objects']
                }
        
        return person_roles
    
    def start_activity(self, activity_name, timestamp, fps, frame_count, person_roles=None):
        """Start tracking an activity
        
        Args:
            activity_name: Name of the activity
            timestamp: Timestamp when activity started
            fps: Frames per second
            frame_count: Frame count when activity started
            person_roles: Dictionary of person roles (optional)
        """
        if not self.activities[activity_name]['active']:
            self.activities[activity_name]['active'] = True
            self.activities[activity_name]['start_time'] = timestamp
            self.activities[activity_name]['start_frame_count'] = frame_count
            self.activities[activity_name]['last_frame_count'] = frame_count
            self.activities[activity_name]['frames'] = list(self.frame_buffer)
            self.activities[activity_name]['duration'] = 0
            self.activities[activity_name]['person_roles'] = person_roles if person_roles else {}
            print(f"[{timestamp}] Activity started: {activity_name}")
    
    def end_activity(self, activity_name, timestamp, fps, frame_count, people_count=1, save_clips=True):
        """End tracking an activity and optionally save evidence (only if meets minimum duration)"""
        if self.activities[activity_name]['active']:
            activity = self.activities[activity_name]
            activity['active'] = False
            
            start_frame = activity.get('start_frame_count', frame_count)
            
            # Calculate duration based on ACTUAL captured frames, not elapsed time
            # This ensures clip duration matches exactly the activity duration
            total_clip_frames = len(activity['frames'])
            actual_clip_duration = total_clip_frames / self.sample_fps  # Duration in seconds based on captured frames
            
            # Check if activity meets minimum duration threshold
            min_duration = self.activity_thresholds[activity_name]['min_duration']
            
            if actual_clip_duration < min_duration:
                print(f"[{timestamp}] Activity '{activity_name}' too short ({actual_clip_duration:.2f}s < {min_duration}s) - discarded")
                activity['frames'] = []
                activity['duration'] = 0
                self.consecutive_detections[activity_name] = 0
                self.grace_counters[activity_name] = 0
                return
            
            start_time_str = activity['start_time']
            
            # Parse activity start time in seconds
            def time_to_seconds(time_str):
                """Convert HH:MM:SS.microseconds to seconds"""
                parts = time_str.split(':')
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            
            activity_start_seconds = time_to_seconds(start_time_str)
            # Calculate end time based on actual clip duration, not elapsed time
            activity_end_seconds = activity_start_seconds + actual_clip_duration
            
            # Generate filenames with composite naming: {video}_{activity}_frame{number}_{counter}
            video_filename = os.path.basename(self.video_path)
            video_name_without_ext = os.path.splitext(video_filename)[0]
            
            clip_filename = f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}_{self.evidence_counter:03d}_clip.mp4"
            image_filename = f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}_{self.evidence_counter:03d}_activity.jpg"
            
            # Generate full paths for clips (even if not saving immediately)
            if self.evidence_clips_dir:
                clip_path = os.path.join(self.evidence_clips_dir, clip_filename)
                image_path = os.path.join(self.evidence_clips_dir, image_filename)
            else:
                # For multiprocessing workers without directories, use relative paths
                clip_path = clip_filename
                image_path = image_filename
            
            # Only save clips/images if save_clips is True and directories exist
            if save_clips and self.evidence_clips_dir:
                # Save video clip at sample FPS for full-duration playback
                # This creates clips with real-time duration instead of fast-motion
                # Example: 13 frames @ 0.5 FPS = 26 seconds (not 0.43 seconds @ 30 FPS)
                self.save_video_clip(activity['frames'], clip_path, self.sample_fps)
                
                # Save activity image (middle frame of the activity)
                if len(activity['frames']) > 0:
                    middle_frame_idx = len(activity['frames']) // 2
                    activity_image = activity['frames'][middle_frame_idx]
                    cv2.imwrite(image_path, activity_image)
            
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
            
            # Determine which crew member performed the activity
            # Default to LP crew info
            activity_crew_name = self.crew_name
            activity_crew_id = self.crew_id
            activity_crew_role = self.crew_role
            performing_role = 'LP'  # Default to LP
            
            # If we have person_roles identified, determine who performed the activity
            if 'person_roles' in activity and activity['person_roles'] and self.crew_members:
                # For now, assume the activity was performed by the first person detected
                # In future, you could use more sophisticated logic (e.g., hand detection, object proximity)
                first_person_idx = min(activity['person_roles'].keys())
                first_person_role = activity['person_roles'][first_person_idx]['role']
                performing_role = first_person_role
                
                # Get crew info from crew_members mapping
                if first_person_role in self.crew_members:
                    activity_crew_name = self.crew_members[first_person_role]['name']
                    activity_crew_id = self.crew_members[first_person_role]['id']
                    # Map role string to numeric value: LP=1, ALP=2
                    activity_crew_role = 1 if first_person_role == 'LP' else 2
                else:
                    # If role not in crew_members, default to LP
                    performing_role = 'LP'
            
            # Create JSON data in the required format
            # Store FULL PATHS for clips and images
            json_data = {
                "tripId": self.trip_id,
                "activityType": self.activity_type_map[activity_name],
                "des": self.activity_descriptions[activity_name],
                "objectType": activity_name.replace('_', ' '),
                "fileUrl": os.path.abspath(self.video_path),
                "fileDuration": video_duration_formatted,
                "activityStartTime": f"{activity_start_seconds:.2f}",
                "activityEndTime": f"{activity_end_seconds:.2f}",
                "crewName": activity_crew_name,
                "crewId": activity_crew_id,
                "crewRole": activity_crew_role,
                "performingRole": performing_role,  # LP or ALP
                "date": current_date,
                "time": current_time,
                "filename": video_filename,
                "peopleCount": len(activity.get('person_roles', {})) if activity.get('person_roles') else people_count,
                "evidence": {"rule": self.evidence_rules[activity_name]},
                "activityImage": os.path.abspath(image_path) if self.evidence_clips_dir else image_filename,
                "activityClip": os.path.abspath(clip_path) if self.evidence_clips_dir else clip_filename
            }
            
            # Add person role information if available
            if 'person_roles' in activity and activity['person_roles']:
                person_roles_list = []
                for person_idx in sorted(activity['person_roles'].keys()):
                    role_info = activity['person_roles'][person_idx]
                    person_roles_list.append({
                        "personIndex": person_idx,
                        "role": role_info['role'],
                        "roleName": role_info['role_name'],
                        "lpScore": role_info['lp_score'],
                        "alpScore": role_info['alp_score']
                    })
                json_data["personRoles"] = person_roles_list
            
            # Add to all activities list
            self.all_activities.append(json_data)
            
            # Calculate end time string for logging
            end_time_str = str(timedelta(seconds=activity_end_seconds))
            
            print(f"[{end_time_str}] Activity ended: {activity_name}")
            print(f"  Clip Duration: {actual_clip_duration:.2f}s ({total_clip_frames} frames @ {self.sample_fps} FPS)")
            print(f"  Min Duration Threshold: {min_duration}s | Required Consecutive: {self.activity_thresholds[activity_name]['required_consecutive']} frames")
            print(f"  Evidence saved: {clip_filename}")
            print(f"  Activity image: {image_filename}")
            
            activity['frames'] = []
            activity['duration'] = 0
            self.consecutive_detections[activity_name] = 0
            self.grace_counters[activity_name] = 0
            
            self.evidence_counter += 1
    
    def save_video_clip(self, frames, output_path, fps):
        """Save frames as video clip at sample FPS for full-duration playback.
        
        Args:
            frames: List of frames to save
            output_path: Path to save video
            fps: FPS to use for video (should be sample_fps for real-time duration)
        """
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Use the provided FPS (sample_fps) to create full-duration clips
        # Example: 13 frames @ 0.5 FPS = 26 seconds (real-time)
        # instead of: 13 frames @ 30 FPS = 0.43 seconds (fast-motion)
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
        print(f"Run directory: {self.run_dir}")
        if self.save_annotated_frames:
            if self.frame_save_interval == 1:
                print(f"  Saving ALL sampled frames (~{expected_samples} frames) to: {self.frames_dir}")
            else:
                print(f"  Saving every {self.frame_save_interval}th sampled frame (~{expected_samples//self.frame_save_interval} frames) to: {self.frames_dir}")
        else:
            print("  Annotated frame saving is disabled (faster processing)")
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
                
                # Run pose-based sleep detection (always, as backup or primary method)
                pose_sleep_detected = False
                pose_microsleep_detected = False
                pose_sleep_info = {}
                
                if pose_results.pose_landmarks:
                    pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.detect_pose_based_sleep(
                        pose_results.pose_landmarks, timestamp_sec
                    )
                
                # Detect objects with pose-guided detection
                detections = self.detect_objects(frame, pose_results.pose_landmarks, use_pose_guided=True)
                
                # Count people in frame  
                people_count = len(detections['person'])
                if people_count == 0:
                    people_count = 1  # Default to 1 if no person detected
                
                # Initialize detection flags AND detect new activities BEFORE frame saving
                # This ensures all detections are available for visualization
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
                
                # Check for cell phone usage with stricter validation
                if pose_results.pose_landmarks and len(detections['cell_phone']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    right_shoulder_y = int(right_shoulder.y * h)
                    left_shoulder_y = int(left_shoulder.y * h)
                    avg_shoulder_y = (right_shoulder_y + left_shoulder_y) / 2
                    nose_y = int(nose.y * h)
                    
                    margin = self.activity_thresholds['cell_phone']['margin']
                    for phone_bbox in detections['cell_phone']:
                        # Check if hand is near phone
                        right_hand_near = self.check_hand_object_interaction(right_hand_coords, phone_bbox, margin)
                        left_hand_near = self.check_hand_object_interaction(left_hand_coords, phone_bbox, margin)
                        
                        if right_hand_near or left_hand_near:
                            # STRICTER VALIDATION: Ensure phone is being actively used, not just in pocket
                            # 1. Phone should be in upper body area (above hip level = top 60% of frame)
                            phone_center_y = (phone_bbox[1] + phone_bbox[3]) / 2
                            phone_in_upper_body = phone_center_y < (h * 0.6)
                            
                            # 2. At least one hand should be raised (above shoulder level or near face)
                            right_hand_raised = right_hand_coords[1] < (avg_shoulder_y + 100)  # Within 100px below shoulder
                            left_hand_raised = left_hand_coords[1] < (avg_shoulder_y + 100)
                            hand_raised = right_hand_raised or left_hand_raised
                            
                            # 3. Hand should be in front of body (typical phone usage), not hanging at sides
                            # Check if hand that's near phone is also elevated
                            active_hand_raised = (right_hand_near and right_hand_raised) or (left_hand_near and left_hand_raised)
                            
                            if phone_in_upper_body and hand_raised and active_hand_raised:
                                cell_phone_detected = True
                                
                                # Log cell phone in hand detection
                                if self.consecutive_detections['cell_phone'] == 0:
                                    print(f"[{timestamp}] Cell phone ACTIVELY USED in hand (frame {frame_idx}, hand raised, upper body)")
                                
                                break
                            else:
                                # Log rejection for debugging
                                if self.consecutive_detections['cell_phone'] == 0:
                                    reason = []
                                    if not phone_in_upper_body: reason.append("phone too low")
                                    if not hand_raised: reason.append("hands down")
                                    if not active_hand_raised: reason.append("active hand not raised")
                                    print(f"[{timestamp}] Cell phone detected but REJECTED - likely in pocket/holder ({', '.join(reason)})")
                
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
                
                # NEW: Check for group detection (more than 2 people)
                group_detected_flag = False
                person_roles = {}  # Store person role information
                
                if len(detections['person']) > 0:
                    # De-duplicate person boxes to get accurate count
                    deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.3)
                    deduplicated_count = len(deduplicated_persons)
                    
                    # Store deduplicated boxes back in detections for visualization
                    detections['deduplicated_person'] = deduplicated_persons
                    
                    # Identify person roles (LP, ALP, etc.)
                    person_roles = self.identify_person_roles(frame, deduplicated_persons, detections)
                    
                    # Log role identification (only once per detection cycle)
                    if self.consecutive_detections['group_detected'] == 0 and person_roles:
                        print(f"[{timestamp}] Person roles identified:")
                        for person_idx in sorted(person_roles.keys()):
                            role_info = person_roles[person_idx]
                            print(f"  Person {person_idx+1}: {role_info['role_name']} (LP score: {role_info['lp_score']}, ALP score: {role_info['alp_score']})")
                    
                    if deduplicated_count > 2:
                        group_detected_flag = True
                        if self.consecutive_detections['group_detected'] == 0:
                            print(f"[{timestamp}] Group detected - {deduplicated_count} people (de-duplicated from {len(detections['person'])} raw detections)")
                else:
                    # No person detected at all
                    detections['deduplicated_person'] = []
                    person_roles = {}
                
                # HEURISTIC: Detect "writing/reading posture" even if YOLO misses objects
                holding_object_heuristic = False
                writing_posture_heuristic = False
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    right_elbow_coords = (int(right_elbow.x * w), int(right_elbow.y * h))
                    right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                    
                    # HEURISTIC 1: Reading posture - hands close together (holding book/phone)
                    hand_distance = np.sqrt((right_hand_coords[0] - left_hand_coords[0])**2 + 
                                           (right_hand_coords[1] - left_hand_coords[1])**2)
                    
                    if hand_distance < 150:
                        holding_object_heuristic = True
                        if not writing_detected and not cell_phone_detected:
                            print(f"[{timestamp}] Heuristic: Hands together (d={hand_distance:.0f}px) - writing posture")
                    
                # HEURISTIC 2: Writing posture - right hand low and forward (writing on desk/logbook)
                # Writing characteristics:
                # 1. Right hand is below elbow (hand lower than elbow = writing down)
                # 2. Right hand is in lower 60% of frame (near desk/table)
                # 3. Elbow is bent (elbow Y between shoulder and hand)
                # 4. Hands are NOT too close together (phone = both hands, writing = one hand)
                # 5. Hand must be NEAR a detected book (no fallback to prevent false positives)
                hand_below_elbow = right_hand_coords[1] > right_elbow_coords[1]  # Y increases downward
                hand_in_lower_area = right_hand_coords[1] > (h * 0.4)  # Lower 60% of frame
                elbow_bent = right_elbow_coords[1] > right_shoulder_coords[1] and right_elbow_coords[1] < right_hand_coords[1]
                
                # Prevent phone false positive - if hands are close together, it's likely phone usage
                hands_separated = hand_distance >= 150  # Hands far apart = writing, close = phone
                
                # Check if hand is actually NEAR any detected book (not just if book exists anywhere)
                hand_near_book = False
                margin = self.activity_thresholds['writing']['margin']
                if len(detections['book']) > 0:
                    for book_bbox in detections['book']:
                        if self.check_hand_object_interaction(right_hand_coords, book_bbox, margin):
                            hand_near_book = True
                            break
                
                # Only trigger writing if hand is NEAR a book (strict detection to prevent false positives)
                # Removed hand_on_surface fallback as it caused too many false positives
                if hand_below_elbow and hand_in_lower_area and elbow_bent and hands_separated and hand_near_book:
                    writing_posture_heuristic = True
                    if not writing_detected and not cell_phone_detected:  # Don't override cell phone detection
                        writing_detected = True  # Activate writing detection via heuristic
                        print(f"[{timestamp}] Heuristic: Writing posture detected (hand Y={right_hand_coords[1]}, elbow Y={right_elbow_coords[1]}, hands separated={hands_separated}, book_near_hand={hand_near_book})")
                
                # NEW: Check for hand gesture (LP/ALP not exchanging hand gesture)
                # AND ALL OTHER ACTIVITIES using COMPREHENSIVE MULTI-PERSON DETECTION
                lp_hand_gesture_detected = False
                alp_hand_gesture_detected = False
                gesture_debug = {}  # Initialize gesture debug info
                multi_person_results = None  # Store for visualization
                
                # Run comprehensive multi-person detection if we have 2+ people
                if len(person_roles) >= 2:
                    # Multi-person scenario: Run COMPREHENSIVE detection on each person's cropped region
                    # This detects ALL activities per person: gestures, sleep, mind diversion, etc.
                    multi_person_results = self.detect_per_person_activities(frame, person_roles, timestamp_sec)
                    
                    # Process results for each person
                    for person_idx, person_result in multi_person_results.items():
                        role = person_result['role']
                        activities = person_result['activities']
                        metrics = person_result['metrics']
                        
                        # 1. Hand Gestures
                        if activities['hand_gesture']:
                            gesture_type = activities['gesture_type']
                            debug_info = metrics['gesture_debug']
                            gesture_debug = debug_info  # Capture debug info
                            
                            if gesture_type == 'lp':
                                lp_hand_gesture_detected = True
                                if self.consecutive_detections['lp_hand_gesture'] == 0:
                                    print(f"[{timestamp}] LP hand gesture detected (Person {person_idx}) - {debug_info.get('hand_raised', 'unknown')} hand raised (multi-person mode)")
                            elif gesture_type == 'alp':
                                alp_hand_gesture_detected = True
                                if self.consecutive_detections['alp_hand_gesture'] == 0:
                                    print(f"[{timestamp}] ALP hand gesture detected (Person {person_idx}) - {debug_info.get('hand_raised', 'unknown')} hand raised (multi-person mode)")
                        
                        # 2. Sleep/Microsleep (per person)
                        if activities['sleep']:
                            sleep_detected = True
                            sleep_duration = metrics['pose_sleep_info'].get('pose_sleep_duration', 0)
                            if sleep_duration > 0:
                                print(f"[{timestamp}] SLEEP detected for {role} (Person {person_idx}) - Duration: {sleep_duration:.1f}s")
                        elif activities['microsleep']:
                            microsleep_detected = True
                            sleep_duration = metrics['pose_sleep_info'].get('pose_sleep_duration', 0)
                            if sleep_duration > 0:
                                print(f"[{timestamp}] MICROSLEEP detected for {role} (Person {person_idx}) - Duration: {sleep_duration:.1f}s")
                        
                        # 3. Mind Diversion (per person)
                        if activities['mind_diversion']:
                            mind_diversion_detected = True
                            head_pose = metrics['head_pose_info']
                            yaw = head_pose.get('yaw', 0)
                            pitch = head_pose.get('pitch', 0)
                            print(f"[{timestamp}] MIND DIVERSION detected for {role} (Person {person_idx}) - Yaw={yaw:.1f}°, Pitch={pitch:.1f}°")
                
                # Fallback to single-person pose detection if only 1 person or multi-person failed
                elif pose_results.pose_landmarks and person_roles:
                    lp_gesture, alp_gesture, gesture_debug = self.detect_hand_gesture(
                        pose_results.pose_landmarks, 
                        frame.shape, 
                        person_roles
                    )
                    
                    if lp_gesture:
                        lp_hand_gesture_detected = True
                        if self.consecutive_detections['lp_hand_gesture'] == 0:
                            print(f"[{timestamp}] LP hand gesture detected - {gesture_debug.get('hand_raised', 'unknown')} hand raised (single-person mode)")
                    
                    if alp_gesture:
                        alp_hand_gesture_detected = True
                        if self.consecutive_detections['alp_hand_gesture'] == 0:
                            print(f"[{timestamp}] ALP hand gesture detected - {gesture_debug.get('hand_raised', 'unknown')} hand raised (single-person mode)")
                
                # NEW: Check for mind diversion (head turned to side AND looking down)
                mind_diversion_detected = False
                head_pose_info = {}
                
                if pose_results.pose_landmarks:
                    # Calculate head pose angles (yaw and pitch)
                    head_pose_info = self.calculate_head_pose_angles(
                        pose_results.pose_landmarks,
                        face_results,
                        frame.shape
                    )
                    
                    # Check if mind diversion is detected based on angles
                    if head_pose_info.get('detected', False):
                        mind_diversion_detected = True
                        
                        # Log detection with angle information
                        if self.consecutive_detections['mind_diversion'] == 0:
                            yaw = head_pose_info.get('yaw', 0)
                            pitch = head_pose_info.get('pitch', 0)
                            method = head_pose_info.get('method', 'unknown')
                            print(f"[{timestamp}] MIND DIVERSION detected - Yaw={yaw:.1f}°, Pitch={pitch:.1f}° (method: {method})")
                
                # CRITICAL: Exclude sleep detection if person is holding objects or in active posture
                # If someone has a phone, book, or backpack in hand, they're clearly NOT sleeping
                # Also use heuristic detection (hands together, writing posture) as fallback
                if cell_phone_detected or writing_detected or packing_detected or holding_object_heuristic:
                    if microsleep_detected or sleep_detected:
                        reason = []
                        if cell_phone_detected: reason.append("phone")
                        if writing_detected: 
                            if writing_posture_heuristic:
                                reason.append("writing-posture-heuristic")
                            else:
                                reason.append("book")
                        if packing_detected: reason.append("backpack")
                        if holding_object_heuristic and not (cell_phone_detected or writing_detected or packing_detected):
                            reason.append("writing-posture-heuristic")
                        print(f"[{timestamp}] Sleep detection OVERRIDDEN - person active ({', '.join(reason)})")
                    microsleep_detected = False
                    sleep_detected = False
                    # Reset sleep tracking counters
                    self.eye_closure_start = None
                    self.eye_closure_duration = 0
                    self.pose_sleep_start = None
                    self.pose_sleep_duration = 0
                
                # Create annotated frame with all detections (pose landmarks + YOLO boxes)
                # This annotated frame will be used for BOTH activity clips AND periodic frame saving
                annotated_frame_for_activity = self.draw_bounding_boxes(
                    frame, detections, show_roi_boxes=True, person_roles=person_roles
                )
                annotated_frame_for_activity = self.draw_mediapipe_outputs(
                    annotated_frame_for_activity,
                    pose_results,
                    face_results,
                    ear_value,
                    self.eye_closure_duration,
                    pose_sleep_info,
                    head_pose_info,
                    multi_person_results  # Pass multi-person results for visualization
                )
                
                # Save annotated frames periodically if enabled (AFTER all detections)
                if (
                    self.save_annotated_frames 
                    and self.frames_dir is not None
                    and sample_idx % self.frame_save_interval == 0
                ):
                    try:
                        # Add gesture debug overlay if enabled
                        frame_to_save = self.draw_hand_gesture_debug(
                            annotated_frame_for_activity.copy(),
                            gesture_debug,
                            lp_hand_gesture_detected,
                            alp_hand_gesture_detected
                        )
                        
                        # Save frame with unique filename
                        frame_filename = f"frame_{frame_idx:08d}.jpg"
                        frame_path = os.path.join(self.frames_dir, frame_filename)
                        
                        # Ensure directory exists (for multiprocessing safety)
                        os.makedirs(self.frames_dir, exist_ok=True)
                        
                        # Save with high quality
                        cv2.imwrite(frame_path, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                    except Exception as e:
                        print(f"[{timestamp}] Error saving frame {frame_idx}: {e}")
                
                # Update activity states with temporal filtering
                activities_map = {
                    'microsleep': microsleep_detected and not sleep_detected,
                    'sleep': sleep_detected,
                    'cell_phone': cell_phone_detected,
                    'writing': writing_detected,
                    'packing_bags': packing_detected,
                    'group_detected': group_detected_flag,
                    'lp_hand_gesture': lp_hand_gesture_detected,
                    'alp_hand_gesture': alp_hand_gesture_detected,
                    'mind_diversion': mind_diversion_detected
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
                                self.start_activity(activity_name, timestamp, fps, frame_idx, person_roles=person_roles)
                            
                            # Continue recording frames ONLY when activity is actively detected
                            if self.activities[activity_name]['active']:
                                # Store annotated frame (with pose landmarks + YOLO boxes) instead of raw frame
                                self.activities[activity_name]['frames'].append(annotated_frame_for_activity.copy())
                                self.activities[activity_name]['last_frame_count'] = frame_idx
                                self.activities[activity_name]['last_detected_frame'] = frame_idx  # Track last actual detection
                                # Update person roles (in case they change during activity)
                                if person_roles:
                                    self.activities[activity_name]['person_roles'] = person_roles
                    else:
                        # Activity not detected - use grace period before resetting
                        if self.consecutive_detections[activity_name] > 0 or self.activities[activity_name]['active']:
                            # Increment grace counter
                            self.grace_counters[activity_name] += 1
                            grace_frames = self.activity_thresholds[activity_name]['grace_frames']
                            
                            # If still within grace period, keep activity alive but DON'T add frames
                            if self.grace_counters[activity_name] <= grace_frames:
                                # Still in grace period - keep activity active but don't record frames
                                # This allows brief interruptions without ending the activity
                                pass
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
        print(f"Run directory: {self.run_dir}")
        print(f"  - Clips: {self.evidence_clips_dir}")
        if self.save_annotated_frames:
            print(f"  - Frames: {self.frames_dir}")
        print(f"  - Activities: {os.path.join(self.run_dir, 'activities.json')}")
        print(f"{'=' * 60}")
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate gesture analysis report
        self.generate_gesture_report()
    
    def process_video_range(self, start_frame: int, end_frame: int, save_clips: bool = False) -> list:
        """
        Process a specific frame range (for multiprocessing support)
        
        This method processes only frames within the specified range and returns
        detected activities without saving clips/images to disk (activities in memory only).
        
        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
            save_clips: Whether to save video clips and images (default: False for multiprocessing)
            
        Returns:
            List of detected activities in this range
        """
        # Get video metadata
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Processing frame range {start_frame}-{end_frame} (worker {os.getpid()})")
        
        sampled_count = 0
        
        # Use the frame sampling generator with range limits
        for sample_idx, timestamp_sec, frame, frame_idx in self.sample_video_frames(
            self.video_path, start_frame=start_frame, end_frame=end_frame
        ):
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
                
                # Calculate EAR for all detected faces
                ear_value = None
                min_ear_value = None
                
                if face_results.multi_face_landmarks:
                    ear_values = []
                    for face_landmarks in face_results.multi_face_landmarks:
                        ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark)
                        if ear is not None:
                            ear_values.append(ear)
                    
                    if ear_values:
                        min_ear_value = min(ear_values)
                        ear_value = min_ear_value
                
                # Run pose-based sleep detection
                pose_sleep_detected = False
                pose_microsleep_detected = False
                pose_sleep_info = {}
                
                if pose_results.pose_landmarks:
                    pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.detect_pose_based_sleep(
                        pose_results.pose_landmarks, timestamp_sec
                    )
                
                # Detect objects with pose-guided detection
                detections = self.detect_objects(frame, pose_results.pose_landmarks, use_pose_guided=True)
                
                # Count people in frame
                people_count = len(detections['person'])
                if people_count == 0:
                    people_count = 1
                
                # Initialize detection flags
                microsleep_detected = False
                sleep_detected = False
                cell_phone_detected = False
                writing_detected = False
                packing_detected = False
                
                # Check for sleep/microsleep (face-based)
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
                else:
                    # Use pose-based detection as fallback
                    if pose_sleep_detected:
                        sleep_detected = True
                    elif pose_microsleep_detected:
                        microsleep_detected = True
                
                # Check for cell phone usage
                if pose_results.pose_landmarks and len(detections['cell_phone']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    right_shoulder_y = int(right_shoulder.y * h)
                    left_shoulder_y = int(left_shoulder.y * h)
                    avg_shoulder_y = (right_shoulder_y + left_shoulder_y) / 2
                    
                    margin = self.activity_thresholds['cell_phone']['margin']
                    for phone_bbox in detections['cell_phone']:
                        right_hand_near = self.check_hand_object_interaction(right_hand_coords, phone_bbox, margin)
                        left_hand_near = self.check_hand_object_interaction(left_hand_coords, phone_bbox, margin)
                        
                        if right_hand_near or left_hand_near:
                            phone_center_y = (phone_bbox[1] + phone_bbox[3]) / 2
                            phone_in_upper_body = phone_center_y < (h * 0.6)
                            
                            right_hand_raised = right_hand_coords[1] < (avg_shoulder_y + 100)
                            left_hand_raised = left_hand_coords[1] < (avg_shoulder_y + 100)
                            hand_raised = right_hand_raised or left_hand_raised
                            
                            active_hand_raised = (right_hand_near and right_hand_raised) or (left_hand_near and left_hand_raised)
                            
                            if phone_in_upper_body and hand_raised and active_hand_raised:
                                cell_phone_detected = True
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
                
                # Check for packing bags
                if pose_results.pose_landmarks and len(detections['backpack']) > 0:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    
                    margin = self.activity_thresholds['packing_bags']['margin']
                    
                    for backpack_bbox in detections['backpack']:
                        if (self.check_hand_object_interaction(right_hand_coords, backpack_bbox, margin) or
                            self.check_hand_object_interaction(left_hand_coords, backpack_bbox, margin)):
                            packing_detected = True
                            break
                
                # Check for group detection
                group_detected_flag = False
                person_roles = {}  # Store person role information
                
                if len(detections['person']) > 0:
                    deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.3)
                    deduplicated_count = len(deduplicated_persons)
                    detections['deduplicated_person'] = deduplicated_persons
                    
                    # Identify person roles (LP, ALP, etc.)
                    person_roles = self.identify_person_roles(frame, deduplicated_persons, detections)
                    
                    if deduplicated_count > 2:
                        group_detected_flag = True
                else:
                    detections['deduplicated_person'] = []
                    person_roles = {}
                
                # Writing/reading posture heuristics
                holding_object_heuristic = False
                writing_posture_heuristic = False
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    h, w = frame.shape[:2]
                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    right_elbow_coords = (int(right_elbow.x * w), int(right_elbow.y * h))
                    right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                    
                    hand_distance = np.sqrt((right_hand_coords[0] - left_hand_coords[0])**2 + 
                                           (right_hand_coords[1] - left_hand_coords[1])**2)
                    
                    if hand_distance < 150:
                        holding_object_heuristic = True
                    
                    # HEURISTIC 2: Writing posture - right hand low and forward (writing on desk/logbook)
                    # Writing characteristics:
                    # 1. Right hand is below elbow (hand lower than elbow = writing down)
                    # 2. Right hand is in lower 60% of frame (near desk/table)
                    # 3. Elbow is bent (elbow Y between shoulder and hand)
                    # 4. Hands are NOT too close together (phone = both hands, writing = one hand)
                    # 5. Hand must be NEAR a detected book (no fallback to prevent false positives)
                    hand_below_elbow = right_hand_coords[1] > right_elbow_coords[1]
                    hand_in_lower_area = right_hand_coords[1] > (h * 0.4)
                    elbow_bent = right_elbow_coords[1] > right_shoulder_coords[1] and right_elbow_coords[1] < right_hand_coords[1]
                    
                    # Prevent phone false positive - if hands are close together, it's likely phone usage
                    hands_separated = hand_distance >= 150  # Hands far apart = writing, close = phone
                    
                    # Check if hand is actually NEAR any detected book (not just if book exists anywhere)
                    hand_near_book = False
                    margin = self.activity_thresholds['writing']['margin']
                    if len(detections['book']) > 0:
                        for book_bbox in detections['book']:
                            if self.check_hand_object_interaction(right_hand_coords, book_bbox, margin):
                                hand_near_book = True
                                break
                    
                    # Only trigger writing if hand is NEAR a book (strict detection to prevent false positives)
                    # Removed hand_on_surface fallback as it caused too many false positives
                    if hand_below_elbow and hand_in_lower_area and elbow_bent and hands_separated and hand_near_book:
                        writing_posture_heuristic = True
                        if not writing_detected and not cell_phone_detected:  # Don't override cell phone detection
                            writing_detected = True
                
                # NEW: Check for hand gesture (LP/ALP not exchanging hand gesture)
                lp_hand_gesture_detected = False
                alp_hand_gesture_detected = False
                gesture_debug = {}  # Initialize gesture debug info
                multi_person_results = None  # Store for visualization
                
                if pose_results.pose_landmarks and person_roles:
                    lp_gesture, alp_gesture, gesture_debug = self.detect_hand_gesture(
                        pose_results.pose_landmarks, 
                        frame.shape, 
                        person_roles
                    )
                    
                    if lp_gesture:
                        lp_hand_gesture_detected = True
                    
                    if alp_gesture:
                        alp_hand_gesture_detected = True
                
                # NEW: Check for mind diversion (head turned to side AND looking down)
                mind_diversion_detected = False
                head_pose_info = {}
                
                if pose_results.pose_landmarks:
                    # Calculate head pose angles (yaw and pitch)
                    head_pose_info = self.calculate_head_pose_angles(
                        pose_results.pose_landmarks,
                        face_results,
                        frame.shape
                    )
                    
                    # Check if mind diversion is detected based on angles
                    if head_pose_info.get('detected', False):
                        mind_diversion_detected = True
                
                # Override sleep detection if person is active
                if cell_phone_detected or writing_detected or packing_detected or holding_object_heuristic:
                    microsleep_detected = False
                    sleep_detected = False
                    self.eye_closure_start = None
                    self.eye_closure_duration = 0
                    self.pose_sleep_start = None
                    self.pose_sleep_duration = 0
                
                # Create annotated frame with all detections (pose landmarks + YOLO boxes)
                # This annotated frame will be used for BOTH activity clips AND periodic frame saving
                annotated_frame_for_activity = self.draw_bounding_boxes(
                    frame, detections, show_roi_boxes=True, person_roles=person_roles
                )
                annotated_frame_for_activity = self.draw_mediapipe_outputs(
                    annotated_frame_for_activity,
                    pose_results,
                    face_results,
                    ear_value,
                    self.eye_closure_duration,
                    pose_sleep_info,
                    head_pose_info,
                    multi_person_results  # Pass multi-person results for visualization
                )
                
                # Save annotated frames periodically if enabled (in process_video_range for multiprocessing)
                if (
                    self.save_annotated_frames 
                    and self.frames_dir is not None
                    and sample_idx % self.frame_save_interval == 0
                ):
                    try:
                        # Add gesture debug overlay if enabled
                        frame_to_save = self.draw_hand_gesture_debug(
                            annotated_frame_for_activity.copy(),
                            gesture_debug,
                            lp_hand_gesture_detected,
                            alp_hand_gesture_detected
                        )
                        
                        # Save frame with unique filename
                        frame_filename = f"frame_{frame_idx:08d}.jpg"
                        frame_path = os.path.join(self.frames_dir, frame_filename)
                        
                        # Ensure directory exists (for multiprocessing safety)
                        os.makedirs(self.frames_dir, exist_ok=True)
                        
                        # Save with high quality
                        cv2.imwrite(frame_path, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                    except Exception as e:
                        print(f"[{timestamp}] Error saving frame {frame_idx}: {e}")
                
                # Update activity states with temporal filtering
                activities_map = {
                    'microsleep': microsleep_detected and not sleep_detected,
                    'sleep': sleep_detected,
                    'cell_phone': cell_phone_detected,
                    'writing': writing_detected,
                    'packing_bags': packing_detected,
                    'group_detected': group_detected_flag,
                    'lp_hand_gesture': lp_hand_gesture_detected,
                    'alp_hand_gesture': alp_hand_gesture_detected,
                    'mind_diversion': mind_diversion_detected
                }
                
                for activity_name, detected in activities_map.items():
                    if detected:
                        self.consecutive_detections[activity_name] += 1
                        self.grace_counters[activity_name] = 0
                        
                        required_consecutive = self.activity_thresholds[activity_name]['required_consecutive']
                        
                        if self.consecutive_detections[activity_name] >= required_consecutive:
                            if not self.activities[activity_name]['active']:
                                self.start_activity(activity_name, timestamp, fps, frame_idx, person_roles=person_roles)
                            
                            if self.activities[activity_name]['active']:
                                # Store annotated frame (with pose landmarks + YOLO boxes) instead of raw frame
                                self.activities[activity_name]['frames'].append(annotated_frame_for_activity.copy())
                                self.activities[activity_name]['last_frame_count'] = frame_idx
                                self.activities[activity_name]['last_detected_frame'] = frame_idx
                                # Update person roles (in case they change during activity)
                                if person_roles:
                                    self.activities[activity_name]['person_roles'] = person_roles
                    else:
                        if self.consecutive_detections[activity_name] > 0 or self.activities[activity_name]['active']:
                            self.grace_counters[activity_name] += 1
                            grace_frames = self.activity_thresholds[activity_name]['grace_frames']
                            
                            if self.grace_counters[activity_name] <= grace_frames:
                                pass
                            else:
                                if self.activities[activity_name]['active']:
                                    self.end_activity(activity_name, timestamp, fps, frame_idx, people_count, save_clips=save_clips)
                                self.consecutive_detections[activity_name] = 0
                                self.grace_counters[activity_name] = 0
                        else:
                            self.grace_counters[activity_name] = 0
            
            except Exception as e:
                print(f"\nError processing sample {sample_idx} (frame {frame_idx}): {e}")
                continue
        
        # End any remaining active activities
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, 1, save_clips=save_clips)
        
        print(f"Frame range {start_frame}-{end_frame} completed: {len(self.all_activities)} activities")
        
        # Return detected activities (without generating summary reports)
        return self.all_activities

    def generate_summary_report(self):
        """Generate activities.json in the run directory"""
        # Save the activities array in the run directory
        activities_json_path = os.path.join(self.run_dir, "activities.json")
        with open(activities_json_path, 'w') as f:
            json.dump(self.all_activities, f, indent=2)
        
        print(f"\nActivities JSON saved: {activities_json_path}")
        print(f"Total activities detected: {len(self.all_activities)}")
        
        # Count and print activity breakdown
        activities_by_type = {}
        for activity in self.all_activities:
            activity_type = activity['des']
            if activity_type not in activities_by_type:
                activities_by_type[activity_type] = 0
            activities_by_type[activity_type] += 1
        
        # Print activity breakdown
        if activities_by_type:
            print("\nActivity Breakdown:")
            for activity_type, count in activities_by_type.items():
                print(f"  - {activity_type}: {count}")


# Usage example
if __name__ == "__main__":
    video_path = "example_data/latest.mp4"
    
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