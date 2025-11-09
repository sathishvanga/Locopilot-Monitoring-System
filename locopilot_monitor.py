import cv2
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
import os

class LocopilotActivityMonitor:
    def __init__(self, video_path, output_dir="evidence", save_annotated_frames=False, frame_save_interval=30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.evidence_clips_dir = os.path.join(output_dir, "clips")
        self.json_dir = os.path.join(output_dir, "json")
        
        # Control annotated frame saving
        self.save_annotated_frames = save_annotated_frames
        self.frame_save_interval = frame_save_interval  # Save 1 frame every N frames
        
        if self.save_annotated_frames:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.frames_dir = os.path.join(output_dir, "frames", f"run_{self.run_timestamp}")
            os.makedirs(self.frames_dir, exist_ok=True)
        
        # Create directories
        os.makedirs(self.evidence_clips_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Initialize models
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8m.pt')
        print("Initializing MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
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
                'min_duration': 5.0,          # Must last 5 seconds minimum
                'required_consecutive': 150,  # 5 seconds @ 30fps
                'margin': 30                  # Tighter proximity check
            },
            'writing': {
                'min_duration': 3.0,          # Must last 3 seconds minimum
                'required_consecutive': 90,   # 3 seconds @ 30fps
                'margin': 50
            },
            'cell_phone': {
                'min_duration': 2.0,          # Must last 2 seconds minimum
                'required_consecutive': 60,   # 2 seconds @ 30fps
                'margin': 50
            },
            'microsleep': {
                'min_duration': 30.0,         # Must last 30 seconds minimum
                'required_consecutive': 30,   # Just need eye closure consistency
                'margin': None                # N/A for eye-based detection
            },
            'sleep': {
                'min_duration': 180.0,        # Must last 180 seconds minimum
                'required_consecutive': 30,   # Just need eye closure consistency
                'margin': None                # N/A for eye-based detection
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
        
        # Buffer for pre-activity frames (5 seconds before)
        self.frame_buffer = deque(maxlen=150)  # 30 fps * 5 seconds
        
        # Eye closure tracking
        self.eye_closure_start = None
        self.eye_closure_duration = 0
        
        # Evidence counter
        self.evidence_counter = 0
        
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
    
    def detect_objects(self, frame):
        """Detect objects using YOLO - enhanced with multiple bag types"""
        results = self.yolo_model(frame, verbose=False)  # Suppress YOLO output
        detections = {
            'person': [],
            'cell_phone': [],
            'book': [],
            'backpack': [],
            'handbag': [],
            'suitcase': []
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
                elif class_name == 'cell phone' and conf > 0.4:
                    detections['cell_phone'].append(xyxy)
                elif class_name == 'book' and conf > 0.4:
                    detections['book'].append(xyxy)
                elif class_name == 'backpack' and conf > 0.4:
                    detections['backpack'].append(xyxy)
                elif class_name == 'handbag' and conf > 0.4:
                    detections['handbag'].append(xyxy)
                elif class_name == 'suitcase' and conf > 0.4:
                    detections['suitcase'].append(xyxy)
        
        return detections
    
    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes on frame for detected objects"""
        annotated_frame = frame.copy()
        
        colors = {
            'person': (0, 255, 0),
            'cell_phone': (0, 0, 255),
            'book': (255, 0, 0),
            'backpack': (0, 255, 255),
            'handbag': (255, 255, 0),
            'suitcase': (255, 0, 255)
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
    
    def draw_mediapipe_outputs(self, frame, pose_results, face_results, ear_value=None, eye_closure_duration=0):
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
                    
                    if eye_closure_duration >= 180:
                        duration_text += " - SLEEP ALERT!"
                        duration_color = (0, 0, 255)
                    elif eye_closure_duration >= 30:
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
    
    def end_activity(self, activity_name, timestamp, fps, frame_count):
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
                return
            
            start_time = activity['start_time']
            end_time = timestamp
            
            clip_filename = f"{activity_name}_{self.evidence_counter:04d}.mp4"
            clip_path = os.path.join(self.evidence_clips_dir, clip_filename)
            
            self.save_video_clip(activity['frames'], clip_path, fps)
            
            total_clip_frames = len(activity['frames'])
            total_clip_duration = total_clip_frames / fps
            
            json_data = {
                "activity_name": activity_name,
                "start_time": start_time,
                "end_time": end_time,
                "activity_duration_seconds": round(duration, 2),
                "total_clip_duration_seconds": round(total_clip_duration, 2),
                "total_frames_in_clip": total_clip_frames,
                "min_duration_threshold": min_duration,
                "required_consecutive_frames": self.activity_thresholds[activity_name]['required_consecutive'],
                "includes_pre_buffer": True,
                "pre_buffer_seconds": 5.0,
                "video_clip": clip_filename,
                "evidence_id": self.evidence_counter,
                "timestamp": datetime.now().isoformat()
            }
            
            json_filename = f"{activity_name}_{self.evidence_counter:04d}.json"
            json_path = os.path.join(self.json_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            print(f"[{end_time}] Activity ended: {activity_name}")
            print(f"  Activity Duration: {duration:.2f}s | Total Clip: {total_clip_duration:.2f}s")
            print(f"  Min Duration Threshold: {min_duration}s | Required Consecutive: {self.activity_thresholds[activity_name]['required_consecutive']} frames")
            print(f"  Evidence saved: {clip_filename} ({total_clip_frames} frames)")
            
            activity['frames'] = []
            activity['duration'] = 0
            self.consecutive_detections[activity_name] = 0
            
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
        """Main video processing loop - PROCESSES EVERY FRAME"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        print(f"Processing video: {self.video_path}")
        print(f"FPS: {fps}")
        print(f"Total frames in video: {total_frames}")
        print(f"Expected duration: {total_frames/fps/60:.2f} minutes")
        print(f"PROCESSING ALL FRAMES (no skipping)")
        if self.save_annotated_frames:
            print(f"Saving annotated frames every {self.frame_save_interval} frames to: {self.frames_dir}")
        else:
            print("Annotated frame saving is disabled (faster processing)")
        print("-" * 60)
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            try:
                timestamp = str(timedelta(seconds=frame_count/fps))
                
                # Add frame to buffer
                self.frame_buffer.append(frame.copy())
                
                # Process pose and face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb_frame)
                face_results = self.face_mesh.process(rgb_frame)
                
                # Calculate EAR
                ear_value = None
                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark
                    ear_value = self.calculate_eye_aspect_ratio(landmarks)
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Save annotated frames periodically if enabled
                if self.save_annotated_frames and frame_count % self.frame_save_interval == 0:
                    annotated_frame = self.draw_bounding_boxes(frame, detections)
                    annotated_frame = self.draw_mediapipe_outputs(
                        annotated_frame, pose_results, face_results, 
                        ear_value, self.eye_closure_duration
                    )
                    frame_filename = f"frame_{frame_count:08d}.jpg"
                    frame_path = os.path.join(self.frames_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)
                
                # Initialize detection flags
                microsleep_detected = False
                sleep_detected = False
                cell_phone_detected = False
                writing_detected = False
                packing_detected = False
                
                # Check for sleep/microsleep
                if face_results.multi_face_landmarks and ear_value is not None:
                    if ear_value < 0.2:
                        if self.eye_closure_start is None:
                            self.eye_closure_start = frame_count / fps
                        
                        self.eye_closure_duration = (frame_count / fps) - self.eye_closure_start
                        
                        if self.eye_closure_duration >= 180:
                            sleep_detected = True
                        elif self.eye_closure_duration >= 30:
                            microsleep_detected = True
                    else:
                        self.eye_closure_start = None
                        self.eye_closure_duration = 0
                
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
                
                # Check for packing bags - enhanced with multiple bag types and tighter margin
                if pose_results.pose_landmarks:
                    # Combine all bag types
                    all_bags = (detections['backpack'] + 
                               detections['handbag'] + 
                               detections['suitcase'])
                    
                    if len(all_bags) > 0:
                        landmarks = pose_results.pose_landmarks.landmark
                        right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                        
                        h, w = frame.shape[:2]
                        right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                        left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                        
                        # Use tighter margin for packing detection (30px instead of 50px)
                        margin = self.activity_thresholds['packing_bags']['margin']
                        
                        for bag_bbox in all_bags:
                            if (self.check_hand_object_interaction(right_hand_coords, bag_bbox, margin) or
                                self.check_hand_object_interaction(left_hand_coords, bag_bbox, margin)):
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
                        # Increment consecutive detection counter
                        self.consecutive_detections[activity_name] += 1
                        
                        # Only start recording after required consecutive frames threshold is met
                        required_consecutive = self.activity_thresholds[activity_name]['required_consecutive']
                        
                        if self.consecutive_detections[activity_name] >= required_consecutive:
                            # Start activity if not already active
                            if not self.activities[activity_name]['active']:
                                self.start_activity(activity_name, timestamp, fps, frame_count)
                            
                            # Continue recording frames
                            if self.activities[activity_name]['active']:
                                self.activities[activity_name]['frames'].append(frame.copy())
                                self.activities[activity_name]['last_frame_count'] = frame_count
                    else:
                        # Activity not detected - reset consecutive counter
                        if self.consecutive_detections[activity_name] > 0:
                            # If we were tracking consecutive detections but haven't started recording yet
                            if self.consecutive_detections[activity_name] < self.activity_thresholds[activity_name]['required_consecutive']:
                                # Just reset the counter - no activity was started
                                self.consecutive_detections[activity_name] = 0
                            else:
                                # Activity was active, now ending
                                if self.activities[activity_name]['active']:
                                    self.end_activity(activity_name, timestamp, fps, frame_count)
                                self.consecutive_detections[activity_name] = 0
                        else:
                            # Normal end of activity
                            if self.activities[activity_name]['active']:
                                self.end_activity(activity_name, timestamp, fps, frame_count)
                
                # Display progress
                if frame_count % 300 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            except Exception as e:
                print(f"\nError processing frame {frame_count}: {e}")
                continue
        
        # End any remaining active activities
        final_timestamp = str(timedelta(seconds=frame_count/fps))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_count)
        
        cap.release()
        print(f"\n{'=' * 60}")
        print(f"Processing complete!")
        print(f"Total frames processed: {frame_count}/{total_frames}")
        print(f"Evidence clips created: {self.evidence_counter}")
        print(f"Evidence saved in: {self.output_dir}")
        if self.save_annotated_frames:
            print(f"Annotated frames saved in: {self.frames_dir}")
        print(f"{'=' * 60}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of all activities"""
        summary = {
            "video_path": self.video_path,
            "processing_date": datetime.now().isoformat(),
            "total_evidence_clips": self.evidence_counter,
            "activities_detected": []
        }
        
        if self.save_annotated_frames:
            summary["run_id"] = self.run_timestamp
            summary["frames_directory"] = self.frames_dir
        
        for json_file in sorted(os.listdir(self.json_dir)):
            if json_file.endswith('.json') and json_file != 'summary_report.json':
                with open(os.path.join(self.json_dir, json_file), 'r') as f:
                    activity_data = json.load(f)
                    summary["activities_detected"].append(activity_data)
        
        summary_path = os.path.join(self.output_dir, "summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nSummary report saved: {summary_path}")


# Usage example
if __name__ == "__main__":
    video_path = "example_data/latest_1.mp4"
    
    # Option 1: Process all frames WITHOUT saving annotated frames (FASTEST)
    monitor = LocopilotActivityMonitor(
        video_path, 
        output_dir="locopilot_evidence",
        save_annotated_frames=True  # Disable frame saving for speed
    )
    
    # Option 2: Process all frames WITH annotated frames (save every 30th frame for review)
    # monitor = LocopilotActivityMonitor(
    #     video_path, 
    #     output_dir="locopilot_evidence",
    #     save_annotated_frames=True,
    #     frame_save_interval=30  # Save 1 frame every 30 frames (~1 per second at 30fps)
    # )
    
    monitor.process_video()