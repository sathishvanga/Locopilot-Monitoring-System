#!/usr/bin/env python3
"""
Video Activity Detection Diagnostic Script

This script analyzes a video at specific timestamps to diagnose WHY expected
violations were not detected by the activity detection system.

It extracts frames, runs YOLO detection and pose estimation, compares against
thresholds, and generates a detailed diagnostic report.

Usage:
    python diagnose_video.py /path/to/video.mp4
    python diagnose_video.py /path/to/video.mp4 --output-dir ./diagnostic_output
"""

import cv2
import json
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add app directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from ultralytics import YOLO
from app.utils.config import get_settings
from app.services.yolo_pose_adapter import YOLO_KEYPOINT_INDICES, YoloPoseLandmarks


# Expected violations for the test video
EXPECTED_VIOLATIONS = [
    {"id": 1, "activity": "packing_bags", "start": "0:59", "end": "1:20"},
    {"id": 2, "activity": "cell_phone", "start": "1:24", "end": "1:40"},
    {"id": 3, "activity": "packing_bags", "start": "1:57", "end": "2:45"},
    {"id": 4, "activity": "mind_diversion", "start": "2:59", "end": "5:15"},
    {"id": 5, "activity": "writing", "start": "5:25", "end": "6:40"},
    {"id": 6, "activity": "mind_diversion", "start": "6:45", "end": "7:50"},
    {"id": 7, "activity": "no_person_detected", "start": "8:02", "end": "9:20"},
    {"id": 8, "activity": "mind_diversion", "start": "9:25", "end": "11:25"},
    {"id": 9, "activity": "packing_bags", "start": "11:40", "end": "11:53"},
    {"id": 10, "activity": "mind_diversion", "start": "12:00", "end": "15:13"},
    {"id": 11, "activity": "writing", "start": "15:14", "end": "15:18"},
    {"id": 12, "activity": "mind_diversion", "start": "15:20", "end": "18:05"},
    {"id": 13, "activity": "no_person_detected", "start": "18:07", "end": "18:29"},
    {"id": 14, "activity": "writing", "start": "21:12", "end": "22:00"},
    {"id": 15, "activity": "no_person_detected", "start": "22:36", "end": "24:25"},
    {"id": 16, "activity": "mind_diversion", "start": "26:45", "end": "29:37"},
    {"id": 17, "activity": "writing", "start": "29:39", "end": "30:05"},
    {"id": 18, "activity": "mind_diversion", "start": "31:00", "end": "34:00"},
]


@dataclass
class DetectionResult:
    """Holds detection results for a single object."""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


@dataclass
class PoseKeypoint:
    """Holds a single pose keypoint."""
    name: str
    x: float  # normalized 0-1
    y: float  # normalized 0-1
    pixel_x: int
    pixel_y: int
    visibility: float


@dataclass
class FrameDiagnosis:
    """Diagnosis for a single frame."""
    timestamp: float
    timestamp_str: str
    position: str  # start, middle, end
    yolo_detections: Dict[str, List[DetectionResult]]
    pose_keypoints: Dict[str, PoseKeypoint]
    person_detected: bool
    person_confidence: float
    angles: Dict[str, float]  # yaw, pitch
    threshold_checks: Dict[str, Any]
    diagnosis: Dict[str, Any]
    annotated_frame_path: Optional[str] = None


class DiagnosticVideoAnalyzer:
    """Analyzes video frames to diagnose activity detection failures."""

    def __init__(self, video_path: str, output_dir: str = "diagnostic_output"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.settings = get_settings()

        # Create output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(output_dir, f"{video_name}_diagnostic_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "raw_detections"), exist_ok=True)

        # Load models
        print(f"Loading YOLO model: {self.settings.yolo_weights}")
        self.yolo_model = YOLO(self.settings.yolo_weights)

        print(f"Loading YOLO-Pose model: {self.settings.yolo_pose_weights}")
        self.yolo_pose_model = YOLO(self.settings.yolo_pose_weights)

        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        print(f"Video: {video_path}")
        print(f"FPS: {self.fps:.2f}, Frames: {self.total_frames}, Duration: {self.duration:.2f}s")

        # Current thresholds from settings
        self.thresholds = {
            'mind_diversion': {
                'yaw_sideways': self.settings.mind_diversion_yaw_sideways,
                'yaw_combined': self.settings.mind_diversion_yaw_combined,
                'pitch_down': self.settings.mind_diversion_pitch_down,
                'pitch_combined': self.settings.mind_diversion_pitch_combined,
                'yaw_max_for_down': self.settings.mind_diversion_yaw_max_for_down,
                'wrist_distance_suppression': self.settings.mind_diversion_wrist_distance_threshold,
            },
            'cell_phone': {
                'confidence': 0.45,
                'proximity_margin': 180,
                'hand_visibility': 0.5,
            },
            'packing_bags': {
                'confidence': 0.45,
                'wrist_inside_margin': 40,
                'hand_proximity_margin': 50,
            },
            'writing': {
                'book_confidence': 0.2,
                'hand_to_book_margin': 180,
                'wrist_proximity': 300,
            },
            'no_person': {
                'person_confidence': 0.5,
                'min_duration': 10.0,
                'required_consecutive': 5,
            },
        }

        # Results storage
        self.all_results = []

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    @staticmethod
    def parse_timestamp(ts_str: str) -> float:
        """Parse timestamp string to seconds. Handles M:SS and H:MM:SS formats."""
        parts = ts_str.strip().split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return float(ts_str)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds to M:SS or H:MM:SS string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def extract_frame(self, timestamp_seconds: float) -> Optional[np.ndarray]:
        """Extract a single frame at the given timestamp."""
        if timestamp_seconds < 0 or timestamp_seconds > self.duration:
            return None

        frame_num = int(timestamp_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        if not ret:
            return None
        return frame

    def run_yolo_detection(self, frame: np.ndarray) -> Dict[str, List[DetectionResult]]:
        """Run YOLO object detection on frame."""
        results = self.yolo_model(frame, verbose=False, imgsz=self.settings.yolo_imgsz)

        detections = defaultdict(list)

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    detections[cls_name].append(DetectionResult(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=bbox
                    ))

        return dict(detections)

    def run_yolo_pose(self, frame: np.ndarray) -> Tuple[Dict[str, PoseKeypoint], List[Dict]]:
        """Run YOLO-Pose on frame and extract keypoints."""
        h, w = frame.shape[:2]
        results = self.yolo_pose_model(frame, verbose=False, imgsz=self.settings.yolo_imgsz)

        keypoints = {}
        all_persons = []

        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Process each detected person
                for person_idx, kpts in enumerate(result.keypoints):
                    person_keypoints = {}

                    if kpts.xy is not None and len(kpts.xy) > 0:
                        xy = kpts.xy[0].cpu().numpy()  # (17, 2)
                        conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(17)

                        for name, idx in YOLO_KEYPOINT_INDICES.items():
                            px, py = xy[idx]
                            visibility = conf[idx] if idx < len(conf) else 0.0

                            kp = PoseKeypoint(
                                name=name,
                                x=px / w if w > 0 else 0,
                                y=py / h if h > 0 else 0,
                                pixel_x=int(px),
                                pixel_y=int(py),
                                visibility=float(visibility)
                            )
                            person_keypoints[name] = kp

                    all_persons.append(person_keypoints)

                    # Use first person as primary
                    if person_idx == 0:
                        keypoints = person_keypoints

        return keypoints, all_persons

    def calculate_head_angles(self, keypoints: Dict[str, PoseKeypoint], frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Calculate head yaw and pitch angles from pose keypoints."""
        h, w = frame_shape[:2]
        result = {'yaw': 0.0, 'pitch': 0.0, 'valid': False}

        required = ['nose', 'left_shoulder', 'right_shoulder', 'left_ear', 'right_ear']
        if not all(k in keypoints for k in required):
            return result

        nose = keypoints['nose']
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        left_ear = keypoints['left_ear']
        right_ear = keypoints['right_ear']

        # Check visibility
        if nose.visibility < 0.5:
            result['reason'] = 'nose_low_visibility'
            return result

        # Convert to pixel coordinates
        nose_coords = np.array([nose.pixel_x, nose.pixel_y])
        left_shoulder_coords = np.array([left_shoulder.pixel_x, left_shoulder.pixel_y])
        right_shoulder_coords = np.array([right_shoulder.pixel_x, right_shoulder.pixel_y])
        left_ear_coords = np.array([left_ear.pixel_x, left_ear.pixel_y])
        right_ear_coords = np.array([right_ear.pixel_x, right_ear.pixel_y])

        # Calculate shoulder midpoint and width
        shoulder_midpoint = (left_shoulder_coords + right_shoulder_coords) / 2
        shoulder_width = np.linalg.norm(right_shoulder_coords - left_shoulder_coords)

        if shoulder_width <= 0:
            result['reason'] = 'invalid_shoulder_width'
            return result

        # YAW: nose offset from shoulder midpoint
        nose_offset_x = nose_coords[0] - shoulder_midpoint[0]
        yaw_normalized = nose_offset_x / (shoulder_width / 2)
        yaw_angle = np.clip(yaw_normalized * 45, -90, 90)

        # PITCH: nose position relative to ears
        ear_midpoint = (left_ear_coords + right_ear_coords) / 2
        nose_offset_y = nose_coords[1] - ear_midpoint[1]
        head_height = shoulder_midpoint[1] - ear_midpoint[1]

        if head_height > 0:
            pitch_normalized = nose_offset_y / head_height
            pitch_angle = np.clip(pitch_normalized * 30, -45, 45)
        else:
            pitch_angle = 0

        result['yaw'] = float(yaw_angle)
        result['pitch'] = float(pitch_angle)
        result['valid'] = True
        result['shoulder_width'] = float(shoulder_width)
        result['nose_visibility'] = float(nose.visibility)
        result['left_ear_visibility'] = float(left_ear.visibility)
        result['right_ear_visibility'] = float(right_ear.visibility)

        return result

    def calculate_wrist_distance(self, keypoints: Dict[str, PoseKeypoint]) -> Dict[str, Any]:
        """Calculate distance between wrists."""
        result = {'distance': -1, 'valid': False}

        if 'left_wrist' not in keypoints or 'right_wrist' not in keypoints:
            return result

        left_wrist = keypoints['left_wrist']
        right_wrist = keypoints['right_wrist']

        if left_wrist.visibility < 0.3 or right_wrist.visibility < 0.3:
            result['reason'] = 'low_wrist_visibility'
            result['left_visibility'] = left_wrist.visibility
            result['right_visibility'] = right_wrist.visibility
            return result

        distance = np.sqrt(
            (left_wrist.pixel_x - right_wrist.pixel_x) ** 2 +
            (left_wrist.pixel_y - right_wrist.pixel_y) ** 2
        )

        result['distance'] = float(distance)
        result['valid'] = True
        result['left_wrist'] = {'x': left_wrist.pixel_x, 'y': left_wrist.pixel_y, 'visibility': left_wrist.visibility}
        result['right_wrist'] = {'x': right_wrist.pixel_x, 'y': right_wrist.pixel_y, 'visibility': right_wrist.visibility}

        return result

    def diagnose_packing_bags(self, detections: Dict, keypoints: Dict, frame_shape: Tuple) -> Dict:
        """Diagnose packing bags detection."""
        result = {
            'should_detect': True,
            'would_detect': False,
            'checks': {},
            'reason': None,
            'recommendations': []
        }

        thresholds = self.thresholds['packing_bags']

        # Check for bag detections
        bag_classes = ['backpack', 'handbag', 'suitcase']
        bags_found = []

        for bag_class in bag_classes:
            if bag_class in detections:
                for det in detections[bag_class]:
                    bags_found.append({
                        'class': bag_class,
                        'confidence': det.confidence,
                        'bbox': det.bbox,
                        'passes_threshold': det.confidence >= thresholds['confidence']
                    })

        result['checks']['bags_detected'] = len(bags_found) > 0
        result['checks']['bags_found'] = bags_found
        result['checks']['confidence_threshold'] = thresholds['confidence']

        if not bags_found:
            result['reason'] = 'no_bag_detected_by_yolo'
            result['recommendations'].append('Check if bag is visible and unoccluded')
            return result

        # Check bag confidence
        passing_bags = [b for b in bags_found if b['passes_threshold']]
        if not passing_bags:
            best_bag = max(bags_found, key=lambda x: x['confidence'])
            result['reason'] = f"bag_confidence_{best_bag['confidence']:.2f}_below_threshold_{thresholds['confidence']}"
            result['recommendations'].append(f"Consider lowering bag confidence from {thresholds['confidence']} to {best_bag['confidence'] - 0.05:.2f}")
            return result

        # Check wrist proximity to bag
        if 'left_wrist' in keypoints and 'right_wrist' in keypoints:
            left_wrist = keypoints['left_wrist']
            right_wrist = keypoints['right_wrist']

            for bag in passing_bags:
                bbox = bag['bbox']
                bag_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # Check if wrist is inside bag bbox (with margin)
                margin = thresholds['wrist_inside_margin']

                for wrist_name, wrist in [('left', left_wrist), ('right', right_wrist)]:
                    if wrist.visibility < 0.5:
                        continue

                    # Check if wrist inside bag bbox
                    inside_x = bbox[0] - margin <= wrist.pixel_x <= bbox[2] + margin
                    inside_y = bbox[1] - margin <= wrist.pixel_y <= bbox[3] + margin
                    wrist_inside = inside_x and inside_y

                    # Calculate distance to bag center
                    dist_to_center = np.sqrt(
                        (wrist.pixel_x - bag_center[0]) ** 2 +
                        (wrist.pixel_y - bag_center[1]) ** 2
                    )

                    result['checks'][f'{wrist_name}_wrist_inside_bag'] = wrist_inside
                    result['checks'][f'{wrist_name}_wrist_distance_to_bag'] = dist_to_center
                    result['checks'][f'{wrist_name}_wrist_visibility'] = wrist.visibility

                    if wrist_inside:
                        result['would_detect'] = True
                        result['reason'] = f'{wrist_name}_wrist_inside_bag'
                        return result

        result['reason'] = 'wrist_not_inside_bag_bbox'
        result['recommendations'].append(f"Wrist may be too far from bag. Current margin: {thresholds['wrist_inside_margin']}px")
        return result

    def diagnose_cell_phone(self, detections: Dict, keypoints: Dict, frame_shape: Tuple) -> Dict:
        """Diagnose cell phone detection."""
        result = {
            'should_detect': True,
            'would_detect': False,
            'checks': {},
            'reason': None,
            'recommendations': []
        }

        thresholds = self.thresholds['cell_phone']

        # Check for cell phone detection
        phones_found = []
        if 'cell phone' in detections:
            for det in detections['cell phone']:
                phones_found.append({
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'passes_threshold': det.confidence >= thresholds['confidence']
                })

        result['checks']['phones_detected'] = len(phones_found) > 0
        result['checks']['phones_found'] = phones_found
        result['checks']['confidence_threshold'] = thresholds['confidence']

        if not phones_found:
            result['reason'] = 'no_phone_detected_by_yolo'
            result['recommendations'].append('Phone may be too small, occluded, or at difficult angle')
            return result

        # Check phone confidence
        passing_phones = [p for p in phones_found if p['passes_threshold']]
        if not passing_phones:
            best_phone = max(phones_found, key=lambda x: x['confidence'])
            result['reason'] = f"phone_confidence_{best_phone['confidence']:.2f}_below_threshold_{thresholds['confidence']}"
            result['recommendations'].append(f"Consider lowering phone confidence from {thresholds['confidence']} to {best_phone['confidence'] - 0.05:.2f}")
            return result

        # Check proximity to hand
        proximity_margin = thresholds['proximity_margin']

        for phone in passing_phones:
            bbox = phone['bbox']
            phone_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            for wrist_name in ['left_wrist', 'right_wrist']:
                if wrist_name in keypoints:
                    wrist = keypoints[wrist_name]
                    if wrist.visibility >= thresholds['hand_visibility']:
                        dist = np.sqrt(
                            (wrist.pixel_x - phone_center[0]) ** 2 +
                            (wrist.pixel_y - phone_center[1]) ** 2
                        )

                        result['checks'][f'{wrist_name}_to_phone_distance'] = dist
                        result['checks'][f'{wrist_name}_visibility'] = wrist.visibility

                        if dist <= proximity_margin:
                            result['would_detect'] = True
                            result['reason'] = f'phone_near_{wrist_name}'
                            return result

        result['reason'] = 'phone_not_near_hand'
        result['recommendations'].append(f"Phone detected but not within {proximity_margin}px of hand")
        return result

    def diagnose_mind_diversion(self, angles: Dict, keypoints: Dict, detections: Dict, frame_shape: Tuple) -> Dict:
        """Diagnose mind diversion detection."""
        result = {
            'should_detect': True,
            'would_detect': False,
            'checks': {},
            'reason': None,
            'sub_type': None,
            'suppression': None,
            'recommendations': []
        }

        thresholds = self.thresholds['mind_diversion']

        if not angles.get('valid', False):
            result['reason'] = f"invalid_pose_angles: {angles.get('reason', 'unknown')}"
            result['checks']['angles_valid'] = False
            return result

        yaw = abs(angles['yaw'])
        pitch = angles['pitch']

        result['checks']['yaw_angle'] = angles['yaw']
        result['checks']['pitch_angle'] = pitch
        result['checks']['angles_valid'] = True

        # Check detection scenarios
        result['checks']['yaw_threshold_sideways'] = thresholds['yaw_sideways']
        result['checks']['yaw_passes_sideways'] = yaw > thresholds['yaw_sideways']

        result['checks']['yaw_threshold_combined'] = thresholds['yaw_combined']
        result['checks']['pitch_threshold_combined'] = thresholds['pitch_combined']
        result['checks']['passes_combined'] = (yaw > thresholds['yaw_combined'] and pitch > thresholds['pitch_combined'])

        result['checks']['pitch_threshold_down'] = thresholds['pitch_down']
        result['checks']['yaw_max_for_down'] = thresholds['yaw_max_for_down']
        result['checks']['passes_looking_down'] = (pitch > thresholds['pitch_down'] and yaw < thresholds['yaw_max_for_down'])

        # Determine detection result
        if yaw > thresholds['yaw_sideways']:
            result['would_detect'] = True
            result['sub_type'] = 'looking_sideways'
        elif yaw > thresholds['yaw_combined'] and pitch > thresholds['pitch_combined']:
            result['would_detect'] = True
            result['sub_type'] = 'looking_away_combined'
        elif pitch > thresholds['pitch_down'] and yaw < thresholds['yaw_max_for_down']:
            result['would_detect'] = True
            result['sub_type'] = 'looking_down_distracted'

        if not result['would_detect']:
            # Generate specific reason
            if yaw < thresholds['yaw_sideways'] and yaw > thresholds['yaw_combined']:
                result['reason'] = f"yaw_{yaw:.1f}_between_combined({thresholds['yaw_combined']})_and_sideways({thresholds['yaw_sideways']})"
                if pitch < thresholds['pitch_combined']:
                    result['reason'] += f"_pitch_{pitch:.1f}_below_combined({thresholds['pitch_combined']})"
            elif pitch < thresholds['pitch_down']:
                result['reason'] = f"pitch_{pitch:.1f}_below_threshold_{thresholds['pitch_down']}"
            else:
                result['reason'] = f"angles_below_all_thresholds_yaw_{yaw:.1f}_pitch_{pitch:.1f}"

            # Recommendations
            if yaw > thresholds['yaw_sideways'] - 10:
                result['recommendations'].append(f"Yaw {yaw:.1f} close to sideways threshold {thresholds['yaw_sideways']}. Consider lowering.")
            if pitch > thresholds['pitch_down'] - 5:
                result['recommendations'].append(f"Pitch {pitch:.1f} close to down threshold {thresholds['pitch_down']}. Consider lowering.")
        else:
            result['reason'] = f"detected_as_{result['sub_type']}"

        # Check suppression conditions
        suppression_reasons = []

        # Book detection suppression
        if 'book' in detections and len(detections['book']) > 0:
            suppression_reasons.append('book_detected')

        # Wrist proximity suppression (writing pose)
        wrist_dist = self.calculate_wrist_distance(keypoints)
        if wrist_dist['valid'] and wrist_dist['distance'] < thresholds['wrist_distance_suppression']:
            suppression_reasons.append(f"wrist_distance_{wrist_dist['distance']:.0f}px_below_{thresholds['wrist_distance_suppression']}px")

        result['checks']['wrist_distance'] = wrist_dist.get('distance', -1)
        result['checks']['wrist_distance_threshold'] = thresholds['wrist_distance_suppression']

        if suppression_reasons:
            result['suppression'] = suppression_reasons
            if result['would_detect']:
                result['reason'] = f"detected_but_suppressed: {', '.join(suppression_reasons)}"

        return result

    def diagnose_writing(self, detections: Dict, keypoints: Dict, angles: Dict, frame_shape: Tuple) -> Dict:
        """Diagnose writing detection."""
        result = {
            'should_detect': True,
            'would_detect': False,
            'checks': {},
            'detection_method': None,
            'reason': None,
            'recommendations': []
        }

        thresholds = self.thresholds['writing']

        # Method 1: Book detection
        books_found = []
        if 'book' in detections:
            for det in detections['book']:
                books_found.append({
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'passes_threshold': det.confidence >= thresholds['book_confidence']
                })

        result['checks']['books_detected'] = len(books_found) > 0
        result['checks']['books_found'] = books_found
        result['checks']['book_confidence_threshold'] = thresholds['book_confidence']

        if books_found:
            passing_books = [b for b in books_found if b['passes_threshold']]
            if passing_books:
                # Check hand near book
                for book in passing_books:
                    bbox = book['bbox']
                    book_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                    for wrist_name in ['left_wrist', 'right_wrist']:
                        if wrist_name in keypoints:
                            wrist = keypoints[wrist_name]
                            if wrist.visibility >= 0.3:
                                dist = np.sqrt(
                                    (wrist.pixel_x - book_center[0]) ** 2 +
                                    (wrist.pixel_y - book_center[1]) ** 2
                                )

                                result['checks'][f'{wrist_name}_to_book_distance'] = dist

                                if dist <= thresholds['hand_to_book_margin']:
                                    result['would_detect'] = True
                                    result['detection_method'] = 'book_detection'
                                    result['reason'] = f'{wrist_name}_near_book'
                                    return result

        # Method 2: Wrist proximity heuristic
        wrist_dist = self.calculate_wrist_distance(keypoints)
        result['checks']['wrist_distance'] = wrist_dist
        result['checks']['wrist_proximity_threshold'] = thresholds['wrist_proximity']

        if wrist_dist['valid']:
            head_down = angles.get('valid', False) and angles.get('pitch', 0) > 10
            result['checks']['head_looking_down'] = head_down

            if wrist_dist['distance'] < thresholds['wrist_proximity'] and head_down:
                result['would_detect'] = True
                result['detection_method'] = 'wrist_proximity'
                result['reason'] = f"wrists_close_{wrist_dist['distance']:.0f}px_and_head_down"
                return result

        # Generate failure reason
        if not books_found:
            result['reason'] = 'no_book_detected'
        elif not [b for b in books_found if b['passes_threshold']]:
            best = max(books_found, key=lambda x: x['confidence'])
            result['reason'] = f"book_confidence_{best['confidence']:.2f}_below_{thresholds['book_confidence']}"
            result['recommendations'].append(f"Consider lowering book confidence to {best['confidence'] - 0.05:.2f}")
        else:
            result['reason'] = 'hand_not_near_book_and_wrists_not_close'

        if wrist_dist['valid'] and wrist_dist['distance'] < thresholds['wrist_proximity'] + 100:
            result['recommendations'].append(f"Wrist distance {wrist_dist['distance']:.0f}px close to threshold {thresholds['wrist_proximity']}px")

        return result

    def diagnose_no_person(self, detections: Dict, all_persons: List) -> Dict:
        """Diagnose no person detected."""
        result = {
            'should_detect': True,
            'would_detect': False,
            'checks': {},
            'reason': None,
            'recommendations': []
        }

        thresholds = self.thresholds['no_person']

        # Check person detections
        persons_found = []
        if 'person' in detections:
            for det in detections['person']:
                persons_found.append({
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'passes_threshold': det.confidence >= thresholds['person_confidence']
                })

        result['checks']['persons_from_yolo'] = len(persons_found)
        result['checks']['persons_from_pose'] = len(all_persons)
        result['checks']['persons_found'] = persons_found
        result['checks']['person_confidence_threshold'] = thresholds['person_confidence']

        passing_persons = [p for p in persons_found if p['passes_threshold']]

        if len(passing_persons) == 0 and len(all_persons) == 0:
            result['would_detect'] = True
            result['reason'] = 'zero_persons_detected'
        else:
            result['reason'] = f"persons_detected: yolo={len(passing_persons)}, pose={len(all_persons)}"

        return result

    def annotate_frame(self, frame: np.ndarray, diagnosis: FrameDiagnosis, violation: Dict) -> np.ndarray:
        """Annotate frame with detection results and diagnosis."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Colors
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        YELLOW = (0, 255, 255)
        CYAN = (255, 255, 0)
        MAGENTA = (255, 0, 255)
        WHITE = (255, 255, 255)
        ORANGE = (0, 165, 255)

        # Draw all YOLO detections
        color_map = {
            'person': GREEN,
            'cell phone': RED,
            'book': ORANGE,
            'backpack': MAGENTA,
            'handbag': MAGENTA,
            'suitcase': MAGENTA,
        }

        for class_name, dets in diagnosis.yolo_detections.items():
            color = color_map.get(class_name, YELLOW)
            for det in dets:
                # det is a dict after asdict() conversion
                bbox = det['bbox'] if isinstance(det, dict) else det.bbox
                conf = det['confidence'] if isinstance(det, dict) else det.confidence
                cls = det['class_name'] if isinstance(det, dict) else det.class_name
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"{cls}: {conf:.2f}"
                cv2.putText(annotated, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw pose keypoints
        keypoint_color = CYAN
        for name, kp in diagnosis.pose_keypoints.items():
            # kp is a dict after asdict() conversion
            vis = kp['visibility'] if isinstance(kp, dict) else kp.visibility
            px = kp['pixel_x'] if isinstance(kp, dict) else kp.pixel_x
            py = kp['pixel_y'] if isinstance(kp, dict) else kp.pixel_y
            if vis > 0.3:
                cv2.circle(annotated, (px, py), 5, keypoint_color, -1)
                if name in ['nose', 'left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']:
                    cv2.putText(annotated, f"{name[:3]}:{vis:.2f}",
                               (px + 5, py),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, keypoint_color, 1)

        # Draw skeleton lines
        skeleton_pairs = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
        ]

        for p1, p2 in skeleton_pairs:
            if p1 in diagnosis.pose_keypoints and p2 in diagnosis.pose_keypoints:
                kp1 = diagnosis.pose_keypoints[p1]
                kp2 = diagnosis.pose_keypoints[p2]
                # Handle dict format
                vis1 = kp1['visibility'] if isinstance(kp1, dict) else kp1.visibility
                vis2 = kp2['visibility'] if isinstance(kp2, dict) else kp2.visibility
                px1 = kp1['pixel_x'] if isinstance(kp1, dict) else kp1.pixel_x
                py1 = kp1['pixel_y'] if isinstance(kp1, dict) else kp1.pixel_y
                px2 = kp2['pixel_x'] if isinstance(kp2, dict) else kp2.pixel_x
                py2 = kp2['pixel_y'] if isinstance(kp2, dict) else kp2.pixel_y
                if vis1 > 0.3 and vis2 > 0.3:
                    cv2.line(annotated, (px1, py1), (px2, py2), CYAN, 2)

        # Info panel at top
        panel_height = 200
        cv2.rectangle(annotated, (0, 0), (w, panel_height), (0, 0, 0), -1)

        y_offset = 20
        line_height = 18

        # Header
        header = f"[{violation['id']:02d}] {violation['activity'].upper()} @ {diagnosis.timestamp_str} ({diagnosis.position})"
        cv2.putText(annotated, header, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        y_offset += line_height + 5

        # Angles
        if diagnosis.angles.get('valid', False):
            yaw = diagnosis.angles['yaw']
            pitch = diagnosis.angles['pitch']
            angle_text = f"YAW: {yaw:.1f}deg  PITCH: {pitch:.1f}deg"
            cv2.putText(annotated, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        else:
            cv2.putText(annotated, f"Angles: INVALID ({diagnosis.angles.get('reason', 'unknown')})",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        y_offset += line_height

        # Detection status
        would_detect = diagnosis.diagnosis.get('would_detect', False)
        status_color = GREEN if would_detect else RED
        status_text = "WOULD DETECT" if would_detect else "WOULD NOT DETECT"
        cv2.putText(annotated, f"Status: {status_text}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        y_offset += line_height

        # Reason
        reason = diagnosis.diagnosis.get('reason', 'unknown')
        if len(reason) > 80:
            reason = reason[:77] + "..."
        cv2.putText(annotated, f"Reason: {reason}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        y_offset += line_height

        # Suppression info
        if 'suppression' in diagnosis.diagnosis and diagnosis.diagnosis['suppression']:
            supp_text = f"Suppression: {', '.join(diagnosis.diagnosis['suppression'])}"
            cv2.putText(annotated, supp_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)
            y_offset += line_height

        # Key threshold checks
        checks = diagnosis.threshold_checks
        if 'wrist_distance' in checks and isinstance(checks['wrist_distance'], (int, float)):
            cv2.putText(annotated, f"Wrist dist: {checks['wrist_distance']:.0f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
            y_offset += line_height

        # Recommendations
        recs = diagnosis.diagnosis.get('recommendations', [])
        if recs:
            cv2.putText(annotated, "Recommendations:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
            y_offset += line_height
            for rec in recs[:2]:  # Max 2 recommendations
                if len(rec) > 70:
                    rec = rec[:67] + "..."
                cv2.putText(annotated, f"  - {rec}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
                y_offset += line_height - 2

        return annotated

    def diagnose_violation(self, violation: Dict) -> List[FrameDiagnosis]:
        """Diagnose a single expected violation."""
        activity = violation['activity']
        start_sec = self.parse_timestamp(violation['start'])
        end_sec = self.parse_timestamp(violation['end'])
        duration = end_sec - start_sec

        # Determine sample points
        sample_points = []
        if duration <= 5:
            # Short duration: start and end
            sample_points = [
                (start_sec, 'start'),
                (end_sec, 'end'),
            ]
        else:
            # Longer duration: start, 25%, 50%, 75%, end
            sample_points = [
                (start_sec, 'start'),
                (start_sec + duration * 0.25, '25%'),
                (start_sec + duration * 0.5, '50%'),
                (start_sec + duration * 0.75, '75%'),
                (end_sec, 'end'),
            ]

        diagnoses = []

        for timestamp, position in sample_points:
            # Skip if timestamp is beyond video duration
            if timestamp > self.duration:
                continue

            frame = self.extract_frame(timestamp)
            if frame is None:
                continue

            frame_shape = frame.shape[:2]

            # Run detections
            yolo_detections = self.run_yolo_detection(frame)
            keypoints, all_persons = self.run_yolo_pose(frame)

            # Calculate angles
            angles = self.calculate_head_angles(keypoints, frame_shape)

            # Convert detections for serialization
            yolo_det_dict = {}
            for cls, dets in yolo_detections.items():
                yolo_det_dict[cls] = [asdict(d) for d in dets]

            # Convert keypoints for serialization
            keypoints_dict = {}
            for name, kp in keypoints.items():
                keypoints_dict[name] = asdict(kp)

            # Person detection check
            person_detected = 'person' in yolo_detections and len(yolo_detections['person']) > 0
            person_confidence = max([d.confidence for d in yolo_detections.get('person', [])], default=0)

            # Activity-specific diagnosis
            if activity == 'packing_bags':
                diagnosis_result = self.diagnose_packing_bags(yolo_detections, keypoints, frame_shape)
            elif activity == 'cell_phone':
                diagnosis_result = self.diagnose_cell_phone(yolo_detections, keypoints, frame_shape)
            elif activity == 'mind_diversion':
                diagnosis_result = self.diagnose_mind_diversion(angles, keypoints, yolo_detections, frame_shape)
            elif activity == 'writing':
                diagnosis_result = self.diagnose_writing(yolo_detections, keypoints, angles, frame_shape)
            elif activity == 'no_person_detected':
                diagnosis_result = self.diagnose_no_person(yolo_detections, all_persons)
            else:
                diagnosis_result = {'should_detect': True, 'would_detect': False, 'reason': 'unknown_activity'}

            # Create frame diagnosis
            frame_diag = FrameDiagnosis(
                timestamp=timestamp,
                timestamp_str=self.format_timestamp(timestamp),
                position=position,
                yolo_detections=yolo_det_dict,
                pose_keypoints=keypoints_dict,
                person_detected=person_detected,
                person_confidence=person_confidence,
                angles=angles,
                threshold_checks=diagnosis_result.get('checks', {}),
                diagnosis=diagnosis_result,
            )

            # Annotate and save frame
            annotated = self.annotate_frame(frame, frame_diag, violation)
            frame_filename = f"{violation['id']:02d}_{activity}_{self.format_timestamp(timestamp).replace(':', '')}_{position}.jpg"
            frame_path = os.path.join(self.run_dir, "frames", frame_filename)
            cv2.imwrite(frame_path, annotated)
            frame_diag.annotated_frame_path = frame_path

            # Save raw detections
            raw_filename = f"{violation['id']:02d}_{activity}_{self.format_timestamp(timestamp).replace(':', '')}_{position}.json"
            raw_path = os.path.join(self.run_dir, "raw_detections", raw_filename)
            with open(raw_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'yolo_detections': yolo_det_dict,
                    'pose_keypoints': keypoints_dict,
                    'angles': angles,
                    'diagnosis': diagnosis_result,
                }, f, indent=2, default=str)

            diagnoses.append(frame_diag)

        return diagnoses

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        lines = []

        lines.append("=" * 80)
        lines.append("           VIDEO ACTIVITY DETECTION DIAGNOSTIC REPORT")
        lines.append("=" * 80)
        lines.append(f"Video: {self.video_path}")
        lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Video Duration: {self.format_timestamp(self.duration)}")
        lines.append(f"Expected Violations: {len(EXPECTED_VIOLATIONS)}")
        lines.append("")

        # Current thresholds
        lines.append("-" * 80)
        lines.append("CURRENT DETECTION THRESHOLDS")
        lines.append("-" * 80)
        for activity, thresholds in self.thresholds.items():
            lines.append(f"\n{activity.upper()}:")
            for key, value in thresholds.items():
                lines.append(f"  {key}: {value}")
        lines.append("")

        # Per-violation analysis
        lines.append("=" * 80)
        lines.append("                    VIOLATION-BY-VIOLATION ANALYSIS")
        lines.append("=" * 80)

        # Statistics
        stats = defaultdict(lambda: {'expected': 0, 'would_detect': 0, 'missed': 0})
        failure_reasons = defaultdict(list)
        recommendations = set()

        for result in self.all_results:
            violation = result['violation']
            diagnoses = result['diagnoses']
            activity = violation['activity']

            stats[activity]['expected'] += 1

            # Check if any frame would be detected
            any_detected = any(d.diagnosis.get('would_detect', False) for d in diagnoses)

            if any_detected:
                stats[activity]['would_detect'] += 1
            else:
                stats[activity]['missed'] += 1

            lines.append("")
            lines.append(f"[{violation['id']:02d}/{len(EXPECTED_VIOLATIONS)}] {activity.upper()} @ {violation['start']} - {violation['end']}")
            lines.append("-" * 70)

            status = "WOULD DETECT" if any_detected else "MISSED"
            lines.append(f"Status: {status}")
            lines.append(f"Frames Analyzed: {len(diagnoses)}")
            lines.append("")

            for diag in diagnoses:
                would_det = diag.diagnosis.get('would_detect', False)
                symbol = "[+]" if would_det else "[-]"
                lines.append(f"  {symbol} Frame @ {diag.timestamp_str} ({diag.position}):")

                # Key metrics
                if diag.angles.get('valid', False):
                    lines.append(f"      Yaw: {diag.angles['yaw']:.1f}deg, Pitch: {diag.angles['pitch']:.1f}deg")
                else:
                    lines.append(f"      Angles: INVALID ({diag.angles.get('reason', 'unknown')})")

                lines.append(f"      Person: {'YES' if diag.person_detected else 'NO'} (conf: {diag.person_confidence:.2f})")

                reason = diag.diagnosis.get('reason', 'unknown')
                lines.append(f"      Result: {'DETECT' if would_det else 'FAIL'} - {reason}")

                if diag.diagnosis.get('suppression'):
                    lines.append(f"      Suppression: {', '.join(diag.diagnosis['suppression'])}")

                # Collect failure reasons and recommendations
                if not would_det:
                    failure_reasons[activity].append(reason)

                for rec in diag.diagnosis.get('recommendations', []):
                    recommendations.add(rec)

                lines.append("")

        # Summary statistics
        lines.append("=" * 80)
        lines.append("                           SUMMARY STATISTICS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"{'Activity Type':<25} {'Expected':>10} {'Detected':>10} {'Missed':>10} {'Rate':>10}")
        lines.append("-" * 65)

        total_expected = 0
        total_detected = 0

        for activity in ['packing_bags', 'cell_phone', 'mind_diversion', 'writing', 'no_person_detected']:
            s = stats[activity]
            rate = (s['would_detect'] / s['expected'] * 100) if s['expected'] > 0 else 0
            lines.append(f"{activity:<25} {s['expected']:>10} {s['would_detect']:>10} {s['missed']:>10} {rate:>9.0f}%")
            total_expected += s['expected']
            total_detected += s['would_detect']

        lines.append("-" * 65)
        total_rate = (total_detected / total_expected * 100) if total_expected > 0 else 0
        lines.append(f"{'TOTAL':<25} {total_expected:>10} {total_detected:>10} {total_expected - total_detected:>10} {total_rate:>9.0f}%")

        # Failure analysis
        lines.append("")
        lines.append("=" * 80)
        lines.append("                         FAILURE ROOT CAUSES")
        lines.append("=" * 80)

        for activity, reasons in failure_reasons.items():
            if reasons:
                lines.append(f"\n{activity.upper()}:")
                reason_counts = defaultdict(int)
                for r in reasons:
                    # Simplify reason for grouping
                    simple_reason = r.split(':')[0] if ':' in r else r
                    reason_counts[simple_reason] += 1

                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  - {reason}: {count} occurrence(s)")

        # Recommendations
        if recommendations:
            lines.append("")
            lines.append("=" * 80)
            lines.append("                          RECOMMENDATIONS")
            lines.append("=" * 80)
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Report saved to: {self.run_dir}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def run_full_diagnosis(self):
        """Run full diagnostic analysis on all expected violations."""
        print(f"\nStarting diagnostic analysis...")
        print(f"Output directory: {self.run_dir}")
        print(f"Analyzing {len(EXPECTED_VIOLATIONS)} expected violations...\n")

        for i, violation in enumerate(EXPECTED_VIOLATIONS):
            print(f"[{i+1}/{len(EXPECTED_VIOLATIONS)}] Analyzing {violation['activity']} @ {violation['start']} - {violation['end']}...")

            diagnoses = self.diagnose_violation(violation)

            self.all_results.append({
                'violation': violation,
                'diagnoses': diagnoses,
            })

            # Quick status
            any_detected = any(d.diagnosis.get('would_detect', False) for d in diagnoses)
            status = "WOULD DETECT" if any_detected else "MISSED"
            print(f"    -> {status}")

        # Generate and save report
        report = self.generate_summary_report()
        report_path = os.path.join(self.run_dir, "summary_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n{report}")
        print(f"\nDiagnostic complete!")
        print(f"Results saved to: {self.run_dir}")

        return self.all_results


def main():
    parser = argparse.ArgumentParser(description="Diagnose video activity detection failures")
    parser.add_argument("video_path", nargs="?",
                       default="/Users/satishvanga/Documents/poc/n_1.mp4",
                       help="Path to video file")
    parser.add_argument("--output-dir", "-o", default="diagnostic_output",
                       help="Output directory for diagnostic results")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    analyzer = DiagnosticVideoAnalyzer(args.video_path, args.output_dir)
    analyzer.run_full_diagnosis()


if __name__ == "__main__":
    main()
