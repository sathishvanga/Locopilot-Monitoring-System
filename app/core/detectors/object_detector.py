"""Object detection using YOLO with pose-guided ROI optimization.

This module extracts object detection logic from locopilot_monitor.py,
providing a reusable ObjectDetector class for detecting objects like
cell phones, books, backpacks, and other items using YOLO inference.

Detection Methods:
1. Full-frame detection for persons, bags, books
2. Pose-guided ROI detection for activity-related objects
3. Batched inference for GPU optimization

Performance Optimizations:
- ROI-based detection reduces false positives
- Batch inference maximizes GPU utilization
- Aspect ratio validation filters spurious detections
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import logging
import time as time_module
import numpy as np
import cv2


class ObjectDetector:
    """Detects objects using YOLO with pose-guided ROI optimization.

    This class provides comprehensive object detection through:
    - Full-frame YOLO inference for persons, bags, and proximity-based objects
    - Pose-guided ROI detection for hands, ears, and lap areas
    - Batched inference for multi-frame GPU optimization
    - Aspect ratio validation to filter false positives

    Attributes:
        yolo_model: YOLO model instance for object detection
        settings: Configuration settings object (optional)
        preprocessing_service: Image preprocessing service for dark frames (optional)
        logger: Logger instance for debug/info output
    """

    def __init__(
        self,
        yolo_model: Any,
        settings: Optional[Any] = None,
        preprocessing_service: Optional[Any] = None,
        get_keypoint_func: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
        yolo_imgsz: int = 640,
        yolo_device: Optional[str] = None,
        cell_phone_confidence: float = 0.45
    ):
        """Initialize the ObjectDetector.

        Args:
            yolo_model: YOLO model instance (from ultralytics)
            settings: Configuration settings object with detection thresholds.
                     If None, uses default values.
            preprocessing_service: ImagePreprocessingService for dark frame enhancement.
                                 If None, preprocessing is skipped.
            get_keypoint_func: Function to get keypoint from pose landmarks.
                              Signature: get_keypoint(landmarks, keypoint_name) -> landmark
            logger: Optional logger instance. If None, creates a new one.
            yolo_imgsz: YOLO input image size (default 640)
            yolo_device: YOLO inference device (None for auto)
            cell_phone_confidence: Confidence threshold for cell phone ROI detection
        """
        self.yolo_model = yolo_model
        self.settings = settings
        self.preprocessing_service = preprocessing_service
        self._get_keypoint = get_keypoint_func
        self.logger = logger or logging.getLogger(__name__)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_device = yolo_device
        self.cell_phone_confidence = cell_phone_confidence

        # Cache for avoiding redundant inference
        self._cached_frame_objects = None
        self._cached_frame_time = 0

        # Initialize thresholds from settings or defaults
        self._init_thresholds()

    def _init_thresholds(self) -> None:
        """Initialize detection thresholds from settings or use defaults."""
        s = self.settings

        # YOLO Confidence Thresholds
        self.YOLO_PERSON_CONFIDENCE = getattr(s, 'yolo_person_confidence', 0.5) if s else 0.5
        self.YOLO_BAG_CONFIDENCE = getattr(s, 'yolo_bag_confidence', 0.45) if s else 0.45
        self.YOLO_BAG_LOG_CONFIDENCE = getattr(s, 'yolo_bag_log_confidence', 0.25) if s else 0.25
        self.YOLO_BOOK_CONFIDENCE = getattr(s, 'yolo_book_confidence', 0.4) if s else 0.4
        self.YOLO_CELL_PHONE_CONFIDENCE = getattr(s, 'yolo_cell_phone_confidence', 0.3) if s else 0.3

        # Object Detection Geometry
        self.BAG_MAX_ASPECT_RATIO = getattr(s, 'bag_max_aspect_ratio', 1.2) if s else 1.2
        self.BAG_MIN_AREA = getattr(s, 'bag_min_area', 5000) if s else 5000
        self.BAG_MAX_AREA = getattr(s, 'bag_max_area', 100000) if s else 100000
        self.BOOK_PERSON_MARGIN = getattr(s, 'book_person_margin', 150) if s else 150
        self.PERSON_BOOK_OVERLAP_MARGIN = getattr(s, 'person_book_overlap_margin', 250) if s else 250

    def get_roi_around_keypoint(
        self,
        keypoint_coords: Any,
        frame_shape: Tuple[int, ...],
        roi_size: int = 150
    ) -> Optional[Tuple[int, int, int, int]]:
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

    def validate_object_aspect_ratio(self, bbox: List[int], object_class: str) -> bool:
        """Validate detected object based on aspect ratio to filter false positives.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            object_class: Class name ('cell phone', 'book', etc.)

        Returns:
            bool: True if aspect ratio is valid for the object class
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height == 0 or width == 0:
            return False

        aspect_ratio = width / height

        aspect_ratio_rules = {
            'cell phone': {
                'min_ratio': 0.4,   # Portrait: ~0.45 (9:20) - tightened from 0.3
                'max_ratio': 2.0,   # Landscape: ~1.78 (16:9)
                'min_size': 30      # Minimum dimension (pixels)
            },
            'book': {
                'min_ratio': 0.5,   # Tall book: ~0.7 (A4 portrait)
                'max_ratio': 2.5,   # Wide book: ~1.4 (A4 landscape)
                'min_size': 40
            }
        }

        if object_class not in aspect_ratio_rules:
            return True  # No validation for other objects

        rules = aspect_ratio_rules[object_class]

        # Check aspect ratio bounds
        if aspect_ratio < rules['min_ratio'] or aspect_ratio > rules['max_ratio']:
            return False

        # Check minimum size
        if min(width, height) < rules['min_size']:
            return False

        return True

    def detect_objects_in_roi(
        self,
        frame: Any,
        roi_bbox: Tuple[int, int, int, int],
        target_classes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection on a specific ROI region.

        Args:
            frame: Full frame
            roi_bbox: (x1, y1, x2, y2) ROI bounding box
            target_classes: List of class names to detect in ROI

        Returns:
            List of detections with global coordinates: [(class_name, conf, x1, y1, x2, y2), ...]
        """
        if target_classes is None:
            target_classes = ['cell phone', 'book', 'pen', 'pencil']

        if roi_bbox is None:
            return []

        x1, y1, x2, y2 = roi_bbox
        roi_frame = frame[y1:y2, x1:x2]

        # Run YOLO on ROI with strict confidence threshold
        results = self.yolo_model(
            roi_frame,
            verbose=False,
            conf=self.cell_phone_confidence,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device
        )

        detections = []
        debug_all_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy_local = box.xyxy[0].cpu().numpy()

                class_name = self.yolo_model.names[cls]
                debug_all_detections.append((class_name, conf))

                # Check if this is a target class
                if class_name in target_classes:
                    # Convert local ROI coordinates to global frame coordinates
                    global_x1 = xyxy_local[0] + x1
                    global_y1 = xyxy_local[1] + y1
                    global_x2 = xyxy_local[2] + x1
                    global_y2 = xyxy_local[3] + y1

                    # Validate aspect ratio before adding detection
                    bbox_for_validation = (global_x1, global_y1, global_x2, global_y2)
                    if self.validate_object_aspect_ratio(bbox_for_validation, class_name):
                        detections.append((class_name, conf, global_x1, global_y1, global_x2, global_y2))

        # DEBUG LOGGING
        if 'cell phone' in target_classes:
            if len(debug_all_detections) > 0:
                cell_phones = [d for d in debug_all_detections if d[0] == 'cell phone']
                if cell_phones:
                    self.logger.info(f"[DEBUG ROI] [OK] Found {len(cell_phones)} cell phone(s): {cell_phones}")
                else:
                    top_detections = sorted(debug_all_detections, key=lambda x: -x[1])[:5]
                    self.logger.info(f"[DEBUG ROI] [FAIL] No phone, but YOLO found {len(debug_all_detections)} objects: {top_detections}")
            else:
                self.logger.info(f"[DEBUG ROI] [WARN] YOLO detected NOTHING in this ROI")

        return detections

    def detect_objects_in_rois_batch(
        self,
        frame: Any,
        roi_bboxes: List[Tuple[int, int, int, int]],
        roi_names: List[str],
        target_classes: List[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """Run YOLO detection on multiple ROI regions in a single batched call.

        PERFORMANCE OPTIMIZATION: This method processes all ROIs in a single YOLO
        inference call instead of N sequential calls, achieving ~4x speedup.

        Args:
            frame: Full frame
            roi_bboxes: List of (x1, y1, x2, y2) ROI bounding boxes
            roi_names: List of ROI names corresponding to each bbox (for debugging)
            target_classes: List of class names to detect in ROIs

        Returns:
            List of lists: [[detections for ROI 1], [detections for ROI 2], ...]
            Each detection: (class_name, conf, x1, y1, x2, y2) with global coordinates
        """
        if target_classes is None:
            target_classes = ['cell phone', 'book', 'pen', 'pencil']

        if not roi_bboxes or len(roi_bboxes) == 0:
            return [[] for _ in range(len(roi_names))]

        # Step 1: Extract all ROI crops
        roi_frames = []
        valid_indices = []

        for idx, roi_bbox in enumerate(roi_bboxes):
            if roi_bbox is None:
                continue

            x1, y1, x2, y2 = roi_bbox
            roi_frame = frame[y1:y2, x1:x2]

            # Validate ROI dimensions
            if roi_frame.size == 0:
                continue

            roi_frames.append(roi_frame)
            valid_indices.append(idx)

        # Initialize results for all ROIs (including invalid ones)
        all_detections = [[] for _ in range(len(roi_bboxes))]

        if len(roi_frames) == 0:
            return all_detections

        # Step 2: Batch YOLO inference on all ROI crops
        batch_results = self.yolo_model(
            roi_frames,
            verbose=False,
            conf=self.cell_phone_confidence,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device
        )

        # Step 3: Process batch results and translate to global coordinates
        for batch_idx, (results_idx, roi_bbox_idx) in enumerate(zip(range(len(batch_results)), valid_indices)):
            roi_bbox = roi_bboxes[roi_bbox_idx]
            x1, y1, x2, y2 = roi_bbox

            detections = []
            debug_all_detections = []

            results = batch_results[batch_idx]
            boxes = results.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy_local = box.xyxy[0].cpu().numpy()

                class_name = self.yolo_model.names[cls]
                debug_all_detections.append((class_name, conf))

                # Check if this is a target class
                if class_name in target_classes:
                    # Convert local ROI coordinates to global frame coordinates
                    global_x1 = xyxy_local[0] + x1
                    global_y1 = xyxy_local[1] + y1
                    global_x2 = xyxy_local[2] + x1
                    global_y2 = xyxy_local[3] + y1

                    # Validate aspect ratio before adding detection
                    bbox_for_validation = (global_x1, global_y1, global_x2, global_y2)
                    if self.validate_object_aspect_ratio(bbox_for_validation, class_name):
                        detections.append((class_name, conf, global_x1, global_y1, global_x2, global_y2))

            # DEBUG LOGGING
            if 'cell phone' in target_classes:
                roi_name = roi_names[roi_bbox_idx]
                if len(debug_all_detections) > 0:
                    cell_phones = [d for d in debug_all_detections if d[0] == 'cell phone']
                    if cell_phones:
                        self.logger.info(f"[DEBUG ROI BATCH] {roi_name}: [OK] Found {len(cell_phones)} cell phone(s): {cell_phones}")
                    else:
                        top_detections = sorted(debug_all_detections, key=lambda x: -x[1])[:5]
                        self.logger.info(f"[DEBUG ROI BATCH] {roi_name}: [FAIL] No phone, found {len(debug_all_detections)} objects: {top_detections}")
                else:
                    self.logger.info(f"[DEBUG ROI BATCH] {roi_name}: [WARN] YOLO detected NOTHING")

            all_detections[roi_bbox_idx] = detections

        return all_detections

    def detect_objects(
        self,
        frame: Any,
        pose_landmarks: Any = None,
        use_pose_guided: bool = True
    ) -> Dict[str, List[Any]]:
        """Detect objects using YOLO with pose-guided detection.

        MULTI-LAYERED DETECTION FLOW:
        1. Full frame detection for persons, backpacks, books near persons
        2. ROI-based detection around pose landmarks for activity objects
        3. Aspect ratio validation filters false positives

        Args:
            frame: Input frame (BGR)
            pose_landmarks: Pose landmarks for ROI-based detection (optional)
            use_pose_guided: Enable pose-guided ROI detection (default True)

        Returns:
            Dictionary with detections and ROI information
        """
        # Stage 1: Full frame detection for person, backpack, and books near person
        results = self.yolo_model(
            frame,
            verbose=False,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device
        )

        # Cache results for potential reuse
        self._cached_frame_objects = results
        self._cached_frame_time = time_module.time()

        detections = {
            'person': [],
            'cell_phone': [],
            'book': [],
            'backpack': [],
            'cup_bottle': [],
            'roi_detections': [],
            'roi_boxes': []
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

                # Person detection
                if class_name == 'person' and conf > self.YOLO_PERSON_CONFIDENCE:
                    detections['person'].append(xyxy)
                    person_boxes.append(xyxy)

                # Bag detection with aspect ratio filtering
                elif class_name in ['backpack', 'handbag', 'suitcase']:
                    if conf > self.YOLO_BAG_LOG_CONFIDENCE:
                        self.logger.debug(f"BAG DETECTED: {class_name} conf={conf:.2f} bbox={xyxy}")
                    if conf > self.YOLO_BAG_CONFIDENCE:
                        bag_width = xyxy[2] - xyxy[0]
                        bag_height = xyxy[3] - xyxy[1]
                        aspect_ratio = bag_width / bag_height if bag_height > 0 else 999
                        bag_area = bag_width * bag_height

                        if aspect_ratio < self.BAG_MAX_ASPECT_RATIO and self.BAG_MIN_AREA < bag_area < self.BAG_MAX_AREA:
                            detections['backpack'].append(xyxy)
                            self.logger.info(f"BAG ADDED: {class_name} conf={conf:.2f} aspect={aspect_ratio:.2f} area={bag_area:.0f}")
                        else:
                            self.logger.debug(f"BAG REJECTED: {class_name} conf={conf:.2f} aspect={aspect_ratio:.2f} area={bag_area:.0f}")

                # Book detection with person proximity check
                elif class_name == 'book' and conf > self.YOLO_BOOK_CONFIDENCE:
                    if len(person_boxes) > 0:
                        book_near_person = False
                        book_center_x = (xyxy[0] + xyxy[2]) / 2
                        book_center_y = (xyxy[1] + xyxy[3]) / 2

                        for person_box in person_boxes:
                            person_x1, person_y1, person_x2, person_y2 = person_box
                            margin = self.BOOK_PERSON_MARGIN
                            if (person_x1 - margin <= book_center_x <= person_x2 + margin and
                                person_y1 - margin <= book_center_y <= person_y2 + margin):
                                book_near_person = True
                                break

                        if book_near_person and self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)
                    else:
                        if self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)

                # Cup/bottle detection for eating/drinking
                elif class_name in ['cup', 'bottle']:
                    cup_floor_conf = getattr(self.settings, 'eating_drinking_cup_floor_confidence', 0.20) if self.settings else 0.20
                    if conf > cup_floor_conf:
                        detections['cup_bottle'].append(xyxy)

        # Stage 2: Pose-guided ROI detection
        if use_pose_guided and pose_landmarks is not None and self._get_keypoint is not None:
            self._detect_pose_guided_rois(frame, pose_landmarks, detections)

        return detections

    def _detect_pose_guided_rois(
        self,
        frame: Any,
        pose_landmarks: Any,
        detections: Dict[str, List[Any]]
    ) -> None:
        """Run pose-guided ROI detection and add results to detections dict.

        Args:
            frame: Input frame (BGR)
            pose_landmarks: Pose landmarks for ROI-based detection
            detections: Detection dictionary to update in-place
        """
        h, w = frame.shape[:2]

        # Define keypoints of interest with ROI sizes
        keypoints_of_interest = [
            ('RIGHT_WRIST', 'right_wrist', 180),
            ('LEFT_WRIST', 'left_wrist', 180),
            ('RIGHT_INDEX', 'right_wrist', 180),
            ('LEFT_INDEX', 'left_wrist', 180),
            ('RIGHT_HIP', 'right_hip', 180),
            ('LEFT_HIP', 'left_hip', 180),
            ('RIGHT_EAR', 'right_ear', 180),
            ('LEFT_EAR', 'left_ear', 180),
        ]

        # Collect all ROIs
        roi_bboxes = []
        roi_names = []

        for display_name, keypoint_name, roi_size in keypoints_of_interest:
            try:
                landmark = self._get_keypoint(pose_landmarks, keypoint_name)

                if landmark.visibility < 0.5:
                    roi_bboxes.append(None)
                    roi_names.append(display_name)
                    continue

                keypoint_coords = (int(landmark.x * w), int(landmark.y * h))
                roi_bbox = self.get_roi_around_keypoint(keypoint_coords, frame.shape, roi_size)

                roi_bboxes.append(roi_bbox)
                roi_names.append(display_name)

                if roi_bbox is not None:
                    detections['roi_boxes'].append((display_name, roi_bbox))

                    if display_name in ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP', 'NOSE']:
                        self.logger.info(f"[DEBUG ROI] Creating {display_name} ROI: size={roi_size}px, coords={keypoint_coords}")

            except Exception as e:
                self.logger.debug(f"Exception in _detect_pose_guided_rois: {e}")
                roi_bboxes.append(None)
                roi_names.append(display_name)
                continue

        # Batch process all ROIs
        valid_roi_count = sum(1 for bbox in roi_bboxes if bbox is not None)

        if valid_roi_count > 0:
            target_classes = ['cell phone', 'book', 'pen', 'pencil', 'paper', 'bottle', 'cup']
            batch_detections = self.detect_objects_in_rois_batch(frame, roi_bboxes, roi_names, target_classes)

            hand_related_keypoints = ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_EAR', 'LEFT_EAR']

            for idx, (keypoint_name, roi_dets) in enumerate(zip(roi_names, batch_detections)):
                for det in roi_dets:
                    class_name, conf, x1, y1, x2, y2 = det
                    detections['roi_detections'].append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'keypoint': keypoint_name,
                        'source': 'pose_guided_roi_batch'
                    })

                    if class_name == 'cell phone':
                        if keypoint_name in hand_related_keypoints:
                            detections['cell_phone'].append([x1, y1, x2, y2])
                    elif class_name == 'book':
                        detections['book'].append([x1, y1, x2, y2])

    def detect_objects_person_rois(
        self,
        frame: Any,
        pose_landmarks: Any
    ) -> Dict[str, List[Any]]:
        """Run ONLY pose-guided ROI detection for a single person (Stage 2 only).

        CR-006 OPTIMIZATION: This method performs only the ROI-based detection
        around a person's keypoints, WITHOUT re-running full-frame YOLO inference.

        Args:
            frame: Input frame (BGR)
            pose_landmarks: Pose landmarks for this person

        Returns:
            Dictionary with 'cell_phone', 'book', 'roi_detections', 'roi_boxes' keys
        """
        person_roi_detections = {
            'cell_phone': [],
            'book': [],
            'roi_detections': [],
            'roi_boxes': []
        }

        if pose_landmarks is None or self._get_keypoint is None:
            return person_roi_detections

        h, w = frame.shape[:2]

        # Define keypoints of interest with ROI sizes
        keypoints_of_interest = [
            ('RIGHT_WRIST', 'right_wrist', 180),
            ('LEFT_WRIST', 'left_wrist', 180),
            ('RIGHT_INDEX', 'right_wrist', 180),
            ('LEFT_INDEX', 'left_wrist', 180),
            ('RIGHT_HIP', 'right_hip', 180),
            ('LEFT_HIP', 'left_hip', 180),
            ('RIGHT_EAR', 'right_ear', 180),
            ('LEFT_EAR', 'left_ear', 180),
        ]

        roi_bboxes = []
        roi_names = []

        for display_name, keypoint_name, roi_size in keypoints_of_interest:
            try:
                landmark = self._get_keypoint(pose_landmarks, keypoint_name)

                if landmark.visibility < 0.5:
                    roi_bboxes.append(None)
                    roi_names.append(display_name)
                    continue

                keypoint_coords = (int(landmark.x * w), int(landmark.y * h))
                roi_bbox = self.get_roi_around_keypoint(keypoint_coords, frame.shape, roi_size)

                roi_bboxes.append(roi_bbox)
                roi_names.append(display_name)

                if roi_bbox is not None:
                    person_roi_detections['roi_boxes'].append((display_name, roi_bbox))

                    if display_name in ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP', 'NOSE']:
                        self.logger.info(f"[DEBUG ROI] Creating {display_name} ROI: size={roi_size}px, coords={keypoint_coords}")

            except Exception as e:
                self.logger.debug(f"Exception in detect_objects_person_rois: {e}")
                roi_bboxes.append(None)
                roi_names.append(display_name)
                continue

        valid_roi_count = sum(1 for bbox in roi_bboxes if bbox is not None)

        if valid_roi_count > 0:
            target_classes = ['cell phone', 'book', 'pen', 'pencil', 'paper', 'bottle', 'cup']
            batch_detections = self.detect_objects_in_rois_batch(frame, roi_bboxes, roi_names, target_classes)

            hand_related_keypoints = ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_EAR', 'LEFT_EAR']

            for idx, (keypoint_name, roi_dets) in enumerate(zip(roi_names, batch_detections)):
                for det in roi_dets:
                    class_name, conf, x1, y1, x2, y2 = det
                    person_roi_detections['roi_detections'].append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'keypoint': keypoint_name,
                        'source': 'pose_guided_roi_batch'
                    })

                    if class_name == 'cell phone':
                        if keypoint_name in hand_related_keypoints:
                            person_roi_detections['cell_phone'].append([x1, y1, x2, y2])
                    elif class_name == 'book':
                        person_roi_detections['book'].append([x1, y1, x2, y2])

        return person_roi_detections

    def _preprocess_frames_for_detection(self, frames: List[Any]) -> List[Any]:
        """Preprocess dark/IR frames to improve YOLO person detection.

        Uses adaptive brightness check on a sample frame. If dark (brightness < threshold),
        applies CLAHE + gamma correction to all frames in the batch.

        Args:
            frames: List of BGR frames

        Returns:
            List of frames (preprocessed copies if dark, originals if not)
        """
        if not self.preprocessing_service or not frames:
            return frames

        # Quick brightness check on first frame
        sample = frames[0]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if len(sample.shape) == 3 else sample
        brightness = float(np.mean(gray)) / 255.0

        threshold = getattr(self.settings, 'yolo_dark_frame_brightness_threshold', 0.4) if self.settings else 0.4
        if brightness >= threshold:
            return frames  # Well-lit, no preprocessing needed

        self.logger.debug(f"[IR PREPROCESS] Dark frames detected (brightness={brightness:.2f}), applying CLAHE+gamma for YOLO")

        try:
            enhanced = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self.preprocessing_service.preprocess_for_mediapipe(rgb)
                bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                enhanced.append(bgr)
            return enhanced
        except Exception as e:
            self.logger.warning(f"[IR PREPROCESS] Preprocessing failed, using original frames: {e}")
            return frames

    def detect_objects_batch(
        self,
        frames: List[Any],
        batch_size: int = 8
    ) -> List[Dict[str, List[Any]]]:
        """Run YOLO object detection on multiple frames in a single batch.

        This maximizes GPU utilization by processing multiple frames at once.

        Args:
            frames: List of BGR frames (numpy arrays)
            batch_size: Maximum batch size for inference (default 8)

        Returns:
            List of detection dictionaries, one per frame
        """
        if not frames:
            return []

        self.logger.debug(f"[GPU BATCH] detect_objects_batch: {len(frames)} frames, batch_size={batch_size}")
        all_detections = []

        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            try:
                batch_results = self.yolo_model(
                    batch_frames,
                    verbose=False,
                    imgsz=self.yolo_imgsz,
                    device=self.yolo_device
                )
            except Exception as e:
                self.logger.error(f"[GPU BATCH] Object detection failed for batch starting at {batch_start}: {e}")
                for _ in batch_frames:
                    all_detections.append({
                        'person': [], 'cell_phone': [], 'book': [],
                        'backpack': [], 'roi_detections': [], 'roi_boxes': []
                    })
                continue

            # Process results for each frame in batch
            for frame_idx, (frame, results) in enumerate(zip(batch_frames, batch_results)):
                detections = {
                    'person': [],
                    'cell_phone': [],
                    'book': [],
                    'backpack': [],
                    'roi_detections': [],
                    'roi_boxes': []
                }

                person_boxes = []
                pending_books = []

                if results.boxes is not None:
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        class_name = self.yolo_model.names[cls]

                        if class_name == 'person' and conf > self.YOLO_PERSON_CONFIDENCE:
                            detections['person'].append(xyxy)
                            person_boxes.append(xyxy)
                        elif class_name in ['backpack', 'handbag', 'suitcase']:
                            if conf > self.YOLO_BAG_CONFIDENCE:
                                bag_width = xyxy[2] - xyxy[0]
                                bag_height = xyxy[3] - xyxy[1]
                                aspect_ratio = bag_width / bag_height if bag_height > 0 else 999
                                bag_area = bag_width * bag_height

                                if aspect_ratio < self.BAG_MAX_ASPECT_RATIO and self.BAG_MIN_AREA < bag_area < self.BAG_MAX_AREA:
                                    detections['backpack'].append(xyxy)
                        elif class_name == 'book' and conf > self.YOLO_BOOK_CONFIDENCE:
                            pending_books.append(xyxy)
                        elif class_name == 'cell phone' and conf > self.YOLO_CELL_PHONE_CONFIDENCE:
                            detections['cell_phone'].append(xyxy)

                # Process pending books - check proximity to persons
                for book_xyxy in pending_books:
                    if len(person_boxes) > 0:
                        for person_box in person_boxes:
                            if self._boxes_overlap_or_near(book_xyxy, person_box, margin=200):
                                detections['book'].append(book_xyxy)
                                break
                    else:
                        detections['book'].append(book_xyxy)

                all_detections.append(detections)

        self.logger.debug(f"[GPU BATCH] detect_objects_batch complete: {len(all_detections)} results")
        return all_detections

    def _boxes_overlap_or_near(
        self,
        box1: Any,
        box2: Any,
        margin: int = 100
    ) -> bool:
        """Check if two boxes overlap or are within margin pixels of each other.

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            margin: Margin in pixels for proximity check

        Returns:
            bool: True if boxes overlap or are within margin
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Expand box2 by margin
        x2_min_expanded = x2_min - margin
        y2_min_expanded = y2_min - margin
        x2_max_expanded = x2_max + margin
        y2_max_expanded = y2_max + margin

        # Check for overlap with expanded box
        return not (x1_max < x2_min_expanded or x1_min > x2_max_expanded or
                   y1_max < y2_min_expanded or y1_min > y2_max_expanded)
