"""YOLO model handler for object and pose detection.

This module encapsulates all YOLO-related operations including:
- Object detection (persons, cell phones, books, bags, etc.)
- Pose detection with keypoint extraction
- ROI-based detection around body keypoints
- Batch processing for GPU optimization
- Frame preprocessing for dark/IR conditions

Extracted from locopilot_monitor.py for modularity and reusability.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# YOLO Keypoint indices (COCO format) - shared across detection methods
YOLO_KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# YOLO pose keypoint index groups (shared across detection methods)
YOLO_HEAD_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
YOLO_BODY_INDICES = [5, 6, 7, 8, 11, 12]  # left/right shoulders, elbows, hips
YOLO_MIN_KEYPOINTS = 13  # Minimum landmarks required (indices 0-12)


def _setup_module_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Setup a file-only logger for the module."""
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, "LocopilotMonitoring.log"))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d [N/A] [N/A] [N/A] [N/A] [%(levelname)s] [%(name)s] [N/A N/A] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class YOLOHandler:
    """Handles YOLO model inference for object and pose detection.

    This class encapsulates all YOLO-related operations including model loading,
    object detection, pose detection, and batch processing for GPU optimization.

    Attributes:
        KEYPOINT_INDICES: Mapping of keypoint names to COCO indices
        HEAD_INDICES: Indices for head-related keypoints
        BODY_INDICES: Indices for body-related keypoints
        MIN_KEYPOINTS: Minimum required keypoints for valid pose
    """

    # Class-level keypoint constants
    KEYPOINT_INDICES = YOLO_KEYPOINT_INDICES
    HEAD_INDICES = YOLO_HEAD_INDICES
    BODY_INDICES = YOLO_BODY_INDICES
    MIN_KEYPOINTS = YOLO_MIN_KEYPOINTS

    def __init__(
        self,
        object_model_path: str = 'yolo26n.pt',
        pose_model_path: str = 'yolo26n-pose.pt',
        device: str = 'cpu',
        imgsz: int = 640,
        preloaded_models: Optional[Dict[str, Any]] = None,
        preprocessing_service: Optional[Any] = None,
        settings: Optional[Any] = None
    ) -> None:
        """Initialize YOLO handler with object and pose detection models.

        Args:
            object_model_path: Path to YOLO object detection weights
            pose_model_path: Path to YOLO pose detection weights
            device: Device for inference ('cpu', 'cuda', 'mps', etc.)
            imgsz: Input image size for inference
            preloaded_models: Optional dict with pre-loaded models:
                - 'yolo': Pre-loaded YOLO object detection model
                - 'yolo_pose': Pre-loaded YoloPoseAdapter instance
            preprocessing_service: Optional ImagePreprocessingService for dark/IR frames
            settings: Optional settings object for configuration thresholds
        """
        self.logger = _setup_module_logger('YOLOHandler')
        self.device = device
        self.imgsz = imgsz
        self.settings = settings
        self.preprocessing_service = preprocessing_service

        # Track if models were preloaded (for cleanup)
        self._models_preloaded = preloaded_models is not None

        # Load or use preloaded models
        if preloaded_models is not None:
            self.object_model = preloaded_models.get('yolo')
            self.pose_adapter = preloaded_models.get('yolo_pose')

            if self.object_model is None or self.pose_adapter is None:
                raise ValueError("preloaded_models must contain 'yolo' and 'yolo_pose'")

            self.logger.info("Using preloaded YOLO models")
        else:
            self._load_models(object_model_path, pose_model_path)

        # Always-preprocess flag (bypass brightness check)
        self.always_preprocess = getattr(settings, 'yolo_always_preprocess', False) if settings else False

        # Zone suppression settings
        self._configure_zone_suppression(settings)


        # Configure confidence thresholds from settings
        self._configure_thresholds(settings)

        # Cache for frame results (avoid redundant inference)
        self._cached_frame_objects = None
        self._cached_frame_time = 0.0

    def _load_models(self, object_model_path: str, pose_model_path: str) -> None:
        """Load YOLO models from disk.

        Args:
            object_model_path: Path to object detection model weights
            pose_model_path: Path to pose detection model weights
        """
        from ultralytics import YOLO

        self.logger.info(f"Loading YOLO object model: {object_model_path}")
        self.object_model = YOLO(object_model_path)

        # Fuse Conv+BatchNorm layers for faster inference (15-20% speedup)
        if hasattr(self.object_model.model, 'fuse'):
            self.object_model.fuse()
            self.logger.info("YOLO model layers fused for optimized inference")

        pose_conf = self.settings.yolo_pose_confidence if self.settings else 0.45
        pose_backend = getattr(self.settings, 'pose_model_backend', 'yolo') if self.settings else 'yolo'

        if pose_backend == 'rtmpose':
            from app.services.rtmpose_adapter import RTMPoseAdapter
            rtm_mode = getattr(self.settings, 'rtmpose_mode', 'balanced') if self.settings else 'balanced'
            rtm_backend = getattr(self.settings, 'rtmpose_backend', 'onnxruntime') if self.settings else 'onnxruntime'
            yolo_device = str(self.device) if self.device else 'cpu'
            rtm_device = 'cuda' if yolo_device not in ('cpu', '') else 'cpu'
            self.logger.info(f"Loading RTMPose: mode={rtm_mode}, backend={rtm_backend}, device={rtm_device}")
            self.pose_adapter = RTMPoseAdapter(
                conf_threshold=pose_conf, device=rtm_device,
                mode=rtm_mode, backend=rtm_backend,
            )
        else:
            from app.services.yolo_pose_adapter import YoloPoseAdapter
            self.logger.info(f"Loading YOLO pose model: {pose_model_path}")
            self.pose_adapter = YoloPoseAdapter(model_path=pose_model_path, conf_threshold=pose_conf)

    def _configure_thresholds(self, settings: Optional[Any]) -> None:
        """Configure detection confidence thresholds from settings.

        Args:
            settings: Settings object with threshold configurations
        """
        # YOLO Confidence Thresholds
        self.person_confidence = settings.yolo_person_confidence if settings else 0.5
        self.bag_confidence = settings.yolo_bag_confidence if settings else 0.45
        self.bag_log_confidence = settings.yolo_bag_log_confidence if settings else 0.25
        self.book_confidence = settings.yolo_book_confidence if settings else 0.4
        self.cell_phone_confidence = settings.yolo_cell_phone_confidence if settings else 0.3
        self.pose_sleep_confidence = settings.yolo_pose_sleep_confidence if settings else 0.30

        # Object Detection Geometry
        # NOTE: 1.5 allows taller/narrower bags while still filtering non-bag shapes.
        self.bag_max_aspect_ratio = settings.bag_max_aspect_ratio if settings else 1.5
        self.bag_min_area = settings.bag_min_area if settings else 5000
        self.bag_max_area = settings.bag_max_area if settings else 100000
        self.book_person_margin = settings.book_person_margin if settings else 150

        # Cell phone confidence for ROI detection (sourced from Settings)
        self.roi_confidence = settings.cell_phone_confidence if settings else 0.40

        # Dark frame preprocessing threshold
        self.dark_frame_brightness_threshold = (
            getattr(settings, 'yolo_dark_frame_brightness_threshold', 0.4) if settings else 0.4
        )

    def _configure_zone_suppression(self, settings: Optional[Any]) -> None:
        """Configure static zone suppression for fixed-camera FP filtering.

        Args:
            settings: Settings object with zone suppression configuration
        """
        self.zone_suppression_enabled = getattr(settings, 'zone_suppression_enabled', False) if settings else False
        self.suppressed_classes = set()
        self.suitcase_suppression_zones = []

        if not self.zone_suppression_enabled:
            return

        # Parse comma-separated suppressed class names
        suppress_str = getattr(settings, 'zone_suppress_classes', '') if settings else ''
        if suppress_str:
            self.suppressed_classes = {c.strip() for c in suppress_str.split(',') if c.strip()}

        # Parse suitcase suppression zones (JSON list of [x1,y1,x2,y2] normalized coords)
        zones_str = getattr(settings, 'zone_suppress_suitcase_regions', '') if settings else ''
        if zones_str:
            try:
                import json
                parsed = json.loads(zones_str)
                if isinstance(parsed, list):
                    for zone in parsed:
                        if isinstance(zone, list) and len(zone) == 4:
                            self.suitcase_suppression_zones.append(tuple(zone))
            except (ValueError, TypeError):
                self.logger.warning(f"[ZONE SUPPRESS] Failed to parse suitcase zones: {zones_str}")

        if self.suppressed_classes or self.suitcase_suppression_zones:
            self.logger.info(
                f"[ZONE SUPPRESS] Enabled: suppress_classes={self.suppressed_classes}, "
                f"suitcase_zones={len(self.suitcase_suppression_zones)}"
            )

    def _is_in_suppression_zone(
        self,
        xyxy: np.ndarray,
        frame_shape: Tuple[int, ...],
        zones: List[Tuple[float, float, float, float]]
    ) -> bool:
        """Check if detection center falls within any suppression zone.

        Uses center-point check (not IoU/overlap) for simplicity and speed.
        For fixed-camera setups, configure zones slightly larger than the
        expected FP region to account for bbox size variation.

        Args:
            xyxy: Bounding box [x1, y1, x2, y2] in pixel coords
            frame_shape: Frame (height, width) for normalization
            zones: List of (x1, y1, x2, y2) in normalized [0,1] coords

        Returns:
            True if detection center is inside any zone
        """
        if not zones:
            return False

        h, w = frame_shape[:2]
        cx = ((xyxy[0] + xyxy[2]) / 2.0) / w
        cy = ((xyxy[1] + xyxy[3]) / 2.0) / h

        for zx1, zy1, zx2, zy2 in zones:
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return True
        return False

    def _is_class_suppressed(self, class_name: str, xyxy: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
        """Check if a detection should be suppressed by zone rules.

        Args:
            class_name: YOLO class name
            xyxy: Bounding box [x1, y1, x2, y2]
            frame_shape: Frame (height, width)

        Returns:
            True if detection should be suppressed
        """
        if not self.zone_suppression_enabled:
            return False

        # Always-suppress classes (e.g., chair)
        if class_name in self.suppressed_classes:
            return True

        # Suitcase zone suppression
        if class_name == 'suitcase' and self.suitcase_suppression_zones:
            return self._is_in_suppression_zone(xyxy, frame_shape, self.suitcase_suppression_zones)

        return False

    def detect_objects(
        self,
        frame: np.ndarray,
        pose_landmarks: Optional[Any] = None,
        use_pose_guided: bool = True,
        get_keypoint_func: Optional[callable] = None,
        get_roi_func: Optional[callable] = None
    ) -> Dict[str, List[Any]]:
        """Detect objects using YOLO with optional pose-guided ROI detection.

        Multi-layered detection flow:
        1. Full frame detection for persons, backpacks, and books near person
        2. ROI-based detection around pose landmarks for activity objects
        3. Aspect ratio validation to filter false positives

        Args:
            frame: Input BGR frame (numpy array)
            pose_landmarks: Optional pose landmarks for ROI-guided detection
            use_pose_guided: Whether to enable pose-guided ROI detection
            get_keypoint_func: Function to get keypoint from landmarks by name
            get_roi_func: Function to create ROI around keypoint coordinates

        Returns:
            Dictionary with detection results:
                - 'person': List of person bounding boxes
                - 'cell_phone': List of cell phone bounding boxes
                - 'book': List of book bounding boxes
                - 'backpack': List of backpack bounding boxes
                - 'cup_bottle': List of cup/bottle bounding boxes
                - 'roi_detections': List of ROI-based detection details
                - 'roi_boxes': List of ROI boxes for visualization
        """
        # Stage 1: Full frame detection
        # Use the lowest per-class confidence as a pre-filter to reduce
        # the number of boxes YOLO returns. Per-class thresholds are
        # still enforced below so this only removes clearly spurious boxes.
        min_conf = min(
            self.person_confidence,
            self.bag_log_confidence,
            self.book_confidence,
            self.cell_phone_confidence,
            getattr(self, 'eating_drinking_cup_floor_confidence',
                    getattr(self.settings, 'eating_drinking_cup_floor_confidence', 0.20)
                    if self.settings else 0.20),
        )
        results = self.object_model(
            frame,
            verbose=False,
            conf=min_conf,
            imgsz=self.imgsz,
            device=self.device
        )

        # Cache results for potential reuse (100ms TTL)
        self._cached_frame_objects = results
        self._cached_frame_time = time.time()

        detections = {
            'person': [],
            'cell_phone': [],
            'book': [],
            'backpack': [],
            'cup_bottle': [],
            'roi_detections': [],
            'roi_boxes': []
        }

        person_boxes = []
        frame_shape = frame.shape
        _raw_confs = []  # Collect (class_name, conf) for detection summary log

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = self.object_model.names[cls].replace('_', ' ')  # Normalize: cell_phone -> cell phone
                _raw_confs.append((class_name, conf))

                # Zone suppression: skip suppressed classes/regions
                if self._is_class_suppressed(class_name, xyxy, frame_shape):
                    self.logger.debug(f"[ZONE SUPPRESS] Suppressed {class_name} conf={conf:.2f}")
                    continue

                # Person detection
                if class_name == 'person' and conf > self.person_confidence:
                    detections['person'].append(xyxy)
                    person_boxes.append(xyxy)

                # Bag detection (backpack, handbag, suitcase)
                elif class_name in ['backpack', 'suitcase']:
                    if conf > self.bag_log_confidence:
                        self.logger.debug(f"BAG DETECTED: {class_name} conf={conf:.2f} bbox={xyxy}")
                    if conf > self.bag_confidence:
                        if self._validate_bag_detection(xyxy):
                            detections['backpack'].append(xyxy)
                            self.logger.info(
                                f"BAG ADDED: {class_name} conf={conf:.2f} "
                                f"aspect={self._get_aspect_ratio(xyxy):.2f} "
                                f"area={self._get_area(xyxy):.0f}"
                            )
                        else:
                            self.logger.debug(
                                f"BAG REJECTED: {class_name} conf={conf:.2f} "
                                f"aspect={self._get_aspect_ratio(xyxy):.2f} "
                                f"area={self._get_area(xyxy):.0f} (filtered)"
                            )

                # Book detection (with person proximity check)
                elif class_name == 'book' and conf > self.book_confidence:
                    if self._is_book_near_person(xyxy, person_boxes):
                        if self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)
                    elif len(person_boxes) == 0:
                        # Fallback: add book anyway if no persons detected
                        if self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)

                # Cup/bottle detection for eating/drinking
                elif class_name == 'bottle':
                    floor_conf = getattr(self.settings, 'eating_drinking_cup_floor_confidence', 0.20) if self.settings else 0.20
                    if conf > floor_conf:
                        detections['cup_bottle'].append(xyxy)

        # Stage 2: Pose-guided ROI detection
        if use_pose_guided and pose_landmarks is not None and get_keypoint_func and get_roi_func:
            self._detect_objects_in_pose_rois(
                frame, pose_landmarks, detections,
                get_keypoint_func, get_roi_func
            )

        # Log per-frame detection summary
        self._log_detection_summary(detections, _raw_confs)

        return detections

    def _detect_objects_in_pose_rois(
        self,
        frame: np.ndarray,
        pose_landmarks: Any,
        detections: Dict[str, List],
        get_keypoint_func: callable,
        get_roi_func: callable
    ) -> None:
        """Detect objects in ROIs around pose keypoints.

        Args:
            frame: Input BGR frame
            pose_landmarks: Pose landmarks object
            detections: Detections dictionary to update
            get_keypoint_func: Function to get keypoint from landmarks by name
            get_roi_func: Function to create ROI around keypoint coordinates
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

        roi_bboxes = []
        roi_names = []

        for display_name, keypoint_name, roi_size in keypoints_of_interest:
            try:
                landmark = get_keypoint_func(pose_landmarks, keypoint_name)

                if landmark.visibility < 0.5:
                    roi_bboxes.append(None)
                    roi_names.append(display_name)
                    continue

                keypoint_coords = (int(landmark.x * w), int(landmark.y * h))
                roi_bbox = get_roi_func(keypoint_coords, frame.shape, roi_size)

                roi_bboxes.append(roi_bbox)
                roi_names.append(display_name)

                if roi_bbox is not None:
                    detections['roi_boxes'].append((display_name, roi_bbox))

                    if display_name in ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP', 'NOSE']:
                        self.logger.debug(
                            f"[DEBUG ROI] Creating {display_name} ROI: "
                            f"size={roi_size}px, coords={keypoint_coords}"
                        )

            except Exception as e:
                self.logger.debug(f"Exception in detect_objects_in_pose_rois: {e}")
                roi_bboxes.append(None)
                roi_names.append(display_name)
                continue

        # Batch process all ROIs
        valid_roi_count = sum(1 for bbox in roi_bboxes if bbox is not None)

        if valid_roi_count > 0:
            target_classes = ['cell phone', 'book', 'bottle']
            batch_detections = self.detect_objects_in_rois_batch(
                frame, roi_bboxes, roi_names, target_classes
            )

            self._process_batch_roi_detections(
                batch_detections, roi_names, detections
            )

    def _process_batch_roi_detections(
        self,
        batch_detections: List[List],
        roi_names: List[str],
        detections: Dict[str, List]
    ) -> None:
        """Process batch ROI detection results and update detections dict.

        Args:
            batch_detections: List of detection lists from batch processing
            roi_names: List of ROI names
            detections: Detections dictionary to update
        """
        hand_related_keypoints = [
            'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_INDEX', 'LEFT_INDEX',
            'RIGHT_EAR', 'LEFT_EAR'
        ]

        for keypoint_name, roi_dets in zip(roi_names, batch_detections):
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
                    # Only add if detected near hands/ears
                    if keypoint_name in hand_related_keypoints:
                        detections['cell_phone'].append([x1, y1, x2, y2])
                elif class_name == 'book':
                    detections['book'].append([x1, y1, x2, y2])

    def detect_objects_person_rois(
        self,
        frame: np.ndarray,
        pose_landmarks: Any,
        get_keypoint_func: callable,
        get_roi_func: callable
    ) -> Dict[str, List]:
        """Run pose-guided ROI detection for a single person (Stage 2 only).

        This method performs only ROI-based detection around a person's keypoints,
        without re-running full-frame YOLO inference.

        Args:
            frame: Input BGR frame
            pose_landmarks: Pose landmarks for this person
            get_keypoint_func: Function to get keypoint from landmarks by name
            get_roi_func: Function to create ROI around keypoint coordinates

        Returns:
            Dictionary with 'cell_phone', 'book', 'roi_detections', 'roi_boxes' keys
        """
        person_roi_detections = {
            'cell_phone': [],
            'book': [],
            'roi_detections': [],
            'roi_boxes': []
        }

        if pose_landmarks is None:
            return person_roi_detections

        h, w = frame.shape[:2]

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
                landmark = get_keypoint_func(pose_landmarks, keypoint_name)

                if landmark.visibility < 0.5:
                    roi_bboxes.append(None)
                    roi_names.append(display_name)
                    continue

                keypoint_coords = (int(landmark.x * w), int(landmark.y * h))
                roi_bbox = get_roi_func(keypoint_coords, frame.shape, roi_size)

                roi_bboxes.append(roi_bbox)
                roi_names.append(display_name)

                if roi_bbox is not None:
                    person_roi_detections['roi_boxes'].append((display_name, roi_bbox))

                    if display_name in ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP', 'NOSE']:
                        self.logger.debug(
                            f"[DEBUG ROI] Creating {display_name} ROI: "
                            f"size={roi_size}px, coords={keypoint_coords}"
                        )

            except Exception as e:
                self.logger.debug(f"Exception in detect_objects_person_rois: {e}")
                roi_bboxes.append(None)
                roi_names.append(display_name)
                continue

        valid_roi_count = sum(1 for bbox in roi_bboxes if bbox is not None)

        if valid_roi_count > 0:
            target_classes = ['cell phone', 'book', 'bottle']
            batch_detections = self.detect_objects_in_rois_batch(
                frame, roi_bboxes, roi_names, target_classes
            )

            self._process_batch_roi_detections(
                batch_detections, roi_names, person_roi_detections
            )

        return person_roi_detections

    def detect_objects_in_rois_batch(
        self,
        frame: np.ndarray,
        roi_bboxes: List[Optional[Tuple[int, int, int, int]]],
        roi_names: List[str],
        target_classes: List[str] = None
    ) -> List[List[Tuple]]:
        """Run YOLO detection on multiple ROI regions in a single batched call.

        Performance optimization: Processes all ROIs in a single YOLO inference call
        instead of N sequential calls, achieving ~4x speedup.

        Args:
            frame: Full BGR frame
            roi_bboxes: List of (x1, y1, x2, y2) ROI bounding boxes (None for invalid)
            roi_names: List of ROI names for debugging
            target_classes: List of class names to detect in ROIs

        Returns:
            List of detection lists, one per ROI.
            Each detection: (class_name, conf, x1, y1, x2, y2) with global coordinates
        """
        if target_classes is None:
            target_classes = ['cell phone', 'book', 'bottle']

        if not roi_bboxes or len(roi_bboxes) == 0:
            return [[] for _ in range(len(roi_names))]

        # Extract valid ROI crops
        roi_frames = []
        valid_indices = []

        for idx, roi_bbox in enumerate(roi_bboxes):
            if roi_bbox is None:
                continue

            x1, y1, x2, y2 = roi_bbox
            roi_frame = frame[y1:y2, x1:x2]

            if roi_frame.size == 0:
                continue

            roi_frames.append(roi_frame)
            valid_indices.append(idx)

        # Initialize results for all ROIs
        all_detections = [[] for _ in range(len(roi_bboxes))]

        if len(roi_frames) == 0:
            return all_detections

        # Batch YOLO inference on all ROI crops
        batch_results = self.object_model(
            roi_frames,
            verbose=False,
            conf=self.roi_confidence,
            imgsz=self.imgsz,
            device=self.device
        )

        # Process batch results
        for batch_idx, roi_bbox_idx in enumerate(valid_indices):
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

                class_name = self.object_model.names[cls].replace('_', ' ')  # Normalize: cell_phone -> cell phone
                debug_all_detections.append((class_name, conf))

                if class_name in target_classes:
                    # Convert local ROI coordinates to global frame coordinates
                    global_x1 = xyxy_local[0] + x1
                    global_y1 = xyxy_local[1] + y1
                    global_x2 = xyxy_local[2] + x1
                    global_y2 = xyxy_local[3] + y1

                    bbox_for_validation = (global_x1, global_y1, global_x2, global_y2)
                    if self.validate_object_aspect_ratio(bbox_for_validation, class_name):
                        detections.append((
                            class_name, conf, global_x1, global_y1, global_x2, global_y2
                        ))

            # Debug logging -- these are diagnostic messages, not operational info
            if 'cell phone' in target_classes:
                roi_name = roi_names[roi_bbox_idx]
                if len(debug_all_detections) > 0:
                    cell_phones = [d for d in debug_all_detections if d[0] == 'cell phone']
                    if cell_phones:
                        self.logger.debug(
                            f"[DEBUG ROI BATCH] {roi_name}: [OK] Found "
                            f"{len(cell_phones)} cell phone(s): {cell_phones}"
                        )
                    else:
                        top_detections = sorted(debug_all_detections, key=lambda x: -x[1])[:5]
                        self.logger.debug(
                            f"[DEBUG ROI BATCH] {roi_name}: [FAIL] No phone, "
                            f"found {len(debug_all_detections)} objects: {top_detections}"
                        )
                else:
                    self.logger.debug(f"[DEBUG ROI BATCH] {roi_name}: [WARN] YOLO detected NOTHING")

            all_detections[roi_bbox_idx] = detections

        return all_detections

    def detect_poses(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Run YOLO pose detection on a single frame.

        Args:
            frame: Input BGR frame
            conf_threshold: Optional confidence threshold override

        Returns:
            Dictionary mapping person index to pose data:
            {
                person_idx: {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_confidence': float,
                    'keypoints': YoloPoseLandmarks object
                }
            }
        """
        return self.pose_adapter.process(frame)

    def detect_objects_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[Dict[str, List]]:
        """Run YOLO object detection on multiple frames in a single batch.

        Maximizes GPU utilization by processing multiple frames at once.

        Args:
            frames: List of BGR frames
            batch_size: Maximum batch size for inference

        Returns:
            List of detection dictionaries, one per frame
        """
        if not frames:
            return []

        self.logger.debug(
            f"[GPU BATCH] detect_objects_batch: {len(frames)} frames, batch_size={batch_size}"
        )
        all_detections = []

        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            try:
                batch_results = self.object_model(
                    batch_frames,
                    verbose=False,
                    imgsz=self.imgsz,
                    device=self.device
                )
            except Exception as e:
                self.logger.error(
                    f"[GPU BATCH] Object detection failed for batch starting at {batch_start}: {e}"
                )
                for _ in batch_frames:
                    all_detections.append({
                        'person': [], 'cell_phone': [], 'book': [],
                        'backpack': [], 'cup_bottle': [],
                        'roi_detections': [], 'roi_boxes': []
                    })
                continue

            for frame, results in zip(batch_frames, batch_results):
                detections = {
                    'person': [],
                    'cell_phone': [],
                    'book': [],
                    'backpack': [],
                    'cup_bottle': [],
                    'roi_detections': [],
                    'roi_boxes': []
                }

                person_boxes = []
                pending_books = []
                frame_shape = frame.shape
                _raw_confs = []  # Collect (class_name, conf) for detection summary log

                if results.boxes is not None:
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        class_name = self.object_model.names[cls].replace('_', ' ')  # Normalize: cell_phone -> cell phone
                        _raw_confs.append((class_name, conf))

                        # Zone suppression: skip suppressed classes/regions
                        if self._is_class_suppressed(class_name, xyxy, frame_shape):
                            continue

                        if class_name == 'person' and conf > self.person_confidence:
                            detections['person'].append(xyxy)
                            person_boxes.append(xyxy)
                        elif class_name in ['backpack', 'suitcase']:
                            if conf > self.bag_confidence:
                                if self._validate_bag_detection(xyxy):
                                    detections['backpack'].append(xyxy)
                        elif class_name == 'book' and conf > self.book_confidence:
                            pending_books.append(xyxy)
                        elif class_name == 'cell phone' and conf > self.cell_phone_confidence:
                            detections['cell_phone'].append(xyxy)

                # Process pending books
                for book_xyxy in pending_books:
                    if len(person_boxes) > 0:
                        for person_box in person_boxes:
                            if self._boxes_overlap_or_near(book_xyxy, person_box, margin=200):
                                detections['book'].append(book_xyxy)
                                break
                    else:
                        detections['book'].append(book_xyxy)

                # Log per-frame detection summary
                self._log_detection_summary(detections, _raw_confs)

                all_detections.append(detections)

        self.logger.debug(
            f"[GPU BATCH] detect_objects_batch complete: {len(all_detections)} results"
        )
        return all_detections

    def detect_poses_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[int, Dict[str, Any]]]:
        """Run pose detection on multiple frames in a single batch.

        Maximizes GPU utilization by processing multiple frames at once.
        Delegates to adapter's process_batch() which works for both
        YoloPoseAdapter and RTMPoseAdapter.

        Args:
            frames: List of BGR frames
            batch_size: Maximum batch size for inference
            conf_threshold: Optional confidence threshold override

        Returns:
            List of pose result dictionaries, one per frame.
            Format: {person_idx: {'bbox': [...], 'bbox_confidence': float, 'keypoints': YoloPoseLandmarks}}
        """
        if not frames:
            return []

        effective_conf = conf_threshold if conf_threshold is not None else self.pose_adapter.conf_threshold
        self.logger.debug(
            f"[GPU BATCH] detect_poses_batch: {len(frames)} frames, "
            f"batch_size={batch_size}, conf={effective_conf}"
        )

        # Delegate to adapter's process_batch (works for both YoloPoseAdapter and RTMPoseAdapter)
        if hasattr(self.pose_adapter, 'process_batch'):
            all_poses = self.pose_adapter.process_batch(
                frames, batch_size=batch_size, conf_threshold=effective_conf, device=self.device
            )
            self.logger.debug(
                f"[GPU BATCH] detect_poses_batch complete: {len(all_poses)} results"
            )
            return all_poses

        # Fallback: legacy per-frame processing
        from app.services.yolo_pose_adapter import YoloPoseLandmarks, PersonKeypoints

        all_poses = []
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            try:
                batch_results = self.pose_adapter.model(
                    batch_frames,
                    verbose=False,
                    conf=effective_conf,
                    device=self.device
                )
            except Exception as e:
                self.logger.error(
                    f"[GPU BATCH] Pose detection failed for batch starting at {batch_start}: {e}"
                )
                for _ in batch_frames:
                    all_poses.append({})
                continue

            for frame, results in zip(batch_frames, batch_results):
                persons = {}

                if results.keypoints is not None and results.boxes is not None:
                    for idx in range(len(results.boxes)):
                        box = results.boxes[idx]
                        person_keypoints = PersonKeypoints(results.keypoints, idx)

                        persons[idx] = {
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'bbox_confidence': float(box.conf[0]),
                            'keypoints': YoloPoseLandmarks(person_keypoints, frame.shape)
                        }

                all_poses.append(persons)

        self.logger.debug(
            f"[GPU BATCH] detect_poses_batch complete: {len(all_poses)} results"
        )
        return all_poses

    def preprocess_frames_for_detection(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocess dark/IR frames to improve YOLO person detection.

        Uses adaptive brightness check on a sample frame. If dark, applies
        CLAHE + gamma correction to all frames in the batch.

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

        is_well_lit = brightness >= self.dark_frame_brightness_threshold

        if not self.always_preprocess and is_well_lit:
            return frames  # Well-lit and always_preprocess disabled, no preprocessing needed

        try:
            # Well-lit + always_preprocess: lightweight pipeline (mild CLAHE only)
            # Dark frames: full pipeline (CLAHE + gamma + noise reduction)
            if self.always_preprocess and is_well_lit:
                self.logger.debug(
                    f"[IR PREPROCESS] Well-lit always_preprocess (brightness={brightness:.2f}), "
                    f"applying lightweight CLAHE only"
                )
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                enhanced = []
                for frame in frames:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l = clahe.apply(l)
                    enhanced.append(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))
                return enhanced

            self.logger.debug(
                f"[IR PREPROCESS] Dark frames (brightness={brightness:.2f}), "
                f"applying full CLAHE+gamma for YOLO"
            )
            enhanced = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = self.preprocessing_service.preprocess_for_mediapipe(rgb)
                bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                enhanced.append(bgr)
            return enhanced
        except Exception as e:
            self.logger.warning(
                f"[IR PREPROCESS] Preprocessing failed, using original frames: {e}"
            )
            return frames

    def process_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        preprocess_dark: bool = True
    ) -> Tuple[List[Dict[str, List]], List[Dict[int, Dict[str, Any]]]]:
        """Process a batch of frames for both object and pose detection.

        Convenience method that runs both detection types with optional
        dark frame preprocessing.

        Args:
            frames: List of BGR frames
            batch_size: Maximum batch size for inference
            preprocess_dark: Whether to preprocess dark/IR frames

        Returns:
            Tuple of (object_detections_list, pose_results_list)
        """
        if preprocess_dark:
            frames = self.preprocess_frames_for_detection(frames)

        object_detections = self.detect_objects_batch(frames, batch_size)
        pose_results = self.detect_poses_batch(frames, batch_size)

        return object_detections, pose_results

    def validate_object_aspect_ratio(
        self,
        bbox: Tuple[float, float, float, float],
        object_class: str
    ) -> bool:
        """Validate detected object based on aspect ratio to filter false positives.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            object_class: Class name ('cell phone', 'book', etc.)

        Returns:
            True if aspect ratio is valid for the object class
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height == 0 or width == 0:
            return False

        aspect_ratio = width / height

        aspect_ratio_rules = {
            'cell phone': {
                'min_ratio': 0.4,   # Portrait: ~0.45 (9:20)
                'max_ratio': 2.0,   # Landscape: ~1.78 (16:9)
                'min_size': 30
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

        if aspect_ratio < rules['min_ratio'] or aspect_ratio > rules['max_ratio']:
            return False

        if min(width, height) < rules['min_size']:
            return False

        return True

    def _validate_bag_detection(self, xyxy: np.ndarray) -> bool:
        """Validate bag detection based on aspect ratio and area.

        Args:
            xyxy: Bounding box [x1, y1, x2, y2]

        Returns:
            True if detection passes validation
        """
        aspect_ratio = self._get_aspect_ratio(xyxy)
        area = self._get_area(xyxy)

        return (
            aspect_ratio < self.bag_max_aspect_ratio and
            self.bag_min_area < area < self.bag_max_area
        )

    def _is_book_near_person(
        self,
        book_xyxy: np.ndarray,
        person_boxes: List[np.ndarray]
    ) -> bool:
        """Check if book is near any detected person.

        Args:
            book_xyxy: Book bounding box
            person_boxes: List of person bounding boxes

        Returns:
            True if book is within margin of any person
        """
        if len(person_boxes) == 0:
            return False

        book_center_x = (book_xyxy[0] + book_xyxy[2]) / 2
        book_center_y = (book_xyxy[1] + book_xyxy[3]) / 2

        for person_box in person_boxes:
            person_x1, person_y1, person_x2, person_y2 = person_box
            margin = self.book_person_margin

            if (person_x1 - margin <= book_center_x <= person_x2 + margin and
                person_y1 - margin <= book_center_y <= person_y2 + margin):
                return True

        return False

    def _boxes_overlap_or_near(
        self,
        box1: np.ndarray,
        box2: np.ndarray,
        margin: int = 100
    ) -> bool:
        """Check if two boxes overlap or are within margin pixels of each other.

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            margin: Proximity margin in pixels

        Returns:
            True if boxes overlap or are within margin
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Expand box2 by margin
        x2_min_expanded = x2_min - margin
        y2_min_expanded = y2_min - margin
        x2_max_expanded = x2_max + margin
        y2_max_expanded = y2_max + margin

        # Check for overlap with expanded box
        return not (
            x1_max < x2_min_expanded or x1_min > x2_max_expanded or
            y1_max < y2_min_expanded or y1_min > y2_max_expanded
        )

    @staticmethod
    def _get_aspect_ratio(xyxy: np.ndarray) -> float:
        """Calculate aspect ratio of bounding box."""
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        return width / height if height > 0 else 999

    @staticmethod
    def _get_area(xyxy: np.ndarray) -> float:
        """Calculate area of bounding box."""
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        return width * height

    def _log_detection_summary(
        self,
        detections: Dict[str, List],
        raw_confs: List[tuple],
    ) -> None:
        """Log a one-line summary of YOLO detections with raw confidences.

        Only logs when at least one non-person object is detected by YOLO
        (regardless of whether it passed filtering), to avoid spamming on
        person-only frames.
        """
        # Group raw confidences by class
        from collections import defaultdict
        conf_by_class = defaultdict(list)
        for class_name, conf in raw_confs:
            conf_by_class[class_name].append(conf)

        # Only log if there's something interesting (non-person detections)
        non_person_classes = {k for k in conf_by_class if k != 'person'}
        if not non_person_classes:
            return

        # Build summary parts
        parts = []
        for cls in ['person', 'backpack', 'suitcase', 'cell phone',
                     'book', 'bottle']:
            confs = conf_by_class.get(cls, [])
            if confs:
                conf_str = ','.join(f'{c:.2f}' for c in sorted(confs, reverse=True))
                parts.append(f"{cls}={len(confs)}({conf_str})")

        # Also show accepted counts
        accepted = []
        for key in ['person', 'backpack', 'cell_phone', 'book', 'cup_bottle']:
            count = len(detections.get(key, []))
            if count > 0:
                accepted.append(f"{key}={count}")

        self.logger.info(
            f"[YOLO] Raw: {', '.join(parts)} | "
            f"Accepted: {', '.join(accepted) if accepted else 'none'}"
        )

    @staticmethod
    def _calculate_iou(box1, box2) -> float:
        """Calculate Intersection over Union between two bounding boxes.

        Args:
            box1: First bbox [x1, y1, x2, y2]
            box2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU value between 0.0 and 1.0
        """
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
        area2 = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def get_roi_around_keypoint(
        keypoint_coords: Tuple[int, int],
        frame_shape: Tuple[int, ...],
        roi_size: int = 150
    ) -> Optional[Tuple[int, int, int, int]]:
        """Create Region of Interest (ROI) box around a keypoint.

        Args:
            keypoint_coords: (x, y) coordinates of keypoint
            frame_shape: (height, width) of frame
            roi_size: Size of ROI box in pixels

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
