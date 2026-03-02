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

Deduplication (Task 0025):
- Shared detection methods are delegated to YOLOHandler to eliminate
  ~80% code duplication between ObjectDetector and YOLOHandler.
- ObjectDetector wraps YOLOHandler, adding pose-guided ROI orchestration.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import logging


class ObjectDetector:
    """Detects objects using YOLO with pose-guided ROI optimization.

    This class provides comprehensive object detection through:
    - Full-frame YOLO inference for persons, bags, and proximity-based objects
    - Pose-guided ROI detection for hands, ears, and lap areas
    - Batched inference for multi-frame GPU optimization
    - Aspect ratio validation to filter false positives

    Shared detection methods (detect_objects, detect_objects_in_rois_batch,
    detect_objects_batch, validate_object_aspect_ratio, _boxes_overlap_or_near,
    get_roi_around_keypoint, preprocess_frames_for_detection) are delegated to
    YOLOHandler to avoid code duplication (Task 0025).

    Attributes:
        yolo_handler: YOLOHandler instance for delegated detection methods
        yolo_model: YOLO model instance for object detection (via yolo_handler)
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
        cell_phone_confidence: float = 0.45,
        yolo_handler: Optional[Any] = None
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
            yolo_handler: Optional pre-configured YOLOHandler instance for delegation.
                         If None, one is created internally from the provided yolo_model.
        """
        self.yolo_model = yolo_model
        self.settings = settings
        self.preprocessing_service = preprocessing_service
        self._get_keypoint = get_keypoint_func
        self.logger = logger or logging.getLogger(__name__)
        self.yolo_imgsz = yolo_imgsz
        self.yolo_device = yolo_device
        self.cell_phone_confidence = cell_phone_confidence

        # Task 0025: Set up YOLOHandler for delegation of shared methods.
        # If a YOLOHandler is provided, use it directly. Otherwise, create
        # one internally by wrapping the raw yolo_model.
        if yolo_handler is not None:
            self.yolo_handler = yolo_handler
        else:
            self.yolo_handler = self._create_yolo_handler(
                yolo_model, settings, preprocessing_service,
                yolo_imgsz, yolo_device, cell_phone_confidence
            )

        # Cache for avoiding redundant inference
        self._cached_frame_objects = None
        self._cached_frame_time = 0

        # Initialize thresholds from settings or defaults
        self._init_thresholds()

    @staticmethod
    def _create_yolo_handler(
        yolo_model: Any,
        settings: Optional[Any],
        preprocessing_service: Optional[Any],
        imgsz: int,
        device: Optional[str],
        cell_phone_confidence: float
    ) -> Any:
        """Create a YOLOHandler instance for internal delegation.

        Constructs a YOLOHandler without requiring a pose model by manually
        setting up the object detection attributes needed for shared methods.

        Args:
            yolo_model: YOLO model instance
            settings: Configuration settings
            preprocessing_service: Preprocessing service for dark frames
            imgsz: Input image size
            device: Inference device
            cell_phone_confidence: Cell phone confidence threshold

        Returns:
            Configured YOLOHandler instance with object detection capabilities
        """
        from app.core.models.yolo_handler import YOLOHandler

        # Create a YOLOHandler without calling __init__ to avoid the
        # pose model requirement. Only object detection methods are needed.
        handler = object.__new__(YOLOHandler)

        # Set up logger
        handler.logger = logging.getLogger('YOLOHandler')

        # Set core attributes needed by shared detection methods
        handler.object_model = yolo_model
        handler.device = device or 'cpu'
        handler.imgsz = imgsz
        handler.settings = settings
        handler.preprocessing_service = preprocessing_service

        # Pose adapter not needed for ObjectDetector delegation
        handler.pose_adapter = None
        handler._models_preloaded = True

        # Configure thresholds (replicate _configure_thresholds logic)
        handler.person_confidence = getattr(settings, 'yolo_person_confidence', 0.5) if settings else 0.5
        handler.bag_confidence = getattr(settings, 'yolo_bag_confidence', 0.45) if settings else 0.45
        handler.bag_log_confidence = getattr(settings, 'yolo_bag_log_confidence', 0.25) if settings else 0.25
        handler.book_confidence = getattr(settings, 'yolo_book_confidence', 0.4) if settings else 0.4
        handler.cell_phone_confidence = getattr(settings, 'yolo_cell_phone_confidence', 0.3) if settings else 0.3
        handler.pose_sleep_confidence = getattr(settings, 'yolo_pose_sleep_confidence', 0.30) if settings else 0.30

        # Object Detection Geometry
        handler.bag_max_aspect_ratio = getattr(settings, 'bag_max_aspect_ratio', 1.5) if settings else 1.5
        handler.bag_min_area = getattr(settings, 'bag_min_area', 5000) if settings else 5000
        handler.bag_max_area = getattr(settings, 'bag_max_area', 100000) if settings else 100000
        handler.book_person_margin = getattr(settings, 'book_person_margin', 150) if settings else 150

        # ROI confidence (sourced from settings, no os.getenv bypass)
        handler.roi_confidence = cell_phone_confidence

        # Dark frame preprocessing threshold
        handler.dark_frame_brightness_threshold = (
            getattr(settings, 'yolo_dark_frame_brightness_threshold', 0.4) if settings else 0.4
        )

        # Always-preprocess flag (Phase 1 SOTA: bypass brightness check)
        handler.always_preprocess = getattr(settings, 'yolo_always_preprocess', False) if settings else False

        # Zone suppression (Phase 1 SOTA: fixed-camera FP filtering)
        handler._configure_zone_suppression(settings)

        # SAHI (Phase 1 SOTA: sliced inference for small objects)
        handler._configure_sahi(settings)

        # Cache for frame results
        handler._cached_frame_objects = None
        handler._cached_frame_time = 0.0

        return handler

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
        # NOTE: bag_max_aspect_ratio controls false positive filtering for bag
        # detections.  A value of 1.5 allows taller/narrower bags (backpacks,
        # duffel bags) while still rejecting clearly non-bag shapes.  Lowering
        # to 1.2 reduces false positives but may reject legitimate tall bags.
        self.BAG_MAX_ASPECT_RATIO = getattr(s, 'bag_max_aspect_ratio', 1.5) if s else 1.5
        self.BAG_MIN_AREA = getattr(s, 'bag_min_area', 5000) if s else 5000
        self.BAG_MAX_AREA = getattr(s, 'bag_max_area', 100000) if s else 100000
        self.BOOK_PERSON_MARGIN = getattr(s, 'book_person_margin', 150) if s else 150
        self.PERSON_BOOK_OVERLAP_MARGIN = getattr(s, 'person_book_overlap_margin', 250) if s else 250

    # =========================================================================
    # Task 0025: DELEGATED METHODS
    # All detection methods delegate to YOLOHandler to eliminate duplication.
    # =========================================================================

    def get_roi_around_keypoint(
        self,
        keypoint_coords: Any,
        frame_shape: Tuple[int, ...],
        roi_size: int = 150
    ) -> Optional[Tuple[int, int, int, int]]:
        """Create Region of Interest (ROI) box around a keypoint.

        Delegates to YOLOHandler.get_roi_around_keypoint().

        Args:
            keypoint_coords: (x, y) coordinates of keypoint
            frame_shape: (height, width) of frame
            roi_size: Size of ROI box in pixels (default 150x150)

        Returns:
            (x1, y1, x2, y2) ROI bounding box, or None if invalid
        """
        return self.yolo_handler.get_roi_around_keypoint(
            keypoint_coords, frame_shape, roi_size
        )

    def validate_object_aspect_ratio(self, bbox: List[int], object_class: str) -> bool:
        """Validate detected object based on aspect ratio to filter false positives.

        Delegates to YOLOHandler.validate_object_aspect_ratio().

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            object_class: Class name ('cell phone', 'book', etc.)

        Returns:
            bool: True if aspect ratio is valid for the object class
        """
        return self.yolo_handler.validate_object_aspect_ratio(bbox, object_class)

    def detect_objects_in_rois_batch(
        self,
        frame: Any,
        roi_bboxes: List[Tuple[int, int, int, int]],
        roi_names: List[str],
        target_classes: List[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """Run YOLO detection on multiple ROI regions in a single batched call.

        Delegates to YOLOHandler.detect_objects_in_rois_batch().

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
        return self.yolo_handler.detect_objects_in_rois_batch(
            frame, roi_bboxes, roi_names, target_classes
        )

    def detect_objects(
        self,
        frame: Any,
        pose_landmarks: Any = None,
        use_pose_guided: bool = True
    ) -> Dict[str, List[Any]]:
        """Detect objects using YOLO with pose-guided detection.

        Delegates full-frame detection (Stage 1) to YOLOHandler.detect_objects().
        Stage 2 (pose-guided ROI detection) is handled locally since it
        uses ObjectDetector-specific keypoint functions.

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
        # Delegate Stage 1 (full-frame detection) to YOLOHandler.
        # Pass get_keypoint_func and get_roi_func for Stage 2 if pose-guided.
        if use_pose_guided and pose_landmarks is not None and self._get_keypoint is not None:
            detections = self.yolo_handler.detect_objects(
                frame,
                pose_landmarks=pose_landmarks,
                use_pose_guided=True,
                get_keypoint_func=self._get_keypoint,
                get_roi_func=self.yolo_handler.get_roi_around_keypoint
            )
        else:
            detections = self.yolo_handler.detect_objects(
                frame,
                pose_landmarks=None,
                use_pose_guided=False
            )

        # Sync cache from handler
        self._cached_frame_objects = self.yolo_handler._cached_frame_objects
        self._cached_frame_time = self.yolo_handler._cached_frame_time

        return detections

    def detect_objects_batch(
        self,
        frames: List[Any],
        batch_size: int = 8
    ) -> List[Dict[str, List[Any]]]:
        """Run YOLO object detection on multiple frames in a single batch.

        Delegates to YOLOHandler.detect_objects_batch().

        This maximizes GPU utilization by processing multiple frames at once.

        Args:
            frames: List of BGR frames (numpy arrays)
            batch_size: Maximum batch size for inference (default 8)

        Returns:
            List of detection dictionaries, one per frame
        """
        return self.yolo_handler.detect_objects_batch(frames, batch_size)

    def _boxes_overlap_or_near(
        self,
        box1: Any,
        box2: Any,
        margin: int = 100
    ) -> bool:
        """Check if two boxes overlap or are within margin pixels of each other.

        Delegates to YOLOHandler._boxes_overlap_or_near().

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            margin: Margin in pixels for proximity check

        Returns:
            bool: True if boxes overlap or are within margin
        """
        return self.yolo_handler._boxes_overlap_or_near(box1, box2, margin)

    def _preprocess_frames_for_detection(self, frames: List[Any]) -> List[Any]:
        """Preprocess dark/IR frames to improve YOLO person detection.

        Delegates to YOLOHandler.preprocess_frames_for_detection().

        Uses adaptive brightness check on a sample frame. If dark (brightness < threshold),
        applies CLAHE + gamma correction to all frames in the batch.

        Args:
            frames: List of BGR frames

        Returns:
            List of frames (preprocessed copies if dark, originals if not)
        """
        return self.yolo_handler.preprocess_frames_for_detection(frames)

    # =========================================================================
    # Task 0025: Additional delegated methods
    # These wrap YOLOHandler methods, supplying ObjectDetector-specific state
    # (e.g. self._get_keypoint) that YOLOHandler receives as parameters.
    # =========================================================================

    def detect_objects_in_roi(
        self,
        frame: Any,
        roi_bbox: Tuple[int, int, int, int],
        target_classes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Run YOLO detection on a specific ROI region.

        Delegates to YOLOHandler.detect_objects_in_rois_batch() with a
        single ROI to avoid duplicating YOLO inference logic.

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

        # Delegate to batch method with a single ROI
        batch_results = self.yolo_handler.detect_objects_in_rois_batch(
            frame, [roi_bbox], ['single_roi'], target_classes
        )

        return batch_results[0] if batch_results else []

    def _detect_pose_guided_rois(
        self,
        frame: Any,
        pose_landmarks: Any,
        detections: Dict[str, List[Any]]
    ) -> None:
        """Run pose-guided ROI detection and add results to detections dict.

        Delegates to YOLOHandler._detect_objects_in_pose_rois() to avoid
        duplicating the ROI collection and batch processing logic.

        Args:
            frame: Input frame (BGR)
            pose_landmarks: Pose landmarks for ROI-based detection
            detections: Detection dictionary to update in-place
        """
        self.yolo_handler._detect_objects_in_pose_rois(
            frame, pose_landmarks, detections,
            self._get_keypoint, self.yolo_handler.get_roi_around_keypoint
        )

    def detect_objects_person_rois(
        self,
        frame: Any,
        pose_landmarks: Any
    ) -> Dict[str, List[Any]]:
        """Run ONLY pose-guided ROI detection for a single person (Stage 2 only).

        CR-006 OPTIMIZATION: This method performs only the ROI-based detection
        around a person's keypoints, WITHOUT re-running full-frame YOLO inference.

        Delegates to YOLOHandler.detect_objects_person_rois() to avoid
        duplicating the ROI collection and batch processing logic.

        Args:
            frame: Input frame (BGR)
            pose_landmarks: Pose landmarks for this person

        Returns:
            Dictionary with 'cell_phone', 'book', 'roi_detections', 'roi_boxes' keys
        """
        if pose_landmarks is None or self._get_keypoint is None:
            return {
                'cell_phone': [],
                'book': [],
                'roi_detections': [],
                'roi_boxes': []
            }

        return self.yolo_handler.detect_objects_person_rois(
            frame, pose_landmarks,
            self._get_keypoint, self.yolo_handler.get_roi_around_keypoint
        )
