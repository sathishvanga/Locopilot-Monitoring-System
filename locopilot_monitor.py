import cv2
import json
import math
import numpy as np
import time as time_module  # Renamed to avoid any potential shadowing issues
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import mediapipe as mp
from ultralytics import YOLO
import os
import logging
from typing import Optional, List, Dict, Tuple, Any, Union, Generator
import gc
import contextlib
import sys
import subprocess

# ---------------------------------------------------------------------------
# Extracted module imports (refactored components)
# ---------------------------------------------------------------------------
from app.core.utils.geometry import calculate_iou, bbox_overlap_with_margin, deduplicate_person_boxes
from app.core.models.yolo_handler import YOLOHandler, YOLO_KEYPOINT_INDICES
from app.core.detectors import SleepDetector, ActivityDetector, GestureDetector, MindDiversionDetector, ObjectDetector
from app.core.tracking import PersonTracker
from app.core.visualization import FrameAnnotator
from app.core.activity_tracker import ActivityTracker, ActivityConfig as ExtractedActivityConfig
from app.core.evidence_manager import EvidenceManager


# ---------------------------------------------------------------------------
# CR-004: ActivityConfig dataclass and ACTIVITY_REGISTRY
# ---------------------------------------------------------------------------
# Consolidates the parallel per-activity dictionaries (activities,
# consecutive_detections, grace_counters, activity_thresholds) into a
# single source of truth.  At runtime the same dict values are produced
# from this registry so behaviour is unchanged.
# ---------------------------------------------------------------------------

@dataclass
class ActivityConfig:
    """Per-activity configuration used to seed runtime tracking dicts."""

    # activity_thresholds fields
    min_duration: float = 0.0
    required_consecutive: int = 1
    margin: Optional[int] = None
    grace_frames: int = 5

    # Extra threshold fields used by specific activities
    region_margin: Optional[int] = None
    wrist_inside_margin: Optional[int] = None
    sustained_proximity_seconds: Optional[float] = None


# Single registry that replaces the 4 hand-written parallel dicts.
# The keys are the canonical activity names.
# Built via function so config-driven margins are resolved at first access.
def _build_activity_registry() -> Dict[str, ActivityConfig]:
    """Build activity registry with config-driven margins."""
    try:
        _settings = get_settings() if get_settings is not None else None
    except Exception:
        _settings = None

    cell_phone_margin = _settings.activity_cell_phone_margin if _settings else 180
    writing_margin = _settings.activity_writing_margin if _settings else 180
    packing_margin = _settings.activity_packing_margin if _settings else 100
    packing_region_margin = _settings.activity_packing_region_margin if _settings else 150
    packing_wrist_inside_margin = _settings.activity_packing_wrist_inside_margin if _settings else 80

    return {
        'microsleep': ActivityConfig(
            min_duration=2.0,
            required_consecutive=1,
            margin=None,
            grace_frames=10,
        ),
        'sleep': ActivityConfig(
            min_duration=2.0,
            required_consecutive=1,
            margin=None,
            grace_frames=10,
        ),
        'cell_phone': ActivityConfig(
            min_duration=0.1,
            required_consecutive=1,
            margin=cell_phone_margin,
            grace_frames=8,
        ),
        'writing': ActivityConfig(
            min_duration=0.1,
            required_consecutive=1,
            margin=writing_margin,
            grace_frames=10,
        ),
        'packing_bags': ActivityConfig(
            min_duration=0.0,
            required_consecutive=1,
            margin=packing_margin,
            grace_frames=5,
            region_margin=packing_region_margin,
            wrist_inside_margin=packing_wrist_inside_margin,
            sustained_proximity_seconds=4.0,
        ),
        'group_detected': ActivityConfig(
            min_duration=0.0,
            required_consecutive=3,
            margin=None,
            grace_frames=8,
        ),
        'lp_hand_gesture': ActivityConfig(
            min_duration=0.0,
            required_consecutive=1,
            margin=None,
            grace_frames=5,
        ),
        'alp_hand_gesture': ActivityConfig(
            min_duration=0.0,
            required_consecutive=1,
            margin=None,
            grace_frames=5,
        ),
        'mind_diversion': ActivityConfig(
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'eating_drinking': ActivityConfig(
            min_duration=0.0,
            required_consecutive=2,
            margin=None,
            grace_frames=5,
        ),
        'no_person_detected': ActivityConfig(
            min_duration=5.0,
            required_consecutive=3,
            margin=None,
            grace_frames=3,
        ),
        'alp_not_standing': ActivityConfig(
            required_consecutive=2,
            grace_frames=3,
        ),
    }


# Add app directory to path for importing preprocessing service
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import preprocessing service, config, and voting service
try:
    from app.services.image_preprocessing_service import ImagePreprocessingService
    from app.utils.config import get_settings
    from app.services.voting_verification_service import VotingVerificationService, ActivityBatchCollector
    # Train motion rule engine imports
    from app.services.rule_engine_service import RuleEngineService, get_rule_engine_service
    from app.services.train_motion_resolver_service import TrainMotionResolverService, get_motion_resolver_service
    from app.services.ocr_timestamp_service import OCRTimestampService, get_ocr_timestamp_service
    from app.services.alp_alertness_service import ALPAlertnessService, get_alp_alertness_service
    from app.models.trip_models import TripSchedule, TrainMotionContext, TrainMotionState
except ImportError:
    # Fallback: try importing as module
    try:
        import importlib.util
        preprocessing_path = os.path.join(script_dir, 'app', 'services', 'image_preprocessing_service.py')
        config_path = os.path.join(script_dir, 'app', 'utils', 'config.py')

        if os.path.exists(preprocessing_path):
            spec = importlib.util.spec_from_file_location("image_preprocessing_service", preprocessing_path)
            if spec and spec.loader:
                preprocessing_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(preprocessing_module)
                ImagePreprocessingService = preprocessing_module.ImagePreprocessingService
            else:
                ImagePreprocessingService = None
        else:
            ImagePreprocessingService = None

        if os.path.exists(config_path):
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                get_settings = config_module.get_settings
            else:
                get_settings = None
        else:
            get_settings = None
    except Exception:
        ImagePreprocessingService = None
        get_settings = None
        VotingVerificationService = None
        ActivityBatchCollector = None

    # Set rule engine imports to None if not available
    RuleEngineService = None
    get_rule_engine_service = None
    TrainMotionResolverService = None
    get_motion_resolver_service = None
    OCRTimestampService = None
    get_ocr_timestamp_service = None
    ALPAlertnessService = None
    get_alp_alertness_service = None
    TripSchedule = None
    TrainMotionContext = None
    TrainMotionState = None

# Import VotingVerificationService separately (may not exist in fallback path)
try:
    if 'VotingVerificationService' not in dir():
        from app.services.voting_verification_service import VotingVerificationService, ActivityBatchCollector
except ImportError:
    VotingVerificationService = None
    ActivityBatchCollector = None

# Build activity registry now that get_settings is available
ACTIVITY_REGISTRY: Dict[str, ActivityConfig] = _build_activity_registry()


# [OK] WINDOWS FIX: Prevent Qt/GUI initialization in worker processes
# If running in a worker process (detected by QT_QPA_PLATFORM=offscreen),
# ensure no Qt imports happen that could create GUI windows
if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
    # Worker process - prevent any Qt/GUI initialization
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('DISPLAY', '')

# Configure module-level logger for file-only logging
# Console output is disabled - all logs go to file only
def _setup_module_logger(logger_name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup a module-level logger with file-only output.
    Console logging is disabled for clean terminal output.
    
    Args:
        logger_name: Name for the logger
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler only - no console output
        file_handler = logging.FileHandler(os.path.join(log_dir, 'LocopilotMonitoring.log'))
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter matching application format
        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d [N/A] [N/A] [N/A] [N/A] [%(levelname)s] [%(name)s] [N/A N/A] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(level)
    return logger

# Module-level loggers removed - consolidated to self.logger instance hierarchy


@contextlib.contextmanager
def video_capture_context(video_path):
    """
    Context manager to ensure VideoCapture is always released.
    This prevents memory leaks from unclosed video captures.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        if cap.isOpened():
            cap.release()


class LocopilotActivityMonitor:
    # YOLO pose keypoint index groups (shared across detection methods)
    YOLO_HEAD_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
    YOLO_BODY_INDICES = [5, 6, 7, 8, 11, 12]  # left/right shoulders, elbows, hips
    YOLO_MIN_KEYPOINTS = 13  # Minimum landmarks required (indices 0-12)

    def __init__(self, video_path: str, output_dir: str = "evidence", save_annotated_frames: bool = False, frame_save_interval: int = 1, sample_fps: float = 1.0, run_dir: Optional[str] = None, create_run_dir: bool = True, preloaded_models: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Locopilot Activity Monitor.
        
        Args:
            video_path: Path to video file
            output_dir: Base output directory for evidence
            save_annotated_frames: Whether to save annotated frames
            frame_save_interval: Save 1 frame every N sampled frames
            sample_fps: Frame sampling rate (e.g., 0.5 = 1 frame every 2 seconds)
            run_dir: Run directory for saving clips (for multiprocessing)
            create_run_dir: Whether to create new run directory
            preloaded_models: Optional dict of pre-loaded models from worker pool
                Keys: 'yolo', 'yolo_pose', 'face_mesh', 'mp_face_mesh', 'preprocessing_service'
                When provided, skips expensive model loading (significant performance gain)
        """
        self.video_path = video_path
        self.output_dir = output_dir

        # CR-011: Cached video metadata (lazily loaded to avoid opening VideoCapture at init)
        self._video_total_frames = None
        self._video_fps = None
        self._video_duration_seconds = None

        # Initialize logger (file-only output, no console)
        self.logger = _setup_module_logger(f'{self.__class__.__name__}', logging.DEBUG)
        
        # Frame sampling configuration
        self.sample_fps = sample_fps  # Sample frames at this rate (e.g., 0.5 = 1 frame every 2 seconds)
        
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
        
        # Import adapter utilities (needed regardless of preloaded models)
        from app.services.yolo_pose_adapter import YOLO_KEYPOINT_INDICES, get_keypoint_by_name
        self.yolo_keypoint_indices = YOLO_KEYPOINT_INDICES
        self._get_keypoint_by_name = get_keypoint_by_name
        
        # Get settings early (needed for both preloaded and fresh model paths)
        settings = get_settings() if get_settings is not None else None
        self.settings = settings  # Store for use in end_activity() clip buffer settings

        # === Config-driven thresholds (replace former class-level constants) ===
        # Wrist/Elbow Detection
        self.MAX_WRIST_DISTANCE = settings.max_wrist_distance if settings else 300
        self.MAX_ELBOW_DISTANCE = settings.max_elbow_distance if settings else 450
        self.MAX_SINGLE_WRIST_DISTANCE = settings.max_single_wrist_distance if settings else 250
        self.WRITING_WRIST_DISTANCE = settings.writing_wrist_distance if settings else 300
        self.RELAXED_WRIST_DISTANCE = settings.relaxed_wrist_distance if settings else 400
        self.ELBOW_VISIBILITY_THRESHOLD = settings.elbow_visibility_threshold if settings else 0.25
        self.WRIST_VISIBILITY_THRESHOLD = settings.wrist_visibility_threshold if settings else 0.3

        # Writing Detection
        self.WRITING_MIN_DURATION = settings.writing_min_duration if settings else 1.0
        self.WRITING_REQUIRED_CONSECUTIVE = settings.writing_required_consecutive if settings else 2
        self.BOOK_POSTURE_MIN_DURATION = settings.book_posture_min_duration if settings else 2.0
        self.BOOK_POSTURE_REQUIRED_CONSECUTIVE = settings.book_posture_required_consecutive if settings else 2

        # Head Tilt / Sleep Detection
        self.HEAD_DOWN_THRESHOLD = settings.head_down_threshold if settings else 0.01
        # Pose-Based Sleep Detection
        self.SLEEP_STRONG_SCORE = settings.sleep_strong_score if settings else 4
        self.SLEEP_STRONG_DURATION = settings.sleep_strong_duration if settings else 2
        self.SLEEP_MODERATE_DURATION = settings.sleep_moderate_duration if settings else 4
        self.SLEEP_MICROSLEEP_DURATION = settings.sleep_microsleep_duration if settings else 2
        self.MINIMAL_MOVEMENT_THRESHOLD = settings.minimal_movement_threshold if settings else 0.15
        self.STABLE_POSTURE_VARIANCE = settings.stable_posture_variance if settings else 100
        self.EYES_NOT_VISIBLE_THRESHOLD = settings.eyes_not_visible_threshold if settings else 0.4

        # Baseline calibration for camera-angle adaptation
        self.SLEEP_BASELINE_ENABLED = settings.sleep_baseline_enabled if settings else True
        self.SLEEP_BASELINE_CALIBRATION_WINDOW = settings.sleep_baseline_calibration_window if settings else 10.0
        self.SLEEP_BASELINE_MIN_SAMPLES = settings.sleep_baseline_min_samples if settings else 5

        # Delta-from-baseline thresholds
        self.SLEEP_BASELINE_NOSE_BELOW_DELTA = settings.sleep_baseline_nose_below_delta if settings else 40
        self.SLEEP_BASELINE_HEAD_TILT_DELTA = settings.sleep_baseline_head_tilt_delta if settings else 25
        self.SLEEP_BASELINE_TORSO_HEIGHT_DELTA = settings.sleep_baseline_torso_height_delta if settings else 40
        self.SLEEP_BASELINE_SHOULDER_WIDTH_DELTA = settings.sleep_baseline_shoulder_width_delta if settings else 20

        # New discriminating signals
        self.SLEEP_SUSTAINED_STILLNESS_THRESHOLD = settings.sleep_sustained_stillness_threshold if settings else 0.02
        self.SLEEP_SUSTAINED_STILLNESS_FRAMES = settings.sleep_sustained_stillness_frames if settings else 3
        self.SLEEP_HANDS_CLASPED_THRESHOLD = settings.sleep_hands_clasped_threshold if settings else 100
        self.SLEEP_HANDS_CLASPED_FRAMES = settings.sleep_hands_clasped_frames if settings else 3
        self.SLEEP_SUSTAINED_LOW_EYE_FRAMES = settings.sleep_sustained_low_eye_frames if settings else 3
        self.SLEEP_HANDS_SPREAD_THRESHOLD = settings.sleep_hands_spread_threshold if settings else 180

        # Head Bob Detection
        self.SLEEP_HEAD_BOB_DRIFT_MAX_RATE = settings.sleep_head_bob_drift_max_rate if settings else 15.0
        self.SLEEP_HEAD_BOB_JERK_MIN_RATE = settings.sleep_head_bob_jerk_min_rate if settings else 20.0
        self.SLEEP_HEAD_BOB_MIN_DRIFT_FRAMES = settings.sleep_head_bob_min_drift_frames if settings else 2
        self.SLEEP_HEAD_BOB_MIN_AMPLITUDE = settings.sleep_head_bob_min_amplitude if settings else 10.0
        self.SLEEP_HEAD_BOB_SCORE_BONUS = settings.sleep_head_bob_score_bonus if settings else 2
        self.SLEEP_HEAD_BOB_BYPASS_EYE_GATE = settings.sleep_head_bob_bypass_eye_gate if settings else True

        # Wrist Velocity Tracking
        self.SLEEP_WRIST_VEL_STILL = settings.sleep_wrist_velocity_still_threshold if settings else 0.005
        self.SLEEP_WRIST_VEL_ACTIVE = settings.sleep_wrist_velocity_active_threshold if settings else 0.03
        self.SLEEP_WRIST_VEL_STILL_FRAMES = settings.sleep_wrist_velocity_still_frames if settings else 2

        # Temporal State Machine
        self.SLEEP_STATE_MACHINE_ENABLED = settings.sleep_state_machine_enabled if settings else True
        self.SLEEP_STATE_HAND_ACTIVITY_THRESHOLD = settings.sleep_state_hand_activity_threshold if settings else 0.02
        self.SLEEP_DROWSY_TO_MICROSLEEP_SEC = settings.sleep_state_drowsy_to_microsleep_sec if settings else 2.0
        self.SLEEP_MICROSLEEP_TO_SLEEP_SEC = settings.sleep_state_microsleep_to_sleep_sec if settings else 4.0

        # Shoulder Slump Rate
        self.SLEEP_SHOULDER_SLUMP_RATE_THRESHOLD = settings.sleep_shoulder_slump_rate_threshold if settings else 0.005
        self.SLEEP_SHOULDER_SLUMP_MIN_FRAMES = settings.sleep_shoulder_slump_min_frames if settings else 3

        # IR Forward Lean Detection
        self.IR_SHOULDER_RELATIVE_THRESHOLD = settings.ir_shoulder_relative_threshold if settings else 0.4
        self.IR_BBOX_ASPECT_RATIO_THRESHOLD = settings.ir_bbox_aspect_ratio_threshold if settings else 1.2
        self.IR_LOW_MOVEMENT_THRESHOLD = settings.ir_low_movement_threshold if settings else 0.02
        self.SUB_THRESHOLD_STREAK_LIMIT = settings.sub_threshold_streak_limit if settings else 3

        # YOLO Confidence Thresholds
        self.YOLO_PERSON_CONFIDENCE = settings.yolo_person_confidence if settings else 0.5
        self.YOLO_BAG_CONFIDENCE = settings.yolo_bag_confidence if settings else 0.45
        self.YOLO_BAG_LOG_CONFIDENCE = settings.yolo_bag_log_confidence if settings else 0.25
        self.YOLO_BOOK_CONFIDENCE = settings.yolo_book_confidence if settings else 0.4
        self.YOLO_CELL_PHONE_CONFIDENCE = settings.yolo_cell_phone_confidence if settings else 0.3
        self.YOLO_SLEEP_POSE_CONFIDENCE = settings.yolo_pose_sleep_confidence if settings else 0.30

        # Object Detection Geometry
        self.BAG_MAX_ASPECT_RATIO = settings.bag_max_aspect_ratio if settings else 1.2
        self.BAG_MIN_AREA = settings.bag_min_area if settings else 5000
        self.BAG_MAX_AREA = settings.bag_max_area if settings else 100000
        self.BOOK_PERSON_MARGIN = settings.book_person_margin if settings else 150
        self.PERSON_BOOK_OVERLAP_MARGIN = settings.person_book_overlap_margin if settings else 250

        # Pose Validation
        self.MIN_POSE_LANDMARKS = settings.min_pose_landmarks if settings else 10
        self.MIN_POSE_VISIBILITY = settings.min_pose_visibility if settings else 0.3
        self.FACE_MESH_DETECTION_CONFIDENCE = settings.face_mesh_detection_confidence if settings else 0.5
        self.FACE_MESH_TRACKING_CONFIDENCE = settings.face_mesh_tracking_confidence if settings else 0.5

        # Initialize models - either use preloaded or load fresh
        if preloaded_models is not None:
            # [OK] PERFORMANCE: Use pre-loaded models from worker pool (fast path)
            self.yolo_model = preloaded_models.get('yolo')
            self.yolo_pose = preloaded_models.get('yolo_pose')
            self.face_mesh = preloaded_models.get('face_mesh')
            self.mp_face_mesh = preloaded_models.get('mp_face_mesh')
            self.preprocessing_service = preloaded_models.get('preprocessing_service')
            
            # Keep MediaPipe references for backward compatibility
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Validate required models
            if self.yolo_model is None or self.yolo_pose is None:
                raise ValueError("preloaded_models must contain 'yolo' and 'yolo_pose'")
        else:
            # Load models fresh (slow path - for standalone use)
            # Get model paths from config (configurable via environment variables)
            # Note: settings is already defined above
            yolo_weights = settings.yolo_weights if settings else 'yolo26n.pt'
            yolo_pose_weights = settings.yolo_pose_weights if settings else 'yolo26n-pose.pt'
            yolo_pose_conf = settings.yolo_pose_confidence if settings else 0.45
            
            self.logger.info(f"Loading YOLO model: {yolo_weights}")
            self.yolo_model = YOLO(yolo_weights)

            # Phase 3.5 Quick Win A: Fuse Conv+BatchNorm layers for faster inference (15-20% speedup)
            if hasattr(self.yolo_model.model, 'fuse'):
                self.yolo_model.fuse()
                self.logger.info("YOLO model layers fused for optimized inference")

            # YOLO-Pose for body pose estimation (replaces MediaPipe Pose)
            self.logger.info(f"Loading YOLO-Pose model: {yolo_pose_weights}")
            from app.services.yolo_pose_adapter import YoloPoseAdapter
            self.yolo_pose = YoloPoseAdapter(model_path=yolo_pose_weights, conf_threshold=yolo_pose_conf)

            self.logger.info("Initializing MediaPipe FaceMesh...")
            # Keep MediaPipe references for backward compatibility with landmark constants
            self.mp_pose = mp.solutions.pose
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            # Face mesh for Eye Aspect Ratio (EAR) calculation
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=self.FACE_MESH_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.FACE_MESH_TRACKING_CONFIDENCE
            )

            # Initialize image preprocessing service
            if ImagePreprocessingService is not None and get_settings is not None:
                try:
                    # Note: settings is already defined above, but refresh it here to ensure we have latest config
                    if settings is None:
                        settings = get_settings()
                    preprocessing_config = {
                        'enable_image_preprocessing': settings.enable_image_preprocessing,
                        'use_clahe': settings.use_clahe,
                        'use_gamma_correction': settings.use_gamma_correction,
                        'use_unsharp_masking': settings.use_unsharp_masking,
                        'use_noise_reduction': settings.use_noise_reduction,
                        'adaptive_preprocessing': settings.adaptive_preprocessing,
                        'clahe_clip_limit': settings.clahe_clip_limit,
                        'clahe_tile_grid_size': settings.clahe_tile_grid_size,
                        'gamma_value': settings.gamma_value,
                        'unsharp_strength': settings.unsharp_strength,
                        'unsharp_radius': settings.unsharp_radius,
                        'noise_reduction_kernel': settings.noise_reduction_kernel
                    }
                    self.preprocessing_service = ImagePreprocessingService(config=preprocessing_config)
                    self.logger.info("Image preprocessing service initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize image preprocessing service: {e}")
                    self.preprocessing_service = None
            else:
                self.preprocessing_service = None

        # Haar cascade classifiers for eye closure detection
        if self.settings and self.settings.haar_eye_detection_enabled:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.logger.info("Haar cascade classifiers loaded for eye closure detection")
        else:
            self.face_cascade = None
            self.profile_face_cascade = None
            self.eye_cascade = None

        # Cell phone detection confidence threshold (configurable via settings)
        self.cell_phone_confidence = self.settings.cell_phone_confidence if self.settings else 0.40

        # Phase 2: Load inference optimization settings (1.5-1.8x speedup)
        self.yolo_imgsz = settings.yolo_imgsz if settings else 640
        self.yolo_device = settings.yolo_device if settings else 'cpu'

        # GPU Batch Processing Settings - maximize GPU utilization
        self.gpu_batch_size = settings.gpu_batch_size if settings else 8
        self.gpu_batch_enabled = settings.gpu_batch_enabled if settings else True

        # Track if models were pre-loaded (don't close them in cleanup)
        self._models_preloaded = preloaded_models is not None

        # Initialize ObjectDetector (extracted module for YOLO-based object detection)
        self.object_detector = ObjectDetector(
            yolo_model=self.yolo_model,
            settings=self.settings,
            preprocessing_service=self.preprocessing_service,
            get_keypoint_func=self.get_keypoint,
            logger=self.logger,
            yolo_imgsz=self.yolo_imgsz,
            yolo_device=self.yolo_device,
            cell_phone_confidence=self.cell_phone_confidence
        )

        # Initialize FrameAnnotator (extracted module for frame visualization)
        self.frame_annotator = FrameAnnotator(logger=self.logger)

        # CR-004: Build the 4 parallel tracking dicts from ACTIVITY_REGISTRY
        # This ensures all activity names stay in sync across dicts.

        # Activity tracking with temporal filtering
        # Each activity tracks OCR timestamps (ocr_start_time, ocr_end_time) for embedded frame timestamps
        self.activities = {
            name: {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0}
            for name in ACTIVITY_REGISTRY
        }

        # TEMPORAL SUPPRESSION: Track recent activities per person for gesture suppression
        # Format: {person_idx: {'writing': last_timestamp, 'packing_bags': last_timestamp, 'cell_phone': last_timestamp}}
        self.recent_person_activities = {}
        self.temporal_suppression_window = self.settings.temporal_suppression_window if self.settings else 10.0

        # Hand gesture coordination temporal window
        # Suppress coordination failure alerts if both LP and ALP raised hands within this window
        self.hand_gesture_coordination_window = self.settings.hand_gesture_coordination_window if self.settings else 5.0

        # Activity thresholds: minimum duration and required consecutive frames before recording starts
        # OPTIMIZED FOR 0.5 FPS SAMPLING (1 frame every 2 seconds)
        # CR-004: Built from ACTIVITY_REGISTRY so thresholds stay in sync with activity names
        self.activity_thresholds = {}
        for _name, _cfg in ACTIVITY_REGISTRY.items():
            _entry: Dict[str, Any] = {
                'min_duration': _cfg.min_duration,
                'required_consecutive': _cfg.required_consecutive,
                'margin': _cfg.margin,
                'grace_frames': _cfg.grace_frames,
            }
            # Include optional fields only when set (preserves original dict shape per activity)
            if _cfg.region_margin is not None:
                _entry['region_margin'] = _cfg.region_margin
            if _cfg.wrist_inside_margin is not None:
                _entry['wrist_inside_margin'] = _cfg.wrist_inside_margin
            if _cfg.sustained_proximity_seconds is not None:
                _entry['sustained_proximity_seconds'] = _cfg.sustained_proximity_seconds
            self.activity_thresholds[_name] = _entry

        # Consecutive detection counters for temporal filtering
        self.consecutive_detections = {name: 0 for name in ACTIVITY_REGISTRY}

        # Grace period counters - allows brief interruptions without resetting
        self.grace_counters = {name: 0 for name in ACTIVITY_REGISTRY}
        
        # Buffer for pre-activity frames (5 seconds before at sampled rate)
        # Calculate buffer size based on sample_fps: 5 seconds * sample_fps
        buffer_size = max(5, int(5 * self.sample_fps))  # At least 5 frames
        self.frame_buffer = deque(maxlen=buffer_size)
        # CR-005: Parallel buffer storing frame indices (not frame copies) for activity tracking
        self.frame_idx_buffer = deque(maxlen=buffer_size)

        self.previous_pose_landmarks = None
        self.movement_history = deque(maxlen=int(30 * self.sample_fps))  # 30 seconds of movement data
        self.head_tilt_history = deque(maxlen=int(10 * self.sample_fps))  # 10 seconds of head tilt data

        # No-pose sleep detection tracking (for IR mode where YOLO pose fails)
        # Format: {person_idx: {'first_seen': timestamp, 'last_bbox': [x1,y1,x2,y2], 'stable_since': timestamp}}
        self.no_pose_sleep_tracking = {}

        # NOTE: per_person_sleep_tracking and ir_forward_lean_tracking are now managed by SleepDetector
        
        # Wrist proximity tracking for writing detection (per person)
        # Format: {person_idx: {'start_time': timestamp, 'duration': seconds, 'consecutive_frames': int}}
        self.wrist_proximity_tracking = {}

        # Per-person consecutive detection tracking for temporal filtering
        # Format: {person_idx: {activity_type: count}} - uses defaultdict to support all activity types
        self.per_person_consecutive_detections = defaultdict(lambda: defaultdict(int))
        self.per_person_grace_counters = defaultdict(lambda: defaultdict(int))

        # Hand position history for velocity/trajectory analysis
        # Format: {person_idx: {'right_wrist': deque([coords]), 'left_wrist': deque([coords]), 'timestamps': deque([t])}}
        self.hand_position_history = {}
        self.hand_history_max_length = 10  # Track last 10 positions (~20s at 0.5 fps)

        # NOTE: packing_motion_history is now managed by ActivityDetector

        # Hand smoothing buffers for coordinate smoothing (CR-015: moved from lazy init)
        # Format: {(person_idx, hand_side): {'positions': deque, 'timestamps': deque}}
        self.hand_smoothing_buffers = {}

        # Cached frame object detection results (CR-015: moved from lazy init)
        self._cached_frame_objects = None
        self._cached_frame_time = 0

        # CR-007: Temporal role tracking state to prevent LP/ALP role flipping between frames
        self._prev_person_boxes = []  # Previous frame's person bounding boxes
        self._prev_person_roles = {}  # Previous frame's person roles {person_idx: role_info}

        # Landmark stability tracking to detect erratic jumps (poor detection quality)
        # Format: {person_idx: {'right_shoulder': deque([coords]), 'left_shoulder': deque([coords])}}
        self.landmark_stability_history = {}
        self.max_landmark_jump_threshold = 100  # pixels - shoulders shouldn't jump more than this

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
            'alp_hand_gesture': 9,
            'mind_diversion': 10,
            'no_person_detected': 11,
            'alp_not_standing': 12,  # ALP not standing in pre-arrival window
            'eating_drinking': 13  # Eating or drinking activity
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
            'mind_diversion': 'Mind diversion - attention diverted from controls',
            'no_person_detected': 'No person detected in frame',
            'alp_not_standing': 'ALP not standing during pre-arrival window',
            'eating_drinking': 'Eating or drinking detected'
        }
        
        # Evidence rules
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'pose_indicators',
            'sleep': 'pose_indicators',
            'writing': 'hand_near_book_or_wrist_proximity',
            'packing_bags': 'wrist_inside_backpack_bbox_or_hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons',
            'lp_hand_gesture': 'lp_hand_raised_gesture_detected',
            'alp_hand_gesture': 'alp_hand_raised_gesture_detected',
            'mind_diversion': 'attention_diverted_from_controls',  # Sub-type stored in evidence details
            'no_person_detected': 'zero_persons_in_frame',
            'alp_not_standing': 'alp_seated_during_pre_arrival_window',
            'eating_drinking': 'cup_or_bottle_near_face'
        }
        
        # Default crew/trip information (None until set by API input)
        self.trip_id = None
        self.crew_name = None
        self.crew_id = None
        self.crew_role = None

        # Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side)
        self.camera_angle = 1  # Default to LP side

        # Initialize PersonTracker (extracted module for person role tracking)
        self.person_tracker = PersonTracker(
            camera_angle=self.camera_angle,
            logger=self.logger
        )

        # Crew members mapping: role (LP/ALP) -> {name, id, role}
        self.crew_members = {}  # Will be populated from API input

        # Store all activities for final JSON array output
        self.all_activities = []

        # Initialize voting verification service for multi-frame voting
        # This reduces false positives by verifying detections across multiple native frames
        self.current_video_path = video_path  # Track current video for voting
        if VotingVerificationService is not None:
            try:
                # Use separate voting models if available (dual-model optimization:
                # nano for detection, large for voting verification)
                voting_yolo = (preloaded_models.get('yolo_voting') if preloaded_models else None) or self.yolo_model
                voting_yolo_pose = (preloaded_models.get('yolo_pose_voting') if preloaded_models else None) or self.yolo_pose
                self.voting_service = VotingVerificationService(
                    yolo_model=voting_yolo,
                    yolo_pose_model=voting_yolo_pose
                )
                if preloaded_models and preloaded_models.get('yolo_voting') is not None and preloaded_models.get('yolo_voting') is not self.yolo_model:
                    self.logger.info("VotingVerificationService initialized with separate voting model")
                else:
                    self.logger.info("VotingVerificationService initialized (same model as detection)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize VotingVerificationService: {e}")
                self.voting_service = None
        else:
            self.voting_service = None
            self.logger.info("VotingVerificationService not available - voting disabled")

        # Initialize train motion rule engine services
        self.trip_schedule = None  # Will be set via set_trip_schedule()
        self.video_start_time = None  # Will be set via set_video_start_time()
        self._prev_motion_frame = None  # Previous frame for optical flow
        self.current_motion_context = None  # Current train motion state
        self.suppress_no_person_without_schedule = getattr(self.settings, 'suppress_no_person_without_schedule', True) if self.settings else True

        # Rule engine service
        if get_rule_engine_service is not None:
            try:
                self.rule_engine = get_rule_engine_service()
                self.logger.info("RuleEngineService initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize RuleEngineService: {e}")
                self.rule_engine = None
        else:
            self.rule_engine = None

        # Motion resolver service
        if get_motion_resolver_service is not None:
            try:
                self.motion_resolver = get_motion_resolver_service()
                self.logger.info("TrainMotionResolverService initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TrainMotionResolverService: {e}")
                self.motion_resolver = None
        else:
            self.motion_resolver = None

        # OCR timestamp service
        if get_ocr_timestamp_service is not None:
            try:
                self.ocr_service = get_ocr_timestamp_service()
                self.logger.info("OCRTimestampService initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OCRTimestampService: {e}")
                self.ocr_service = None
        else:
            self.ocr_service = None

        # ALP alertness service
        if get_alp_alertness_service is not None:
            try:
                self.alp_alertness_service = get_alp_alertness_service()
                self.logger.info("ALPAlertnessService initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ALPAlertnessService: {e}")
                self.alp_alertness_service = None
        else:
            self.alp_alertness_service = None

        # ---------------------------------------------------------------------------
        # Initialize extracted detector modules
        # These provide modular detection logic extracted from this monitor class.
        # TODO: Gradually migrate detection calls to use these instances.
        # ---------------------------------------------------------------------------
        self.sleep_detector = SleepDetector(
            settings=self.settings,
            sample_fps=self.sample_fps,
            logger=self.logger
        )

        self.activity_detector = ActivityDetector(
            settings=self.settings,
            get_keypoint_func=self._get_keypoint_by_name
        )

        self.gesture_detector = GestureDetector(
            settings=self.settings,
            session_timeout=10.0,
            coordination_window=self.hand_gesture_coordination_window,
            get_keypoint_func=self._get_keypoint_by_name
        )

        self.mind_diversion_detector = MindDiversionDetector(
            settings=self.settings,
            logger=self.logger
        )

        # Evidence manager for clip/report generation (only if run_dir is set)
        if self.run_dir:
            self.evidence_manager = EvidenceManager(
                output_dir=self.output_dir,
                run_dir=self.run_dir,
                save_annotated_frames=self.save_annotated_frames,
                frame_save_interval=self.frame_save_interval,
                logger=self.logger
            )
        else:
            self.evidence_manager = None

        self.logger.info("Extracted detector modules initialized: SleepDetector, ActivityDetector, GestureDetector, MindDiversionDetector")

    def set_trip_schedule(self, trip_schedule: Any) -> None:
        """
        Set the trip schedule for motion-based rule evaluation.

        Args:
            trip_schedule: TripSchedule object or None
        """
        self.trip_schedule = trip_schedule
        if trip_schedule:
            self.logger.info(
                f"Trip schedule set for train {trip_schedule.train_number} "
                f"on {trip_schedule.journey_date} with {len(trip_schedule.halts)} halts"
            )
        else:
            self.logger.info(
                "Trip schedule not available - no_person_detected will be suppressed "
                "(cannot distinguish station halts from running without schedule)"
            )

    def set_video_start_time(self, start_time_str: str) -> None:
        """
        Set the video's actual start time (time of day when recording began).

        This allows calculating real timestamps from video offsets when OCR is unavailable.

        Args:
            start_time_str: Start time in HH:MM:SS format (e.g., "14:30:00")
        """
        self.video_start_time = start_time_str
        if start_time_str:
            self.logger.info(f"Video start time set to {start_time_str} - will use for motion rules when OCR unavailable")
        else:
            self.video_start_time = None

    def _convert_video_to_real_time(self, video_seconds: float) -> str:
        """
        Convert video timestamp (seconds from start) to real time of day.

        If video_start_time is set, calculates actual time by adding video offset.
        Otherwise returns video-relative time (which won't match station schedules).

        Args:
            video_seconds: Seconds from video start

        Returns:
            Time string in HH:MM:SS format
        """
        if self.video_start_time:
            try:
                # Parse video start time
                parts = self.video_start_time.split(':')
                start_hours = int(parts[0])
                start_minutes = int(parts[1])
                start_seconds = int(parts[2]) if len(parts) > 2 else 0
                total_start_seconds = start_hours * 3600 + start_minutes * 60 + start_seconds

                # Add video offset
                total_seconds = total_start_seconds + int(video_seconds)

                # Handle day rollover (wrap at 24 hours)
                total_seconds = total_seconds % (24 * 3600)

                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to parse video_start_time '{self.video_start_time}': {e}")

        # Fallback: return video-relative time
        hours = int(video_seconds // 3600)
        minutes = int((video_seconds % 3600) // 60)
        seconds = int(video_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _resolve_motion_context(self, timestamp_str, frame=None, prev_frame=None):
        """
        Resolve current train motion context from timestamp.

        Enhanced with optical flow verification for unscheduled stops.

        Args:
            timestamp_str: Current timestamp in HH:MM:SS format
            frame: Current video frame (optional, for optical flow)
            prev_frame: Previous video frame (optional, for optical flow)

        Returns:
            TrainMotionContext or None
        """
        if self.motion_resolver is None or self.trip_schedule is None:
            return None

        try:
            # Use optical flow-enhanced method if frame is provided
            if frame is not None and hasattr(self.motion_resolver, 'resolve_motion_state_with_optical_flow'):
                context = self.motion_resolver.resolve_motion_state_with_optical_flow(
                    timestamp_str, self.trip_schedule, frame, prev_frame
                )
            else:
                # Fall back to standard schedule-based resolution
                context = self.motion_resolver.resolve_motion_state(
                    timestamp_str, self.trip_schedule
                )
            return context
        except Exception as e:
            self.logger.warning(f"Error resolving motion context: {e}")
            return None

    def _extract_ocr_timestamp(self, frame):
        """
        Extract timestamp from video frame using OCR.

        Args:
            frame: Video frame (numpy array)

        Returns:
            Extracted timestamp string (HH:MM:SS) or None
        """
        if self.ocr_service is None:
            return None

        try:
            result = self.ocr_service.extract_timestamp(frame)
            if result.success:
                return result.timestamp
            return None
        except Exception as e:
            self.logger.debug(f"OCR extraction failed: {e}")
            return None

    def _apply_motion_rules(self, activities_map, motion_context):
        """
        Apply motion-based rules to filter activities.

        Args:
            activities_map: Dictionary of activity_name -> detected (bool)
            motion_context: Current TrainMotionContext

        Returns:
            Filtered activities_map with exemptions applied
        """
        if self.rule_engine is None:
            return activities_map

        try:
            return self.rule_engine.get_filtered_activities_map(
                activities_map, motion_context
            )
        except Exception as e:
            self.logger.warning(f"Error applying motion rules: {e}")
            return activities_map

    def _check_alp_standing(self, persons_data, person_roles, frame_shape):
        """
        Check if ALP is standing (for pre-arrival alertness detection).

        Args:
            persons_data: Per-person detection data from process_all_persons_activities
            person_roles: Person role assignments
            frame_shape: Frame shape (height, width, channels)

        Returns:
            bool: True if ALP is NOT standing (violation), False otherwise
        """
        if self.alp_alertness_service is None:
            return False

        # Find ALP person
        alp_person_idx = None
        for person_idx, role_info in person_roles.items():
            if role_info.get('role') == 'ALP':
                alp_person_idx = person_idx
                break

        if alp_person_idx is None:
            # No ALP identified - can't check standing
            return False

        # Get pose keypoints for ALP
        person_data = persons_data.get(alp_person_idx)
        if person_data is None:
            return False

        pose_keypoints = person_data.get('pose_landmarks')
        if pose_keypoints is None:
            return False

        try:
            result = self.alp_alertness_service.is_alp_standing(
                pose_keypoints, frame_shape, 'ALP'
            )
            # Return True if NOT standing (violation)
            return not result.get('is_standing', False)
        except Exception as e:
            self.logger.debug(f"ALP standing check failed: {e}")
            return False

    def get_keypoint(self, landmarks: Any, keypoint_name: str) -> Any:
        """Get a keypoint from landmarks by name (works with both YOLO and MediaPipe formats).

        This method provides backward compatibility for code that was written for MediaPipe
        by mapping keypoint names to YOLO indices.

        Args:
            landmarks: YoloPoseLandmarks or MediaPipe NormalizedLandmarkList
            keypoint_name: String name like 'nose', 'left_wrist', 'RIGHT_SHOULDER', etc.
                          Case-insensitive, supports both 'left_wrist' and 'LEFT_WRIST'

        Returns:
            Landmark object with x, y, z, visibility attributes

        Example:
            nose = self.get_keypoint(landmarks, 'nose')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')
        """
        return self._get_keypoint_by_name(landmarks, keypoint_name)

    def update_per_person_detection(self, person_idx: int, activity_type: str, detected: bool, timestamp_sec: float) -> bool:
        """
        Update per-person consecutive detection counters with temporal filtering.

        Args:
            person_idx: Person index (0, 1, 2, ...)
            activity_type: Activity name ('cell_phone', 'writing', 'packing_bags')
            detected: Boolean - was activity detected in current frame?
            timestamp_sec: Current timestamp

        Returns:
            bool: True if activity should trigger alert (threshold met)
        """
        # Access tracking for this person (defaultdict auto-initializes for any activity type)
        person_counters = self.per_person_consecutive_detections[person_idx]
        person_grace = self.per_person_grace_counters[person_idx]

        required_consecutive = self.activity_thresholds[activity_type]['required_consecutive']
        grace_frames = self.activity_thresholds[activity_type]['grace_frames']

        if detected:
            person_counters[activity_type] += 1
            person_grace[activity_type] = 0

            if person_counters[activity_type] >= required_consecutive:
                return True  # Trigger activity
        else:
            if person_counters[activity_type] > 0:
                person_grace[activity_type] += 1

                if person_grace[activity_type] > grace_frames:
                    person_counters[activity_type] = 0
                    person_grace[activity_type] = 0

        return False

    def sample_video_frames(self, video_path: str, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> Generator[Tuple[int, float, Any, int], None, None]:
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
        with video_capture_context(video_path) as cap:
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            
            # Determine frame range
            start_frame = start_frame if start_frame is not None else 0
            end_frame = end_frame if end_frame is not None else total_frames
            
            # Calculate stride: how many frames to skip between samples
            step = max(1, int(round(native_fps / max(1e-6, float(self.sample_fps)))))
            
            self.logger.debug(f"[Frame Sampling] Native FPS: {native_fps:.2f}, Sample FPS: {self.sample_fps}")
            self.logger.debug(f"[Frame Sampling] Step: {step} (sampling 1 frame every {step} frames)")
            self.logger.debug(f"[Frame Sampling] Frame range: {start_frame} - {end_frame}")
            self.logger.debug(f"[Frame Sampling] Expected sampled frames: ~{((end_frame - start_frame) // step)}")
            
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
            
            self.logger.debug(f"[Frame Sampling] Completed sampling, total samples: {sampled_idx}")
        
    # Sleep detection methods now accessed via self.sleep_detector directly

    # Activity detection methods now accessed via self.activity_detector directly

    def check_hands_below_shoulders(self, pose_landmarks: Any) -> bool:
        """Check if both hands are below shoulder level.

        A more relaxed check than "hands in lap" for various camera angles.

        Args:
            pose_landmarks: YoloPoseLandmarks or MediaPipe pose landmarks

        Returns:
            bool: True if both hands are below shoulders, False otherwise
        """
        try:
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
            right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')

            if any(p is None for p in [left_shoulder, right_shoulder, left_wrist, right_wrist]):
                return False

            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            return left_wrist.y > shoulder_y and right_wrist.y > shoulder_y

        except Exception as e:
            self.logger.debug(f"Exception in check_hands_below_shoulders: {e}")
            return False

    def detect_writing_by_wrist_proximity(self, pose_landmarks: Any, frame_shape: Tuple[int, ...], person_idx: int, timestamp_sec: float) -> bool:
        """Detect writing activity based on wrist/elbow proximity + head posture heuristic.

        When both wrists (or elbows as fallback) are close together AND head is tilted down
        (typical writing posture) for a sustained duration, it indicates writing activity.
        This method serves as a fallback when book detection doesn't trigger but person is clearly writing.

        Required Conditions (both must be true):
        1. Wrist/Elbow proximity: Left and right wrists within 300px (or elbows within 450px)
        2. Head posture: Head tilted down (nose below eye line)
        3. Temporal: Sustained for 1+ seconds across 2+ consecutive frames

        Args:
            pose_landmarks: MediaPipe pose landmarks (must include wrist/elbow + head keypoints)
            frame_shape: Tuple of (height, width) of the frame
            person_idx: Index of the person being analyzed
            timestamp_sec: Current timestamp in seconds

        Returns:
            bool: True if writing detected by pose-based heuristic, False otherwise
        """
        # Calculate distance between wrists (with elbow fallback)
        distance_result = self.activity_detector.calculate_wrist_distance(pose_landmarks, frame_shape)

        # Handle tuple return (distance, source)
        if isinstance(distance_result, tuple):
            distance, source = distance_result
        else:
            # Backward compatibility if function returns single value
            distance = distance_result
            source = 'wrist' if distance is not None else None

        # DEBUG: Log distance calculation
        if distance is None:
            self.logger.debug(f"Person {person_idx}: Wrist/elbow distance = None (landmarks missing)")
            return False
        else:
            self.logger.debug(f"Person {person_idx}: {source.capitalize()} distance = {distance:.1f}px")

        # Initialize tracking for this person if needed
        if person_idx not in self.wrist_proximity_tracking:
            self.wrist_proximity_tracking[person_idx] = {
                'start_time': None,
                'duration': 0.0,
                'consecutive_frames': 0
            }

        # Configurable thresholds - different for wrist vs elbow

        # Select threshold based on detection source
        if source == 'wrist':
            max_distance = self.MAX_WRIST_DISTANCE
        elif source == 'single_wrist':
            max_distance = self.MAX_SINGLE_WRIST_DISTANCE
        else:
            max_distance = self.MAX_ELBOW_DISTANCE

        person_tracking = self.wrist_proximity_tracking[person_idx]

        # Check if distance is within threshold
        if distance <= max_distance:
            # NEW: Check if head is looking down (required for writing posture)
            head_looking_down = self.activity_detector.detect_head_looking_down(pose_landmarks)

            # DEBUG: Log head state
            self.logger.debug(f"Person {person_idx}: Head looking down = {head_looking_down} (source={source})")

            if not head_looking_down:
                # Head not down - reset tracking (not a writing posture)
                if person_tracking['start_time'] is not None:
                    # Was tracking, now stopped because head is up
                    self.logger.debug(
                        f"Person {person_idx}: {source.capitalize()}s close ({distance:.1f}px) but head not down - "
                        f"resetting writing tracker"
                    )
                person_tracking['start_time'] = None
                person_tracking['duration'] = 0.0
                person_tracking['consecutive_frames'] = 0
                return False

            # BOTH conditions met: distance close AND head down
            # Update tracking
            if person_tracking['start_time'] is None:
                # Start new proximity event
                person_tracking['start_time'] = timestamp_sec
                person_tracking['consecutive_frames'] = 1
                self.logger.debug(
                    f"Person {person_idx}: Writing posture started ({source}) - dist={distance:.1f}px, head=down"
                )
            else:
                # Continue existing proximity event
                person_tracking['duration'] = timestamp_sec - person_tracking['start_time']
                person_tracking['consecutive_frames'] += 1
                self.logger.debug(
                    f"Person {person_idx}: Writing posture continuing ({source}) - dist={distance:.1f}px, "
                    f"head=down, frames={person_tracking['consecutive_frames']}, "
                    f"duration={person_tracking['duration']:.1f}s"
                )

            # Check if thresholds are met
            if (person_tracking['consecutive_frames'] >= self.WRITING_REQUIRED_CONSECUTIVE and
                person_tracking['duration'] >= self.WRITING_MIN_DURATION):
                self.logger.info(
                    f"Person {person_idx}: WRITING CONFIRMED via {source} - distance close + head down for "
                    f"{person_tracking['duration']:.1f}s ({person_tracking['consecutive_frames']} frames)"
                )
                return True
        else:
            # Distance too far apart - reset tracking
            if person_tracking['start_time'] is not None:
                self.logger.debug(
                    f"Person {person_idx}: Writing posture lost - {source}s too far ({distance:.1f}px) - "
                    f"resetting tracker"
                )
            person_tracking['start_time'] = None
            person_tracking['duration'] = 0.0
            person_tracking['consecutive_frames'] = 0

        return False

    def detect_writing_by_book_and_posture(self, pose_landmarks: Any, person_bbox: List[int], book_bboxes: List[List[int]], person_idx: int, timestamp_sec: float) -> bool:
        """Fallback writing detection when wrists are not visible.

        Detects writing based on:
        1. Book detected in person's region
        2. Head looking down (reading/writing posture)
        3. Sustained for minimum duration

        This is a fallback when wrist/elbow detection fails.

        Args:
            pose_landmarks: Pose landmarks for this person
            person_bbox: [x1, y1, x2, y2] bounding box of person
            book_bboxes: List of book bounding boxes detected in frame
            person_idx: Index of person being analyzed
            timestamp_sec: Current timestamp in seconds

        Returns:
            bool: True if writing detected via book+posture, False otherwise
        """
        if not book_bboxes or len(book_bboxes) == 0:
            return False

        # Initialize tracking if needed
        tracking_key = f"book_posture_{person_idx}"
        if tracking_key not in self.wrist_proximity_tracking:
            self.wrist_proximity_tracking[tracking_key] = {
                'start_time': None,
                'duration': 0.0,
                'consecutive_frames': 0
            }

        person_tracking = self.wrist_proximity_tracking[tracking_key]

        # Check if any book is in person's region
        person_book_margin = self.PERSON_BOOK_OVERLAP_MARGIN  # Same margin used elsewhere
        book_in_region = False
        for book_bbox in book_bboxes:
            if bbox_overlap_with_margin(book_bbox, person_bbox, person_book_margin):
                book_in_region = True
                break

        if not book_in_region:
            person_tracking['start_time'] = None
            person_tracking['duration'] = 0.0
            person_tracking['consecutive_frames'] = 0
            return False

        # Check head posture (must be looking down toward book)
        head_looking_down = self.activity_detector.detect_head_looking_down(pose_landmarks)

        if not head_looking_down:
            if person_tracking['start_time'] is not None:
                self.logger.debug(
                    f"Person {person_idx}: Book in region but head not down - resetting book+posture tracker"
                )
            person_tracking['start_time'] = None
            person_tracking['duration'] = 0.0
            person_tracking['consecutive_frames'] = 0
            return False

        # Both conditions met: book in region + head down

        if person_tracking['start_time'] is None:
            person_tracking['start_time'] = timestamp_sec
            person_tracking['consecutive_frames'] = 1
            self.logger.debug(
                f"Person {person_idx}: Book+posture writing started - book in region, head down"
            )
        else:
            person_tracking['duration'] = timestamp_sec - person_tracking['start_time']
            person_tracking['consecutive_frames'] += 1
            self.logger.debug(
                f"Person {person_idx}: Book+posture continuing - frames={person_tracking['consecutive_frames']}, "
                f"duration={person_tracking['duration']:.1f}s"
            )

        if (person_tracking['consecutive_frames'] >= self.BOOK_POSTURE_REQUIRED_CONSECUTIVE and
            person_tracking['duration'] >= self.BOOK_POSTURE_MIN_DURATION):
            self.logger.info(
                f"Person {person_idx}: WRITING CONFIRMED via book+posture fallback - "
                f"book in region + head down for {person_tracking['duration']:.1f}s"
            )
            return True

        return False

    # Object detection methods now accessed via self.object_detector directly

    def detect_poses_batch(self, frames: List[Any], batch_size: int = 8, conf_threshold: Optional[float] = None) -> List[Any]:
        """Run YOLO pose detection on multiple frames in a single batch.

        This maximizes GPU utilization by processing multiple frames at once.

        Args:
            frames: List of BGR frames (numpy arrays)
            batch_size: Maximum batch size for inference (default 8)
            conf_threshold: Optional confidence threshold override (default: use model's conf_threshold)

        Returns:
            List of pose result dictionaries, one per frame.
            Format matches self.yolo_pose.process() output:
            {person_idx: {'bbox': [...], 'bbox_confidence': float, 'keypoints': YoloPoseLandmarks}}
        """
        # Import at method level (not inside loop)
        from app.services.yolo_pose_adapter import YoloPoseLandmarks, PersonKeypoints

        if not frames:
            return []

        effective_conf = conf_threshold if conf_threshold is not None else self.yolo_pose.conf_threshold
        self.logger.debug(f"[GPU BATCH] detect_poses_batch: {len(frames)} frames, batch_size={batch_size}, conf={effective_conf}")
        all_poses = []

        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            try:
                # Run batch inference on pose model with device parameter
                batch_results = self.yolo_pose.model(
                    batch_frames,
                    verbose=False,
                    conf=effective_conf,
                    device=self.yolo_device
                )
            except Exception as e:
                self.logger.error(f"[GPU BATCH] Pose detection failed for batch starting at {batch_start}: {e}")
                # Fallback: return empty poses for this batch
                for _ in batch_frames:
                    all_poses.append({})
                continue

            # Process results for each frame in batch
            for frame_idx, (frame, results) in enumerate(zip(batch_frames, batch_results)):
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

        self.logger.debug(f"[GPU BATCH] detect_poses_batch complete: {len(all_poses)} results")
        return all_poses

    # Visualization methods now accessed via self.frame_annotator directly
    
    def draw_mediapipe_outputs(self, frame: Any, pose_results: Any, face_results: Any, pose_sleep_info: Optional[Dict[str, Any]] = None, head_pose_info: Optional[Dict[str, Any]] = None) -> Any:
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
                
                if duration >= self.SLEEP_STRONG_DURATION:
                    duration_text += " - SLEEP DETECTED!"
                    duration_color = (0, 0, 255)
                elif duration >= self.SLEEP_MICROSLEEP_DURATION:
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
    

    def draw_multi_person_mediapipe_outputs(self, frame: Any, persons_data: Dict[int, Dict[str, Any]], face_results: Any) -> Any:
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
                        start = self.get_keypoint(pose_landmarks, start_name)
                        end = self.get_keypoint(pose_landmarks, end_name)

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
                        landmark = self.get_keypoint(pose_landmarks, keypoint_name)

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
                    nose = self.get_keypoint(pose_landmarks, 'nose')
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

    def _get_smoothed_hand_position(self, person_idx: int, hand_side: str, landmark: Any,
                                     w: int, h: int, timestamp_sec: float) -> Tuple[int, int]:
        """Get temporally smoothed hand position to reduce pose estimation noise.

        Uses simple average over last 3 positions (6 seconds @ 0.5fps).

        Args:
            person_idx: Person identifier
            hand_side: 'right' or 'left'
            landmark: Pose landmark with .x, .y attributes (normalised coords)
            w, h: Frame dimensions in pixels
            timestamp_sec: Current timestamp

        Returns:
            tuple: (smoothed_x, smoothed_y) pixel coordinates
        """
        key = (person_idx, hand_side)
        if key not in self.hand_smoothing_buffers:
            self.hand_smoothing_buffers[key] = {
                'positions': deque(maxlen=3),
                'timestamps': deque(maxlen=3)
            }

        buffer = self.hand_smoothing_buffers[key]

        current_x = int(landmark.x * w)
        current_y = int(landmark.y * h)
        buffer['positions'].append((current_x, current_y))
        buffer['timestamps'].append(timestamp_sec)

        if len(buffer['positions']) > 0:
            avg_x = sum(pos[0] for pos in buffer['positions']) / len(buffer['positions'])
            avg_y = sum(pos[1] for pos in buffer['positions']) / len(buffer['positions'])
            return (int(avg_x), int(avg_y))

        return (current_x, current_y)

    def check_hand_object_interaction(self, hand_coords: Tuple[float, float], object_bbox: List[int], margin: int = 50) -> bool:
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

    # NOTE: detect_pose_per_person removed - replaced by YOLO26-Pose
    # NOTE: translate_pose_landmarks removed - not needed with YOLO26-Pose (native multi-person)

    def validate_pose_landmarks(self, pose_landmarks: Any, min_landmarks: Optional[int] = None, min_visibility: Optional[float] = None) -> bool:
        """Validate that pose landmarks are valid and usable for activity detection.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            min_landmarks: Minimum number of landmarks required (default: MIN_POSE_LANDMARKS)
            min_visibility: Minimum average visibility score (default: MIN_POSE_VISIBILITY)
        
        Returns:
            bool: True if landmarks are valid, False otherwise
        """
        if min_landmarks is None:
            min_landmarks = self.MIN_POSE_LANDMARKS
        if min_visibility is None:
            min_visibility = self.MIN_POSE_VISIBILITY
        if pose_landmarks is None:
            return False
        
        # Support both YoloPoseLandmarks (has .landmark) and plain list
        landmark_list = pose_landmarks.landmark if hasattr(pose_landmarks, 'landmark') else pose_landmarks if isinstance(pose_landmarks, list) else None
        if landmark_list is None or len(landmark_list) < min_landmarks:
            return False

        # Validate coordinates are within valid range (0-1 for normalized)
        valid_count = 0
        total_visibility = 0.0

        for landmark in landmark_list:
            # Check if coordinates are valid (normalized 0-1)
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                valid_count += 1
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                total_visibility += visibility
        
        # Check we have enough valid landmarks
        if valid_count < min_landmarks:
            return False
        
        # Check average visibility
        avg_visibility = total_visibility / valid_count if valid_count > 0 else 0.0
        if avg_visibility < min_visibility:
            return False
        
        return True

    def validate_anatomical_consistency(self, pose_landmarks: Any, frame_shape: Tuple[int, ...]) -> Tuple[bool, str]:
        """
        Validate that pose landmarks follow anatomical rules.
        Rejects physically impossible configurations.

        Returns: (is_valid: bool, reason: str)
        """
        h, w = frame_shape[:2]

        try:
            # Get key landmarks
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_elbow = self.get_keypoint(pose_landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(pose_landmarks, 'left_elbow')
            right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
            right_hip = self.get_keypoint(pose_landmarks, 'right_hip')
            left_hip = self.get_keypoint(pose_landmarks, 'left_hip')
            nose = self.get_keypoint(pose_landmarks, 'nose')

            # Rule 1: Shoulders should be roughly horizontal (±30 degrees)
            shoulder_y_diff = abs(right_shoulder.y - left_shoulder.y) * h
            shoulder_x_diff = abs(right_shoulder.x - left_shoulder.x) * w
            if shoulder_x_diff > 0:
                shoulder_slope = shoulder_y_diff / shoulder_x_diff
                if shoulder_slope > 0.6:  # ~30 degrees
                    return False, "Shoulders not horizontal (slope too steep)"

            # Rule 2: Shoulder-elbow-wrist distances must be reasonable
            # Forearm (elbow-wrist) typically 80-120% of upper arm (shoulder-elbow)
            def distance(lm1, lm2):
                dx = (lm1.x - lm2.x) * w
                dy = (lm1.y - lm2.y) * h
                return (dx**2 + dy**2)**0.5

            right_upper_arm = distance(right_shoulder, right_elbow)
            right_forearm = distance(right_elbow, right_wrist)
            left_upper_arm = distance(left_shoulder, left_elbow)
            left_forearm = distance(left_elbow, left_wrist)

            # Check proportions (forearm should be 50-150% of upper arm length)
            if right_upper_arm > 10:  # Avoid division by very small numbers
                right_ratio = right_forearm / right_upper_arm
                if right_ratio < 0.5 or right_ratio > 1.5:
                    return False, f"Right arm proportions invalid ({right_ratio:.2f})"

            if left_upper_arm > 10:
                left_ratio = left_forearm / left_upper_arm
                if left_ratio < 0.5 or left_ratio > 1.5:
                    return False, f"Left arm proportions invalid ({left_ratio:.2f})"

            # Rule 3: Nose should be above shoulders (not inverted person)
            avg_shoulder_y = (right_shoulder.y + left_shoulder.y) / 2
            if nose.y > avg_shoulder_y + 0.1:  # Nose more than 10% below shoulders
                return False, "Nose below shoulders (inverted detection)"

            # Rule 4: Hips should be below shoulders
            avg_hip_y = (right_hip.y + left_hip.y) / 2
            if avg_hip_y < avg_shoulder_y:
                return False, "Hips above shoulders (inverted detection)"

            # Rule 5: High visibility required for key landmarks
            if (right_shoulder.visibility < 0.5 or left_shoulder.visibility < 0.5 or
                nose.visibility < 0.5):
                return False, "Low visibility for critical landmarks"

            return True, "Valid"

        except (IndexError, AttributeError) as e:
            return False, f"Missing landmarks: {e}"

    def check_landmark_stability(self, person_idx: int, pose_landmarks: Any, frame_shape: Tuple[int, ...]) -> Tuple[bool, float]:
        """
        Check if landmarks are stable over time (not jumping erratically).
        Erratic jumps indicate poor detection quality.

        Returns: (is_stable: bool, max_jump: float)
        """
        h, w = frame_shape[:2]

        if person_idx not in self.landmark_stability_history:
            self.landmark_stability_history[person_idx] = {
                'right_shoulder': deque(maxlen=3),
                'left_shoulder': deque(maxlen=3)
            }

        history = self.landmark_stability_history[person_idx]

        # Get current shoulder positions (most stable body parts)
        right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
        left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')

        right_pos = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))

        history['right_shoulder'].append(right_pos)
        history['left_shoulder'].append(left_pos)

        # Need at least 2 positions to check stability
        if len(history['right_shoulder']) < 2:
            return True, 0  # Not enough data, assume stable

        # Calculate maximum jump between consecutive frames
        max_jump = 0
        for key in ['right_shoulder', 'left_shoulder']:
            positions = list(history[key])
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                jump = (dx**2 + dy**2)**0.5
                max_jump = max(max_jump, jump)

        # Shoulders shouldn't jump more than 100px between frames (person sitting, camera static)
        is_stable = max_jump < self.max_landmark_jump_threshold

        return is_stable, max_jump

    def detect_hand_gesture(self, pose_landmarks: Any, frame_shape: Tuple[int, ...], person_roles: Dict[int, Dict[str, Any]], yolo_person_boxes: Optional[List[List[int]]] = None, 
                           person_activities: Optional[Dict[str, Any]] = None, backpack_detections: Optional[List[Any]] = None, 
                           person_idx: Optional[int] = None, current_timestamp: Optional[float] = None, frame_number: Optional[int] = None) -> Tuple[bool, bool, Dict[str, Any]]:
        """Detect hand gesture (raised hand) for LP/ALP hand exchange signal.
        
        CRITICAL: This function ensures pose landmarks belong to the SAME person
        we're analyzing by matching pose to YOLO person bounding boxes.
        
        The gesture should be detected only when ONE person is doing it (not both).
        
        ROBUST FALSE POSITIVE PREVENTION:
        - Filters out hands reaching toward/operating control panels
        - Only detects deliberate hand-raising gestures (signaling)
        - Uses control panel proximity, forward reach detection, and arm geometry
        - Context-aware filtering: suppresses gestures during packing bags activity
        - Object proximity detection: checks if hand is near backpack/bag
        - Temporal suppression: maintains recent activity history per person
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (tracks 1 person)
            frame_shape: (height, width) of the frame
            person_roles: Dictionary of person roles from identify_person_roles()
                         Format: {person_idx: {'bbox': [x1, y1, x2, y2], 'role': 'LP'/'ALP', ...}}
            yolo_person_boxes: List of YOLO person bounding boxes (for validation)
            person_activities: Dictionary of current person activities (for context-aware filtering)
            backpack_detections: List of backpack bounding boxes (for object proximity detection)
            person_idx: Person index for temporal history tracking
            current_timestamp: Current timestamp in seconds for temporal suppression
            frame_number: Frame number for logging/debugging
            
        Returns:
            tuple: (lp_gesture_detected, alp_gesture_detected, debug_info)
                   Returns (False, False, {}) if both are gesturing or no one is
        """
        if not pose_landmarks or not person_roles:
            return False, False, {}
        
        # Validate pose landmarks before using (RELAXED for hand gesture detection)
        # For hand gestures, we only need key landmarks (wrist, elbow, shoulder) so we relax requirements
        # Overhead cameras may have lower visibility for some landmarks
        if not self.validate_pose_landmarks(pose_landmarks, min_landmarks=8, min_visibility=0.25):
            return False, False, {}

        # Validate anatomical consistency (reject physically impossible poses)
        is_valid, reason = self.validate_anatomical_consistency(pose_landmarks, frame_shape)
        if not is_valid:
            return False, False, {'anatomical_validation_failed': reason}

        # Check landmark stability (reject jittery detections)
        if person_idx is not None:
            is_stable, max_jump = self.check_landmark_stability(person_idx, pose_landmarks, frame_shape)
            if not is_stable:
                return False, False, {
                    'unstable_landmarks': True,
                    'max_jump': max_jump,
                    'threshold': self.max_landmark_jump_threshold
                }

        h, w = frame_shape[:2]
        # Get key body landmarks
        try:
            right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_elbow = self.get_keypoint(pose_landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(pose_landmarks, 'left_elbow')
            right_hip = self.get_keypoint(pose_landmarks, 'right_hip')
            left_hip = self.get_keypoint(pose_landmarks, 'left_hip')
            nose = self.get_keypoint(pose_landmarks, 'nose')
        except (IndexError, AttributeError, ValueError):
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
        
        matched_person_idx = None
        matched_role = None
        best_overlap_score = 0  # Initialize for both single and multi-person cases
        
        # OPTIMIZATION: If person_roles contains only ONE person, use it directly
        # This is the common case when called from process_all_persons_activities()
        if len(person_roles) == 1:
            matched_person_idx = list(person_roles.keys())[0]
            matched_role = person_roles[matched_person_idx].get('role', 'UNKNOWN')
            # For single person, assume perfect match (overlap_score = 1.0)
            best_overlap_score = 1.0
        else:
            # MULTI-PERSON CASE: Match pose to person bounding box
            # Strategy: Check which person's bounding box contains the pose landmarks
            # We'll use nose/shoulders as the key landmarks to match
            
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
                
                # Require at least 40% of key landmarks to be within the box (RELAXED from 50% for overhead camera)
                # Overhead cameras may have less precise bounding boxes
                if overlap_score > best_overlap_score and overlap_score >= 0.4:
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
        # Allow some margin for extended arms (especially for raised hands)
        matched_bbox = person_roles[matched_person_idx]['bbox']
        mx1, my1, mx2, my2 = matched_bbox
        
        # Expand box for arm extension tolerance (FURTHER INCREASED for overhead camera)
        # Overhead cameras show arms extending more, especially when hands are raised high
        # Raised hands can extend significantly above the person's bbox
        # When two people are close together, wrist may appear further from bbox center
        box_width = mx2 - mx1
        box_height = my2 - my1
        margin_x = box_width * 1.0  # INCREASED from 0.8 to 1.0 (100% of box width)
        margin_y = box_height * 1.5  # INCREASED from 1.2 to 1.5 for high raised hands
        
        expanded_x1 = mx1 - margin_x
        expanded_y1 = my1 - margin_y  # Allow significant upward extension
        expanded_x2 = mx2 + margin_x
        expanded_y2 = my2 + margin_y
        
        # Check if wrists are within expanded box (for extended arms, especially raised hands)
        # RELAXED: For raised hands, we're more lenient - wrist can be above the expanded box
        right_wrist_in_expanded = (
            expanded_x1 <= right_wrist_coords[0] <= expanded_x2 and 
            right_wrist_coords[1] >= expanded_y1  # Allow wrist above expanded top (raised hand)
        )
        left_wrist_in_expanded = (
            expanded_x1 <= left_wrist_coords[0] <= expanded_x2 and 
            left_wrist_coords[1] >= expanded_y1  # Allow wrist above expanded top (raised hand)
        )
        
        # ==================================================================================
        # TEMPORAL SUPPRESSION (FALSE POSITIVE PREVENTION #0 - HIGHEST PRIORITY)
        # ==================================================================================
        # Check if this person was recently engaged in work activities (writing, packing, phone)
        # Suppress hand gestures for N seconds after last work activity detection
        # This handles cases where work activity isn't detected in current frame but was recent
        # ==================================================================================
        if person_idx is not None and current_timestamp is not None:
            if matched_person_idx in self.recent_person_activities:
                recent_acts = self.recent_person_activities[matched_person_idx]
                
                # Check each work activity type
                for activity_type in ['writing', 'packing_bags', 'cell_phone']:
                    if activity_type in recent_acts:
                        time_since_activity = current_timestamp - recent_acts[activity_type]
                        
                        if time_since_activity < self.temporal_suppression_window:
                            frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
                            # gesture_logger.info(f"{frame_info}TEMPORAL SUPPRESSION - Person {matched_person_idx} ({matched_role}) - {activity_type} detected {time_since_activity:.1f}s ago")
                            return False, False, {
                                'suppressed': True,
                                'reason': f'Recent {activity_type} activity ({time_since_activity:.1f}s ago)',
                                'matched_person_idx': matched_person_idx,
                                'matched_role': matched_role,
                                'time_since_activity': time_since_activity
                            }
        
        # ==================================================================================
        # CONTEXT-AWARE FILTERING (FALSE POSITIVE PREVENTION #1)
        # ==================================================================================
        # SUPPRESS hand gesture detection if person is engaged in activities where
        # hands are naturally raised (packing bags, writing, using phone, etc.)
        # This prevents false positives when someone raises their hand while working
        # ==================================================================================
        if person_activities:
            # Check if this person is doing packing bags activity
            if person_activities.get('packing_bags', False):
                frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
                # gesture_logger.info(f"{frame_info}GESTURE SUPPRESSED - Person {matched_person_idx} ({matched_role}) - packing bags activity detected")
                return False, False, {
                    'suppressed': True,
                    'reason': 'Person engaged in packing bags activity',
                    'matched_person_idx': matched_person_idx,
                    'matched_role': matched_role
                }
            
            # Also suppress for writing (may raise hand while writing in logbook)
            if person_activities.get('writing', False):
                frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
                # gesture_logger.info(f"{frame_info}GESTURE SUPPRESSED - Person {matched_person_idx} ({matched_role}) - writing activity detected")
                return False, False, {
                    'suppressed': True,
                    'reason': 'Person engaged in writing activity',
                    'matched_person_idx': matched_person_idx,
                    'matched_role': matched_role
                }
            
            # Suppress for eating/drinking (hand raised to mouth)
            if person_activities.get('eating_drinking', False):
                frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
                return False, False, {
                    'suppressed': True,
                    'reason': 'Person engaged in eating/drinking activity',
                    'matched_person_idx': matched_person_idx,
                    'matched_role': matched_role
                }

            # Suppress for cell phone use (hand raised to ear or viewing phone)
            if person_activities.get('cell_phone', False):
                frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
                # gesture_logger.info(f"{frame_info}GESTURE SUPPRESSED - Person {matched_person_idx} ({matched_role}) - cell phone use detected")
                return False, False, {
                    'suppressed': True,
                    'reason': 'Person using cell phone',
                    'matched_person_idx': matched_person_idx,
                    'matched_role': matched_role
                }
        
        # ==================================================================================
        # OBJECT PROXIMITY DETECTION (FALSE POSITIVE PREVENTION #2)
        # ==================================================================================
        # Check if hands are near backpack/bag objects
        # If hand is near a backpack, it's likely packing activity, NOT a signal gesture
        # ==================================================================================
        if backpack_detections and len(backpack_detections) > 0:
            # Check proximity threshold (pixels) - INCREASED for better detection
            proximity_threshold = 250  # Hand within 250px of backpack center (increased from 150px)
            
            frame_info = f"[Frame {frame_number}] " if frame_number is not None else ""
            self.logger.debug(f"{frame_info}GESTURE DEBUG - Checking {len(backpack_detections)} backpack(s) for person {matched_person_idx} ({matched_role})")
            
            for backpack_bbox in backpack_detections:
                bx1, by1, bx2, by2 = backpack_bbox[:4]
                backpack_center_x = (bx1 + bx2) / 2
                backpack_center_y = (by1 + by2) / 2
                
                # Check if either wrist is near the backpack
                right_dist = ((right_wrist_coords[0] - backpack_center_x) ** 2 + 
                             (right_wrist_coords[1] - backpack_center_y) ** 2) ** 0.5
                left_dist = ((left_wrist_coords[0] - backpack_center_x) ** 2 + 
                            (left_wrist_coords[1] - backpack_center_y) ** 2) ** 0.5
                
                self.logger.debug(f"{frame_info}GESTURE DEBUG - Backpack at ({backpack_center_x:.0f}, {backpack_center_y:.0f})")
                self.logger.debug(f"{frame_info}GESTURE DEBUG - Right wrist at {right_wrist_coords}, dist: {right_dist:.0f}px")
                self.logger.debug(f"{frame_info}GESTURE DEBUG - Left wrist at {left_wrist_coords}, dist: {left_dist:.0f}px")
                
                if right_dist < proximity_threshold or left_dist < proximity_threshold:
                    # gesture_logger.info(f"{frame_info}GESTURE SUPPRESSED - Person {matched_person_idx} ({matched_role}) - hand near backpack (right: {right_dist:.0f}px, left: {left_dist:.0f}px)")
                    return False, False, {
                        'suppressed': True,
                        'reason': 'Hand near backpack object (likely packing, not signaling)',
                        'matched_person_idx': matched_person_idx,
                        'matched_role': matched_role,
                        'right_dist_to_backpack': right_dist,
                        'left_dist_to_backpack': left_dist
                    }
        
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
        
        # CRITICAL: Detect control panel region (assume control panel is in front of operator)
        # In locomotive cab, control panel is typically in the upper-front area
        # We detect if hand is reaching forward toward this region by checking if:
        # 1. Hand is in front of body center (forward reach)
        # 2. Hand is not significantly laterally extended
        
        # For RIGHT hand:
        # - Control panel reach: wrist X is roughly between shoulder X and far right of person bbox
        # - Signaling gesture: wrist X is laterally away from body center (to the right for right hand)
        
        # Calculate if hand is in "control panel reach zone" (forward reach pattern)
        # For a seated operator, control panel is typically in the frontal zone
        # We define this as: hand in upper portion of frame AND not laterally extended
        
        # M-04: Scale control zone pixel thresholds by person bbox height
        # At 1080p with ~500px person bbox, original values were 30, 100, 50, 30.
        # Scale proportionally for other resolutions.
        bbox_height = max(my2 - my1, 1)
        cz_wrist_shoulder_min = max(20, int(0.06 * bbox_height))   # ~30px at 500px bbox
        cz_wrist_shoulder_max = max(20, int(0.20 * bbox_height))   # ~100px at 500px bbox
        cz_wrist_elbow_max = max(20, int(0.10 * bbox_height))      # ~50px at 500px bbox
        cz_elbow_shoulder_offset = max(20, int(0.06 * bbox_height)) # ~30px at 500px bbox

        # Right hand: Check if it's in control operation zone
        # This identifies forward reaches to operate controls vs upward signaling
        # IMPROVED: More specific detection - only filter if it's CLEARLY a forward reach, not upward raise
        # For overhead cameras, we need to be more lenient to avoid false negatives
        right_in_control_zone = (
            # Hand is in reasonable vertical range (not extremely high for signaling)
            right_wrist_coords[1] > (my1 + (my2 - my1) * 0.2) and
            right_wrist_coords[1] < (my1 + (my2 - my1) * 0.8) and

            # CRITICAL: Wrist above shoulder but NOT too far (scaled control panel range)
            # True hand signals are typically above this range
            cz_wrist_shoulder_min < right_wrist_shoulder_vertical < cz_wrist_shoulder_max and

            # IMPROVED: Elbow-wrist distance check - must be SMALL (forward reach, not vertical extension)
            # Control panel: elbow NOT significantly below wrist (forward reach pattern)
            # Hand signal: elbow MUST be significantly below wrist (vertical arm extension)
            right_wrist_elbow_distance < cz_wrist_elbow_max and

            # ADDITIONAL: Elbow must be BELOW shoulder (forward reach pattern)
            # If elbow is at or above shoulder, it's likely an upward raise, not forward reach
            right_elbow_coords[1] > right_shoulder_coords[1] + cz_elbow_shoulder_offset
        )

        # Left hand: Check if it's in control operation zone
        # IMPROVED: More specific detection - only filter if it's CLEARLY a forward reach, not upward raise
        left_in_control_zone = (
            # Hand is in reasonable vertical range (not extremely high for signaling)
            left_wrist_coords[1] > (my1 + (my2 - my1) * 0.2) and
            left_wrist_coords[1] < (my1 + (my2 - my1) * 0.8) and

            # CRITICAL: Wrist above shoulder but NOT too far (scaled control panel range)
            left_wrist_shoulder_vertical > cz_wrist_shoulder_min and
            left_wrist_shoulder_vertical < cz_wrist_shoulder_max and

            # IMPROVED: Elbow-wrist distance check - must be SMALL (forward reach, not vertical extension)
            left_wrist_elbow_distance < cz_wrist_elbow_max and

            # ADDITIONAL: Elbow must be BELOW shoulder (forward reach pattern)
            left_elbow_coords[1] > left_shoulder_coords[1] + cz_elbow_shoulder_offset
        )
        
        # M-06: Scale gesture detection thresholds by person bbox height
        # Calibrated at 1080p with ~500px person bbox: 80, -30, 20, 150 pixels
        wrist_shoulder_min = max(20, int(0.16 * bbox_height))      # ~80px at 500px bbox
        wrist_elbow_min = -max(20, int(0.06 * bbox_height))        # ~-30px at 500px bbox
        arm_extension_min = max(20, int(0.04 * bbox_height))       # ~20px at 500px bbox
        elbow_shoulder_tolerance = max(20, int(0.30 * bbox_height)) # ~150px at 500px bbox

        # Right hand gesture detection (HAND RAISED TO FACE OR ABOVE)
        # RELAXED thresholds to detect drinking/eating and any hand-to-face actions
        # Detects: hand raised to face level, drinking, eating, signaling gestures
        right_hand_raised = (
            # CRITICAL: Wrist must belong to the same person (within expanded bbox)
            right_wrist_in_expanded and

            # Control zone filter - reject if wrist is inside control zone
            not right_in_control_zone and

            # Hand significantly above shoulder level (scaled by bbox height)
            right_wrist_shoulder_vertical > wrist_shoulder_min and

            # RELAXED: Allow bent arm (drinking position has wrist near elbow level)
            right_wrist_elbow_distance > wrist_elbow_min and

            # RELAXED: Allow arm close to body (drinking/eating position)
            right_arm_extension > arm_extension_min and

            # Allow elbow to be below shoulder (natural drinking/eating position)
            (right_elbow_coords[1] < right_shoulder_coords[1] + elbow_shoulder_tolerance) and

            # Visibility checks (FURTHER RELAXED for overhead cameras)
            right_wrist.visibility > 0.3 and
            right_elbow.visibility > 0.3 and
            right_shoulder.visibility > 0.4 and

            # Within frame bounds
            0 < right_wrist_coords[0] < w and
            0 < right_wrist_coords[1] < h
        )

        # Left hand gesture detection (HAND RAISED TO FACE OR ABOVE)
        # RELAXED thresholds to detect drinking/eating and any hand-to-face actions
        # Detects: hand raised to face level, drinking, eating, signaling gestures
        left_hand_raised = (
            # CRITICAL: Wrist must belong to the same person (within expanded bbox)
            left_wrist_in_expanded and

            # Control zone filter - reject if wrist is inside control zone
            not left_in_control_zone and

            # Hand significantly above shoulder level (scaled by bbox height)
            left_wrist_shoulder_vertical > wrist_shoulder_min and

            # RELAXED: Allow bent arm (drinking position has wrist near elbow level)
            left_wrist_elbow_distance > wrist_elbow_min and

            # RELAXED: Allow arm close to body (drinking/eating position)
            left_arm_extension > arm_extension_min and

            # Allow elbow to be below shoulder (natural drinking/eating position)
            (left_elbow_coords[1] < left_shoulder_coords[1] + elbow_shoulder_tolerance) and

            # Visibility checks (FURTHER RELAXED for overhead cameras)
            left_wrist.visibility > 0.3 and
            left_elbow.visibility > 0.3 and
            left_shoulder.visibility > 0.4 and

            # Within frame bounds
            0 < left_wrist_coords[0] < w and
            0 < left_wrist_coords[1] < h
        )
        
        # Either hand raised counts as gesture
        hand_gesture_detected = right_hand_raised or left_hand_raised

        if not hand_gesture_detected:
            return False, False, {}

        # Analyze velocity and trajectory
        velocity_analysis = self.analyze_hand_velocity_and_trajectory(
            matched_person_idx, pose_landmarks, frame_shape, current_timestamp
        )

        # Velocity gate: use rapid_raise_detected to filter false positives
        # When analysis quality is good, require a rapid raise to confirm gesture
        if velocity_analysis.get('analysis_quality') == 'good':
            rapid_raise = velocity_analysis['rapid_raise_detected']
            self.logger.debug(
                f"[VELOCITY] Person {matched_person_idx}: "
                f"R_vel={velocity_analysis['right_velocity']:.1f}px/s ({velocity_analysis['right_trajectory']}), "
                f"L_vel={velocity_analysis['left_velocity']:.1f}px/s ({velocity_analysis['left_trajectory']}), "
                f"Rapid raise: {rapid_raise}"
            )

            if not rapid_raise:
                self.logger.debug(f"[VELOCITY] No rapid raise detected - suppressing hand gesture (likely control operation)")
                return False, False, {}

        # Return result based on the MATCHED person's role
        if matched_role == 'LP':
            return True, False, {
                'hand_raised': 'right' if right_hand_raised else 'left',
                'shoulder_y': avg_shoulder_y,
                'wrist_y': right_wrist_coords[1] if right_hand_raised else left_wrist_coords[1],
                'right_wrist_shoulder_vertical': right_wrist_shoulder_vertical,
                'left_wrist_shoulder_vertical': left_wrist_shoulder_vertical,
                'right_wrist_elbow_distance': right_wrist_elbow_distance,
                'left_wrist_elbow_distance': left_wrist_elbow_distance,
                'person_role': 'LP',
                'matched_person_idx': matched_person_idx,
                'overlap_score': best_overlap_score,
                'velocity_analysis': velocity_analysis
            }
        elif matched_role == 'ALP':
            return False, True, {
                'hand_raised': 'right' if right_hand_raised else 'left',
                'shoulder_y': avg_shoulder_y,
                'wrist_y': right_wrist_coords[1] if right_hand_raised else left_wrist_coords[1],
                'right_wrist_shoulder_vertical': right_wrist_shoulder_vertical,
                'left_wrist_shoulder_vertical': left_wrist_shoulder_vertical,
                'right_wrist_elbow_distance': right_wrist_elbow_distance,
                'left_wrist_elbow_distance': left_wrist_elbow_distance,
                'person_role': 'ALP',
                'matched_person_idx': matched_person_idx,
                'overlap_score': best_overlap_score,
                'velocity_analysis': velocity_analysis
            }

        # Unknown role
        return False, False, {}

    def _check_hand_gesture_coordination(self, lp_detected, alp_detected, current_time):
        """
        Check for hand gesture coordination failures with temporal window support.

        Prevents false positives when both people raise hands within a time window
        (collaborative discussion) but not in the exact same frame.

        Args:
            lp_detected: LP hand gesture detected in current frame
            alp_detected: ALP hand gesture detected in current frame
            current_time: Current timestamp in seconds

        Returns:
            tuple: (lp_not_coordinating, alp_not_coordinating)
                - lp_not_coordinating: True if ALP raised hand but LP failed to coordinate
                - alp_not_coordinating: True if LP raised hand but ALP failed to coordinate
        """
        # Get last hand raise times from recent activities
        lp_last_raise_time = None
        alp_last_raise_time = None

        for person_idx, activities in self.recent_person_activities.items():
            if 'lp_hand_raise' in activities:
                t = activities['lp_hand_raise']
                if lp_last_raise_time is None or t > lp_last_raise_time:
                    lp_last_raise_time = t
            if 'alp_hand_raise' in activities:
                t = activities['alp_hand_raise']
                if alp_last_raise_time is None or t > alp_last_raise_time:
                    alp_last_raise_time = t

        # Helper: Check if both raised hands within coordination window
        def both_within_window(lp_time, alp_time):
            if lp_time is None or alp_time is None:
                return False
            lp_recent = (current_time - lp_time) <= self.hand_gesture_coordination_window
            alp_recent = (current_time - alp_time) <= self.hand_gesture_coordination_window
            return lp_recent and alp_recent

        # Check coordination with temporal window logic
        lp_not_coordinating = False
        alp_not_coordinating = False

        if alp_detected and not lp_detected:
            # ALP raised hand, LP didn't in current frame
            # Check if LP raised recently (within window)
            if not both_within_window(lp_last_raise_time, alp_last_raise_time):
                lp_not_coordinating = True  # True coordination failure

        if lp_detected and not alp_detected:
            # LP raised hand, ALP didn't in current frame
            # Check if ALP raised recently (within window)
            if not both_within_window(lp_last_raise_time, alp_last_raise_time):
                alp_not_coordinating = True  # True coordination failure

        return lp_not_coordinating, alp_not_coordinating

    def analyze_hand_velocity_and_trajectory(self, person_idx: int, landmarks: Any, frame_shape: Tuple[int, ...], timestamp_sec: float) -> Dict[str, Any]:
        """
        Analyze hand velocity and trajectory patterns to enhance gesture detection.

        Detects rapid hand raises (signaling) vs static positions (control operations).

        Args:
            person_idx: Person index
            landmarks: MediaPipe pose landmarks
            frame_shape: Frame dimensions (h, w, c)
            timestamp_sec: Current timestamp

        Returns:
            dict: Velocity/trajectory analysis results
        """
        import numpy as np
        h, w = frame_shape[:2]

        # Initialize history for this person
        if person_idx not in self.hand_position_history:
            self.hand_position_history[person_idx] = {
                'right_wrist': deque(maxlen=self.hand_history_max_length),
                'left_wrist': deque(maxlen=self.hand_history_max_length),
                'timestamps': deque(maxlen=self.hand_history_max_length)
            }

        history = self.hand_position_history[person_idx]

        # Get current wrist positions
        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        right_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

        # Append current positions
        history['right_wrist'].append(right_coords)
        history['left_wrist'].append(left_coords)
        history['timestamps'].append(timestamp_sec)

        # Need at least 3 positions to analyze velocity
        if len(history['timestamps']) < 3:
            return {
                'right_velocity': 0.0,
                'left_velocity': 0.0,
                'right_trajectory': 'unknown',
                'left_trajectory': 'unknown',
                'rapid_raise_detected': False,
                'analysis_quality': 'insufficient_data'
            }

        # Calculate velocities (pixels per second)
        def calculate_velocity(position_history, timestamps):
            if len(position_history) < 2:
                return 0.0, 'unknown'

            recent_positions = list(position_history)[-3:]
            recent_times = list(timestamps)[-3:]

            dx = recent_positions[-1][0] - recent_positions[0][0]
            dy = recent_positions[-1][1] - recent_positions[0][1]
            dt = recent_times[-1] - recent_times[0]

            if dt == 0:
                return 0.0, 'unknown'

            displacement = np.sqrt(dx**2 + dy**2)
            velocity = displacement / dt  # pixels/second

            # Determine trajectory
            if abs(dy) > abs(dx) * 1.5:
                trajectory = 'upward' if dy < 0 else 'downward'  # Y increases downward
            elif abs(dx) > abs(dy) * 1.5:
                trajectory = 'lateral'
            else:
                trajectory = 'diagonal'

            return velocity, trajectory

        right_vel, right_traj = calculate_velocity(history['right_wrist'], history['timestamps'])
        left_vel, left_traj = calculate_velocity(history['left_wrist'], history['timestamps'])

        # Detect rapid hand raise: velocity > 150 px/s AND upward trajectory
        rapid_raise = (
            (right_vel > 150 and right_traj == 'upward') or
            (left_vel > 150 and left_traj == 'upward')
        )

        return {
            'right_velocity': right_vel,
            'left_velocity': left_vel,
            'right_trajectory': right_traj,
            'left_trajectory': left_traj,
            'rapid_raise_detected': rapid_raise,
            'analysis_quality': 'good' if len(history['timestamps']) >= 5 else 'limited'
        }

    def analyze_packing_hand_motion(self, person_idx: int, landmarks: Any, frame_shape: Tuple[int, ...], timestamp_sec: float, backpack_bbox: List[int]) -> Dict[str, Any]:
        """Analyze hand motion patterns to detect actual packing activity - delegates to ActivityDetector."""
        return self.activity_detector.analyze_packing_hand_motion(
            person_idx, landmarks, frame_shape, timestamp_sec, backpack_bbox
        )

    # NOTE: detect_multi_person_pose_and_gestures removed - replaced by YOLO26-Pose

    def _match_pose_to_roles(self, yolo_pose_results, person_roles):
        """Match YOLO26-Pose detections to identified person roles by bounding box IoU.

        DELEGATION: This method delegates to PersonTracker.match_pose_to_roles().
        Kept for backward compatibility during refactoring transition.
        """
        return self.person_tracker.match_pose_to_roles(yolo_pose_results, person_roles)

    def process_all_persons_activities(self, frame: Any, detections: Dict[str, List[Any]], person_roles: Dict[int, Dict[str, Any]], timestamp_sec: float, face_results: Any = None, frame_number: Optional[int] = None, precomputed_pose_results: Optional[Any] = None, precomputed_sleep_pose_results: Optional[Any] = None, is_dark_frame: Optional[bool] = None) -> Dict[str, Any]:
        """Process all detected persons for ALL activity detections (mind diversion, sleep, etc.)

        This is the MAIN multi-person processing method that:
        1. Runs YOLO26-Pose once to get all persons with keypoints (or uses precomputed results)
        2. Matches YOLO detections to person_roles by bounding box IoU
        3. Detects ALL activities for EACH person (mind diversion, sleep, cell phone, writing, etc.)
        4. Returns aggregated results for all persons

        Args:
            frame: The full frame image (BGR format)
            detections: YOLO detections dictionary containing 'person', 'cell_phone', 'book', etc.
            person_roles: Dictionary of person roles from identify_person_roles()
            timestamp_sec: Current timestamp in seconds
            face_results: MediaPipe face mesh results (optional, for mind diversion detection)
            frame_number: Frame number for logging/debugging (optional)
            precomputed_pose_results: Pre-computed YOLO pose results (optional, for GPU batch optimization)
            precomputed_sleep_pose_results: Low-confidence YOLO pose results for sleep detection fallback (optional)
            is_dark_frame: Whether the frame is dark/IR (optional, computed from brightness if None)

        Returns:
            dict: {
                'persons': {
                    person_idx: {
                        'pose_landmarks': translated landmarks,
                        'role': 'LP'/'ALP'/etc.,
                        'bbox': [x1, y1, x2, y2],
                        'activities': {
                            'mind_diversion': bool,
                            'sleep': bool,
                            'microsleep': bool,
                            'cell_phone': bool,
                            'writing': bool,
                            'packing_bags': bool,
                            'lp_hand_gesture': bool,
                            'alp_hand_gesture': bool
                        },
                        'debug_info': {
                            'head_pose': {...},
                            'sleep_info': {...},
                            'gesture_debug': {...}
                        }
                    }
                },
                'aggregated': {
                    'mind_diversion_detected': bool,
                    'sleep_detected': bool,
                    'microsleep_detected': bool,
                    'cell_phone_detected': bool,
                    'writing_detected': bool,
                    'packing_detected': bool,
                    'lp_hand_gesture_detected': bool,
                    'alp_hand_gesture_detected': bool,
                    'performing_person': int (person_idx who performed the activity, or -1 for aggregated)
                }
            }
        """
        if not person_roles or len(person_roles) == 0:
            # No persons detected, return empty results
            return {
                'persons': {},
                'aggregated': {
                    'mind_diversion_detected': False,
                    'sleep_detected': False,
                    'microsleep_detected': False,
                    'cell_phone_detected': False,
                    'writing_detected': False,
                    'packing_detected': False,
                    'lp_hand_gesture_detected': False,
                    'alp_hand_gesture_detected': False,
                    'performing_person': -1
                }
            }
        
        h, w = frame.shape[:2]
        persons_data = {}

        # ============ YOLO26-POSE: Single inference for all persons ============
        # Run YOLO26-Pose once on the full frame to get all persons with keypoints
        # This replaces the per-person MediaPipe cropping loop for better performance
        # If precomputed_pose_results is provided (from GPU batch inference), use it directly
        if precomputed_pose_results is not None:
            yolo_pose_results = precomputed_pose_results
        else:
            yolo_pose_results = self.yolo_pose.process(frame)

        # Match YOLO pose detections to person_roles by bounding box IoU
        matched_poses = self._match_pose_to_roles(yolo_pose_results, person_roles)

        # Match low-confidence sleep poses as fallback for persons not found at normal confidence
        matched_sleep_poses = {}
        if precomputed_sleep_pose_results is not None and precomputed_sleep_pose_results:
            matched_sleep_poses = self._match_pose_to_roles(precomputed_sleep_pose_results, person_roles)

        # Dark frame flag for IR forward lean detection (compute if not passed by caller)
        if is_dark_frame is None:
            is_dark_frame = False
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                frame_brightness = float(np.mean(gray)) / 255.0
                is_dark_frame = frame_brightness < self.settings.yolo_dark_frame_brightness_threshold
            except Exception as e:
                self.logger.debug(f"[DARK FRAME] Failed to check frame brightness: {e}")

        # Process each person individually
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue

            bbox = person_data['bbox']  # [x1, y1, x2, y2]

            # Get matched pose keypoints for this person
            # YOLO26-Pose provides full-frame coordinates directly (no cropping/translation needed)
            try:
                has_pose = True
                translated_landmarks = None

                if person_idx not in matched_poses:
                    has_pose = False
                else:
                    # Get the matched YoloPoseLandmarks (MediaPipe-compatible interface)
                    translated_landmarks = matched_poses[person_idx]

                    # Validate landmarks are valid before using for activity detection
                    if translated_landmarks is None or len(translated_landmarks.landmark) == 0:
                        has_pose = False
                        translated_landmarks = None
                    else:
                        # Check if at least some keypoints have good visibility
                        visible_count = sum(1 for lm in translated_landmarks.landmark if lm.visibility > 0.3)
                        if visible_count < 5:
                            has_pose = False
                            translated_landmarks = None

                h, w = frame.shape[:2]

                if has_pose and translated_landmarks is not None:
                    # ============ KEYPOINT CONSISTENCY VALIDATION ============
                    # Verify that torso center falls within (or near) the person's bbox
                    # This catches cases where pose matching assigned wrong skeleton to person
                    left_shoulder = translated_landmarks.landmark[5]  # YOLO index
                    right_shoulder = translated_landmarks.landmark[6]  # YOLO index

                    # Calculate torso center in pixel coords
                    torso_center_x = ((left_shoulder.x + right_shoulder.x) / 2) * w
                    torso_center_y = ((left_shoulder.y + right_shoulder.y) / 2) * h

                    # Check if torso center is within expanded bbox (1.5x margin)
                    bbox_margin = 0.5  # 50% expansion
                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    expanded_x1 = x1 - bbox_width * bbox_margin
                    expanded_x2 = x2 + bbox_width * bbox_margin
                    expanded_y1 = y1 - bbox_height * bbox_margin
                    expanded_y2 = y2 + bbox_height * bbox_margin

                    torso_in_bbox = (
                        expanded_x1 <= torso_center_x <= expanded_x2 and
                        expanded_y1 <= torso_center_y <= expanded_y2
                    )

                    if not torso_in_bbox:
                        self.logger.warning(
                            f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                            f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) outside expanded bbox "
                            f"[{expanded_x1:.0f}-{expanded_x2:.0f}, {expanded_y1:.0f}-{expanded_y2:.0f}] - SKIPPING"
                        )
                        has_pose = False
                        translated_landmarks = None
                    else:
                        self.logger.debug(
                            f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                            f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) VALID within bbox"
                        )

                # ============ NO-POSE PATH: Limited detection for persons without pose ============
                if not has_pose:
                    # Run object-based eating/drinking detection and no-pose sleep tracking
                    no_pose_activities = {
                        'mind_diversion': False,
                        'sleep': False,
                        'microsleep': False,
                        'cell_phone': False,
                        'writing': False,
                        'packing_bags': False,
                        'lp_hand_gesture': False,
                        'alp_hand_gesture': False,
                        'eating_drinking': False
                    }
                    no_pose_debug = {
                        'head_pose': {},
                        'sleep_info': {'no_pose': True},
                        'gesture_debug': {}
                    }

                    # --- Object-based eating/drinking (cup directly overlaps person bbox) ---
                    if getattr(self.settings, 'eating_drinking_detection_enabled', True):
                        cup_bottle_bboxes = []
                        cup_conf_threshold = getattr(self.settings, 'eating_drinking_cup_confidence', 0.25)
                        for roi_det in detections.get('roi_detections', []):
                            if roi_det['class'] in ('cup', 'bottle') and roi_det['confidence'] > cup_conf_threshold:
                                det_bbox = roi_det['bbox']
                                if bbox_overlap_with_margin(det_bbox, bbox, 50):
                                    cup_bottle_bboxes.append(det_bbox)
                        for cb_xyxy in detections.get('cup_bottle', []):
                            cb_bbox = [float(cb_xyxy[0]), float(cb_xyxy[1]), float(cb_xyxy[2]), float(cb_xyxy[3])]
                            if bbox_overlap_with_margin(cb_bbox, bbox, 50):
                                cup_bottle_bboxes.append(cb_bbox)

                        if cup_bottle_bboxes:
                            # Cup overlaps person bbox directly → eating/drinking (no hand-face check possible)
                            no_pose_activities['eating_drinking'] = True
                            no_pose_debug['head_pose']['sub_type'] = 'eating_drinking'
                            no_pose_debug['head_pose']['detected'] = True
                            no_pose_debug['head_pose']['method'] = 'no_pose_cup_overlap'
                            self.logger.info(
                                f"[NO-POSE EATING/DRINKING] Person {person_idx}: cup/bottle overlaps bbox, "
                                f"no pose available - flagging eating/drinking"
                            )

                    # --- Low-confidence pose fallback for sleep detection ---
                    # When normal confidence misses a sleeping person, try low-confidence poses
                    if matched_sleep_poses and person_idx in matched_sleep_poses:
                        sleep_fallback_landmarks = matched_sleep_poses[person_idx]
                        if sleep_fallback_landmarks is not None and len(sleep_fallback_landmarks.landmark) > 0:
                            visible_kps = sum(1 for lm in sleep_fallback_landmarks.landmark if lm.visibility > 0.3)
                            if visible_kps >= 5:
                                sleep_det, microsleep_det, sleep_info = self.sleep_detector.detect_pose_based_sleep(
                                    sleep_fallback_landmarks, timestamp_sec, person_idx=person_idx,
                                    frame_shape=frame.shape
                                )
                                if sleep_det:
                                    no_pose_activities['sleep'] = True
                                    no_pose_debug['sleep_info'] = sleep_info
                                    no_pose_debug['sleep_info']['method'] = 'low_conf_pose_fallback'
                                if microsleep_det:
                                    no_pose_activities['microsleep'] = True

                    # --- IR forward-lean sleep detection (body-only keypoints in dark frames) ---
                    # CR-NEW-003: Safe settings access with default
                    ir_fl_enabled = getattr(self.settings, 'ir_forward_lean_enabled', True) if self.settings else True
                    if is_dark_frame and ir_fl_enabled and not no_pose_activities.get('sleep', False):
                        # Try to use low-confidence sleep landmarks for body-only analysis
                        ir_landmarks = None
                        if matched_sleep_poses and person_idx in matched_sleep_poses:
                            ir_landmarks = matched_sleep_poses[person_idx]
                        elif person_idx in matched_poses:
                            ir_landmarks = matched_poses[person_idx]

                        if ir_landmarks is not None and hasattr(ir_landmarks, 'landmark') and len(ir_landmarks.landmark) > 0:
                            ir_sleep, ir_microsleep, ir_info = self.sleep_detector.detect_ir_forward_lean_sleep(
                                ir_landmarks, bbox, timestamp_sec, person_idx, frame.shape
                            )
                            if ir_sleep:
                                no_pose_activities['sleep'] = True
                                no_pose_debug['sleep_info'] = ir_info
                                no_pose_debug['sleep_info']['method'] = 'ir_forward_lean'
                            elif ir_microsleep:
                                no_pose_activities['microsleep'] = True
                                no_pose_debug['sleep_info'] = ir_info
                                no_pose_debug['sleep_info']['method'] = 'ir_forward_lean'

                    # --- No-pose sleep detection (bbox stability tracking) ---
                    sleep_no_pose_enabled = getattr(self.settings, 'sleep_no_pose_enabled', True)
                    if sleep_no_pose_enabled:
                        # Use shorter duration for IR/dark frames (15s vs 30s)
                        if is_dark_frame:
                            min_duration = getattr(self.settings, 'ir_sleep_no_pose_min_duration', 15.0)
                        else:
                            min_duration = getattr(self.settings, 'sleep_no_pose_min_duration', 30.0)
                        stability_threshold = getattr(self.settings, 'sleep_no_pose_bbox_stability_threshold', 0.15)

                        if person_idx not in self.no_pose_sleep_tracking:
                            self.no_pose_sleep_tracking[person_idx] = {
                                'first_seen': timestamp_sec,
                                'last_bbox': list(bbox),
                                'stable_since': timestamp_sec
                            }
                        else:
                            tracker = self.no_pose_sleep_tracking[person_idx]
                            # Calculate IoU between current and last bbox to measure stability
                            # CR-NEW-001: Use consolidated calculate_iou method
                            last_bbox = tracker['last_bbox']
                            iou = calculate_iou(bbox, last_bbox)
                            bbox_change = 1.0 - iou  # How much the bbox moved

                            if bbox_change > stability_threshold:
                                # Person moved significantly, reset stability timer
                                tracker['stable_since'] = timestamp_sec

                            tracker['last_bbox'] = list(bbox)

                            # Check if person has been stable (not moving) for min_duration
                            stable_duration = timestamp_sec - tracker['stable_since']
                            if stable_duration >= min_duration:
                                no_pose_activities['sleep'] = True
                                no_pose_debug['sleep_info']['no_pose_sleep'] = True
                                no_pose_debug['sleep_info']['stable_duration'] = stable_duration
                                self.logger.info(
                                    f"[NO-POSE SLEEP] Person {person_idx}: no pose for {stable_duration:.1f}s "
                                    f"with stable bbox (change={bbox_change:.3f}) - flagging sleep"
                                )

                    persons_data[person_idx] = {
                        'pose_landmarks': None,
                        'role': person_data.get('role', 'UNKNOWN'),
                        'role_name': person_data.get('role_name', 'Unknown'),
                        'bbox': bbox,
                        'activities': no_pose_activities,
                        'debug_info': no_pose_debug
                    }
                    continue

                # Initialize activity detection results for this person
                person_activities = {
                    'mind_diversion': False,
                    'sleep': False,
                    'microsleep': False,
                    'cell_phone': False,
                    'writing': False,
                    'packing_bags': False,
                    'lp_hand_gesture': False,
                    'alp_hand_gesture': False,
                    'eating_drinking': False
                }

                person_debug_info = {
                    'head_pose': {},
                    'sleep_info': {},
                    'gesture_debug': {}
                }

                # ============ BATCH VOTING VERIFICATION COLLECTOR ============
                # Collect all activities that need verification for this person
                # Then verify in batch at the end (single inference pass)
                voting_collector = None
                if self.voting_service is not None and ActivityBatchCollector is not None:
                    voting_collector = ActivityBatchCollector()
                
                # ============ PER-PERSON OBJECT DETECTION (ROI ONLY) ============
                # CR-006: Run ONLY pose-guided ROI detection for this person's keypoints.
                # Full-frame YOLO inference (Stage 1) is already done via the 'detections'
                # parameter passed into this method -- no need to re-run it per person.
                person_detections = self.object_detector.detect_objects_person_rois(frame, translated_landmarks)
                
                # FIX C-01: Create per-person scoped detection lists instead of mutating
                # the shared 'detections' dict. Using .extend() on the shared dict causes
                # cross-person contamination: person 0's ROI detections leak into person 1's
                # checks because the list grows cumulatively across loop iterations.
                person_cell_phones = detections['cell_phone'] + person_detections['cell_phone']
                person_books = detections['book'] + person_detections['book']
                
                # DEBUG: Log per-person ROI detection results
                if person_detections['cell_phone']:
                    self.logger.info(f"[MULTI-PERSON ROI] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): Found {len(person_detections['cell_phone'])} cell phone(s)")
                
                # ============ ACTIVITY DETECTION FOR THIS PERSON ============
                
                # 1. MIND DIVERSION DETECTION
                head_pose_info = self.calculate_head_pose_angles(
                    translated_landmarks,
                    face_results,
                    frame.shape
                )
                person_debug_info['head_pose'] = head_pose_info
                mind_diversion_detected = head_pose_info.get('detected', False)

                # Stage 2: Collect for batch voting verification (if enabled)
                if mind_diversion_detected and voting_collector is not None:
                    voting_collector.add('mind_diversion', person_idx, list(bbox))
                    # Will be verified in batch at end of person processing
                else:
                    person_activities['mind_diversion'] = mind_diversion_detected

                # NOTE: Mind diversion book suppression moved to AFTER writing detection
                # This allows us to check both book presence AND writing activity
                
                # 2. SLEEP / MICROSLEEP DETECTION (pose-based)
                # Run Haar eye detection once (shared between score boost and fallback)
                haar_eye_result = None
                # CR-NEW-003: Safe settings access with default
                haar_enabled = getattr(self.settings, 'haar_eye_detection_enabled', True) if self.settings else True
                if (haar_enabled and self.eye_cascade is not None):
                    haar_eye_result = self.sleep_detector.detect_eye_closure_haar(
                        frame, translated_landmarks, person_idx, bbox, timestamp_sec
                    )
                    person_debug_info['haar_eye_info'] = haar_eye_result

                pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.sleep_detector.detect_pose_based_sleep(
                    translated_landmarks, timestamp_sec, person_idx=person_idx,
                    frame_shape=frame.shape, haar_result=haar_eye_result
                )
                person_debug_info['sleep_info'] = pose_sleep_info
                person_activities['sleep'] = pose_sleep_detected
                person_activities['microsleep'] = pose_microsleep_detected

                # 2b. IR FORWARD LEAN FALLBACK (when dark frame and normal sleep detection missed)
                # CR-NEW-003: Safe settings access with default
                ir_fl_enabled = getattr(self.settings, 'ir_forward_lean_enabled', True) if self.settings else True
                if (is_dark_frame and ir_fl_enabled and
                        not pose_sleep_detected and not pose_microsleep_detected):
                    ir_sleep, ir_microsleep, ir_info = self.sleep_detector.detect_ir_forward_lean_sleep(
                        translated_landmarks, bbox, timestamp_sec, person_idx, frame.shape
                    )
                    if ir_sleep:
                        person_activities['sleep'] = True
                        person_debug_info['sleep_info'] = ir_info
                        person_debug_info['sleep_info']['method'] = 'ir_forward_lean_pose_fallback'
                    elif ir_microsleep:
                        person_activities['microsleep'] = True
                        person_debug_info['sleep_info'] = ir_info
                        person_debug_info['sleep_info']['method'] = 'ir_forward_lean_pose_fallback'

                # 2c. HAAR EYE CLOSURE FALLBACK (reuse result from step 2 — no second call)
                if (haar_eye_result is not None and
                        not person_activities.get('sleep') and not person_activities.get('microsleep')):
                    if haar_eye_result.get('is_sleep'):
                        person_activities['sleep'] = True
                        person_debug_info['sleep_info'] = {'method': 'haar_eye_closure', **haar_eye_result}
                    elif haar_eye_result.get('is_microsleep'):
                        person_activities['microsleep'] = True
                        person_debug_info['sleep_info'] = {'method': 'haar_eye_closure', **haar_eye_result}

                # 3. CELL PHONE DETECTION (check if hand near phone in THIS person's region)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                if len(person_cell_phones) > 0:
                    # DEBUG: Log when cell phones are detected
                    if self.consecutive_detections.get('cell_phone', 0) == 0:
                        self.logger.info(f"[DEBUG CELL PHONE] {len(person_cell_phones)} phone(s) detected in frame")
                    right_hand = self.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = self.get_keypoint(translated_landmarks, 'left_wrist')

                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    
                    # STRICTER MARGIN: Reduced from default to ensure phone is really near hand
                    margin = 100  # Reduced from activity_thresholds margin to be more strict
                    
                    for phone_bbox in person_cell_phones:
                        # Check if phone bbox overlaps with person bbox (with margin)
                        phone_in_person_region = bbox_overlap_with_margin(phone_bbox, bbox, margin)
                        
                        if phone_in_person_region:
                            # Check if hand is near the phone (stricter check)
                            right_hand_near = self.check_hand_object_interaction(right_hand_coords, phone_bbox, margin)
                            left_hand_near = self.check_hand_object_interaction(left_hand_coords, phone_bbox, margin)

                            if right_hand_near or left_hand_near:
                                # Use per-person temporal filtering (2-3 frame requirement)
                                should_trigger = self.update_per_person_detection(
                                    person_idx, 'cell_phone', True, timestamp_sec
                                )

                                # Stage 2: Collect for batch voting verification (if enabled)
                                if should_trigger and voting_collector is not None:
                                    voting_collector.add('cell_phone', person_idx, list(bbox))
                                    # Will be verified in batch at end of person processing
                                else:
                                    person_activities['cell_phone'] = should_trigger
                                break

                # 3b. EATING/DRINKING DETECTION (cup/bottle near face = mind diversion)
                eating_drinking_detected = False
                if getattr(self.settings, 'eating_drinking_detection_enabled', True) and not person_activities.get('mind_diversion', False):
                    # Check if cup/bottle detected in ROI near this person
                    cup_bottle_bboxes = []
                    cup_conf_threshold = getattr(self.settings, 'eating_drinking_cup_confidence', 0.25)
                    for roi_det in detections.get('roi_detections', []):
                        if roi_det['class'] in ('cup', 'bottle') and roi_det['confidence'] > cup_conf_threshold:
                            det_bbox = roi_det['bbox']
                            if bbox_overlap_with_margin(det_bbox, bbox, 100):
                                cup_bottle_bboxes.append(det_bbox)

                    # Also check full-frame cup/bottle detections
                    for cb_xyxy in detections.get('cup_bottle', []):
                        cb_bbox = [float(cb_xyxy[0]), float(cb_xyxy[1]), float(cb_xyxy[2]), float(cb_xyxy[3])]
                        if bbox_overlap_with_margin(cb_bbox, bbox, 100):
                            cup_bottle_bboxes.append(cb_bbox)

                    if cup_bottle_bboxes:
                        # Check if hand is near face level AND holding cup/bottle
                        right_wrist = self.get_keypoint(translated_landmarks, 'right_wrist')
                        left_wrist = self.get_keypoint(translated_landmarks, 'left_wrist')
                        nose = self.get_keypoint(translated_landmarks, 'nose')

                        hand_face_margin = getattr(self.settings, 'eating_drinking_hand_face_margin', 80)
                        hand_obj_margin = getattr(self.settings, 'eating_drinking_hand_object_margin', 150)

                        if nose and nose.visibility > 0.3:
                            nose_y = nose.y * h
                            for wrist in [right_wrist, left_wrist]:
                                if wrist and wrist.visibility > 0.3:
                                    wrist_y = wrist.y * h
                                    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                                    # Hand is at or above shoulder/chin level (drinking position)
                                    if wrist_y < nose_y + hand_face_margin:
                                        for cb_bbox in cup_bottle_bboxes:
                                            if self.check_hand_object_interaction(wrist_coords, cb_bbox, hand_obj_margin):
                                                eating_drinking_detected = True
                                                break
                                if eating_drinking_detected:
                                    break

                        # Fallback: cup directly overlaps person bbox AND at least one wrist visible
                        # Handles overhead camera angles where hand-face proximity is unreliable
                        if not eating_drinking_detected:
                            any_wrist_visible = any(
                                w_kp and w_kp.visibility > 0.3
                                for w_kp in [right_wrist, left_wrist]
                            )
                            if any_wrist_visible:
                                for cb_bbox in cup_bottle_bboxes:
                                    # Cup must directly overlap person bbox (no margin = stricter spatial check)
                                    if bbox_overlap_with_margin(cb_bbox, bbox, 0):
                                        eating_drinking_detected = True
                                        self.logger.info(
                                            f"[EATING/DRINKING FALLBACK] Cup/bottle directly overlaps person {person_idx} bbox "
                                            f"with wrist visible - flagging eating/drinking"
                                        )
                                        break

                    if eating_drinking_detected:
                        person_activities['eating_drinking'] = True
                        person_debug_info['head_pose']['sub_type'] = 'eating_drinking'
                        person_debug_info['head_pose']['detected'] = True
                        person_debug_info['head_pose']['method'] = 'object_proximity'
                        # Skip voting for eating/drinking — voting re-checks head pose angles
                        # which doesn't apply to cup-based detection, causing 0/10 false rejections

                # 4. WRITING DETECTION (check if hand near book OR wrist/elbow proximity heuristic)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                writing_detected_by_book = False
                writing_detected_by_wrist = False
                writing_detected_by_book_posture = False  # NEW fallback method

                # Method 1: Book detection (existing method - requires wrists visible)
                if len(person_books) > 0:
                    right_hand = self.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = self.get_keypoint(translated_landmarks, 'left_wrist')

                    # Check if wrists are visible enough for hand-based detection
                    wrists_visible = (right_hand.visibility >= 0.5 or left_hand.visibility >= 0.5)

                    if wrists_visible:
                        right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                        left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))

                        hand_margin = self.activity_thresholds['writing']['margin']
                        # Use larger margin for book-to-person association since book is typically in lap area
                        # (below the person's detected bounding box which mainly covers upper body)
                        person_book_margin = 250  # INCREASED from 150 to 250 - books in lap area extend beyond person bbox
                        for book_bbox in person_books:
                            # Check if book is in this person's region (use large margin for lap area)
                            book_in_person_region = bbox_overlap_with_margin(book_bbox, bbox, person_book_margin)

                            if book_in_person_region:
                                # Check visible hands for interaction with book (use tighter margin)
                                right_hand_near_book = right_hand.visibility >= 0.5 and self.check_hand_object_interaction(right_hand_coords, book_bbox, hand_margin)
                                left_hand_near_book = left_hand.visibility >= 0.5 and self.check_hand_object_interaction(left_hand_coords, book_bbox, hand_margin)
                                if right_hand_near_book or left_hand_near_book:
                                    writing_detected_by_book = True
                                    break
                    else:
                        # FALLBACK: Wrists not visible, use book + posture detection
                        self.logger.debug(f"Person {person_idx}: Wrists not visible, trying book+posture fallback")
                        writing_detected_by_book_posture = self.detect_writing_by_book_and_posture(
                            translated_landmarks,
                            bbox,
                            person_books,
                            person_idx,
                            timestamp_sec
                        )

                # Method 2: Wrist/Elbow proximity heuristic (temporal - requires sustained duration)
                # DEBUG: Log that we're checking writing detection
                self.logger.debug(f"Person {person_idx}: Calling writing detection (frame {frame_number})")
                writing_detected_by_wrist = self.detect_writing_by_wrist_proximity(
                    translated_landmarks,
                    frame.shape,
                    person_idx,
                    timestamp_sec
                )
                self.logger.debug(f"Person {person_idx}: Writing detection result = {writing_detected_by_wrist}")

                # Combine all detection methods (book+hand, wrist/elbow proximity, book+posture fallback)
                writing_detected_raw = (
                    writing_detected_by_book or
                    writing_detected_by_wrist or
                    writing_detected_by_book_posture  # NEW fallback
                )
                should_trigger = self.update_per_person_detection(
                    person_idx, 'writing', writing_detected_raw, timestamp_sec
                )

                # Stage 2: Collect for batch voting verification (if enabled)
                if should_trigger and voting_collector is not None:
                    voting_collector.add('writing', person_idx, list(bbox))
                    # Will be verified in batch at end of person processing
                else:
                    person_activities['writing'] = should_trigger

                # Store detection method in debug info for analysis
                if writing_detected_by_book:
                    person_debug_info['writing_method'] = 'book_hand'
                elif writing_detected_by_book_posture:
                    person_debug_info['writing_method'] = 'book_posture_fallback'
                elif writing_detected_by_wrist:
                    person_debug_info['writing_method'] = 'pose_based'  # Could be wrist or elbow
                else:
                    person_debug_info['writing_method'] = 'none'
                
                # SUPPRESS MIND DIVERSION IF LEGITIMATE WORK ACTIVITY DETECTED
                # Uses comprehensive suppression logic that checks:
                # 1. Writing activity detected
                # 2. Recent writing activity (within grace period)
                # 3. Book/document present in frame
                # 4. Hands in writing position (wrists close together, below face)
                if person_activities['mind_diversion']:
                    # Don't suppress eating/drinking detections — they are object-based, not head-pose
                    sub_type = person_debug_info.get('head_pose', {}).get('sub_type')
                    if sub_type != 'eating_drinking':
                        # FIX C-01: Pass per-person scoped detections to avoid
                        # cross-person book contamination in suppression logic
                        person_scoped_detections = {**detections, 'book': person_books, 'cell_phone': person_cell_phones}
                        should_suppress, suppress_reason = self.should_suppress_mind_diversion(
                            person_idx=person_idx,
                            person_activities=person_activities,
                            pose_landmarks=translated_landmarks,
                            detections=person_scoped_detections,
                            frame_shape=frame.shape,
                            current_time=timestamp_sec
                        )

                        if should_suppress:
                            person_activities['mind_diversion'] = False
                            person_debug_info['head_pose']['suppressed'] = True
                            person_debug_info['head_pose']['suppressed_reason'] = suppress_reason
                        else:
                            person_debug_info['head_pose']['suppressed'] = False
                            person_debug_info['head_pose']['suppressed_reason'] = None

                # 5. PACKING DETECTION (check if hand near backpack in THIS person's region)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                if len(detections['backpack']) > 0:
                    right_hand = self.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = self.get_keypoint(translated_landmarks, 'left_wrist')

                    # Check wrist visibility - only use if visible enough
                    right_wrist_visible = right_hand.visibility > 0.3
                    left_wrist_visible = left_hand.visibility > 0.3

                    # Use smoothed hand positions to reduce pose estimation noise (only if visible)
                    right_hand_coords = None
                    left_hand_coords = None

                    if right_wrist_visible:
                        right_hand_coords = self._get_smoothed_hand_position(
                            person_idx, 'right', right_hand, w, h, timestamp_sec
                        )
                    elif left_wrist_visible:
                        # Fallback: if right wrist not visible, try using right elbow as approximation
                        right_elbow = self.get_keypoint(translated_landmarks, 'right_elbow')
                        if right_elbow.visibility > 0.3:
                            right_hand_coords = (int(right_elbow.x * w), int(right_elbow.y * h))

                    if left_wrist_visible:
                        left_hand_coords = self._get_smoothed_hand_position(
                            person_idx, 'left', left_hand, w, h, timestamp_sec
                        )
                    elif right_wrist_visible:
                        # Fallback: if left wrist not visible, try using left elbow as approximation
                        left_elbow = self.get_keypoint(translated_landmarks, 'left_elbow')
                        if left_elbow.visibility > 0.3:
                            left_hand_coords = (int(left_elbow.x * w), int(left_elbow.y * h))

                    # Separate margins: region overlap vs. hand proximity
                    region_margin = self.activity_thresholds['packing_bags'].get('region_margin', 100)
                    proximity_margin = self.activity_thresholds['packing_bags']['margin']

                    # ============ SIMPLIFIED PACKING DETECTION ============
                    # Core logic: If wrist is inside/near backpack bbox -> Packing detected!
                    # M-01 FIX: Track the best match across ALL backpacks instead of
                    # stopping at the first match. Priority: wrist-inside > motion-confirmed.
                    # Within the same priority, prefer the closest backpack (smallest distance).
                    packing_motion_analysis = None
                    packing_detected_simple = False
                    best_pack_type = None        # 'wrist_inside' | 'motion' | None
                    best_pack_distance = float('inf')
                    best_pack_bbox = None
                    best_pack_motion = None
                    best_pack_debug = None

                    for backpack_bbox in detections['backpack']:
                        # Check if backpack is in this person's region (wider margin)
                        backpack_in_person_region = bbox_overlap_with_margin(
                            backpack_bbox, bbox, region_margin
                        )

                        if not backpack_in_person_region:
                            continue

                        # ===== SIMPLIFIED CHECK: Is wrist INSIDE backpack bbox? =====
                        right_inside, right_dist = self.activity_detector.is_wrist_inside_backpack(
                            right_hand_coords, backpack_bbox, margin=40
                        )
                        left_inside, left_dist = self.activity_detector.is_wrist_inside_backpack(
                            left_hand_coords, backpack_bbox, margin=40
                        )

                        wrist_inside_backpack = right_inside or left_inside
                        closest_distance = min(right_dist, left_dist)

                        cur_debug = {
                            'right_wrist_inside': right_inside,
                            'left_wrist_inside': left_inside,
                            'right_dist': right_dist,
                            'left_dist': left_dist,
                            'closest_distance': closest_distance,
                            'backpack_bbox': list(backpack_bbox[:4])
                        }

                        # ===== PRIMARY: Wrist inside backpack bbox =====
                        if wrist_inside_backpack:
                            if best_pack_type != 'wrist_inside' or closest_distance < best_pack_distance:
                                best_pack_type = 'wrist_inside'
                                best_pack_distance = closest_distance
                                best_pack_bbox = backpack_bbox
                                best_pack_debug = cur_debug
                            continue  # Check remaining backpacks for a closer match

                        # ===== FALLBACK: Hand near backpack with motion analysis =====
                        # Only consider if we have not found a wrist-inside match yet
                        if best_pack_type == 'wrist_inside':
                            continue

                        hand_near_backpack = (
                            self.check_hand_object_interaction(right_hand_coords, backpack_bbox, proximity_margin) or
                            self.check_hand_object_interaction(left_hand_coords, backpack_bbox, proximity_margin)
                        )

                        if hand_near_backpack:
                            cur_motion = self.analyze_packing_hand_motion(
                                person_idx, translated_landmarks, frame.shape, timestamp_sec, backpack_bbox
                            )
                            motion_confirmed = cur_motion['packing_motion_detected']
                            sustained_proximity = cur_motion.get('sustained_proximity', False) and \
                                                 cur_motion.get('sustained_proximity_time', False)

                            if motion_confirmed or sustained_proximity:
                                if best_pack_type != 'motion' or closest_distance < best_pack_distance:
                                    best_pack_type = 'motion'
                                    best_pack_distance = closest_distance
                                    best_pack_bbox = backpack_bbox
                                    best_pack_motion = cur_motion
                                    best_pack_debug = cur_debug

                    # ===== APPLY BEST MATCH RESULT AFTER LOOP =====
                    if best_pack_debug is not None:
                        person_debug_info['packing_wrist_check'] = best_pack_debug

                    if best_pack_type == 'wrist_inside':
                        packing_detected_simple = True
                        self.logger.info(
                            f"PACKING DETECTED (SIMPLE): Wrist inside backpack bbox! "
                            f"Distance: {best_pack_distance:.0f}px, "
                            f"Backpack: {list(best_pack_bbox[:4])}"
                        )
                        should_trigger = self.update_per_person_detection(
                            person_idx, 'packing_bags', True, timestamp_sec
                        )
                        if should_trigger and voting_collector is not None:
                            voting_collector.add('packing_bags', person_idx, list(bbox))
                            person_activities['packing_bags'] = True
                        else:
                            person_activities['packing_bags'] = should_trigger
                        if person_idx not in self.recent_person_activities:
                            self.recent_person_activities[person_idx] = {}
                        self.recent_person_activities[person_idx]['packing_bags'] = timestamp_sec

                    elif best_pack_type == 'motion':
                        packing_motion_analysis = best_pack_motion
                        person_debug_info['packing_motion'] = packing_motion_analysis
                        should_trigger = self.update_per_person_detection(
                            person_idx, 'packing_bags', True, timestamp_sec
                        )
                        if should_trigger and voting_collector is not None:
                            voting_collector.add('packing_bags', person_idx, list(bbox))
                            person_activities['packing_bags'] = True
                        else:
                            person_activities['packing_bags'] = should_trigger
                        if person_idx not in self.recent_person_activities:
                            self.recent_person_activities[person_idx] = {}
                        self.recent_person_activities[person_idx]['packing_bags'] = timestamp_sec

                    else:
                        # No match found across all backpacks - reset counter
                        should_trigger = self.update_per_person_detection(
                            person_idx, 'packing_bags', False, timestamp_sec
                        )
                        person_activities['packing_bags'] = should_trigger
                
                # UPDATE TEMPORAL HISTORY for writing and cell phone too
                if person_activities.get('writing', False):
                    if person_idx not in self.recent_person_activities:
                        self.recent_person_activities[person_idx] = {}
                    self.recent_person_activities[person_idx]['writing'] = timestamp_sec
                
                if person_activities.get('cell_phone', False):
                    if person_idx not in self.recent_person_activities:
                        self.recent_person_activities[person_idx] = {}
                    self.recent_person_activities[person_idx]['cell_phone'] = timestamp_sec
                
                # 6. HAND GESTURE DETECTION (LP/ALP)
                # CRITICAL: This runs AFTER packing/writing/phone detection for context-aware filtering
                # Pass person_activities, backpack detections, person_idx, and timestamp for full suppression
                single_person_roles = {person_idx: person_data}
                lp_gesture, alp_gesture, gesture_debug = self.detect_hand_gesture(
                    translated_landmarks,
                    frame.shape,
                    single_person_roles,
                    yolo_person_boxes=None,
                    person_activities=person_activities,
                    backpack_detections=detections.get('backpack', []),
                    person_idx=person_idx,
                    current_timestamp=timestamp_sec,
                    frame_number=frame_number
                )
                person_debug_info['gesture_debug'] = gesture_debug

                # Stage 2: Collect for batch voting verification (if enabled)
                if lp_gesture and voting_collector is not None:
                    voting_collector.add('lp_hand_gesture', person_idx, list(bbox))
                    # Will be verified in batch at end of person processing
                else:
                    person_activities['lp_hand_gesture'] = lp_gesture

                if alp_gesture and voting_collector is not None:
                    voting_collector.add('alp_hand_gesture', person_idx, list(bbox))
                    # Will be verified in batch at end of person processing
                else:
                    person_activities['alp_hand_gesture'] = alp_gesture

                # ============ BATCH VOTING VERIFICATION WITH MOTION CHECK ============
                # Verify all collected activities in a single batch (shared inference)
                # Uses optical flow motion detection to exempt violations when train is stopped
                if voting_collector is not None and voting_collector.has_activities():
                    try:
                        # Use motion-aware verification if available
                        if hasattr(self.voting_service, 'verify_batch_with_motion_check'):
                            batch_results = self.voting_service.verify_batch_with_motion_check(
                                video_path=self.current_video_path,
                                timestamp_sec=timestamp_sec,
                                activities=voting_collector.get_activities(),
                                trip_schedule=self.trip_schedule,
                                motion_context=self.current_motion_context
                            )
                        else:
                            batch_results = self.voting_service.verify_batch(
                                video_path=self.current_video_path,
                                timestamp_sec=timestamp_sec,
                                activities=voting_collector.get_activities()
                            )

                        # Apply results to person_activities
                        for activity_key, (is_confirmed, vote_details) in batch_results.items():
                            # Parse key: 'cell_phone_p0' -> ('cell_phone', 0)
                            parts = activity_key.rsplit('_p', 1)
                            if len(parts) == 2:
                                activity_type = parts[0]

                                # Map activity type to person_activities key
                                activity_key_map = {
                                    'mind_diversion': 'mind_diversion',
                                    'cell_phone': 'cell_phone',
                                    'writing': 'writing',
                                    'packing_bags': 'packing_bags',
                                    'lp_hand_gesture': 'lp_hand_gesture',
                                    'alp_hand_gesture': 'alp_hand_gesture',
                                    'eating_drinking': 'eating_drinking'
                                }

                                person_key = activity_key_map.get(activity_type, activity_type)
                                person_activities[person_key] = is_confirmed

                                # Log result with motion exemption info
                                motion_exempted = vote_details.get('motion_exempted', False)
                                if is_confirmed:
                                    self.logger.info(f"[VOTING BATCH] {activity_type} CONFIRMED: {vote_details.get('vote_breakdown', [])}")
                                elif motion_exempted:
                                    self.logger.info(f"[VOTING MOTION] {activity_type} EXEMPTED: Train stopped (optical flow)")
                                else:
                                    self.logger.info(f"[VOTING BATCH] {activity_type} REJECTED: {vote_details.get('vote_breakdown', [])}")
                    except Exception as e:
                        self.logger.error(f"[VOTING BATCH] Error in batch verification: {e}", exc_info=True)
                        # On error, set all collected activities to False (safe default)
                        for activity in voting_collector.get_activities():
                            activity_type = activity['type']
                            activity_key_map = {
                                'mind_diversion': 'mind_diversion',
                                'cell_phone': 'cell_phone',
                                'writing': 'writing',
                                'packing_bags': 'packing_bags',
                                'lp_hand_gesture': 'lp_hand_gesture',
                                'alp_hand_gesture': 'alp_hand_gesture',
                                'eating_drinking': 'eating_drinking'
                            }
                            person_key = activity_key_map.get(activity_type, activity_type)
                            person_activities[person_key] = False

                # Track hand raise timestamps for temporal coordination window
                if person_activities['lp_hand_gesture']:
                    if person_idx not in self.recent_person_activities:
                        self.recent_person_activities[person_idx] = {}
                    self.recent_person_activities[person_idx]['lp_hand_raise'] = timestamp_sec

                if person_activities['alp_hand_gesture']:
                    if person_idx not in self.recent_person_activities:
                        self.recent_person_activities[person_idx] = {}
                    self.recent_person_activities[person_idx]['alp_hand_raise'] = timestamp_sec

                # Store this person's data
                persons_data[person_idx] = {
                    'pose_landmarks': translated_landmarks,
                    'role': person_data.get('role', 'UNKNOWN'),
                    'role_name': person_data.get('role_name', 'Unknown'),
                    'bbox': bbox,
                    'activities': person_activities,
                    'debug_info': person_debug_info
                }
                
            except Exception as e:
                self.logger.error(f"Error processing person {person_idx}: {e}", exc_info=True)
                continue
        
        # ============ CLEAN UP STALE PER-PERSON TRACKING (CR-012) ============
        # Remove tracking for persons no longer detected to prevent stale state
        active_person_indices = set(persons_data.keys())
        self._cleanup_stale_person_tracking(active_person_indices)

        # Also clear no-pose tracking for persons that now have pose (they moved to the pose path)
        for person_idx in list(self.no_pose_sleep_tracking.keys()):
            if person_idx in persons_data and persons_data[person_idx].get('pose_landmarks') is not None:
                del self.no_pose_sleep_tracking[person_idx]

        # ============ AGGREGATE RESULTS ACROSS ALL PERSONS ============
        aggregated = {
            'mind_diversion_detected': False,
            'sleep_detected': False,
            'microsleep_detected': False,
            'cell_phone_detected': False,
            'writing_detected': False,
            'packing_detected': False,
            'lp_hand_gesture_detected': False,
            'alp_hand_gesture_detected': False,
            'eating_drinking_detected': False,
            'performing_person': -1,
            'performing_persons': []  # List of person indices who performed activities
        }
        
        # Aggregate: if ANY person has an activity, mark it as detected
        # Per-person state machine gate (H-02 fix): only aggregate sleep/microsleep
        # if THAT SPECIFIC person's state machine is in DROWSY or beyond.
        # This prevents person 0's SLEEPING state from letting person 1's
        # microsleep bypass the gate.
        for person_idx, person_data in persons_data.items():
            activities = person_data['activities']

            if activities['mind_diversion']:
                aggregated['mind_diversion_detected'] = True
                aggregated['performing_persons'].append(person_idx)

            # Per-person state machine gate for sleep/microsleep
            person_sleep_info = person_data.get('debug_info', {}).get('sleep_info', {})
            person_sleep_state = person_sleep_info.get('sleep_state', 'ALERT')
            person_state_machine_ready = person_sleep_state in ('DROWSY', 'MICROSLEEP', 'SLEEPING')

            if activities['sleep']:
                if person_state_machine_ready:
                    aggregated['sleep_detected'] = True
                else:
                    # Suppress this person's sleep - state machine not ready
                    activities['sleep'] = False
            if activities['microsleep']:
                if person_state_machine_ready:
                    aggregated['microsleep_detected'] = True
                else:
                    # Suppress this person's microsleep - state machine not ready
                    activities['microsleep'] = False
            if activities['cell_phone']:
                aggregated['cell_phone_detected'] = True
            if activities['writing']:
                aggregated['writing_detected'] = True
            if activities['packing_bags']:
                aggregated['packing_detected'] = True
            if activities['lp_hand_gesture']:
                aggregated['lp_hand_gesture_detected'] = True
            if activities['alp_hand_gesture']:
                aggregated['alp_hand_gesture_detected'] = True
            if activities.get('eating_drinking'):
                aggregated['eating_drinking_detected'] = True

        # Set performing_person to the first detected person (for backward compatibility)
        if aggregated['performing_persons']:
            aggregated['performing_person'] = aggregated['performing_persons'][0]
        
        return {
            'persons': persons_data,
            'aggregated': aggregated
        }
    
    # CR-NEW-001: _calculate_bbox_iou removed - use calculate_iou() instead (consolidated IoU methods)
    # CR-NEW-002: bbox_overlap_with_margin removed - use bbox_overlap_with_margin() from app.core.utils.geometry

    def calculate_head_pose_angles(self, pose_landmarks: Any, face_landmarks: Any, frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Calculate head pose angles (yaw and pitch) to detect mind diversion - delegates to MindDiversionDetector."""
        return self.mind_diversion_detector.calculate_head_pose_angles(
            pose_landmarks, face_landmarks, frame_shape
        )

    def should_suppress_mind_diversion(self, person_idx: int, person_activities: Dict[str, Any], pose_landmarks: Any, detections: Dict[str, List[Any]], frame_shape: Tuple[int, ...], current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Suppress mind diversion if person is doing legitimate work activity.

        This function checks multiple conditions to prevent false positives when the LP
        is legitimately working on documents (logbook, papers, etc.).

        Args:
            person_idx: Index of the person being checked
            person_activities: Dict of detected activities for this person
            pose_landmarks: Pose landmarks for the person
            detections: YOLO detections dict (may contain 'book', etc.)
            frame_shape: (height, width) of the frame
            current_time: Current timestamp (optional, for recent activity check)

        Returns:
            tuple: (should_suppress: bool, reason: str or None)
        """
        h, w = frame_shape[:2]
        settings = self.settings

        # 1. WRITING ACTIVITY SUPPRESSION
        if settings.mind_diversion_suppress_with_writing:
            if person_activities.get('writing', False):
                return True, "suppressed_writing_active"

            # Check recent writing (within grace period)
            if current_time is not None and hasattr(self, 'recent_person_activities'):
                writing_timestamp = self.recent_person_activities.get(person_idx, {}).get('writing')
                if writing_timestamp and (current_time - writing_timestamp) < settings.mind_diversion_writing_grace_seconds:
                    return True, "suppressed_recent_writing"

        # 2. BOOK DETECTION SUPPRESSION
        if detections and 'book' in detections and len(detections.get('book', [])) > 0:
            return True, "suppressed_book_detected"

        # 3. HAND POSITION HEURISTIC (Critical for camera angle)
        # If both wrists visible and close together in lap area → likely document work
        if pose_landmarks:
            try:
                left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
                right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')
                nose = self.get_keypoint(pose_landmarks, 'nose')

                if left_wrist.visibility > 0.3 and right_wrist.visibility > 0.3:
                    # Calculate wrist positions
                    left_wrist_coords = np.array([left_wrist.x * w, left_wrist.y * h])
                    right_wrist_coords = np.array([right_wrist.x * w, right_wrist.y * h])
                    wrist_distance = np.linalg.norm(left_wrist_coords - right_wrist_coords)

                    # Check if wrists are in "lap area" (below nose, in front of body)
                    nose_y = nose.y * h
                    avg_wrist_y = (left_wrist_coords[1] + right_wrist_coords[1]) / 2
                    wrists_below_face = avg_wrist_y > nose_y

                    # If wrists close together AND below face → writing pose
                    if wrist_distance < settings.mind_diversion_wrist_distance_threshold and wrists_below_face:
                        return True, "suppressed_writing_pose_detected"
            except (AttributeError, IndexError):
                pass  # Landmarks not available, continue without suppression

        # 4. NO SUPPRESSION - Allow detection
        return False, None

    # CR-NEW-003: calculate_iou, deduplicate_person_boxes, _compute_iou removed
    # Use calculate_iou(), deduplicate_person_boxes() from app.core.utils.geometry

    def identify_person_roles(self, frame: Any, person_boxes: List[List[int]], detections: Dict[str, List[Any]]) -> Dict[int, Dict[str, Any]]:
        """Identify LP (Loco Pilot) and ALP (Assistant Loco Pilot) based on camera angle.

        DELEGATION: This method delegates to PersonTracker.identify_person_roles().
        Kept for backward compatibility during refactoring transition.
        """
        # Sync camera_angle in case it was changed
        self.person_tracker.camera_angle = self.camera_angle
        return self.person_tracker.identify_person_roles(person_boxes, frame, detections)
    
    def start_activity(self, activity_name: str, timestamp: float, fps: float, frame_count: int, person_roles: Optional[Dict[int, Dict[str, Any]]] = None, ocr_timestamp: Optional[str] = None) -> None:
        """Start tracking an activity

        Args:
            activity_name: Name of the activity
            timestamp: Timestamp when activity started (video playback time)
            fps: Frames per second
            frame_count: Frame count when activity started
            person_roles: Dictionary of person roles (optional)
            ocr_timestamp: OCR-extracted timestamp from frame (HH:MM:SS format, optional)
        """
        if not self.activities[activity_name]['active']:
            self.activities[activity_name]['active'] = True
            self.activities[activity_name]['start_time'] = timestamp

            # OCR timestamp (disabled)
            ocr_ts = ocr_timestamp if ocr_timestamp else None
            self.activities[activity_name]['ocr_start_time'] = ocr_ts
            self.activities[activity_name]['start_frame_count'] = frame_count
            self.activities[activity_name]['last_frame_count'] = frame_count
            # CR-005: Store frame indices instead of frame copies to reduce memory usage
            self.activities[activity_name]['frames'] = list(self.frame_idx_buffer)
            self.activities[activity_name]['duration'] = 0
            self.activities[activity_name]['person_roles'] = person_roles if person_roles else {}

            # Track actual detection timestamps for precise clip duration
            self.activities[activity_name]['first_detection_time'] = timestamp
            self.activities[activity_name]['last_detection_time'] = timestamp

            # Log with OCR timestamp if available
            if ocr_ts:
                self.logger.info(f"[{timestamp}] Activity started: {activity_name} (Frame timestamp: {ocr_ts})")
            else:
                self.logger.info(f"[{timestamp}] Activity started: {activity_name}")
    
    def _cleanup_stale_person_tracking(self, active_person_indices):
        """CR-012: Remove entries from per-person tracking dicts for persons no longer detected.

        This prevents unbounded memory growth when person indices change over time.

        Args:
            active_person_indices: Set of person indices currently detected in the frame.
        """
        active_set = set(active_person_indices)

        # Delegate sleep-detector cleanup to its own method (C-11 fix)
        self.sleep_detector.cleanup_stale_tracking(active_set)

        # All per-person tracking dictionaries to clean up
        tracking_dicts = [
            ('per_person_consecutive_detections', self.per_person_consecutive_detections),
            ('per_person_grace_counters', self.per_person_grace_counters),
            ('hand_position_history', self.hand_position_history),
            ('landmark_stability_history', self.landmark_stability_history),
            ('wrist_proximity_tracking', self.wrist_proximity_tracking),
            ('no_pose_sleep_tracking', self.no_pose_sleep_tracking),
            ('recent_person_activities', self.recent_person_activities),
        ]

        total_removed = 0
        for dict_name, tracking_dict in tracking_dicts:
            stale_keys = set(tracking_dict.keys()) - active_set
            for stale_key in stale_keys:
                del tracking_dict[stale_key]
                total_removed += 1

        # M-03: Clean up tuple-keyed dicts where key is (person_idx, hand_side).
        # These cannot be cleaned by simple set subtraction against active_set
        # because the keys are tuples, not plain integers.
        tuple_keyed_dicts = [
            ('hand_smoothing_buffers', self.hand_smoothing_buffers),
        ]

        for dict_name, tracking_dict in tuple_keyed_dicts:
            stale_keys = [
                key for key in tracking_dict
                if key[0] not in active_set
            ]
            for stale_key in stale_keys:
                del tracking_dict[stale_key]
                total_removed += 1

        if total_removed > 0:
            self.logger.debug(f"[CR-012] Cleaned up {total_removed} stale person tracking entries "
                              f"(active persons: {sorted(active_set)})")

    def _get_video_metadata(self):
        """CR-011: Lazily load and cache video metadata (total_frames, fps, duration).

        Avoids reopening VideoCapture every time metadata is needed.
        Returns:
            tuple: (total_frames, fps, duration_seconds)
        """
        if self._video_total_frames is None:
            with video_capture_context(self.video_path) as cap:
                self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._video_duration_seconds = self._video_total_frames / self._video_fps
            self.logger.debug(f"[CR-011] Cached video metadata: {self._video_total_frames} frames, "
                              f"{self._video_fps:.2f} fps, {self._video_duration_seconds:.1f}s duration")
        return self._video_total_frames, self._video_fps, self._video_duration_seconds

    def end_activity(self, activity_name: str, timestamp: float, fps: float, frame_count: int, people_count: int = 1, save_clips: bool = True, ocr_timestamp: Optional[str] = None) -> None:
        """End tracking an activity and optionally save evidence (only if meets minimum duration)"""
        if self.activities[activity_name]['active']:
            activity = self.activities[activity_name]
            activity['active'] = False

            # For end timestamp, we calculate from start + duration instead of OCR
            # This is more reliable because _current_frame may point to a later frame
            # (OCR on end frame often gives wrong timestamps due to frame timing issues)
            ocr_ts = None  # Will be calculated from start + duration below
            if ocr_timestamp:
                # Only use provided timestamp if explicitly passed
                ocr_ts = ocr_timestamp
                self.logger.info(f"[OCR END] Using provided timestamp: {ocr_ts}")
            else:
                self.logger.info(f"[OCR END] Will calculate from start + duration (more reliable)")

            activity['ocr_end_time'] = ocr_ts

            start_frame = activity.get('start_frame_count', frame_count)

            # Calculate duration based on ACTUAL captured frames, not elapsed time
            # This ensures clip duration matches exactly the activity duration
            total_clip_frames = len(activity['frames'])
            actual_clip_duration = total_clip_frames / self.sample_fps  # Duration in seconds based on captured frames

            # Check if activity meets minimum duration threshold
            min_duration = self.activity_thresholds[activity_name]['min_duration']

            if actual_clip_duration < min_duration:
                self.logger.debug(f"[{timestamp}] Activity '{activity_name}' too short ({actual_clip_duration:.2f}s < {min_duration}s) - discarded")
                activity['frames'] = []
                activity['duration'] = 0
                self.consecutive_detections[activity_name] = 0
                self.grace_counters[activity_name] = 0
                return

            start_time_str = activity['start_time']

            # Parse activity start time in seconds (video playback time)
            def time_to_seconds(time_str):
                """Convert HH:MM:SS.microseconds to seconds"""
                parts = time_str.split(':')
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

            # Use ACTUAL detection timestamps for precise clip duration
            first_detection = activity.get('first_detection_time', start_time_str)
            last_detection = activity.get('last_detection_time', start_time_str)

            first_detection_seconds = time_to_seconds(first_detection)
            last_detection_seconds = time_to_seconds(last_detection)

            # Calculate precise activity duration from detection window
            actual_activity_duration = last_detection_seconds - first_detection_seconds

            # Apply configurable buffer (default: 1 sec before/after)
            # Use getattr for safe attribute access in case settings wasn't initialized
            settings = getattr(self, 'settings', None)
            buffer_before = settings.clip_buffer_before if settings else 1.0
            buffer_after = settings.clip_buffer_after if settings else 1.0

            activity_start_seconds = max(0, first_detection_seconds - buffer_before)
            activity_end_seconds = last_detection_seconds + buffer_after

            # Ensure minimum clip duration (at least min_duration)
            if activity_end_seconds - activity_start_seconds < min_duration:
                activity_end_seconds = activity_start_seconds + min_duration

            self.logger.info(f"  Precise clip: {first_detection_seconds:.2f}s - {last_detection_seconds:.2f}s (activity: {actual_activity_duration:.2f}s, buffer: +/-{buffer_before}s)")

            # Calculate OCR timestamps if available
            ocr_start_time_str = activity.get('ocr_start_time')
            ocr_end_time_str = activity.get('ocr_end_time')

            # If we have OCR start but not end, calculate end from duration
            if ocr_start_time_str and not ocr_end_time_str:
                ocr_start_seconds = time_to_seconds(ocr_start_time_str)
                ocr_end_seconds = ocr_start_seconds + actual_clip_duration
                hours = int(ocr_end_seconds // 3600)
                minutes = int((ocr_end_seconds % 3600) // 60)
                seconds = int(ocr_end_seconds % 60)
                ocr_end_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                activity['ocr_end_time'] = ocr_end_time_str
            
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
            
            # Always save clips/images (for UI evidence), regardless of save_clips flag
            # The save_clips flag now only controls whether frames are saved
            if self.evidence_clips_dir:
                # Extract video segment directly from source for smooth playback
                # This preserves original frame rate instead of reconstructing from sampled frames
                self.extract_video_segment(
                    self.video_path,
                    clip_path,
                    activity_start_seconds,
                    activity_end_seconds
                )

                # Save activity image (middle frame of the activity)
                # CR-005: frames now stores frame indices; extract frame on-demand from video
                if len(activity['frames']) > 0:
                    middle_list_idx = len(activity['frames']) // 2
                    middle_frame_number = activity['frames'][middle_list_idx]
                    # M-08: Extract the frame on-demand with proper resource cleanup
                    cap = cv2.VideoCapture(self.video_path)
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
                        ret, activity_image = cap.read()
                        if ret and activity_image is not None:
                            cv2.imwrite(image_path, activity_image)
                    finally:
                        cap.release()
            
            # Get video duration in HH:MM:SS format
            # CR-011: Use cached video metadata instead of reopening VideoCapture
            video_total_frames, _cached_fps, video_duration_seconds = self._get_video_metadata()

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

            # CR-019: Warn if evidence is being created with unset crew/trip data
            if any(v is None for v in (self.trip_id, self.crew_name, self.crew_id, self.crew_role)):
                self.logger.warning(
                    f"Creating evidence for '{activity_name}' with default (None) crew/trip data. "
                    f"trip_id={self.trip_id}, crew_name={self.crew_name}, "
                    f"crew_id={self.crew_id}, crew_role={self.crew_role}. "
                    f"Ensure crew data is set via API input before processing."
                )
            
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
            # Use OCR timestamps (embedded frame timestamps) when available, fallback to video playback time
            ocr_start = activity.get('ocr_start_time')
            ocr_end = activity.get('ocr_end_time')

            # Determine which timestamps to use for startTime/endTime
            # Priority: OCR timestamps (actual frame time) > Video playback timestamps
            if ocr_start and ocr_end:
                # Use OCR timestamps directly (HH:MM:SS format)
                final_start_time = ocr_start
                final_end_time = ocr_end
                self.logger.info(f"  Using OCR timestamps: {final_start_time} - {final_end_time}")
            else:
                # Fallback to video playback timestamps (seconds format)
                final_start_time = f"{activity_start_seconds:.2f}"
                final_end_time = f"{activity_end_seconds:.2f}"
                self.logger.info(f"  Using video timestamps (OCR failed): {final_start_time}s - {final_end_time}s (ocr_start={ocr_start}, ocr_end={ocr_end})")

            json_data = {
                "tripId": self.trip_id,
                "activityType": self.activity_type_map[activity_name],
                "des": self.activity_descriptions[activity_name],
                "objectType": activity_name.replace('_', ' '),
                "fileUrl": os.path.abspath(self.video_path),
                "fileDuration": video_duration_formatted,
                "activityStartTime": final_start_time,
                "activityEndTime": final_end_time,
                # Also store video playback timestamps for reference/clip extraction
                "videoStartTime": f"{activity_start_seconds:.2f}",
                "videoEndTime": f"{activity_end_seconds:.2f}",
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
                        "bboxArea": role_info.get('bbox_area', 0)
                    })
                json_data["personRoles"] = person_roles_list
            
            # Add to all activities list
            self.all_activities.append(json_data)
            
            # Calculate end time string for logging
            end_time_str = str(timedelta(seconds=activity_end_seconds))
            
            self.logger.info(f"[{end_time_str}] Activity ended: {activity_name}")
            self.logger.info(f"  Clip Duration: {actual_clip_duration:.2f}s ({total_clip_frames} frames @ {self.sample_fps} FPS)")
            self.logger.debug(f"  Min Duration Threshold: {min_duration}s | Required Consecutive: {self.activity_thresholds[activity_name]['required_consecutive']} frames")
            self.logger.info(f"  Evidence saved: {clip_filename}")
            self.logger.info(f"  Activity image: {image_filename}")
            
            activity['frames'] = []
            activity['duration'] = 0
            self.consecutive_detections[activity_name] = 0
            self.grace_counters[activity_name] = 0
            
            self.evidence_counter += 1

    def _reencode_to_h264(self, input_path: str) -> bool:
        """Re-encode video to H.264 for browser compatibility.

        OpenCV's mp4v codec (MPEG-4 Part 2) doesn't play in browsers.
        This re-encodes to H.264 which has universal browser support.

        Args:
            input_path: Path to the video file to re-encode

        Returns:
            True if re-encoding succeeded, False otherwise
        """
        import subprocess
        temp_path = input_path + ".temp.mp4"
        try:
            result = subprocess.run([
                '/usr/bin/ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                temp_path
            ], capture_output=True, timeout=120)
            if result.returncode == 0 and os.path.exists(temp_path):
                os.replace(temp_path, input_path)
                self.logger.debug(f"Re-encoded to H.264: {input_path}")
                return True
            else:
                stderr = result.stderr.decode() if result.stderr else ""
                self.logger.warning(f"H.264 re-encoding failed (code {result.returncode}): {stderr}")
        except FileNotFoundError:
            self.logger.warning("ffmpeg not found - videos will use mp4v codec (may not play in browsers)")
        except subprocess.TimeoutExpired:
            self.logger.warning(f"H.264 re-encoding timed out for: {input_path}")
        except Exception as e:
            self.logger.warning(f"H.264 re-encoding failed: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
        return False

    def save_video_clip(self, frames: List[Any], output_path: str, fps: float) -> None:
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

        # Re-encode to H.264 for browser compatibility
        # (mp4v codec from OpenCV doesn't play in browsers)
        self._reencode_to_h264(output_path)

    def extract_video_segment(self, source_video: str, output_path: str, start_seconds: float, end_seconds: float) -> bool:
        """Extract video segment - delegates to EvidenceManager.

        Args:
            source_video: Path to the source video file
            output_path: Path to save the extracted clip
            start_seconds: Start time in seconds
            end_seconds: End time in seconds

        Returns:
            True if extraction succeeded, False otherwise
        """
        if self.evidence_manager:
            # EvidenceManager expects output_filename (basename), not full path
            output_filename = os.path.basename(output_path)
            result = self.evidence_manager.extract_video_segment(
                source_video, start_seconds, end_seconds, output_filename
            )
            return result is not None
        return False

    def _process_frames_core(
        self,
        frame,
        frame_idx: int,
        timestamp_sec: float,
        sample_idx: int,
        total_frames: int,
        fps: float,
        batch_object_detections=None,
        batch_pose_results=None,
        batch_sleep_pose_results=None,
        batch_idx: int = 0,
        save_clips: bool = True,
        log_per_person_detections: bool = True,
        enable_stale_cleanup: bool = True,
    ) -> None:
        """Core frame processing logic shared by process_video and process_video_range.

        CR-003: Extracted from duplicated code in process_video() and process_video_range().
        This method processes a single sampled frame through the full detection pipeline:
        face mesh, YOLO detection, person deduplication, multi-person activity processing,
        activity lifecycle management, and motion rule engine integration.

        Args:
            frame: The video frame (BGR numpy array)
            frame_idx: Frame index in the source video
            timestamp_sec: Timestamp in seconds for this frame
            sample_idx: Sequential sample index
            total_frames: Total frames in the video (for progress logging)
            fps: Video native FPS
            batch_object_detections: Pre-computed YOLO object detections (batch mode)
            batch_pose_results: Pre-computed YOLO pose results (batch mode)
            batch_sleep_pose_results: Pre-computed low-conf pose results for sleep (batch mode)
            batch_idx: Index into batch results arrays
            save_clips: Whether to pass save_clips to end_activity (False for multiprocessing)
            log_per_person_detections: Whether to log per-person detection details
            enable_stale_cleanup: Whether to call _cleanup_stale_person_tracking
        """
        # Initialize variables for memory cleanup in finally block
        rgb_frame = None
        annotated_frame_for_activity = None

        try:
            # Convert timestamp to HH:MM:SS format
            timestamp = str(timedelta(seconds=timestamp_sec))

            # Add frame to buffer
            self.frame_buffer.append(frame.copy())
            # CR-005: Track frame indices in parallel buffer for activity frame storage
            self.frame_idx_buffer.append(frame_idx)

            # STEP 1: Run MediaPipe Face Mesh on full frame (for head pose/mind diversion detection)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb_frame)

            # STEP 2: Detect objects with YOLO
            # GPU BATCH: Use pre-computed detections from batch inference if available
            if batch_object_detections is not None and batch_idx < len(batch_object_detections):
                detections = batch_object_detections[batch_idx]
            else:
                # Per-frame detection (process_video path or fallback)
                # Preprocess single frame for dark/IR conditions if batch mode fallback
                if batch_object_detections is not None:
                    detection_frame = self.object_detector._preprocess_frames_for_detection([frame])[0]
                    detections = self.object_detector.detect_objects(detection_frame, None, use_pose_guided=False)
                else:
                    detections = self.object_detector.detect_objects(frame, None, use_pose_guided=False)

            # STEP 3: Identify person roles and count people
            people_count = len(detections['person'])
            if people_count == 0:
                people_count = 1  # Default to 1 if no person detected

            # De-duplicate person boxes and identify roles
            group_detected_flag = False
            person_roles = {}

            if len(detections['person']) > 0:
                # De-duplicate person boxes to get accurate count
                # Increased IOU threshold from 0.3 to 0.5 to better filter duplicate detections
                deduplicated_persons = deduplicate_person_boxes(detections['person'], iou_threshold=0.5)

                # Store deduplicated boxes in detections for visualization
                # NOTE: Removed pose validation as it was filtering out legitimate people
                # (MediaPipe struggles with back views, partial occlusions, overhead cameras)
                detections['deduplicated_person'] = deduplicated_persons
                deduplicated_count = len(deduplicated_persons)

                # Identify person roles (LP, ALP, etc.)
                person_roles = self.identify_person_roles(frame, deduplicated_persons, detections)

                # Log role identification (only once per detection cycle)
                if self.consecutive_detections['group_detected'] == 0 and person_roles:
                    self.logger.debug(f"[{timestamp}] Person roles identified:")
                    for person_idx in sorted(person_roles.keys()):
                        role_info = person_roles[person_idx]
                        self.logger.debug(f"  Person {person_idx+1}: {role_info['role_name']} (bbox_area: {role_info.get('bbox_area', 0):.0f})")

                if deduplicated_count > 2:
                    # Stage 2: Voting verification for group_detected (if enabled)
                    if self.voting_service is not None:
                        is_confirmed, vote_details = self.voting_service.verify_activity(
                            video_path=self.current_video_path,
                            timestamp_sec=timestamp_sec,
                            activity_type='group_detected',
                            person_bbox=[0, 0, frame.shape[1], frame.shape[0]]  # Full frame for group detection
                        )
                        if is_confirmed:
                            group_detected_flag = True
                            self.logger.info(f"[VOTING] group_detected CONFIRMED: {vote_details.get('vote_breakdown', [])}")
                        else:
                            group_detected_flag = False
                            self.logger.info(f"[VOTING] group_detected REJECTED: {vote_details.get('vote_breakdown', [])}")
                    else:
                        group_detected_flag = True
                    if group_detected_flag and self.consecutive_detections['group_detected'] == 0:
                        self.logger.info(f"[{timestamp}] Group detected - {deduplicated_count} people (de-duplicated from {len(detections['person'])} raw detections)")
            else:
                # No person detected at all
                detections['deduplicated_person'] = []
                person_roles = {}
                # DEBUG: Log when no person is detected (will be tracked as activity)
                if self.consecutive_detections['no_person_detected'] == 0:
                    raw_detections = len(detections['person'])
                    self.logger.debug(f"[{timestamp}] NO PERSON detected in frame (raw YOLO detections: {raw_detections})")

            # STEP 4: *** MULTI-PERSON PROCESSING ***
            # Process ALL persons individually for ALL activities
            # GPU BATCH: Pass pre-computed pose results if available
            precomputed_poses = None
            if batch_pose_results is not None and batch_idx < len(batch_pose_results):
                precomputed_poses = batch_pose_results[batch_idx]

            precomputed_sleep_poses = None
            if batch_sleep_pose_results is not None and batch_idx < len(batch_sleep_pose_results):
                precomputed_sleep_poses = batch_sleep_pose_results[batch_idx]

            multi_person_results = self.process_all_persons_activities(
                frame, detections, person_roles, timestamp_sec, face_results, frame_idx,
                precomputed_pose_results=precomputed_poses,
                precomputed_sleep_pose_results=precomputed_sleep_poses
            )

            # Extract aggregated detection flags
            persons_data = multi_person_results['persons']
            aggregated = multi_person_results['aggregated']

            # Initialize detection flags from aggregated results
            microsleep_detected = aggregated['microsleep_detected']
            sleep_detected = aggregated['sleep_detected']
            cell_phone_detected = aggregated['cell_phone_detected']
            writing_detected = aggregated['writing_detected']
            packing_detected = aggregated['packing_detected']
            lp_hand_gesture_detected = aggregated['lp_hand_gesture_detected']
            alp_hand_gesture_detected = aggregated['alp_hand_gesture_detected']
            mind_diversion_detected = aggregated['mind_diversion_detected']
            eating_drinking_detected = aggregated.get('eating_drinking_detected', False)

            # Log detections for each person (only on first detection)
            if log_per_person_detections:
                for person_idx, person_data in persons_data.items():
                    activities = person_data['activities']
                    role_name = person_data['role_name']
                    debug_info = person_data['debug_info']

                    # Log mind diversion
                    if activities.get('mind_diversion', False) and self.consecutive_detections.get('mind_diversion', 0) == 0:
                        head_pose = debug_info.get('head_pose', {})
                        yaw = head_pose.get('yaw', 0)
                        pitch = head_pose.get('pitch', 0)
                        method = head_pose.get('method', 'unknown')
                        self.logger.info(f"[{timestamp}] MIND DIVERSION detected for {role_name} (Person {person_idx+1}) - Yaw={yaw:.1f}, Pitch={pitch:.1f} (method: {method})")

                    # Log sleep detection
                    if activities['sleep'] and self.consecutive_detections['sleep'] == 0:
                        sleep_info = debug_info.get('sleep_info', {})
                        self.logger.info(f"[{timestamp}] SLEEP detected for {role_name} (Person {person_idx+1}) - pose-based")

                    # Log microsleep detection
                    if activities['microsleep'] and self.consecutive_detections['microsleep'] == 0:
                        self.logger.info(f"[{timestamp}] MICROSLEEP detected for {role_name} (Person {person_idx+1}) - pose-based")

                    # Log hand gestures
                    if activities['lp_hand_gesture'] and self.consecutive_detections['lp_hand_gesture'] == 0:
                        gesture_debug = debug_info.get('gesture_debug', {})
                        self.logger.info(f"[{timestamp}] LP hand gesture detected for {role_name} (Person {person_idx+1}) - {gesture_debug.get('hand_raised', 'unknown')} hand raised")

                    if activities['alp_hand_gesture'] and self.consecutive_detections['alp_hand_gesture'] == 0:
                        gesture_debug = debug_info.get('gesture_debug', {})
                        self.logger.info(f"[{timestamp}] ALP hand gesture detected for {role_name} (Person {person_idx+1}) - {gesture_debug.get('hand_raised', 'unknown')} hand raised")

                    # Log cell phone, writing, packing
                    if activities['cell_phone'] and self.consecutive_detections['cell_phone'] == 0:
                        self.logger.info(f"[{timestamp}] Cell phone ACTIVELY USED by {role_name} (Person {person_idx+1})")

                    if activities['writing'] and self.consecutive_detections['writing'] == 0:
                        self.logger.info(f"[{timestamp}] WRITING detected for {role_name} (Person {person_idx+1})")

                    if activities.get('packing_bags', False) and self.consecutive_detections.get('packing_bags', 0) == 0:
                        self.logger.info(f"[{timestamp}] PACKING detected for {role_name} (Person {person_idx+1})")

            # GATE: Per-person state machine check (H-02 fix).
            # The primary per-person gate is now applied inside process_all_persons_activities
            # before aggregation. This secondary gate verifies that at least one person with
            # active sleep/microsleep has their own state machine in DROWSY or beyond.
            # Unlike the old code, we only check persons who actually have sleep/microsleep
            # detected (not all persons), preventing cross-person state leakage.
            if microsleep_detected or sleep_detected:
                state_machine_ready = False
                for _pidx, _pdata in persons_data.items():
                    _activities = _pdata.get('activities', {})
                    if not (_activities.get('sleep') or _activities.get('microsleep')):
                        continue  # Skip persons without active sleep/microsleep
                    _sleep_info = _pdata.get('debug_info', {}).get('sleep_info', {})
                    _state = _sleep_info.get('sleep_state', 'ALERT')
                    if _state in ('DROWSY', 'MICROSLEEP', 'SLEEPING'):
                        state_machine_ready = True
                        break
                if not state_machine_ready:
                    self.logger.debug(
                        f"[{timestamp}] [Frame {frame_idx}] Sleep/microsleep SUPPRESSED - "
                        f"no person with active sleep/microsleep has state machine in DROWSY/MICROSLEEP/SLEEPING"
                    )
                    microsleep_detected = False
                    sleep_detected = False

            # CRITICAL: Exclude sleep detection if person is holding objects or in active posture
            # If someone has a phone, book, or backpack in hand, they're clearly NOT sleeping
            # EXCEPTION: If the sleep state machine is in DROWSY/MICROSLEEP/SLEEPING, don't let
            # writing suppress sleep — during microsleep, hands-in-lap + head-down can look like
            # writing posture but the state machine has already determined the person is drowsy.
            # FIX: Also check raw drowsiness indicators from pose_sleep_info directly, so that
            # on the first frame of sleep onset (when state machine is still ALERT), strong
            # drowsiness signals can override writing suppression and allow the state machine
            # to advance. This prevents the chicken-and-egg problem where writing suppresses
            # sleep before the state machine ever reaches DROWSY.
            sleep_state_overrides_writing = False
            if sleep_detected or microsleep_detected:
                # H-02 fix: Only check persons who have active sleep/microsleep,
                # not all persons. This prevents cross-person state leakage where
                # person 0's SLEEPING state overrides writing suppression for person 1.
                for _pidx, _pdata in persons_data.items():
                    _activities = _pdata.get('activities', {})
                    if not (_activities.get('sleep') or _activities.get('microsleep')):
                        continue  # Skip persons without active sleep/microsleep
                    _sleep_info = _pdata.get('debug_info', {}).get('sleep_info', {})
                    _state = _sleep_info.get('sleep_state', 'ALERT')
                    if _state in ('DROWSY', 'MICROSLEEP', 'SLEEPING'):
                        sleep_state_overrides_writing = True
                        break
                    # Check raw drowsiness indicators even when state machine is still ALERT.
                    # head_drop_detected or significant nose_y_drop indicate the person's head
                    # is dropping -- a strong physical signal that should not be suppressed by
                    # writing detection. haar_eye_closed similarly indicates closed eyes.
                    _head_drop = _sleep_info.get('head_drop_detected', False)
                    _nose_y_drop = _sleep_info.get('nose_y_drop', 0.0)
                    _haar_eye_closed = _sleep_info.get('haar_eye_closed', False)
                    if _head_drop or _nose_y_drop > 0.05 or _haar_eye_closed:
                        sleep_state_overrides_writing = True
                        self.logger.debug(
                            f"[{timestamp}] Writing suppression overridden by raw drowsiness "
                            f"indicators: head_drop={_head_drop}, nose_y_drop={_nose_y_drop:.4f}, "
                            f"haar_eye_closed={_haar_eye_closed}"
                        )
                        break
            suppress_activities = cell_phone_detected or packing_detected
            if writing_detected and not sleep_state_overrides_writing:
                suppress_activities = True
            if suppress_activities:
                if log_per_person_detections and (microsleep_detected or sleep_detected):
                    reason = []
                    if cell_phone_detected: reason.append("phone")
                    if writing_detected and not sleep_state_overrides_writing: reason.append("book")
                    if packing_detected: reason.append("backpack")
                    self.logger.debug(f"[{timestamp}] Sleep detection OVERRIDDEN - person active ({', '.join(reason)})")
                microsleep_detected = False
                sleep_detected = False

            # Debug: log sleep detection state after override check
            if sleep_detected or microsleep_detected:
                self.logger.info(
                    f"[{timestamp}] [Frame {frame_idx}] SLEEP/MICROSLEEP PASSED override check: "
                    f"sleep={sleep_detected}, microsleep={microsleep_detected}, "
                    f"writing={writing_detected}, override={sleep_state_overrides_writing}"
                )

            # Create annotated frame with all detections (pose landmarks + YOLO boxes)
            # This annotated frame will be used for BOTH activity clips AND periodic frame saving
            annotated_frame_for_activity = self.frame_annotator.draw_bounding_boxes(
                frame, detections, show_roi_boxes=True, person_roles=person_roles
            )
            # NEW: Draw MediaPipe outputs for ALL persons (not just one)
            annotated_frame_for_activity = self.draw_multi_person_mediapipe_outputs(
                annotated_frame_for_activity,
                persons_data,  # All persons' pose landmarks and activities
                face_results
            )

            # Draw sleep detection debug overlay for each person
            for pidx, pdata in persons_data.items():
                sleep_info = pdata.get('debug_info', {}).get('sleep_info')
                if sleep_info:
                    annotated_frame_for_activity = self.frame_annotator.draw_sleep_debug_overlay(
                        annotated_frame_for_activity, sleep_info, pidx,
                        pdata.get('activities', {}), timestamp_sec
                    )

            # Save annotated frames periodically if enabled (AFTER all detections)
            if (
                self.save_annotated_frames
                and self.frames_dir is not None
                and sample_idx % self.frame_save_interval == 0
            ):
                try:
                    # Save frame with unique filename
                    frame_filename = f"frame_{frame_idx:08d}.jpg"
                    frame_path = os.path.join(self.frames_dir, frame_filename)

                    # Ensure directory exists (for multiprocessing safety)
                    os.makedirs(self.frames_dir, exist_ok=True)

                    # Save with high quality
                    cv2.imwrite(frame_path, annotated_frame_for_activity, [cv2.IMWRITE_JPEG_QUALITY, 95])

                except Exception as e:
                    self.logger.error(f"[{timestamp}] Error saving frame {frame_idx}: {e}")

            # CRITICAL: Hand gesture coordination check
            # Activity Type 8 (LP not exchanging): Triggers when ALP raises hand BUT LP does NOT
            # Activity Type 9 (ALP not exchanging): Triggers when LP raises hand BUT ALP does NOT
            # This ensures we detect COORDINATION FAILURES, not individual gestures
            # Uses temporal window to prevent false positives when both people raise hands within window
            lp_not_coordinating, alp_not_coordinating = self._check_hand_gesture_coordination(
                lp_hand_gesture_detected,
                alp_hand_gesture_detected,
                timestamp_sec
            )

            # Debug logging for coordination check
            if lp_not_coordinating and self.consecutive_detections['lp_hand_gesture'] == 0:
                self.logger.info(f"[{timestamp}] [Frame {frame_idx}] COORDINATION FAILURE: ALP raised hand but LP did NOT respond")
            if alp_not_coordinating and self.consecutive_detections['alp_hand_gesture'] == 0:
                self.logger.info(f"[{timestamp}] [Frame {frame_idx}] COORDINATION FAILURE: LP raised hand but ALP did NOT respond")

            # Detect when no person is in frame
            no_person_detected_flag = (len(detections.get('deduplicated_person', [])) == 0)

            # ==========================================
            # TRAIN MOTION RULE ENGINE INTEGRATION
            # ==========================================
            # 1. Extract OCR timestamp from frame (if enabled)
            ocr_timestamp = self._extract_ocr_timestamp(frame)

            # 2. Resolve train motion context
            # Priority: OCR timestamp > video_start_time + offset > video-relative time
            if ocr_timestamp:
                motion_timestamp = ocr_timestamp
                # Log OCR success periodically (every 30 frames to avoid spam)
                if frame_idx % 30 == 0:
                    self.logger.info(f"[MOTION-RULES] Frame {frame_idx}: OCR extracted timestamp: {motion_timestamp}")
            else:
                # Convert video offset to real time (if video_start_time is set)
                # Use timestamp_sec directly (it's already numeric) instead of parsing timestamp string
                video_seconds = timestamp_sec
                motion_timestamp = self._convert_video_to_real_time(video_seconds)
                has_start_time = hasattr(self, 'video_start_time') and self.video_start_time
                # Log periodically (every 60 frames)
                if frame_idx % 60 == 0:
                    if has_start_time:
                        self.logger.info(
                            f"[MOTION-RULES] Frame {frame_idx}: Using video_start_time + offset = {motion_timestamp}"
                        )
                    else:
                        self.logger.info(
                            f"[MOTION-RULES] Frame {frame_idx}: No OCR/start_time - using video-relative: {motion_timestamp}"
                        )
            # Pass frame for optical flow-enhanced motion detection
            # prev_motion_frame is set at end of loop
            prev_motion_frame = self._prev_motion_frame
            self.current_motion_context = self._resolve_motion_context(
                motion_timestamp, frame=frame, prev_frame=prev_motion_frame
            )
            # Store current frame for next iteration's optical flow
            self._prev_motion_frame = frame.copy()

            # Log motion state periodically
            if frame_idx % 60 == 0 and self.current_motion_context:
                self.logger.info(
                    f"[MOTION-RULES] Frame {frame_idx}: State={self.current_motion_context.motion_state.value}, "
                    f"Station={self.current_motion_context.current_station.station_code if self.current_motion_context.current_station else 'N/A'}, "
                    f"Source={self.current_motion_context.resolution_source}"
                )

            # 3. Check ALP pre-arrival alertness (if in pre-arrival window)
            alp_not_standing_detected = False
            if (self.rule_engine is not None and
                self.current_motion_context is not None and
                self.rule_engine.should_check_alp_alertness(self.current_motion_context)):
                alp_not_standing_detected = self._check_alp_standing(
                    persons_data, person_roles, frame.shape
                )
                if alp_not_standing_detected and self.consecutive_detections.get('alp_not_standing', 0) == 0:
                    self.logger.info(
                        f"[{timestamp}] ALP NOT STANDING in pre-arrival window - "
                        f"{self.current_motion_context.seconds_to_arrival}s to arrival"
                    )

            # Update activity states with temporal filtering
            activities_map = {
                'microsleep': microsleep_detected and not sleep_detected,
                'sleep': sleep_detected,
                'cell_phone': cell_phone_detected,
                'writing': writing_detected,
                'packing_bags': packing_detected,
                'group_detected': group_detected_flag,
                'lp_hand_gesture': lp_not_coordinating,  # LP fails to respond when ALP raises hand
                'alp_hand_gesture': alp_not_coordinating,  # ALP fails to respond when LP raises hand
                'mind_diversion': mind_diversion_detected,
                'eating_drinking': eating_drinking_detected,
                'no_person_detected': no_person_detected_flag,
                'alp_not_standing': alp_not_standing_detected  # ALP not standing in pre-arrival
            }

            # 3.5. Suppress no_person_detected when trip schedule is unavailable
            if self.suppress_no_person_without_schedule and self.trip_schedule is None:
                if activities_map.get('no_person_detected', False):
                    activities_map['no_person_detected'] = False
                    self.logger.debug(
                        f"[{timestamp}] no_person_detected suppressed - "
                        f"no trip schedule available to distinguish station halts"
                    )

            # 4. Apply motion-based rule engine to filter exempted activities
            activities_map = self._apply_motion_rules(activities_map, self.current_motion_context)

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
                            self.start_activity(activity_name, timestamp, fps, frame_idx, person_roles=person_roles, ocr_timestamp=ocr_timestamp)

                        # Continue recording frames ONLY when activity is actively detected
                        if self.activities[activity_name]['active']:
                            # CR-005: Store frame index instead of frame copy to reduce memory usage
                            self.activities[activity_name]['frames'].append(frame_idx)
                            self.activities[activity_name]['last_frame_count'] = frame_idx
                            self.activities[activity_name]['last_detected_frame'] = frame_idx  # Track last actual detection
                            self.activities[activity_name]['last_detection_time'] = timestamp  # Track for precise clip duration
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
                                # No ocr_timestamp: current frame is post-grace-period, not when activity ended.
                                # end_activity computes ocr_end from ocr_start + duration instead (more accurate).
                                if save_clips:
                                    self.end_activity(activity_name, timestamp, fps, frame_idx, people_count)
                                else:
                                    self.end_activity(activity_name, timestamp, fps, frame_idx, people_count, save_clips=save_clips)
                            self.consecutive_detections[activity_name] = 0
                            self.grace_counters[activity_name] = 0
                    else:
                        # Reset grace counter if nothing is being tracked
                        self.grace_counters[activity_name] = 0

            # CR-012: Clean up stale per-person tracking dicts after each frame
            if enable_stale_cleanup and persons_data:
                self._cleanup_stale_person_tracking(set(persons_data.keys()))

            # Display progress with detection status
            if sample_idx % 50 == 0:  # Show progress every 50 sampled frames
                progress = (frame_idx / total_frames) * 100
                self.logger.info(f"Progress: {sample_idx} samples processed (frame {frame_idx}/{total_frames}, {progress:.1f}%)")

                # Show current detection counts for debugging
                active_detections = []
                for act_name, count in self.consecutive_detections.items():
                    if count > 0:
                        threshold = self.activity_thresholds[act_name]['required_consecutive']
                        status = "RECORDING" if self.activities[act_name]['active'] else f"building {count}/{threshold}"
                        active_detections.append(f"{act_name}: {status}")

                if active_detections:
                    self.logger.debug(f"  Active detections: {', '.join(active_detections)}")

        except Exception as e:
            self.logger.error(f"Error processing sample {sample_idx} (frame {frame_idx}): {e}")
        finally:
            # MEMORY FIX: Explicitly delete frame after processing to free memory
            if annotated_frame_for_activity is not None:
                del annotated_frame_for_activity
            if rgb_frame is not None:
                del rgb_frame


    def process_video(self) -> None:
        """Main video processing loop - SAMPLES FRAMES AT SPECIFIED RATE"""
        # Get video metadata
        # CR-011: Use cached video metadata instead of reopening VideoCapture
        total_frames, fps, _duration = self._get_video_metadata()

        # Calculate expected sampled frames
        step = max(1, int(round(fps / max(1e-6, float(self.sample_fps)))))
        expected_samples = (total_frames // step)
        
        self.logger.info(f"Processing video: {self.video_path}")
        self.logger.info(f"Native FPS: {fps:.2f}")
        self.logger.info(f"Sample FPS: {self.sample_fps} (1 frame every {1.0/self.sample_fps:.1f} seconds)")
        self.logger.info(f"Total frames in video: {total_frames}")
        self.logger.info(f"Expected duration: {total_frames/fps/60:.2f} minutes")
        self.logger.info(f"Expected sampled frames: ~{expected_samples}")
        self.logger.info(f"Processing speed-up: ~{step}x faster")
        self.logger.info(f"Run directory: {self.run_dir}")
        if self.save_annotated_frames:
            if self.frame_save_interval == 1:
                self.logger.info(f"  Saving ALL sampled frames (~{expected_samples} frames) to: {self.frames_dir}")
            else:
                self.logger.info(f"  Saving every {self.frame_save_interval}th sampled frame (~{expected_samples//self.frame_save_interval} frames) to: {self.frames_dir}")
        else:
            self.logger.info("  Annotated frame saving is disabled (faster processing)")
        self.logger.info("-" * 60)
        
        sampled_count = 0
        
        # Use the frame sampling generator
        for sample_idx, timestamp_sec, frame, frame_idx in self.sample_video_frames(self.video_path):
            sampled_count += 1

            self._process_frames_core(
                frame=frame,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                sample_idx=sample_idx,
                total_frames=total_frames,
                fps=fps,
                save_clips=True,
                log_per_person_detections=True,
                enable_stale_cleanup=True,
            )
            # MEMORY FIX: Explicitly delete frame after processing to free memory
            if frame is not None:
                del frame

        # End any remaining active activities
        # No ocr_timestamp: video ended, no specific end frame to OCR.
        # end_activity computes ocr_end from ocr_start + duration instead.
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, people_count=1)

        # [OK] MEMORY FIX: Clear frame buffers and activity frames to free memory
        self.frame_buffer.clear()
        self.frame_idx_buffer.clear()
        for activity_name in self.activities:
            if 'frames' in self.activities[activity_name]:
                self.activities[activity_name]['frames'].clear()

        # [OK] MEMORY FIX: Force garbage collection
        gc.collect()
        
        self.logger.info("=" * 60)
        self.logger.info("Processing complete!")
        self.logger.info(f"Total frames sampled: {sampled_count}/{total_frames}")
        self.logger.info(f"Sampling rate: {self.sample_fps} FPS (1 frame every {1.0/self.sample_fps:.1f} seconds)")
        self.logger.info(f"Processing speed-up: ~{step}x faster than full-frame processing")
        self.logger.info(f"Evidence clips created: {self.evidence_counter}")
        self.logger.info(f"Run directory: {self.run_dir}")
        self.logger.info(f"  - Clips: {self.evidence_clips_dir}")
        if self.save_annotated_frames:
            self.logger.info(f"  - Frames: {self.frames_dir}")
        self.logger.info(f"  - Activities: {os.path.join(self.run_dir, 'activities.json')}")

        # Log voting verification cache statistics
        if self.voting_service is not None:
            try:
                cache_stats = self.voting_service.get_cache_stats()
                self.logger.info("-" * 40)
                self.logger.info("[VOTING CACHE STATS]")
                self.logger.info(f"  Cache hits: {cache_stats.get('hits', 0)}")
                self.logger.info(f"  Cache misses: {cache_stats.get('misses', 0)}")
                self.logger.info(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
                self.logger.info(f"  Cache size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
                # Clear cache after video processing to free memory
                self.voting_service.clear_cache()
                self.logger.info("  Cache cleared for next video")
            except Exception as e:
                self.logger.debug(f"Could not get cache stats: {e}")

        self.logger.info("=" * 60)

        # Generate summary report
        self.generate_summary_report()
    
    def process_video_range(self, start_frame: int, end_frame: int, save_clips: bool = False) -> list:
        """
        Process a specific frame range (for multiprocessing support)

        This method processes only frames within the specified range and returns
        detected activities without saving clips/images to disk (activities in memory only).

        GPU OPTIMIZED: Uses batch inference for YOLO object and pose detection.
        Phase 1: Collect all frames
        Phase 2: Batch YOLO object detection
        Phase 3: Batch YOLO pose detection
        Phase 4: Sequential per-frame activity processing with pre-computed detections

        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
            save_clips: Whether to save video clips and images (default: False for multiprocessing)

        Returns:
            List of detected activities in this range
        """
        # Get video metadata
        # CR-011: Use cached video metadata instead of reopening VideoCapture
        total_frames, fps, _duration = self._get_video_metadata()

        self.logger.info(f"Processing frame range {start_frame}-{end_frame} (worker {os.getpid()})")

        # NOTE: required_consecutive thresholds are preserved for multiprocessing chunks.
        # Lowering them to 1 caused excessive false positives (e.g., no_person_detected
        # triggering on single frames). Voting verification handles cross-chunk consistency.

        # =========================================================================
        # GPU BATCH OPTIMIZATION: Collect frames, run batch inference, then process
        # =========================================================================

        # Get batch settings from instance variables
        batch_size = self.gpu_batch_size
        batch_enabled = self.gpu_batch_enabled

        # CR-013: Initialize optical flow state for chunk boundaries
        # In multiprocessing mode, each worker starts with _prev_motion_frame = None,
        # so the first frame(s) of each chunk produce no optical flow results.
        # Fix: read the frame immediately before this chunk's start to seed optical flow.
        if start_frame > 0 and self._prev_motion_frame is None:
            try:
                with video_capture_context(self.video_path) as _init_cap:
                    _init_cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame - 1))
                    _init_ret, _init_frame = _init_cap.read()
                    if _init_ret and _init_frame is not None:
                        self._prev_motion_frame = _init_frame.copy()
                        self.logger.debug(
                            f"[CR-013] Initialized optical flow with frame {start_frame - 1} "
                            f"for chunk starting at {start_frame} (worker {os.getpid()})"
                        )
                    else:
                        self.logger.debug(
                            f"[CR-013] Could not read frame {start_frame - 1} for optical flow init"
                        )
            except Exception as e:
                self.logger.debug(f"[CR-013] Optical flow pre-init failed: {e}")

        # PHASE 1: Collect all frames for this chunk
        frames_data = []  # List of (frame, frame_idx, timestamp_sec, sample_idx)
        for sample_idx, timestamp_sec, frame, frame_idx in self.sample_video_frames(
            self.video_path, start_frame=start_frame, end_frame=end_frame
        ):
            frames_data.append((frame.copy(), frame_idx, timestamp_sec, sample_idx))

        if not frames_data:
            self.logger.warning(f"No frames sampled in range {start_frame}-{end_frame}")
            return self.all_activities

        self.logger.info(f"[GPU BATCH] Collected {len(frames_data)} frames for batch processing (batch_size={batch_size})")

        # PHASE 2 & 3: Batch YOLO inference (object + pose detection)
        if batch_enabled and len(frames_data) > 1:
            # Run batch object detection
            frames_only = [fd[0] for fd in frames_data]

            # Preprocess dark/IR frames for better YOLO person detection
            detection_frames = self.object_detector._preprocess_frames_for_detection(frames_only)

            self.logger.debug(f"[GPU BATCH] Running batch object detection on {len(frames_only)} frames")
            batch_object_detections = self.object_detector.detect_objects_batch(detection_frames, batch_size)

            # Run batch pose detection at normal confidence (for hand gestures, mind diversion, etc.)
            self.logger.debug(f"[GPU BATCH] Running batch pose detection on {len(frames_only)} frames")
            batch_pose_results = self.detect_poses_batch(frames_only, batch_size)

            # Run a second lower-confidence pose pass for sleep detection
            # Sleeping persons often have low YOLO confidence; 0.30 captures more frames
            sleep_pose_conf = getattr(self.settings, 'yolo_pose_sleep_confidence', self.YOLO_SLEEP_POSE_CONFIDENCE) if self.settings else self.YOLO_SLEEP_POSE_CONFIDENCE
            if sleep_pose_conf < self.yolo_pose.conf_threshold:
                self.logger.debug(f"[GPU BATCH] Running low-confidence pose pass for sleep detection (conf={sleep_pose_conf})")
                batch_sleep_pose_results = self.detect_poses_batch(frames_only, batch_size, conf_threshold=sleep_pose_conf)
            else:
                batch_sleep_pose_results = None

            self.logger.info(f"[GPU BATCH] Batch inference complete: {len(batch_object_detections)} object results, {len(batch_pose_results)} pose results")

            # Release frames_only to free memory after batch inference
            del frames_only
            gc.collect()
        else:
            # Fallback: No batching (will compute per-frame)
            batch_object_detections = None
            batch_pose_results = None
            batch_sleep_pose_results = None

        # PHASE 4: Sequential per-frame activity processing with pre-computed detections
        sampled_count = 0
        timestamp_sec = 0
        frame_idx = start_frame

        for idx, (frame, frame_idx, timestamp_sec, sample_idx) in enumerate(frames_data):
            sampled_count += 1

            self._process_frames_core(
                frame=frame,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                sample_idx=sample_idx,
                total_frames=total_frames,
                fps=fps,
                batch_object_detections=batch_object_detections,
                batch_pose_results=batch_pose_results,
                batch_sleep_pose_results=batch_sleep_pose_results,
                batch_idx=idx,
                save_clips=save_clips,
                log_per_person_detections=False,
                enable_stale_cleanup=False,
            )
            # MEMORY FIX: Explicitly delete frame after processing to free memory
            if frame is not None:
                del frame

        # Guard against empty frame ranges where timestamp_sec/frame_idx may not be set
        if sampled_count == 0:
            timestamp_sec = start_frame / fps if fps > 0 else 0.0
            frame_idx = start_frame
            self.logger.warning(f"No frames sampled in range {start_frame}-{end_frame}, skipping activity finalization")
            return self.all_activities

        # End any remaining active activities
        # No ocr_timestamp: video range ended, no specific end frame to OCR.
        # end_activity computes ocr_end from ocr_start + duration instead.
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, people_count=1, save_clips=save_clips)

        # [OK] MEMORY FIX: Clear frame buffers and activity frames to free memory
        self.frame_buffer.clear()
        self.frame_idx_buffer.clear()
        for activity_name in self.activities:
            if 'frames' in self.activities[activity_name]:
                self.activities[activity_name]['frames'].clear()

        # [OK] MEMORY FIX: Force garbage collection
        gc.collect()

        self.logger.info(f"Frame range {start_frame}-{end_frame} completed: {len(self.all_activities)} activities")
        
        # Return detected activities (without generating summary reports)
        return self.all_activities

    def cleanup(self) -> None:
        """
        [OK] MEMORY FIX: Cleanup method to release model resources

        This method mirrors POC_2's MediaPipeService.close() pattern.
        Call this after processing to free GPU/CPU resources.
        
        NOTE: If models were pre-loaded (worker pool), they are NOT closed
        since they are shared across tasks in the same worker.
        """
        try:
            # Only close models if they were loaded fresh (not pre-loaded from worker pool)
            if not getattr(self, '_models_preloaded', False):
                # Close YOLO26-Pose model
                if hasattr(self, 'yolo_pose') and self.yolo_pose is not None:
                    self.yolo_pose = None

                # Close MediaPipe FaceMesh (still used for EAR calculation)
                if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                    self.face_mesh.close()
                    self.face_mesh = None
            else:
                # Pre-loaded models: just clear references (don't close shared models)
                self.yolo_pose = None
                self.yolo_model = None
                self.face_mesh = None
            
            # Clear frame buffers (always)
            if hasattr(self, 'frame_buffer'):
                self.frame_buffer.clear()
            if hasattr(self, 'frame_idx_buffer'):
                self.frame_idx_buffer.clear()

            # Clear activity frames (always)
            if hasattr(self, 'activities'):
                for activity_name in self.activities:
                    if 'frames' in self.activities[activity_name]:
                        self.activities[activity_name]['frames'].clear()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Cleanup completed: Models closed, buffers cleared")
        except Exception as e:
            self.logger.warning(f"Warning during cleanup: {e}")
    
    def __del__(self):
        """
        [OK] MEMORY FIX: Destructor to ensure cleanup on object deletion
        Note: __del__ is not reliable for cleanup - prefer explicit cleanup() calls
        """
        try:
            self.cleanup()
        except Exception as e:
            # Log at debug level - exceptions in __del__ should not be propagated
            try:
                self.logger.debug(f"Exception in __del__ cleanup: {e}")
            except Exception:
                pass  # Logger may not be available during interpreter shutdown
    
    def generate_summary_report(self) -> None:
        """Generate activities.json in the run directory - delegates to EvidenceManager."""
        if self.evidence_manager:
            self.evidence_manager.generate_summary_report(self.all_activities, save_json=True)
        else:
            # Fallback: save activities.json directly if no evidence manager
            activities_json_path = os.path.join(self.run_dir, "activities.json")
            with open(activities_json_path, 'w') as f:
                json.dump(self.all_activities, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)
            self.logger.info(f"Activities JSON saved: {activities_json_path}")
            self.logger.info(f"Total activities detected: {len(self.all_activities)}")


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