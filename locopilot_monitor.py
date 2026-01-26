import cv2
import json
import numpy as np
import time as time_module  # Renamed to avoid any potential shadowing issues
from datetime import datetime, timedelta
from collections import deque
import mediapipe as mp
from ultralytics import YOLO
import os
import logging
import gc
import contextlib
import sys
import subprocess

# Add app directory to path for importing preprocessing service
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import preprocessing service, config, and voting service
try:
    from app.services.image_preprocessing_service import ImagePreprocessingService
    from app.utils.config import get_settings
    from app.services.voting_verification_service import VotingVerificationService, ActivityBatchCollector
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

# Import VotingVerificationService separately (may not exist in fallback path)
try:
    if 'VotingVerificationService' not in dir():
        from app.services.voting_verification_service import VotingVerificationService, ActivityBatchCollector
except ImportError:
    VotingVerificationService = None
    ActivityBatchCollector = None

# Import TrainStateDetector for stopped train detection
try:
    from app.services.train_state_detection_service import TrainStateDetector, TrainState
except ImportError:
    TrainStateDetector = None
    TrainState = None


# ✅ WINDOWS FIX: Prevent Qt/GUI initialization in worker processes
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

# Module-level loggers (file-only output)
gesture_logger = _setup_module_logger('HandGestureDetection')
monitor_logger = _setup_module_logger('LocopilotMonitor')


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
    def __init__(self, video_path, output_dir="evidence", save_annotated_frames=False, frame_save_interval=1, sample_fps=1.0, run_dir=None, create_run_dir=True, preloaded_models=None):
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

        # Initialize models - either use preloaded or load fresh
        if preloaded_models is not None:
            # ✅ PERFORMANCE: Use pre-loaded models from worker pool (fast path)
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
            yolo_weights = settings.yolo_weights if settings else 'yolo11m.pt'
            yolo_pose_weights = settings.yolo_pose_weights if settings else 'yolo11m-pose.pt'
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
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
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

        # Cell phone detection confidence threshold (configurable)
        self.cell_phone_confidence = float(os.getenv("CELL_PHONE_CONFIDENCE", "0.40"))

        # Phase 2: Load inference optimization settings (1.5-1.8x speedup)
        self.yolo_imgsz = settings.yolo_imgsz if settings else 416
        self.yolo_device = settings.yolo_device if settings else 'cpu'

        # GPU Batch Processing Settings - maximize GPU utilization
        # Use settings if available, otherwise fall back to environment variables
        self.gpu_batch_size = getattr(settings, 'gpu_batch_size', None) or int(os.getenv('GPU_BATCH_SIZE', '8'))
        self.gpu_batch_enabled = getattr(settings, 'gpu_batch_enabled', None) if settings else bool(int(os.getenv('GPU_BATCH_ENABLED', '1')))

        # Track if models were pre-loaded (don't close them in cleanup)
        self._models_preloaded = preloaded_models is not None

        # Activity tracking with temporal filtering
        # Each activity now also tracks OCR timestamps (ocr_start_time, ocr_end_time) for embedded frame timestamps
        self.activities = {
            'microsleep': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'sleep': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'cell_phone': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'writing': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'packing_bags': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'group_detected': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'lp_hand_gesture': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'alp_hand_gesture': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'mind_diversion': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'no_person_detected': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0},
            'alp_not_standing_before_stop': {'active': False, 'start_time': None, 'ocr_start_time': None, 'frames': [], 'duration': 0}
        }
        
        # TEMPORAL SUPPRESSION: Track recent activities per person for gesture suppression
        # Format: {person_idx: {'writing': last_timestamp, 'packing': last_timestamp, 'cell_phone': last_timestamp}}
        self.recent_person_activities = {}
        self.temporal_suppression_window = 10.0  # Suppress hand gestures for 10 seconds after detecting work activity

        # Hand gesture coordination temporal window (legacy - kept for reference)
        # Suppress coordination failure alerts if both LP and ALP raised hands within this window
        self.hand_gesture_coordination_window = float(os.getenv('HAND_GESTURE_COORDINATION_WINDOW', '5.0'))

        # Session-based hand gesture coordination tracking
        # Replaces rolling window approach to fix false positives when ALP raises multiple times
        # after LP's single instruction raise
        self.hand_coordination_session = {
            'active': False,
            'start_time': None,
            'lp_raised': False,
            'alp_raised': False,
            'last_activity_time': None,
            'lp_raise_count': 0,
            'alp_raise_count': 0
        }
        self.hand_coordination_session_timeout = float(os.getenv('HAND_GESTURE_SESSION_TIMEOUT', '10.0'))

        # ALP Standing Before Stop tracking
        # Tracks if ALP stands up before train stops at station
        self.alp_standing_tracking = {
            'monitoring_active': False,      # True when APPROACHING_STOP detected
            'approaching_start_time': None,  # When APPROACHING_STOP began
            'alp_stood_time': None,          # When ALP first stood (None if never)
            'alp_visible': False,            # Whether ALP pose is visible
            'violation_flagged': False       # Prevent duplicate violations per stop
        }
        # Get ALP standing required seconds from config
        self.alp_standing_required_seconds = settings.alp_standing_required_seconds if settings else 30.0

        # Activity thresholds: minimum duration and required consecutive frames before recording starts
        # OPTIMIZED FOR 0.5 FPS SAMPLING (1 frame every 2 seconds)
        self.activity_thresholds = {
            'packing_bags': {
                'min_duration': 0.0,          # NO minimum duration - any detection creates activity
                'required_consecutive': 1,    # Immediate detection when wrist inside backpack bbox
                'margin': 100,                # Hand proximity margin in pixels (was 50)
                'region_margin': 150,         # Region overlap margin for person-backpack association
                'grace_frames': 5,            # Allow 5 samples (~10s) gap to group nearby detections
                'wrist_inside_margin': 80,    # Margin for wrist-inside-bbox check (was 40 - INCREASED)
                'sustained_proximity_seconds': 4.0  # If hand near backpack for 4+ seconds, detect as packing
            },
            'writing': {
                'min_duration': 0.1,          # NO minimum duration - any detection creates activity
                'required_consecutive': 1,    # Instant detection when book+hand seen
                'margin': 180,                # Hand-to-book proximity - INCREASED from 100 to 150 for better capture
                'grace_frames': 10,            # Allow 10 samples (~20s) gap to group nearby detections
                # NOTE: Pose-based detection (wrist proximity + head down) uses separate internal
                #       threshold in detect_writing_by_wrist_proximity()
            },
            'cell_phone': {
                'min_duration': 0.1,          # NO minimum duration - any detection creates activity
                'required_consecutive': 1,    # Instant detection on first frame
                'margin': 180,                # MAXIMUM proximity for detecting phone near hand/ear/shoulder
                'grace_frames': 8             # Allow 8 samples (~16s) gap to group nearby detections
            },
            'microsleep': {
                'min_duration': 3.0,          # Must last 3 seconds minimum (reduced from 5.0 for early detection)
                'required_consecutive': 2,    # 2 samples @ 0.5fps = 4 seconds (reduced from 3)
                'margin': None,               # N/A for eye-based detection
                'grace_frames': 10            # Allow 10 frames (~20s) of non-detection
            },
            'sleep': {
                'min_duration': 20.0,         # Must last 20 seconds minimum (reduced from 30s)
                'required_consecutive': 4,    # 4 samples @ 0.5fps = 8 seconds (reduced from 5)
                'margin': None,               # N/A for eye-based detection
                'grace_frames': 10            # Allow 10 frames (~20s) of non-detection
            },
            'group_detected': {
                'min_duration': 0.0,          # NO minimum duration - any detection creates activity
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds - INCREASED for temporal consistency
                'margin': None,               # N/A for person count detection
                'grace_frames': 8             # Allow 8 samples (~16s) gap
            },
            'lp_hand_gesture': {
                'min_duration': 0.0,          # NO minimum duration - any coordination failure creates activity
                'required_consecutive': 1,    # Instant detection on first frame
                'margin': None,               # N/A for hand gesture detection
                'grace_frames': 5             # Allow 5 samples (~10s) gap to handle multiple raises
            },
            'alp_hand_gesture': {
                'min_duration': 0.0,          # NO minimum duration - any coordination failure creates activity
                'required_consecutive': 1,    # Instant detection on first frame
                'margin': None,               # N/A for hand gesture detection
                'grace_frames': 5             # Allow 5 samples (~10s) gap to handle multiple raises
            },
            'mind_diversion': {
                'min_duration': 0.0,          # NO minimum duration - any detection creates activity
                'required_consecutive': 2,    # 2 samples @ 0.5fps = 4 seconds (reduced from 3)
                'margin': None,               # N/A for head pose detection
                'grace_frames': 5             # Allow 5 samples (~10s) gap
            },
            'no_person_detected': {
                'min_duration': 5.0,          # Must last 5 seconds minimum (increased from 2.0)
                'required_consecutive': 3,    # 3 samples @ 0.5fps = 6 seconds (increased from 1)
                'margin': None,               # N/A for person detection
                'grace_frames': 3             # Allow 3 samples (~6s) gap for brief detection failures (reduced from 5)
            },
            'alp_not_standing_before_stop': {
                'min_duration': 0.0,          # Instant violation when train stops without ALP standing
                'required_consecutive': 1,    # Single detection triggers violation
                'margin': None,               # N/A for pose-based detection
                'grace_frames': 0             # No grace - violation is one-time per stop
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
            'mind_diversion': 0,
            'no_person_detected': 0,
            'alp_not_standing_before_stop': 0
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
            'mind_diversion': 0,
            'no_person_detected': 0,
            'alp_not_standing_before_stop': 0
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
        
        # Wrist proximity tracking for writing detection (per person)
        # Format: {person_idx: {'start_time': timestamp, 'duration': seconds, 'consecutive_frames': int}}
        self.wrist_proximity_tracking = {}

        # Per-person consecutive detection tracking for temporal filtering
        # Format: {person_idx: {'cell_phone': count, 'writing': count, 'packing_bags': count}}
        self.per_person_consecutive_detections = {}
        self.per_person_grace_counters = {}

        # Hand position history for velocity/trajectory analysis
        # Format: {person_idx: {'right_wrist': deque([coords]), 'left_wrist': deque([coords]), 'timestamps': deque([t])}}
        self.hand_position_history = {}
        self.hand_history_max_length = 10  # Track last 10 positions (~20s at 0.5 fps)

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
            'alp_not_standing_before_stop': 12
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
            'alp_not_standing_before_stop': 'ALP not standing before station stop'
        }
        
        # Evidence rules
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'eyes_closed_5s_or_pose_indicators',
            'sleep': 'eyes_closed_30s_or_pose_indicators',
            'writing': 'hand_near_book_or_wrist_proximity',
            'packing_bags': 'wrist_inside_backpack_bbox_or_hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons',
            'lp_hand_gesture': 'lp_hand_raised_gesture_detected',
            'alp_hand_gesture': 'alp_hand_raised_gesture_detected',
            'mind_diversion': 'attention_diverted_from_controls',  # Sub-type stored in evidence details
            'no_person_detected': 'zero_persons_in_frame'
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

        # Initialize voting verification service for multi-frame voting
        # This reduces false positives by verifying detections across multiple native frames
        self.current_video_path = video_path  # Track current video for voting
        if VotingVerificationService is not None:
            try:
                self.voting_service = VotingVerificationService(
                    yolo_model=self.yolo_model,
                    yolo_pose_model=self.yolo_pose
                )
                self.logger.info("VotingVerificationService initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize VotingVerificationService: {e}")
                self.voting_service = None
        else:
            self.voting_service = None
            self.logger.info("VotingVerificationService not available - voting disabled")

        # Initialize train state detector for stopped train detection
        # When train is stopped, certain activities (cell_phone, writing, packing_bags, mind_diversion) are exempted
        if TrainStateDetector is not None and settings is not None:
            if getattr(settings, 'train_state_detection_enabled', True):
                try:
                    train_state_config = {
                        'motion_threshold': getattr(settings, 'train_state_motion_threshold', 2.0),
                        'min_stopped_duration': getattr(settings, 'train_state_min_stopped_duration', 5.0),
                        # ROI: Side Window - TOP ONLY (above person height)
                        'roi_x_start': getattr(settings, 'train_state_roi_x_start', 0.37),
                        'roi_x_end': getattr(settings, 'train_state_roi_x_end', 0.52),
                        'roi_y_start': getattr(settings, 'train_state_roi_y_start', 0.0),
                        'roi_y_end': getattr(settings, 'train_state_roi_y_end', 0.15),
                        'adaptive_roi': getattr(settings, 'train_state_adaptive_roi', False),
                        'debug_frames': getattr(settings, 'train_state_debug_frames', False),
                        'debug_dir': getattr(settings, 'train_state_debug_dir', 'train_state_debug'),
                        'debug_interval': getattr(settings, 'train_state_debug_interval', 1),
                    }
                    self.train_state_detector = TrainStateDetector(train_state_config)
                    self.exempt_activities = getattr(
                        settings, 'stopped_state_exempt_activities',
                        ['cell_phone', 'writing', 'packing_bags', 'mind_diversion']
                    )
                    self.logger.info(
                        f"TrainStateDetector initialized - exempt activities: {self.exempt_activities}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize TrainStateDetector: {e}")
                    self.train_state_detector = None
                    self.exempt_activities = []
            else:
                self.train_state_detector = None
                self.exempt_activities = []
                self.logger.info("Train state detection disabled by configuration")
        else:
            self.train_state_detector = None
            self.exempt_activities = []
            if TrainStateDetector is None:
                self.logger.info("TrainStateDetector not available - stopped train exemption disabled")

    def get_keypoint(self, landmarks, keypoint_name):
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

    def update_per_person_detection(self, person_idx, activity_type, detected, timestamp_sec):
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
        # Initialize tracking for this person
        if person_idx not in self.per_person_consecutive_detections:
            self.per_person_consecutive_detections[person_idx] = {
                'cell_phone': 0, 'writing': 0, 'packing_bags': 0
            }
            self.per_person_grace_counters[person_idx] = {
                'cell_phone': 0, 'writing': 0, 'packing_bags': 0
            }

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
            nose = self.get_keypoint(landmarks, 'nose')
            left_shoulder = self.get_keypoint(landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(landmarks, 'right_shoulder')

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
            key_landmark_names = [
                'nose', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
            ]

            total_movement = 0.0
            for landmark_name in key_landmark_names:
                curr = self.get_keypoint(current_landmarks, landmark_name)
                prev = self.get_keypoint(previous_landmarks, landmark_name)

                # Calculate Euclidean distance
                distance = np.sqrt(
                    (curr.x - prev.x) ** 2 +
                    (curr.y - prev.y) ** 2
                )
                total_movement += distance
            
            # Normalize by number of landmarks
            movement_score = total_movement / len(key_landmark_names)
            
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
        
        # Validate pose landmarks before using
        if not self.validate_pose_landmarks(pose_landmarks):
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
                self.logger.debug(f"[Pose-Based Sleep] Started tracking - head_tilt={avg_head_tilt:.1f}°, movement={avg_movement:.4f}")
            
            self.pose_sleep_duration = timestamp_sec - self.pose_sleep_start
            
            # Check thresholds
            is_sleeping = self.pose_sleep_duration >= 30  # 30 seconds
            is_microsleeping = self.pose_sleep_duration >= 5 and not is_sleeping  # 5 seconds
            
            debug_info['pose_sleep_duration'] = self.pose_sleep_duration
            
            return is_sleeping, is_microsleeping, debug_info
        else:
            # Reset if conditions not met
            if self.pose_sleep_start is not None:
                self.logger.debug("[Pose-Based Sleep] Stopped - indicators not met")
            self.pose_sleep_start = None
            self.pose_sleep_duration = 0
            
            return False, False, debug_info
    
    def calculate_wrist_distance(self, pose_landmarks, frame_shape):
        """Calculate Euclidean distance between left and right wrists.
        Falls back to elbow distance if wrists are not visible.

        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: Tuple of (height, width) of the frame

        Returns:
            tuple: (distance in pixels, source) where source is 'wrist' or 'elbow'
                   or (None, None) if neither detectable
        """
        if not pose_landmarks:
            return None, None

        # Validate pose landmarks before using
        if not self.validate_pose_landmarks(pose_landmarks):
            return None, None

        try:
            landmarks = pose_landmarks.landmark
            h, w = frame_shape[:2]

            # Get wrist landmarks
            right_wrist = self.get_keypoint(landmarks, 'right_wrist')
            left_wrist = self.get_keypoint(landmarks, 'left_wrist')

            # Try wrists first (primary method)
            if right_wrist.visibility >= 0.5 and left_wrist.visibility >= 0.5:
                # Convert normalized coordinates to pixel coordinates
                right_wrist_px = (right_wrist.x * w, right_wrist.y * h)
                left_wrist_px = (left_wrist.x * w, left_wrist.y * h)

                # Calculate Euclidean distance
                distance = np.sqrt(
                    (right_wrist_px[0] - left_wrist_px[0])**2 +
                    (right_wrist_px[1] - left_wrist_px[1])**2
                )
                return distance, 'wrist'

            # FALLBACK: Use elbows if wrists not visible
            # Elbows are typically more visible from overhead camera angles
            right_elbow = self.get_keypoint(landmarks, 'right_elbow')
            left_elbow = self.get_keypoint(landmarks, 'left_elbow')

            ELBOW_VISIBILITY_THRESHOLD = 0.4  # Lower threshold since elbows more reliable
            if right_elbow.visibility >= ELBOW_VISIBILITY_THRESHOLD and left_elbow.visibility >= ELBOW_VISIBILITY_THRESHOLD:
                right_elbow_px = (right_elbow.x * w, right_elbow.y * h)
                left_elbow_px = (left_elbow.x * w, left_elbow.y * h)
                distance = np.sqrt(
                    (right_elbow_px[0] - left_elbow_px[0])**2 +
                    (right_elbow_px[1] - left_elbow_px[1])**2
                )
                # Return elbow distance (elbows are typically wider apart)
                return distance, 'elbow'

            return None, None
        except Exception as e:
            return None, None

    def detect_writing_posture(self, pose_landmarks, frame_shape):
        """Instantly detect writing posture based on hand position.

        Checks if hands are in typical writing position using multiple criteria:
        1. Hands below shoulders (relaxed check for camera angles)
        2. Hands in lap area (strict check)
        3. Head looking down (indicates reading/writing posture)

        Args:
            pose_landmarks: YoloPoseLandmarks or MediaPipe pose landmarks
            frame_shape: Tuple of (height, width) of the frame

        Returns:
            bool: True if writing posture detected, False otherwise
        """
        if not pose_landmarks:
            return False

        # Validate pose landmarks before using
        if not self.validate_pose_landmarks(pose_landmarks):
            return False

        try:
            h, w = frame_shape[:2]

            # Get key body points
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_hip = self.get_keypoint(pose_landmarks, 'left_hip')
            right_hip = self.get_keypoint(pose_landmarks, 'right_hip')
            left_wrist = self.get_keypoint(pose_landmarks, 'left_wrist')
            right_wrist = self.get_keypoint(pose_landmarks, 'right_wrist')

            # Calculate vertical positions (normalized 0-1)
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_y = (left_hip.y + right_hip.y) / 2
            left_wrist_y = left_wrist.y
            right_wrist_y = right_wrist.y

            # Calculate wrist positions relative to body
            # Option 1: Lap area (strict) - wrists below hips
            left_in_lap = left_wrist_y > hip_y
            right_in_lap = right_wrist_y > hip_y

            # Option 2: Below shoulders (relaxed) - better for various camera angles
            left_below_shoulders = left_wrist_y > shoulder_y
            right_below_shoulders = right_wrist_y > shoulder_y

            # Calculate wrist distance (in pixels)
            left_wrist_x = int(left_wrist.x * w)
            left_wrist_y_px = int(left_wrist.y * h)
            right_wrist_x = int(right_wrist.x * w)
            right_wrist_y_px = int(right_wrist.y * h)

            wrist_distance = ((left_wrist_x - right_wrist_x) ** 2 +
                            (left_wrist_y_px - right_wrist_y_px) ** 2) ** 0.5

            # Check if head is looking down (indicates reading/writing)
            head_looking_down = self.detect_head_looking_down(pose_landmarks)

            # Writing posture criteria (RELAXED):
            # 1. Wrists close together (increased from 200 to 300 pixels)
            # 2. Either: hands in lap OR hands below shoulders
            # 3. Bonus: head looking down strengthens detection
            WRITING_WRIST_DISTANCE = 300  # pixels - increased from 200

            # Hands in writing position (either strict lap OR relaxed below-shoulders)
            hands_in_lap = left_in_lap and right_in_lap
            hands_below_shoulders = left_below_shoulders and right_below_shoulders
            hands_in_writing_position = hands_in_lap or hands_below_shoulders

            # Detect if wrists are close enough
            wrists_close = wrist_distance <= WRITING_WRIST_DISTANCE

            # Writing detected if:
            # - Hands below shoulders + wrists close, OR
            # - Head looking down + hands below shoulders (even with wider wrist distance)
            if hands_in_writing_position and wrists_close:
                return True

            # Extra: Head looking down with hands below shoulders (wider tolerance)
            RELAXED_WRIST_DISTANCE = 400  # even more relaxed when head is down
            if head_looking_down and hands_below_shoulders and wrist_distance <= RELAXED_WRIST_DISTANCE:
                return True

            return False

        except Exception as e:
            return False

    def detect_head_looking_down(self, pose_landmarks):
        """Check if head is tilted down (looking at lap area).

        Uses nose position relative to eyes to detect downward head tilt,
        which indicates reading/writing posture.

        Args:
            pose_landmarks: YoloPoseLandmarks or MediaPipe pose landmarks

        Returns:
            bool: True if head is looking down, False otherwise
        """
        try:
            nose = self.get_keypoint(pose_landmarks, 'nose')
            left_eye = self.get_keypoint(pose_landmarks, 'left_eye')
            right_eye = self.get_keypoint(pose_landmarks, 'right_eye')

            if nose is None or left_eye is None or right_eye is None:
                return False

            # Calculate average eye Y position
            eye_y = (left_eye.y + right_eye.y) / 2

            # Head is looking down when nose is significantly below eye line
            # Using normalized coordinates (0-1), so 0.01 = ~1% of frame height
            # REDUCED from 0.02 to 0.01 to better capture slight head tilt while reading/writing
            HEAD_DOWN_THRESHOLD = 0.01
            return nose.y > eye_y + HEAD_DOWN_THRESHOLD

        except Exception as e:
            return False

    def check_hands_below_shoulders(self, pose_landmarks):
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
            return False

    def detect_writing_by_wrist_proximity(self, pose_landmarks, frame_shape, person_idx, timestamp_sec):
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
        distance_result = self.calculate_wrist_distance(pose_landmarks, frame_shape)

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
        MAX_WRIST_DISTANCE = 300  # pixels - wrists close together during writing
        MAX_ELBOW_DISTANCE = 450  # pixels - elbows typically wider apart during writing
        MIN_DURATION = 1.0  # seconds - for faster detection
        REQUIRED_CONSECUTIVE = 2  # frames @ 0.5fps = 4 seconds total

        # Select threshold based on detection source
        max_distance = MAX_WRIST_DISTANCE if source == 'wrist' else MAX_ELBOW_DISTANCE

        person_tracking = self.wrist_proximity_tracking[person_idx]

        # Check if distance is within threshold
        if distance <= max_distance:
            # NEW: Check if head is looking down (required for writing posture)
            head_looking_down = self.detect_head_looking_down(pose_landmarks)

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
            if (person_tracking['consecutive_frames'] >= REQUIRED_CONSECUTIVE and
                person_tracking['duration'] >= MIN_DURATION):
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

    def detect_writing_by_book_and_posture(self, pose_landmarks, person_bbox, book_bboxes, person_idx, timestamp_sec):
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
        person_book_margin = 250  # Same margin used elsewhere
        book_in_region = False
        for book_bbox in book_bboxes:
            if self.bbox_overlap_with_margin(book_bbox, person_bbox, person_book_margin):
                book_in_region = True
                break

        if not book_in_region:
            person_tracking['start_time'] = None
            person_tracking['duration'] = 0.0
            person_tracking['consecutive_frames'] = 0
            return False

        # Check head posture (must be looking down toward book)
        head_looking_down = self.detect_head_looking_down(pose_landmarks)

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
        MIN_DURATION = 2.0  # Require longer duration for this fallback (more confidence needed)
        REQUIRED_CONSECUTIVE = 2

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

        if (person_tracking['consecutive_frames'] >= REQUIRED_CONSECUTIVE and
            person_tracking['duration'] >= MIN_DURATION):
            self.logger.info(
                f"Person {person_idx}: WRITING CONFIRMED via book+posture fallback - "
                f"book in region + head down for {person_tracking['duration']:.1f}s"
            )
            return True

        return False

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

    def detect_standing_posture(self, pose_landmarks, frame_shape):
        """
        Detect if a person is standing based on YOLO-Pose keypoints.

        Uses hip-knee-ankle keypoint alignment to determine posture:
        - Standing: Legs mostly straight (hip-knee-ankle vertically aligned)
        - Sitting: Knees bent significantly (knee forward of hip-ankle line)

        YOLO-Pose keypoint indices:
        - Left Hip: 11, Right Hip: 12
        - Left Knee: 13, Right Knee: 14
        - Left Ankle: 15, Right Ankle: 16

        Args:
            pose_landmarks: YOLO-Pose landmarks for a person
            frame_shape: (height, width) of frame

        Returns:
            dict: {
                'is_standing': bool,
                'confidence': float (0-1),
                'left_leg_ratio': float or None,
                'right_leg_ratio': float or None,
                'visible': bool  # True if enough keypoints visible
            }
        """
        result = {
            'is_standing': False,
            'confidence': 0.0,
            'left_leg_ratio': None,
            'right_leg_ratio': None,
            'visible': False
        }

        if pose_landmarks is None:
            return result

        h, w = frame_shape[:2]
        min_visibility = 0.3  # Minimum visibility threshold for keypoints

        # Get keypoints using the adapter
        left_hip = self._get_keypoint_by_name(pose_landmarks, 'left_hip')
        right_hip = self._get_keypoint_by_name(pose_landmarks, 'right_hip')
        left_knee = self._get_keypoint_by_name(pose_landmarks, 'left_knee')
        right_knee = self._get_keypoint_by_name(pose_landmarks, 'right_knee')
        left_ankle = self._get_keypoint_by_name(pose_landmarks, 'left_ankle')
        right_ankle = self._get_keypoint_by_name(pose_landmarks, 'right_ankle')

        def get_visibility(kp):
            """Get visibility score for keypoint."""
            if kp is None:
                return 0.0
            # YOLO-Pose keypoints: [x, y, conf]
            if len(kp) >= 3:
                return kp[2]
            return 0.0

        def get_coords(kp):
            """Get (x, y) pixel coordinates from keypoint."""
            if kp is None:
                return None
            x = kp[0] * w if kp[0] <= 1 else kp[0]
            y = kp[1] * h if kp[1] <= 1 else kp[1]
            return (x, y)

        # Check left leg visibility
        left_visible = (
            get_visibility(left_hip) >= min_visibility and
            get_visibility(left_knee) >= min_visibility and
            get_visibility(left_ankle) >= min_visibility
        )

        # Check right leg visibility
        right_visible = (
            get_visibility(right_hip) >= min_visibility and
            get_visibility(right_knee) >= min_visibility and
            get_visibility(right_ankle) >= min_visibility
        )

        # Need at least one leg visible
        if not left_visible and not right_visible:
            return result

        result['visible'] = True

        def calculate_leg_verticality(hip, knee, ankle):
            """
            Calculate how vertical a leg is (standing vs sitting).

            Standing: knee is on the line between hip and ankle (ratio near 0)
            Sitting: knee is forward/bent (high ratio)

            Returns ratio of knee forward displacement to leg height.
            """
            hip_coords = get_coords(hip)
            knee_coords = get_coords(knee)
            ankle_coords = get_coords(ankle)

            if not all([hip_coords, knee_coords, ankle_coords]):
                return None

            # Calculate vertical distance (hip to ankle)
            leg_height = abs(ankle_coords[1] - hip_coords[1])
            if leg_height < 50:  # Too small to measure reliably
                return None

            # Calculate how far forward the knee is from the hip-ankle line
            # For a standing person, knee should be roughly on the line
            # For sitting, knee will be forward

            # Use simple vertical alignment check
            hip_x, hip_y = hip_coords
            knee_x, knee_y = knee_coords
            ankle_x, ankle_y = ankle_coords

            # Expected knee x if standing (linear interpolation)
            t = (knee_y - hip_y) / leg_height if leg_height > 0 else 0.5
            expected_knee_x = hip_x + t * (ankle_x - hip_x)

            # Horizontal deviation of knee from expected position
            knee_deviation = abs(knee_x - expected_knee_x)

            # Ratio of deviation to leg height (standing = low, sitting = high)
            verticality_ratio = knee_deviation / leg_height

            return verticality_ratio

        left_ratio = None
        right_ratio = None

        if left_visible:
            left_ratio = calculate_leg_verticality(left_hip, left_knee, left_ankle)
            result['left_leg_ratio'] = left_ratio

        if right_visible:
            right_ratio = calculate_leg_verticality(right_hip, right_knee, right_ankle)
            result['right_leg_ratio'] = right_ratio

        # Determine standing based on leg verticality
        # Standing threshold: ratio < 0.15 (knee close to hip-ankle line)
        # Sitting threshold: ratio > 0.25 (knee significantly forward)
        standing_threshold = 0.15

        ratios = [r for r in [left_ratio, right_ratio] if r is not None]

        if not ratios:
            return result

        # Person is standing if at least one leg shows standing posture
        min_ratio = min(ratios)
        result['is_standing'] = min_ratio < standing_threshold

        # Confidence based on how clearly standing vs sitting
        # Higher confidence when ratio is further from threshold
        if result['is_standing']:
            result['confidence'] = max(0.0, min(1.0, (standing_threshold - min_ratio) / standing_threshold))
        else:
            result['confidence'] = max(0.0, min(1.0, (min_ratio - standing_threshold) / 0.3))

        return result

    def check_alp_standing_before_stop(
        self,
        train_state,
        train_state_changed: bool,
        timestamp_sec: float,
        persons_data: dict,
        person_roles: dict,
        frame_shape: tuple
    ) -> bool:
        """
        Check if ALP stood up in time before train stopped.

        Logic:
        1. When APPROACHING_STOP detected -> start monitoring ALP pose
        2. If ALP stands -> record timestamp
        3. If ALP not visible -> skip check (avoid false positives)
        4. When STOPPED detected -> check if ALP stood >= 30s before stop
        5. If ALP was visible but didn't stand in time -> flag violation
        6. When MOVING detected -> reset tracking

        Args:
            train_state: Current TrainState enum value
            train_state_changed: Whether state changed this frame
            timestamp_sec: Current timestamp in seconds
            persons_data: Multi-person results from process_all_persons_activities
            person_roles: Dict mapping person_idx to role info
            frame_shape: (height, width) of frame

        Returns:
            bool: True if violation should be flagged (ALP didn't stand in time)
        """
        from app.models.activity_models import TrainStateEnum as TrainState

        violation_detected = False

        # Handle state transitions
        if train_state_changed:
            if train_state == TrainState.APPROACHING_STOP:
                # Start monitoring ALP standing
                self.alp_standing_tracking['monitoring_active'] = True
                self.alp_standing_tracking['approaching_start_time'] = timestamp_sec
                self.alp_standing_tracking['alp_stood_time'] = None
                self.alp_standing_tracking['alp_visible'] = False
                self.alp_standing_tracking['violation_flagged'] = False
                self.logger.info(
                    f"[{timestamp_sec:.2f}s] ALP standing monitoring STARTED (approaching stop)"
                )

            elif train_state == TrainState.STOPPED:
                # Train stopped - check if ALP stood in time
                if self.alp_standing_tracking['monitoring_active']:
                    stop_time = timestamp_sec
                    alp_stood_time = self.alp_standing_tracking['alp_stood_time']
                    alp_visible = self.alp_standing_tracking['alp_visible']

                    if alp_visible and not self.alp_standing_tracking['violation_flagged']:
                        if alp_stood_time is not None:
                            # ALP stood - check if it was >= 30s before stop
                            time_before_stop = stop_time - alp_stood_time
                            if time_before_stop < self.alp_standing_required_seconds:
                                # ALP stood too late
                                violation_detected = True
                                self.alp_standing_tracking['violation_flagged'] = True
                                self.logger.warning(
                                    f"[{timestamp_sec:.2f}s] ALP VIOLATION: Stood only {time_before_stop:.1f}s "
                                    f"before stop (required: {self.alp_standing_required_seconds}s)"
                                )
                            else:
                                self.logger.info(
                                    f"[{timestamp_sec:.2f}s] ALP stood {time_before_stop:.1f}s before stop - OK"
                                )
                        else:
                            # ALP was visible but never stood
                            violation_detected = True
                            self.alp_standing_tracking['violation_flagged'] = True
                            self.logger.warning(
                                f"[{timestamp_sec:.2f}s] ALP VIOLATION: Never stood before stop"
                            )
                    elif not alp_visible:
                        # ALP not visible - skip check (no violation)
                        self.logger.info(
                            f"[{timestamp_sec:.2f}s] ALP not visible during approach - skipping standing check"
                        )

                # Keep monitoring active in case train starts moving again
                self.alp_standing_tracking['monitoring_active'] = False

            elif train_state == TrainState.MOVING:
                # Train moving again - reset tracking
                self.alp_standing_tracking['monitoring_active'] = False
                self.alp_standing_tracking['approaching_start_time'] = None
                self.alp_standing_tracking['alp_stood_time'] = None
                self.alp_standing_tracking['alp_visible'] = False
                self.alp_standing_tracking['violation_flagged'] = False

        # During APPROACHING_STOP, monitor ALP posture
        if self.alp_standing_tracking['monitoring_active']:
            # Find ALP in person_roles
            alp_person_idx = None
            for person_idx, role_info in person_roles.items():
                if role_info.get('role') == 'ALP':
                    alp_person_idx = person_idx
                    break

            if alp_person_idx is not None and alp_person_idx in persons_data:
                person_data = persons_data[alp_person_idx]
                pose_landmarks = person_data.get('pose_landmarks')

                if pose_landmarks is not None:
                    self.alp_standing_tracking['alp_visible'] = True

                    # Check if ALP is standing
                    standing_result = self.detect_standing_posture(pose_landmarks, frame_shape)

                    if standing_result['is_standing'] and standing_result['confidence'] > 0.5:
                        # ALP is standing - record first time
                        if self.alp_standing_tracking['alp_stood_time'] is None:
                            self.alp_standing_tracking['alp_stood_time'] = timestamp_sec
                            self.logger.info(
                                f"[{timestamp_sec:.2f}s] ALP stood up (confidence: {standing_result['confidence']:.2f})"
                            )
                else:
                    # Pose not detected for this person
                    pass
            else:
                # ALP not identified in frame - don't mark as visible
                pass

        return violation_detected

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
        
        # Run YOLO on ROI with strict confidence threshold to minimize false positives
        # Increased from 0.01 → 0.15 → 0.25 → 0.38 → 0.45 (configurable)
        results = self.yolo_model(roi_frame, verbose=False, conf=self.cell_phone_confidence,
                                  imgsz=self.yolo_imgsz, device=self.yolo_device)
        
        detections = []
        debug_all_detections = []  # Track all detections for debugging
        
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
        
        # DEBUG LOGGING: Log what YOLO detected (especially for cell phone debugging)
        # Use logging module instead of print for multiprocessing compatibility
        import logging
        debug_logger = logging.getLogger('locopilot_monitor')
        
        if 'cell phone' in target_classes:
            if len(debug_all_detections) > 0:
                cell_phones = [d for d in debug_all_detections if d[0] == 'cell phone']
                if cell_phones:
                    debug_logger.info(f"[DEBUG ROI] ✓ Found {len(cell_phones)} cell phone(s): {cell_phones}")
                else:
                    # Log what was detected instead of phone (top 5 by confidence)
                    top_detections = sorted(debug_all_detections, key=lambda x: -x[1])[:5]
                    debug_logger.info(f"[DEBUG ROI] ✗ No phone, but YOLO found {len(debug_all_detections)} objects: {top_detections}")
            else:
                # YOLO detected absolutely nothing in this ROI
                debug_logger.info(f"[DEBUG ROI] ⚠ YOLO detected NOTHING in this ROI (empty detection)")
        
        return detections

    def detect_objects_in_rois_batch(self, frame, roi_bboxes, roi_names, target_classes=['cell phone', 'book', 'pen', 'pencil']):
        """Run YOLO detection on multiple ROI regions in a single batched call.

        PERFORMANCE OPTIMIZATION: This method processes all ROIs in a single YOLO
        inference call instead of N sequential calls, achieving ~4x speedup for
        ROI processing (1200ms → 300ms for 8 ROIs per person).

        Args:
            frame: Full frame
            roi_bboxes: List of (x1, y1, x2, y2) ROI bounding boxes
            roi_names: List of ROI names corresponding to each bbox (for debugging)
            target_classes: List of class names to detect in ROIs

        Returns:
            List of lists: [[detections for ROI 1], [detections for ROI 2], ...]
            Each detection: (class_name, conf, x1, y1, x2, y2) with global coordinates
        """
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
        # This is the KEY optimization: 1 call instead of N calls
        batch_results = self.yolo_model(roi_frames, verbose=False, conf=self.cell_phone_confidence,
                                         imgsz=self.yolo_imgsz, device=self.yolo_device)

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

            # DEBUG LOGGING (same as original method)
            import logging
            debug_logger = logging.getLogger('locopilot_monitor')

            if 'cell phone' in target_classes:
                roi_name = roi_names[roi_bbox_idx]
                if len(debug_all_detections) > 0:
                    cell_phones = [d for d in debug_all_detections if d[0] == 'cell phone']
                    if cell_phones:
                        debug_logger.info(f"[DEBUG ROI BATCH] {roi_name}: ✓ Found {len(cell_phones)} cell phone(s): {cell_phones}")
                    else:
                        top_detections = sorted(debug_all_detections, key=lambda x: -x[1])[:5]
                        debug_logger.info(f"[DEBUG ROI BATCH] {roi_name}: ✗ No phone, found {len(debug_all_detections)} objects: {top_detections}")
                else:
                    debug_logger.info(f"[DEBUG ROI BATCH] {roi_name}: ⚠ YOLO detected NOTHING")

            all_detections[roi_bbox_idx] = detections

        return all_detections

    def validate_object_aspect_ratio(self, bbox, object_class):
        """
        Validate detected object based on aspect ratio to filter false positives.

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

    def detect_objects(self, frame, pose_landmarks=None, use_pose_guided=True):
        """Detect objects using YOLO with pose-guided detection.
        
        MULTI-LAYERED DETECTION FLOW:
        1. Full frame detection for:
           - Person (for counting)
           - Backpack (for packing detection)
           - Book (low confidence, only if near person with aspect ratio validation)
        2. ROI-based detection around landmarks for activity objects:
           - Hands (wrists, index) - 180px radius (OPTIMIZED for YOLOv8m)
           - Lap/Torso (hips) - 180px radius (OPTIMIZED for YOLOv8m)
           - Ears, mouth - 180px radius (for phone/eating detection)
        3. Aspect ratio validation filters false positives (phones, books)
        4. This provides comprehensive detection while minimizing false positives
        
        Args:
            frame: Input frame
            pose_landmarks: MediaPipe pose landmarks (optional)
            use_pose_guided: Enable pose-guided ROI detection (default True)
            
        Returns:
            Dictionary with detections and ROI information
        """
        # Stage 1: Full frame detection for person, backpack, and books near person
        results = self.yolo_model(frame, verbose=False, imgsz=self.yolo_imgsz, device=self.yolo_device)

        # Phase 3: Cache results for potential reuse (avoid redundant inference)
        # Store results with timestamp for cache validation (100ms TTL)
        self._cached_frame_objects = results
        self._cached_frame_time = time_module.time()

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
                # Detect person and bag types (backpack, handbag, suitcase) from full frame
                if class_name == 'person' and conf > 0.5:
                    detections['person'].append(xyxy)
                    person_boxes.append(xyxy)
                elif class_name in ['backpack', 'handbag', 'suitcase']:
                    # Log all bag detections for debugging
                    if conf > 0.25:
                        self.logger.debug(f"BAG DETECTED: {class_name} conf={conf:.2f} bbox={xyxy}")
                    if conf > 0.45:  # LOWERED from 0.75 to catch more bags (aspect ratio filter handles false positives)
                        # Validate aspect ratio and size to filter out seats/equipment
                        bag_width = xyxy[2] - xyxy[0]
                        bag_height = xyxy[3] - xyxy[1]
                        aspect_ratio = bag_width / bag_height if bag_height > 0 else 999
                        bag_area = bag_width * bag_height

                        # Filter: backpacks are typically taller than wide (aspect ratio < 1.2)
                        # Train seats are wide (aspect ratio > 1.2)
                        # Also filter very large detections (> 100000 px area = ~316x316)
                        # And filter very small detections (< 5000 px area = ~70x70)
                        if aspect_ratio < 1.2 and 5000 < bag_area < 100000:
                            detections['backpack'].append(xyxy)
                            self.logger.info(f"BAG ADDED: {class_name} conf={conf:.2f} aspect={aspect_ratio:.2f} area={bag_area:.0f}")
                        else:
                            self.logger.debug(f"BAG REJECTED: {class_name} conf={conf:.2f} aspect={aspect_ratio:.2f} area={bag_area:.0f} (filtered)")
                # OPTION 3: Re-enable book detection in full frame with moderate confidence
                # But only if book is within reasonable distance of a person
                elif class_name == 'book' and conf > 0.4:  # Increased to 0.4 to reduce false positives
                    # Check if book is near any detected person
                    if len(person_boxes) > 0:
                        book_near_person = False
                        book_center_x = (xyxy[0] + xyxy[2]) / 2
                        book_center_y = (xyxy[1] + xyxy[3]) / 2
                        
                        for person_box in person_boxes:
                            # Check if book center is within expanded person bounding box
                            person_x1, person_y1, person_x2, person_y2 = person_box
                            margin = 150  # Reduced from 200px to 150px for stricter book-to-person association
                            if (person_x1 - margin <= book_center_x <= person_x2 + margin and
                                person_y1 - margin <= book_center_y <= person_y2 + margin):
                                book_near_person = True
                                break
                        
                        if book_near_person and self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)
                    else:
                        # No person detected, add book anyway (fallback) if aspect ratio is valid
                        if self.validate_object_aspect_ratio(xyxy, 'book'):
                            detections['book'].append(xyxy)
        
        # Stage 2: Pose-guided ROI detection (if pose landmarks available)
        if use_pose_guided and pose_landmarks is not None:
            h, w = frame.shape[:2]

            # Define keypoints of interest with ROI sizes
            # OPTIMIZED CONFIGURATION: Uniform 180px ROI for YOLOv8m
            # Format: (display_name, keypoint_name_for_lookup, roi_size)
            # Note: RIGHT_INDEX/LEFT_INDEX map to wrist in YOLO (no finger keypoints)
            keypoints_of_interest = [
                # Hands (for phone, book, pen, pencil) - FOCUSED ON HANDS ONLY
                ('RIGHT_WRIST', 'right_wrist', 180),  # Optimized for YOLOv8m
                ('LEFT_WRIST', 'left_wrist', 180),    # Optimized for YOLOv8m
                ('RIGHT_INDEX', 'right_wrist', 180),  # Maps to wrist (YOLO has no finger keypoints)
                ('LEFT_INDEX', 'left_wrist', 180),    # Maps to wrist (YOLO has no finger keypoints)

                # Lap/Torso area (for books, reading, writing on lap) - REDUCED SIZE
                ('RIGHT_HIP', 'right_hip', 180),  # Optimized for YOLOv8m
                ('LEFT_HIP', 'left_hip', 180),    # Optimized for YOLOv8m

                # Ears (for phone calls) - REDUCED SIZE
                ('RIGHT_EAR', 'right_ear', 180),  # Reduced from 240px to 180px
                ('LEFT_EAR', 'left_ear', 180),    # Reduced from 240px to 180px

                # REMOVED: Shoulders, Mouth, Nose ROIs (causing too many false positives)
            ]

            # OPTIMIZATION: Collect all ROIs first instead of processing sequentially
            # This enables batch YOLO inference for massive performance improvement
            roi_bboxes = []
            roi_names = []

            import logging
            import time
            debug_logger = logging.getLogger('locopilot_monitor')

            for display_name, keypoint_name, roi_size in keypoints_of_interest:
                try:
                    landmark = self.get_keypoint(pose_landmarks, keypoint_name)

                    # Check visibility
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

                        # DEBUG: Log ROI creation for key body parts
                        if display_name in ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP', 'NOSE']:
                            debug_logger.info(f"[DEBUG ROI] Creating {display_name} ROI: size={roi_size}px, coords={keypoint_coords}")

                except Exception as e:
                    roi_bboxes.append(None)
                    roi_names.append(display_name)
                    continue

            # OPTIMIZATION: Batch process all ROIs in single YOLO call
            valid_roi_count = sum(1 for bbox in roi_bboxes if bbox is not None)

            if valid_roi_count > 0:
                target_classes = ['cell phone', 'book', 'pen', 'pencil', 'paper', 'bottle', 'cup']
                batch_detections = self.detect_objects_in_rois_batch(frame, roi_bboxes, roi_names, target_classes)

                # Process batch results (same logic as before, but from batch)
                for idx, (keypoint_name, roi_detections) in enumerate(zip(roi_names, batch_detections)):
                    for det in roi_detections:
                        class_name, conf, x1, y1, x2, y2 = det
                        detections['roi_detections'].append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'keypoint': keypoint_name,
                            'source': 'pose_guided_roi_batch'  # Updated source indicator
                        })

                        # FILTER: Only add cell phones from HAND/WRIST/EAR ROIs (not hips)
                        # This reduces false positives from phones detected near lap/seat areas
                        hand_related_keypoints = ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_EAR', 'LEFT_EAR']

                        if class_name == 'cell phone':
                            # Only add if detected near hands/ears (actual phone usage)
                            if keypoint_name in hand_related_keypoints:
                                detections['cell_phone'].append([x1, y1, x2, y2])
                        elif class_name == 'book':
                            # Books can be detected from all ROIs (hands, lap, etc.)
                            detections['book'].append([x1, y1, x2, y2])
        
        return detections

    # =========================================================================
    # BATCH INFERENCE METHODS - GPU OPTIMIZATION
    # =========================================================================
    # These methods process multiple frames at once to maximize GPU utilization.
    # Instead of running inference N times (once per frame), we run it once on
    # a batch of N frames, keeping the GPU busy and reducing overhead.
    # =========================================================================

    def detect_objects_batch(self, frames, batch_size=8):
        """Run YOLO object detection on multiple frames in a single batch.

        This maximizes GPU utilization by processing multiple frames at once
        instead of one at a time. The GPU stays busy with larger batches.

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
                # Run batch inference - YOLO handles list of frames efficiently
                batch_results = self.yolo_model(
                    batch_frames,
                    verbose=False,
                    imgsz=self.yolo_imgsz,
                    device=self.yolo_device
                )
            except Exception as e:
                self.logger.error(f"[GPU BATCH] Object detection failed for batch starting at {batch_start}: {e}")
                # Fallback: return empty detections for this batch
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
                pending_books = []  # Store books to check after all persons are found

                # Process detections from this frame's results
                if results.boxes is not None:
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()

                        class_name = self.yolo_model.names[cls]

                        if class_name == 'person' and conf > 0.5:
                            detections['person'].append(xyxy)
                            person_boxes.append(xyxy)
                        elif class_name in ['backpack', 'handbag', 'suitcase']:
                            if conf > 0.45:
                                bag_width = xyxy[2] - xyxy[0]
                                bag_height = xyxy[3] - xyxy[1]
                                aspect_ratio = bag_width / bag_height if bag_height > 0 else 999
                                bag_area = bag_width * bag_height

                                if aspect_ratio < 1.2 and 5000 < bag_area < 100000:
                                    detections['backpack'].append(xyxy)
                        elif class_name == 'book' and conf > 0.4:
                            # Store book for later processing (after all persons found)
                            pending_books.append(xyxy)
                        elif class_name == 'cell phone' and conf > 0.3:
                            detections['cell_phone'].append(xyxy)

                # Process pending books - check proximity to persons
                for book_xyxy in pending_books:
                    if len(person_boxes) > 0:
                        for person_box in person_boxes:
                            if self._boxes_overlap_or_near(book_xyxy, person_box, margin=200):
                                detections['book'].append(book_xyxy)
                                break
                    else:
                        # Fallback: add book anyway if no persons detected
                        detections['book'].append(book_xyxy)

                all_detections.append(detections)

        self.logger.debug(f"[GPU BATCH] detect_objects_batch complete: {len(all_detections)} results")
        return all_detections

    def detect_poses_batch(self, frames, batch_size=8):
        """Run YOLO pose detection on multiple frames in a single batch.

        This maximizes GPU utilization by processing multiple frames at once.

        Args:
            frames: List of BGR frames (numpy arrays)
            batch_size: Maximum batch size for inference (default 8)

        Returns:
            List of pose result dictionaries, one per frame.
            Format matches self.yolo_pose.process() output:
            {person_idx: {'bbox': [...], 'bbox_confidence': float, 'keypoints': YoloPoseLandmarks}}
        """
        # Import at method level (not inside loop)
        from app.services.yolo_pose_adapter import YoloPoseLandmarks, PersonKeypoints

        if not frames:
            return []

        self.logger.debug(f"[GPU BATCH] detect_poses_batch: {len(frames)} frames, batch_size={batch_size}")
        all_poses = []

        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            try:
                # Run batch inference on pose model with device parameter
                batch_results = self.yolo_pose.model(
                    batch_frames,
                    verbose=False,
                    conf=self.yolo_pose.conf_threshold,
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

    def _boxes_overlap_or_near(self, box1, box2, margin=100):
        """Check if two boxes overlap or are within margin pixels of each other."""
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

    # =========================================================================
    # END BATCH INFERENCE METHODS
    # =========================================================================

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
    
    def draw_mediapipe_outputs(self, frame, pose_results, face_results, ear_value=None, eye_closure_duration=0, pose_sleep_info=None, head_pose_info=None):
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
    
    def draw_multi_person_mediapipe_outputs(self, frame, persons_data, face_results, ear_value=None, eye_closure_duration=0):
        """Draw MediaPipe pose landmarks for ALL detected persons
        
        Args:
            frame: The frame image
            persons_data: Dictionary of person data from process_all_persons_activities()
                         Format: {person_idx: {'pose_landmarks': landmarks, 'role': 'LP', 'activities': {...}, ...}}
            face_results: MediaPipe face mesh results
            ear_value: Eye aspect ratio value
            eye_closure_duration: Duration of eye closure
            
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
                    except:
                        continue

                # Draw keypoints
                for i in range(17):
                    try:
                        landmark = pose_landmarks.landmark[i]
                        if landmark.visibility > 0.5:
                            pt = (int(landmark.x * w), int(landmark.y * h))
                            cv2.circle(annotated_frame, pt, 8, (0, 255, 0), -1)
                    except:
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
                    except:
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
                except:
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
        
        # Draw status text for face/eye detection
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
                
                if eye_closure_duration > 0:
                    duration_text = f"Closed Duration: {eye_closure_duration:.1f}s"
                    duration_color = (0, 165, 255)
                    
                    if eye_closure_duration >= 30:
                        duration_text += " - SLEEP ALERT!"
                        duration_color = (0, 0, 255)
                    elif eye_closure_duration >= 5:
                        duration_text += " - MICROSLEEP!"
                        duration_color = (0, 140, 255)
                    
                    cv2.putText(annotated_frame, duration_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, duration_color, 2, cv2.LINE_AA)
        else:
            no_face_text = "FACE NOT DETECTED"
            cv2.putText(annotated_frame, no_face_text, (10, 30), 
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

    def is_wrist_inside_backpack(self, wrist_coords, backpack_bbox, margin=30):
        """Check if wrist keypoint is inside or very close to backpack bounding box.
        
        SIMPLIFIED PACKING DETECTION: If wrist is inside/near backpack bbox → Packing detected!
        
        This is based on the observation that when a person is handling/packing a bag,
        their wrist keypoints will be inside or very close to the bag's bounding box.
        
        Args:
            wrist_coords: (x, y) coordinates of wrist keypoint
            backpack_bbox: [x1, y1, x2, y2] bounding box of backpack/bag
            margin: additional margin around bbox (default 30px for tight detection)
            
        Returns:
            tuple: (is_inside, distance_to_center)
                - is_inside: True if wrist is inside/near backpack bbox
                - distance_to_center: Distance from wrist to backpack center (for confidence)
        """
        if wrist_coords is None or backpack_bbox is None:
            return False, float('inf')
        
        wx, wy = wrist_coords
        x1, y1, x2, y2 = backpack_bbox[:4]
        
        # Calculate backpack center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate distance to center
        import math
        distance = math.sqrt((wx - center_x) ** 2 + (wy - center_y) ** 2)
        
        # Check if wrist is inside bbox (with margin)
        is_inside = (x1 - margin <= wx <= x2 + margin and
                     y1 - margin <= wy <= y2 + margin)
        
        return is_inside, distance

    # NOTE: detect_pose_per_person removed - replaced by YOLOv8-Pose
    # NOTE: translate_pose_landmarks removed - not needed with YOLOv8-Pose (native multi-person)

    def validate_pose_landmarks(self, pose_landmarks, min_landmarks=10, min_visibility=0.3):
        """Validate that pose landmarks are valid and usable for activity detection.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            min_landmarks: Minimum number of landmarks required (default: 10)
            min_visibility: Minimum average visibility score (default: 0.3)
        
        Returns:
            bool: True if landmarks are valid, False otherwise
        """
        if pose_landmarks is None:
            return False
        
        if not hasattr(pose_landmarks, 'landmark') or len(pose_landmarks.landmark) < min_landmarks:
            return False
        
        # Validate coordinates are within valid range (0-1 for normalized)
        valid_count = 0
        total_visibility = 0.0
        
        for landmark in pose_landmarks.landmark:
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

    def validate_anatomical_consistency(self, pose_landmarks, frame_shape):
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

    def check_landmark_stability(self, person_idx, pose_landmarks, frame_shape):
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

    def detect_hand_gesture(self, pose_landmarks, frame_shape, person_roles, yolo_person_boxes=None, 
                           person_activities=None, backpack_detections=None, 
                           person_idx=None, current_timestamp=None, frame_number=None):
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
                for activity_type in ['writing', 'packing', 'cell_phone']:
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
            if person_activities.get('packing', False):
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
            gesture_logger.debug(f"{frame_info}GESTURE DEBUG - Checking {len(backpack_detections)} backpack(s) for person {matched_person_idx} ({matched_role})")
            
            for backpack_bbox in backpack_detections:
                bx1, by1, bx2, by2 = backpack_bbox[:4]
                backpack_center_x = (bx1 + bx2) / 2
                backpack_center_y = (by1 + by2) / 2
                
                # Check if either wrist is near the backpack
                right_dist = ((right_wrist_coords[0] - backpack_center_x) ** 2 + 
                             (right_wrist_coords[1] - backpack_center_y) ** 2) ** 0.5
                left_dist = ((left_wrist_coords[0] - backpack_center_x) ** 2 + 
                            (left_wrist_coords[1] - backpack_center_y) ** 2) ** 0.5
                
                gesture_logger.debug(f"{frame_info}GESTURE DEBUG - Backpack at ({backpack_center_x:.0f}, {backpack_center_y:.0f})")
                gesture_logger.debug(f"{frame_info}GESTURE DEBUG - Right wrist at {right_wrist_coords}, dist: {right_dist:.0f}px")
                gesture_logger.debug(f"{frame_info}GESTURE DEBUG - Left wrist at {left_wrist_coords}, dist: {left_dist:.0f}px")
                
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
        
        # Right hand: Check if it's in control operation zone
        # This identifies forward reaches to operate controls vs upward signaling
        # IMPROVED: More specific detection - only filter if it's CLEARLY a forward reach, not upward raise
        # For overhead cameras, we need to be more lenient to avoid false negatives
        right_in_control_zone = (
            # Hand is in reasonable vertical range (not extremely high for signaling)
            right_wrist_coords[1] > (my1 + (my2 - my1) * 0.2) and
            right_wrist_coords[1] < (my1 + (my2 - my1) * 0.8) and

            # CRITICAL: Wrist above shoulder but NOT too far (control panel operations: 30-100px)
            # True hand signals are typically >100px above shoulder (RELAXED from 120px)
            # Only filter if wrist is in the ambiguous range (30-100px) AND other conditions suggest control operation
            30 < right_wrist_shoulder_vertical < 100 and

            # IMPROVED: Elbow-wrist distance check - must be SMALL (forward reach, not vertical extension)
            # Control panel: elbow NOT significantly below wrist (forward reach pattern)
            # Hand signal: elbow MUST be significantly below wrist (vertical arm extension)
            # RELAXED: Only filter if wrist-elbow distance is very small (<50px), indicating forward reach
            right_wrist_elbow_distance < 50 and  # RELAXED from 80 to 50 - only filter clear forward reaches

            # ADDITIONAL: Elbow must be BELOW shoulder (forward reach pattern)
            # If elbow is at or above shoulder, it's likely an upward raise, not forward reach
            right_elbow_coords[1] > right_shoulder_coords[1] + 30  # Elbow significantly below shoulder
        )
        
        # Left hand: Check if it's in control operation zone
        # IMPROVED: More specific detection - only filter if it's CLEARLY a forward reach, not upward raise
        left_in_control_zone = (
            # Hand is in reasonable vertical range (not extremely high for signaling)
            left_wrist_coords[1] > (my1 + (my2 - my1) * 0.2) and
            left_wrist_coords[1] < (my1 + (my2 - my1) * 0.8) and

            # CRITICAL: Wrist above shoulder but NOT too far (control panel operations: 30-100px)
            # True hand signals are typically >100px above shoulder (RELAXED from 120px)
            # Only filter if wrist is in the ambiguous range (30-100px) AND other conditions suggest control operation
            30 < left_wrist_shoulder_vertical < 100 and

            # IMPROVED: Elbow-wrist distance check - must be SMALL (forward reach, not vertical extension)
            # Control panel: elbow NOT significantly below wrist (forward reach pattern)
            # Hand signal: elbow MUST be significantly below wrist (vertical arm extension)
            # RELAXED: Only filter if wrist-elbow distance is very small (<50px), indicating forward reach
            left_wrist_elbow_distance < 50 and  # RELAXED from 80 to 50 - only filter clear forward reaches

            # ADDITIONAL: Elbow must be BELOW shoulder (forward reach pattern)
            # If elbow is at or above shoulder, it's likely an upward raise, not forward reach
            left_elbow_coords[1] > left_shoulder_coords[1] + 30  # Elbow significantly below shoulder
        )
        
        # Right hand gesture detection (HAND RAISED TO FACE OR ABOVE)
        # RELAXED thresholds to detect drinking/eating and any hand-to-face actions (2024-12-04)
        # Detects: hand raised to face level, drinking, eating, signaling gestures
        right_hand_raised = (
            # CRITICAL: Wrist must belong to the same person (within expanded bbox)
            right_wrist_in_expanded and

            # REMOVED: Control zone filter - we now want to detect hand-to-face actions
            # not right_in_control_zone and

            # Hand at or above shoulder level (includes face-level actions like drinking)
            # >0 means wrist is at or above shoulder height
            right_wrist_shoulder_vertical > 0 and

            # RELAXED: Allow bent arm (drinking position has wrist near elbow level)
            # >-30 allows wrist to be slightly below elbow (bent arm holding cup)
            right_wrist_elbow_distance > -30 and

            # RELAXED: Allow arm close to body (drinking/eating position)
            # >20px minimal extension (hand not completely against body)
            right_arm_extension > 20 and

            # Allow elbow to be below shoulder (natural drinking/eating position)
            (right_elbow_coords[1] < right_shoulder_coords[1] + 150) and

            # Visibility checks (FURTHER RELAXED for overhead cameras)
            right_wrist.visibility > 0.3 and
            right_elbow.visibility > 0.3 and
            right_shoulder.visibility > 0.4 and

            # Within frame bounds
            0 < right_wrist_coords[0] < w and
            0 < right_wrist_coords[1] < h
        )
        
        # Left hand gesture detection (HAND RAISED TO FACE OR ABOVE)
        # RELAXED thresholds to detect drinking/eating and any hand-to-face actions (2024-12-04)
        # Detects: hand raised to face level, drinking, eating, signaling gestures
        left_hand_raised = (
            # CRITICAL: Wrist must belong to the same person (within expanded bbox)
            left_wrist_in_expanded and

            # REMOVED: Control zone filter - we now want to detect hand-to-face actions
            # not left_in_control_zone and

            # Hand at or above shoulder level (includes face-level actions like drinking)
            # >0 means wrist is at or above shoulder height
            left_wrist_shoulder_vertical > 0 and

            # RELAXED: Allow bent arm (drinking position has wrist near elbow level)
            # >-30 allows wrist to be slightly below elbow (bent arm holding cup)
            left_wrist_elbow_distance > -30 and

            # RELAXED: Allow arm close to body (drinking/eating position)
            # >20px minimal extension (hand not completely against body)
            left_arm_extension > 20 and

            # Allow elbow to be below shoulder (natural drinking/eating position)
            (left_elbow_coords[1] < left_shoulder_coords[1] + 150) and

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
        
        # COMPREHENSIVE DEBUG LOGGING for all checks
        frame_info = f"[Frame {frame_number}]" if frame_number is not None else ""
        # DEBUG LOGGING ENABLED for hand gesture troubleshooting
        gesture_logger.info(f"\n{'='*80}")
        gesture_logger.info(f"{frame_info} HAND GESTURE DEBUG - Person {matched_person_idx} ({matched_role})")
        gesture_logger.info(f"{'='*80}")
        gesture_logger.info(f"MATCHED BBOX: {matched_bbox}")
        gesture_logger.info(f"EXPANDED BBOX: x=[{expanded_x1:.0f}, {expanded_x2:.0f}], y=[{expanded_y1:.0f}, {expanded_y2:.0f}]")
        gesture_logger.info(f"RIGHT HAND ANALYSIS:")
        gesture_logger.info(f"  - Wrist coords: {right_wrist_coords}")
        gesture_logger.info(f"  - Shoulder coords: {right_shoulder_coords}")
        gesture_logger.info(f"  - Elbow coords: {right_elbow_coords}")
        gesture_logger.info(f"  - Wrist above Shoulder: {right_wrist_shoulder_vertical:.1f}px (need >0)")
        gesture_logger.info(f"  - Wrist above Elbow: {right_wrist_elbow_distance:.1f}px (need >-30)")
        gesture_logger.info(f"  - Arm extension (lateral): {right_arm_extension:.1f}px (need >20)")
        gesture_logger.info(f"  - Elbow position check: {right_elbow_coords[1] - right_shoulder_coords[1]:.1f}px (need <150)")
        gesture_logger.info(f"  - Wrist in expanded bbox: {right_wrist_in_expanded}")
        gesture_logger.info(f"  - Visibility - Wrist: {right_wrist.visibility:.2f} (need>0.3), Elbow: {right_elbow.visibility:.2f} (need>0.3), Shoulder: {right_shoulder.visibility:.2f} (need>0.4)")
        gesture_logger.info(f"  - Within frame bounds: {0 < right_wrist_coords[0] < w and 0 < right_wrist_coords[1] < h}")
        gesture_logger.info(f"  - RIGHT HAND RAISED: {right_hand_raised}")

        gesture_logger.info(f"LEFT HAND ANALYSIS:")
        gesture_logger.info(f"  - Wrist coords: {left_wrist_coords}")
        gesture_logger.info(f"  - Shoulder coords: {left_shoulder_coords}")
        gesture_logger.info(f"  - Elbow coords: {left_elbow_coords}")
        gesture_logger.info(f"  - Wrist above Shoulder: {left_wrist_shoulder_vertical:.1f}px (need >0)")
        gesture_logger.info(f"  - Wrist above Elbow: {left_wrist_elbow_distance:.1f}px (need >-30)")
        gesture_logger.info(f"  - Arm extension (lateral): {left_arm_extension:.1f}px (need >20)")
        gesture_logger.info(f"  - Elbow position check: {left_elbow_coords[1] - left_shoulder_coords[1]:.1f}px (need <150)")
        gesture_logger.info(f"  - Wrist in expanded bbox: {left_wrist_in_expanded}")
        gesture_logger.info(f"  - Visibility - Wrist: {left_wrist.visibility:.2f} (need>0.3), Elbow: {left_elbow.visibility:.2f} (need>0.3), Shoulder: {left_shoulder.visibility:.2f} (need>0.4)")
        gesture_logger.info(f"  - Within frame bounds: {0 < left_wrist_coords[0] < w and 0 < left_wrist_coords[1] < h}")
        gesture_logger.info(f"  - LEFT HAND RAISED: {left_hand_raised}")

        gesture_logger.info(f"FINAL RESULT: {'GESTURE DETECTED' if hand_gesture_detected else 'NO GESTURE'}")
        gesture_logger.info(f"{'='*80}\n")
        
        if not hand_gesture_detected:
            return False, False, {}

        # Analyze velocity and trajectory
        velocity_analysis = self.analyze_hand_velocity_and_trajectory(
            matched_person_idx, pose_landmarks, frame_shape, current_timestamp
        )

        # Log velocity analysis for debugging
        if velocity_analysis.get('analysis_quality') == 'good':
            rapid_raise = velocity_analysis['rapid_raise_detected']
            gesture_logger.debug(
                f"[VELOCITY] Person {matched_person_idx}: "
                f"R_vel={velocity_analysis['right_velocity']:.1f}px/s ({velocity_analysis['right_trajectory']}), "
                f"L_vel={velocity_analysis['left_velocity']:.1f}px/s ({velocity_analysis['left_trajectory']}), "
                f"Rapid raise: {rapid_raise}"
            )

            if not rapid_raise:
                gesture_logger.debug(f"[VELOCITY] No rapid raise - may be control operation")

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
        Check for hand gesture coordination failures using session-based tracking.

        Session-based approach:
        - A session starts when either LP or ALP raises their hand
        - Within a session, track if each person raised at least once
        - Session ends after configurable timeout (no hand raises)
        - Violation only if one person NEVER raised during the entire session

        This fixes false positives when:
        - LP raises hand once to instruct ALP
        - ALP raises hand multiple times in response
        - The old rolling window approach incorrectly flagged this as LP not coordinating

        Args:
            lp_detected: LP hand gesture detected in current frame
            alp_detected: ALP hand gesture detected in current frame
            current_time: Current timestamp in seconds

        Returns:
            tuple: (lp_not_coordinating, alp_not_coordinating)
                - lp_not_coordinating: True if ALP raised hand but LP NEVER raised in session
                - alp_not_coordinating: True if LP raised hand but ALP NEVER raised in session
        """
        session = self.hand_coordination_session
        timeout = self.hand_coordination_session_timeout

        # Case 1: No session active, check if we need to start one
        if not session['active']:
            if lp_detected or alp_detected:
                # Start a new session
                session['active'] = True
                session['start_time'] = current_time
                session['lp_raised'] = lp_detected
                session['alp_raised'] = alp_detected
                session['last_activity_time'] = current_time
                session['lp_raise_count'] = 1 if lp_detected else 0
                session['alp_raise_count'] = 1 if alp_detected else 0
                self.logger.debug(f"[COORDINATION SESSION] Started at {current_time:.2f}s - "
                                f"LP: {lp_detected}, ALP: {alp_detected}")
            # No violation on session start or when nothing detected
            return False, False

        # Case 2: Session is active
        # First, check if session has timed out
        time_since_last = current_time - session['last_activity_time']

        if time_since_last > timeout and not lp_detected and not alp_detected:
            # Session timed out with no new activity
            # Check if one person NEVER raised during the session
            lp_not_coordinating = session['alp_raised'] and not session['lp_raised']
            alp_not_coordinating = session['lp_raised'] and not session['alp_raised']

            if lp_not_coordinating or alp_not_coordinating:
                self.logger.info(f"[COORDINATION SESSION] Timeout violation at {current_time:.2f}s - "
                               f"LP raised: {session['lp_raised']} ({session['lp_raise_count']}x), "
                               f"ALP raised: {session['alp_raised']} ({session['alp_raise_count']}x), "
                               f"Session duration: {current_time - session['start_time']:.2f}s")
            else:
                self.logger.debug(f"[COORDINATION SESSION] Ended at {current_time:.2f}s - "
                                f"Both participated, no violation. "
                                f"LP: {session['lp_raise_count']}x, ALP: {session['alp_raise_count']}x")

            # Reset session
            self._reset_coordination_session()
            return lp_not_coordinating, alp_not_coordinating

        # Session is active and not timed out
        if lp_detected or alp_detected:
            # Update session with new activity
            session['lp_raised'] = session['lp_raised'] or lp_detected
            session['alp_raised'] = session['alp_raised'] or alp_detected
            session['last_activity_time'] = current_time
            if lp_detected:
                session['lp_raise_count'] += 1
            if alp_detected:
                session['alp_raise_count'] += 1
            self.logger.debug(f"[COORDINATION SESSION] Activity at {current_time:.2f}s - "
                            f"LP: {lp_detected} (total: {session['lp_raise_count']}), "
                            f"ALP: {alp_detected} (total: {session['alp_raise_count']})")

        # No violation while session is active with ongoing activity
        return False, False

    def _reset_coordination_session(self):
        """Reset the hand coordination session to initial state."""
        self.hand_coordination_session = {
            'active': False,
            'start_time': None,
            'lp_raised': False,
            'alp_raised': False,
            'last_activity_time': None,
            'lp_raise_count': 0,
            'alp_raise_count': 0
        }

    def _finalize_coordination_session(self, current_time):
        """
        Finalize any active coordination session at video end.

        Called when video processing completes to ensure pending sessions
        are evaluated and violations are not missed.

        Args:
            current_time: Final timestamp in seconds

        Returns:
            tuple: (lp_not_coordinating, alp_not_coordinating) or (False, False) if no active session
        """
        session = self.hand_coordination_session

        if not session['active']:
            return False, False

        # Session was active at video end - evaluate it
        lp_not_coordinating = session['alp_raised'] and not session['lp_raised']
        alp_not_coordinating = session['lp_raised'] and not session['alp_raised']

        if lp_not_coordinating or alp_not_coordinating:
            self.logger.info(f"[COORDINATION SESSION] Video-end violation at {current_time:.2f}s - "
                           f"LP raised: {session['lp_raised']} ({session['lp_raise_count']}x), "
                           f"ALP raised: {session['alp_raised']} ({session['alp_raise_count']}x)")
        else:
            self.logger.debug(f"[COORDINATION SESSION] Video-end evaluation at {current_time:.2f}s - "
                            f"Both participated, no violation")

        # Reset session
        self._reset_coordination_session()
        return lp_not_coordinating, alp_not_coordinating

    def analyze_hand_velocity_and_trajectory(self, person_idx, landmarks, frame_shape, timestamp_sec):
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

    def analyze_packing_hand_motion(self, person_idx, landmarks, frame_shape, timestamp_sec, backpack_bbox):
        """Analyze hand motion patterns to detect actual packing activity.

        Detects repeated back-and-forth movement between body and backpack:
        - Tracks hand distance from backpack center over time
        - Identifies direction changes (closer → farther → closer → farther)
        - Validates moderate velocity (30-120 pixels per frame)
        - Requires at least 2 direction changes for packing pattern

        Args:
            person_idx: Person identifier
            landmarks: MediaPipe pose landmarks
            frame_shape: (height, width, channels)
            timestamp_sec: Current timestamp
            backpack_bbox: (x1, y1, x2, y2) backpack bounding box

        Returns:
            dict with motion analysis results:
            - 'packing_motion_detected': bool
            - 'direction_changes': int
            - 'avg_velocity': float (pixels per frame)
            - 'history_length': int
        """
        h, w = frame_shape[:2]

        # Calculate backpack center
        bp_x1, bp_y1, bp_x2, bp_y2 = backpack_bbox
        backpack_center_x = (bp_x1 + bp_x2) / 2
        backpack_center_y = (bp_y1 + bp_y2) / 2

        # Get current hand positions (use closer hand to backpack)
        right_wrist = self.get_keypoint(landmarks, 'right_wrist')
        left_wrist = self.get_keypoint(landmarks, 'left_wrist')

        right_x, right_y = int(right_wrist.x * w), int(right_wrist.y * h)
        left_x, left_y = int(left_wrist.x * w), int(left_wrist.y * h)

        # Calculate distances to backpack center
        import math
        right_dist = math.sqrt((right_x - backpack_center_x)**2 + (right_y - backpack_center_y)**2)
        left_dist = math.sqrt((left_x - backpack_center_x)**2 + (left_y - backpack_center_y)**2)

        # Use the hand closer to backpack
        current_distance = min(right_dist, left_dist)
        active_hand = 'right' if right_dist < left_dist else 'left'

        # Initialize packing motion history for this person
        if not hasattr(self, 'packing_motion_history'):
            self.packing_motion_history = {}

        if person_idx not in self.packing_motion_history:
            self.packing_motion_history[person_idx] = {
                'distances': deque(maxlen=6),  # Track last 6 frames (12 seconds @ 0.5fps)
                'timestamps': deque(maxlen=6),
                'active_hand': deque(maxlen=6)
            }

        history = self.packing_motion_history[person_idx]

        # Add current position
        history['distances'].append(current_distance)
        history['timestamps'].append(timestamp_sec)
        history['active_hand'].append(active_hand)

        # REDUCED: Need at least 3 samples to detect pattern (6 seconds @ 0.5fps)
        # This allows faster detection while still having enough data
        if len(history['distances']) < 3:
            return {
                'packing_motion_detected': False,
                'direction_changes': 0,
                'avg_velocity': 0.0,
                'history_length': len(history['distances']),
                'reason': 'insufficient_history'
            }

        # Analyze motion pattern
        distances = list(history['distances'])

        # Calculate velocity (pixels per second, accounting for frame timing)
        # At 0.5fps, frames are 2 seconds apart, so we need to normalize
        velocities = []
        for i in range(1, len(distances)):
            distance_change = abs(distances[i] - distances[i-1])
            time_diff = history['timestamps'][i] - history['timestamps'][i-1]
            if time_diff > 0:
                # Convert to pixels per second
                velocity = distance_change / time_diff
            else:
                # Fallback: assume 2 seconds between frames at 0.5fps
                velocity = distance_change / 2.0
            velocities.append(velocity)

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0

        # Detect direction changes (closer → farther → closer → farther)
        direction_changes = 0
        prev_direction = None

        for i in range(1, len(distances)):
            current_direction = 'closer' if distances[i] < distances[i-1] else 'farther'

            if prev_direction and current_direction != prev_direction:
                direction_changes += 1

            prev_direction = current_direction

        # Packing motion criteria (RELAXED):
        # 1. At least 1 direction change (back-and-forth motion) - REDUCED from 2
        # 2. Moderate velocity (15-200 px/sec - RELAXED range, accounting for frame timing)
        # 3. Hand consistency (same hand used in recent frames) - RELAXED
        # 4. OR: Sustained proximity (hand near backpack for extended time)

        # Check hand consistency (at least 2 of last 3 frames use same hand) - RELAXED
        recent_hands = list(history['active_hand'])[-3:]
        hand_consistency = recent_hands.count(active_hand) >= 2 if len(recent_hands) >= 2 else True

        # Check sustained proximity (hand consistently close to backpack)
        # If all distances are below threshold, it's sustained proximity
        proximity_threshold = 100  # pixels
        sustained_proximity = all(d < proximity_threshold for d in distances[-3:]) if len(distances) >= 3 else False
        
        # Calculate time span of history
        time_span = history['timestamps'][-1] - history['timestamps'][0] if len(history['timestamps']) >= 2 else 0
        sustained_proximity_time = time_span >= 4.0  # 4+ seconds of proximity

        # Packing detected if:
        # - Motion pattern detected (direction changes + velocity + consistency), OR
        # - Sustained proximity for 4+ seconds (simpler case: hand just stays near backpack)
        packing_detected = (
            (direction_changes >= 1 and  # REDUCED from 2
             15 <= avg_velocity <= 200 and  # RELAXED range: 15-200 px/sec (was 30-120 px/frame)
             hand_consistency) or
            (sustained_proximity and sustained_proximity_time)  # NEW: Sustained proximity fallback
        )

        return {
            'packing_motion_detected': packing_detected,
            'direction_changes': direction_changes,
            'avg_velocity': avg_velocity,
            'history_length': len(history['distances']),
            'hand_consistency': hand_consistency,
            'active_hand': active_hand,
            'sustained_proximity': sustained_proximity,
            'sustained_proximity_time': sustained_proximity_time,
            'time_span': time_span,
            'reason': 'valid_pattern' if packing_detected else ('sustained_proximity' if (sustained_proximity and sustained_proximity_time) else 'no_packing_pattern')
        }

    # NOTE: detect_multi_person_pose_and_gestures removed - replaced by YOLOv8-Pose

    def _match_pose_to_roles(self, yolo_pose_results, person_roles):
        """Match YOLOv8-Pose detections to identified person roles by bounding box IoU.

        Enhanced with torso-center fallback for cases where bboxes overlap significantly.

        Args:
            yolo_pose_results: Dict from YoloPoseAdapter.process() containing:
                {person_idx: {'bbox': [...], 'keypoints': YoloPoseLandmarks}}
            person_roles: Dict from identify_person_roles() containing:
                {person_idx: {'bbox': [...], 'role': 'LP'/'ALP', ...}}

        Returns:
            Dict mapping person_idx (from person_roles) to YoloPoseLandmarks
        """
        matched = {}
        used_yolo_indices = set()

        gesture_logger.info(f"[POSE MATCHING] Matching {len(yolo_pose_results)} YOLO poses to {len(person_roles)} person roles")

        for person_idx, role_data in person_roles.items():
            if 'bbox' not in role_data:
                continue

            role_bbox = role_data['bbox']
            role_center_x = (role_bbox[0] + role_bbox[2]) / 2
            role_center_y = (role_bbox[1] + role_bbox[3]) / 2
            role_name = role_data.get('role', 'UNKNOWN')

            # Collect all candidates with their IoU scores
            candidates = []
            for yolo_idx, yolo_data in yolo_pose_results.items():
                if yolo_idx in used_yolo_indices:
                    continue

                iou = self.calculate_iou(role_bbox, yolo_data['bbox'])
                candidates.append({
                    'yolo_idx': yolo_idx,
                    'iou': iou,
                    'keypoints': yolo_data['keypoints'],
                    'bbox': yolo_data['bbox']
                })

            # Sort by IoU descending
            candidates.sort(key=lambda x: x['iou'], reverse=True)

            # Log all candidates
            for c in candidates:
                gesture_logger.debug(f"  [{role_name}] Candidate YOLO {c['yolo_idx']}: IoU={c['iou']:.3f}")

            if not candidates:
                gesture_logger.warning(f"[POSE MATCHING] No candidates for {role_name} (person {person_idx})")
                continue

            best_candidate = candidates[0]

            # Check if top two candidates have similar IoU (within 0.15) - use torso center as tiebreaker
            if len(candidates) >= 2 and candidates[0]['iou'] - candidates[1]['iou'] < 0.15:
                gesture_logger.info(f"[POSE MATCHING] Close IoU scores for {role_name}: {candidates[0]['iou']:.3f} vs {candidates[1]['iou']:.3f} - using torso center")

                # Calculate torso center for each candidate using shoulders
                best_dist = float('inf')
                for c in candidates[:2]:  # Only compare top 2
                    keypoints = c['keypoints']
                    if len(keypoints.landmark) >= 7:  # Need at least shoulders
                        # Get shoulder positions (indices 5 and 6 for left/right shoulder)
                        left_shoulder = keypoints.landmark[5]
                        right_shoulder = keypoints.landmark[6]

                        # Get frame dimensions from bbox (approximate)
                        bbox = c['bbox']
                        frame_w = max(bbox[2], 1920)  # Estimate frame width
                        frame_h = max(bbox[3], 1080)  # Estimate frame height

                        # Calculate torso center in pixel coords
                        torso_x = ((left_shoulder.x + right_shoulder.x) / 2) * frame_w
                        torso_y = ((left_shoulder.y + right_shoulder.y) / 2) * frame_h

                        # Distance from role bbox center to torso center
                        dist = ((torso_x - role_center_x) ** 2 + (torso_y - role_center_y) ** 2) ** 0.5

                        gesture_logger.debug(f"    Candidate {c['yolo_idx']}: torso=({torso_x:.0f}, {torso_y:.0f}), dist={dist:.0f}px")

                        if dist < best_dist:
                            best_dist = dist
                            best_candidate = c

                gesture_logger.info(f"[POSE MATCHING] Torso-based selection: YOLO {best_candidate['yolo_idx']} (dist={best_dist:.0f}px)")

            # Match if IoU is above threshold (LOWERED from 0.3 to 0.2 for overlapping cases)
            if best_candidate['iou'] > 0.2:
                matched[person_idx] = best_candidate['keypoints']
                used_yolo_indices.add(best_candidate['yolo_idx'])
                gesture_logger.info(f"[POSE MATCHING] Matched {role_name} (person {person_idx}) -> YOLO {best_candidate['yolo_idx']} (IoU={best_candidate['iou']:.3f})")
            else:
                gesture_logger.warning(f"[POSE MATCHING] No match for {role_name} (person {person_idx}): best IoU={best_candidate['iou']:.3f} < 0.2")

        return matched

    def process_all_persons_activities(self, frame, detections, person_roles, timestamp_sec, face_results=None, frame_number=None, precomputed_pose_results=None):
        """Process all detected persons for ALL activity detections (mind diversion, sleep, etc.)

        This is the MAIN multi-person processing method that:
        1. Runs YOLOv8-Pose once to get all persons with keypoints (or uses precomputed results)
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
                            'packing': bool,
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

        # ============ YOLOV8-POSE: Single inference for all persons ============
        # Run YOLOv8-Pose once on the full frame to get all persons with keypoints
        # This replaces the per-person MediaPipe cropping loop for better performance
        # If precomputed_pose_results is provided (from GPU batch inference), use it directly
        if precomputed_pose_results is not None:
            yolo_pose_results = precomputed_pose_results
        else:
            yolo_pose_results = self.yolo_pose.process(frame)

        # Match YOLO pose detections to person_roles by bounding box IoU
        matched_poses = self._match_pose_to_roles(yolo_pose_results, person_roles)

        # Process each person individually
        for person_idx, person_data in person_roles.items():
            if 'bbox' not in person_data:
                continue

            bbox = person_data['bbox']  # [x1, y1, x2, y2]

            # Get matched pose keypoints for this person
            # YOLOv8-Pose provides full-frame coordinates directly (no cropping/translation needed)
            try:
                if person_idx not in matched_poses:
                    # No pose detected for this person, skip
                    continue

                # Get the matched YoloPoseLandmarks (MediaPipe-compatible interface)
                translated_landmarks = matched_poses[person_idx]

                # Validate landmarks are valid before using for activity detection
                if translated_landmarks is None or len(translated_landmarks.landmark) == 0:
                    continue

                # Check if at least some keypoints have good visibility
                visible_count = sum(1 for lm in translated_landmarks.landmark if lm.visibility > 0.3)
                if visible_count < 5:
                    # Not enough visible keypoints, skip this person
                    continue

                # ============ KEYPOINT CONSISTENCY VALIDATION ============
                # Verify that torso center falls within (or near) the person's bbox
                # This catches cases where pose matching assigned wrong skeleton to person
                h, w = frame.shape[:2]
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
                    gesture_logger.warning(
                        f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                        f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) outside expanded bbox "
                        f"[{expanded_x1:.0f}-{expanded_x2:.0f}, {expanded_y1:.0f}-{expanded_y2:.0f}] - SKIPPING"
                    )
                    continue
                else:
                    gesture_logger.debug(
                        f"[KEYPOINT VALIDATION] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): "
                        f"Torso center ({torso_center_x:.0f}, {torso_center_y:.0f}) VALID within bbox"
                    )

                # Initialize activity detection results for this person
                person_activities = {
                    'mind_diversion': False,
                    'sleep': False,
                    'microsleep': False,
                    'cell_phone': False,
                    'writing': False,
                    'packing': False,
                    'lp_hand_gesture': False,
                    'alp_hand_gesture': False
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
                
                # ============ PER-PERSON OBJECT DETECTION ============
                # Run ROI-based object detection around THIS person's hands/body parts
                # This creates ROIs specifically for this person's pose landmarks
                person_detections = self.detect_objects(frame, translated_landmarks, use_pose_guided=True)
                
                # Merge person-specific detections into the main detections dict
                # This ensures each person's hand ROIs are checked for cell phones
                if person_detections['cell_phone']:
                    detections['cell_phone'].extend(person_detections['cell_phone'])
                if person_detections['book']:
                    detections['book'].extend(person_detections['book'])
                
                # DEBUG: Log per-person ROI detection results
                import logging
                debug_logger = logging.getLogger('locopilot_monitor')
                if person_detections['cell_phone']:
                    debug_logger.info(f"[MULTI-PERSON ROI] Person {person_idx} ({person_data.get('role', 'UNKNOWN')}): Found {len(person_detections['cell_phone'])} cell phone(s)")
                
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
                pose_sleep_detected, pose_microsleep_detected, pose_sleep_info = self.detect_pose_based_sleep(
                    translated_landmarks, timestamp_sec
                )
                person_debug_info['sleep_info'] = pose_sleep_info
                person_activities['sleep'] = pose_sleep_detected
                person_activities['microsleep'] = pose_microsleep_detected
                
                # 3. CELL PHONE DETECTION (check if hand near phone in THIS person's region)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                if len(detections['cell_phone']) > 0:
                    # DEBUG: Log when cell phones are detected
                    import logging
                    debug_logger = logging.getLogger('locopilot_monitor')
                    if self.consecutive_detections.get('cell_phone', 0) == 0:
                        debug_logger.info(f"[DEBUG CELL PHONE] {len(detections['cell_phone'])} phone(s) detected in frame")
                    right_hand = self.get_keypoint(translated_landmarks, 'right_wrist')
                    left_hand = self.get_keypoint(translated_landmarks, 'left_wrist')

                    right_hand_coords = (int(right_hand.x * w), int(right_hand.y * h))
                    left_hand_coords = (int(left_hand.x * w), int(left_hand.y * h))
                    
                    # STRICTER MARGIN: Reduced from default to ensure phone is really near hand
                    margin = 100  # Reduced from activity_thresholds margin to be more strict
                    
                    for phone_bbox in detections['cell_phone']:
                        # Check if phone bbox overlaps with person bbox (with margin)
                        phone_in_person_region = self.bbox_overlap_with_margin(phone_bbox, bbox, margin)
                        
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
                
                # 4. WRITING DETECTION (check if hand near book OR wrist/elbow proximity heuristic)
                # MOVED BEFORE HAND GESTURE: Need to detect this first for context-aware filtering
                writing_detected_by_book = False
                writing_detected_by_wrist = False
                writing_detected_by_book_posture = False  # NEW fallback method

                # Method 1: Book detection (existing method - requires wrists visible)
                if len(detections['book']) > 0:
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
                        for book_bbox in detections['book']:
                            # Check if book is in this person's region (use large margin for lap area)
                            book_in_person_region = self.bbox_overlap_with_margin(book_bbox, bbox, person_book_margin)

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
                            detections['book'],
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
                    should_suppress, suppress_reason = self.should_suppress_mind_diversion(
                        person_idx=person_idx,
                        person_activities=person_activities,
                        pose_landmarks=translated_landmarks,
                        detections=detections,
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

                # Helper method for temporal smoothing of hand positions
                def _get_smoothed_hand_position(person_idx, hand_side, landmark, w, h, timestamp_sec):
                    """Get temporally smoothed hand position to reduce pose estimation noise.

                    Uses simple average over last 3 positions (6 seconds @ 0.5fps).

                    Args:
                        person_idx: Person identifier
                        hand_side: 'right' or 'left'
                        landmark: MediaPipe hand landmark
                        w, h: Frame dimensions
                        timestamp_sec: Current timestamp

                    Returns:
                        tuple: (smoothed_x, smoothed_y) coordinates
                    """
                    # Initialize smoothing buffers
                    if not hasattr(self, 'hand_smoothing_buffers'):
                        self.hand_smoothing_buffers = {}

                    key = (person_idx, hand_side)
                    if key not in self.hand_smoothing_buffers:
                        self.hand_smoothing_buffers[key] = {
                            'positions': deque(maxlen=3),  # Last 3 positions (6 seconds)
                            'timestamps': deque(maxlen=3)
                        }

                    buffer = self.hand_smoothing_buffers[key]

                    # Add current position
                    current_x = int(landmark.x * w)
                    current_y = int(landmark.y * h)
                    buffer['positions'].append((current_x, current_y))
                    buffer['timestamps'].append(timestamp_sec)

                    # Calculate average position
                    if len(buffer['positions']) > 0:
                        avg_x = sum(pos[0] for pos in buffer['positions']) / len(buffer['positions'])
                        avg_y = sum(pos[1] for pos in buffer['positions']) / len(buffer['positions'])
                        return (int(avg_x), int(avg_y))

                    return (current_x, current_y)

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
                        right_hand_coords = _get_smoothed_hand_position(
                            person_idx, 'right', right_hand, w, h, timestamp_sec
                        )
                    elif left_wrist_visible:
                        # Fallback: if right wrist not visible, try using right elbow as approximation
                        right_elbow = self.get_keypoint(translated_landmarks, 'right_elbow')
                        if right_elbow.visibility > 0.3:
                            right_hand_coords = (int(right_elbow.x * w), int(right_elbow.y * h))

                    if left_wrist_visible:
                        left_hand_coords = _get_smoothed_hand_position(
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
                    # Core logic: If wrist is inside/near backpack bbox → Packing detected!
                    # This is simpler and more direct than motion analysis.
                    packing_motion_analysis = None
                    packing_detected_simple = False
                    
                    for backpack_bbox in detections['backpack']:
                        # Check if backpack is in this person's region (wider margin)
                        backpack_in_person_region = self.bbox_overlap_with_margin(
                            backpack_bbox, bbox, region_margin
                        )

                        if backpack_in_person_region:
                            # ===== SIMPLIFIED CHECK: Is wrist INSIDE backpack bbox? =====
                            # This directly detects if hand is interacting with the bag
                            right_inside, right_dist = self.is_wrist_inside_backpack(
                                right_hand_coords, backpack_bbox, margin=40  # 40px margin for tolerance
                            )
                            left_inside, left_dist = self.is_wrist_inside_backpack(
                                left_hand_coords, backpack_bbox, margin=40
                            )
                            
                            wrist_inside_backpack = right_inside or left_inside
                            closest_distance = min(right_dist, left_dist)
                            
                            # Store debug info
                            person_debug_info['packing_wrist_check'] = {
                                'right_wrist_inside': right_inside,
                                'left_wrist_inside': left_inside,
                                'right_dist': right_dist,
                                'left_dist': left_dist,
                                'closest_distance': closest_distance,
                                'backpack_bbox': list(backpack_bbox[:4])
                            }
                            
                            # ===== PRIMARY DETECTION: Wrist inside backpack bbox =====
                            if wrist_inside_backpack:
                                packing_detected_simple = True
                                self.logger.info(f"PACKING DETECTED (SIMPLE): Wrist inside backpack bbox! "
                                               f"Right: {right_inside} ({right_dist:.0f}px), "
                                               f"Left: {left_inside} ({left_dist:.0f}px)")

                                # Trigger packing detection immediately
                                should_trigger = self.update_per_person_detection(
                                    person_idx, 'packing_bags', True, timestamp_sec
                                )

                                # Stage 2: Collect for batch voting verification (if enabled)
                                if should_trigger and voting_collector is not None:
                                    voting_collector.add('packing_bags', person_idx, list(bbox))
                                    # Will be verified in batch at end of person processing
                                    # Tentatively set to True, will be updated after batch verification
                                    person_activities['packing'] = True
                                else:
                                    person_activities['packing'] = should_trigger

                                # Update temporal history for hand gesture suppression
                                if person_idx not in self.recent_person_activities:
                                    self.recent_person_activities[person_idx] = {}
                                self.recent_person_activities[person_idx]['packing'] = timestamp_sec
                                break

                            # ===== FALLBACK: Hand near backpack with motion analysis =====
                            hand_near_backpack = (
                                self.check_hand_object_interaction(right_hand_coords, backpack_bbox, proximity_margin) or
                                self.check_hand_object_interaction(left_hand_coords, backpack_bbox, proximity_margin)
                            )

                            if hand_near_backpack:
                                # Analyze hand motion patterns to confirm packing activity
                                packing_motion_analysis = self.analyze_packing_hand_motion(
                                    person_idx, translated_landmarks, frame.shape, timestamp_sec, backpack_bbox
                                )

                                # Store motion analysis in debug info
                                person_debug_info['packing_motion'] = packing_motion_analysis

                                # Trigger if motion analysis confirms packing pattern OR sustained proximity
                                motion_confirmed = packing_motion_analysis['packing_motion_detected']
                                sustained_proximity = packing_motion_analysis.get('sustained_proximity', False) and \
                                                     packing_motion_analysis.get('sustained_proximity_time', False)

                                if motion_confirmed or sustained_proximity:
                                    should_trigger = self.update_per_person_detection(
                                        person_idx, 'packing_bags', True, timestamp_sec
                                    )

                                    # Stage 2: Collect for batch voting verification (if enabled)
                                    if should_trigger and voting_collector is not None:
                                        voting_collector.add('packing_bags', person_idx, list(bbox))
                                        # Will be verified in batch at end of person processing
                                        # Tentatively set to True, will be updated after batch verification
                                        person_activities['packing'] = True
                                    else:
                                        person_activities['packing'] = should_trigger
                                    # UPDATE TEMPORAL HISTORY (for hand gesture suppression)
                                    if person_idx not in self.recent_person_activities:
                                        self.recent_person_activities[person_idx] = {}
                                    self.recent_person_activities[person_idx]['packing'] = timestamp_sec
                                else:
                                    # Hand is near but no packing motion - reset counter
                                    should_trigger = self.update_per_person_detection(
                                        person_idx, 'packing_bags', False, timestamp_sec
                                    )
                                    person_activities['packing'] = should_trigger
                                break
                
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

                # ============ BATCH VOTING VERIFICATION ============
                # Verify all collected activities in a single batch (shared inference)
                if voting_collector is not None and voting_collector.has_activities():
                    try:
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
                                    'packing_bags': 'packing',
                                    'lp_hand_gesture': 'lp_hand_gesture',
                                    'alp_hand_gesture': 'alp_hand_gesture'
                                }

                                person_key = activity_key_map.get(activity_type, activity_type)
                                person_activities[person_key] = is_confirmed

                                # Log result
                                if is_confirmed:
                                    self.logger.info(f"[VOTING BATCH] {activity_type} CONFIRMED: {vote_details.get('vote_breakdown', [])}")
                                else:
                                    self.logger.info(f"[VOTING BATCH] {activity_type} REJECTED: {vote_details.get('vote_breakdown', [])}")
                    except Exception as e:
                        self.logger.error(f"[VOTING BATCH] Error in batch verification: {e}")
                        # On error, set all collected activities to False (safe default)
                        for activity in voting_collector.get_activities():
                            activity_type = activity['type']
                            activity_key_map = {
                                'mind_diversion': 'mind_diversion',
                                'cell_phone': 'cell_phone',
                                'writing': 'writing',
                                'packing_bags': 'packing',
                                'lp_hand_gesture': 'lp_hand_gesture',
                                'alp_hand_gesture': 'alp_hand_gesture'
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
                self.logger.error(f"Error processing person {person_idx}: {e}")
                continue
        
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
            'performing_person': -1,
            'performing_persons': []  # List of person indices who performed activities
        }
        
        # Aggregate: if ANY person has an activity, mark it as detected
        for person_idx, person_data in persons_data.items():
            activities = person_data['activities']
            
            if activities['mind_diversion']:
                aggregated['mind_diversion_detected'] = True
                aggregated['performing_persons'].append(person_idx)
            if activities['sleep']:
                aggregated['sleep_detected'] = True
            if activities['microsleep']:
                aggregated['microsleep_detected'] = True
            if activities['cell_phone']:
                aggregated['cell_phone_detected'] = True
            if activities['writing']:
                aggregated['writing_detected'] = True
            if activities['packing']:
                aggregated['packing_detected'] = True
            if activities['lp_hand_gesture']:
                aggregated['lp_hand_gesture_detected'] = True
            if activities['alp_hand_gesture']:
                aggregated['alp_hand_gesture_detected'] = True
        
        # Set performing_person to the first detected person (for backward compatibility)
        if aggregated['performing_persons']:
            aggregated['performing_person'] = aggregated['performing_persons'][0]
        
        return {
            'persons': persons_data,
            'aggregated': aggregated
        }
    
    def bbox_overlap_with_margin(self, obj_bbox, person_bbox, margin):
        """Check if object bbox overlaps with person bbox (with margin)
        
        Args:
            obj_bbox: [x1, y1, x2, y2] object bounding box
            person_bbox: [x1, y1, x2, y2] person bounding box
            margin: margin to expand person bbox
            
        Returns:
            bool: True if object overlaps with person region
        """
        ox1, oy1, ox2, oy2 = obj_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Expand person bbox with margin
        px1_expanded = px1 - margin
        py1_expanded = py1 - margin
        px2_expanded = px2 + margin
        py2_expanded = py2 + margin
        
        # Check overlap
        if ox2 < px1_expanded or ox1 > px2_expanded:
            return False
        if oy2 < py1_expanded or oy1 > py2_expanded:
            return False
        
        return True
    
    def calculate_head_pose_angles(self, pose_landmarks, face_landmarks, frame_shape):
        """Calculate head pose angles (yaw and pitch) to detect mind diversion.

        Detects three types of mind diversion:
        1. looking_sideways - head turned > 55° (configurable)
        2. looking_away_combined - head turned > 40° AND down > 20°
        3. looking_down_distracted - head down > 30° (configurable)

        Uses both pose landmarks (nose, shoulders) and face mesh landmarks for accuracy.

        Args:
            pose_landmarks: MediaPipe pose landmarks
            face_landmarks: MediaPipe face mesh landmarks (can be None)
            frame_shape: (height, width) of frame

        Returns:
            dict: {
                'yaw': float,       # Side turn angle in degrees (-90 to +90)
                'pitch': float,     # Up/down tilt angle in degrees (-90 to +90)
                'detected': bool,   # True if mind diversion detected
                'sub_type': str,    # 'looking_sideways'|'looking_down_distracted'|'looking_away_combined'|None
                'method': str       # Detection method used ('pose_landmarks'|'face_mesh'|'none')
            }
        """
        h, w = frame_shape[:2]
        result = {'yaw': 0, 'pitch': 0, 'detected': False, 'sub_type': None, 'method': 'none'}
        
        if not pose_landmarks:
            return result

        try:
            # Get pose landmarks
            nose = self.get_keypoint(pose_landmarks, 'nose')
            left_shoulder = self.get_keypoint(pose_landmarks, 'left_shoulder')
            right_shoulder = self.get_keypoint(pose_landmarks, 'right_shoulder')
            left_ear = self.get_keypoint(pose_landmarks, 'left_ear')
            right_ear = self.get_keypoint(pose_landmarks, 'right_ear')

            # Check visibility
            if not nose or nose.visibility < 0.5:
                # FALLBACK: When nose not visible, use ear asymmetry for yaw estimation
                left_ear_vis = left_ear.visibility if left_ear else 0
                right_ear_vis = right_ear.visibility if right_ear else 0

                if left_ear_vis > 0.5 and right_ear_vis < 0.3:
                    # Right ear hidden = turned right (looking away from track)
                    yaw_angle = 60  # Estimate significant right turn
                    result['yaw'] = yaw_angle
                    result['method'] = 'ear_asymmetry'

                    # Check if this exceeds sideways threshold
                    settings = self.settings
                    # Positive yaw (looking right/away) - always check threshold
                    if yaw_angle > settings.mind_diversion_yaw_sideways:
                        result['detected'] = True
                        result['sub_type'] = 'looking_sideways'
                    return result
                elif right_ear_vis > 0.5 and left_ear_vis < 0.3:
                    # Left ear hidden = turned left (looking toward track - LEGITIMATE)
                    yaw_angle = -60  # Estimate significant left turn
                    result['yaw'] = yaw_angle
                    result['method'] = 'ear_asymmetry'

                    # Check if this exceeds sideways threshold
                    settings = self.settings
                    # Negative yaw (looking left/forward toward track)
                    # Only trigger if forward-looking exemption is disabled
                    exempt_forward = getattr(settings, 'mind_diversion_exempt_forward_looking', True)
                    if not exempt_forward and abs(yaw_angle) > settings.mind_diversion_yaw_sideways:
                        result['detected'] = True
                        result['sub_type'] = 'looking_sideways'
                    # If exempt_forward is True, negative yaw does NOT trigger detection
                    return result
                else:
                    # Neither ear pattern matches, cannot determine head pose
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

            # DETECTION LOGIC: Multi-scenario mind diversion with configurable thresholds
            # Uses settings from config for all thresholds
            settings = self.settings

            # Sub-type detection with priority order:
            # 1. looking_sideways - head turned significantly to side (HIGH CONFIDENCE)
            # 2. looking_away_combined - head turned AND down (HIGH CONFIDENCE)
            # 3. looking_down_distracted - only head down, not sideways (MEDIUM CONFIDENCE)

            sub_type = None

            # Forward-looking exemption: Camera is behind-right of crew
            # - Negative yaw = looking LEFT toward track/window (LEGITIMATE WORK)
            # - Positive yaw = looking RIGHT away from track (POTENTIAL DIVERSION)
            # When exempt_forward_looking is True, only positive yaw triggers detection
            exempt_forward = getattr(settings, 'mind_diversion_exempt_forward_looking', True)

            # Calculate effective yaw for threshold comparison
            # If exempting forward-looking, only positive (rightward) yaw counts
            # If not exempting, use absolute value (both directions count)
            if exempt_forward:
                effective_yaw = yaw_angle if yaw_angle > 0 else 0  # Only positive yaw triggers
            else:
                effective_yaw = abs(yaw_angle)  # Both directions trigger (original behavior)

            # Scenario 1: looking_sideways (head turned > threshold, regardless of pitch)
            if effective_yaw > settings.mind_diversion_yaw_sideways:
                sub_type = 'looking_sideways'
                result['detected'] = True
            # Scenario 2: looking_away_combined (turned AND down)
            elif (effective_yaw > settings.mind_diversion_yaw_combined and
                  pitch_angle > settings.mind_diversion_pitch_combined):
                sub_type = 'looking_away_combined'
                result['detected'] = True
            # Scenario 3: looking_down_distracted (only down, not sideways)
            elif (pitch_angle > settings.mind_diversion_pitch_down and
                  effective_yaw < settings.mind_diversion_yaw_max_for_down):
                sub_type = 'looking_down_distracted'
                result['detected'] = True

            result['sub_type'] = sub_type
            
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

                    # Re-evaluate detection with face mesh data using new thresholds
                    yaw_fm = result['yaw']
                    pitch_fm = result['pitch']
                    sub_type = None

                    # Forward-looking exemption (same logic as pose-based detection)
                    exempt_forward = getattr(settings, 'mind_diversion_exempt_forward_looking', True)
                    if exempt_forward:
                        effective_yaw_fm = yaw_fm if yaw_fm > 0 else 0  # Only positive yaw triggers
                    else:
                        effective_yaw_fm = abs(yaw_fm)  # Both directions trigger

                    # Scenario 1: looking_sideways
                    if effective_yaw_fm > settings.mind_diversion_yaw_sideways:
                        sub_type = 'looking_sideways'
                        result['detected'] = True
                    # Scenario 2: looking_away_combined
                    elif (effective_yaw_fm > settings.mind_diversion_yaw_combined and
                          pitch_fm > settings.mind_diversion_pitch_combined):
                        sub_type = 'looking_away_combined'
                        result['detected'] = True
                    # Scenario 3: looking_down_distracted
                    elif (pitch_fm > settings.mind_diversion_pitch_down and
                          effective_yaw_fm < settings.mind_diversion_yaw_max_for_down):
                        sub_type = 'looking_down_distracted'
                        result['detected'] = True
                    else:
                        result['detected'] = False

                    result['sub_type'] = sub_type

                except Exception as e:
                    # If face mesh processing fails, keep pose-based result
                    pass
            
            return result
            
        except (IndexError, AttributeError, ZeroDivisionError) as e:
            return {'yaw': 0, 'pitch': 0, 'detected': False, 'sub_type': None, 'method': 'error'}

    def should_suppress_mind_diversion(self, person_idx, person_activities, pose_landmarks, detections, frame_shape, current_time=None):
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
        
        # Phase 3: Check if we can reuse cached full-frame detection (avoid redundant inference)
        cache_age = time_module.time() - getattr(self, '_cached_frame_time', 0)
        if (hasattr(self, '_cached_frame_objects') and
            hasattr(self, '_cached_frame_time') and
            cache_age < 0.1):  # Cache valid for 100ms only
            # Reuse cached results instead of re-running inference
            yolo_results = self._cached_frame_objects
        else:
            # Cache miss or stale - run full-frame YOLO detection
            # Look for: tv/monitor, keyboard, mouse, laptop, book, backpack, cell phone
            yolo_results = self.yolo_model(frame, verbose=False, conf=0.3,
                                            imgsz=self.yolo_imgsz, device=self.yolo_device)
        
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
            # Two people - assign LP and ALP based on camera position
            # Logic: Person CLOSER to camera = LP (driver), person FURTHER from camera = ALP (assistant)
            #
            # How to determine "closer to camera":
            # 1. Bounding box area - larger area means person appears bigger (closer)
            # 2. Bottom Y coordinate - higher Y value means lower in frame (closer to camera)
            #
            # We use bounding box area as primary indicator (more reliable)

            def get_bbox_area(person):
                bbox = person['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return width * height

            def get_bottom_y(person):
                return person['bbox'][3]  # y2 = bottom of bounding box

            # Sort by bounding box area (largest first = closest to camera = LP)
            sorted_persons = sorted(person_scores, key=lambda x: get_bbox_area(x), reverse=True)

            self.logger.debug(f"Role assignment by camera position: "
                            f"Person {sorted_persons[0]['person_idx']} (area={get_bbox_area(sorted_persons[0]):.0f}) -> LP, "
                            f"Person {sorted_persons[1]['person_idx']} (area={get_bbox_area(sorted_persons[1]):.0f}) -> ALP")

            # Person with higher lp_score (or leftmost position) is LP
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
            # Sort by bounding box area (largest first = closest to camera)
            def get_bbox_area(person):
                bbox = person['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return width * height

            sorted_persons = sorted(person_scores, key=lambda x: get_bbox_area(x), reverse=True)
            self.logger.debug(f"Role assignment (3+ people) by camera position - areas: {[get_bbox_area(p) for p in sorted_persons]}")
            
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
    
    def start_activity(self, activity_name, timestamp, fps, frame_count, person_roles=None, ocr_timestamp=None):
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
            self.activities[activity_name]['frames'] = list(self.frame_buffer)
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
    
    def end_activity(self, activity_name, timestamp, fps, frame_count, people_count=1, save_clips=True, ocr_timestamp=None):
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
                if len(activity['frames']) > 0:
                    middle_frame_idx = len(activity['frames']) // 2
                    activity_image = activity['frames'][middle_frame_idx]
                    cv2.imwrite(image_path, activity_image)
            
            # Get video duration in HH:MM:SS format
            # ✅ MEMORY FIX: Use context manager to ensure video capture is released
            with video_capture_context(self.video_path) as cap:
                video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration_seconds = video_total_frames / fps
            
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
                        "lpScore": role_info['lp_score'],
                        "alpScore": role_info['alp_score']
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
                except:
                    pass
        return False

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

        # Re-encode to H.264 for browser compatibility
        # (mp4v codec from OpenCV doesn't play in browsers)
        self._reencode_to_h264(output_path)

    def extract_video_segment(self, source_video, output_path, start_seconds, end_seconds):
        """Extract video segment directly from source using ffmpeg for smooth playback.

        This method extracts the original video segment instead of reconstructing from
        sampled frames, resulting in smooth playback at the original frame rate.

        Args:
            source_video: Path to the source video file
            output_path: Path to save the extracted clip
            start_seconds: Start time in seconds
            end_seconds: End time in seconds
        """
        try:
            duration = end_seconds - start_seconds
            if duration <= 0:
                self.logger.warning(f"Invalid duration for clip extraction: {duration}")
                return False

            # Use ffmpeg to extract the segment directly from source video
            # -ss before -i for fast seeking, -t for duration
            # -c:v libx264 for H.264 encoding (browser compatible)
            # -movflags +faststart for web streaming optimization
            ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')  # Use system PATH
            cmd = [
                ffmpeg_path, '-y',
                '-ss', str(start_seconds),
                '-i', source_video,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',  # No audio needed for evidence clips
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120
            )

            if result.returncode == 0 and os.path.exists(output_path):
                self.logger.debug(f"Extracted video segment: {start_seconds:.2f}s - {end_seconds:.2f}s -> {output_path}")
                return True
            else:
                self.logger.warning(f"ffmpeg extraction failed: {result.stderr.decode()[:200]}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Video segment extraction timed out for: {output_path}")
            return False
        except Exception as e:
            self.logger.warning(f"Video segment extraction failed: {e}")
            return False

    def process_video(self):
        """Main video processing loop - SAMPLES FRAMES AT SPECIFIED RATE"""
        # Get video metadata
        # ✅ MEMORY FIX: Use context manager to ensure video capture is released
        with video_capture_context(self.video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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

            try:
                # Convert timestamp to HH:MM:SS format
                timestamp = str(timedelta(seconds=timestamp_sec))

                # Add frame to buffer
                self.frame_buffer.append(frame.copy())

                # TRAIN STATE DETECTION: Analyze if train is stopped or moving
                # Uses ROI-based optical flow on window regions (excludes cabin interior)
                train_is_stopped = False
                train_state = None
                train_state_changed = False
                if self.train_state_detector is not None:
                    train_state = self.train_state_detector.analyze_frame(frame, timestamp_sec)
                    train_is_stopped = self.train_state_detector.is_stopped()
                    train_state_changed = self.train_state_detector.state_changed()

                    # Log state changes
                    if train_state_changed:
                        state_name = train_state.name if hasattr(train_state, 'name') else str(train_state)
                        self.logger.info(f"[{timestamp}] Train state changed to: {state_name}")

                # WHEN STOPPED: Run minimal person detection only
                # - Check no_person_detected violation (0 persons = violation)
                # - Skip ALL other activities (including group_detected - exempted when stopped)
                if train_is_stopped:
                    # Log periodically that we're in stopped mode
                    if sample_idx % 100 == 0:
                        self.logger.debug(f"[{timestamp}] Train STOPPED - minimal person detection only")

                    # End any active activities since train is now stopped
                    for activity_name in self.activities:
                        if self.activities[activity_name]['active']:
                            self.end_activity(activity_name, timestamp, fps, frame_idx, people_count=1)
                        # Reset detection counters (except no_person_detected)
                        if activity_name != 'no_person_detected':
                            self.consecutive_detections[activity_name] = 0
                            self.grace_counters[activity_name] = 0

                    # Run MINIMAL person detection to check no_person_detected
                    detections = self.detect_objects(frame, None, use_pose_guided=False)

                    if len(detections['person']) > 0:
                        deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
                        person_count = len(deduplicated_persons)
                    else:
                        person_count = 0

                    # Only check no_person_detected (existing violation)
                    # group_detected is EXEMPTED when stopped
                    no_person_flag = (person_count == 0)

                    # Update no_person_detected activity tracking
                    if no_person_flag:
                        self.consecutive_detections['no_person_detected'] += 1
                        self.grace_counters['no_person_detected'] = 0

                        threshold = self.activity_thresholds['no_person_detected']
                        if self.consecutive_detections['no_person_detected'] >= threshold['required_consecutive']:
                            if not self.activities['no_person_detected']['active']:
                                self.start_activity('no_person_detected', timestamp, frame)
                                self.logger.info(f"[{timestamp}] NO PERSON detected (train stopped)")
                    else:
                        self.grace_counters['no_person_detected'] += 1
                        grace_frames = self.activity_thresholds['no_person_detected']['grace_frames']
                        if self.grace_counters['no_person_detected'] > grace_frames:
                            self.consecutive_detections['no_person_detected'] = 0
                            if self.activities['no_person_detected']['active']:
                                self.end_activity('no_person_detected', timestamp, fps, frame_idx, people_count=person_count)

                    continue  # Skip to next frame - no other activity checks needed

                # STEP 1: Run MediaPipe Face Mesh on full frame (for face-based sleep/EAR detection)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                
                # STEP 2: Detect objects with YOLO (without pose-guided detection yet)
                # We need person boxes first before we can do per-person pose detection
                detections = self.detect_objects(frame, None, use_pose_guided=False)
                
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
                    deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
                    
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
                            self.logger.debug(f"  Person {person_idx+1}: {role_info['role_name']} (LP score: {role_info['lp_score']}, ALP score: {role_info['alp_score']})")
                    
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
                
                # STEP 4: *** NEW MULTI-PERSON PROCESSING ***
                # Process ALL persons individually for ALL activities
                multi_person_results = self.process_all_persons_activities(
                    frame, detections, person_roles, timestamp_sec, face_results, frame_idx
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
                
                # Log detections for each person (only on first detection)
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
                        self.logger.info(f"[{timestamp}] MIND DIVERSION detected for {role_name} (Person {person_idx+1}) - Yaw={yaw:.1f}°, Pitch={pitch:.1f}° (method: {method})")
                    
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
                    
                    if activities.get('packing', False) and self.consecutive_detections.get('packing_bags', 0) == 0:
                        self.logger.info(f"[{timestamp}] PACKING detected for {role_name} (Person {person_idx+1})")

                # ALP STANDING CHECK: Check if ALP stands before train stops
                if self.train_state_detector is not None and train_state is not None:
                    alp_standing_violation = self.check_alp_standing_before_stop(
                        train_state=train_state,
                        train_state_changed=train_state_changed,
                        timestamp_sec=timestamp_sec,
                        persons_data=persons_data,
                        person_roles=person_roles,
                        frame_shape=frame.shape
                    )
                    if alp_standing_violation:
                        # Start ALP standing violation activity
                        if not self.activities['alp_not_standing_before_stop']['active']:
                            self.start_activity('alp_not_standing_before_stop', timestamp, frame)
                        # Immediately end it (one-time violation per stop)
                        self.end_activity('alp_not_standing_before_stop', timestamp, fps, frame_idx, people_count=people_count)

                # Face-based sleep detection (still use EAR as additional signal)
                if face_results.multi_face_landmarks and ear_value is not None:
                    if ear_value < 0.2:
                        if self.eye_closure_start is None:
                            self.eye_closure_start = timestamp_sec

                        self.eye_closure_duration = timestamp_sec - self.eye_closure_start

                        # Merge with pose-based detection
                        if self.eye_closure_duration >= 30:
                            sleep_detected = True
                        elif self.eye_closure_duration >= 5:
                            microsleep_detected = True
                    else:
                        self.eye_closure_start = None
                        self.eye_closure_duration = 0
                
                # DEPRECATED: Old single-person detection code removed
                # The new process_all_persons_activities() method above handles all detections
                
                # CRITICAL: Exclude sleep detection if person is holding objects or in active posture
                # If someone has a phone, book, or backpack in hand, they're clearly NOT sleeping
                if cell_phone_detected or writing_detected or packing_detected:
                    if microsleep_detected or sleep_detected:
                        reason = []
                        if cell_phone_detected: reason.append("phone")
                        if writing_detected: reason.append("book")
                        if packing_detected: reason.append("backpack")
                        self.logger.debug(f"[{timestamp}] Sleep detection OVERRIDDEN - person active ({', '.join(reason)})")
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
                # NEW: Draw MediaPipe outputs for ALL persons (not just one)
                annotated_frame_for_activity = self.draw_multi_person_mediapipe_outputs(
                    annotated_frame_for_activity,
                    persons_data,  # All persons' pose landmarks and activities
                    face_results,
                    ear_value,
                    self.eye_closure_duration
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
                    'no_person_detected': no_person_detected_flag
                }

                # Note: Train stopped check with `continue` happens earlier - if we reach here, train is moving

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
                                # Store raw frame (without annotations) for clean evidence clips
                                self.activities[activity_name]['frames'].append(frame.copy())
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
                                    self.end_activity(activity_name, timestamp, fps, frame_idx, people_count)
                                self.consecutive_detections[activity_name] = 0
                                self.grace_counters[activity_name] = 0
                        else:
                            # Reset grace counter if nothing is being tracked
                            self.grace_counters[activity_name] = 0
                
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
                continue
            finally:
                # ✅ MEMORY FIX: Explicitly delete frame after processing to free memory
                del frame
                del annotated_frame_for_activity
                if 'rgb_frame' in locals():
                    del rgb_frame
        
        # Finalize any pending hand coordination session at video end
        lp_not_coordinating, alp_not_coordinating = self._finalize_coordination_session(timestamp_sec)
        if lp_not_coordinating:
            # Record LP coordination failure if session ended with violation
            if not self.activities['lp_hand_gesture']['active']:
                self.start_activity('lp_hand_gesture', str(timedelta(seconds=timestamp_sec)), fps, frame_idx, person_roles=None)
        if alp_not_coordinating:
            # Record ALP coordination failure if session ended with violation
            if not self.activities['alp_hand_gesture']['active']:
                self.start_activity('alp_hand_gesture', str(timedelta(seconds=timestamp_sec)), fps, frame_idx, person_roles=None)

        # Finalize train state detection and log stopped periods
        if self.train_state_detector is not None:
            self.train_state_detector.finalize(timestamp_sec)
            stopped_periods = self.train_state_detector.get_stopped_periods()
            if stopped_periods:
                self.logger.info("-" * 40)
                self.logger.info("[TRAIN STATE DETECTION SUMMARY]")
                self.logger.info(f"  Total stopped periods: {len(stopped_periods)}")
                for i, (start, end) in enumerate(stopped_periods, 1):
                    duration = (end - start) if end else "ongoing"
                    end_str = f"{end:.2f}s" if end else "video end"
                    self.logger.info(f"  Period {i}: {start:.2f}s - {end_str} (duration: {duration}s)")

        # End any remaining active activities
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, people_count=1)  # Default to 1 person for final activities
        
        # ✅ MEMORY FIX: Clear frame buffers and activity frames to free memory
        self.frame_buffer.clear()
        for activity_name in self.activities:
            if 'frames' in self.activities[activity_name]:
                self.activities[activity_name]['frames'].clear()
        
        # ✅ MEMORY FIX: Force garbage collection
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
        # ✅ MEMORY FIX: Use context manager to ensure video capture is released
        with video_capture_context(self.video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Processing frame range {start_frame}-{end_frame} (worker {os.getpid()})")

        # =========================================================================
        # GPU BATCH OPTIMIZATION: Collect frames, run batch inference, then process
        # =========================================================================

        # Get batch settings from instance variables
        batch_size = self.gpu_batch_size
        batch_enabled = self.gpu_batch_enabled

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
            self.logger.debug(f"[GPU BATCH] Running batch object detection on {len(frames_only)} frames")
            batch_object_detections = self.detect_objects_batch(frames_only, batch_size)

            # Run batch pose detection
            self.logger.debug(f"[GPU BATCH] Running batch pose detection on {len(frames_only)} frames")
            batch_pose_results = self.detect_poses_batch(frames_only, batch_size)

            self.logger.info(f"[GPU BATCH] Batch inference complete: {len(batch_object_detections)} object results, {len(batch_pose_results)} pose results")

            # Release frames_only to free memory after batch inference
            del frames_only
            gc.collect()
        else:
            # Fallback: No batching (will compute per-frame)
            batch_object_detections = None
            batch_pose_results = None

        # PHASE 4: Sequential per-frame activity processing with pre-computed detections
        sampled_count = 0
        timestamp_sec = 0
        frame_idx = start_frame

        for idx, (frame, frame_idx, timestamp_sec, sample_idx) in enumerate(frames_data):
            sampled_count += 1

            try:
                # Convert timestamp to HH:MM:SS format
                timestamp = str(timedelta(seconds=timestamp_sec))

                # Add frame to buffer
                self.frame_buffer.append(frame.copy())

                # TRAIN STATE DETECTION: Analyze if train is stopped or moving
                # Uses ROI-based optical flow on window regions (excludes cabin interior)
                train_is_stopped = False
                train_state = None
                train_state_changed = False
                if self.train_state_detector is not None:
                    train_state = self.train_state_detector.analyze_frame(frame, timestamp_sec)
                    train_is_stopped = self.train_state_detector.is_stopped()
                    train_state_changed = self.train_state_detector.state_changed()

                    # Log state changes
                    if train_state_changed:
                        state_name = train_state.name if hasattr(train_state, 'name') else str(train_state)
                        self.logger.info(f"[{timestamp}] Train state changed to: {state_name}")

                # WHEN STOPPED: Run minimal person detection only
                # - Check no_person_detected violation (0 persons = violation)
                # - Skip ALL other activities (including group_detected - exempted when stopped)
                if train_is_stopped:
                    # Log periodically that we're in stopped mode
                    if sample_idx % 100 == 0:
                        self.logger.debug(f"[{timestamp}] Train STOPPED - minimal person detection only")

                    # End any active activities since train is now stopped
                    for activity_name in self.activities:
                        if self.activities[activity_name]['active']:
                            self.end_activity(activity_name, timestamp, fps, frame_idx, people_count=1, save_clips=save_clips)
                        # Reset detection counters (except no_person_detected)
                        if activity_name != 'no_person_detected':
                            self.consecutive_detections[activity_name] = 0
                            self.grace_counters[activity_name] = 0

                    # Use pre-computed batch detections if available, otherwise run detection
                    if batch_object_detections is not None and idx < len(batch_object_detections):
                        detections = batch_object_detections[idx]
                    else:
                        detections = self.detect_objects(frame, None, use_pose_guided=False)

                    if len(detections['person']) > 0:
                        deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
                        person_count = len(deduplicated_persons)
                    else:
                        person_count = 0

                    # Only check no_person_detected (existing violation)
                    # group_detected is EXEMPTED when stopped
                    no_person_flag = (person_count == 0)

                    # Update no_person_detected activity tracking
                    if no_person_flag:
                        self.consecutive_detections['no_person_detected'] += 1
                        self.grace_counters['no_person_detected'] = 0

                        threshold = self.activity_thresholds['no_person_detected']
                        if self.consecutive_detections['no_person_detected'] >= threshold['required_consecutive']:
                            if not self.activities['no_person_detected']['active']:
                                self.start_activity('no_person_detected', timestamp, frame)
                                self.logger.info(f"[{timestamp}] NO PERSON detected (train stopped)")
                    else:
                        self.grace_counters['no_person_detected'] += 1
                        grace_frames = self.activity_thresholds['no_person_detected']['grace_frames']
                        if self.grace_counters['no_person_detected'] > grace_frames:
                            self.consecutive_detections['no_person_detected'] = 0
                            if self.activities['no_person_detected']['active']:
                                self.end_activity('no_person_detected', timestamp, fps, frame_idx, people_count=person_count, save_clips=save_clips)

                    continue  # Skip to next frame - no other activity checks needed

                # STEP 1: Run MediaPipe Face Mesh on full frame (for face-based sleep/EAR detection)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                # STEP 2: Detect objects with YOLO (use pre-computed batch results if available)
                # GPU BATCH: Use pre-computed detections from batch inference
                if batch_object_detections is not None and idx < len(batch_object_detections):
                    detections = batch_object_detections[idx]
                else:
                    # Fallback: Run per-frame detection
                    detections = self.detect_objects(frame, None, use_pose_guided=False)

                # STEP 3: Identify person roles and count people
                people_count = len(detections['person'])
                if people_count == 0:
                    people_count = 1

                # De-duplicate person boxes and identify roles
                group_detected_flag = False
                person_roles = {}
                
                if len(detections['person']) > 0:
                    deduplicated_persons = self.deduplicate_person_boxes(detections['person'], iou_threshold=0.5)
                    
                    # Store deduplicated boxes (no pose validation - it filters out legitimate people)
                    deduplicated_count = len(deduplicated_persons)
                    detections['deduplicated_person'] = deduplicated_persons
                    person_roles = self.identify_person_roles(frame, deduplicated_persons, detections)
                    
                    if deduplicated_count > 2:
                        # Stage 2: Voting verification for group_detected (if enabled)
                        if self.voting_service is not None:
                            is_confirmed, vote_details = self.voting_service.verify_activity(
                                video_path=self.current_video_path,
                                timestamp_sec=timestamp_sec,
                                activity_type='group_detected',
                                person_bbox=[0, 0, frame.shape[1], frame.shape[0]]
                            )
                            if is_confirmed:
                                group_detected_flag = True
                                self.logger.info(f"[VOTING] group_detected CONFIRMED: {vote_details.get('vote_breakdown', [])}")
                            else:
                                group_detected_flag = False
                                self.logger.info(f"[VOTING] group_detected REJECTED: {vote_details.get('vote_breakdown', [])}")
                        else:
                            group_detected_flag = True
                else:
                    detections['deduplicated_person'] = []
                    person_roles = {}

                # STEP 4: *** NEW MULTI-PERSON PROCESSING ***
                # Process ALL persons individually for ALL activities
                # GPU BATCH: Pass pre-computed pose results if available
                precomputed_poses = None
                if batch_pose_results is not None and idx < len(batch_pose_results):
                    precomputed_poses = batch_pose_results[idx]

                multi_person_results = self.process_all_persons_activities(
                    frame, detections, person_roles, timestamp_sec, face_results, frame_idx,
                    precomputed_pose_results=precomputed_poses
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

                # ALP STANDING CHECK: Check if ALP stands before train stops
                if self.train_state_detector is not None and train_state is not None:
                    alp_standing_violation = self.check_alp_standing_before_stop(
                        train_state=train_state,
                        train_state_changed=train_state_changed,
                        timestamp_sec=timestamp_sec,
                        persons_data=persons_data,
                        person_roles=person_roles,
                        frame_shape=frame.shape
                    )
                    if alp_standing_violation:
                        # Start ALP standing violation activity
                        if not self.activities['alp_not_standing_before_stop']['active']:
                            self.start_activity('alp_not_standing_before_stop', timestamp, frame)
                        # Immediately end it (one-time violation per stop)
                        self.end_activity('alp_not_standing_before_stop', timestamp, fps, frame_idx, people_count=people_count, save_clips=save_clips)

                # Face-based sleep detection (still use EAR as additional signal)
                if face_results.multi_face_landmarks and ear_value is not None:
                    if ear_value < 0.2:
                        if self.eye_closure_start is None:
                            self.eye_closure_start = timestamp_sec

                        self.eye_closure_duration = timestamp_sec - self.eye_closure_start

                        # Merge with pose-based detection
                        if self.eye_closure_duration >= 30:
                            sleep_detected = True
                        elif self.eye_closure_duration >= 5:
                            microsleep_detected = True
                    else:
                        self.eye_closure_start = None
                        self.eye_closure_duration = 0

                # DEPRECATED: Old single-person detection code removed
                # The new process_all_persons_activities() method above handles all detections
                
                # Exclude sleep detection if person is holding objects or in active posture
                if cell_phone_detected or writing_detected or packing_detected:
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
                # NEW: Draw MediaPipe outputs for ALL persons (not just one)
                annotated_frame_for_activity = self.draw_multi_person_mediapipe_outputs(
                    annotated_frame_for_activity,
                    persons_data,  # All persons' pose landmarks and activities
                    face_results,
                    ear_value,
                    self.eye_closure_duration
                )
                
                # Save annotated frames periodically if enabled (in process_video_range for multiprocessing)
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
                #
                # NOTE: With 15s chunk duration (> 10s coordination timeout), session-based
                # tracking works reliably as most coordination events fit within a single chunk.
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
                    'no_person_detected': no_person_detected_flag
                }

                # Note: Train stopped check with `continue` happens earlier - if we reach here, train is moving

                for activity_name, detected in activities_map.items():
                    if detected:
                        self.consecutive_detections[activity_name] += 1
                        self.grace_counters[activity_name] = 0

                        required_consecutive = self.activity_thresholds[activity_name]['required_consecutive']

                        if self.consecutive_detections[activity_name] >= required_consecutive:
                            if not self.activities[activity_name]['active']:
                                self.start_activity(activity_name, timestamp, fps, frame_idx, person_roles=person_roles)

                            if self.activities[activity_name]['active']:
                                # Store raw frame (without annotations) for clean evidence clips
                                self.activities[activity_name]['frames'].append(frame.copy())
                                self.activities[activity_name]['last_frame_count'] = frame_idx
                                self.activities[activity_name]['last_detected_frame'] = frame_idx
                                self.activities[activity_name]['last_detection_time'] = timestamp  # Track for precise clip duration
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
                self.logger.error(f"Error processing sample {sample_idx} (frame {frame_idx}): {e}")
                continue
            finally:
                # ✅ MEMORY FIX: Explicitly delete frame after processing to free memory
                if 'frame' in locals():
                    del frame
                if 'annotated_frame_for_activity' in locals():
                    del annotated_frame_for_activity
                if 'rgb_frame' in locals():
                    del rgb_frame

        # Guard against empty frame ranges where timestamp_sec/frame_idx may not be set
        if sampled_count == 0:
            timestamp_sec = start_frame / fps if fps > 0 else 0.0
            frame_idx = start_frame
            self.logger.warning(f"No frames sampled in range {start_frame}-{end_frame}, skipping activity finalization")
            return self.all_activities

        # NOTE: Do NOT finalize coordination sessions in process_video_range()
        # This method processes small chunks (~5s) in parallel, and session state
        # doesn't carry across chunks. Coordination checking requires full video context.
        # Coordination failures are detected during chunk processing via the session
        # timeout mechanism, not at chunk boundaries. The _finalize_coordination_session()
        # is only appropriate for process_video() which processes the entire video sequentially.

        # Finalize train state detection for this chunk
        # Note: Unlike process_video(), we don't log a summary since this is a partial view
        if self.train_state_detector is not None:
            self.train_state_detector.finalize(timestamp_sec)

        # End any remaining active activities
        final_timestamp = str(timedelta(seconds=timestamp_sec))
        for activity_name in self.activities:
            if self.activities[activity_name]['active']:
                self.end_activity(activity_name, final_timestamp, fps, frame_idx, people_count=1, save_clips=save_clips)

        # ✅ MEMORY FIX: Clear frame buffers and activity frames to free memory
        self.frame_buffer.clear()
        for activity_name in self.activities:
            if 'frames' in self.activities[activity_name]:
                self.activities[activity_name]['frames'].clear()

        # ✅ MEMORY FIX: Force garbage collection
        gc.collect()

        self.logger.info(f"Frame range {start_frame}-{end_frame} completed: {len(self.all_activities)} activities")
        
        # Return detected activities (without generating summary reports)
        return self.all_activities

    def cleanup(self):
        """
        ✅ MEMORY FIX: Cleanup method to release model resources

        This method mirrors POC_2's MediaPipeService.close() pattern.
        Call this after processing to free GPU/CPU resources.
        
        NOTE: If models were pre-loaded (worker pool), they are NOT closed
        since they are shared across tasks in the same worker.
        """
        try:
            # Only close models if they were loaded fresh (not pre-loaded from worker pool)
            if not getattr(self, '_models_preloaded', False):
                # Close YOLOv8-Pose model
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

            # Clear activity frames (always)
            if hasattr(self, 'activities'):
                for activity_name in self.activities:
                    if 'frames' in self.activities[activity_name]:
                        self.activities[activity_name]['frames'].clear()

            # Reset train state detector (always)
            if hasattr(self, 'train_state_detector') and self.train_state_detector is not None:
                self.train_state_detector.reset()

            # Force garbage collection
            gc.collect()

            self.logger.info("Cleanup completed: Models closed, buffers cleared")
        except Exception as e:
            self.logger.warning(f"Warning during cleanup: {e}")
    
    def __del__(self):
        """
        ✅ MEMORY FIX: Destructor to ensure cleanup on object deletion
        """
        try:
            self.cleanup()
        except Exception:
            pass
    
    def generate_summary_report(self):
        """Generate activities.json in the run directory"""
        # Save the activities array in the run directory
        activities_json_path = os.path.join(self.run_dir, "activities.json")
        with open(activities_json_path, 'w') as f:
            json.dump(self.all_activities, f, indent=2)
        
        self.logger.info(f"Activities JSON saved: {activities_json_path}")
        self.logger.info(f"Total activities detected: {len(self.all_activities)}")
        
        # Count and log activity breakdown
        activities_by_type = {}
        for activity in self.all_activities:
            activity_type = activity['des']
            if activity_type not in activities_by_type:
                activities_by_type[activity_type] = 0
            activities_by_type[activity_type] += 1
        
        # Log activity breakdown
        if activities_by_type:
            self.logger.info("Activity Breakdown:")
            for activity_type, count in activities_by_type.items():
                self.logger.info(f"  - {activity_type}: {count}")


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