"""
Configuration management for the Locopilot Monitoring System

Uses environment variables with sensible defaults for production deployment.
"""

import os
import json
import tempfile
from typing import Optional, List
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Thread configuration for CPU inference optimization
# Allow worker processes to set optimal thread counts
# Conservative default of 2, worker initializer will override based on actual worker count
default_threads = 2  # Safe default, workers will set optimal value
os.environ.setdefault('OMP_NUM_THREADS', str(default_threads))
os.environ.setdefault('MKL_NUM_THREADS', str(default_threads))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(default_threads))

# Phase 3.5 Quick Win B: OpenCV threading for faster preprocessing (5-10% speedup)
import cv2
opencv_threads = int(os.getenv('OPENCV_THREADS', '4'))  # Match worker thread count
cv2.setNumThreads(opencv_threads)


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    
    All settings can be overridden via environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = "Locopilot Monitoring System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File upload settings
    max_upload_size: int = 5 * 1024 * 1024 * 1024  # 5 GB
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    # Use cross-platform temp directory (works on Windows, macOS, and Linux)
    upload_dir: str = os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "locopilot_uploads"))
    
    # Output settings
    # Convert to absolute path to avoid path resolution issues
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        "locopilot_evidence"
    )
    save_annotated_frames: bool = True
    frame_save_interval: int = 1
    
    # Video processing settings
    sample_fps: float = 0.5  # Sample at 0.5 FPS (1 frame every 2 seconds)
    
    # Multiprocessing settings
    enable_multiprocessing: bool = bool(int(os.getenv("ENABLE_MULTIPROCESSING", "1")))
    # ✅ 15s chunks ensure hand gesture coordination detection works correctly
    # Coordination window is 10s, so 15s chunks capture full coordination sequences
    # Tradeoff: fewer chunks (~118 for 30-min video) but reliable coordination detection
    mp_chunk_duration: float = 15.0  # Chunk duration in seconds (optimized for coordination detection)
    mp_max_workers: Optional[int] = None  # None = auto-detect (uses min(CPU count, max_workers_cap))
    mp_max_workers_cap: int = 12  # Maximum number of workers (11 cores + slight oversubscription)
    
    # Model settings - YOLO11 nano (fast scanning, VLM handles verification accuracy)
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11n.pt")  # YOLO11n nano for fast object detection
    yolo_pose_weights: str = os.getenv("YOLO_POSE_WEIGHTS", "yolo11n-pose.pt")  # YOLO11n-pose nano for fast pose
    yolo_pose_confidence: float = float(os.getenv("YOLO_POSE_CONFIDENCE", "0.35"))  # Lowered for nano (VLM filters FPs)

    # Phase 2: Inference optimization settings
    # CHANGED from 416 to 640 for better accuracy on small objects (cell phones)
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "640"))  # Model input size (640 for better small object detection)
    yolo_device: str = "cpu"  # Device for YOLO inference (cpu, 0 for GPU)

    # GPU Settings - Enable GPU acceleration for video processing
    gpu_enabled: bool = bool(int(os.getenv("GPU_ENABLED", "1")))  # Enable GPU if available
    gpu_device: str = os.getenv("GPU_DEVICE", "cuda:0")  # CUDA device identifier
    gpu_memory_fraction: float = float(os.getenv("GPU_MEMORY_FRACTION", "0.85"))  # Max GPU memory to use (85%)

    # Concurrency Settings - Control parallel video processing
    max_concurrent_videos: int = int(os.getenv("MAX_CONCURRENT_VIDEOS", "3"))  # Max videos processed simultaneously
    inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))  # Frames per inference batch
    job_queue_max_size: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "10"))  # Max pending jobs in queue

    # Memory Management - OOM recovery and CUDA allocator settings
    pytorch_cuda_alloc_conf: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    oom_retry_enabled: bool = bool(int(os.getenv("OOM_RETRY_ENABLED", "1")))  # Enable OOM recovery

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = os.getenv("LOG_DIR", "logs")  # Directory for log files
    environment: str = os.getenv("ENVIRONMENT", "development")  # production or development
    prod_log_level: str = os.getenv("PROD_LOG_LEVEL", "INFO")
    dev_log_level: str = os.getenv("DEV_LOG_LEVEL", "DEBUG")
    
    # External API settings (CVVR API)
    # Use {division} placeholder in URL - will be replaced with actual division value at runtime
    cvvr_api_url: str = os.getenv(
        "CVVR_API_URL",
        "https://api.mindcoinapps.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulk"
    )
    cvvr_api_url_no_events: str = os.getenv(
        "CVVR_API_URL_NO_EVENTS",
        "https://api.mindcoinapps.com/ai_{division}_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"
    )
    cvvr_api_default_division: str = os.getenv("CVVR_API_DEFAULT_DIVISION", "ai_demo_api")
    cvvr_api_token: Optional[str] = os.getenv("CVVR_API_TOKEN", None)
    cvvr_api_timeout: int = int(os.getenv("CVVR_API_TIMEOUT", "30"))
    cvvr_api_enabled: bool = bool(int(os.getenv("CVVR_API_ENABLED", "1")))  # Enable by default
    host_url: str = os.getenv("HOST_URL", "http://103.195.244.66:8000")  # URL for building fileUrl

    # MinIO settings for video downloads
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "mind.snikbtel.uk:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "admin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "login123")
    minio_secure: bool = bool(int(os.getenv("MINIO_SECURE", "1")))
    minio_bucket: str = os.getenv("MINIO_BUCKET", "cvss")
    
    # Image preprocessing settings (for MediaPipe detection enhancement)
    enable_image_preprocessing: bool = bool(int(os.getenv("ENABLE_IMAGE_PREPROCESSING", "1")))  # Enable by default
    use_clahe: bool = bool(int(os.getenv("USE_CLAHE", "1")))  # CLAHE is most effective
    use_gamma_correction: bool = bool(int(os.getenv("USE_GAMMA_CORRECTION", "1")))
    use_unsharp_masking: bool = bool(int(os.getenv("USE_UNSHARP_MASKING", "0")))  # Optional, can add artifacts
    use_noise_reduction: bool = bool(int(os.getenv("USE_NOISE_REDUCTION", "1")))
    adaptive_preprocessing: bool = bool(int(os.getenv("ADAPTIVE_PREPROCESSING", "1")))  # Use quality metrics
    clahe_clip_limit: float = float(os.getenv("CLAHE_CLIP_LIMIT", "1.5"))  # REDUCED from 2.0 (less aggressive CLAHE)
    
    # Parse tile grid size from environment variable (JSON array string)
    @field_validator('clahe_tile_grid_size', mode='before')
    @classmethod
    def parse_tile_grid_size(cls, v):
        """Parse tile grid size from JSON string or return default"""
        if v is None:
            return [8, 8]
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list) and len(parsed) == 2:
                    return [int(parsed[0]), int(parsed[1])]
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass
            return [8, 8]
        if isinstance(v, list) and len(v) == 2:
            return [int(v[0]), int(v[1])]
        return [8, 8]
    
    clahe_tile_grid_size: List[int] = json.loads(os.getenv("CLAHE_TILE_GRID_SIZE", "[16, 16]"))  # INCREASED from [8,8] (larger tiles, smoother)
    gamma_value: float = float(os.getenv("GAMMA_VALUE", "1.2"))
    unsharp_strength: float = float(os.getenv("UNSHARP_STRENGTH", "1.5"))
    unsharp_radius: int = int(os.getenv("UNSHARP_RADIUS", "1"))
    noise_reduction_kernel: int = int(os.getenv("NOISE_REDUCTION_KERNEL", "3"))

    # Concurrent activity grouping settings
    # Group overlapping activities of different types into combined records with arrays
    concurrent_grouping_enabled: bool = bool(int(os.getenv("CONCURRENT_GROUPING_ENABLED", "1")))

    # Voting verification settings
    # Two-stage detection: when activity detected, verify with multiple native frames
    voting_enabled: bool = bool(int(os.getenv("VOTING_ENABLED", "1")))
    voting_num_frames: int = int(os.getenv("VOTING_NUM_FRAMES", "10"))
    voting_frame_spread_ms: int = int(os.getenv("VOTING_FRAME_SPREAD_MS", "400"))  # 400ms window at 25fps

    # Per-activity voting thresholds (percentage of frames required for confirmation)
    # Default 50% (5/10 frames must detect the activity)
    voting_threshold_cell_phone: float = float(os.getenv("VOTING_THRESHOLD_CELL_PHONE", "0.5"))
    voting_threshold_writing: float = float(os.getenv("VOTING_THRESHOLD_WRITING", "0.5"))
    voting_threshold_packing_bags: float = float(os.getenv("VOTING_THRESHOLD_PACKING_BAGS", "0.75"))  # 75% - stricter for false positive reduction (was 60%)
    voting_threshold_lp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_LP_GESTURE", "0.6"))
    voting_threshold_alp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_ALP_GESTURE", "0.6"))
    voting_threshold_mind_diversion: float = float(os.getenv("VOTING_THRESHOLD_MIND_DIVERSION", "0.5"))
    voting_threshold_eating_drinking: float = float(os.getenv("VOTING_THRESHOLD_EATING_DRINKING", "0.4"))  # Lower threshold - cups harder to detect in IR
    voting_threshold_group_detected: float = float(os.getenv("VOTING_THRESHOLD_GROUP", "0.5"))

    # Hand Gesture Coordination Session Settings
    # Session-based tracking: session starts on first raise, ends after timeout with no activity
    # Violation only if one person NEVER raised during the entire session
    hand_gesture_session_timeout: float = float(os.getenv("HAND_GESTURE_SESSION_TIMEOUT", "10.0"))

    # Voting debug settings - save annotated frames for troubleshooting
    voting_save_debug_frames: bool = bool(int(os.getenv("VOTING_SAVE_DEBUG_FRAMES", "0")))  # Disabled by default (enable locally for debugging)
    voting_debug_frames_dir: str = os.getenv("VOTING_DEBUG_FRAMES_DIR", "voting_debug_frames")

    # Packing bags verification thresholds (stricter than initial detection)
    # TUNED 2026-01-21: Stricter thresholds to reduce false positives from bags on floor near seated crew
    packing_wrist_visibility_threshold: float = float(os.getenv("PACKING_WRIST_VIS", "0.5"))  # Min wrist visibility 50% (was 40%)
    packing_voting_margin: int = int(os.getenv("PACKING_VOTING_MARGIN", "0"))  # No margin - wrist must be truly inside (was 30)
    packing_max_distance_ratio: float = float(os.getenv("PACKING_MAX_DIST_RATIO", "0.30"))  # Wrist must be within 30% of bag diagonal from center (was 0.45)
    packing_min_bag_area: int = int(os.getenv("PACKING_MIN_BAG_AREA", "25000"))  # Min bag area 25,000 sq pixels (was 20000)
    packing_require_wrist_truly_inside: bool = bool(int(os.getenv("PACKING_STRICT_INSIDE", "1")))  # Require wrist truly inside bbox (no margin)

    # Clip duration settings - Precise clip extraction matching actual activity duration
    clip_buffer_before: float = float(os.getenv("CLIP_BUFFER_BEFORE", "1.0"))  # Seconds before activity start
    clip_buffer_after: float = float(os.getenv("CLIP_BUFFER_AFTER", "1.0"))    # Seconds after activity end

    # Mind Diversion Detection Thresholds
    # Three sub-types: looking_sideways, looking_down_distracted, looking_away_combined
    # TUNED 2026-01-21 v2: Further increased thresholds - camera angle makes forward-looking appear as ~60-70° yaw
    mind_diversion_yaw_sideways: float = float(os.getenv("MIND_DIV_YAW_SIDEWAYS", "78"))  # Head turned > 78° for sideways (was 68°, orig 55°)
    mind_diversion_yaw_combined: float = float(os.getenv("MIND_DIV_YAW_COMBINED", "58"))  # Yaw threshold for combined detection (was 50°, orig 40°)
    mind_diversion_pitch_down: float = float(os.getenv("MIND_DIV_PITCH_DOWN", "45"))  # Head down > 45° for looking_down (was 40°, orig 30°)
    mind_diversion_pitch_combined: float = float(os.getenv("MIND_DIV_PITCH_COMBINED", "35"))  # Pitch threshold for combined (was 28°, orig 20°)
    mind_diversion_yaw_max_for_down: float = float(os.getenv("MIND_DIV_YAW_MAX_DOWN", "55"))  # Max yaw for pure looking_down (was 50°, orig 40°)

    # Forward-looking exemption: Camera is behind-right of crew, so:
    # - Negative yaw = looking LEFT toward track/window (LEGITIMATE WORK)
    # - Positive yaw = looking RIGHT away from track (POTENTIAL DIVERSION)
    # When enabled, only positive yaw triggers mind diversion (exempts looking at track)
    mind_diversion_exempt_forward_looking: bool = os.getenv("MIND_DIV_EXEMPT_FORWARD", "true").lower() == "true"

    # Sleep Detection - Reclined Posture Settings (overhead/behind camera angles)
    # When crew sleeps by leaning BACK against wall, torso elongates and nose moves higher in frame
    sleep_reclined_torso_height_threshold: float = float(os.getenv("SLEEP_TORSO_HEIGHT_THRESH", "175"))  # px, torso > this = reclined
    sleep_reclined_nose_y_norm_threshold: float = float(os.getenv("SLEEP_NOSE_Y_NORM_THRESH", "0.40"))  # normalized, nose_y < this = reclined (higher in frame)
    sleep_reclined_shoulder_width_threshold: float = float(os.getenv("SLEEP_SHOULDER_WIDTH_THRESH", "60"))  # px, shoulders < this = compressed (reclined)

    # No-pose sleep detection (for IR mode where YOLO pose fails)
    sleep_no_pose_enabled: bool = bool(int(os.getenv("SLEEP_NO_POSE_ENABLED", "1")))
    sleep_no_pose_min_duration: float = float(os.getenv("SLEEP_NO_POSE_MIN_DURATION", "30.0"))  # Seconds of stable no-pose person before flagging sleep
    sleep_no_pose_bbox_stability_threshold: float = float(os.getenv("SLEEP_NO_POSE_BBOX_STABILITY", "0.15"))  # IoU change threshold for bbox stability

    # Eating/Drinking Detection (mind diversion sub-type)
    eating_drinking_detection_enabled: bool = bool(int(os.getenv("EATING_DRINKING_ENABLED", "1")))
    eating_drinking_cup_confidence: float = float(os.getenv("EATING_DRINKING_CUP_CONF", "0.25"))  # Lower threshold for cups in IR
    eating_drinking_hand_face_margin: int = int(os.getenv("EATING_DRINKING_HAND_FACE_MARGIN", "80"))  # Wrist within 80px of nose height
    eating_drinking_hand_object_margin: int = int(os.getenv("EATING_DRINKING_HAND_OBJ_MARGIN", "150"))  # Hand-to-cup proximity
    eating_drinking_cup_floor_confidence: float = float(os.getenv("EATING_DRINKING_CUP_FLOOR_CONF", "0.20"))  # Pre-filter floor for cup/bottle in full-frame detection

    # Mind Diversion Suppression Settings
    # Suppress false positives when LP is doing legitimate document work
    mind_diversion_suppress_with_writing: bool = os.getenv("MIND_DIV_SUPPRESS_WRITING", "true").lower() == "true"
    mind_diversion_writing_grace_seconds: float = float(os.getenv("MIND_DIV_WRITING_GRACE", "5.0"))
    mind_diversion_wrist_distance_threshold: float = float(os.getenv("MIND_DIV_WRIST_DIST", "200"))  # Max wrist distance for writing pose

    # Job Queue Settings - Async video processing queue management
    # Note: max_concurrent_videos and job_queue_max_size defined in GPU Settings section above
    job_queue_num_workers: int = int(os.getenv("JOB_QUEUE_NUM_WORKERS", "3"))  # Number of worker tasks processing jobs

    # GPU Memory Management Settings
    gpu_memory_warning_threshold: float = float(os.getenv("GPU_MEMORY_WARNING_THRESHOLD", "80.0"))  # Percentage

    # ==========================================
    # VLM Verification Settings (Qwen2.5-VL)
    # ==========================================
    # Hybrid pipeline: YOLO nano fast-scan → VLM semantic verification
    vlm_enabled: bool = bool(int(os.getenv("VLM_ENABLED", "0")))  # Opt-in: set VLM_ENABLED=1 after installing transformers+autoawq
    vlm_model_name: str = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    vlm_max_new_tokens: int = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))
    vlm_num_verification_frames: int = int(os.getenv("VLM_NUM_VERIFICATION_FRAMES", "5"))  # Frames sent to VLM per verification

    # ==========================================
    # Train Motion Rules Settings
    # ==========================================
    # Enable/disable train motion-based rule engine
    train_motion_rules_enabled: bool = bool(int(os.getenv("TRAIN_MOTION_RULES_ENABLED", "0")))

    # Trip API Settings (RailRadar API)
    trip_api_url: str = os.getenv("TRIP_API_URL", "https://api.railradar.in/api/v1/trains")
    trip_api_timeout: int = int(os.getenv("TRIP_API_TIMEOUT", "10"))

    # OCR Timestamp Extraction Settings
    ocr_enabled: bool = bool(int(os.getenv("OCR_ENABLED", "0")))
    ocr_engine: str = os.getenv("OCR_ENGINE", "auto")  # 'easyocr' (recommended), 'tesseract', or 'auto'
    ocr_roi_position: str = os.getenv("OCR_ROI_POSITION", "top-left")  # top-right, top-left, bottom-right, bottom-left
    ocr_roi_x: int = int(os.getenv("OCR_ROI_X", "10"))  # X offset from edge
    ocr_roi_y: int = int(os.getenv("OCR_ROI_Y", "10"))  # Y offset from edge
    ocr_roi_width: int = int(os.getenv("OCR_ROI_WIDTH", "200"))  # ROI width
    ocr_roi_height: int = int(os.getenv("OCR_ROI_HEIGHT", "50"))  # ROI height

    # Pre-Arrival ALP Alertness Settings
    pre_arrival_window_start: int = int(os.getenv("PRE_ARRIVAL_WINDOW_START", "60"))  # 60s before arrival
    pre_arrival_window_end: int = int(os.getenv("PRE_ARRIVAL_WINDOW_END", "30"))  # 30s before arrival

    # Halt Grace Period - allow exemptions for short time after scheduled departure
    halt_grace_period: int = int(os.getenv("HALT_GRACE_PERIOD", "120"))  # 120s after departure

    # ==========================================
    # Optical Flow Motion Detection Settings
    # ==========================================
    # Enable/disable optical flow motion verification
    optical_flow_enabled: bool = bool(int(os.getenv("OPTICAL_FLOW_ENABLED", "0")))

    # Side window ROI configuration (ratios of frame dimensions)
    # Captures the narrow left side window/door opening for motion detection
    # IPCamera 02 cabin view: outside scenery visible through left side bars/window
    # Calibrated from "Writing 13 25.mp4" (1280x720, 25fps)
    motion_roi_x_ratio: float = float(os.getenv("MOTION_ROI_X", "0.0"))
    motion_roi_width_ratio: float = float(os.getenv("MOTION_ROI_WIDTH", "0.08"))
    motion_roi_y_ratio: float = float(os.getenv("MOTION_ROI_Y", "0.15"))
    motion_roi_height_ratio: float = float(os.getenv("MOTION_ROI_HEIGHT", "0.50"))

    # Motion classification thresholds (optical flow magnitude - 90th percentile)
    # Calibrated from video analysis (consecutive frames at 25fps):
    #   - STOPPED: p90 magnitude typically 0.13-0.14
    #   - RUNNING: p90 magnitude typically 1.0-2.4
    #   - TRANSITION: p90 magnitude 0.5-0.7
    motion_stopped_threshold: float = float(os.getenv("MOTION_STOPPED_THRESHOLD", "0.3"))
    motion_running_threshold: float = float(os.getenv("MOTION_RUNNING_THRESHOLD", "0.8"))
    motion_confidence_threshold: float = float(os.getenv("MOTION_CONFIDENCE_THRESHOLD", "0.7"))

    # ==========================================
    # etrain.info Delay Integration Settings
    # ==========================================
    # Enable/disable etrain.info delay data fetching
    etrain_enabled: bool = bool(int(os.getenv("ETRAIN_ENABLED", "0")))

    # etrain.info base URL for train live status
    etrain_base_url: str = os.getenv("ETRAIN_BASE_URL", "https://etrain.info/train")

    # Cache TTL for delay data (in seconds, default 30 minutes)
    etrain_cache_ttl: int = int(os.getenv("ETRAIN_CACHE_TTL", "1800"))


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()