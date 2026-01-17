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
    max_upload_size: int = 500 * 1024 * 1024  # 500 MB
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
    enable_multiprocessing: bool = True
    # ✅ PERFORMANCE BOOST: 6s chunks maximize parallelism with minimal overhead
    # 6s chunks: ~380 chunks, better load distribution across more workers
    # Smaller chunks = workers stay busy, no idle time waiting for long tasks
    mp_chunk_duration: float = 6.0  # Chunk duration in seconds (optimized for 11-core system)
    mp_max_workers: Optional[int] = None  # None = auto-detect (uses min(CPU count, max_workers_cap))
    mp_max_workers_cap: int = 12  # Maximum number of workers (11 cores + slight oversubscription)
    
    # Model settings - YOLO11 (latest, better accuracy, faster, fewer parameters)
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11m.pt")  # YOLO11m for object detection
    yolo_pose_weights: str = os.getenv("YOLO_POSE_WEIGHTS", "yolo11m-pose.pt")  # YOLO11m-pose for multi-person pose
    yolo_pose_confidence: float = float(os.getenv("YOLO_POSE_CONFIDENCE", "0.45"))  # Pose detection confidence

    # Detection-tier models (Stage 1 - fast, nano for quick scanning)
    yolo_detection_weights: str = os.getenv("YOLO_DETECTION_WEIGHTS", "yolo11n.pt")
    yolo_detection_pose_weights: str = os.getenv("YOLO_DETECTION_POSE_WEIGHTS", "yolo11n-pose.pt")

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
    cvvr_api_url: str = os.getenv(
        "CVVR_API_URL",
        "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulk"
    )
    cvvr_api_url_no_events: str = os.getenv(
        "CVVR_API_URL_NO_EVENTS",
        "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"
    )
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
    voting_threshold_packing_bags: float = float(os.getenv("VOTING_THRESHOLD_PACKING_BAGS", "0.6"))  # 60% - stricter for false positive reduction
    voting_threshold_lp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_LP_GESTURE", "0.5"))
    voting_threshold_alp_hand_gesture: float = float(os.getenv("VOTING_THRESHOLD_ALP_GESTURE", "0.5"))
    voting_threshold_mind_diversion: float = float(os.getenv("VOTING_THRESHOLD_MIND_DIVERSION", "0.5"))
    voting_threshold_group_detected: float = float(os.getenv("VOTING_THRESHOLD_GROUP", "0.5"))

    # Voting debug settings - save annotated frames for troubleshooting
    voting_save_debug_frames: bool = bool(int(os.getenv("VOTING_SAVE_DEBUG_FRAMES", "1")))  # Enable by default
    voting_debug_frames_dir: str = os.getenv("VOTING_DEBUG_FRAMES_DIR", "voting_debug_frames")

    # Packing bags verification thresholds (stricter than initial detection)
    packing_wrist_visibility_threshold: float = float(os.getenv("PACKING_WRIST_VIS", "0.4"))  # Min wrist visibility (40%)
    packing_voting_margin: int = int(os.getenv("PACKING_VOTING_MARGIN", "30"))  # Stricter bbox margin for voting
    packing_max_distance_ratio: float = float(os.getenv("PACKING_MAX_DIST_RATIO", "0.45"))  # Wrist must be within 45% of bag diagonal from center (reduced from 0.6)
    packing_min_bag_area: int = int(os.getenv("PACKING_MIN_BAG_AREA", "20000"))  # Min bag area 20,000 sq pixels (filters very small/spurious detections)
    packing_require_wrist_truly_inside: bool = bool(int(os.getenv("PACKING_STRICT_INSIDE", "1")))  # Require wrist truly inside bbox (no margin)

    # Clip duration settings - Precise clip extraction matching actual activity duration
    clip_buffer_before: float = float(os.getenv("CLIP_BUFFER_BEFORE", "1.0"))  # Seconds before activity start
    clip_buffer_after: float = float(os.getenv("CLIP_BUFFER_AFTER", "1.0"))    # Seconds after activity end

    # Mind Diversion Detection Thresholds
    # Three sub-types: looking_sideways, looking_down_distracted, looking_away_combined
    mind_diversion_yaw_sideways: float = float(os.getenv("MIND_DIV_YAW_SIDEWAYS", "55"))  # Head turned > 55° for sideways
    mind_diversion_yaw_combined: float = float(os.getenv("MIND_DIV_YAW_COMBINED", "40"))  # Yaw threshold for combined detection
    mind_diversion_pitch_down: float = float(os.getenv("MIND_DIV_PITCH_DOWN", "30"))  # Head down > 30° for looking_down
    mind_diversion_pitch_combined: float = float(os.getenv("MIND_DIV_PITCH_COMBINED", "20"))  # Pitch threshold for combined
    mind_diversion_yaw_max_for_down: float = float(os.getenv("MIND_DIV_YAW_MAX_DOWN", "40"))  # Max yaw for pure looking_down

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


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()