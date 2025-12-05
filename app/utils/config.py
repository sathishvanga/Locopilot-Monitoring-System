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

# Suppress PyTorch/YOLO NNPACK warnings early (before torch imports)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


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
    
    # Model settings
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolov8m.pt")  # YOLOv8m for faster CPU
    yolo_pose_weights: str = os.getenv("YOLO_POSE_WEIGHTS", "yolov8m-pose.pt")  # YOLOv8m-pose for multi-person pose
    yolo_pose_confidence: float = float(os.getenv("YOLO_POSE_CONFIDENCE", "0.45"))  # Pose detection confidence
    preload_ocr: bool = bool(int(os.getenv("PRELOAD_OCR", "0")))

    # TIER 2 OPTIMIZATION: ONNX Runtime settings (3x faster CPU inference)
    use_onnx_runtime: bool = bool(int(os.getenv("USE_ONNX_RUNTIME", "1")))  # Enable by default
    onnx_yolo_weights: str = os.getenv("ONNX_YOLO_WEIGHTS", "yolov8m.onnx")
    onnx_yolo_pose_weights: str = os.getenv("ONNX_YOLO_POSE_WEIGHTS", "yolov8m-pose.onnx")

    # TIER 3 OPTIMIZATION: Motion-based frame skipping (30-50% frame reduction)
    enable_motion_skipping: bool = bool(int(os.getenv("ENABLE_MOTION_SKIPPING", "1")))  # Enable by default
    motion_threshold_low: float = float(os.getenv("MOTION_THRESHOLD_LOW", "0.005"))  # Skip if motion < 0.5%
    motion_threshold_high: float = float(os.getenv("MOTION_THRESHOLD_HIGH", "0.05"))  # Always process if > 5%
    baseline_frame_interval: int = int(os.getenv("BASELINE_FRAME_INTERVAL", "4"))  # Process every 4th frame minimum

    # TIER 4.1 OPTIMIZATION: INT8 Quantization (2-3x additional speedup)
    use_int8_quantization: bool = bool(int(os.getenv("USE_INT8_QUANTIZATION", "0")))  # Disabled by default (requires quantized models)
    int8_yolo_weights: str = os.getenv("INT8_YOLO_WEIGHTS", "yolov8m_int8.onnx")
    int8_yolo_pose_weights: str = os.getenv("INT8_YOLO_POSE_WEIGHTS", "yolov8m-pose_int8.onnx")

    # TIER 4.2 OPTIMIZATION: Async Frame Reader (20-30% I/O overlap)
    use_async_frame_reader: bool = bool(int(os.getenv("USE_ASYNC_FRAME_READER", "0")))  # Disabled by default
    async_buffer_size: int = int(os.getenv("ASYNC_BUFFER_SIZE", "15"))  # 10-20 recommended for optimal performance

    # PHASE 1.2 OPTIMIZATION: Frame Resolution Reduction for Detection (25-40% speedup)
    detection_width: int = int(os.getenv("DETECTION_WIDTH", "640"))  # Default 640px (reduced from 1280)
    detection_height: int = int(os.getenv("DETECTION_HEIGHT", "480"))  # Default 480px (reduced from 720)

    @property
    def detection_resolution(self) -> tuple:
        """Detection resolution as (width, height) tuple"""
        return (self.detection_width, self.detection_height)

    # PHASE 1.3 OPTIMIZATION: Pose Cache Expansion (10-15% reduction in pose detection calls)
    enable_pose_cache: bool = bool(int(os.getenv("ENABLE_POSE_CACHE", "1")))  # Enable by default
    pose_cache_duration: float = float(os.getenv("POSE_CACHE_DURATION", "1.0"))  # Cache for 1 second
    pose_cache_bbox_threshold: float = float(os.getenv("POSE_CACHE_BBOX_THRESHOLD", "0.1"))  # 10% movement threshold
    pose_cache_stats_interval: int = int(os.getenv("POSE_CACHE_STATS_INTERVAL", "100"))  # Log stats every N frames

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
    host_url: str = os.getenv("HOST_URL", "https://celebxmedia.info")  # URL for building fileUrl

    # Chunked upload settings
    chunk_size: int = 8 * 1024 * 1024  # Fixed 8 MB chunk size
    upload_session_timeout: int = 3600  # 1 hour in seconds
    max_upload_sessions: int = 100  # Prevent memory exhaustion
    chunks_cleanup_interval: int = 300  # 5 minutes
    
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


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()