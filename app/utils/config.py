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
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress TensorFlow warnings


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
    max_upload_size: int = 1 * 1024 * 1024 * 1024  # 1 GB (upgraded from 500 MB)
    allowed_video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    # Use cross-platform temp directory (works on Windows, macOS, and Linux)
    upload_dir: str = os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "locopilot_uploads"))

    # Chunked upload settings (for resumable uploads)
    chunk_upload_dir: str = os.getenv(
        "CHUNK_UPLOAD_DIR",
        os.path.join(tempfile.gettempdir(), "locopilot_uploads_chunks")
    )
    chunk_size_default: int = 10 * 1024 * 1024  # 10 MB default chunk size
    chunk_session_ttl_hours: int = 24  # Auto-cleanup sessions after 24 hours
    disk_reserve_gb: int = 5  # Reserve 5GB minimum disk space
    stream_chunk_size: int = 1 * 1024 * 1024  # 1 MB read size for streaming uploads
    
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
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11s.pt")
    preload_ocr: bool = bool(int(os.getenv("PRELOAD_OCR", "0")))
    
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


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()