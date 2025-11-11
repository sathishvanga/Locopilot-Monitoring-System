"""
Configuration management for the Locopilot Monitoring System

Uses environment variables with sensible defaults for production deployment.
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    
    All settings can be overridden via environment variables.
    """
    
    # Application settings
    app_name: str = "Locopilot Monitoring System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # File upload settings
    max_upload_size: int = 500 * 1024 * 1024  # 500 MB
    allowed_video_extensions: list = [".mp4", ".avi", ".mov", ".mkv"]
    upload_dir: str = "uploads"
    
    # Output settings
    output_dir: str = "locopilot_evidence"
    save_annotated_frames: bool = False
    frame_save_interval: int = 1
    
    # Video processing settings
    sample_fps: float = 0.5  # Sample at 0.5 FPS (1 frame every 2 seconds)
    
    # Multiprocessing settings
    enable_multiprocessing: bool = True
    mp_chunk_duration: float = 6.0  # Chunk duration in seconds
    mp_max_workers: int = 5  # 0 = auto-detect
    mp_max_workers_cap: int = 8  # Maximum number of workers
    
    # Model settings
    yolo_weights: str = os.getenv("YOLO_WEIGHTS_PRELOAD", "yolo11s.pt")
    preload_ocr: bool = bool(int(os.getenv("PRELOAD_OCR", "0")))
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS settings
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses LRU cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()

