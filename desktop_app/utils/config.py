"""
Configuration management using Pydantic Settings
"""

import multiprocessing
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """
    Application settings
    
    Loads configuration from environment variables with defaults
    """
    
    # Application info
    app_name: str = Field(default="Locopilot CVVR Desktop", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    
    # API Configuration
    api_base_url: str = Field(
        default="https://api.mindcoinapps.com/ai_demo_api",
        description="Remote API base URL"
    )
    local_backend_url: str = Field(
        default="http://localhost:8000",
        description="Local FastAPI backend URL"
    )
    local_backend_port: int = Field(
        default=8000,
        description="Local FastAPI backend port"
    )
    
    # Timeouts (in seconds)
    request_timeout: int = Field(default=30, description="HTTP request timeout")
    upload_timeout: int = Field(default=300, description="File upload timeout (5 minutes)")
    processing_timeout: int = Field(default=3600, description="Video processing timeout (1 hour)")
    
    # Retry configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts for failed requests")
    retry_delay: int = Field(default=2, description="Delay between retries in seconds")
    
    # Upload configuration
    upload_chunk_size: int = Field(default=8192, description="Upload chunk size in bytes")
    max_file_size: int = Field(default=2 * 1024 * 1024 * 1024, description="Max file size (2GB)")
    
    # Video file extensions
    allowed_video_extensions: list[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"],
        description="Allowed video file extensions"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="desktop_app.log", description="Log file path")
    
    # UI Configuration
    window_width: int = Field(default=1200, description="Default window width")
    window_height: int = Field(default=800, description="Default window height")
    
    # Development mode
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Backend management
    auto_start_backend: bool = Field(default=True, description="Automatically start local backend on app launch")
    backend_startup_timeout: int = Field(default=10, description="Seconds to wait for backend startup")
    
    # Backend worker configuration (for CPU utilization optimization)
    # These can be overridden via environment variables: CVVR_BACKEND_WORKERS, CVVR_BACKEND_THREADS
    backend_workers: Optional[int] = Field(
        default=None,
        description="Number of uvicorn workers (None = auto-detect based on CPU count)"
    )
    backend_threads: int = Field(
        default=1,
        description="Number of threads per worker (uvicorn supports this but typically use workers instead)"
    )
    backend_max_workers_cap: int = Field(
        default=4,
        description="Maximum number of backend workers (to prevent memory issues - each worker loads models)"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="CVVR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @computed_field
    @property
    def effective_backend_workers(self) -> int:
        """
        Calculate effective number of backend workers based on CPU count
        
        Similar to gunicorn_config.py logic but optimized for desktop app:
        - Desktop app can use more workers since it's single-user
        - Formula: min(cpu_count // 2, max_workers_cap) for better CPU utilization
        - Minimum 1 worker, maximum capped to prevent memory issues
        
        Returns:
            int: Number of workers to use
        """
        if self.backend_workers is not None:
            # User explicitly set workers
            return max(1, min(self.backend_workers, self.backend_max_workers_cap))
        
        # Auto-detect based on CPU count
        cpu_count = multiprocessing.cpu_count()
        # Desktop app: use cpu_count // 2 for better utilization (more aggressive than server)
        # But cap at max_workers_cap to prevent memory issues
        workers = max(1, min(cpu_count // 2, self.backend_max_workers_cap))
        return workers


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton)
    
    Returns:
        Settings: Application configuration
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

