"""
Configuration management using Pydantic Settings
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


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
    allowed_video_extensions: List[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"],
        description="Allowed video file extensions"
    )
    
    @field_validator('local_backend_port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number"""
        if not (1 <= v <= 65535):
            raise ValueError("local_backend_port must be between 1 and 65535")
        return v
    
    @field_validator('max_file_size')
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Validate max file size is reasonable"""
        if v <= 0:
            raise ValueError("max_file_size must be positive")
        if v > 10 * 1024 * 1024 * 1024:  # 10 GB
            raise ValueError("max_file_size cannot exceed 10 GB")
        return v
    
    @field_validator('request_timeout', 'upload_timeout', 'processing_timeout')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout values"""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        if v > 86400:  # 24 hours
            raise ValueError("Timeout cannot exceed 24 hours")
        return v
    
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
    backend_startup_timeout: int = Field(default=30, description="Seconds to wait for backend startup (increased for ML model loading)")
    
    model_config = SettingsConfigDict(
        env_prefix="CVVR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


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

