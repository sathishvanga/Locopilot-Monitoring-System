"""
Configuration management using Pydantic Settings
"""

import os
import sys
from pathlib import Path
from typing import Optional
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
    allowed_video_extensions: list[str] = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"],
        description="Allowed video file extensions"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path (auto-determined if None)")
    
    @field_validator('log_file', mode='before')
    @classmethod
    def set_log_file_path(cls, v: Optional[str]) -> str:
        """Set log file path to a user-accessible location"""
        if v:  # If explicitly set, use it
            return v
        
        # Determine log file location based on platform
        if sys.platform == 'win32':
            # Windows: Use Desktop or AppData
            desktop = Path.home() / 'Desktop'
            if desktop.exists():
                log_path = desktop / 'LocopilotCVVR.log'
            else:
                # Fallback to AppData
                appdata = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming'))
                log_path = appdata / 'LocopilotCVVR' / 'LocopilotCVVR.log'
        elif sys.platform == 'darwin':
            # macOS: Use Desktop
            log_path = Path.home() / 'Desktop' / 'LocopilotCVVR.log'
        else:
            # Linux: Use home directory
            log_path = Path.home() / 'LocopilotCVVR.log'
        
        return str(log_path)
    
    # UI Configuration
    window_width: int = Field(default=1200, description="Default window width")
    window_height: int = Field(default=800, description="Default window height")
    
    # Development mode
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Backend management
    auto_start_backend: bool = Field(default=True, description="Automatically start local backend on app launch")
    backend_startup_timeout: int = Field(default=10, description="Seconds to wait for backend startup")
    
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

