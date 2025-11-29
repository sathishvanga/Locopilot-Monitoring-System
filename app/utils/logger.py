"""
Enhanced logging configuration with request context support

Provides structured logging with request tracking, file rotation,
and custom formatting for production environments.
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from .request_context import get_request_context
from .config import get_settings


settings = get_settings()

# Create logs directory if it doesn't exist
log_dir = settings.log_dir
os.makedirs(log_dir, exist_ok=True)


class RequestFormatter(logging.Formatter):
    """
    Custom formatter that includes request context in log messages
    
    Extracts context metadata (user_id, request_id, etc.) and includes
    it in formatted log output.
    """
    
    def format(self, record):
        """
        Format log record with request context
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        context = get_request_context()
        record.cookie_id = context.get("cookie_id", "N/A")
        record.user_id = context.get("user_id", "N/A")
        record.url = context.get("url", "N/A")
        record.method = context.get("method", "N/A")
        record.request_id = context.get("request_id", "N/A")
        record.source_request_id = context.get("source_request_id", "N/A")
        return super().format(record)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup application logging with file rotation and request tracking
    
    Configures:
    - Root logger with appropriate level
    - File handler with daily rotation (4-day retention)
    - Console handler for errors
    - Custom formatter with request context
    - Disables noisy third-party loggers
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              If None, uses settings based on environment
    """
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Determine log level based on environment
    if level:
        log_level = getattr(logging, level.upper())
    else:
        if settings.environment == "production":
            log_level = getattr(logging, settings.prod_log_level.upper())
        else:
            log_level = getattr(logging, settings.dev_log_level.upper())
    
    root_logger.setLevel(log_level)
    
    # Disable noisy third-party loggers
    noisy_loggers = [
        "httpcore.connection",
        "httpcore.http11",
        "openai._base_client",
        "langsmith",
        "langsmith._internal._serde",
        "urllib3.connectionpool",
        "langsmith.client",
        "asyncio",
        "multipart.multipart",
        "PIL.PngImagePlugin",
        "PIL.TiffImagePlugin"
    ]
    
    # Disable PyTorch/YOLO warnings (NNPACK, etc.)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
    warnings.filterwarnings('ignore', message='.*Error decoding JSON.*')
    warnings.filterwarnings('ignore', message='.*settings.json.*')
    warnings.filterwarnings('ignore', message='.*inference_feedback_manager.*')
    warnings.filterwarnings('ignore', message='.*landmark_projection_calculator.*')
    warnings.filterwarnings('ignore', message='.*NORM_RECT.*')
    warnings.filterwarnings('ignore', message='.*IMAGE_DIMENSIONS.*')
    
    import logging as std_logging
    std_logging.getLogger('ultralytics').setLevel(std_logging.ERROR)
    std_logging.getLogger('absl').setLevel(std_logging.ERROR)  # Suppress TensorFlow/MediaPipe absl warnings
    
    # Set environment variable to suppress NNPACK warnings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).disabled = True
    
    # Create log file path
    log_file_path = os.path.join(log_dir, "LocopilotMonitoring.log")
    
    # File handler with daily rotation (keeps 4 days of logs)
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",
        interval=1,
        backupCount=4,
        encoding="utf-8",
        utc=True,
    )
    
    # Only add handler if not already present (avoid duplicates)
    if not any(
        isinstance(h, TimedRotatingFileHandler) 
        and getattr(h, "baseFilename", None) == file_handler.baseFilename 
        for h in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)
    
    # Console handler for errors (production) or all messages (development)
    stream_handler = logging.StreamHandler()
    if settings.environment == "production":
        stream_handler.setLevel(logging.ERROR)
    else:
        stream_handler.setLevel(logging.DEBUG)
    
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(stream_handler)
    
    # Custom formatter with request context
    formatter = RequestFormatter(
        "%(asctime)s [%(user_id)s] [%(cookie_id)s] [%(source_request_id)s] "
        "[%(request_id)s] [%(levelname)s] [%(name)s] [%(method)s %(url)s] %(message)s"
    )
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}, "
                f"Environment: {settings.environment}, Log file: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
