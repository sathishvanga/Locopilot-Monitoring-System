"""
Logging utilities for the Locopilot Monitoring System

Provides centralized logging configuration with proper formatting.
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance
    
    Creates a logger with proper formatting and handlers. If the logger
    already exists, returns the existing instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set log level
        log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Create formatter
        default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(
            log_format or default_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def setup_logging(level: str = "INFO", log_format: Optional[str] = None) -> None:
    """
    Setup application-wide logging configuration
    
    Configures the root logger with proper formatting.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format or default_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress overly verbose third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)

