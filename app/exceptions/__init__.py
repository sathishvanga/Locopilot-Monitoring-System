"""
Custom exceptions for the Locopilot Monitoring System

This module provides domain-specific exception classes for better error handling
and more descriptive error messages throughout the application.
"""

from .video_exceptions import (
    VideoProcessingError,
    VideoValidationError,
    VideoNotFoundError,
    VideoUploadError,
)
from .api_exceptions import (
    APIError,
    ExternalAPIError,
    S3UploadError,
    InvalidRequestError,
)

__all__ = [
    "VideoProcessingError",
    "VideoValidationError",
    "VideoNotFoundError",
    "VideoUploadError",
    "APIError",
    "ExternalAPIError",
    "S3UploadError",
    "InvalidRequestError",
]

