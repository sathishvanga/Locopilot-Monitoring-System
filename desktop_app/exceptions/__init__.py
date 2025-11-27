"""
Custom exceptions for the Locopilot CVVR Desktop Application

This module provides domain-specific exception classes for better error handling
and more descriptive error messages throughout the application.
"""

from .auth_exceptions import (
    AuthenticationError,
    LoginError,
    InvalidCredentialsError,
    SessionExpiredError,
)
from .backend_exceptions import (
    BackendError,
    BackendStartupError,
    BackendNotRunningError,
    BackendHealthCheckError,
)
from .upload_exceptions import (
    UploadError,
    FileValidationError,
    UploadTimeoutError,
    UploadConnectionError,
)

__all__ = [
    "AuthenticationError",
    "LoginError",
    "InvalidCredentialsError",
    "SessionExpiredError",
    "BackendError",
    "BackendStartupError",
    "BackendNotRunningError",
    "BackendHealthCheckError",
    "UploadError",
    "FileValidationError",
    "UploadTimeoutError",
    "UploadConnectionError",
]

