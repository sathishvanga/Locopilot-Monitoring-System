"""
Backend-related exceptions

Custom exception classes for backend management and health check errors.
"""

from typing import Optional


class BackendError(Exception):
    """
    Base exception for backend-related errors.
    
    Attributes:
        message: Human-readable error message
        backend_url: Backend URL (if applicable)
        process_pid: Process PID (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        backend_url: Optional[str] = None,
        process_pid: Optional[int] = None
    ):
        """
        Initialize backend error.
        
        Args:
            message: Error message
            backend_url: Backend URL (optional)
            process_pid: Process PID (optional)
        """
        self.message = message
        self.backend_url = backend_url
        self.process_pid = process_pid
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.backend_url:
            parts.append(f"URL: {self.backend_url}")
        if self.process_pid:
            parts.append(f"PID: {self.process_pid}")
        return " | ".join(parts)


class BackendStartupError(BackendError):
    """
    Exception raised when backend fails to start.
    
    Used for startup failures, process crashes, or timeout issues.
    """
    pass


class BackendNotRunningError(BackendError):
    """
    Exception raised when backend is not running.
    
    Used when backend is expected to be running but health checks fail.
    """
    pass


class BackendHealthCheckError(BackendError):
    """
    Exception raised when backend health check fails.
    
    Used when backend process exists but doesn't respond to health checks.
    """
    pass

