"""
Upload-related exceptions

Custom exception classes for file upload and validation errors.
"""

from typing import Optional


class UploadError(Exception):
    """
    Base exception for upload-related errors.
    
    Attributes:
        message: Human-readable error message
        file_path: Path to file that failed (if applicable)
        file_size: File size in bytes (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None
    ):
        """
        Initialize upload error.
        
        Args:
            message: Error message
            file_path: File path (optional)
            file_size: File size (optional)
        """
        self.message = message
        self.file_path = file_path
        self.file_size = file_size
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.file_size:
            size_mb = self.file_size / (1024 * 1024)
            parts.append(f"Size: {size_mb:.2f} MB")
        return " | ".join(parts)


class FileValidationError(UploadError):
    """
    Exception raised when file validation fails.
    
    Used for invalid file formats, sizes, or other validation issues.
    """
    pass


class UploadTimeoutError(UploadError):
    """
    Exception raised when upload times out.
    
    Used when upload operation exceeds timeout limit.
    """
    pass


class UploadConnectionError(UploadError):
    """
    Exception raised when upload connection fails.
    
    Used for network connection errors during upload.
    """
    pass

