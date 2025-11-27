"""
Video processing exceptions

Custom exception classes for video-related operations.
"""

from typing import Optional


class VideoProcessingError(Exception):
    """
    Base exception for video processing errors.
    
    Attributes:
        message: Human-readable error message
        video_path: Path to the video file that caused the error (if applicable)
        trip_id: Trip identifier associated with the error (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        trip_id: Optional[str] = None
    ):
        """
        Initialize video processing error.
        
        Args:
            message: Error message
            video_path: Path to video file (optional)
            trip_id: Trip identifier (optional)
        """
        self.message = message
        self.video_path = video_path
        self.trip_id = trip_id
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.video_path:
            parts.append(f"Video: {self.video_path}")
        if self.trip_id:
            parts.append(f"Trip: {self.trip_id}")
        return " | ".join(parts)


class VideoValidationError(VideoProcessingError):
    """
    Exception raised when video file validation fails.
    
    Used for invalid file formats, sizes, or other validation issues.
    """
    pass


class VideoNotFoundError(VideoProcessingError):
    """
    Exception raised when a video file cannot be found.
    
    Used when video file path is invalid or file doesn't exist.
    """
    pass


class VideoUploadError(VideoProcessingError):
    """
    Exception raised when video upload fails.
    
    Used for upload-related errors such as I/O failures or network issues.
    """
    pass

