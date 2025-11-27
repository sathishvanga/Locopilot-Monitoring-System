"""
API-related exceptions

Custom exception classes for API operations and external service interactions.
"""

from typing import Optional, Dict, Any


class APIError(Exception):
    """
    Base exception for API-related errors.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code (if applicable)
        response_data: Response data from API (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code (optional)
            response_data: Response data (optional)
        """
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        return " | ".join(parts)


class ExternalAPIError(APIError):
    """
    Exception raised when external API calls fail.
    
    Used for errors when communicating with external services
    such as the CVVR API.
    
    Attributes:
        api_url: URL of the API endpoint that failed
        trip_id: Trip identifier associated with the error (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        api_url: Optional[str] = None,
        trip_id: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize external API error.
        
        Args:
            message: Error message
            api_url: API URL that failed (optional)
            trip_id: Trip identifier (optional)
            status_code: HTTP status code (optional)
            response_data: Response data (optional)
        """
        self.api_url = api_url
        self.trip_id = trip_id
        super().__init__(message, status_code, response_data)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.api_url:
            parts.append(f"API: {self.api_url}")
        if self.trip_id:
            parts.append(f"Trip: {self.trip_id}")
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        return " | ".join(parts)


class S3UploadError(APIError):
    """
    Exception raised when S3 upload operations fail.
    
    Used for errors during file uploads to S3 storage.
    
    Attributes:
        file_path: Path to the file that failed to upload
        subfolder: S3 subfolder name (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        subfolder: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize S3 upload error.
        
        Args:
            message: Error message
            file_path: File path that failed (optional)
            subfolder: S3 subfolder (optional)
            status_code: HTTP status code (optional)
            response_data: Response data (optional)
        """
        self.file_path = file_path
        self.subfolder = subfolder
        super().__init__(message, status_code, response_data)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        parts = [self.message]
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        if self.subfolder:
            parts.append(f"Subfolder: {self.subfolder}")
        return " | ".join(parts)


class InvalidRequestError(APIError):
    """
    Exception raised when API request validation fails.
    
    Used for invalid request parameters, missing required fields, etc.
    """
    pass

