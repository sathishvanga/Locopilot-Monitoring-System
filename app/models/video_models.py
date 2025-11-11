"""
Video models - Request and response models for video processing API
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from .activity_models import ActivityModel


class VideoUploadRequest(BaseModel):
    """
    Request model for video upload and processing
    
    Note: The actual video file will be uploaded via multipart/form-data,
    this model handles the metadata.
    """
    
    tripId: str = Field(
        ..., 
        description="Unique trip identifier",
        min_length=1,
        max_length=100
    )
    crewName: Optional[str] = Field(
        default="John Doe",
        description="Crew member name"
    )
    crewId: Optional[str] = Field(
        default="C-001",
        description="Crew member ID"
    )
    crewRole: Optional[int] = Field(
        default=1,
        description="Crew role (1 = primary loco pilot)",
        ge=1,
        le=10
    )
    
    @validator('tripId')
    def validate_trip_id(cls, v):
        """Validate tripId is not empty after stripping whitespace"""
        if not v.strip():
            raise ValueError('tripId cannot be empty or whitespace')
        return v.strip()
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "tripId": "TRIP-20251110-001",
                "crewName": "John Doe",
                "crewId": "C-001",
                "crewRole": 1
            }
        }


class VideoProcessingResponse(BaseModel):
    """
    Response model for video processing endpoint
    
    Contains processing status, metadata, and all detected activities.
    """
    
    status: str = Field(
        ...,
        description="Processing status (success, error, processing)"
    )
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    tripId: str = Field(
        ...,
        description="Trip identifier from request"
    )
    videoFilename: str = Field(
        ...,
        description="Uploaded video filename"
    )
    runDirectory: str = Field(
        ...,
        description="Output directory path for this processing run"
    )
    activitiesJsonPath: str = Field(
        ...,
        description="Path to activities.json file"
    )
    activitiesCount: int = Field(
        ...,
        description="Total number of activities detected"
    )
    activities: List[ActivityModel] = Field(
        default_factory=list,
        description="List of all detected activities"
    )
    processingTime: Optional[float] = Field(
        None,
        description="Total processing time in seconds"
    )
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Video processed successfully",
                "tripId": "TRIP-20251110-001",
                "videoFilename": "uploaded_video.mp4",
                "runDirectory": "/path/to/locopilot_evidence/run_20251110_143045",
                "activitiesJsonPath": "/path/to/locopilot_evidence/run_20251110_143045/activities.json",
                "activitiesCount": 5,
                "activities": [],
                "processingTime": 45.67
            }
        }


class VideoProcessingError(BaseModel):
    """Error response model for video processing failures"""
    
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    error: str = Field(..., description="Detailed error information")
    tripId: Optional[str] = Field(None, description="Trip ID if available")
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Failed to process video",
                "error": "Invalid video format: file corrupted",
                "tripId": "TRIP-20251110-001"
            }
        }

