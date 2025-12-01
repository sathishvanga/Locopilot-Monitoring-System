"""
Video models - Request and response models for video processing API
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from .activity_models import ActivityModel


class CrewMember(BaseModel):
    """
    Model for individual crew member information
    """
    
    name: Optional[str] = Field(
        None,
        description="Crew member name",
        min_length=1,
        max_length=100
    )
    id: Optional[str] = Field(
        None,
        description="Unique crew member ID",
        min_length=1,
        max_length=50
    )
    role: Optional[str] = Field(
        None,
        description="Crew role (LP or ALP)",
        pattern="^(LP|ALP)$"
    )
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "id": "LP-001",
                "role": "LP"
            }
        }


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
    lpCrew: Optional[CrewMember] = Field(
        None,
        description="Loco Pilot crew member information"
    )
    alpCrew: Optional[CrewMember] = Field(
        None,
        description="Assistant Loco Pilot crew member information"
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
                "lpCrew": {
                    "name": "John Doe",
                    "id": "LP-001",
                    "role": "LP"
                },
                "alpCrew": {
                    "name": "Jane Smith",
                    "id": "ALP-002",
                    "role": "ALP"
                }
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


class ChunkedUploadInitiateResponse(BaseModel):
    """
    Response model for initiating a chunked upload session
    """

    status: str = Field(
        default="initiated",
        description="Upload initiation status"
    )
    uploadId: str = Field(
        ...,
        description="Unique upload session identifier (UUID)"
    )
    totalChunks: int = Field(
        ...,
        description="Total number of chunks to upload"
    )
    chunkSize: int = Field(
        default=8388608,
        description="Size of each chunk in bytes (8 MB)"
    )
    expiresAt: str = Field(
        ...,
        description="ISO timestamp when this upload session expires"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "initiated",
                "uploadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "totalChunks": 13,
                "chunkSize": 8388608,
                "expiresAt": "2025-11-30T15:30:00Z"
            }
        }


class ChunkedUploadChunkResponse(BaseModel):
    """
    Response model for uploading a single chunk
    """

    status: str = Field(
        default="received",
        description="Chunk upload status"
    )
    uploadId: str = Field(
        ...,
        description="Upload session identifier"
    )
    chunkIndex: int = Field(
        ...,
        description="Index of the chunk that was uploaded"
    )
    receivedChunks: int = Field(
        ...,
        description="Total number of chunks received so far"
    )
    totalChunks: int = Field(
        ...,
        description="Total number of chunks expected"
    )
    complete: bool = Field(
        default=False,
        description="Whether all chunks have been received"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "status": "received",
                "uploadId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "chunkIndex": 5,
                "receivedChunks": 6,
                "totalChunks": 13,
                "complete": False
            }
        }

