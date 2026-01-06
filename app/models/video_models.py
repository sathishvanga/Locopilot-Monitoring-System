"""
Video models - Request and response models for video processing API
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from .activity_models import ActivityModel


class ViolationModel(BaseModel):
    """
    Model for violation data - matches the format posted to external CVVR API

    v2: Now uses arrays for types, descriptions, objectTypes to support combined activities
    """

    tripId: str = Field(..., description="Trip identifier")
    types: List[int] = Field(..., description="Array of activity type codes")
    startTime: str = Field(..., description="Violation start time in HH:mm:ss format or seconds")
    endTime: str = Field(..., description="Violation end time in HH:mm:ss format or seconds")
    clipDuration: str = Field(..., description="Duration of the clip in HH:mm:ss format")
    remarks: str = Field(default="", description="Remarks about the violation")
    reason: str = Field(default="Automated detection", description="Reason for the violation")
    descriptions: List[str] = Field(..., description="Array of activity descriptions")
    objectTypes: List[str] = Field(..., description="Array of detected object types")
    fileName: str = Field(..., description="Original video filename")
    fileDuration: str = Field(..., description="Total video duration in HH:mm:ss format")
    crewName: str = Field(..., description="Crew member name")
    fileType: int = Field(default=2, description="File type (2 = video)")
    fileUrl: str = Field(default="", description="URL to the evidence clip (S3 or local)")
    createdDate: str = Field(..., description="ISO timestamp when violation was created")
    createdBy: str = Field(default="system", description="Creator of the violation record")
    status: int = Field(default=1, description="Status (1 = active/complete)")
    roleType: int = Field(default=1, description="Role type (1 = LP, 2 = ALP)")

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "tripId": "TRIP-2024-001234",
                "types": [2, 8],
                "startTime": "00:02:15",
                "endTime": "00:02:45",
                "clipDuration": "00:00:30",
                "remarks": "",
                "reason": "Automated detection",
                "descriptions": ["Using mobile phone", "LP not exchanging hand gesture"],
                "objectTypes": ["cell phone", "lp hand gesture"],
                "fileName": "video_20241217_143022.mp4",
                "fileDuration": "00:15:30",
                "crewName": "John Doe",
                "fileType": 2,
                "fileUrl": "https://bucket.s3.amazonaws.com/clips/clip.mp4",
                "createdDate": "2024-12-17T14:35:22",
                "createdBy": "system",
                "status": 1,
                "roleType": 1
            }
        }


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

    Contains processing status, metadata, and all detected violations.
    Returns the same format that gets posted to external CVVR API.
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
        description="Total number of violations detected"
    )
    violations: List[ViolationModel] = Field(
        default_factory=list,
        description="List of all detected violations (same format as external API)"
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
                "activitiesCount": 2,
                "violations": [
                    {
                        "tripId": "TRIP-20251110-001",
                        "type": 1,
                        "startTime": "00:02:15",
                        "endTime": "00:02:45",
                        "clipDuration": "00:00:30",
                        "description": "Phone usage detected",
                        "objectTypes": "phone",
                        "fileName": "video.mp4",
                        "fileDuration": "00:15:30",
                        "crewName": "John Doe",
                        "fileUrl": "https://bucket.s3.amazonaws.com/clips/clip.mp4"
                    }
                ],
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



