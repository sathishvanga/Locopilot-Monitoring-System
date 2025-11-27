"""
Trip models for CVVR system
"""

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class TripModel(BaseModel):
    """
    CVVR Trip model
    
    Represents a pending trip that requires video upload and processing
    """
    uuid: str = Field(..., description="Trip unique identifier")
    dateTime: Optional[str] = Field(None, description="Trip date and time")
    fromStationId: Optional[str] = Field(None, description="Starting station ID")
    toStationId: Optional[str] = Field(None, description="Destination station ID")
    sectionId: Optional[str] = Field(None, description="Section ID")
    trainNo: Optional[str] = Field(None, description="Train number")
    locoNo: Optional[str] = Field(None, description="Locomotive number")
    createdBy: Optional[str] = Field(None, description="Created by user")
    analysisType: Optional[int] = Field(None, description="Type of analysis (0=default, etc.)")
    status: Optional[int] = Field(None, description="Trip status")
    
    # Additional fields that might be in the API response
    fromStation: Optional[str] = Field(None, description="Starting station name")
    toStation: Optional[str] = Field(None, description="Destination station name")
    sectionName: Optional[str] = Field(None, description="Section name")
    divId: Optional[str] = Field(None, description="Division ID")
    createdDate: Optional[str] = Field(None, description="Creation timestamp")
    
    class Config:
        # Allow extra fields from API
        extra = "allow"


class TripsAPIResponse(BaseModel):
    """
    Wrapper for the trips API response structure
    
    The API returns trips wrapped in a content array with metadata
    """
    mssg: str = Field(..., description="Response message")
    content: List[TripModel] = Field(..., description="List of trip data")
    status: int = Field(..., description="Response status code")


class UploadStatus(BaseModel):
    """
    Track upload status for a trip
    """
    trip_uuid: str = Field(..., description="Trip UUID")
    status: str = Field(default="pending", description="Upload status: pending, uploading, processing, completed, error")
    progress: int = Field(default=0, description="Progress percentage (0-100)")
    message: str = Field(default="", description="Status message")
    video_url: Optional[str] = Field(None, description="Uploaded video S3 URL")
    evidence_urls: list[str] = Field(default_factory=list, description="Uploaded evidence clip S3 URLs")
    error: Optional[str] = Field(None, description="Error message if failed")


class S3UploadResponse(BaseModel):
    """
    Response from S3 upload API
    """
    url: str = Field(..., description="S3 URL of uploaded file")


class ProcessingResult(BaseModel):
    """
    Result from local video processing and S3 upload
    """
    success: bool = Field(..., description="Whether processing succeeded")
    run_dir: Optional[str] = Field(None, description="Output directory path")
    clips_dir: Optional[str] = Field(None, description="Evidence clips directory path")
    activities_count: int = Field(default=0, description="Number of detected activities")
    clip_files: list[str] = Field(default_factory=list, description="List of evidence clip file paths (local)")
    video_url: Optional[str] = Field(None, description="S3 URL of uploaded video")
    evidence_urls: list[str] = Field(default_factory=list, description="S3 URLs of uploaded evidence clips")
    clips_uploaded: int = Field(default=0, description="Number of clips successfully uploaded to S3")
    error: Optional[str] = Field(None, description="Error message if failed")

