"""
Pydantic models for chunked/resumable upload workflow.

This module defines the request and response models for the 3-step chunked upload process:
1. Initiate upload session
2. Upload individual chunks
3. Complete and assemble final file
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class InitiateUploadRequest(BaseModel):
    """Request model for initiating a chunked upload session."""

    filename: str = Field(..., description="Original filename of the video")
    total_size: int = Field(..., description="Total file size in bytes", gt=0)
    tripId: str = Field(..., description="Unique trip identifier")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate that filename is not empty and has valid extension."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()

    @field_validator("tripId")
    @classmethod
    def validate_trip_id(cls, v: str) -> str:
        """Validate that tripId is not empty."""
        if not v or not v.strip():
            raise ValueError("tripId cannot be empty")
        return v.strip()


class InitiateUploadResponse(BaseModel):
    """Response model after initiating upload session."""

    upload_id: str = Field(..., description="Unique upload session ID")
    chunk_size_recommendation: int = Field(..., description="Recommended chunk size in bytes")
    total_chunks: int = Field(..., description="Expected total number of chunks")
    expires_at: str = Field(..., description="ISO format timestamp when session expires")


class UploadChunkRequest(BaseModel):
    """Request model for uploading a single chunk."""

    upload_id: str = Field(..., description="Upload session ID from initiate step")
    part_number: int = Field(..., description="1-based part number", ge=1)

    @field_validator("upload_id")
    @classmethod
    def validate_upload_id(cls, v: str) -> str:
        """Validate that upload_id is not empty."""
        if not v or not v.strip():
            raise ValueError("upload_id cannot be empty")
        return v.strip()


class UploadChunkResponse(BaseModel):
    """Response model after uploading a chunk."""

    status: str = Field(default="ok", description="Status of chunk upload")
    part: int = Field(..., description="Part number that was uploaded")
    message: Optional[str] = Field(default=None, description="Optional message")


class CompleteUploadRequest(BaseModel):
    """Request model for completing chunked upload."""

    upload_id: str = Field(..., description="Upload session ID")

    @field_validator("upload_id")
    @classmethod
    def validate_upload_id(cls, v: str) -> str:
        """Validate that upload_id is not empty."""
        if not v or not v.strip():
            raise ValueError("upload_id cannot be empty")
        return v.strip()


class UploadStatusResponse(BaseModel):
    """Response model for upload session status check."""

    upload_id: str = Field(..., description="Upload session ID")
    filename: str = Field(..., description="Original filename")
    total_size: int = Field(..., description="Total expected file size in bytes")
    trip_id: str = Field(..., description="Associated trip ID")
    created_at: str = Field(..., description="ISO format timestamp when session was created")
    expires_at: str = Field(..., description="ISO format timestamp when session expires")
    chunks_uploaded: List[int] = Field(..., description="List of successfully uploaded chunk numbers")
    total_chunks_expected: int = Field(..., description="Total number of chunks expected")
    is_complete: bool = Field(..., description="Whether all chunks have been uploaded")
    bytes_uploaded: int = Field(..., description="Total bytes uploaded so far")


class CancelUploadRequest(BaseModel):
    """Request model for canceling an upload session."""

    upload_id: str = Field(..., description="Upload session ID to cancel")

    @field_validator("upload_id")
    @classmethod
    def validate_upload_id(cls, v: str) -> str:
        """Validate that upload_id is not empty."""
        if not v or not v.strip():
            raise ValueError("upload_id cannot be empty")
        return v.strip()


class CancelUploadResponse(BaseModel):
    """Response model after canceling upload session."""

    status: str = Field(default="cancelled", description="Status of cancellation")
    upload_id: str = Field(..., description="Upload session ID that was cancelled")
    message: str = Field(..., description="Cancellation confirmation message")


class UploadSessionMetadata(BaseModel):
    """Internal model for upload session metadata stored in meta.json."""

    filename: str
    total_size: int
    trip_id: str
    created_at: str  # ISO format datetime
    chunk_size_recommendation: int
    total_chunks_expected: int


class ErrorResponse(BaseModel):
    """Standard error response model."""

    status: str = Field(default="error", description="Always 'error' for error responses")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
