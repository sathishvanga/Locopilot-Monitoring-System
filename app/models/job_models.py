"""
Job models - Pydantic models for async job queue system

Defines the data structures for job management, including status tracking,
request/response schemas, and queue status reporting.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """
    Enumeration of possible job states in the processing queue.

    Job lifecycle: PENDING -> QUEUED -> PROCESSING -> COMPLETED/FAILED/CANCELLED
    """
    PENDING = "pending"       # Job created, not yet queued
    QUEUED = "queued"         # Job added to queue, waiting for worker
    PROCESSING = "processing" # Worker actively processing job
    COMPLETED = "completed"   # Job finished successfully
    FAILED = "failed"         # Job encountered an error
    CANCELLED = "cancelled"   # Job was cancelled by user


class Job(BaseModel):
    """
    Internal job representation with full tracking metadata.

    Tracks the complete lifecycle of a video processing job including
    timing, progress, results, and error information.
    """
    id: str = Field(..., description="Unique job identifier (UUID)")
    video_path: str = Field(..., description="Path to video file for processing")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job configuration parameters"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current job status"
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Processing progress percentage (0-100)"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing result data when completed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when job was created"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when processing completed/failed"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "video_path": "/tmp/uploads/video.mp4",
                "config": {"trip_id": "TRIP-001", "use_mock": False},
                "status": "processing",
                "progress": 45,
                "result": None,
                "error": None,
                "created_at": "2025-01-06T10:30:00Z",
                "started_at": "2025-01-06T10:30:05Z",
                "completed_at": None
            }
        }


class JobSubmitRequest(BaseModel):
    """
    Request model for submitting a new video processing job.

    The video_path should point to a valid video file accessible by the server.
    Config can contain job-specific parameters like trip_id, crew info, etc.
    """
    video_path: str = Field(
        ...,
        description="Path to the video file to process",
        min_length=1
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job configuration parameters (trip_id, crew_members, etc.)"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "video_path": "/tmp/uploads/trip_video.mp4",
                "config": {
                    "trip_id": "TRIP-20250106-001",
                    "lp_crew_name": "John Doe",
                    "lp_crew_id": "LP-001",
                    "save_clips": True
                }
            }
        }


class JobSubmitResponse(BaseModel):
    """
    Response model for job submission.

    Returns the job_id immediately for tracking and status polling.
    """
    success: bool = Field(..., description="Whether job was submitted successfully")
    job_id: str = Field(..., description="Unique job identifier for tracking")
    message: str = Field(..., description="Status message")
    queue_position: Optional[int] = Field(
        default=None,
        description="Position in queue (if queued)"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Job submitted successfully",
                "queue_position": 3
            }
        }


class JobStatusResponse(BaseModel):
    """
    Response model for job status queries.

    Provides full job details including progress and timing information.
    """
    success: bool = Field(..., description="Whether the request was successful")
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., description="Processing progress percentage (0-100)")
    video_path: str = Field(..., description="Path to video being processed")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job configuration"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None,
        description="Processing start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Processing completion timestamp"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 75,
                "video_path": "/tmp/uploads/video.mp4",
                "config": {"trip_id": "TRIP-001"},
                "error": None,
                "created_at": "2025-01-06T10:30:00Z",
                "started_at": "2025-01-06T10:30:05Z",
                "completed_at": None,
                "processing_time_seconds": 45.5
            }
        }


class JobResultResponse(BaseModel):
    """
    Response model for completed job results.

    Returns the full processing result including detected activities.
    """
    success: bool = Field(..., description="Whether the request was successful")
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status (should be COMPLETED)")
    result: Dict[str, Any] = Field(..., description="Processing result data")
    processing_time_seconds: Optional[float] = Field(
        default=None,
        description="Total processing time in seconds"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "result": {
                    "trip_id": "TRIP-001",
                    "activities_count": 5,
                    "activities": [],
                    "run_directory": "/output/run_20250106_103000"
                },
                "processing_time_seconds": 120.5
            }
        }


class QueueStatusResponse(BaseModel):
    """
    Response model for queue status queries.

    Provides overview of queue health and job counts.
    """
    success: bool = Field(default=True, description="Request status")
    queue_depth: int = Field(..., description="Number of jobs waiting in queue")
    active_jobs: int = Field(..., description="Number of jobs currently processing")
    pending_jobs: int = Field(..., description="Number of jobs in pending state")
    completed_jobs: int = Field(..., description="Number of completed jobs (in memory)")
    failed_jobs: int = Field(..., description="Number of failed jobs (in memory)")
    cancelled_jobs: int = Field(..., description="Number of cancelled jobs (in memory)")
    total_jobs: int = Field(..., description="Total jobs tracked in memory")
    max_queue_size: int = Field(..., description="Maximum queue capacity")
    num_workers: int = Field(..., description="Number of worker tasks")
    queue_full: bool = Field(..., description="Whether queue is at capacity")

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "success": True,
                "queue_depth": 5,
                "active_jobs": 2,
                "pending_jobs": 3,
                "completed_jobs": 10,
                "failed_jobs": 1,
                "cancelled_jobs": 0,
                "total_jobs": 16,
                "max_queue_size": 10,
                "num_workers": 3,
                "queue_full": False
            }
        }


class JobCancelResponse(BaseModel):
    """
    Response model for job cancellation requests.
    """
    success: bool = Field(..., description="Whether cancellation was successful")
    job_id: str = Field(..., description="Job identifier")
    message: str = Field(..., description="Status message")
    previous_status: Optional[JobStatus] = Field(
        default=None,
        description="Job status before cancellation"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "success": True,
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Job cancelled successfully",
                "previous_status": "queued"
            }
        }
