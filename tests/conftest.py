"""
Pytest configuration and shared fixtures

This file contains pytest fixtures and configuration used across all tests.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

from app.utils.config import Settings
from app.repositories.activity_repository import ActivityRepository
from app.services.video_processing_service import VideoProcessingService
from app.services.activity_detection_service import ActivityDetectionService
from app.services.s3_upload_service import S3UploadService


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for testing.
    
    Yields:
        str: Path to temporary directory
        
    The directory is automatically cleaned up after the test.
    """
    temp_path = tempfile.mkdtemp()
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir: str) -> Settings:
    """
    Create test settings with temporary directories.
    
    Args:
        temp_dir: Temporary directory path
        
    Returns:
        Settings: Test settings instance
    """
    return Settings(
        output_dir=os.path.join(temp_dir, "output"),
        upload_dir=os.path.join(temp_dir, "uploads"),
        debug=True,
        log_level="DEBUG"
    )


@pytest.fixture
def activity_repository(temp_dir: str) -> ActivityRepository:
    """
    Create an activity repository for testing.
    
    Args:
        temp_dir: Temporary directory path
        
    Returns:
        ActivityRepository: Repository instance
    """
    output_dir = os.path.join(temp_dir, "output")
    return ActivityRepository(output_dir=output_dir)


@pytest.fixture
def video_processing_service(activity_repository: ActivityRepository) -> VideoProcessingService:
    """
    Create a video processing service for testing.
    
    Args:
        activity_repository: Activity repository instance
        
    Returns:
        VideoProcessingService: Service instance
    """
    return VideoProcessingService(
        activity_repository=activity_repository,
        activity_detection_service=ActivityDetectionService()
    )


@pytest.fixture
def sample_activities() -> list:
    """
    Sample activities data for testing.
    
    Returns:
        list: List of sample activity dictionaries
    """
    return [
        {
            "tripId": "TEST-001",
            "activityType": 2,
            "des": "Using mobile phone",
            "objectType": "cell phone",
            "fileUrl": "/path/to/video.mp4",
            "fileDuration": "00:10:30",
            "activityStartTime": "125.50",
            "activityEndTime": "132.75",
            "crewName": "John Doe",
            "crewId": "LP-001",
            "crewRole": 1,
            "date": "2025-11-10",
            "time": "14:30:45",
            "filename": "test_video.mp4",
            "peopleCount": 1,
            "evidence": {"rule": "phone_in_hand"},
            "activityImage": "test_activity.jpg",
            "activityClip": "test_clip.mp4"
        }
    ]

