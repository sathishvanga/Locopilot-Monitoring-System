"""
Tests for VideoProcessingService

Tests video processing, validation, and workflow operations.
"""

import pytest
import os
from pathlib import Path

from app.services.video_processing_service import VideoProcessingService
from app.exceptions import VideoValidationError, VideoNotFoundError


class TestVideoProcessingService:
    """Test suite for VideoProcessingService"""
    
    def test_validate_video_file_valid(self, video_processing_service: VideoProcessingService):
        """Test validation of valid video file"""
        is_valid, error = video_processing_service.validate_video_file(
            filename="test_video.mp4",
            file_size=10 * 1024 * 1024  # 10 MB
        )
        assert is_valid is True
        assert error is None
    
    def test_validate_video_file_invalid_extension(
        self, 
        video_processing_service: VideoProcessingService
    ):
        """Test validation rejects invalid file extension"""
        is_valid, error = video_processing_service.validate_video_file(
            filename="test_video.txt",
            file_size=10 * 1024 * 1024
        )
        assert is_valid is False
        assert "extension" in error.lower()
    
    def test_validate_video_file_too_large(
        self, 
        video_processing_service: VideoProcessingService
    ):
        """Test validation rejects files that are too large"""
        is_valid, error = video_processing_service.validate_video_file(
            filename="test_video.mp4",
            file_size=600 * 1024 * 1024  # 600 MB (exceeds default 500 MB limit)
        )
        assert is_valid is False
        assert "large" in error.lower() or "size" in error.lower()
    
    def test_validate_video_file_empty(
        self, 
        video_processing_service: VideoProcessingService
    ):
        """Test validation rejects empty files"""
        is_valid, error = video_processing_service.validate_video_file(
            filename="test_video.mp4",
            file_size=0
        )
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_cleanup_uploaded_video(
        self, 
        video_processing_service: VideoProcessingService,
        temp_dir: str
    ):
        """Test cleanup of uploaded video file"""
        # Create a test file
        test_file = os.path.join(temp_dir, "test_video.mp4")
        with open(test_file, 'wb') as f:
            f.write(b"test video content")
        
        assert os.path.exists(test_file)
        
        # Cleanup
        video_processing_service.cleanup_uploaded_video(test_file)
        
        # File should be deleted
        assert not os.path.exists(test_file)
    
    def test_cleanup_uploaded_video_nonexistent(
        self, 
        video_processing_service: VideoProcessingService
    ):
        """Test cleanup handles nonexistent files gracefully"""
        # Should not raise an exception
        video_processing_service.cleanup_uploaded_video("/nonexistent/path/video.mp4")

