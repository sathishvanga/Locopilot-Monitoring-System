"""
Tests for UploadService

Tests file upload, validation, and S3 operations.
"""

import pytest
import os
import tempfile
from pathlib import Path

from desktop_app.services.upload_service import UploadService
from desktop_app.exceptions import FileValidationError, UploadError


class TestUploadService:
    """Test suite for UploadService"""
    
    def test_validate_file_exists(self, temp_dir: str):
        """Test file validation with existing file"""
        service = UploadService()
        
        # Create a test file
        test_file = os.path.join(temp_dir, "test_video.mp4")
        with open(test_file, 'wb') as f:
            f.write(b"test video content")
        
        is_valid, error = service.validate_file(test_file)
        assert is_valid is True
        assert error is None
    
    def test_validate_file_not_exists(self):
        """Test file validation rejects non-existent file"""
        service = UploadService()
        is_valid, error = service.validate_file("/nonexistent/path/video.mp4")
        
        assert is_valid is False
        assert "not exist" in error.lower() or "not found" in error.lower()
    
    def test_validate_file_empty(self, temp_dir: str):
        """Test file validation rejects empty file"""
        service = UploadService()
        
        # Create empty file
        test_file = os.path.join(temp_dir, "empty.mp4")
        with open(test_file, 'wb'):
            pass  # Create empty file
        
        is_valid, error = service.validate_file(test_file)
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_file_path_traversal_protection(self):
        """Test file validation prevents path traversal"""
        service = UploadService()
        
        # Try path traversal attack
        malicious_path = "../../../etc/passwd"
        is_valid, error = service.validate_file(malicious_path)
        
        # Should fail validation
        assert is_valid is False or error is not None

