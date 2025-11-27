"""
Pytest configuration and shared fixtures for desktop app tests

This file contains pytest fixtures and configuration used across all tests.
"""

import pytest
import os
import tempfile
import shutil
from typing import Generator
from pathlib import Path
from unittest.mock import Mock, MagicMock

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject

from desktop_app.services.auth_service import AuthService
from desktop_app.services.trip_service import TripService
from desktop_app.services.upload_service import UploadService
from desktop_app.services.local_processing_service import LocalProcessingService
from desktop_app.services.backend_manager import BackendManager
from desktop_app.models.auth_models import LoginResponse
from desktop_app.models.trip_models import TripModel
from desktop_app.utils.config import Settings


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """
    Create QApplication instance for Qt tests.
    
    Yields:
        QApplication: Application instance
        
    Note: This fixture is session-scoped to avoid creating multiple QApplications.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


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
        api_base_url="https://test-api.example.com",
        local_backend_url="http://localhost:8000",
        local_backend_port=8000,
        debug=True,
        log_level="DEBUG"
    )


@pytest.fixture
def mock_auth_service() -> Mock:
    """
    Create a mock authentication service.
    
    Returns:
        Mock: Mocked AuthService instance
    """
    service = Mock(spec=AuthService)
    service.get_token.return_value = "test_token_123"
    service.is_authenticated.return_value = True
    service.get_user_info.return_value = LoginResponse(
        uuid="test-uuid",
        name="Test User",
        actionId=1,
        divId="DIV001",
        mobileNumber="1234567890",
        divName="Test Division",
        token="test_token_123",
        designation="Test",
        roleId=1,
        status=1,
        createdDate="2025-01-01"
    )
    return service


@pytest.fixture
def mock_trip_service() -> Mock:
    """
    Create a mock trip service.
    
    Returns:
        Mock: Mocked TripService instance
    """
    service = Mock(spec=TripService)
    service.get_pending_trips.return_value = (True, [], None)
    return service


@pytest.fixture
def sample_trip() -> TripModel:
    """
    Create a sample trip for testing.
    
    Returns:
        TripModel: Sample trip instance
    """
    return TripModel(
        uuid="test-trip-uuid-123",
        dateTime="2025-11-27T10:00:00",
        fromStationId="ST001",
        toStationId="ST002",
        sectionId="SEC001",
        trainNo="T12345",
        locoNo="L67890"
    )


@pytest.fixture
def sample_trips(sample_trip: TripModel) -> list[TripModel]:
    """
    Create a list of sample trips for testing.
    
    Args:
        sample_trip: Sample trip fixture
        
    Returns:
        list[TripModel]: List of sample trips
    """
    return [sample_trip]

