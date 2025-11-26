"""
Unit tests for services
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desktop_app.services.auth_service import AuthService
from desktop_app.services.trip_service import TripService
from desktop_app.services.upload_service import UploadService
from desktop_app.models.auth_models import LoginResponse
from desktop_app.models.trip_models import TripModel


class TestAuthService(unittest.TestCase):
    """Test authentication service"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.auth_service = AuthService()
    
    @patch('desktop_app.services.auth_service.APIClient')
    def test_login_success(self, mock_api_client):
        """Test successful login"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "uuid": "test-uuid",
            "name": "Test User",
            "actionId": 1,
            "divId": "div-123",
            "mobileNumber": "9705589009",
            "divName": "Test Division",
            "token": "test-token-123",
            "designation": "admin",
            "roleId": 7,
            "status": 1,
            "createdDate": "2024-01-01 10:00:00"
        }
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_api_client.return_value = mock_client_instance
        
        # Reinitialize service with mocked client
        self.auth_service = AuthService()
        
        # Test login
        success, user_info, error = self.auth_service.login("9705589009", "test123")
        
        # Assertions
        self.assertTrue(success)
        self.assertIsNotNone(user_info)
        self.assertIsNone(error)
        self.assertEqual(user_info.name, "Test User")
        self.assertEqual(user_info.token, "test-token-123")
    
    def test_login_validation(self):
        """Test login input validation"""
        # Empty username
        success, user_info, error = self.auth_service.login("", "password")
        self.assertFalse(success)
        self.assertIsNotNone(error)
        
        # Empty password
        success, user_info, error = self.auth_service.login("9705589009", "")
        self.assertFalse(success)
        self.assertIsNotNone(error)
    
    def test_is_authenticated(self):
        """Test authentication status check"""
        # Initially not authenticated
        self.assertFalse(self.auth_service.is_authenticated())
        
        # Mock authentication state
        from desktop_app.models.auth_models import AuthState
        AuthState.set_auth("test-token", Mock(spec=LoginResponse))
        
        # Now authenticated
        self.assertTrue(self.auth_service.is_authenticated())
        
        # Cleanup
        AuthState.clear()


class TestTripService(unittest.TestCase):
    """Test trip service"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.trip_service = TripService(auth_token="test-token")
    
    @patch('desktop_app.services.trip_service.APIClient')
    def test_get_pending_trips_success(self, mock_api_client):
        """Test fetching pending trips successfully"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "uuid": "trip-123",
                "dateTime": "2024-01-01 10:00:00",
                "fromStationId": "STN1",
                "toStationId": "STN2",
                "trainNo": "12345",
                "locoNo": "WAP7-123",
                "status": 0
            }
        ]
        
        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_api_client.return_value = mock_client_instance
        
        # Reinitialize service with mocked client
        self.trip_service = TripService(auth_token="test-token")
        
        # Test get trips
        success, trips, error = self.trip_service.get_pending_trips()
        
        # Assertions
        self.assertTrue(success)
        self.assertEqual(len(trips), 1)
        self.assertIsNone(error)
        self.assertEqual(trips[0].uuid, "trip-123")
    
    def test_get_trips_no_token(self):
        """Test fetching trips without authentication token"""
        service = TripService()
        success, trips, error = service.get_pending_trips()
        
        self.assertFalse(success)
        self.assertEqual(len(trips), 0)
        self.assertIsNotNone(error)


class TestUploadService(unittest.TestCase):
    """Test upload service"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.upload_service = UploadService(auth_token="test-token")
    
    def test_validate_file_not_exists(self):
        """Test file validation for non-existent file"""
        is_valid, error = self.upload_service.validate_file("/nonexistent/file.mp4")
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_file_empty(self, mock_getsize, mock_exists):
        """Test file validation for empty file"""
        mock_exists.return_value = True
        mock_getsize.return_value = 0
        
        is_valid, error = self.upload_service.validate_file("test.mp4")
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_file_success(self, mock_getsize, mock_exists):
        """Test successful file validation"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        is_valid, error = self.upload_service.validate_file("test.mp4")
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()

