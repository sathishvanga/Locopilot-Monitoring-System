"""
Unit tests for data models
"""

import unittest
from pydantic import ValidationError

from desktop_app.models.auth_models import LoginRequest, LoginResponse, AuthState
from desktop_app.models.trip_models import TripModel, UploadStatus


class TestAuthModels(unittest.TestCase):
    """Test authentication models"""
    
    def test_login_request_creation(self):
        """Test creating a login request"""
        request = LoginRequest(
            username="9705589009",
            password="test123",
            osType=1,
            captchaToken=""
        )
        
        self.assertEqual(request.username, "9705589009")
        self.assertEqual(request.password, "test123")
        self.assertEqual(request.osType, 1)
        self.assertEqual(request.captchaToken, "")
    
    def test_login_request_defaults(self):
        """Test login request default values"""
        request = LoginRequest(
            username="9705589009",
            password="test123"
        )
        
        self.assertEqual(request.osType, 1)
        self.assertEqual(request.captchaToken, "")
    
    def test_login_response_creation(self):
        """Test creating a login response"""
        response = LoginResponse(
            uuid="test-uuid",
            name="Test User",
            actionId=1,
            divId="div-123",
            mobileNumber="9705589009",
            divName="Test Division",
            token="test-token",
            designation="admin",
            roleId=7,
            status=1,
            createdDate="2024-01-01 10:00:00"
        )
        
        self.assertEqual(response.name, "Test User")
        self.assertEqual(response.token, "test-token")
    
    def test_auth_state_singleton(self):
        """Test AuthState singleton pattern"""
        state1 = AuthState()
        state2 = AuthState()
        
        self.assertIs(state1, state2)
    
    def test_auth_state_storage(self):
        """Test storing and retrieving authentication data"""
        # Clear state
        AuthState.clear()
        
        # Not authenticated initially
        self.assertFalse(AuthState.is_authenticated())
        
        # Store auth data
        mock_response = LoginResponse(
            uuid="test-uuid",
            name="Test User",
            actionId=1,
            divId="div-123",
            mobileNumber="9705589009",
            divName="Test Division",
            token="test-token",
            designation="admin",
            roleId=7,
            status=1,
            createdDate="2024-01-01 10:00:00"
        )
        
        AuthState.set_auth("test-token", mock_response)
        
        # Verify storage
        self.assertTrue(AuthState.is_authenticated())
        self.assertEqual(AuthState.get_token(), "test-token")
        self.assertEqual(AuthState.get_user_info().name, "Test User")
        
        # Clear and verify
        AuthState.clear()
        self.assertFalse(AuthState.is_authenticated())


class TestTripModels(unittest.TestCase):
    """Test trip models"""
    
    def test_trip_model_creation(self):
        """Test creating a trip model"""
        trip = TripModel(
            uuid="trip-123",
            dateTime="2024-01-01 10:00:00",
            fromStationId="STN1",
            toStationId="STN2",
            trainNo="12345",
            locoNo="WAP7-123"
        )
        
        self.assertEqual(trip.uuid, "trip-123")
        self.assertEqual(trip.trainNo, "12345")
    
    def test_trip_model_optional_fields(self):
        """Test trip model with optional fields"""
        trip = TripModel(uuid="trip-123")
        
        self.assertEqual(trip.uuid, "trip-123")
        self.assertIsNone(trip.dateTime)
        self.assertIsNone(trip.trainNo)
    
    def test_upload_status_creation(self):
        """Test creating upload status"""
        status = UploadStatus(
            trip_uuid="trip-123",
            status="uploading",
            progress=50,
            message="Uploading video..."
        )
        
        self.assertEqual(status.trip_uuid, "trip-123")
        self.assertEqual(status.status, "uploading")
        self.assertEqual(status.progress, 50)
    
    def test_upload_status_defaults(self):
        """Test upload status default values"""
        status = UploadStatus(trip_uuid="trip-123")
        
        self.assertEqual(status.status, "pending")
        self.assertEqual(status.progress, 0)
        self.assertEqual(status.message, "")


if __name__ == '__main__':
    unittest.main()

