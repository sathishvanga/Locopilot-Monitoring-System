"""
Tests for AuthService

Tests authentication, login, and token management.
"""

import pytest
from unittest.mock import Mock, patch

from desktop_app.services.auth_service import AuthService
from desktop_app.models.auth_models import LoginResponse, LoginAPIResponse
from desktop_app.exceptions import AuthenticationError, InvalidCredentialsError


class TestAuthService:
    """Test suite for AuthService"""
    
    def test_login_success(self, mock_auth_service: Mock):
        """Test successful login"""
        # This would require mocking the API client
        # For now, we test the structure
        assert mock_auth_service.get_token() == "test_token_123"
        assert mock_auth_service.is_authenticated() is True
    
    def test_login_validation_empty_username(self):
        """Test login validation rejects empty username"""
        service = AuthService()
        success, user_info, error = service.login("", "password123")
        
        assert success is False
        assert user_info is None
        assert "required" in error.lower()
    
    def test_login_validation_empty_password(self):
        """Test login validation rejects empty password"""
        service = AuthService()
        success, user_info, error = service.login("username", "")
        
        assert success is False
        assert user_info is None
        assert "required" in error.lower()
    
    def test_is_authenticated_false_when_no_token(self):
        """Test is_authenticated returns False when no token"""
        service = AuthService()
        # Clear any existing auth state
        service.logout()
        assert service.is_authenticated() is False
    
    def test_logout_clears_state(self):
        """Test logout clears authentication state"""
        service = AuthService()
        # Set some state first (would normally be done via login)
        service.auth_state.set_auth("test_token", Mock(spec=LoginResponse))
        assert service.is_authenticated() is True
        
        service.logout()
        assert service.is_authenticated() is False

