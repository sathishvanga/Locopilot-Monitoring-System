"""
Authentication models for API communication
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """
    Login request payload
    
    Attributes:
        username: Mobile number or username
        password: User password
        osType: Operating system type (1 for desktop)
        captchaToken: reCAPTCHA token (empty for desktop app)
    """
    username: str = Field(..., description="Mobile number or username")
    password: str = Field(..., description="User password")
    osType: int = Field(default=1, description="OS type - 1 for desktop")
    captchaToken: str = Field(default="", description="reCAPTCHA token")


class LoginResponse(BaseModel):
    """
    Login response from API
    
    Contains user information and authentication token
    """
    uuid: str = Field(..., description="User UUID")
    name: str = Field(..., description="User name")
    actionId: int = Field(..., description="Action ID")
    divId: str = Field(..., description="Division ID")
    mobileNumber: str = Field(..., description="User mobile number")
    divName: str = Field(..., description="Division name")
    token: str = Field(..., description="JWT authentication token")
    designation: str = Field(..., description="User designation")
    roleId: int = Field(..., description="Role ID")
    status: int = Field(..., description="User status")
    createdDate: str = Field(..., description="Account creation date")


class LoginAPIResponse(BaseModel):
    """
    Wrapper for the new API response structure
    
    The API returns user data wrapped in a content array with metadata
    """
    mssg: str = Field(..., description="Response message")
    content: List[LoginResponse] = Field(..., description="List of user data (always contains 1 item)")
    status: int = Field(..., description="Response status code")


class AuthState:
    """
    Singleton class to manage authentication state
    """
    _instance = None
    _token: Optional[str] = None
    _user_info: Optional[LoginResponse] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuthState, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def set_auth(cls, token: str, user_info: LoginResponse):
        """Store authentication token and user info"""
        instance = cls()
        instance._token = token
        instance._user_info = user_info
    
    @classmethod
    def get_token(cls) -> Optional[str]:
        """Get stored authentication token"""
        instance = cls()
        return instance._token
    
    @classmethod
    def get_user_info(cls) -> Optional[LoginResponse]:
        """Get stored user information"""
        instance = cls()
        return instance._user_info
    
    @classmethod
    def clear(cls):
        """Clear authentication state"""
        instance = cls()
        instance._token = None
        instance._user_info = None
    
    @classmethod
    def is_authenticated(cls) -> bool:
        """Check if user is authenticated"""
        instance = cls()
        return instance._token is not None

