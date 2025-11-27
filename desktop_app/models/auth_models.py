"""
Authentication models for API communication
"""

from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    pass

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
    Thread-safe singleton class to manage authentication state.
    
    Uses a lock to ensure thread safety in multi-threaded environments.
    """
    _instance: Optional['AuthState'] = None
    _lock = None  # Will be initialized on first use
    
    def __init__(self):
        """Initialize authentication state"""
        import threading
        if not hasattr(self, '_token'):  # Only initialize once
            if AuthState._lock is None:
                AuthState._lock = threading.Lock()
            self._token: Optional[str] = None
            self._user_info: Optional[LoginResponse] = None
    
    def __new__(cls) -> 'AuthState':
        """Create singleton instance with thread safety"""
        import threading
        if cls._instance is None:
            # Use class-level lock for instance creation
            if not hasattr(cls, '_creation_lock'):
                cls._creation_lock = threading.Lock()
            with cls._creation_lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(AuthState, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def set_auth(cls, token: str, user_info: LoginResponse) -> None:
        """
        Store authentication token and user info (thread-safe).
        
        Args:
            token: Authentication token
            user_info: User information
        """
        instance = cls()
        with instance._lock:
            instance._token = token
            instance._user_info = user_info
    
    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get stored authentication token (thread-safe).
        
        Returns:
            Optional[str]: Authentication token or None
        """
        instance = cls()
        with instance._lock:
            return instance._token
    
    @classmethod
    def get_user_info(cls) -> Optional[LoginResponse]:
        """
        Get stored user information (thread-safe).
        
        Returns:
            Optional[LoginResponse]: User info or None
        """
        instance = cls()
        with instance._lock:
            return instance._user_info
    
    @classmethod
    def clear(cls) -> None:
        """Clear authentication state (thread-safe)"""
        instance = cls()
        with instance._lock:
            instance._token = None
            instance._user_info = None
    
    @classmethod
    def is_authenticated(cls) -> bool:
        """
        Check if user is authenticated (thread-safe).
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        instance = cls()
        with instance._lock:
            return instance._token is not None

