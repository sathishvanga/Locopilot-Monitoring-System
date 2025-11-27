"""
Authentication exceptions

Custom exception classes for authentication-related errors.
"""

from typing import Optional


class AuthenticationError(Exception):
    """
    Base exception for authentication errors.
    
    Attributes:
        message: Human-readable error message
        username: Username that caused the error (if applicable)
    """
    
    def __init__(self, message: str, username: Optional[str] = None):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            username: Username (optional)
        """
        self.message = message
        self.username = username
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.username:
            return f"{self.message} (User: {self.username})"
        return self.message


class LoginError(AuthenticationError):
    """
    Exception raised when login fails.
    
    Used for general login failures that don't fit other categories.
    """
    pass


class InvalidCredentialsError(AuthenticationError):
    """
    Exception raised when credentials are invalid.
    
    Used when username/password combination is incorrect.
    """
    pass


class SessionExpiredError(AuthenticationError):
    """
    Exception raised when user session has expired.
    
    Used when authentication token is no longer valid.
    """
    pass

