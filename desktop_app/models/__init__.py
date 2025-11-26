"""
Data models for API communication
"""

from .auth_models import LoginRequest, LoginResponse
from .trip_models import TripModel

__all__ = ["LoginRequest", "LoginResponse", "TripModel"]

