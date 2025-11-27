"""
Data models for API communication
"""

from .auth_models import LoginRequest, LoginResponse, LoginAPIResponse
from .trip_models import TripModel, TripsAPIResponse

__all__ = ["LoginRequest", "LoginResponse", "LoginAPIResponse", "TripModel", "TripsAPIResponse"]

