"""
Trip service for fetching and managing CVVR trips
"""

from typing import List, Optional
import requests

from ..models.trip_models import TripModel, TripsAPIResponse
from ..utils.api_client import APIClient
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class TripService:
    """
    Service for managing CVVR trips
    
    Handles fetching pending trips and trip status updates
    """
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize trip service
        
        Args:
            auth_token: Authentication token for API requests
        """
        self.api_client = APIClient(base_url=settings.api_base_url)
        self.auth_token = auth_token
        logger.info("Trip service initialized")
    
    def set_auth_token(self, token: str) -> None:
        """
        Set authentication token
        
        Args:
            token: JWT authentication token
        """
        self.auth_token = token
        logger.debug("Authentication token updated")
    
    def get_pending_trips(self) -> tuple[bool, List[TripModel], Optional[str]]:
        """
        Fetch all pending trips from API
        
        Returns:
            tuple[bool, List[TripModel], Optional[str]]: 
                (success, trips_list, error_message)
        """
        try:
            if not self.auth_token:
                logger.error("Cannot fetch trips - no authentication token")
                return False, [], "Authentication required - please login"
            
            logger.info("Fetching pending trips from API")
            
            # Make API request
            response = self.api_client.get(
                endpoint="/cvvr/cvvrTrips/getAllPendingTrips",
                token=self.auth_token
            )
            
            # Parse response
            response_data = response.json()
            
            # Handle new wrapped response structure
            if isinstance(response_data, dict) and "content" in response_data and "status" in response_data:
                # New API format with wrapped response
                try:
                    api_response = TripsAPIResponse(**response_data)
                    
                    # Check if request was successful
                    if api_response.status != 1:
                        error_msg = api_response.mssg or "Failed to fetch trips"
                        logger.error(f"API returned error: {error_msg}")
                        return False, [], error_msg
                    
                    trips = api_response.content
                    logger.info(f"Successfully fetched {len(trips)} pending trips")
                    return True, trips, None
                    
                except Exception as e:
                    logger.error(f"Failed to parse wrapped API response: {e}")
                    return False, [], f"Invalid response format: {str(e)}"
            
            # Handle legacy response structures (for backward compatibility)
            elif isinstance(response_data, list):
                trips_data = response_data
            elif isinstance(response_data, dict) and "data" in response_data:
                trips_data = response_data["data"]
            elif isinstance(response_data, dict) and "trips" in response_data:
                trips_data = response_data["trips"]
            else:
                logger.warning(f"Unexpected response structure: {response_data}")
                return False, [], "Unexpected response format from server"
            
            # Parse legacy format into TripModel objects
            trips = []
            for trip_dict in trips_data:
                try:
                    trip = TripModel(**trip_dict)
                    trips.append(trip)
                except Exception as e:
                    logger.warning(f"Failed to parse trip: {trip_dict} - Error: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(trips)} pending trips")
            return True, trips, None
            
        except requests.HTTPError as e:
            logger.error(f"Failed to fetch trips (HTTP error): {e}")
            
            if e.response.status_code == 401:
                error_msg = "Session expired - please login again"
            elif e.response.status_code == 403:
                error_msg = "Access denied - insufficient permissions"
            else:
                error_msg = f"Failed to fetch trips (HTTP {e.response.status_code})"
            
            return False, [], error_msg
            
        except requests.Timeout:
            logger.error("Request timed out while fetching trips")
            return False, [], "Request timed out - please try again"
            
        except requests.ConnectionError:
            logger.error("Connection error while fetching trips")
            return False, [], "Cannot connect to server - please check your internet connection"
            
        except Exception as e:
            logger.error(f"Unexpected error fetching trips: {e}", exc_info=True)
            return False, [], f"Failed to fetch trips: {str(e)}"
    
    def refresh_trips(self) -> tuple[bool, List[TripModel], Optional[str]]:
        """
        Refresh trips list (alias for get_pending_trips)
        
        Returns:
            tuple[bool, List[TripModel], Optional[str]]: 
                (success, trips_list, error_message)
        """
        logger.info("Refreshing trips list")
        return self.get_pending_trips()

