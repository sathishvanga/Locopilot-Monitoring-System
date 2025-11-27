"""
Authentication service for remote API
"""

from typing import Optional
import requests

from ..models.auth_models import LoginRequest, LoginResponse, LoginAPIResponse, AuthState
from ..utils.api_client import APIClient
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class AuthService:
    """
    Service for user authentication
    
    Handles login, token management, and authentication state
    """
    
    def __init__(self):
        """Initialize authentication service"""
        self.api_client = APIClient(base_url=settings.api_base_url)
        self.auth_state = AuthState()
        logger.info("Authentication service initialized")
    
    def login(self, username: str, password: str) -> tuple[bool, Optional[LoginResponse], Optional[str]]:
        """
        Authenticate user with remote API
        
        Args:
            username: Mobile number or username
            password: User password
            
        Returns:
            tuple[bool, Optional[LoginResponse], Optional[str]]: 
                (success, user_info, error_message)
        """
        try:
            # Validate inputs
            if not username or not username.strip():
                return False, None, "Mobile number is required"
            
            if not password or not password.strip():
                return False, None, "Password is required"
            
            # Create login request
            login_request = LoginRequest(
                username=username.strip(),
                password=password.strip(),
                osType=1,  # Desktop
                captchaToken=""  # Skip captcha for desktop
            )
            
            logger.info(f"Attempting login for user: {username}")
            
            # Make API request
            response = self.api_client.post(
                endpoint="/auth/user/loginByMobilePassword",
                json=login_request.model_dump()
            )
            
            # Parse response
            response_data = response.json()
            
            # Validate response structure - new API returns wrapped response
            if "content" not in response_data or "status" not in response_data:
                logger.error(f"Invalid response structure: {response_data}")
                return False, None, "Invalid response from server"
            
            # Parse the wrapped API response
            api_response = LoginAPIResponse(**response_data)
            
            # Check if login was successful
            if api_response.status != 1:
                error_msg = api_response.mssg or "Login failed"
                logger.error(f"Login failed: {error_msg}")
                return False, None, error_msg
            
            # Extract user info from content array (should have exactly 1 item)
            if not api_response.content or len(api_response.content) == 0:
                logger.error("No user data in response")
                return False, None, "No user data received from server"
            
            user_info = api_response.content[0]
            
            # Store authentication state
            self.auth_state.set_auth(user_info.token, user_info)
            
            logger.info(f"Login successful for user: {user_info.name} ({user_info.mobileNumber})")
            
            return True, user_info, None
            
        except requests.HTTPError as e:
            logger.error(f"Login HTTP error: {e}")
            
            # Parse error message from response
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", "Invalid credentials")
            except:
                if e.response.status_code == 401:
                    error_msg = "Invalid mobile number or password"
                elif e.response.status_code == 403:
                    error_msg = "Account is disabled or not authorized"
                else:
                    error_msg = f"Authentication failed (HTTP {e.response.status_code})"
            
            return False, None, error_msg
            
        except requests.Timeout:
            logger.error("Login request timed out")
            return False, None, "Connection timeout - please check your internet connection"
            
        except requests.ConnectionError:
            logger.error("Login connection error")
            return False, None, "Cannot connect to server - please check your internet connection"
            
        except Exception as e:
            logger.error(f"Unexpected login error: {e}", exc_info=True)
            return False, None, f"Login failed: {str(e)}"
    
    def logout(self) -> None:
        """
        Logout user and clear authentication state
        """
        logger.info("User logged out")
        self.auth_state.clear()
    
    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        return self.auth_state.is_authenticated()
    
    def get_token(self) -> Optional[str]:
        """
        Get current authentication token
        
        Returns:
            Optional[str]: Authentication token or None
        """
        return self.auth_state.get_token()
    
    def get_user_info(self) -> Optional[LoginResponse]:
        """
        Get current user information
        
        Returns:
            Optional[LoginResponse]: User info or None
        """
        return self.auth_state.get_user_info()

