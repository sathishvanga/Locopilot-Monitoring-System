"""
HTTP API client with retry logic and authentication
"""

import time
from typing import Optional, Dict, Any, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests import Response

from .logger import get_logger
from .config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class APIClient:
    """
    HTTP client wrapper with automatic retry and token management
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url or settings.api_base_url
        self.timeout = timeout or settings.request_timeout
        self.max_retries = max_retries or settings.max_retries
        
        # Create session with retry strategy
        self.session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"API client initialized - Base URL: {self.base_url}")
    
    def _get_headers(self, token: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Build request headers
        
        Args:
            token: Authentication token
            extra_headers: Additional headers to include
            
        Returns:
            Dict[str, str]: Request headers
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        if extra_headers:
            headers.update(extra_headers)
        
        return headers
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Response:
        """
        Make GET request
        
        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            token: Authentication token
            timeout: Request timeout
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(token)
        timeout = timeout or self.timeout
        
        logger.debug(f"GET {url} - Params: {params}")
        
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            logger.debug(f"GET {url} - Status: {response.status_code}")
            return response
            
        except requests.RequestException as e:
            logger.error(f"GET {url} failed: {e}")
            raise
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Response:
        """
        Make POST request
        
        Args:
            endpoint: API endpoint (relative to base_url)
            data: Form data
            json: JSON payload
            files: Files to upload
            token: Authentication token
            timeout: Request timeout
            progress_callback: Callback for upload progress (bytes_sent, total_bytes)
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        
        # Adjust headers for file uploads
        if files:
            headers = {"Accept": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
        else:
            headers = self._get_headers(token)
        
        timeout = timeout or self.timeout
        
        logger.debug(f"POST {url} - Has files: {files is not None}")
        
        try:
            response = self.session.post(
                url,
                data=data,
                json=json,
                files=files,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            logger.debug(f"POST {url} - Status: {response.status_code}")
            return response
            
        except requests.RequestException as e:
            logger.error(f"POST {url} failed: {e}")
            raise
    
    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Response:
        """
        Make PUT request
        
        Args:
            endpoint: API endpoint (relative to base_url)
            data: Form data
            json: JSON payload
            token: Authentication token
            timeout: Request timeout
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(token)
        timeout = timeout or self.timeout
        
        logger.debug(f"PUT {url}")
        
        try:
            response = self.session.put(
                url,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            logger.debug(f"PUT {url} - Status: {response.status_code}")
            return response
            
        except requests.RequestException as e:
            logger.error(f"PUT {url} failed: {e}")
            raise
    
    def delete(
        self,
        endpoint: str,
        token: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Response:
        """
        Make DELETE request
        
        Args:
            endpoint: API endpoint (relative to base_url)
            token: Authentication token
            timeout: Request timeout
            
        Returns:
            requests.Response: HTTP response
            
        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(token)
        timeout = timeout or self.timeout
        
        logger.debug(f"DELETE {url}")
        
        try:
            response = self.session.delete(
                url,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            logger.debug(f"DELETE {url} - Status: {response.status_code}")
            return response
            
        except requests.RequestException as e:
            logger.error(f"DELETE {url} failed: {e}")
            raise

