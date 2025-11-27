"""
Backend health check utilities

Shared functions for checking backend health status.
"""

import socket
from typing import Optional
import requests

from .logger import get_logger
from .config import get_settings


logger = get_logger(__name__)
settings = get_settings()


def check_backend_health(
    backend_url: Optional[str] = None,
    backend_port: Optional[int] = None,
    timeout: int = 5
) -> bool:
    """
    Check if local FastAPI backend is running and healthy.
    
    This function performs both port connectivity check and HTTP health check.
    
    Args:
        backend_url: Backend URL (default: from settings)
        backend_port: Backend port (default: from settings)
        timeout: Timeout for health check in seconds
        
    Returns:
        bool: True if backend is running and healthy, False otherwise
    """
    # Use settings defaults if not provided
    if backend_url is None:
        backend_url = settings.local_backend_url
    if backend_port is None:
        backend_port = settings.local_backend_port
    
    try:
        # Step 1: Check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', backend_port))
        sock.close()
        
        if result != 0:
            # Port is not open
            logger.debug(f"Backend port {backend_port} is not open")
            return False
        
        # Step 2: Check if it's our API with health endpoint
        try:
            health_url = f"{backend_url}/health"
            response = requests.get(health_url, timeout=timeout)
            is_healthy = response.status_code == 200
            logger.debug(f"Backend health check: {health_url} - Status: {response.status_code}")
            return is_healthy
        except requests.RequestException as e:
            logger.debug(f"Backend health check failed: {e}")
            return False
        
    except Exception as e:
        logger.debug(f"Backend health check error: {e}")
        return False

