"""
S3 Upload Service - Handles file uploads to S3 via remote API

This service centralizes all S3 upload logic in the backend.
"""

import os
import threading
import time
from typing import Optional, List, Tuple
from pathlib import Path
import requests

from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


RETRIABLE_STATUS_CODES = {429, 500, 502, 503}
MAX_RETRIES = 3
BACKOFF_BASE = 1  # Base delay in seconds for exponential backoff


class S3UploadService:
    """
    Service for uploading files to S3 via remote API
    """

    def __init__(self):
        """Initialize S3 upload service"""
        self.settings = settings
        self.api_url = self.settings.s3_upload_api_url
        self.timeout = 300  # 5 minutes timeout for uploads
        logger.info("S3 upload service initialized")
    
    def upload_file(
        self,
        file_path: str,
        subfolder: str = "cvvr",
        auth_token: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a single file to S3
        
        Args:
            file_path: Local file path to upload
            subfolder: S3 subfolder name (default: "cvvr")
            auth_token: Optional authentication token
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: 
                (success, s3_url, error_message)
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return False, None, f"File not found: {file_path}"
            
            file_name = Path(file_path).name
            file_size = os.path.getsize(file_path)
            
            logger.info(
                f"Uploading {file_name} ({file_size / (1024**2):.2f} MB) "
                f"to S3 subfolder '{subfolder}'"
            )
            
            # Prepare headers - Use Authorization header (same as desktop app)
            headers = {}
            if auth_token:
                # Use Bearer token in Authorization header (same format as desktop app)
                headers['Authorization'] = f'Bearer {auth_token}'
                logger.debug(f"S3 upload with Bearer token (length: {len(auth_token)})")
            else:
                logger.warning("S3 upload attempted without auth token")

            # Upload with exponential backoff retry for retriable errors
            response = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Re-open file each attempt so the stream starts from the beginning
                    with open(file_path, 'rb') as f:
                        files = {
                            'file': (file_name, f, 'application/octet-stream')
                        }
                        data = {
                            'subFolderName': subfolder
                        }
                        response = requests.post(
                            self.api_url,
                            files=files,
                            data=data,
                            headers=headers,
                            timeout=self.timeout
                        )

                    # Check if we got a retriable status code
                    if response.status_code in RETRIABLE_STATUS_CODES:
                        logger.warning(
                            f"S3 upload attempt {attempt}/{MAX_RETRIES} returned "
                            f"status {response.status_code} for {file_name}"
                        )
                        if attempt < MAX_RETRIES:
                            delay = BACKOFF_BASE * (2 ** (attempt - 1))
                            logger.info(f"Retrying in {delay}s...")
                            time.sleep(delay)
                            continue
                        # Final attempt still retriable -- raise so outer handler catches it
                        response.raise_for_status()

                    # Non-retriable status -- raise immediately if error
                    response.raise_for_status()
                    # Success -- break out of retry loop
                    break

                except (requests.ConnectionError, requests.Timeout) as exc:
                    logger.warning(
                        f"S3 upload attempt {attempt}/{MAX_RETRIES} failed "
                        f"with {type(exc).__name__} for {file_name}"
                    )
                    if attempt < MAX_RETRIES:
                        delay = BACKOFF_BASE * (2 ** (attempt - 1))
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    # Exhausted retries -- re-raise so outer handlers catch it
                    raise

            # Parse response
            result = response.json()
            s3_url = None
            
            # Check for error in response
            if isinstance(result, dict):
                # Check for error status
                if result.get("status") != 1 and "mssg" in result:
                    error_msg = result.get("mssg", "Upload failed")
                    logger.error(f"S3 upload error: {error_msg} - Full response: {result}")
                    return False, None, error_msg
                
                # Extract S3 URL - try multiple possible response formats
                # Format 1: Direct url field
                if "url" in result:
                    s3_url = result["url"]
                # Format 2: url in content object (actual API format)
                elif "content" in result and isinstance(result["content"], dict):
                    s3_url = result["content"].get("url")
                # Format 3: url in data object (alternative format)
                elif "data" in result and isinstance(result["data"], dict):
                    s3_url = result["data"].get("url")
                else:
                    logger.error(f"Could not extract S3 URL from response: {result}")
                    return False, None, "Invalid S3 API response format. Response: " + str(result)
            else:
                logger.error(f"Unexpected S3 response type: {type(result)}")
                return False, None, "Invalid response from S3 upload API"
            
            if not s3_url:
                logger.error(f"Could not extract S3 URL from response: {result}")
                return False, None, "Invalid S3 API response format. Response: " + str(result)
            
            logger.info(f"Successfully uploaded {file_name} to S3: {s3_url}")
            
            return True, s3_url, None
            
        except requests.HTTPError as e:
            logger.error(f"S3 upload HTTP error for {file_path}: {e}")
            # Try to extract error message from response
            error_msg = f"Upload failed (HTTP {e.response.status_code})"
            try:
                error_data = e.response.json()
                if isinstance(error_data, dict) and "mssg" in error_data:
                    error_msg = error_data.get("mssg", error_msg)
            except (ValueError, KeyError):
                pass
            return False, None, error_msg
            
        except requests.Timeout:
            logger.error(f"S3 upload timeout for {file_path}")
            return False, None, "Upload timed out"
            
        except requests.ConnectionError:
            logger.error(f"S3 upload connection error for {file_path}")
            return False, None, "Cannot connect to S3 upload service"
            
        except Exception as e:
            logger.error(f"Unexpected S3 upload error for {file_path}: {e}", exc_info=True)
            return False, None, f"Upload failed: {str(e)}"
    
    def upload_multiple_files(
        self,
        file_paths: List[str],
        subfolder: str = "cvvr",
        auth_token: Optional[str] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Upload multiple files to S3
        
        Args:
            file_paths: List of local file paths to upload
            subfolder: S3 subfolder name
            auth_token: Optional authentication token
            
        Returns:
            Tuple[bool, List[str], List[str]]: 
                (all_success, successful_urls, error_messages)
        """
        logger.info(f"Uploading {len(file_paths)} files to S3")
        
        successful_urls = []
        error_messages = []
        
        for file_path in file_paths:
            success, s3_url, error = self.upload_file(file_path, subfolder, auth_token)
            
            if success and s3_url:
                successful_urls.append(s3_url)
            else:
                error_msg = error or "Unknown error"
                error_messages.append(f"{Path(file_path).name}: {error_msg}")
        
        all_success = len(error_messages) == 0
        
        logger.info(
            f"Upload batch completed - "
            f"Success: {len(successful_urls)}/{len(file_paths)}, "
            f"Errors: {len(error_messages)}"
        )
        
        return all_success, successful_urls, error_messages


# Singleton instance
_s3_service: Optional[S3UploadService] = None
_s3_service_lock = threading.Lock()


def get_s3_upload_service() -> S3UploadService:
    """
    Get S3 upload service (singleton).

    M-25: Thread-safe double-checked locking pattern.

    Returns:
        S3UploadService: S3 upload service instance
    """
    global _s3_service
    if _s3_service is None:
        with _s3_service_lock:
            if _s3_service is None:
                _s3_service = S3UploadService()
    return _s3_service

