"""
S3 upload service for video and evidence files
"""

import os
from typing import Optional, Callable, List
from pathlib import Path
import requests

from ..models.trip_models import S3UploadResponse
from ..utils.api_client import APIClient
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class UploadService:
    """
    Service for uploading files to S3
    
    Handles video and evidence clip uploads with progress tracking
    """
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize upload service
        
        Args:
            auth_token: Authentication token for API requests
        """
        self.api_client = APIClient(base_url=settings.api_base_url)
        self.auth_token = auth_token
        logger.info("Upload service initialized")
    
    def set_auth_token(self, token: str) -> None:
        """
        Set authentication token
        
        Args:
            token: JWT authentication token
        """
        self.auth_token = token
        logger.debug("Authentication token updated")
    
    def upload_file(
        self,
        file_path: str,
        subfolder: str = "cvvr",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a file to S3
        
        Args:
            file_path: Path to file to upload
            subfolder: S3 subfolder name (default: "cvvr")
            progress_callback: Callback for progress updates (bytes_sent, total_bytes)
            
        Returns:
            tuple[bool, Optional[str], Optional[str]]: 
                (success, s3_url, error_message)
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False, None, f"File not found: {file_path}"
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            
            # Check file size
            if file_size > settings.max_file_size:
                max_size_gb = settings.max_file_size / (1024 ** 3)
                return False, None, f"File too large. Maximum: {max_size_gb:.1f} GB"
            
            logger.info(f"Uploading {file_name} ({file_size / (1024**2):.1f} MB) to S3 subfolder '{subfolder}'")
            
            # Prepare multipart form data
            with open(file_path, 'rb') as f:
                files = {
                    'file': (file_name, f, 'application/octet-stream')
                }
                
                data = {
                    'subFolderName': subfolder
                }
                
                # Make upload request
                response = self.api_client.post(
                    endpoint="/amazonUpload/uploadWithFolder",
                    data=data,
                    files=files,
                    token=self.auth_token,
                    timeout=settings.upload_timeout
                )
            
            # Parse response
            response_data = response.json()
            
            # Extract S3 URL
            if "url" in response_data:
                s3_url = response_data["url"]
            elif isinstance(response_data, dict) and "data" in response_data:
                s3_url = response_data["data"].get("url")
            else:
                logger.error(f"Unexpected upload response: {response_data}")
                return False, None, "Invalid response from upload server"
            
            logger.info(f"Successfully uploaded {file_name} to S3: {s3_url}")
            
            return True, s3_url, None
            
        except requests.HTTPError as e:
            logger.error(f"Upload HTTP error: {e}")
            
            if e.response.status_code == 401:
                error_msg = "Session expired - please login again"
            elif e.response.status_code == 413:
                error_msg = "File too large"
            else:
                error_msg = f"Upload failed (HTTP {e.response.status_code})"
            
            return False, None, error_msg
            
        except requests.Timeout:
            logger.error(f"Upload timeout for {file_path}")
            return False, None, "Upload timed out - please try again"
            
        except requests.ConnectionError:
            logger.error(f"Connection error during upload: {file_path}")
            return False, None, "Cannot connect to server - please check your internet connection"
            
        except Exception as e:
            logger.error(f"Unexpected upload error: {e}", exc_info=True)
            return False, None, f"Upload failed: {str(e)}"
    
    def upload_multiple_files(
        self,
        file_paths: List[str],
        subfolder: str = "cvvr",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> tuple[bool, List[str], List[str]]:
        """
        Upload multiple files to S3
        
        Args:
            file_paths: List of file paths to upload
            subfolder: S3 subfolder name
            progress_callback: Callback for progress (file_index, total_files, current_file)
            
        Returns:
            tuple[bool, List[str], List[str]]: 
                (all_success, successful_urls, error_messages)
        """
        logger.info(f"Uploading {len(file_paths)} files to S3")
        
        successful_urls = []
        error_messages = []
        
        for idx, file_path in enumerate(file_paths):
            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(file_paths), os.path.basename(file_path))
            
            # Upload file
            success, s3_url, error = self.upload_file(file_path, subfolder)
            
            if success and s3_url:
                successful_urls.append(s3_url)
            else:
                error_msg = error or "Unknown error"
                error_messages.append(f"{os.path.basename(file_path)}: {error_msg}")
        
        all_success = len(error_messages) == 0
        
        logger.info(
            f"Upload batch completed - "
            f"Success: {len(successful_urls)}/{len(file_paths)}, "
            f"Errors: {len(error_messages)}"
        )
        
        return all_success, successful_urls, error_messages
    
    def validate_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate file before upload
        
        Args:
            file_path: Path to file
            
        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "File is empty"
        
        if file_size > settings.max_file_size:
            max_size_gb = settings.max_file_size / (1024 ** 3)
            return False, f"File too large. Maximum: {max_size_gb:.1f} GB"
        
        # Check file extension for videos
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in settings.allowed_video_extensions:
            return True, None
        
        # Allow other files (evidence clips)
        return True, None

