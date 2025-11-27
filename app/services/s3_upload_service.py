"""
S3 Upload Service - Handles file uploads to S3 via remote API

This service centralizes all S3 upload logic in the backend.
"""

import os
from typing import Optional, List, Tuple
from pathlib import Path
import requests

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..exceptions import S3UploadError


logger = get_logger(__name__)
settings = get_settings()


class S3UploadService:
    """
    Service for uploading files to S3 via remote API
    """
    
    def __init__(self):
        """Initialize S3 upload service"""
        self.api_url = "https://api.mindcoinapps.com/ai_demo_api/amazonUpload/uploadWithFolder"
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
            
            # Prepare multipart form data
            with open(file_path, 'rb') as f:
                files = {
                    'file': (file_name, f, 'application/octet-stream')
                }
                
                data = {
                    'subFolderName': subfolder
                }
                
                # Prepare headers
                headers = {}
                if auth_token:
                    headers['Authorization'] = f'Bearer {auth_token}'
                
                # Make upload request
                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract S3 URL
            if "url" in result:
                s3_url = result["url"]
            elif isinstance(result, dict) and "data" in result:
                s3_url = result["data"].get("url")
            else:
                logger.error(f"Unexpected S3 response: {result}")
                return False, None, "Invalid response from S3 upload API"
            
            logger.info(f"Successfully uploaded {file_name} to S3: {s3_url}")
            
            return True, s3_url, None
            
        except requests.HTTPError as e:
            logger.error(f"S3 upload HTTP error for {file_path}: {e}")
            status_code = e.response.status_code if e.response else None
            error_msg = f"Upload failed (HTTP {status_code})"
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


def get_s3_upload_service() -> S3UploadService:
    """
    Get S3 upload service (singleton)
    
    Returns:
        S3UploadService: S3 upload service instance
    """
    global _s3_service
    if _s3_service is None:
        _s3_service = S3UploadService()
    return _s3_service

