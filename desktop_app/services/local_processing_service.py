"""
Local video processing service - Interface with FastAPI backend
"""

import os
import time
import subprocess
import socket
from typing import Optional, List
from pathlib import Path
import requests

from ..models.trip_models import ProcessingResult
from ..utils.api_client import APIClient
from ..utils.logger import get_logger
from ..utils.config import get_settings


logger = get_logger(__name__)
settings = get_settings()


class LocalProcessingService:
    """
    Service for local video processing
    
    Interfaces with the local FastAPI backend to process videos
    and extract evidence clips
    """
    
    def __init__(self):
        """Initialize local processing service"""
        self.local_api = APIClient(base_url=settings.local_backend_url)
        self.backend_warning_shown = False  # Track if backend warning was shown (persists across controller recreations)
        logger.info("Local processing service initialized")
    
    def is_backend_running(self) -> bool:
        """
        Check if local FastAPI backend is running
        
        Returns:
            bool: True if backend is running, False otherwise
        """
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', settings.local_backend_port))
            sock.close()
            
            if result == 0:
                # Port is open, check if it's our API
                try:
                    response = requests.get(
                        f"{settings.local_backend_url}/health",
                        timeout=5
                    )
                    return response.status_code == 200
                except:
                    return False
            
            return False
            
        except Exception as e:
            logger.debug(f"Backend check error: {e}")
            return False
    
    def wait_for_backend(self, timeout: int = 30) -> bool:
        """
        Wait for backend to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if backend became available, False if timeout
        """
        logger.info(f"Waiting for local backend (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_backend_running():
                logger.info("Local backend is ready")
                return True
            time.sleep(1)
        
        logger.warning("Timeout waiting for local backend")
        return False
    
    def process_and_upload_video(
        self,
        video_path: str,
        trip_id: str,
        auth_token: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process video and upload to S3 using local FastAPI backend
        
        This is the new preferred method - backend handles everything including S3 upload.
        
        Args:
            video_path: Path to video file
            trip_id: Trip UUID
            auth_token: Optional authentication token for S3 upload
            
        Returns:
            ProcessingResult: Processing result with S3 URLs
        """
        try:
            # Check if backend is running
            if not self.is_backend_running():
                logger.error("Local backend is not running")
                return ProcessingResult(
                    success=False,
                    error="Local processing backend is not running. Please start the backend service."
                )
            
            # Validate video file
            if not os.path.exists(video_path):
                return ProcessingResult(
                    success=False,
                    error=f"Video file not found: {video_path}"
                )
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                return ProcessingResult(
                    success=False,
                    error="Video file is empty"
                )
            
            logger.info(f"Processing and uploading video: {video_path} (Trip: {trip_id})")
            
            # Prepare multipart upload
            with open(video_path, 'rb') as f:
                files = {
                    'video_file': (os.path.basename(video_path), f, 'video/mp4')
                }
                
                data = {
                    'tripId': trip_id,
                    'subFolderName': 'cvvr'
                }
                
                # Add auth token if provided
                if auth_token:
                    data['authToken'] = auth_token
                
                # Make processing + upload request to NEW endpoint
                response = self.local_api.post(
                    endpoint="/api/v1/video/process-and-upload",  # ← New endpoint
                    data=data,
                    files=files,
                    timeout=settings.processing_timeout
                )
            
            # Parse response
            result_data = response.json()
            
            logger.debug(f"Processing and upload response: {result_data}")
            
            # Extract information from response
            if result_data.get("status") != "success":
                error_msg = result_data.get("message", "Processing and upload failed")
                return ProcessingResult(
                    success=False,
                    error=error_msg
                )
            
            # Get data from response
            data = result_data.get("data", {})
            run_dir = data.get("run_dir")
            activities_count = data.get("activities_count", 0)
            video_url = data.get("video_url")
            evidence_urls = data.get("evidence_clips", [])
            clips_uploaded = data.get("clips_uploaded", 0)
            upload_errors = data.get("upload_errors", [])
            
            if upload_errors:
                logger.warning(f"Some clips failed to upload: {upload_errors}")
            
            logger.info(
                f"Processing and upload completed - "
                f"Activities: {activities_count}, "
                f"Video URL: {video_url}, "
                f"Clips uploaded: {clips_uploaded}"
            )
            
            return ProcessingResult(
                success=True,
                run_dir=run_dir,
                activities_count=activities_count,
                video_url=video_url,
                evidence_urls=evidence_urls,
                clips_uploaded=clips_uploaded,
                clip_files=[]  # No longer needed as files are on S3
            )
            
        except requests.HTTPError as e:
            logger.error(f"Processing HTTP error: {e}")
            error_msg = f"Processing failed (HTTP {e.response.status_code})"
            
            try:
                error_data = e.response.json()
                error_msg = error_data.get("message", error_msg) or error_data.get("detail", error_msg)
            except:
                pass
            
            return ProcessingResult(success=False, error=error_msg)
            
        except requests.Timeout:
            logger.error("Processing timeout")
            return ProcessingResult(
                success=False,
                error="Processing timed out - video may be too long or complex"
            )
            
        except requests.ConnectionError:
            logger.error("Connection error to local backend")
            return ProcessingResult(
                success=False,
                error="Cannot connect to local processing backend"
            )
            
        except Exception as e:
            logger.error(f"Unexpected processing error: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                error=f"Processing failed: {str(e)}"
            )
    
    def get_evidence_clips(self, run_dir: str) -> List[str]:
        """
        Get list of evidence clip files from processing output
        
        Args:
            run_dir: Processing run directory
            
        Returns:
            List[str]: List of evidence clip file paths
        """
        clips_dir = os.path.join(run_dir, "clips")
        
        if not os.path.exists(clips_dir):
            logger.warning(f"Clips directory not found: {clips_dir}")
            return []
        
        # Find all video files in clips directory
        clip_files = []
        for file in os.listdir(clips_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                clip_files.append(os.path.join(clips_dir, file))
        
        clip_files.sort()
        logger.info(f"Found {len(clip_files)} evidence clips in {clips_dir}")
        
        return clip_files

