"""
Video processing service - Orchestrates video upload and activity detection

This service handles the main business logic for processing uploaded videos.
"""

import os
import shutil
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..repositories.activity_repository import ActivityRepository
from .activity_detection_service import ActivityDetectionService
from .external_api_service import get_external_api_service


logger = get_logger(__name__)
settings = get_settings()


class VideoProcessingService:
    """
    Service for processing uploaded videos and detecting activities
    
    Orchestrates the complete workflow:
    1. Save uploaded video to disk
    2. Run activity detection
    3. Save results to activities.json
    4. Return processing results
    """
    
    def __init__(
        self,
        activity_repository: Optional[ActivityRepository] = None,
        activity_detection_service: Optional[ActivityDetectionService] = None
    ):
        """
        Initialize the video processing service
        
        Args:
            activity_repository: Repository for activity persistence
            activity_detection_service: Service for activity detection
        """
        self.activity_repository = activity_repository or ActivityRepository(
            output_dir=settings.output_dir
        )
        self.activity_detection_service = activity_detection_service or ActivityDetectionService()
        
        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        logger.info(
            f"🚀 Video processing service initialized - "
            f"Output dir: {settings.output_dir}, Upload dir: {settings.upload_dir}"
        )
    
    def validate_video_file(self, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded video file
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            
        Returns:
            tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            return False, f"Invalid file extension. Allowed: {', '.join(settings.allowed_video_extensions)}"
        
        # Check file size
        if file_size > settings.max_upload_size:
            max_size_mb = settings.max_upload_size / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_size_mb:.0f} MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, None
    
    async def save_uploaded_video(
        self,
        file_content: bytes,
        filename: str,
        trip_id: str
    ) -> str:
        """
        Save uploaded video file to disk
        
        Args:
            file_content: Video file content
            filename: Original filename
            trip_id: Trip identifier (used in saved filename)
            
        Returns:
            str: Path to saved video file
            
        Raises:
            IOError: If file saving fails
        """
        try:
            # Create safe filename
            file_ext = os.path.splitext(filename)[1].lower()
            safe_filename = f"{trip_id}_{int(time.time())}{file_ext}"
            file_path = os.path.join(settings.upload_dir, safe_filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Video saved to {file_path} ({len(file_content)} bytes)")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save video: {e}", exc_info=True)
            raise IOError(f"Failed to save video: {str(e)}") from e
    
    def process_video(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Dict[str, Dict[str, str]] = None,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1,
        use_mock_detection: bool = False,
        use_multiprocessing: bool = False,
        save_clips: bool = True
    ) -> Dict[str, Any]:
        """
        Process video and detect activities
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_members: Dictionary mapping role (LP/ALP) to crew member info {name, id, role}
            crew_name: [Legacy] Crew member name
            crew_id: [Legacy] Crew member ID
            crew_role: [Legacy] Crew role
            use_mock_detection: Use mock detection instead of real ML models
            use_multiprocessing: Enable multiprocessing for faster processing
            save_clips: Whether to save video clips and images (default: True)
            
        Returns:
            Dict[str, Any]: Processing results with activities
            
        Raises:
            Exception: If processing fails
        """
        start_time = time.time()
        
        try:
            logger.info(
                f"🎬 Starting video processing for trip {trip_id} - "
                f"Multiprocessing: {'enabled' if use_multiprocessing else 'disabled'}, "
                f"Save clips: {save_clips}, Mock detection: {use_mock_detection}"
            )
            
            # Validate video file exists
            if not os.path.exists(video_path):
                logger.error(f"❌ Video file not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            logger.info(f"✅ Video file validated: {video_path}")
            
            # Create run directory ONCE at the top level
            run_dir = self.activity_repository.create_run_directory(base_name=f"run")
            logger.info(f"📁 Created run directory: {run_dir}")
            
            # Run activity detection
            if use_mock_detection:
                logger.info("🎭 Using mock activity detection")
                activities = self.activity_detection_service.detect_activities_mock(
                    video_path=video_path,
                    trip_id=trip_id,
                    crew_members=crew_members,
                    crew_name=crew_name,
                    crew_id=crew_id,
                    crew_role=crew_role
                )
            else:
                logger.info(
                    f"🔍 Using real activity detection - "
                    f"Multiprocessing: {'enabled' if use_multiprocessing else 'disabled'}"
                )
                
                # Pass run_dir and save_clips settings
                if use_multiprocessing:
                    activities = self.activity_detection_service._detect_activities_multiprocess(
                        video_path=video_path,
                        trip_id=trip_id,
                        crew_members=crew_members,
                        crew_name=crew_name,
                        crew_id=crew_id,
                        crew_role=crew_role,
                        output_dir=settings.output_dir,
                        sample_fps=settings.sample_fps,
                        run_dir=run_dir,  # Pass existing run_dir to avoid nested directories
                        save_clips=save_clips
                    )
                else:
                    activities = self.activity_detection_service._detect_activities_single_process(
                        video_path=video_path,
                        trip_id=trip_id,
                        crew_members=crew_members,
                        crew_name=crew_name,
                        crew_id=crew_id,
                        crew_role=crew_role,
                        output_dir=settings.output_dir,
                        sample_fps=settings.sample_fps,
                        run_dir=run_dir  # Pass existing run_dir
                    )
            
            # Save activities to JSON
            activities_json_path = self.activity_repository.save_activities(
                activities=activities,
                run_dir=run_dir
            )
            logger.info(f"💾 Saved {len(activities)} activities to {activities_json_path}")
            
            # Post results to external API (non-blocking, errors don't fail the job)
            api_result = None
            try:
                logger.info(f"🌐 Attempting to post results to external API...")
                external_api_service = get_external_api_service()
                
                # Extract run_id from run_dir for constructing job_id
                run_id = os.path.basename(run_dir)
                
                # Post to external API
                api_result = external_api_service.post_cvvr_results(
                    trip_id=trip_id,
                    events=activities,
                    job_id=run_id,
                    host_url=settings.host_url
                )
                
                if api_result.get("success"):
                    logger.info(
                        f"✅ [external_api] Successfully posted {api_result.get('violations_count', 0)} "
                        f"violations to external API for trip {trip_id}"
                    )
                else:
                    logger.warning(
                        f"⚠️ [external_api] Failed to post to external API: {api_result.get('message')}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ [external_api] Exception while posting to external API: {e}",
                    exc_info=True
                )
                api_result = {
                    "success": False,
                    "message": f"Exception: {str(e)}",
                    "posted": False
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate summary
            summary = self.activity_repository.get_activity_summary(activities)
            
            logger.info(
                f"✅ Video processing completed in {processing_time:.2f}s - "
                f"Found {len(activities)} activities for trip {trip_id}"
            )
            
            # Build response with API result
            response = {
                "status": "success",
                "message": "Video processed successfully",
                "tripId": trip_id,
                "videoFilename": os.path.basename(video_path),
                "runDirectory": run_dir,
                "activitiesJsonPath": activities_json_path,
                "activitiesCount": len(activities),
                "activities": activities,
                "processingTime": processing_time,
                "summary": summary,
                "multiprocessingEnabled": use_multiprocessing,
                "clipsGenerated": save_clips
            }
            
            # Add external API result if available
            if api_result is not None:
                response["externalApiResult"] = api_result
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Video processing failed after {processing_time:.2f}s for trip {trip_id}: {e}",
                exc_info=True
            )
            raise
    
    def cleanup_uploaded_video(self, video_path: str) -> None:
        """
        Clean up uploaded video file after processing
        
        Args:
            video_path: Path to video file to delete
        """
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"🗑️ Cleaned up uploaded video: {video_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup video {video_path}: {e}")
    
    def get_processing_status(self, run_dir: str) -> Dict[str, Any]:
        """
        Get processing status for a run directory
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            Dict[str, Any]: Status information
        """
        try:
            activities_json_path = os.path.join(run_dir, "activities.json")
            
            if not os.path.exists(activities_json_path):
                return {
                    "status": "processing",
                    "message": "Processing in progress"
                }
            
            # Load activities
            activities = self.activity_repository.load_activities(activities_json_path)
            summary = self.activity_repository.get_activity_summary(activities)
            
            return {
                "status": "completed",
                "message": "Processing completed",
                "activitiesCount": len(activities),
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

