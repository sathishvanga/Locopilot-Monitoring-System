"""
Video processing service - Orchestrates video upload and activity detection

This service handles the main business logic for processing uploaded videos.
"""

import os
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..repositories.activity_repository import ActivityRepository
from .activity_detection_service import ActivityDetectionService
from .external_api_service import get_external_api_service
from ..exceptions import VideoProcessingError, VideoValidationError, VideoNotFoundError, VideoUploadError
from ..utils.crew_helpers import get_default_crew_name, get_default_crew_id


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
    
    def validate_video_file(self, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded video file
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in settings.allowed_video_extensions:
            error_msg = f"Invalid file extension. Allowed: {', '.join(settings.allowed_video_extensions)}"
            return False, error_msg
        
        # Check file size
        if file_size > settings.max_upload_size:
            max_size_mb = settings.max_upload_size / (1024 * 1024)
            error_msg = f"File too large. Maximum size: {max_size_mb:.0f} MB"
            return False, error_msg
        
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
        Save uploaded video file to disk asynchronously.
        
        Uses asyncio.to_thread to perform file I/O in a thread pool,
        preventing blocking of the event loop.
        
        Args:
            file_content: Video file content
            filename: Original filename
            trip_id: Trip identifier (used in saved filename)
            
        Returns:
            str: Path to saved video file
            
        Raises:
            VideoUploadError: If file saving fails
        """
        import asyncio
        
        def _write_file() -> str:
            """Synchronous file write function to run in thread pool."""
            try:
                # Create safe filename
                file_ext = os.path.splitext(filename)[1].lower()
                safe_filename = f"{trip_id}_{int(time.time())}{file_ext}"
                file_path = os.path.join(settings.upload_dir, safe_filename)
                
                # Ensure upload directory exists
                os.makedirs(settings.upload_dir, exist_ok=True)
                
                # Save file
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                
                logger.info(f"Video saved to {file_path} ({len(file_content)} bytes)")
                
                return file_path
                
            except Exception as e:
                logger.error(f"Failed to save video: {e}", exc_info=True)
                raise VideoUploadError(
                    message=f"Failed to save video: {str(e)}",
                    video_path=os.path.join(settings.upload_dir, filename),
                    trip_id=trip_id
                ) from e
        
        try:
            # Run file I/O in thread pool to avoid blocking event loop
            file_path = await asyncio.to_thread(_write_file)
            return file_path
        except VideoUploadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving video: {e}", exc_info=True)
            raise VideoUploadError(
                message=f"Unexpected error saving video: {str(e)}",
                video_path=os.path.join(settings.upload_dir, filename),
                trip_id=trip_id
            ) from e
    
    def process_video(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Optional[Dict[str, Dict[str, str]]] = None,
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
                raise VideoNotFoundError(
                    message=f"Video file not found: {video_path}",
                    video_path=video_path,
                    trip_id=trip_id
                )
            
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
            
        except (VideoProcessingError, VideoNotFoundError, VideoValidationError):
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Video processing failed after {processing_time:.2f}s for trip {trip_id}: {e}",
                exc_info=True
            )
            raise VideoProcessingError(
                message=f"Video processing failed: {str(e)}",
                video_path=video_path,
                trip_id=trip_id
            ) from e
    
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
    
    def process_and_upload_workflow(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Optional[Dict[str, Dict[str, str]]],
        subfolder_name: str,
        auth_token: Optional[str],
        use_mock_detection: bool,
        use_multiprocessing: bool,
        save_clips: bool,
        s3_upload_service: Any  # S3UploadService type, avoiding circular import
    ) -> Dict[str, Any]:
        """
        Complete workflow: process video and upload to S3.
        
        This method orchestrates:
        1. Process video (activity detection)
        2. Upload original video to S3
        3. Upload evidence clips to S3
        4. Update activities.json with S3 URLs
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_members: Dictionary mapping role to crew member info
            subfolder_name: S3 subfolder name
            auth_token: Optional authentication token for S3
            use_mock_detection: Use mock detection
            use_multiprocessing: Enable multiprocessing
            save_clips: Whether to save clips
            s3_upload_service: S3 upload service instance
            
        Returns:
            Dict with processing results and S3 URLs
            
        Raises:
            VideoProcessingError: If processing fails
            S3UploadError: If S3 upload fails
        """
        import json
        
        logger.info(f"🎬 Starting process-and-upload workflow for trip: {trip_id}")
        
        # Step 1: Process video
        result = self.process_video(
            video_path=video_path,
            trip_id=trip_id,
            crew_members=crew_members,
            crew_name=get_default_crew_name(crew_members) if crew_members else "Unknown",
            crew_id=get_default_crew_id(crew_members) if crew_members else "N/A",
            crew_role=1,
            use_mock_detection=use_mock_detection,
            use_multiprocessing=use_multiprocessing,
            save_clips=save_clips
        )
        
        run_dir = result.get('runDirectory', result.get('run_dir', ''))
        clip_files = result.get('clip_files', result.get('clipFiles', []))
        
        # Step 2: Upload original video to S3
        logger.info(f"☁️ Uploading original video to S3 (subfolder: {subfolder_name})")
        video_upload_success, video_s3_url, video_error = s3_upload_service.upload_file(
            file_path=video_path,
            subfolder=subfolder_name,
            auth_token=auth_token
        )
        
        if not video_upload_success:
            logger.error(f"Failed to upload video to S3: {video_error}")
            raise VideoUploadError(
                message=f"Video processing succeeded but S3 upload failed: {video_error}",
                video_path=video_path,
                trip_id=trip_id
            )
        
        logger.info(f"✅ Video uploaded to S3: {video_s3_url}")
        
        # Step 3: Upload evidence files (clips + images) to S3
        evidence_urls: List[str] = []
        upload_errors: List[str] = []
        s3_file_mapping: Dict[str, str] = {}
        
        if clip_files:
            logger.info(f"☁️ Uploading {len(clip_files)} evidence files (clips + images) to S3")
            
            # Collect all files to upload (clips + their corresponding images)
            all_files_to_upload: List[str] = []
            for clip_file in clip_files:
                all_files_to_upload.append(clip_file)
                
                # Find corresponding image file
                image_file = clip_file.replace('_clip.mp4', '_activity.jpg')
                if os.path.exists(image_file):
                    all_files_to_upload.append(image_file)
            
            logger.info(f"☁️ Total files to upload: {len(all_files_to_upload)} (clips + images)")
            
            # Upload all files
            clips_success, file_urls, clip_errors = s3_upload_service.upload_multiple_files(
                file_paths=all_files_to_upload,
                subfolder=subfolder_name,
                auth_token=auth_token
            )
            
            # Create mapping from local path to S3 URL
            for local_path, s3_url in zip(all_files_to_upload, file_urls):
                s3_file_mapping[local_path] = s3_url
            
            evidence_urls = [url for url in file_urls if '_clip.mp4' in url]
            upload_errors = clip_errors
            
            if not clips_success:
                logger.warning(f"Some files failed to upload: {clip_errors}")
            
            logger.info(f"✅ Uploaded {len(file_urls)}/{len(all_files_to_upload)} files to S3")
        
        # Step 4: Update activities.json with S3 URLs
        logger.info("📝 Updating activities.json with S3 URLs")
        
        activities_json_path = os.path.join(run_dir, 'activities.json')
        updated_activities: List[Dict[str, Any]] = []
        
        if os.path.exists(activities_json_path):
            # Read existing activities
            with open(activities_json_path, 'r', encoding='utf-8') as f:
                activities = json.load(f)
            
            # Update each activity with S3 URLs
            for activity in activities:
                # Update activityClip with S3 URL
                if 'activityClip' in activity and activity['activityClip']:
                    local_clip_path = activity['activityClip']
                    if local_clip_path in s3_file_mapping:
                        activity['activityClip'] = s3_file_mapping[local_clip_path]
                        logger.debug(f"Updated clip URL: {local_clip_path} -> {s3_file_mapping[local_clip_path]}")
                
                # Update activityImage with S3 URL
                if 'activityImage' in activity and activity['activityImage']:
                    local_image_path = activity['activityImage']
                    if local_image_path in s3_file_mapping:
                        activity['activityImage'] = s3_file_mapping[local_image_path]
                        logger.debug(f"Updated image URL: {local_image_path} -> {s3_file_mapping[local_image_path]}")
                
                updated_activities.append(activity)
            
            # Save updated activities.json with S3 URLs
            with open(activities_json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_activities, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Updated {len(updated_activities)} activities with S3 URLs")
            activities = updated_activities
        else:
            activities = result.get('activities', [])
        
        # Prepare response
        response_data = {
            "status": "success",
            "message": "Video processed and uploaded successfully",
            "data": {
                "tripId": trip_id,
                "run_id": os.path.basename(run_dir) if run_dir else "",
                "run_dir": run_dir,
                "activities_count": len(activities),
                "processing_time_seconds": result.get('processingTime', result.get('processing_time', 0)),
                "video_url": video_s3_url,
                "evidence_clips": evidence_urls,
                "clips_uploaded": len(evidence_urls),
                "total_clips": len(clip_files),
                "upload_errors": upload_errors if upload_errors else None,
                "activities": activities
            }
        }
        
        logger.info(f"✅ Complete workflow finished for trip: {trip_id}")
        return response_data

