"""
Video processing service - Orchestrates video upload and activity detection

This service handles the main business logic for processing uploaded videos.
"""

import os
import time
import threading
from typing import Dict, Any, List, Optional

from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..repositories.activity_repository import ActivityRepository
from .activity_detection_service import ActivityDetectionService
from .external_api_service import get_external_api_service
from .vlm_verification_service import get_vlm_verification_service

# Train motion rule engine imports
try:
    from .trip_data_service import get_trip_data_service
except ImportError:
    get_trip_data_service = None


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

    Implements thread-safe singleton pattern.
    """

    _instance: Optional['VideoProcessingService'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'VideoProcessingService':
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

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
        # Skip re-initialization for singleton
        if VideoProcessingService._initialized:
            return

        self.activity_repository = activity_repository or ActivityRepository(
            output_dir=settings.output_dir
        )
        self.activity_detection_service = activity_detection_service or ActivityDetectionService()

        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)

        logger.info(
            f"[OK] Video processing service initialized - "
            f"Output dir: {settings.output_dir}, Upload dir: {settings.upload_dir}"
        )

        VideoProcessingService._initialized = True
    
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
        save_clips: bool = True,
        skip_external_api: bool = False,
        skip_vlm_verification: bool = False,
        division: Optional[str] = None,
        train_number: Optional[str] = None,
        trip_date: Optional[str] = None,
        video_start_time: Optional[str] = None,
        camera_angle: int = 1
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
            skip_external_api: Skip posting to external API (for process-and-upload endpoint)
            division: Division identifier for external API URL
            train_number: Train number for motion-based rule engine (optional)
            trip_date: Trip date in YYYY-MM-DD format for schedule lookup (optional)
            video_start_time: Video recording start time in HH:MM:SS format (optional, for motion rules when OCR unavailable)
            camera_angle: Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side)

        Returns:
            Dict[str, Any]: Processing results with activities

        Raises:
            Exception: If processing fails
        """
        start_time = time.time()

        try:
            logger.info(
                f"[START] Starting video processing for trip {trip_id} - "
                f"Multiprocessing: {'enabled' if use_multiprocessing else 'disabled'}, "
                f"Save clips: {save_clips}, Mock detection: {use_mock_detection}"
            )

            # Validate video file exists
            if not os.path.exists(video_path):
                logger.error(f"[ERROR] Video file not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")

            logger.info(f"[OK] Video file validated: {video_path}")

            # Fetch trip schedule for motion-based rule engine (if train info provided)
            trip_schedule = None
            logger.info(
                f"[MOTION-RULES] Configuration check - "
                f"train_number: {train_number}, trip_date: {trip_date}, "
                f"rules_enabled: {settings.train_motion_rules_enabled}"
            )

            if train_number and trip_date and settings.train_motion_rules_enabled:
                logger.info(
                    f"[MOTION-RULES] [OK] All conditions met - fetching trip schedule "
                    f"for train {train_number} on {trip_date}"
                )
                try:
                    if get_trip_data_service is not None:
                        trip_data_service = get_trip_data_service()
                        # Use delay-enhanced schedule fetch if etrain integration is enabled
                        if hasattr(trip_data_service, 'fetch_trip_schedule_with_delays') and settings.etrain_enabled:
                            logger.info(f"[MOTION-RULES] Calling TripDataService.fetch_trip_schedule_with_delays() (etrain.info enabled)")
                            trip_schedule = trip_data_service.fetch_trip_schedule_with_delays(
                                train_number=train_number,
                                journey_date=trip_date,
                                division=division
                            )
                        else:
                            logger.info(f"[MOTION-RULES] Calling TripDataService.fetch_trip_schedule()")
                            trip_schedule = trip_data_service.fetch_trip_schedule(
                                train_number=train_number,
                                journey_date=trip_date,
                                division=division
                            )
                        if trip_schedule:
                            logger.info(
                                f"[MOTION-RULES] [TRAIN] Successfully fetched trip schedule for train {train_number}: "
                                f"{len(trip_schedule.halts)} station halts"
                            )
                            # Log first few halts for debugging (include delay info if available)
                            for i, halt in enumerate(trip_schedule.halts[:3]):
                                delay_info = f", Delay: {halt.delay_minutes}min" if hasattr(halt, 'delay_minutes') and halt.delay_minutes > 0 else ""
                                logger.info(
                                    f"[MOTION-RULES]   Halt {i+1}: {halt.station_name} ({halt.station_code}) - "
                                    f"Arr: {halt.scheduled_arrival}, Dep: {halt.scheduled_departure}{delay_info}"
                                )
                            if len(trip_schedule.halts) > 3:
                                logger.info(f"[MOTION-RULES]   ... and {len(trip_schedule.halts) - 3} more halts")
                        else:
                            logger.warning(
                                f"[MOTION-RULES] [WARN] Could not fetch trip schedule for train {train_number} "
                                f"on {trip_date} - no_person_detected will be suppressed (cannot distinguish station halts)"
                            )
                    else:
                        logger.warning("[MOTION-RULES] [WARN] Trip data service not available - motion rules disabled")
                except Exception as e:
                    logger.warning(f"[MOTION-RULES] [WARN] Error fetching trip schedule: {e} - no_person_detected will be suppressed")
            else:
                missing = []
                if not train_number:
                    missing.append("train_number")
                if not trip_date:
                    missing.append("trip_date")
                if not settings.train_motion_rules_enabled:
                    missing.append("rules_enabled=False")
                logger.info(
                    f"[MOTION-RULES] [SKIP] Skipping motion-based rules - missing: {', '.join(missing)}. "
                    f"All detected activities will be treated as violations."
                )

            # Create run directory ONCE at the top level
            run_dir = self.activity_repository.create_run_directory(base_name=f"run")
            logger.info(f"[OK] Created run directory: {run_dir}")
            
            # Run activity detection
            if use_mock_detection:
                logger.info("[MOCK] Using mock activity detection")
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
                    f"[DETECT] Using real activity detection - "
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
                        save_clips=save_clips,
                        trip_schedule=trip_schedule,  # Pass trip schedule for motion rules
                        video_start_time=video_start_time,  # Pass video start time for motion rules
                        camera_angle=camera_angle
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
                        run_dir=run_dir,  # Pass existing run_dir
                        trip_schedule=trip_schedule,  # Pass trip schedule for motion rules
                        video_start_time=video_start_time,  # Pass video start time for motion rules
                        camera_angle=camera_angle
                    )
            
            # Save activities to JSON
            activities_json_path = self.activity_repository.save_activities(
                activities=activities,
                run_dir=run_dir
            )
            logger.info(f"[SAVE] Saved {len(activities)} activities to {activities_json_path}")

            # Pipeline-2: VLM verification before external API post.
            # Skipped when skip_vlm_verification=True (the /v1/process-and-upload
            # controller runs its own hook with extra clip_files filtering).
            # Fail-open: a verifier exception leaves activities unchanged.
            if not skip_vlm_verification:
                vlm_service = get_vlm_verification_service()
                if vlm_service.is_enabled():
                    try:
                        pre_count = len(activities)
                        activities, vlm_stats = vlm_service.verify_activities(activities)
                        # Re-save activities through the repository so the
                        # post-VLM rewrite uses the same atomic + locked +
                        # numpy-aware writer as Pipeline-1 (Task 0002).
                        self.activity_repository.save_activities(
                            activities=activities,
                            run_dir=run_dir,
                        )
                        logger.info(
                            f"[VLM] verified pre={pre_count} post={len(activities)} "
                            f"dropped={vlm_stats['dropped']} uncertain={vlm_stats['uncertain']} "
                            f"skipped_unavail={vlm_stats['skipped_unavailable']}"
                        )
                    except Exception as vlm_exc:
                        logger.warning(
                            f"[VLM] verifier failed unexpectedly, passing through "
                            f"Pipeline-1 results: {vlm_exc}",
                            exc_info=True,
                        )

            # Post results to external API (non-blocking, errors don't fail the job)
            # Skip if called from process-and-upload endpoint (will be called later with S3 URLs)
            api_result = None
            if not skip_external_api:
                try:
                    logger.info(f"[API] Attempting to post results to external API...")
                    external_api_service = get_external_api_service()
                    
                    # Extract run_id from run_dir for constructing job_id
                    run_id = os.path.basename(run_dir)
                    
                    # Filter out STOPPED activities (only post RUNNING and UNCERTAIN),
                    # EXCEPT safety-critical types whitelisted via
                    # ``MOTION_FILTER_BYPASS_TYPES`` (defaults to cell_phone +
                    # microsleep). Per spec, those violations matter regardless
                    # of train motion state and must reach the external API.
                    _bypass_raw = getattr(settings, 'motion_filter_bypass_types', '') or ''
                    _bypass = {
                        s.strip().lower().replace(' ', '_')
                        for s in _bypass_raw.split(',') if s.strip()
                    }
                    def _postable(a):
                        if (a.get('motionState') or 'UNKNOWN').upper() != 'STOPPED':
                            return True
                        ot = (a.get('objectType') or '').strip().lower().replace(' ', '_')
                        return ot in _bypass
                    postable_activities = [a for a in activities if _postable(a)]
                    _bypassed = sum(
                        1 for a in activities
                        if (a.get('motionState') or '').upper() == 'STOPPED' and _postable(a)
                    )
                    _excluded = len(activities) - len(postable_activities)
                    logger.info(
                        f"[API] Motion filter: {len(postable_activities)}/{len(activities)} "
                        f"activities to post (excluded {_excluded} STOPPED, "
                        f"bypassed {_bypassed} safety-critical STOPPED)"
                    )

                    # Post to external API. ``run_dir`` is forwarded so a
                    # retries-exhausted payload lands in
                    # ``<run_dir>/_failed_external_api/`` per spec 0004.
                    api_result = external_api_service.post_cvvr_results(
                        trip_id=trip_id,
                        events=postable_activities,
                        job_id=run_id,
                        host_url=settings.host_url,
                        division=division,
                        run_dir=run_dir,
                    )
                    
                    if api_result.get("success"):
                        logger.info(
                            f"[OK] [external_api] Successfully posted {api_result.get('violations_count', 0)} "
                            f"violations to external API for trip {trip_id}"
                        )
                    else:
                        logger.warning(
                            f"[WARN] [external_api] Failed to post to external API: {api_result.get('message')}"
                        )
                except Exception as e:
                    logger.error(
                        f"[ERROR] [external_api] Exception while posting to external API: {e}",
                        exc_info=True
                    )
                    api_result = {
                        "success": False,
                        "message": f"Exception: {str(e)}",
                        "posted": False
                    }
            else:
                logger.info(f"[SKIP] [external_api] Skipping external API call (will be called later with S3 URLs)")
                api_result = {
                    "success": False,
                    "message": "Skipped - will be called later with S3 URLs",
                    "posted": False
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate summary
            summary = self.activity_repository.get_activity_summary(activities)
            
            logger.info(
                f"[OK] Video processing completed in {processing_time:.2f}s - "
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
                f"[ERROR] Video processing failed after {processing_time:.2f}s for trip {trip_id}: {e}",
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
                logger.info(f"[CLEANUP] Cleaned up uploaded video: {video_path}")
        except Exception as e:
            logger.warning(f"[WARN] Failed to cleanup video {video_path}: {e}")
    
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

