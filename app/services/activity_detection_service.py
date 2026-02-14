"""
Activity detection service - Handles activity detection logic
"""

import gc
from typing import List, Dict, Any

from ..utils.logger import get_logger
from ..models.activity_models import ActivityTypeEnum
from ..utils.config import get_settings


logger = get_logger(__name__)


class ActivityDetectionService:
    """
    Service for detecting activities in video frames
    
    Service for detecting activities in video frames
    """
    
    def __init__(self):
        """Initialize the activity detection service"""
        self.activity_type_map = {
            'cell_phone': ActivityTypeEnum.CELL_PHONE,
            'microsleep': ActivityTypeEnum.MICROSLEEP,
            'sleep': ActivityTypeEnum.SLEEP,
            'writing': ActivityTypeEnum.WRITING,
            'packing_bags': ActivityTypeEnum.PACKING_BAGS,
            'group_detected': ActivityTypeEnum.GROUP_DETECTED,
            'lp_hand_gesture': ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE,
            'alp_hand_gesture': ActivityTypeEnum.ALP_NOT_EXCHANGING_HAND_GESTURE,
            'mind_diversion': ActivityTypeEnum.MIND_DIVERSION,
            'no_person_detected': ActivityTypeEnum.NO_PERSON_DETECTED
        }
        
        self.activity_descriptions = {
            'cell_phone': 'Using mobile phone',
            'microsleep': 'Micro-sleep detected (5+ seconds)',
            'sleep': 'Sleep detected (30+ seconds)',
            'writing': 'WRITING LOG BOOK WHILE RUNNING',
            'packing_bags': 'Packing bags activity detected',
            'group_detected': 'More than 2 people (group) detected',
            'lp_hand_gesture': 'LP not exchanging hand gesture',
            'alp_hand_gesture': 'ALP not exchanging hand gesture',
            'mind_diversion': 'Mind diversion - attention diverted from controls',
            'no_person_detected': 'No person detected in frame'
        }
        
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'pose_indicators',
            'sleep': 'pose_indicators',
            'writing': 'hand_near_book',
            'packing_bags': 'wrist_inside_backpack_bbox_or_hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons',
            'lp_hand_gesture': 'lp_hand_raised_gesture_detected',
            'alp_hand_gesture': 'alp_hand_raised_gesture_detected',
            'mind_diversion': 'attention_diverted_from_controls',  # Sub-type (looking_sideways, looking_down_distracted, looking_away_combined) in evidence
            'no_person_detected': 'zero_persons_in_frame'
        }

        # Mind diversion sub-type evidence descriptions
        self.mind_diversion_sub_types = {
            'looking_sideways': 'head_turned_sideways_sustained',
            'looking_down_distracted': 'head_looking_down_sustained',
            'looking_away_combined': 'head_turned_and_looking_down'
        }
        
        self.settings = get_settings()
        logger.info("Activity detection service initialized")
    
    def detect_activities_real(
        self,
        video_path: str,
        trip_id: str,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1,
        output_dir: str = "locopilot_evidence",
        sample_fps: float = 0.5,
        use_multiprocessing: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Real activity detection using the LocopilotActivityMonitor
        
        This integrates with the existing locopilot_monitor.py logic.
        Supports both single-process and multi-process execution.
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            output_dir: Output directory for evidence
            sample_fps: Frame sampling rate
            use_multiprocessing: Enable multiprocessing (default: False)
            
        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        if use_multiprocessing:
            logger.info(f"Running real activity detection with MULTIPROCESSING for {video_path}")
            return self._detect_activities_multiprocess(
                video_path=video_path,
                trip_id=trip_id,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                output_dir=output_dir,
                sample_fps=sample_fps
            )
        else:
            logger.info(f"Running real activity detection with SINGLE PROCESS for {video_path}")
            return self._detect_activities_single_process(
                video_path=video_path,
                trip_id=trip_id,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                output_dir=output_dir,
                sample_fps=sample_fps
            )
    
    def _detect_activities_single_process(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Dict[str, Dict[str, str]] = None,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1,
        output_dir: str = "locopilot_evidence",
        sample_fps: float = 1.0,
        run_dir: str = None,
        trip_schedule = None,
        video_start_time: str = None,
        camera_angle: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Single-process activity detection (original implementation)

        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            output_dir: Output directory for evidence (base directory)
            sample_fps: Frame sampling rate
            run_dir: Run directory to use (if None, creates new one)
            trip_schedule: TripSchedule object for motion-based rule engine (optional)
            video_start_time: Video recording start time in HH:MM:SS format (optional)
            camera_angle: Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side)

        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        from locopilot_monitor import LocopilotActivityMonitor

        # Create monitor instance (with or without run_dir)
        if run_dir:
            # Use existing run directory
            monitor = LocopilotActivityMonitor(
                video_path=video_path,
                output_dir=output_dir,
                save_annotated_frames=self.settings.save_annotated_frames,
                frame_save_interval=self.settings.frame_save_interval,
                sample_fps=sample_fps,
                run_dir=run_dir,
                create_run_dir=False  # Don't create new directory
            )
        else:
            # Create new run directory
            monitor = LocopilotActivityMonitor(
                video_path=video_path,
                output_dir=output_dir,
                save_annotated_frames=self.settings.save_annotated_frames,
                frame_save_interval=self.settings.frame_save_interval,
                sample_fps=sample_fps
            )

        # Set trip and crew information
        monitor.trip_id = trip_id
        monitor.crew_name = crew_name
        monitor.crew_id = crew_id
        monitor.crew_role = crew_role

        # Set crew members mapping if provided
        if crew_members:
            monitor.crew_members = crew_members

        # Set camera angle for LP/ALP role assignment
        monitor.camera_angle = camera_angle

        # Set trip schedule for motion-based rule engine
        if trip_schedule is not None and hasattr(monitor, 'set_trip_schedule'):
            monitor.set_trip_schedule(trip_schedule)

        # Set video start time for motion rules (when OCR unavailable)
        if video_start_time and hasattr(monitor, 'set_video_start_time'):
            monitor.set_video_start_time(video_start_time)

        # Process video
        monitor.process_video()
        
        # Get activities before cleanup
        activities = monitor.all_activities.copy()

        # Get run_dir for aggregation (may be from monitor if not provided)
        actual_run_dir = run_dir or getattr(monitor, 'run_dir', None)

        # ✅ MEMORY FIX: Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
        monitor.cleanup()

        # Group overlapping activities of different types into combined records
        from .concurrent_activity_grouping_service import get_concurrent_grouping_service
        concurrent_grouping_service = get_concurrent_grouping_service()
        activities = concurrent_grouping_service.group_concurrent_activities(activities, actual_run_dir)

        # Return detected activities
        logger.info(f"Single-process detection found {len(activities)} activities")
        return activities
    
    def _detect_activities_multiprocess(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Dict[str, Dict[str, str]] = None,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1,
        output_dir: str = "locopilot_evidence",
        sample_fps: float = 1.0,
        run_dir: str = None,
        save_clips: bool = True,
        trip_schedule = None,
        video_start_time: str = None,
        camera_angle: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Multi-process activity detection using parallel processing

        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            output_dir: Output directory for evidence (base directory)
            sample_fps: Frame sampling rate
            run_dir: Run directory to use (if None, creates new one)
            save_clips: Whether to save video clips and images (default: True)
            trip_schedule: TripSchedule object for motion-based rule engine (optional)
            video_start_time: Video recording start time in HH:MM:SS format (optional)
            camera_angle: Camera angle for LP/ALP role assignment (1 = LP Side, 2 = ALP Side)

        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        from ..utils.video_multiprocessing import VideoMultiprocessingOrchestrator
        from ..utils.multiprocessing_config import MultiprocessingConfig
        from ..repositories.activity_repository import ActivityRepository
        
        # Get settings for configuration
        settings = get_settings()
        
        # Create run directory only if not provided
        if run_dir is None:
            activity_repo = ActivityRepository(output_dir=output_dir)
            run_dir = activity_repo.create_run_directory(base_name="run")
        
        # Create multiprocessing configuration
        # ✅ PERFORMANCE: 10s chunks optimize load balancing vs overhead
        # Smaller chunks (10s) provide better work distribution across 8 workers
        # Each chunk processes faster (~15-20s), keeping workers busy and reducing idle time
        config = MultiprocessingConfig(
            chunk_duration_seconds=settings.mp_chunk_duration,  # Use config value (default 10.0s)
            max_workers=None,  # Auto-detect
            max_workers_cap=settings.mp_max_workers_cap,  # Use config value (default 8)
            preload_models=True,
            yolo_device=settings.yolo_device,  # GPU device (0 for GPU, cpu for CPU)
        )
        
        # Create orchestrator (use shared global pool to mimic POC_2 behavior)
        orchestrator = VideoMultiprocessingOrchestrator(
            config=config,
            output_dir=run_dir,
            use_shared_pool=True,
        )
        
        try:
            # Process video in parallel with clip generation
            # Note: trip_schedule is passed for motion-based rules
            # Multiprocessing workers will need to re-fetch or receive serialized schedule
            activities = orchestrator.process_video_parallel(
                video_path=video_path,
                trip_id=trip_id,
                crew_members=crew_members,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                sample_fps=sample_fps,
                run_dir=run_dir,
                save_clips=save_clips,
                trip_schedule=trip_schedule,
                video_start_time=video_start_time,
                camera_angle=camera_angle
            )
            
            # ✅ MEMORY FIX: Force garbage collection after processing
            gc.collect()

            # Group overlapping activities of different types into combined records
            from .concurrent_activity_grouping_service import get_concurrent_grouping_service
            concurrent_grouping_service = get_concurrent_grouping_service()
            activities = concurrent_grouping_service.group_concurrent_activities(activities, run_dir)

            logger.info(f"Multi-process detection found {len(activities)} activities "
                       f"(clips {'generated' if save_clips else 'not generated'})")
            return activities
            
        finally:
            # Cleanup: shutdown shared pool to release GPU memory after each job
            # orchestrator.shutdown_pool() is a no-op for shared pools,
            # so we call shutdown_shared_pool() directly to terminate workers
            # and free GPU VRAM between jobs.
            from ..utils.video_multiprocessing import shutdown_shared_pool
            shutdown_shared_pool(wait=True)

            # Force PyTorch to release cached GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Force garbage collection after shutdown
            gc.collect()

