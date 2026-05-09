"""
Activity detection service - Handles activity detection logic

This service can be extended with actual ML models for production use.
For now, it provides a placeholder/mock implementation.
"""

import random
import gc
from typing import List, Dict, Any
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..models.activity_models import ActivityTypeEnum
from ..core.activity_registry import ACTIVITY_REGISTRY
from ..utils.config import get_settings


logger = get_logger(__name__)


class ActivityDetectionService:
    """
    Service for detecting activities in video frames

    This is a mock/placeholder implementation that can be replaced
    with actual ML-based detection logic (YOLO, MediaPipe, etc.)
    """

    def __init__(self):
        """Initialize the activity detection service"""
        # Task 0001 (2026-04): the per-activity metadata dicts used to be
        # redefined here, were incomplete (missing ``eating_drinking`` and
        # ``alp_not_standing``) and silently drifted from the real monitor.
        # They are now derived from the single registry in
        # ``app.core.activity_registry`` so this mock automatically covers
        # every activity the real pipeline emits.
        self.activity_type_map: Dict[str, ActivityTypeEnum] = {
            name: ActivityTypeEnum(cfg.type_code)
            for name, cfg in ACTIVITY_REGISTRY.items()
        }
        self.activity_descriptions: Dict[str, str] = {
            name: cfg.description for name, cfg in ACTIVITY_REGISTRY.items()
        }
        self.evidence_rules: Dict[str, str] = {
            name: cfg.evidence_rule for name, cfg in ACTIVITY_REGISTRY.items()
        }

        # Mind diversion sub-type evidence descriptions (mock-only detail).
        self.mind_diversion_sub_types = {
            'looking_sideways': 'head_turned_sideways_sustained',
            'looking_down_distracted': 'head_looking_down_sustained',
            'looking_away_combined': 'head_turned_and_looking_down'
        }

        self.settings = get_settings()
        logger.info("Activity detection service initialized (mock mode)")
    
    def detect_activities_mock(
        self,
        video_path: str,
        trip_id: str,
        crew_members: Dict[str, Dict[str, str]] = None,
        crew_name: str = "John Doe",
        crew_id: str = "C-001",
        crew_role: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Mock activity detection
        
        Generates sample activities for demonstration purposes.
        Replace this with actual detection logic for production.
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            
        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        import os
        import cv2

        logger.info(f"Running mock activity detection for {video_path}")

        # Get video metadata with proper resource cleanup
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration_seconds = total_frames / fps
        finally:
            if cap is not None:
                cap.release()
        
        video_duration_formatted = str(timedelta(seconds=int(video_duration_seconds)))
        video_filename = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]
        
        # Generate mock activities
        activities = []
        
        # Randomly generate 1-3 activities
        num_activities = random.randint(1, 3)
        activity_types = random.sample(
            list(self.activity_type_map.keys()), 
            min(num_activities, len(self.activity_type_map))
        )
        
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        
        for idx, activity_name in enumerate(activity_types):
            # Generate random timestamps
            start_time = random.uniform(0, max(0, video_duration_seconds - 10))
            end_time = start_time + random.uniform(5, 15)
            end_time = min(end_time, video_duration_seconds)
            
            # Generate filenames
            start_frame = int(start_time * fps)
            clip_filename = f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}_{idx:03d}_clip.mp4"
            image_filename = f"{video_name_without_ext}_{activity_name}_frame{start_frame:08d}_{idx:03d}_activity.jpg"
            
            activity = {
                "tripId": trip_id,
                "activityType": self.activity_type_map[activity_name],
                "des": self.activity_descriptions[activity_name],
                "objectType": activity_name.replace('_', ' '),
                "fileUrl": os.path.abspath(video_path),
                "fileDuration": video_duration_formatted,
                "activityStartTime": f"{start_time:.2f}",
                "activityEndTime": f"{end_time:.2f}",
                "crewName": crew_name,
                "crewId": crew_id,
                "crewRole": crew_role,
                "date": current_date,
                "time": current_time,
                "filename": video_filename,
                "peopleCount": random.randint(1, 2),
                "evidence": {"rule": self.evidence_rules[activity_name]},
                "activityImage": image_filename,
                "activityClip": clip_filename
            }
            
            activities.append(activity)
        
        logger.info(f"Mock detection generated {len(activities)} activities")
        return activities
    
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

        # MEMORY FIX: Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
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
        # PERFORMANCE: 15s chunks ensure hand gesture coordination detection works correctly
        # Coordination window is 10s, so 15s chunks capture full coordination sequences
        # Each chunk processes in ~15-20s, keeping workers busy and reducing idle time
        config = MultiprocessingConfig(
            chunk_duration_seconds=settings.mp_chunk_duration,  # Use config value (default 15.0s)
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
                video_start_time=video_start_time,
                camera_angle=camera_angle
            )
            
            # MEMORY FIX: Force garbage collection after processing
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

