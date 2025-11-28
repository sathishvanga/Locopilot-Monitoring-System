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
        self.activity_type_map = {
            'cell_phone': ActivityTypeEnum.CELL_PHONE,
            'microsleep': ActivityTypeEnum.MICROSLEEP,
            'sleep': ActivityTypeEnum.SLEEP,
            'writing': ActivityTypeEnum.WRITING,
            'packing_bags': ActivityTypeEnum.PACKING_BAGS,
            'group_detected': ActivityTypeEnum.GROUP_DETECTED,
            'lp_hand_gesture': ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE,
            'alp_hand_gesture': ActivityTypeEnum.ALP_NOT_EXCHANGING_HAND_GESTURE,
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
            'no_person_detected': 'No person detected in frame'
        }
        
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'eyes_closed_5s_or_pose_indicators',
            'sleep': 'eyes_closed_30s_or_pose_indicators',
            'writing': 'hand_near_book',
            'packing_bags': 'hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons',
            'lp_hand_gesture': 'lp_hand_raised_gesture_detected',
            'alp_hand_gesture': 'alp_hand_raised_gesture_detected',
            'no_person_detected': 'zero_persons_in_frame'
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
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_seconds = total_frames / fps
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
        run_dir: str = None
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
        
        # Process video
        monitor.process_video()
        
        # Get activities before cleanup
        activities = monitor.all_activities.copy()
        
        # ✅ MEMORY FIX: Explicit cleanup (closes MediaPipe, clears buffers, forces GC)
        monitor.cleanup()
        
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
        save_clips: bool = True
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
            
        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        from ..utils.video_multiprocessing import VideoMultiprocessingOrchestrator
        from ..utils.multiprocessing_config import MultiprocessingConfig
        from ..repositories.activity_repository import ActivityRepository
        
        # Create run directory only if not provided
        if run_dir is None:
            activity_repo = ActivityRepository(output_dir=output_dir)
            run_dir = activity_repo.create_run_directory(base_name="run")
        
        # Create multiprocessing configuration
        # Uses default max_workers_cap=6 for optimal CPU utilization (~60%) while maintaining memory safety
        config = MultiprocessingConfig(
            chunk_duration_seconds=6.0,
            max_workers=None,  # Auto-detect
            max_workers_cap=6,  # Consistent with default (increased from 2 for better CPU utilization)
            preload_models=True
        )
        
        # Create orchestrator (use shared global pool to mimic POC_2 behavior)
        orchestrator = VideoMultiprocessingOrchestrator(
            config=config,
            output_dir=run_dir,
            use_shared_pool=True,
        )
        
        try:
            # Process video in parallel with clip generation
            activities = orchestrator.process_video_parallel(
                video_path=video_path,
                trip_id=trip_id,
                crew_members=crew_members,
                crew_name=crew_name,
                crew_id=crew_id,
                crew_role=crew_role,
                sample_fps=sample_fps,
                run_dir=run_dir,
                save_clips=save_clips
            )
            
            # ✅ MEMORY FIX: Force garbage collection after processing
            gc.collect()
            
            logger.info(f"Multi-process detection found {len(activities)} activities "
                       f"(clips {'generated' if save_clips else 'not generated'})")
            return activities
            
        finally:
            # Cleanup: shutdown pool
            orchestrator.shutdown_pool(wait=True)
            
            # ✅ MEMORY FIX: Force garbage collection after shutdown
            gc.collect()

