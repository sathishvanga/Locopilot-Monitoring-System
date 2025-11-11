"""
Activity detection service - Handles activity detection logic

This service can be extended with actual ML models for production use.
For now, it provides a placeholder/mock implementation.
"""

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from ..models.activity_models import ActivityTypeEnum


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
            'group_detected': ActivityTypeEnum.GROUP_DETECTED
        }
        
        self.activity_descriptions = {
            'cell_phone': 'Using mobile phone',
            'microsleep': 'Micro-sleep detected (5+ seconds)',
            'sleep': 'Sleep detected (30+ seconds)',
            'writing': 'Writing activity detected',
            'packing_bags': 'Packing bags activity detected',
            'group_detected': 'More than 2 people (group) detected'
        }
        
        self.evidence_rules = {
            'cell_phone': 'phone_in_hand',
            'microsleep': 'eyes_closed_5s_or_pose_indicators',
            'sleep': 'eyes_closed_30s_or_pose_indicators',
            'writing': 'hand_near_book',
            'packing_bags': 'hand_near_backpack',
            'group_detected': 'more_than_2_deduplicated_persons'
        }
        
        logger.info("Activity detection service initialized (mock mode)")
    
    def detect_activities_mock(
        self,
        video_path: str,
        trip_id: str,
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
        sample_fps: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Real activity detection using the LocopilotActivityMonitor
        
        This integrates with the existing locopilot_monitor.py logic.
        
        Args:
            video_path: Path to video file
            trip_id: Trip identifier
            crew_name: Crew member name
            crew_id: Crew member ID
            crew_role: Crew role
            output_dir: Output directory for evidence
            sample_fps: Frame sampling rate
            
        Returns:
            List[Dict[str, Any]]: List of detected activities
        """
        from locopilot_monitor import LocopilotActivityMonitor
        
        logger.info(f"Running real activity detection for {video_path}")
        
        # Create monitor instance
        monitor = LocopilotActivityMonitor(
            video_path=video_path,
            output_dir=output_dir,
            save_annotated_frames=False,  # Disable for API use
            frame_save_interval=1,
            sample_fps=sample_fps
        )
        
        # Set trip and crew information
        monitor.trip_id = trip_id
        monitor.crew_name = crew_name
        monitor.crew_id = crew_id
        monitor.crew_role = crew_role
        
        # Process video
        monitor.process_video()
        
        # Return detected activities
        logger.info(f"Real detection found {len(monitor.all_activities)} activities")
        return monitor.all_activities

