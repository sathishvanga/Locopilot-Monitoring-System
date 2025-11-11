"""
Activity repository - Handles file I/O operations for activities.json

This repository manages reading and writing activity data to JSON files.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.logger import get_logger
from ..models.activity_models import ActivityModel


logger = get_logger(__name__)


class ActivityRepository:
    """
    Repository for managing activity data persistence
    
    Handles all file I/O operations for activities.json files,
    ensuring data integrity and proper error handling.
    """
    
    def __init__(self, output_dir: str = "locopilot_evidence"):
        """
        Initialize the activity repository
        
        Args:
            output_dir: Base directory for storing evidence and activities
        """
        self.output_dir = output_dir
        
    def save_activities(
        self, 
        activities: List[Dict[str, Any]], 
        run_dir: str
    ) -> str:
        """
        Save activities to activities.json in the run directory
        
        Args:
            activities: List of activity dictionaries
            run_dir: Directory path for this processing run
            
        Returns:
            str: Path to the saved activities.json file
            
        Raises:
            IOError: If file writing fails
        """
        try:
            # Ensure run directory exists
            os.makedirs(run_dir, exist_ok=True)
            
            # Create activities.json path
            activities_json_path = os.path.join(run_dir, "activities.json")
            
            # Write activities to JSON file with proper formatting
            with open(activities_json_path, 'w', encoding='utf-8') as f:
                json.dump(activities, f, indent=2, ensure_ascii=False)
            
            logger.info(
                f"Successfully saved {len(activities)} activities to {activities_json_path}"
            )
            
            return activities_json_path
            
        except Exception as e:
            logger.error(f"Failed to save activities: {e}", exc_info=True)
            raise IOError(f"Failed to save activities.json: {str(e)}") from e
    
    def load_activities(self, activities_json_path: str) -> List[Dict[str, Any]]:
        """
        Load activities from activities.json file
        
        Args:
            activities_json_path: Path to activities.json file
            
        Returns:
            List[Dict[str, Any]]: List of activity dictionaries
            
        Raises:
            FileNotFoundError: If activities.json doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        try:
            if not os.path.exists(activities_json_path):
                raise FileNotFoundError(f"Activities file not found: {activities_json_path}")
            
            with open(activities_json_path, 'r', encoding='utf-8') as f:
                activities = json.load(f)
            
            logger.info(f"Successfully loaded {len(activities)} activities from {activities_json_path}")
            
            return activities
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in activities file: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to load activities: {e}", exc_info=True)
            raise
    
    def get_latest_run_dir(self) -> Optional[str]:
        """
        Get the most recent run directory
        
        Returns:
            Optional[str]: Path to latest run directory, or None if none exist
        """
        try:
            if not os.path.exists(self.output_dir):
                return None
            
            # List all run directories
            run_dirs = [
                d for d in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith("run_")
            ]
            
            if not run_dirs:
                return None
            
            # Sort by name (timestamp-based) and get the latest
            latest_run = sorted(run_dirs)[-1]
            return os.path.join(self.output_dir, latest_run)
            
        except Exception as e:
            logger.error(f"Failed to get latest run directory: {e}", exc_info=True)
            return None
    
    def validate_activities(self, activities: List[Dict[str, Any]]) -> bool:
        """
        Validate activities data structure
        
        Args:
            activities: List of activity dictionaries to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validate each activity using Pydantic model
            for activity in activities:
                ActivityModel(**activity)
            return True
            
        except Exception as e:
            logger.warning(f"Activity validation failed: {e}")
            return False
    
    def create_run_directory(self, base_name: str = "run") -> str:
        """
        Create a new timestamped run directory
        
        Args:
            base_name: Base name for the directory (default: "run")
            
        Returns:
            str: Path to the newly created run directory
        """
        from datetime import datetime
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"{base_name}_{timestamp}")
        
        # Create directories
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
        
        logger.info(f"Created run directory: {run_dir}")
        
        return run_dir
    
    def get_activity_summary(self, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from activities
        
        Args:
            activities: List of activity dictionaries
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        summary = {
            "total_activities": len(activities),
            "activity_breakdown": {},
            "total_duration": 0.0
        }
        
        for activity in activities:
            # Count by description
            desc = activity.get("des", "Unknown")
            summary["activity_breakdown"][desc] = summary["activity_breakdown"].get(desc, 0) + 1
            
            # Calculate total duration
            try:
                start_time = float(activity.get("activityStartTime", 0))
                end_time = float(activity.get("activityEndTime", 0))
                summary["total_duration"] += (end_time - start_time)
            except (ValueError, TypeError):
                pass
        
        return summary

