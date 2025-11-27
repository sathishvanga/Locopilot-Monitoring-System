"""
Tests for ActivityRepository

Tests activity data persistence and retrieval.
"""

import pytest
import json
import os

from app.repositories.activity_repository import ActivityRepository
from app.models.activity_models import ActivityModel


class TestActivityRepository:
    """Test suite for ActivityRepository"""
    
    def test_save_activities(
        self, 
        activity_repository: ActivityRepository,
        sample_activities: list
    ):
        """Test saving activities to JSON file"""
        run_dir = activity_repository.create_run_directory()
        
        activities_json_path = activity_repository.save_activities(
            activities=sample_activities,
            run_dir=run_dir
        )
        
        assert os.path.exists(activities_json_path)
        assert activities_json_path.endswith("activities.json")
        
        # Verify file contents
        with open(activities_json_path, 'r', encoding='utf-8') as f:
            loaded_activities = json.load(f)
        
        assert len(loaded_activities) == len(sample_activities)
        assert loaded_activities[0]['tripId'] == sample_activities[0]['tripId']
    
    def test_load_activities(
        self,
        activity_repository: ActivityRepository,
        sample_activities: list
    ):
        """Test loading activities from JSON file"""
        run_dir = activity_repository.create_run_directory()
        activities_json_path = activity_repository.save_activities(
            activities=sample_activities,
            run_dir=run_dir
        )
        
        loaded_activities = activity_repository.load_activities(activities_json_path)
        
        assert len(loaded_activities) == len(sample_activities)
        assert loaded_activities[0]['tripId'] == sample_activities[0]['tripId']
    
    def test_load_activities_not_found(self, activity_repository: ActivityRepository):
        """Test loading activities from nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            activity_repository.load_activities("/nonexistent/path/activities.json")
    
    def test_get_activity_summary(
        self,
        activity_repository: ActivityRepository,
        sample_activities: list
    ):
        """Test generating activity summary"""
        summary = activity_repository.get_activity_summary(sample_activities)
        
        assert summary['total_activities'] == len(sample_activities)
        assert 'activity_breakdown' in summary
        assert 'total_duration' in summary
    
    def test_create_run_directory(self, activity_repository: ActivityRepository):
        """Test creating run directory"""
        run_dir = activity_repository.create_run_directory()
        
        assert os.path.exists(run_dir)
        assert os.path.isdir(run_dir)
        assert os.path.exists(os.path.join(run_dir, "clips"))
        assert os.path.exists(os.path.join(run_dir, "frames"))
    
    def test_validate_activities(
        self,
        activity_repository: ActivityRepository,
        sample_activities: list
    ):
        """Test activity validation"""
        is_valid = activity_repository.validate_activities(sample_activities)
        assert is_valid is True
    
    def test_validate_activities_invalid(self, activity_repository: ActivityRepository):
        """Test activity validation with invalid data"""
        invalid_activities = [{"invalid": "data"}]
        is_valid = activity_repository.validate_activities(invalid_activities)
        assert is_valid is False

