"""
Activity models - Domain models for activity detection and monitoring
"""

from enum import IntEnum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class ActivityTypeEnum(IntEnum):
    """Activity type enumeration matching existing system"""
    UNKNOWN = 1
    CELL_PHONE = 2
    MICROSLEEP = 3
    SLEEP = 4
    WRITING = 5
    PACKING_BAGS = 6
    GROUP_DETECTED = 7
    LP_NOT_EXCHANGING_HAND_GESTURE = 8
    ALP_NOT_EXCHANGING_HAND_GESTURE = 9
    MIND_DIVERSION = 10
    NO_PERSON_DETECTED = 11


class TrainStateEnum(IntEnum):
    """
    Train state enumeration for stopped train detection.

    Used to track whether the train is moving or stopped,
    enabling exemption of non-safety-critical violations
    during stopped periods (at stations, signals, etc.).
    """
    UNKNOWN = 0   # State not yet determined
    MOVING = 1    # Train is in motion
    STOPPED = 2   # Train is stationary


class EvidenceModel(BaseModel):
    """Evidence details for an activity"""
    rule: str = Field(..., description="Evidence rule that triggered the activity")


class PersonRoleModel(BaseModel):
    """Person role information with LP/ALP identification"""
    personIndex: int = Field(..., description="Index of the person (0, 1, 2, ...)")
    role: str = Field(..., description="Role code (LP, ALP, SUPERVISOR, TRAINEE, VISITOR)")
    roleName: str = Field(..., description="Human-readable role name")
    lpScore: int = Field(..., description="Loco Pilot score based on detected objects")
    alpScore: int = Field(..., description="Assistant Loco Pilot score based on detected objects")


class ActivityModel(BaseModel):
    """
    Activity detection model matching the existing activities.json format
    
    This model represents a single detected activity with all metadata,
    timestamps, and evidence information.
    """
    
    tripId: str = Field(..., description="Unique trip identifier")
    activityType: int = Field(..., description="Activity type code (2-7)")
    des: str = Field(..., description="Human-readable activity description")
    objectType: str = Field(..., description="Object type involved in activity")
    fileUrl: str = Field(..., description="Absolute path to source video file")
    fileDuration: str = Field(..., description="Total video duration (HH:MM:SS)")
    activityStartTime: str = Field(..., description="Activity start time in seconds")
    activityEndTime: str = Field(..., description="Activity end time in seconds")
    crewName: str = Field(..., description="Crew member name who performed the activity")
    crewId: str = Field(..., description="Crew member ID who performed the activity")
    crewRole: int = Field(..., description="Crew role (1 = LP, 2 = ALP)")
    performingRole: Optional[str] = Field(None, description="Role of crew member who performed activity (LP or ALP)")
    date: str = Field(..., description="Date of activity (YYYY-MM-DD)")
    time: str = Field(..., description="Time of activity (HH:MM:SS)")
    filename: str = Field(..., description="Source video filename")
    peopleCount: int = Field(..., description="Number of people detected")
    evidence: EvidenceModel = Field(..., description="Evidence details")
    activityImage: str = Field(..., description="Activity screenshot filename")
    activityClip: str = Field(..., description="Activity video clip filename")
    personRoles: Optional[List[PersonRoleModel]] = Field(None, description="List of person roles identified (LP, ALP, etc.)")
    
    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "tripId": "TRIP-123",
                "activityType": 2,
                "des": "Using mobile phone",
                "objectType": "cell phone",
                "fileUrl": "/path/to/video.mp4",
                "fileDuration": "00:10:30",
                "activityStartTime": "125.50",
                "activityEndTime": "132.75",
                "crewName": "John Doe",
                "crewId": "LP-001",
                "crewRole": 1,
                "performingRole": "LP",
                "date": "2025-11-10",
                "time": "14:30:45",
                "filename": "latest.mp4",
                "peopleCount": 1,
                "evidence": {"rule": "phone_in_hand"},
                "activityImage": "latest_cell_phone_frame00001250_001_activity.jpg",
                "activityClip": "latest_cell_phone_frame00001250_001_clip.mp4",
                "personRoles": [
                    {
                        "personIndex": 0,
                        "role": "LP",
                        "roleName": "Loco Pilot",
                        "lpScore": 5,
                        "alpScore": 1
                    }
                ]
            }
        }

