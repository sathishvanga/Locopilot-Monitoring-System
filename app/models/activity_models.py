"""
Activity models - Domain models for activity detection and monitoring
"""

from enum import IntEnum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

# Task 0001 (2026-04): the activity metadata source of truth is now
# ``app.core.activity_registry``. We import it here so the IntEnum members
# can be derived from the registry, eliminating the silent drift that
# previously existed between this enum and the runtime monitor/mock-service
# dicts.
from app.core.activity_registry import ACTIVITY_REGISTRY


# Map canonical registry key -> preferred enum member name. The name must be
# a valid Python identifier and should preserve the legacy names so external
# imports (e.g. ``ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE``) keep
# working without a follow-up rename.
_REGISTRY_TO_ENUM_NAME: Dict[str, str] = {
    'cell_phone': 'CELL_PHONE',
    'microsleep': 'MICROSLEEP',
    'sleep': 'SLEEP',
    'writing': 'WRITING',
    'packing_bags': 'PACKING_BAGS',
    'group_detected': 'GROUP_DETECTED',
    'lp_hand_gesture': 'LP_NOT_EXCHANGING_HAND_GESTURE',
    'alp_hand_gesture': 'ALP_NOT_EXCHANGING_HAND_GESTURE',
    'mind_diversion': 'MIND_DIVERSION',
    'no_person_detected': 'NO_PERSON_DETECTED',
    'alp_not_standing': 'ALP_NOT_STANDING_PRE_ARRIVAL',
    'eating_drinking': 'EATING_DRINKING',
}


def _build_activity_type_enum_members() -> Dict[str, int]:
    """Derive enum members from ``ACTIVITY_REGISTRY``.

    Guarantees every registry entry has a matching enum member and that
    ``type_code`` values never drift. Includes a historical ``UNKNOWN=1``
    slot (no registry equivalent) for backwards compatibility with consumers
    that fall back to this code.
    """

    members: Dict[str, int] = {"UNKNOWN": 1}
    missing = [
        key for key in ACTIVITY_REGISTRY if key not in _REGISTRY_TO_ENUM_NAME
    ]
    if missing:
        raise RuntimeError(
            "ActivityTypeEnum mapping is missing registry keys: "
            f"{missing}. Update _REGISTRY_TO_ENUM_NAME in activity_models.py."
        )
    for key, cfg in ACTIVITY_REGISTRY.items():
        members[_REGISTRY_TO_ENUM_NAME[key]] = cfg.type_code
    return members


ActivityTypeEnum = IntEnum(  # type: ignore[misc]
    "ActivityTypeEnum",
    _build_activity_type_enum_members(),
)
ActivityTypeEnum.__doc__ = (
    "Activity type enumeration derived from ``app.core.activity_registry``. "
    "Changing this enum in isolation is not supported — add entries to the "
    "registry instead."
)


def _assert_enum_matches_registry() -> None:
    """Fail loudly at import time if the enum and registry disagree.

    Defensive: because ``ActivityTypeEnum`` is now generated from
    ``ACTIVITY_REGISTRY`` it should always match, but a reviewer could still
    hand-edit ``_REGISTRY_TO_ENUM_NAME`` or the registry in isolation.
    """

    for key, cfg in ACTIVITY_REGISTRY.items():
        enum_name = _REGISTRY_TO_ENUM_NAME[key]
        member = ActivityTypeEnum[enum_name]
        if int(member) != cfg.type_code:
            raise RuntimeError(
                f"ActivityTypeEnum.{enum_name}={int(member)} does not match "
                f"ACTIVITY_REGISTRY['{key}'].type_code={cfg.type_code}."
            )


_assert_enum_matches_registry()


class EvidenceModel(BaseModel):
    """Evidence details for an activity"""
    rule: str = Field(..., description="Evidence rule that triggered the activity")


class PersonRoleModel(BaseModel):
    """Person role information with LP/ALP identification"""
    personIndex: int = Field(..., description="Index of the person (0, 1, 2, ...)")
    role: str = Field(..., description="Role code (LP, ALP, VISITOR)")
    roleName: str = Field(..., description="Human-readable role name")
    bboxArea: float = Field(..., description="Bounding box area used for camera proximity-based role assignment")


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
                        "bboxArea": 12345.0
                    }
                ]
            }
        }

