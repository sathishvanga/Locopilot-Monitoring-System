"""
Trip models - Data models for train motion state and schedule tracking

These models support the train movement-based rule engine that dynamically
changes violation detection based on whether the train is running or stopped.
"""

from enum import Enum
from typing import Optional, List
from datetime import datetime, time
from pydantic import BaseModel, Field


class TrainMotionState(str, Enum):
    """Train motion state enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class StationHalt(BaseModel):
    """
    Model for a station halt in the train schedule

    Represents a single station stop with arrival/departure times.
    """
    station_code: str = Field(..., description="3-4 letter station identifier")
    station_name: str = Field(..., description="Full station name")
    scheduled_arrival: str = Field(..., description="Scheduled arrival time (HH:MM:SS)")
    scheduled_departure: str = Field(..., description="Scheduled departure time (HH:MM:SS)")
    halt_duration_minutes: int = Field(default=0, description="Halt duration at station in minutes")
    distance_km: float = Field(default=0.0, description="Distance from origin in kilometers")
    journey_day: int = Field(default=1, description="Day of journey (1 = start day, 2 = next day)")
    platform: Optional[str] = Field(None, description="Platform number if available")

    # Calculated fields for easy comparison
    arrival_minutes: int = Field(default=0, description="Arrival time in minutes from midnight")
    departure_minutes: int = Field(default=0, description="Departure time in minutes from midnight")

    class Config:
        json_schema_extra = {
            "example": {
                "station_code": "SPJ",
                "station_name": "Samastipur Jn",
                "scheduled_arrival": "02:15:00",
                "scheduled_departure": "02:20:00",
                "halt_duration_minutes": 5,
                "distance_km": 58,
                "journey_day": 1,
                "platform": "2",
                "arrival_minutes": 135,
                "departure_minutes": 140
            }
        }


class TripSchedule(BaseModel):
    """
    Model for complete train trip schedule

    Contains all station halts for a train journey.
    """
    train_number: str = Field(..., description="5-digit train number")
    train_name: Optional[str] = Field(None, description="Train name")
    journey_date: str = Field(..., description="Journey date in YYYY-MM-DD format")
    origin_station: Optional[str] = Field(None, description="Origin station code")
    destination_station: Optional[str] = Field(None, description="Destination station code")
    halts: List[StationHalt] = Field(default_factory=list, description="List of station halts")
    total_distance_km: Optional[float] = Field(None, description="Total journey distance")
    fetched_at: datetime = Field(default_factory=datetime.now, description="Timestamp when schedule was fetched")

    def get_halt_at_time(self, time_str: str) -> Optional[StationHalt]:
        """
        Find the station halt that includes the given time

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            StationHalt if train is at a station at this time, None otherwise
        """
        try:
            parts = time_str.split(':')
            query_minutes = int(parts[0]) * 60 + int(parts[1])

            for halt in self.halts:
                # Check if time falls within halt period
                if halt.arrival_minutes <= query_minutes <= halt.departure_minutes:
                    return halt

            return None
        except (ValueError, IndexError):
            return None

    def get_next_halt(self, time_str: str) -> Optional[StationHalt]:
        """
        Find the next upcoming station halt after the given time

        Args:
            time_str: Time in HH:MM:SS format

        Returns:
            Next StationHalt if found, None otherwise
        """
        try:
            parts = time_str.split(':')
            query_minutes = int(parts[0]) * 60 + int(parts[1])

            for halt in self.halts:
                if halt.arrival_minutes > query_minutes:
                    return halt

            return None
        except (ValueError, IndexError):
            return None

    class Config:
        json_schema_extra = {
            "example": {
                "train_number": "12345",
                "train_name": "Express Mail",
                "journey_date": "2025-01-27",
                "origin_station": "NED",
                "destination_station": "HWH",
                "halts": [],
                "total_distance_km": 1500.0
            }
        }


class TrainMotionContext(BaseModel):
    """
    Model for current train motion context at a specific timestamp

    Combines motion state with relevant station information for rule evaluation.
    """
    motion_state: TrainMotionState = Field(default=TrainMotionState.UNKNOWN, description="Current motion state")
    timestamp: str = Field(..., description="Timestamp in HH:MM:SS format")

    # Station context (populated when stopped or approaching)
    current_station: Optional[StationHalt] = Field(None, description="Current station if stopped")
    next_station: Optional[StationHalt] = Field(None, description="Next upcoming station")

    # Pre-arrival context
    is_pre_arrival_window: bool = Field(default=False, description="True if within 30-60s of arrival")
    seconds_to_arrival: Optional[int] = Field(None, description="Seconds until next station arrival")

    # Grace period tracking
    seconds_since_departure: Optional[int] = Field(None, description="Seconds since last departure")
    in_grace_period: bool = Field(default=False, description="True if within grace period after departure")

    # Debug info
    resolution_source: str = Field(default="unknown", description="How state was determined")

    class Config:
        json_schema_extra = {
            "example": {
                "motion_state": "stopped",
                "timestamp": "14:30:45",
                "current_station": None,
                "next_station": None,
                "is_pre_arrival_window": False,
                "seconds_to_arrival": None,
                "seconds_since_departure": None,
                "in_grace_period": False,
                "resolution_source": "schedule_lookup"
            }
        }


class OCRTimestampResult(BaseModel):
    """
    Model for OCR timestamp extraction result

    Contains the extracted timestamp and metadata about the extraction.
    """
    success: bool = Field(default=False, description="Whether extraction succeeded")
    timestamp: Optional[str] = Field(None, description="Extracted timestamp in HH:MM:SS format")
    raw_text: Optional[str] = Field(None, description="Raw OCR text before parsing")
    confidence: float = Field(default=0.0, description="OCR confidence score (0-1)")

    # Extraction metadata
    roi_position: str = Field(default="top-right", description="ROI position used")
    roi_coords: Optional[tuple] = Field(None, description="ROI coordinates (x, y, width, height)")
    extraction_time_ms: float = Field(default=0.0, description="Time taken for extraction in milliseconds")

    # Error info
    error_message: Optional[str] = Field(None, description="Error message if extraction failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "timestamp": "14:30:45",
                "raw_text": "14:30:45",
                "confidence": 0.95,
                "roi_position": "top-right",
                "roi_coords": (1720, 10, 200, 50),
                "extraction_time_ms": 45.2,
                "error_message": None
            }
        }


class ViolationRuleResult(BaseModel):
    """
    Model for rule evaluation result

    Contains the decision about whether an activity is a violation
    based on train motion state.
    """
    activity_type: int = Field(..., description="Activity type code")
    activity_name: str = Field(..., description="Activity name")
    is_violation: bool = Field(..., description="Whether this is a violation")
    is_exempted: bool = Field(default=False, description="Whether activity is exempted due to motion state")

    # Rule context
    motion_state: TrainMotionState = Field(..., description="Train motion state when evaluated")
    rule_applied: str = Field(..., description="Name of rule applied")
    reason: str = Field(..., description="Human-readable reason for decision")

    # Original detection
    was_detected: bool = Field(default=False, description="Whether activity was detected")

    class Config:
        json_schema_extra = {
            "example": {
                "activity_type": 5,
                "activity_name": "writing",
                "is_violation": False,
                "is_exempted": True,
                "motion_state": "stopped",
                "rule_applied": "STOPPED_WRITING_EXEMPTION",
                "reason": "Writing is allowed when train is stopped at station",
                "was_detected": True
            }
        }
